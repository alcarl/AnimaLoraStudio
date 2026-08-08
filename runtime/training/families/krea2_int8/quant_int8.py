"""Krea2 int8（ConvRot INT8）底模量化的封装层。

与 ``quant_fp8.py``（fp8 底模）并列的结构：int8 底模是 krea2 训练/推理的另一种
权重量化形态，前向通过 ConvRot INT8（Hadamard 旋转 + 行内 INT8 + GEMM 反量化）
核实现。本模块不直接实现量化/前向算法，而是**封装调用复制自 musubi-tuner 的
vendored 代码**（``krea2_int8.vendor.convrot_int8_utils``），对外提供与本仓库
fp8 路径对称的接口：

- ``patch_convrot_int8_linears(model, state_dict, bwd_mode=...)``：把量化后 state_dict
  里的 ``.scale_weight`` 层挂上 ConvRot INT8 前向 patch（等价 fp8 的
  ``patch_fp8_linears``）。
- ``build_convrot_quantizer()`` / ``quantize_state_dict(...)``：包装 vendored 的
  ``ConvRotInt8Quantizer``，用于动态量化 bf16 底模。
- ``model_has_int8_layers(model)``：探测模型是否已是 int8 量化形态。

前向 patch 后权重 ``requires_grad=False``，底模恒 frozen，梯度只流经 LoRA 参数
——与 fp8 同款 fp8_base/「int8_base」语义（kohya/musubi 生态标准做法）。

注意：ConvRot INT8 使用自定义 ``autograd.Function``（Triton 核无 autograd 支持），
因此 ``torch.compile`` 不能 trace 这些 Linear，且若 ``bwd_mode="int8"`` 需要 triton。
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from training.families.krea2_int8.vendor import convrot_int8_utils as _vendor
from training.families.krea2_int8.vendor.convrot_int8_kernels import HAS_TRITON

logger = logging.getLogger(__name__)


#: 默认 ConvRot group size（与 musubi/ComfyUI 一致；为 4 的幂，见 vendor 说明）
DEFAULT_CONVROT_GROUPSIZE = _vendor.CONVROT_GROUPSIZE

#: 与 fp8 路径对齐的 target/exclude 范围：量化所有主块（blocks.*）里的 Linear 权重，
#: 排除调制（mod.）、RMSNorm 与文本融合（txtfusion）——后者需保持计算精度。
INT8_TARGET_KEYS = ["blocks."]
INT8_EXCLUDE_KEYS = ["mod.", "norm", "txtfusion"]


def convrot_available() -> bool:
    """是否可用 Triton 融合 INT8 核（不可用时退化为 eager dequant 路径）。"""
    return HAS_TRITON


def build_convrot_quantizer(
    target_keys: Sequence[str] | None = INT8_TARGET_KEYS,
    exclude_keys: Sequence[str] | None = INT8_EXCLUDE_KEYS,
    allowed_groupsizes: Sequence[int] = (DEFAULT_CONVROT_GROUPSIZE,),
) -> _vendor.ConvRotInt8Quantizer:
    """构造 vendored 的 ConvRot INT8 量化器（包装，参数透传）。"""
    return _vendor.ConvRotInt8Quantizer(
        target_layer_keys=list(target_keys) if target_keys is not None else None,
        exclude_layer_keys=list(exclude_keys) if exclude_keys is not None else None,
        allowed_groupsizes=tuple(allowed_groupsizes),
    )


def quantize_state_dict(
    quantizer: _vendor.ConvRotInt8Quantizer,
    path: str,
    calc_device: torch.device,
    *,
    move_to_device: bool = True,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """用 vendored 量化器把 ``path`` 里的 bf16/fp16 底模流式量化成 ConvRot INT8。

    返回的 state_dict 键与 fp8 路径一致：目标 Linear 的 ``{layer}.weight`` 为
    int8（旋转基），并新增 ``{layer}.scale_weight`` F32 标量。``calc_device``
    为量化计算设备（旋转/量化在 GPU 上跑，避免 CPU 极慢）。``move_to_device``
    为 True 时张量直接落在 ``calc_device``（本仓库训练即此语义）。

    ``dtype``（可选）：统一把**未量化**（passthrough）的浮点权重 cast 到该 dtype，
    使前向里 bf16 latent 与 bf16 权重 dtype 一致（int8 权重与 F32 scale 不受影响）。
    训练时应传底模计算 dtype，避免 checkpoint 存储 dtype（可能 fp32）导致 dtype 错配。
    """
    sd = quantizer.load_and_quantize(
        model_files=[path],
        calc_device=calc_device,
        move_to_device=move_to_device,
        weight_hook=None,
    )
    if dtype is not None:
        for key in list(sd.keys()):
            tensor = sd[key]
            if tensor.dtype == torch.int8:
                continue  # int8 权重（旋转基）不 cast
            if key.endswith(".scale_weight"):
                continue  # ConvRot scale 恒为 F32，前向/反向依赖其 fp32
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                continue
            sd[key] = tensor.to(dtype)
    return sd


def patch_convrot_int8_linears(
    model: torch.nn.Module,
    optimized_state_dict: dict[str, torch.Tensor],
    *,
    bwd_mode: str = "bf16",
) -> int:
    """给持有 ConvRot INT8 权重的 Linear 挂 fused 前向 patch；返回 patch 数。

    封装 ``vendor.apply_convrot_int8_monkey_patch``：遍历 ``optimized_state_dict``
    里的 ``{layer}.scale_weight``，找到对应 ``nn.Linear``，注册非持久
    ``scale_weight`` buffer 并替换 ``forward``。``bwd_mode``：
    ``"bf16"``（默认，瞬时反量化最准）或 ``"int8"``（复用融合 GEMM，更快，需 triton）。
    """
    if bwd_mode not in ("bf16", "int8"):
        raise ValueError(f"int8 backward 模式仅支持 bf16/int8，收到：{bwd_mode}")
    if bwd_mode == "int8" and not HAS_TRITON:
        raise ValueError("int8 反向模式需要 triton（Windows 用 triton-windows）")
    _vendor.apply_convrot_int8_monkey_patch(
        model,
        optimized_state_dict,
        bwd_mode=bwd_mode,
        groupsize=DEFAULT_CONVROT_GROUPSIZE,
        groupsize_map=quantizer_module_groupsizes(optimized_state_dict),
    )
    return int(getattr(model, "convrot_int8_layer_count", 0))


def quantizer_module_groupsizes(optimized_state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """从量化 state_dict 推导 per-module group size（当前固定 256）。

    vendored 的 ``apply_convrot_int8_monkey_patch`` 需要 per-module group size 映射；
    动态量化路径（本 loader 用）恒用 ``DEFAULT_CONVROT_GROUPSIZE``，故这里构造
    等价映射。若未来支持 ComfyUI 预量化多 group size，应改由 quantizer 携带。
    """
    return {
        layer: DEFAULT_CONVROT_GROUPSIZE
        for layer in _scale_weight_layers(optimized_state_dict)
    }


def _scale_weight_layers(state_dict: dict[str, torch.Tensor]) -> list[str]:
    return [
        key[: -len(".scale_weight")]
        for key in state_dict if key.endswith(".scale_weight")
    ]


def model_has_int8_layers(model: object) -> bool:
    """采样/LoRA 路径判断底模是否为 ConvRot INT8 量化形态。

    非 nn.Module（测试 fake / 尚未加载）一律 False。以 patched 后模块持有的
    ``scale_weight`` buffer 或 int8 权重为判据。
    """
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    for m in modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if m.weight.dtype == torch.int8 or hasattr(m, "scale_weight"):
            return True
    return False
