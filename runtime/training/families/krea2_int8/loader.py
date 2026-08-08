"""Krea2 int8（ConvRot INT8）底模加载与冻结。

与 ``families/krea2/loader.py``（fp8/bf16 loader）**并列**的独立 loader：把单文件
bf16/fp16 的 Krea2 Raw 底模在加载期动态量化成 ConvRot INT8（Hadamard 旋转 + 行内
INT8），前向走 fused Triton GEMM（无 triton 时退化 eager dequant），并冻结底模——
即 kohya/musubi 生态的「int8_base」语义（底模 frozen 无梯度，LoRA 参数全精度，
显存收益依赖 grad checkpointing）。

本 loader **只接受 bf16/fp16 底模作为量化输入**（int8 由加载期动态量化产生），
与 fp8 loader 的「直接读入 fp8 权重」路径不同；两者都是把「量化」当作 checkpoint
的派生属性而非模型族开关。

实现上调用复制自 musubi-tuner 的 vendored 代码：``quant_int8.build_convrot_quantizer``
+ ``quant_int8.quantize_state_dict`` 产出 int8 state dict，``patch_convrot_int8_linears``
挂上 fused 前向。结构校验（单文件、键/形状指纹）复用本仓库 krea2 家族的一致口径。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors import safe_open

from modeling.krea2 import Krea2Config, SingleStreamDiT
from training.families.krea2_int8 import quant_int8

logger = logging.getLogger(__name__)


#: 动态量化输入只接受浮点 dtype（bf16/fp16/fp32）
_FLOAT_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
#: safetensors header 字符串 → 是否「已是 int8」形态（无法再作为量化输入）
_INT8_DTYPES = {"I8"}


def _checkpoint_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Krea2 int8 checkpoint 不存在：{resolved}")
    if resolved.is_dir():
        raise ValueError(
            "Krea2 int8 loader 需要单文件 raw.safetensors，不能传 diffusers transformer "
            f"分片目录：{resolved}"
        )
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError(f"Krea2 int8 checkpoint 必须是 .safetensors 文件：{resolved}")
    return resolved


def _validate_input(path: str | Path) -> None:
    """header 只读校验输入底模形态（不读 payload）。

    接受两类输入：
    1) bf16/fp16/fp32 底模 —— 加载期由 vendored 量化器动态量化为 ConvRot INT8；
    2) ComfyUI 预量化的 ConvRot INT8 checkpoint —— ``.comfy_quant`` + ``.weight_scale``
       + int8 ``.weight`` 三元组，由 vendored ``load_and_quantize`` 原地转成 Musubi
       layout（``.scale_weight``）直接加载。

    只有「已是 int8 权重但缺少 ComfyUI ConvRot 规格（``.comfy_quant``）」的畸形输入才
    拒绝 —— 这种文件既不能动态量化（权重已非浮点）也不是合法预量化 layout。
    """
    checkpoint = _checkpoint_path(path)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        int8_weights: list[str] = []
        has_comfy_spec = False
        for key in handle.keys():
            if key.endswith(quant_int8.COMFY_QUANT_SUFFIX):
                has_comfy_spec = True
                continue
            dtype_str = str(handle.get_slice(key).get_dtype()).upper()
            if dtype_str in _INT8_DTYPES:
                int8_weights.append(key)
    if not int8_weights:
        return  # 纯浮点底模，正常动态量化
    if has_comfy_spec:
        return  # 合法 ComfyUI 预量化 ConvRot INT8，vendored 代码直接加载
    raise ValueError(
        f"Krea2 int8 loader 输入 {int8_weights[0]} 已是 int8 权重（I8），但缺少 ComfyUI "
        "ConvRot 预量化规格（.comfy_quant + .weight_scale）。本 loader 只接受：bf16/fp16 "
        "底模（加载期动态量化）或 ComfyUI 预量化的 ConvRot INT8 checkpoint。"
    )


def load_krea2_int8_model(
    path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    config: Krea2Config | None = None,
    purpose: str = "train",
    blocks_to_swap: int = 0,
    bwd_mode: str = "bf16",
) -> SingleStreamDiT:
    """加载并动态量化一个 Krea2 底模为 ConvRot INT8，冻结后返回。

    ``path``：单文件 bf16/fp16 safetensors。``dtype``：量化计算/非量化层 dtype
    （bf16）。``bwd_mode``：int8 反向模式（``"bf16"`` 瞬时反量化 / ``"int8"`` 需 triton）。

    ``blocks_to_swap`` 当前保留参数位（与 fp8 loader 对齐），int8 路径尚未实现
    block swap 下的 per-block 量化流式编排，>0 时抛错以免静默退化。
    """
    if dtype not in _FLOAT_INPUT_DTYPES:
        raise ValueError(f"Krea2 int8 loader 的计算 dtype 必须是浮点：{dtype}")
    if blocks_to_swap > 0:
        raise NotImplementedError(
            "Krea2 int8（ConvRot）当前不支持 block swap：int8 权重需在加载期动态量化，"
            "block swap 的 CPU 流式编排尚未接线。请先关闭 block swap 使用 int8 训练。"
        )
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("Krea2 int8 loader 的目标 device 不能是 meta")

    checkpoint = _checkpoint_path(path)
    _validate_input(checkpoint)

    if config is None:
        config = Krea2Config()
    with torch.device("meta"):
        model = SingleStreamDiT(config)

    # 1) 用 vendored 量化器流式动态量化目标 Linear（blocks.*），产出 int8 state dict
    quantizer = quant_int8.build_convrot_quantizer()
    logger.info("Krea2 int8：动态量化 %s 为 ConvRot INT8（bwd=%s）", checkpoint, bwd_mode)
    sd = quant_int8.quantize_state_dict(
        quantizer,
        str(checkpoint),
        target_device,
        move_to_device=True,
        dtype=dtype,
    )

    # 2) 先挂前向 patch（注册 scale_weight buffer），再 assign-load int8 权重。
    #    int8 张量不能作为 requires_grad=True 的 Parameter（仅浮点/复数 dtype 可），
    #    且 assign=True 会用 meta 参数的 requires_grad 重新包裹输入张量——因此必须
    #    **先** requires_grad_(False) 再 load（musubi 同款处理，见 krea2_utils.py）。
    quant_int8.patch_convrot_int8_linears(model, sd, bwd_mode=bwd_mode)
    model.requires_grad_(False)
    model.load_state_dict(sd, strict=True, assign=True)
    del sd
    logger.info("Krea2 int8：已加载并冻结 %d 个 ConvRot INT8 Linear", model.convrot_int8_layer_count)
    return model


def checkpoint_contains_int8(path: str | Path) -> bool:
    """轻量探测：safetensors header 里是否有 int8 权重（不读 payload）。

    供训练启动期防呆用（见 phases/models.py 的 int8_base 校验）。非 safetensors /
    读失败返回 False——真正的结构校验由 loader 兜底。
    """
    try:
        checkpoint = _checkpoint_path(path)
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            return any(
                str(handle.get_slice(key).get_dtype()).upper() in _INT8_DTYPES
                for key in handle.keys()
            )
    except Exception:  # noqa: BLE001
        return False
