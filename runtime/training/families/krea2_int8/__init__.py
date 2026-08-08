"""Krea2 int8（ConvRot INT8）ModelFamily 实现。

与 ``families/krea2``（fp8/bf16 krea2 家族）**并列**的独立家族：底模在加载期动态
量化为 ConvRot INT8（frozen、int8_base 语义），前向走 fused Triton GEMM，LoRA
参数全精度叠加。采样/文本编码/preset 与 fp8 家族共用同一实现（架构与 TE 相同），
仅底模加载路径不同——这里通过复制自 musubi-tuner 的 vendored convrot int8 代码封装。

登记与 fp8 家族同款：family_id 走 ``"krea2_int8"``（与 ``"krea2"`` 并列的独立族），
供训练管线按族 dispatch 到底模加载（见 families/__init__.py 的 get_family）。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

from .preset import KREA2_PRESET
from .sampling import (
    KREA2_RAW_GUIDANCE,
    KREA2_RAW_STEPS,
    KREA2_SAMPLER,
    KREA2_SCHEDULER,
    Krea2SamplingCondition,
    prepare_sampling_condition,
    resolve_sampling_settings,
    sample_image,
)
from .text_encoding import (
    KREA2_TEXT_FINGERPRINT,
    Krea2TextCondition,
    Krea2TextStack,
    load_krea2_text_stack,
)
from ..krea2 import Krea2Family
from ..latent_spaces import WAN21_F8C16
from ..spec import (
    ConstantShift,
    LoraOutputSpec,
    ModelSpec,
    SamplingDefaults,
    TextSpec,
)
from studio.domain.common import (
    FAMILY_CAPABILITIES,
    FAMILY_CONFIG_DEFAULTS,
    FAMILY_SAMPLING,
)


logger = logging.getLogger(__name__)


KREA2_INT8_SPEC = ModelSpec(
    family_id="krea2_int8",
    display_name="Krea 2 (INT8)",
    objective="rectified_flow",
    latent=WAN21_F8C16,
    text=TextSpec(
        strategy="cached_varlen",
        max_seq_len=512,
        fingerprint=KREA2_TEXT_FINGERPRINT,
    ),
    sampling=SamplingDefaults(
        samplers=FAMILY_SAMPLING["krea2_int8"]["samplers"],
        schedulers=FAMILY_SAMPLING["krea2_int8"]["schedulers"],
        default_sampler=KREA2_SAMPLER,
        default_scheduler=KREA2_SCHEDULER,
        default_steps=KREA2_RAW_STEPS,
        default_cfg=KREA2_RAW_GUIDANCE,
        shift_policy=ConstantShift(shift=1.15),
    ),
    capabilities=FAMILY_CAPABILITIES["krea2_int8"],
    lora=LoraOutputSpec(prefix="lora_unet", preset_name="krea2_full"),
    config_defaults=FAMILY_CONFIG_DEFAULTS["krea2_int8"],
)


class Krea2Int8Family(Krea2Family):
    """int8 底模的 krea2 家族：仅覆盖底模加载，其余复用 ``Krea2Family``。"""

    spec = KREA2_INT8_SPEC

    def load_dit(self, path, device, dtype, *,
                 attention_backend: str = "flash_attn", repo_root=None,
                 purpose: str = "train", blocks_to_swap: int = 0):
        from training.families.krea2_int8.loader import load_krea2_int8_model

        if attention_backend != "none":
            logger.info(
                "Krea2 int8 当前固定使用 PyTorch SDPA；忽略 attention_backend=%s",
                attention_backend,
            )
        return load_krea2_int8_model(
            path, device, dtype, purpose=purpose, blocks_to_swap=blocks_to_swap,
        )


__all__ = [
    "KREA2_INT8_SPEC",
    "Krea2Int8Family",
    "Krea2SamplingCondition",
    "Krea2TextCondition",
    "Krea2TextStack",
    "load_krea2_text_stack",
    "prepare_sampling_condition",
    "sample_image",
]
