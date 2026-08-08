"""Krea2 int8 采样复用层。

int8 底模与 fp8/bf16 底模是同一架构、同一采样器/scheduler/步数/guidance，故全部
复用 ``families/krea2/sampling.py`` 的实现，本模块只做转发封装（便于 int8 家族
引用点集中、语义清晰）。
"""

from __future__ import annotations

from training.families.krea2.sampling import (  # noqa: F401
    KREA2_BASE_IMAGE_SEQ_LEN,
    KREA2_BASE_SHIFT,
    KREA2_FIXED_MU,
    KREA2_MAX_IMAGE_SEQ_LEN,
    KREA2_MAX_SHIFT,
    KREA2_RAW_GUIDANCE,
    KREA2_RAW_STEPS,
    KREA2_SAMPLER,
    KREA2_SCHEDULER,
    KREA2_TURBO_GUIDANCE,
    KREA2_TURBO_STEPS,
    Krea2SamplingCondition,
    prepare_sampling_condition,
    resolve_sampling_settings,
    sample_image,
)

__all__ = [
    "KREA2_BASE_IMAGE_SEQ_LEN",
    "KREA2_BASE_SHIFT",
    "KREA2_FIXED_MU",
    "KREA2_MAX_IMAGE_SEQ_LEN",
    "KREA2_MAX_SHIFT",
    "KREA2_RAW_GUIDANCE",
    "KREA2_RAW_STEPS",
    "KREA2_SAMPLER",
    "KREA2_SCHEDULER",
    "KREA2_TURBO_GUIDANCE",
    "KREA2_TURBO_STEPS",
    "Krea2SamplingCondition",
    "prepare_sampling_condition",
    "resolve_sampling_settings",
    "sample_image",
]
