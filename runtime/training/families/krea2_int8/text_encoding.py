"""Krea2 int8 文本编码复用层。

int8 底模与 fp8/bf16 底模共用同一个 Qwen3-VL 文本编码器与缓存协议，故复用
``families/krea2/text_encoding.py`` 的全部实现，本模块只做转发封装。
"""

from __future__ import annotations

from training.families.krea2.text_encoding import (  # noqa: F401
    KREA2_MAX_LENGTH,
    KREA2_SELECTED_LAYERS,
    KREA2_TEXT_FINGERPRINT,
    KREA2_TEXT_WIDTH,
    Krea2TextCondition,
    Krea2TextStack,
    load_krea2_text_stack,
)

__all__ = [
    "KREA2_MAX_LENGTH",
    "KREA2_SELECTED_LAYERS",
    "KREA2_TEXT_FINGERPRINT",
    "KREA2_TEXT_WIDTH",
    "Krea2TextCondition",
    "Krea2TextStack",
    "load_krea2_text_stack",
]
