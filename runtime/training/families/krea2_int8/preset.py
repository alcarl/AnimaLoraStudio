"""Krea2 int8 LoRA/LoKr target preset（与 fp8 家族共用）。

int8 底模与 fp8/bf16 底模共用同一套 LoRA 目标范围（全部 264 个 ``nn.Linear``，
含 attention gates 与 text-fusion stack），故直接复用 ``families/krea2/preset.py``
的 ``KREA2_PRESET``，不另造一份。
"""

from __future__ import annotations

from typing import Any

from training.families.krea2.preset import KREA2_PRESET


def krea2_int8_preset() -> dict[str, Any]:
    """返回与 fp8/bf16 krea2 相同的 LoRA preset（目标全 Linear）。"""
    return dict(KREA2_PRESET)


__all__ = ["KREA2_PRESET", "krea2_int8_preset"]
