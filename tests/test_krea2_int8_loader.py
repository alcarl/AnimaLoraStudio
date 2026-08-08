"""Krea2 int8（ConvRot INT8）loader 的冒烟/回归测试。

覆盖：从 bf16 底模动态量化出 ConvRot INT8、前向 fused patch、底模冻结、
int8 权重/scale 布局，以及 int8_base 组合校验（phases/models）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from modeling.krea2 import Krea2Config, SingleStreamDiT
from training.families.krea2_int8.loader import (
    checkpoint_contains_int8,
    load_krea2_int8_model,
)
from training.families.krea2_int8.quant_int8 import (
    DEFAULT_CONVROT_GROUPSIZE,
    model_has_int8_layers,
)

#: 块内 Linear 的 in_features 需被 256 整除才会被 ConvRot 量化；用 features=256、
#: multiplier=4（mlpdim 圆整到 768）使 attn/mlp 的 gate/up/down 全部可量化。
def _int8_tiny_config() -> Krea2Config:
    return Krea2Config(
        features=256,
        tdim=32,
        txtdim=64,
        heads=8,
        kvheads=4,
        multiplier=4,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=3,
        txtheads=8,
        txtkvheads=2,
    )


def _state_dict(config: Krea2Config) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        key: value.detach().contiguous()
        for key, value in SingleStreamDiT(config).state_dict().items()
    }


def _write_checkpoint(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "",
) -> None:
    save_file({f"{prefix}{key}": value for key, value in state_dict.items()}, str(path))


def test_load_int8_quantizes_and_freezes(tmp_path: Path) -> None:
    config = _int8_tiny_config()
    checkpoint = tmp_path / "raw.safetensors"
    _write_checkpoint(checkpoint, _state_dict(config))

    model = load_krea2_int8_model(
        checkpoint, device="cpu", dtype=torch.bfloat16, config=config
    )

    assert isinstance(model, SingleStreamDiT)
    assert getattr(model, "is_convrot_int8", False) is True
    assert model.convrot_int8_layer_count >= 1
    assert all(not p.requires_grad for p in model.parameters())
    # 目标块 Linear 持有 int8 权重（旋转基）与 F32 scale buffer
    wq = model.blocks[0].attn.wq
    assert wq.weight.dtype == torch.int8
    assert hasattr(wq, "scale_weight")
    assert tuple(wq.scale_weight.shape) == (wq.out_features, 1)


def test_load_int8_forward_does_not_crash(tmp_path: Path) -> None:
    config = _int8_tiny_config()
    checkpoint = tmp_path / "raw.safetensors"
    _write_checkpoint(checkpoint, _state_dict(config))

    model = load_krea2_int8_model(
        checkpoint, device="cpu", dtype=torch.bfloat16, config=config
    )
    batch, c, h, w = 2, config.channels, 16, 16
    x = torch.randn(batch, c, h, w, dtype=torch.bfloat16)
    timesteps = torch.rand(batch, dtype=torch.bfloat16)
    context = torch.randn(
        batch, 6, config.txtlayers * config.txtdim, dtype=torch.bfloat16
    )
    attention_mask = torch.ones(batch, 6, dtype=torch.bool)

    with torch.no_grad():
        out = model(x, timesteps, context, attention_mask=attention_mask)

    assert tuple(out.shape) == (batch, c, h, w)
    assert torch.isfinite(out).all()


def test_load_int8_detects_base(tmp_path: Path) -> None:
    config = _int8_tiny_config()
    checkpoint = tmp_path / "raw.safetensors"
    _write_checkpoint(checkpoint, _state_dict(config))

    assert checkpoint_contains_int8(checkpoint) is False
    model = load_krea2_int8_model(
        checkpoint, device="cpu", dtype=torch.bfloat16, config=config
    )
    assert model_has_int8_layers(model) is True


def test_load_int8_rejects_prequantized_int8_input(tmp_path: Path) -> None:
    """loader 的输入必须是 bf16/fp16 底模；已是 int8 的输入直接拒绝。"""
    config = _int8_tiny_config()
    sd = _state_dict(config)
    sd["blocks.0.attn.wq.weight"] = torch.zeros(
        sd["blocks.0.attn.wq.weight"].shape, dtype=torch.int8
    )
    checkpoint = tmp_path / "prequant.safetensors"
    save_file(sd, str(checkpoint))

    assert checkpoint_contains_int8(checkpoint) is True
    with pytest.raises(ValueError, match="已是 int8"):
        load_krea2_int8_model(checkpoint, device="cpu", dtype=torch.bfloat16, config=config)


def test_load_int8_rejects_block_swap(tmp_path: Path) -> None:
    config = _int8_tiny_config()
    checkpoint = tmp_path / "raw.safetensors"
    _write_checkpoint(checkpoint, _state_dict(config))

    with pytest.raises(NotImplementedError, match="block swap"):
        load_krea2_int8_model(
            checkpoint, device="cpu", dtype=torch.bfloat16, config=config, blocks_to_swap=1
        )


def test_default_groupsize_is_power_of_four() -> None:
    # ConvRot 正则 Hadamard 只对 4 的幂 size 存在；默认 256 符合。
    assert DEFAULT_CONVROT_GROUPSIZE == 256
    value = DEFAULT_CONVROT_GROUPSIZE
    assert value >= 4 and (value & (value - 1)) == 0
