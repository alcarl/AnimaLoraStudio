"""VAEWrapper OOM-fallback tiled decode 单测（issue #200）。

tiled encode/decode 已从 cosine-blend+acc/wsum 方案迁移到 musubi-tuner /
diffusers 官方 tiled 算法（固定 tile/stride + 相邻 tile 线性 blend + 前 stride
裁减拼接）。

测：
- `_blend_v` / `_blend_h` 线性 blend 数值正确性 + 边界 clamp
- `VAEWrapper.decode` 走 try-full 路径不触发 tile
- `VAEWrapper.decode` 在 model.decode 抛 OOM 时走 tile fallback 并产出形状一致的结果
- `_tiled_decode` / `_tiled_encode` 拼接：mock 一个反映 WanVAE 内部结构的线性
  模型，tile 重建结果与整图重建数值一致（验证固定 stride 裁减 + 线性 blend 正确性）
"""
from __future__ import annotations

import logging

import pytest
import torch

from training.vae import (
    VAEWrapper,
    _blend_h,
    _blend_v,
)


# ---------------------------------------------------------------------------
# _blend_v / _blend_h
# ---------------------------------------------------------------------------


def test_blend_v_linear_interpolation_correct() -> None:
    """垂直 blend：b 顶部 blend_extent 行 = a 底部与 b 顶部的线性插值。"""
    a = torch.full((1, 1, 1, 8, 8), 0.0)   # 上方 tile 值 0
    b = torch.full((1, 1, 1, 8, 8), 10.0)  # 当前 tile 值 10
    blend = 4
    out = _blend_v(a, b, blend)
    # y=0 → 全取 a (=0)；y=3 → 接近全取 b (=10)
    assert out[0, 0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[0, 0, 0, 3, 0].item() == pytest.approx(10.0 * 3 / 4, abs=1e-5)
    # blend 之外的行不修改
    assert out[0, 0, 0, 5, 0].item() == pytest.approx(10.0, abs=1e-5)


def test_blend_h_linear_interpolation_correct() -> None:
    """水平 blend：b 左部 blend_extent 列 = a 右侧与 b 左侧的线性插值。"""
    a = torch.full((1, 1, 1, 8, 8), 0.0)
    b = torch.full((1, 1, 1, 8, 8), 20.0)
    blend = 4
    out = _blend_h(a, b, blend)
    assert out[0, 0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[0, 0, 0, 0, 3].item() == pytest.approx(20.0 * 3 / 4, abs=1e-5)
    assert out[0, 0, 0, 0, 5].item() == pytest.approx(20.0, abs=1e-5)


def test_blend_extent_clamped_to_min_dim() -> None:
    """blend_extent 超出 tile 维度时 clamp 到 min(a,b,extent)，不越界。"""
    a = torch.zeros(1, 1, 1, 4, 8)
    b = torch.zeros(1, 1, 1, 4, 8)
    out = _blend_v(a, b, 100)  # extent 100 > tile h=4
    assert out.shape == (1, 1, 1, 4, 8)


def test_blend_zero_extent_returns_b_unchanged() -> None:
    a = torch.zeros(1, 1, 1, 8, 8)
    b = torch.full((1, 1, 1, 8, 8), 5.0)
    out = _blend_v(a, b, 0)
    assert torch.all(out == 5.0)


# ---------------------------------------------------------------------------
# VAEWrapper 分块决策
# ---------------------------------------------------------------------------


class _WanLikeLinearModel:
    """反映 WanVAE 内部结构的 mock：
    - 顶层 `encode`/`decode`：8× avg_pool / nearest upsample，带 scale 归一化
    - 内部 `encoder`/`conv1`/`conv2`/`decoder`：tiled 路径用的线性层
    - `z_dim` / `clear_cache` / `_feat_map` / `_conv_idx` / `_enc_feat_map` / `_enc_conv_idx`
    """

    z_dim = 16

    def __init__(self, oom_on_full=False, oom_on_full_enc=False):
        self.calls: list[tuple[int, ...]] = []
        self.enc_calls: list[tuple[int, ...]] = []
        self.oom_on_full = oom_on_full
        self.oom_on_full_enc = oom_on_full_enc
        self.clear_cache()

    def clear_cache(self):
        self._conv_num = 1
        self._conv_idx = [0]
        self._feat_map = [None]
        self._enc_conv_num = 1
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None]

    def encoder(self, x, feat_cache=None, feat_idx=None):
        """线性 encode：8× avg_pool → [b,3,t,h,w]→[b,3,t,h/8,w/8]；补零到 z_dim*2。"""
        b, _c, t, H, W = x.shape
        z = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 8, 8))  # [b,3,t,h/8,w/8]
        if z.shape[1] < self.z_dim * 2:
            z = torch.nn.functional.pad(z, (0, 0, 0, 0, 0, 0, 0, self.z_dim * 2 - z.shape[1]))
        return z

    def conv1(self, x):
        """把 z_dim*2 → z_dim*2（identity），模拟 quant_conv；无归一化。"""
        return x

    def conv2(self, x):
        """z_dim → z_dim（identity），模拟 post_quant_conv。"""
        return x

    def decoder(self, x, feat_cache=None, feat_idx=None):
        """线性 decode：前 3 通道 nearest 8× upsample → [b,3,t,h*8,w*8]。"""
        b, _c, t, h, w = x.shape
        upsampled = torch.nn.functional.interpolate(
            x[:, :3].reshape(b * t, 3, h, w),
            scale_factor=8,
            mode="nearest",
        ).reshape(b, 3, t, h * 8, w * 8)
        return upsampled

    def encode(self, pixels, scale):
        """顶层整图 encode：内部 encoder + conv1 + 归一化。"""
        self.enc_calls.append(tuple(pixels.shape))
        b, _c, t, H, W = pixels.shape
        if self.oom_on_full_enc and H >= 1024 and len(self.enc_calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on full encode")
        raw = self.encoder(pixels)
        mu = self.conv1(raw)[:, : self.z_dim]
        mean = scale[0].view(1, -1, 1, 1, 1)
        std = scale[1].view(1, -1, 1, 1, 1)
        return (mu - mean) * std

    def decode(self, z, scale):
        """顶层整图 decode：反归一化 + conv2 + decoder。"""
        self.calls.append(tuple(z.shape))
        b, _c, t, h, w = z.shape
        if self.oom_on_full and h >= 128 and len(self.calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on full decode")
        mean = scale[0].view(1, -1, 1, 1, 1)
        std = scale[1].view(1, -1, 1, 1, 1)
        z_raw = z / std + mean
        x = self.conv2(z_raw)
        return self.decoder(x)


def _make_wrapper(model, tiling="auto") -> VAEWrapper:
    mean = torch.zeros(16)
    std = torch.ones(16)
    return VAEWrapper(model, mean, std, tiling=tiling)


def test_decode_full_path_calls_model_once() -> None:
    """大显存路径：try full 成功 → 不进 tile，model.decode 只调一次。"""
    model = _WanLikeLinearModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)  # 256×256 输出
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 256, 256)
    assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# VAEWrapper.decode OOM fallback
# ---------------------------------------------------------------------------


def test_decode_oom_fallback_invokes_tiled_decode() -> None:
    """整图 OOM → catch + empty_cache + 分块 decode。

    1024 reg 规格：z = [1,16,1,128,128]；tile=64 stride=48 →
    range(0,128,48) = [0,48,96] → 3×3 = 9 个 latent tile。
    整图调用记录 1 次（先失败）。
    """
    model = _WanLikeLinearModel(oom_on_full=True)
    wrapper = _make_wrapper(model)
    z = torch.zeros(1, 16, 1, 128, 128)  # 1024×1024 reg
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    # 只记录 1 次整图调用（tiled 走内部 decoder，不记录在 model.calls）
    assert len(model.calls) == 1
    assert model.calls[0] == (1, 16, 1, 128, 128)


# ---------------------------------------------------------------------------
# Tile 拼接数值正确性
# ---------------------------------------------------------------------------


def test_tiled_decode_reconstructs_full_for_linear_decoder() -> None:
    """对一个线性 decode（nearest upsample 8×），tile + 线性 blend 拼接
    结果与整图 decode 数值一致。

    用 latent[:, :3] 作 "image"，nearest 8× 上采。每个像素的实际值与位置无
    关，固定 stride 裁减 + 线性 blend 加权后必须恢复原值。
    """
    model = _WanLikeLinearModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    torch.manual_seed(0)
    z = torch.randn(1, 16, 1, 128, 128)

    full = wrapper.decode(z)
    tiled = wrapper._tiled_decode(z)

    # 数值应几乎一致；线性 blend 只引入极小数值误差
    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol=1e-4)


def test_tiled_decode_handles_small_input_without_tiling() -> None:
    """size ≤ tile 时 range 只产出 [0]，等同整图 decode。"""
    model = _WanLikeLinearModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)  # < tile=64
    out = wrapper._tiled_decode(z)
    assert out.shape == (1, 3, 1, 256, 256)


# ---------------------------------------------------------------------------
# tiling 模式（auto / on / off）+ 峰值估算 + auto 判定阈值
# ---------------------------------------------------------------------------


def test_tiling_on_always_tiles_no_full_call() -> None:
    """tiling='on'：直接走 _tiled_decode，没有整图 decode 调用。

    128 latent → range(0,128,48)=[0,48,96] → 9 个 latent tile。
    """
    model = _WanLikeLinearModel(oom_on_full=False)
    wrapper = _make_wrapper(model, tiling="on")
    z = torch.zeros(1, 16, 1, 128, 128)
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    # tiled 走内部 decoder，不记录顶层调用
    assert len(model.calls) == 0


def test_tiling_off_uses_whole_image_then_oom_net() -> None:
    """tiling='off'：整图优先；仍保留真 OOM 兜底分块（小显存安全网）。"""
    model = _WanLikeLinearModel(oom_on_full=True)
    wrapper = _make_wrapper(model, tiling="off")
    z = torch.zeros(1, 16, 1, 128, 128)
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    # 1 次失败整图（OOM 兜底仍在，tiled 走内部层不记录顶层）
    assert model.calls[0] == (1, 16, 1, 128, 128)
    assert len(model.calls) == 1


def test_est_decode_peak_scales_with_pixels_and_dtype() -> None:
    """峰值估算 ∝ 输出像素 × 元素大小。fp32 1024² ≈ 11.5G，bf16 减半。"""
    model = _WanLikeLinearModel()
    wrapper = _make_wrapper(model)
    z32 = torch.zeros(1, 16, 1, 128, 128, dtype=torch.float32)   # 1024² 输出
    est_fp32 = wrapper._est_decode_peak_bytes(z32)
    assert est_fp32 == pytest.approx(11000 * 1024 * 1024, rel=1e-6)
    z16 = torch.zeros(1, 16, 1, 128, 128, dtype=torch.bfloat16)
    assert wrapper._est_decode_peak_bytes(z16) == pytest.approx(est_fp32 / 2, rel=1e-6)


def test_should_auto_tile_threshold_matches_measured_cliff() -> None:
    """auto 阈值复刻实测崖（RTX 5090 31.8G）：fp32 1536² 分块、其余快路径整图。"""
    GB = 1024 ** 3
    total = int(31.8 * GB)
    model = _WanLikeLinearModel()
    w = _make_wrapper(model)

    def est(res, dtype):
        h = res // 8
        return w._est_decode_peak_bytes(torch.zeros(1, 16, 1, h, h, dtype=dtype))

    light = int(2.2 * GB)  # 仅 VAE 常驻
    # fp32 1024（实测 0.34s 整图安全）→ 不分块
    assert not VAEWrapper._should_auto_tile(light, est(1024, torch.float32), total)
    # fp32 1536（实测整图 196s）→ 分块
    assert VAEWrapper._should_auto_tile(light, est(1536, torch.float32), total)
    # bf16 1536（实测 0.4s 整图安全）→ 不分块
    assert not VAEWrapper._should_auto_tile(light, est(1536, torch.bfloat16), total)
    # bf16 1536 但叠加 ~12G 常驻模型（训练 sample 场景）→ 分块
    heavy = int(13 * GB)
    assert VAEWrapper._should_auto_tile(heavy, est(1536, torch.bfloat16), total)


# ---------------------------------------------------------------------------
# encode 分块（latent 缓存路径）
# ---------------------------------------------------------------------------


def test_encode_full_path_calls_model_once() -> None:
    """CPU（非 cuda）走整图 encode：model.encode 只调一次。"""
    model = _WanLikeLinearModel()
    wrapper = _make_wrapper(model)
    px = torch.randn(1, 3, 1, 256, 256)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 32, 32)
    assert len(model.enc_calls) == 1


def test_encode_tiling_on_tiles_in_pixel_space() -> None:
    """tiling='on'：1024px → 像素 tile=512/stride=384 起点 [0,384,512] → 9 个 512² tile。"""
    model = _WanLikeLinearModel()
    wrapper = _make_wrapper(model, tiling="on")
    px = torch.zeros(1, 3, 1, 1024, 1024)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 128, 128)


def test_encode_oom_fallback_invokes_tiled_encode() -> None:
    """整图 encode OOM → catch + tile 兜底（off 模式也保留安全网）。

    tiled encode 走内部 encoder/conv1，不记录顶层 enc_calls。
    """
    model = _WanLikeLinearModel(oom_on_full_enc=True)
    wrapper = _make_wrapper(model, tiling="off")
    px = torch.zeros(1, 3, 1, 1024, 1024)
    z = wrapper.encode(px)
    assert z.shape == (1, 16, 1, 128, 128)
    assert model.enc_calls[0] == (1, 3, 1, 1024, 1024)  # 先试整图
    assert len(model.enc_calls) == 1                     # 失败后 tiled 走内部层


def test_tiled_encode_reconstructs_full_for_linear_encoder() -> None:
    """线性 encode（avg_pool）下，tile + 线性 blend 拼接 ≈ 整图 encode。"""
    model = _WanLikeLinearModel()
    wrapper = _make_wrapper(model)
    torch.manual_seed(0)
    px = torch.randn(1, 3, 1, 1024, 1024)
    full = wrapper.model.encode(px, wrapper.scale)
    tiled = wrapper._tiled_encode(px)
    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol=1e-4)


def test_should_offload_for_whole_decode_false_on_cpu() -> None:
    """CPU 张量（无 cuda）下不 offload：守卫返回 False，不触碰 mem_get_info。"""
    wrapper = _make_wrapper(_WanLikeLinearModel())
    z = torch.zeros(1, 16, 1, 128, 128)  # CPU
    assert wrapper.should_offload_for_whole_decode(z) is False


def test_est_encode_peak_scales_with_pixels_and_dtype() -> None:
    """encode 峰值估算 ∝ 输入像素 × 元素大小。fp32 1024² ≈ 5.5G，bf16 减半。"""
    wrapper = _make_wrapper(_WanLikeLinearModel())
    px32 = torch.zeros(1, 3, 1, 1024, 1024, dtype=torch.float32)
    est_fp32 = wrapper._est_encode_peak_bytes(px32)
    assert est_fp32 == pytest.approx(5500 * 1024 * 1024, rel=1e-6)
    px16 = torch.zeros(1, 3, 1, 1024, 1024, dtype=torch.bfloat16)
    assert wrapper._est_encode_peak_bytes(px16) == pytest.approx(est_fp32 / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# 分块决策 / OOM 回退日志去重（缓存逐图调用不刷屏）
# ---------------------------------------------------------------------------


def test_log_once_same_key_logs_only_first(caplog) -> None:
    """同一 key 调多次只记一次：latent 缓存逐 bucket 调 encode 不刷屏。"""
    wrapper = _make_wrapper(_WanLikeLinearModel())
    logger = logging.getLogger("training.vae")
    with caplog.at_level(logging.INFO, logger="training.vae"):
        for i in range(200):
            wrapper._log_once("auto_encode", logger.info, "主动分块 #%d", i)
    hits = [r for r in caplog.records if "主动分块" in r.getMessage()]
    assert len(hits) == 1
    assert "#0" in hits[0].getMessage()  # 记的是首次那条


def test_log_once_distinct_keys_each_logged_once(caplog) -> None:
    """不同事件（encode/decode 决策、OOM 回退）各自独立去重，互不抑制。"""
    wrapper = _make_wrapper(_WanLikeLinearModel())
    logger = logging.getLogger("training.vae")
    with caplog.at_level(logging.WARNING, logger="training.vae"):
        wrapper._log_once("auto_encode", logger.warning, "encode 分块")
        wrapper._log_once("auto_decode", logger.warning, "decode 分块")
        wrapper._log_once("auto_encode", logger.warning, "encode 分块")  # 重复，抑制
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == ["encode 分块", "decode 分块"]
