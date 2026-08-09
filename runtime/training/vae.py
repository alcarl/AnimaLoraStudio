"""Anima Transformer / VAE / 文本编码器加载（公开 API，sister script 直接 import）。

抽自原 runtime/anima_train.py L614-775（ADR 0003 PR-A）。

公开（被 anima_daemon / anima_generate / anima_reg_ai 通过 anima_train.X 调用）：
- load_anima_model — Anima Transformer + flash_attn 开关 + checkpoint 推断配置
- load_vae — WAN VAE + 归一化 wrapper
- load_text_encoders — Qwen + T5 tokenizer

内部：
- ensure_models_namespace — 把模型代码目录加进 sys.path
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from training.model_loading import (
    _load_safetensors_state_dict,
    _load_weights_best_effort,
)


logger = logging.getLogger(__name__)


class VAEWrapper:
    """WAN VAE + 归一化参数 + 整图 decode OOM 自动回退到 tiled decode。

    Issue #200：8GB 类小显存跑 1024×1024 reg 生成时 transformer/Qwen/T5 常驻
    GPU 后剩 ~1GB，整图 decode 工作内存吃满 → OOM。本类在 `decode()` 入口先
    `try` 整图，CUDA OOM 才走 cosine-blend 切块 decode（每 tile 64 latent /
    512 pixel，单 tile 工作峰值约 75MB）。

    后续（VRAM 崖修复）：`decode()` / `encode()` 都按 `tiling`（auto/on/off）决策，
    auto 在「已用 + 预计峰值」越过总显存阈值时主动分块——不只是等 OOM。大显存卡上
    整图 op 接近占满显存会触发 WDDM 显存换页、单次 op 从 <1s 退化到上百秒，且不抛
    干净 OOM，reactive 兜底救不了，故需 proactive。

    调用方应走 `wrapper.encode(pixels)` / `wrapper.decode(z)`（带分块决策），不要直接
    调 `wrapper.model.encode/decode(...)`（绕过分块）。
    """

    # tile 几何：tile=512px / overlap=128px / 4-stage VAE 8× upsample
    _TILE_LATENT = 64
    _STRIDE_LATENT = 48
    _UPSAMPLE = 8

    # 整图 decode 峰值显存 ≈ _DECODE_PEAK_BYTES_PER_OUT_PX × (elem/4) × 输出像素 × B × T。
    # 标定自 tools/spike/vae_stress.py（RTX 5090 / WAN VAE dim=96）：fp32 1024²≈10.4G、
    # 1536²≈22.6G；bf16 减半。取 11000（实测 ~9.8k）留 ~12% 余量。
    _DECODE_PEAK_BYTES_PER_OUT_PX = 11000
    # 整图 encode 峰值 ≈ _ENCODE_PEAK_BYTES_PER_IN_PX × (elem/4) × 输入像素 × B × T。
    # 标定自 tools/spike/vae_stress.py（fp32）：1024²≈5.5G、1536²≈11.6G、2048²≈20.3G。
    _ENCODE_PEAK_BYTES_PER_IN_PX = 5500
    # auto：当「当前已用 + 预计峰值」超过总显存此比例就分块。崖在 ~50% 而非满显存——
    # fp32 1536²(峰值 22.6G / 总 31.8G = 71%) 即便「装得下」也会因 WDDM 显存换页退化到
    # ~190s；fp32 1024²(10.4G / 33%) 正常。0.5 让快路径(fp32≤1024 / bf16≤1536)走整图，
    # 把会撞崖的(大图、fp32、或叠加常驻模型)切到分块。
    _TILE_VRAM_FRACTION = 0.5

    def __init__(self, model, mean, std, tiling: str = "auto"):
        self.model = model
        self.mean = mean
        self.std = std
        self.scale = [mean, 1.0 / std]
        # VAE 权重精度（mean/std 与 model 同 dtype）。encode/decode 入口按此 cast 输入，
        # 这样 fp16 训练 + fp32 VAE 时调用方传 fp16 latent / pixel 也不会 dtype mismatch。
        self.dtype = mean.dtype
        # 分块模式：
        #   auto（默认）= 按 free VRAM 估算，整图峰值逼近可用显存时主动分块；
        #   on          = 始终分块（省显存，慢约 30%）；
        #   off          = 整图，仅真 OOM 时回退分块（旧行为）。
        # auto 解决大显存卡整图 decode 接近占满时触发「系统内存回退」→ 单次 decode
        # 从 <1s 退化到上百秒的卡死（reactive OOM 兜底救不了，因为没抛干净 OOM）。
        self.tiling = str(tiling or "auto").lower().strip()
        # 分块决策 / OOM 回退日志去重：latent 缓存逐 bucket 调 encode（200 张不同尺寸
        # 的图 → 200 次调用），每次都记会刷屏。同一 wrapper 实例下每种事件只记一次。
        self._logged_once: set[str] = set()

    def _log_once(self, key: str, log_fn, msg: str, *args) -> None:
        """同一 wrapper 实例下，相同 ``key`` 的日志只发一次（其余静默）。

        缓存阶段 transformer/Qwen 已驻留 GPU（models_phase 早于 dataset_phase），
        free 显存低 → auto 判定对几乎每张图都分块。分块本身是对的（整图 op 会把
        占用推过 WDDM 显存换页崖），只是逐图重复记日志没有信息量，故去重。
        """
        if key in self._logged_once:
            return
        self._logged_once.add(key)
        log_fn(msg, *args)

    def to(self, device):
        """Move the complete VAE wrapper, including non-module scale tensors.

        ``mean``/``std`` are plain tensors rather than registered buffers on the
        underlying WAN VAE.  Moving only ``model`` therefore leaves decode with
        tensors on mixed devices.  Test generation uses this method to park the
        VAE in system RAM between decodes without keeping any VAE weights in
        VRAM.
        """
        target = torch.device(device)
        self.model.to(target)
        self.mean = self.mean.to(target)
        self.std = self.std.to(target)
        self.scale = [self.mean, 1.0 / self.std]
        return self

    def _est_decode_peak_bytes(self, z) -> int:
        b, _c, t, H, W = z.shape
        out_px = (H * self._UPSAMPLE) * (W * self._UPSAMPLE)
        elem = z.element_size()  # 2=bf16/fp16, 4=fp32
        return int(self._DECODE_PEAK_BYTES_PER_OUT_PX * (elem / 4.0) * out_px * b * max(1, t))

    def _est_encode_peak_bytes(self, pixels) -> int:
        b, _c, t, H, W = pixels.shape  # H/W 为像素分辨率
        in_px = H * W
        elem = pixels.element_size()
        return int(self._ENCODE_PEAK_BYTES_PER_IN_PX * (elem / 4.0) * in_px * b * max(1, t))

    @classmethod
    def _should_auto_tile(cls, used_bytes: int, est_peak_bytes: int, total_bytes: int) -> bool:
        """auto 判定：当前已用 + 预计 decode 峰值 是否越过总显存的分块阈值。"""
        return (used_bytes + est_peak_bytes) > total_bytes * cls._TILE_VRAM_FRACTION

    def should_offload_for_whole_decode(self, z) -> bool:
        """采样 decode 前是否值得把非活跃模块（DiT/Qwen）挪到 CPU 腾显存。

        仅当「显存紧张（当前会分块）**且** 峰值仍在崖下」时 True：此时腾出常驻模块
        就能整图 decode、保住 parity / 无 tile 缝。峰值已越崖（如 fp32 1536²）时 False
        —— 腾显存也救不了整图（崖按总显存比例算、与 free 无关），交给分块更省系统内存。
        """
        if not (torch.is_tensor(z) and z.is_cuda and torch.cuda.is_available()):
            return False
        free, total = torch.cuda.mem_get_info()
        est = self._est_decode_peak_bytes(z)
        return self._should_auto_tile(total - free, est, total) and est < total * self._TILE_VRAM_FRACTION

    def decode(self, z):
        """latent → pixel；按 self.tiling 决定整图 / 分块。

        z: ``[b, 16, t, H, W]`` latent
        return: ``[b, 3, t, H*8, W*8]``（与底层 `WanVAE_.decode` 一致，未 clamp）
        """
        z = z.to(self.dtype)  # 对齐 VAE 权重精度（fp16 训练 / fp32 VAE；dtype 一致时为 no-op）
        if self.tiling == "on":
            return self._tiled_decode(z)

        if self.tiling == "auto" and z.is_cuda and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            est = self._est_decode_peak_bytes(z)
            if self._should_auto_tile(used, est, total):
                # debug 级：缓存阶段模型驻留 → free 低 → 几乎张张分块属正常，
                # 默认不显示，免得「已用 X > 总显存」被误读成显存不足。
                self._log_once(
                    "auto_decode", logger.debug,
                    "VAE decode 主动分块（tiling=auto）：已用 %.1fG + 预计峰值 %.1fG > 总显存 %.1fG×%.2f",
                    used / 1024 ** 3, est / 1024 ** 3, total / 1024 ** 3, self._TILE_VRAM_FRACTION,
                )
                return self._tiled_decode(z)

        # off，或 auto 判定显存够：整图，仍保留 OOM 兜底（小显存卡的安全网）。
        try:
            return self.model.decode(z, self.scale)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log_once(
                "oom_decode", logger.warning,
                "VAE 整图 decode OOM，回退到 tiled decode "
                "(tile=%dpx, overlap=%dpx)（同类后续不再重复）",
                self._TILE_LATENT * self._UPSAMPLE,
                (self._TILE_LATENT - self._STRIDE_LATENT) * self._UPSAMPLE,
            )
            return self._tiled_decode(z)

    def encode(self, pixels):
        """pixel → latent；按 self.tiling 决定整图 / 分块。

        pixels: ``[b, 3, t, H, W]``（H/W 为像素分辨率，须为 8 的倍数）
        return: ``[b, 16, t, H/8, W/8]`` latent（与 `WanVAE_.encode` 一致）
        """
        pixels = pixels.to(self.dtype)  # 对齐 VAE 权重精度（同 decode；dtype 一致时为 no-op）
        if self.tiling == "on":
            return self._tiled_encode(pixels)

        if self.tiling == "auto" and pixels.is_cuda and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            est = self._est_encode_peak_bytes(pixels)
            if self._should_auto_tile(used, est, total):
                # debug 级：缓存阶段模型驻留 → free 低 → 几乎张张分块属正常，
                # 默认不显示，免得「已用 X > 总显存」被误读成显存不足。
                self._log_once(
                    "auto_encode", logger.debug,
                    "VAE encode 主动分块（tiling=auto）：已用 %.1fG + 预计峰值 %.1fG > 总显存 %.1fG×%.2f",
                    used / 1024 ** 3, est / 1024 ** 3, total / 1024 ** 3, self._TILE_VRAM_FRACTION,
                )
                return self._tiled_encode(pixels)

        try:
            return self.model.encode(pixels, self.scale)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log_once(
                "oom_encode", logger.warning,
                "VAE 整图 encode OOM，回退到 tiled encode "
                "(tile=%dpx, overlap=%dpx)（同类后续不再重复）",
                self._TILE_LATENT * self._UPSAMPLE,
                (self._TILE_LATENT - self._STRIDE_LATENT) * self._UPSAMPLE,
            )
            return self._tiled_encode(pixels)

    @torch.no_grad()
    def _tiled_decode(self, z):
        """在 H/W 维度切块独立 decode（移植自 musubi-tuner / diffusers AutoencoderKL
        官方 tiled_decode）：固定 tile / 固定 stride 步进 + 相邻 tile 线性 blend +
        每 tile 只取前 ``stride`` 部分裁减拼接。

        相比旧的 cosine-blend + ``acc/wsum`` 方案：
        - 不做归一化除，边界 tile 的拼接就是最后 tile 的实际输出，**不会因
          wsum→0 / 归一化放大而把 VAE 边缘噪声放大**；
        - 边界 tile 被切片自动 clamp，天然无边界 mask 归零问题。

        输入 ``z`` 为已归一化 latent；先整体反归一化到 raw latent，再逐 tile
        走 ``decoder + conv2``，最后 clamp 到 [-1, 1]（对齐整图 decode 语义）。

        ``z``: [b, 16, t, H, W]（latent 空间）。返回 [b, 3, t, H*8, W*8]（像素）。
        """
        b, _c, t, H, W = z.shape
        up = self._UPSAMPLE
        model = self.model

        # 反归一化：latent = (raw - mean) * (1/std)  ⇒  raw = z / (1/std) + mean = z * std + mean
        mean = self.mean.to(z.dtype)
        std = self.std.to(z.dtype)
        z_raw = z * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)

        # 固定 tile 尺寸（对齐 musubi：tile=64 latent, stride=48 latent，overlap=16）
        tile_lat = self._TILE_LATENT          # 64 latent
        stride_lat = self._STRIDE_LATENT      # 48 latent
        # decoder 在像素空间 upsample 8×，故 blend 与裁减都在像素空间进行
        tile_px = tile_lat * up               # 512 px
        stride_px = stride_lat * up           # 384 px
        blend_px = tile_px - stride_px        # 128 px

        rows = []
        for i in range(0, H, stride_lat):
            row = []
            for j in range(0, W, stride_lat):
                model.clear_cache()
                time = []
                for k in range(t):
                    model._conv_idx = [0]
                    z_tile = z_raw[:, :, k:k + 1, i:i + tile_lat, j:j + tile_lat]
                    x = model.conv2(z_tile)
                    decoded = model.decoder(x, feat_cache=model._feat_map, feat_idx=model._conv_idx)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        model.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = _blend_v(rows[i - 1][j], tile, blend_px)
                if j > 0:
                    tile = _blend_h(row[j - 1], tile, blend_px)
                result_row.append(tile[:, :, :, :stride_px, :stride_px])
            result_rows.append(torch.cat(result_row, dim=-1))

        out = torch.cat(result_rows, dim=3)[:, :, :, :H * up, :W * up]
        # 与整图 decode（self.model.decode）一致：不在此 clamp，由调用方
        # （如 _decode_to_pil）统一 clamp，保证 tile/整图路径语义等价。
        return out

    @torch.no_grad()
    def _tiled_encode(self, pixels, tile_px=None, overlap_px=None):
        """在 H/W 维度切块独立 encode（移植自 musubi-tuner / diffusers 官方
        tiled_encode）：固定 tile / 固定 stride 步进 + 相邻 tile 线性 blend +
        每 tile 只取前 ``stride`` 部分裁减拼接。

        在 **raw latent 层** 操作（不经 scale 归一化），拼出完整 raw mu 后
        最后统一归一化。避免旧的 ``acc/wsum`` cosine 方案在边界 wsum 归零放大噪声。

        ``pixels``: [b, 3, t, H, W]（像素）。``tile_px`` / ``overlap_px``（像素）
        非 None 时可覆盖默认（供缓存分块按 config 传入），须为 ``up`` 的整倍数。
        返回 [b, 16, t, H/8, W/8] 归一化 latent。
        """
        b, _c, t, H, W = pixels.shape
        up = self._UPSAMPLE
        model = self.model

        if tile_px is None:
            tile_px = self._TILE_LATENT * up
            stride_px = self._STRIDE_LATENT * up
        else:
            tile_px = int(tile_px)
            ov_px = int(overlap_px) if overlap_px is not None else (self._TILE_LATENT - self._STRIDE_LATENT) * up
            for _name, _v in (("tile_px", tile_px), ("overlap_px", ov_px)):
                if _v % up != 0:
                    raise ValueError(
                        f"_tiled_encode 要求 {_name}={_v} 是 VAE 下采样 {up} 的整倍数"
                        "（latent 块边界需落整格）。"
                    )
            if not (0 <= ov_px < tile_px):
                raise ValueError(f"overlap_px={ov_px} 必须满足 0 <= overlap < tile_px={tile_px}")
            stride_px = tile_px - ov_px

        lat_h, lat_w = H // up, W // up
        tile_lat = tile_px // up          # 64（默认）
        stride_lat = stride_px // up      # 48（默认）
        blend_lat = tile_lat - stride_lat

        rows = []
        for i in range(0, H, stride_px):
            row = []
            for j in range(0, W, stride_px):
                model.clear_cache()
                time = []
                # WanVAE encoder 时间上按 1/4/4/... 分帧；这里 t=1 恒为单帧
                frame_range = 1 + (t - 1) // 4
                for k in range(frame_range):
                    model._enc_conv_idx = [0]
                    if k == 0:
                        px = pixels[:, :, :1, i:i + tile_px, j:j + tile_px]
                    else:
                        px = pixels[:, :, 1 + 4 * (k - 1):1 + 4 * k, i:i + tile_px, j:j + tile_px]
                    raw = model.encoder(px, feat_cache=model._enc_feat_map, feat_idx=model._enc_conv_idx)
                    raw = model.conv1(raw)
                    time.append(raw)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        model.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = _blend_v(rows[i - 1][j], tile, blend_lat)
                if j > 0:
                    tile = _blend_h(row[j - 1], tile, blend_lat)
                result_row.append(tile[:, :, :, :stride_lat, :stride_lat])
            result_rows.append(torch.cat(result_row, dim=-1))

        raw_mu = torch.cat(result_rows, dim=3)[:, :, :, :lat_h, :lat_w]
        # conv1 输出前 z_dim 通道是 mu（WanVAE_.encode 的 chunk(2, dim=1) 语义）
        mu = raw_mu[:, : self.model.z_dim]
        # 归一化：latent = (mu - mean) * (1/std)
        mean = self.mean.to(mu.dtype)
        std = self.std.to(mu.dtype)
        return ((mu - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)).to(pixels.dtype)


def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """垂直方向相邻 tile 线性 blend（musubi / diffusers 官方实现）。

    ``a`` 为上方 tile，``b`` 为当前 tile；把 ``b`` 顶部 ``blend_extent`` 行替换成
    与 ``a`` 底部的线性插值，消 tile 缝。原地修改 ``b`` 并返回。
    """
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    if blend_extent <= 0:
        return b
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
    return b


def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """水平方向相邻 tile 线性 blend（musubi / diffusers 官方实现）。

    ``a`` 为左侧 tile，``b`` 为当前 tile；把 ``b`` 左部 ``blend_extent`` 列替换成
    与 ``a`` 右侧的线性插值，消 tile 缝。原地修改 ``b`` 并返回。
    """
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    if blend_extent <= 0:
        return b
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
    return b


def load_vae(vae_path, device, dtype, repo_root, *, tiling: str = "auto"):
    """加载 VAE。``tiling`` 透传给 VAEWrapper（auto/on/off）。"""
    # 正常 import（exec-load 退役，多模型 PR-2a）；repo_root 参数保留但不再使用
    from modeling.wan.vae2_1 import WanVAE_ as WanVAE

    cfg = dict(
        dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn_scales=[],
        temperal_downsample=[False, True, True], dropout=0.0,
    )

    model = WanVAE(**cfg).eval().requires_grad_(False)

    sd = _load_safetensors_state_dict(Path(vae_path))
    _load_weights_best_effort(model, sd, label="VAE")
    model = model.to(device=device, dtype=dtype)

    # VAE 归一化参数
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=dtype, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=dtype, device=device)

    logger.info("VAE 加载完成")
    return VAEWrapper(model, mean, std, tiling=tiling)
