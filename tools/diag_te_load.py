"""Diagnose why Qwen3-VL TE safetensors -> .to(device) is slow.

Run from the training venv on the Linux box:
    python tools/diag_te_load.py /path/to/models/text_encoders/Qwen_Qwen3-VL-4B-Instruct

Prints per-stage timings to locate the bottleneck (mmap read vs H2D copy vs cast).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open


def main(model_dir: str) -> None:
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        print(f"no index at {index_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}  vram_free={torch.cuda.mem_get_info()[0] / 2**30:.1f}GB")
    import psutil
    print(f"ram_available={psutil.virtual_memory().available / 2**30:.1f}GB")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    shards = list(dict.fromkeys(weight_map.values()))
    print(f"shards={len(shards)}  tensors={len(weight_map)}")

    # Pick the largest single shard.
    shard_sizes = []
    for shard in shards:
        shard_sizes.append((shard, (model_path / shard).stat().st_size))
    shard_sizes.sort(key=lambda x: x[1], reverse=True)
    print("top shards:", [(s, round(b / 2**30, 2)) for s, b in shard_sizes[:3]], "GB")

    # Largest tensor in the largest shard.
    big_shard, big_bytes = shard_sizes[0]
    big_key = None
    big_shape = None
    with safe_open(str(model_path / big_shard), framework="pt", device="cpu") as h:
        for k in h.keys():
            info = h.get_slice(k).get_shape()
            if big_key is None or _numel(info) > _numel(big_shape):
                big_key, big_shape = k, info
    print(f"largest tensor: {big_key} shape={big_shape} in {big_shard}")

    # 1) get_tensor (mmap) time only.
    t0 = time.perf_counter()
    with safe_open(str(model_path / big_shard), framework="pt", device="cpu") as h:
        t = h.get_tensor(big_key)
    t1 = time.perf_counter()
    print(f"[get_tensor]  {big_key}: {t1 - t0:.3f}s  dtype={t.dtype}  elem={_numel(t.shape) / 1e9:.2f}B")

    # 2) .to(device) H2D copy time (pageable source).
    t0 = time.perf_counter()
    t_gpu = t.to(device=device)
    if device != "cpu":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[to_device]        {big_key}: {t1 - t0:.3f}s  (pageable src)")

    # 2b) .to(device) via pinned memory (DMA path) — expected much faster on ROCm.
    t_pin = t.pin_memory()
    t0 = time.perf_counter()
    _ = t_pin.to(device=device, non_blocking=True)
    if device != "cpu":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[to_device_pinned] {big_key}: {t1 - t0:.3f}s  (pinned src)")

    # 3) cast-to-bf16 path (only if source isn't already bf16).
    if t.dtype != torch.bfloat16:
        t0 = time.perf_counter()
        _ = t.to(device=device, dtype=torch.bfloat16)
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"[to_bf16]     {big_key}: {t1 - t0:.3f}s   (source was {t.dtype})")
    else:
        print(f"[to_bf16]     source already bf16, cast is no-op")

    # 4) Whole-shard read via get_tensor loop (real file I/O + cpu buffers).
    t0 = time.perf_counter()
    n = 0
    with safe_open(str(model_path / big_shard), framework="pt", device="cpu") as h:
        for k in h.keys():
            _ = h.get_tensor(k)
            n += 1
    t1 = time.perf_counter()
    print(f"[read_all_cpu] {big_shard}: {t1 - t0:.3f}s  ({n} tensors, file {big_bytes / 2**30:.2f}GB)")

    # 5) Per-tensor move to device across the whole shard (mirrors loader).
    t0 = time.perf_counter()
    with safe_open(str(model_path / big_shard), framework="pt", device="cpu") as h:
        for k in h.keys():
            _ = h.get_tensor(k).to(device=device)
    if device != "cpu":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"[read_to_gpu]  {big_shard}: {t1 - t0:.3f}s")


def _numel(shape):
    if shape is None:
        return 0
    n = 1
    for s in shape:
        n *= s
    return n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tools/diag_te_load.py <model_dir>")
        sys.exit(1)
    main(sys.argv[1])
