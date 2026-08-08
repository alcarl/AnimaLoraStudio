#!/usr/bin/env python
"""Anima LoRA Trainer v2 — main() 编排入口。

本模块的实现层已按 ADR 0003 PR-A 拆到 runtime/training/ 子包：
  bootstrap / cli / observability / model_loading / models / text_encoding /
  state / dataset / sampling / timestep_sampling / noise / loss_weighting

顶部的 re-export 段保留 anima_train.X 访问路径，给 sister script
（anima_daemon / anima_generate / anima_reg_ai）和 tests/ 不变。新代码请
直接 `from training.X import Y`。

LoRA / LoKr 实现：见 utils.lycoris_adapter.AnimaLycorisAdapter（ADR 0001）。
"""

import logging
import os
import sys
from pathlib import Path

# 小显存优化：减少 CUDA 显存碎片，缓解 8GB 卡 LoKr full-matrix OOM。
# - 必须在 torch 链式 import 之前设置：torch 在 import 阶段就读 PYTORCH_CUDA_ALLOC_CONF
#   并缓存，之后再改无效。
# - expandable_segments 的 CUDA backend 实现需要 PYTORCH_C10_DRIVER_API_SUPPORTED 宏，
#   PyTorch 的 c10/cuda/CMakeLists.txt 把该宏 gate 在 `if(NOT WIN32)`，因此 Windows wheel
#   不包含该 backend，运行时会 emit `TORCH_WARN_ONCE("expandable_segments not supported
#   on this platform")` 并强制 disable。为避免 Windows 用户看无用 warning，只在 Linux 设。
# - setdefault 不覆盖用户已显式设置的值。
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 脚本在 runtime/ 下按裸脚本启动（`python runtime/anima_train.py`）。
# 把仓库根 + runtime/ 注入 sys.path，让 `import utils.*` / `import train_monitor` /
# `import training.*` 等不需要改成包导入。
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# Windows 控制台默认 cp936，logging / print 写中文会 UnicodeEncodeError，
# 默认 handler 的 errors='backslashreplace' 会把中文转成 \uXXXX 形式 ——
# 这就是 task log 里看到的「检查 VAE」之类乱码的来源。
# 强制 stdout/stderr UTF-8 + replace 让中文 / emoji 永远直出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─── Re-exports for sister script / tests (ADR 0003 PR-A) ────────────────────
# 这些名字被 anima_daemon / anima_generate / anima_reg_ai (`import anima_train as _T`
# 然后 _T.X) 以及 tests/test_anima_train_migration.py 等直接读取。新代码请
# 直接 import 子模块，不要再依赖 anima_train 顶层。
from training.bootstrap import (  # noqa: E402
    apply_yaml_config,
    ensure_dependencies,
    init_progress,
    load_yaml_config,
)
from training.observability import (  # noqa: E402
    WandBMonitor,
    init_wandb_monitor,
    render_curve_panel,
    render_loss_curve,
)
from training.model_loading import (  # noqa: E402
    _load_safetensors_state_dict,
    _load_weights_best_effort,
    _pick_best_prefix_remap,
    _strip_prefixes,
    enable_xformers,
    find_diffusion_pipe_root,
    forward_with_optional_checkpoint,
    resolve_path_best_effort,
)
from training.families.anima.text_encoding import (  # noqa: E402
    _build_qwen_text_from_prompt,
    _parse_weighted_tag,
    encode_qwen,
    tokenize_t5_weighted,
)
from training.state import load_training_state, save_training_state  # noqa: E402
from training.model_loading import ensure_models_namespace  # noqa: E402
from training.vae import load_vae  # noqa: E402
from training.families import get_family, resolve_family  # noqa: E402  # 派发咽喉（D8'）
from training.families.anima.loader import (  # noqa: E402
    load_anima_model,
    load_text_encoders,
)
from training.families.anima.sampling import sample_image  # noqa: E402
from training.dataset import (  # noqa: E402
    BucketBatchSampler,
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    MergedDataset,
    RepeatDataset,
    collate_fn,
    collate_fn_cached,
)
from training.cli import (  # noqa: E402
    parse_args,
    prompt_for_args,
)
from training.timestep_sampling import sample_t  # noqa: E402
from training.noise import make_noise  # noqa: E402
from training.loss_weighting import compute_loss_weight  # noqa: E402


# ============================================================================
# 主函数
# ============================================================================


def _log_allocator_state() -> None:
    """把 CUDA 进程显存上限提到 95% 并打印实际 allocator 配置。

    定位「显存充足却分不到小块」的 OOM 时发现：torch 2.11 的 ``cudaMallocAsync``
    backend 默认把进程 pool 锁在 ~16GB（24GB 卡的 2/3，实测 ``PyTorch limit:
    17179869184``），即使物理显存还有 8GB+。用 ``set_per_process_memory_fraction``
    显式提到 0.95 反制（对两种 backend 都生效：cudaMallocAsync 通过 pool 大小、
    expandable_segments 通过上限）。必须在 CUDA 初始化后、任何大分配前调用。
    打印设置后的 fraction 与 mem_get_info 供日志核对。
    """
    import logging
    import os

    import torch

    log = logging.getLogger(__name__)
    if not torch.cuda.is_available():
        log.info("[allocator] CUDA 不可用，跳过")
        return
    try:
        # 显式提高进程显存上限到 95%：解除 cudaMallocAsync 默认 ~16GB pool 限制
        torch.cuda.set_per_process_memory_fraction(0.95)
    except Exception as exc:  # pragma: no cover - 仅诊断/兜底
        log.warning("[allocator] set_per_process_memory_fraction 失败: %s", exc)
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<未设置>")
    try:
        frac = torch.cuda.memory.get_per_process_memory_fraction()
    except Exception as exc:  # pragma: no cover - 仅诊断
        frac = f"<err {exc}>"
    free, total = torch.cuda.mem_get_info()
    log.info(
        "[allocator] PYTORCH_CUDA_ALLOC_CONF=%s | memory_fraction=%s | mem_free=%sGiB total=%sGiB",
        conf,
        frac,
        round(free / (1024**3), 2),
        round(total / (1024**3), 2),
    )


def main():
    """ADR 0003 PR-B：main() 现在只编排 phase。

    每个 phase 是个 `run(ctx)` 函数，按顺序 in-place mutate TrainingContext。
    具体实现在 runtime/training/phases/。
    """
    from training import phases
    from training.context import TrainingContext
    from training import loop

    args = parse_args()
    ctx = TrainingContext(args=args)
    phases.bootstrap.run(ctx)
    _log_allocator_state()
    phases.models.run(ctx)
    phases.dataset.run(ctx)
    phases.text_cache.run(ctx)
    phases.models.finish(ctx)
    phases.optimizer.run(ctx)
    phases.resume.run(ctx)
    loop.run(ctx)
    phases.finalize.run(ctx)


if __name__ == "__main__":
    main()
