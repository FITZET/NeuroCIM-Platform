from __future__ import annotations

import gc

import torch


GIB = 1024 ** 3


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_aware_batch_size(
    requested: int,
    device: torch.device | str,
    *,
    reserve_bytes: int = GIB,
    estimated_peak_bytes_per_sample: int = 3 * GIB,
) -> tuple[int, dict[str, int]]:
    """Choose a conservative differential-MAC batch from live CUDA memory.

    Differential LUT convolutions allocate large temporary integer tensors and
    need substantially more transient memory than ordinary float convolutions.
    The returned value is only the initial size; callers still back off on an
    actual allocator OOM because fragmentation and layer shapes vary by model.
    """
    requested = max(1, int(requested))
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return requested, {"free_bytes": 0, "total_bytes": 0}
    device_index = resolved.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    usable = max(0, int(free_bytes) - int(reserve_bytes))
    safe = max(1, usable // max(1, int(estimated_peak_bytes_per_sample)))
    return min(requested, safe), {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
