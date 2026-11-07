from __future__ import annotations

import gc
import os
import sys

import torch


GIB = 1024 ** 3
MIB = 1024 ** 2
DEFAULT_GPU_TARGET_FRACTION = 0.90
DEFAULT_RAM_TARGET_FRACTION = 0.82


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def system_memory_snapshot() -> dict[str, int]:
    """Return live physical-memory totals without requiring psutil."""

    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "ram_total_bytes": int(memory.total),
            "ram_available_bytes": int(memory.available),
            "ram_used_bytes": int(memory.total - memory.available),
        }
    except (ImportError, AttributeError):
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]
        status = MemoryStatus(); status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "ram_total_bytes": int(status.total_physical),
                "ram_available_bytes": int(status.available_physical),
                "ram_used_bytes": int(status.total_physical - status.available_physical),
            }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES")
        available = page_size * os.sysconf("SC_AVPHYS_PAGES")
        return {
            "ram_total_bytes": int(total), "ram_available_bytes": int(available),
            "ram_used_bytes": int(total - available),
        }
    except (AttributeError, OSError, ValueError):
        return {"ram_total_bytes": 0, "ram_available_bytes": 0, "ram_used_bytes": 0}


def runtime_memory_snapshot(device: torch.device | str) -> dict[str, int]:
    snapshot = system_memory_snapshot()
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        snapshot.update({
            "free_bytes": int(free), "total_bytes": int(total),
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        })
    else:
        snapshot.update({
            "free_bytes": 0, "total_bytes": 0,
            "allocated_bytes": 0, "reserved_bytes": 0,
        })
    return snapshot


def memory_aware_batch_size(
    requested: int,
    device: torch.device | str,
    *,
    reserve_bytes: int | None = None,
    estimated_peak_bytes_per_sample: int = 3 * GIB,
    gpu_target_fraction: float = DEFAULT_GPU_TARGET_FRACTION,
    estimated_host_bytes_per_sample: int = 0,
    ram_target_fraction: float = DEFAULT_RAM_TARGET_FRACTION,
) -> tuple[int, dict[str, int]]:
    """Choose an aggressive-but-bounded batch from live RAM and CUDA memory.

    Differential LUT convolutions allocate large temporary integer tensors and
    need substantially more transient memory than ordinary float convolutions.
    The returned value is only the initial size; callers still back off on an
    actual allocator OOM because fragmentation and layer shapes vary by model.
    """
    requested = max(1, int(requested))
    resolved = torch.device(device)
    snapshot = runtime_memory_snapshot(resolved)
    host_limit = requested
    if estimated_host_bytes_per_sample > 0 and snapshot["ram_total_bytes"]:
        host_budget = min(
            snapshot["ram_available_bytes"],
            max(0, int(snapshot["ram_total_bytes"] * ram_target_fraction)
                - snapshot["ram_used_bytes"]),
        )
        host_limit = max(1, host_budget // estimated_host_bytes_per_sample)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return min(requested, host_limit), snapshot
    free_bytes, total_bytes = snapshot["free_bytes"], snapshot["total_bytes"]
    reserve = (
        max(384 * MIB, int(total_bytes * (1.0 - gpu_target_fraction)))
        if reserve_bytes is None else int(reserve_bytes)
    )
    current_used = total_bytes - free_bytes
    target_headroom = max(0, int(total_bytes * gpu_target_fraction) - current_used)
    usable = max(0, min(int(free_bytes) - reserve, target_headroom))
    safe = max(1, usable // max(1, int(estimated_peak_bytes_per_sample)))
    snapshot.update({
        "gpu_target_bytes": int(total_bytes * gpu_target_fraction),
        "gpu_usable_bytes": int(usable),
        "ram_target_bytes": int(snapshot["ram_total_bytes"] * ram_target_fraction),
    })
    return min(requested, host_limit, safe), snapshot


def empirical_cuda_batch_size(
    requested: int,
    current: int,
    device: torch.device | str,
    *,
    baseline_allocated_bytes: int,
    peak_allocated_bytes: int,
    processed_samples: int,
    target_fraction: float = DEFAULT_GPU_TARGET_FRACTION,
) -> int:
    """Grow a successful batch using its measured CUDA peak as evidence.

    Growth is capped at 2x per observation. Any overestimate is still guarded
    by the caller's OOM backoff, but normal jobs converge without repeatedly
    emptying the CUDA allocator cache.
    """

    requested, current = max(1, int(requested)), max(1, int(current))
    resolved = torch.device(device)
    if (resolved.type != "cuda" or not torch.cuda.is_available()
            or current >= requested or processed_samples < 1):
        return min(requested, current)
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    _free, total = torch.cuda.mem_get_info(index)
    transient = max(1, int(peak_allocated_bytes) - int(baseline_allocated_bytes))
    per_sample = transient / processed_samples
    budget = max(0, int(total * target_fraction) - int(baseline_allocated_bytes))
    measured_limit = max(1, int((budget * 0.94) / max(per_sample, 1.0)))
    return min(requested, max(current, min(current * 2, measured_limit)))
