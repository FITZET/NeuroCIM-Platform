from __future__ import annotations

import torch

from neurocim_opt import runtime_memory


def _snapshot(*, ram_total: int, ram_available: int, gpu_total: int = 0,
              gpu_free: int = 0) -> dict[str, int]:
    return {
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_available,
        "ram_used_bytes": ram_total - ram_available,
        "free_bytes": gpu_free,
        "total_bytes": gpu_total,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
    }


def test_cpu_batch_respects_live_ram_target(monkeypatch) -> None:
    gib = 1024 ** 3
    monkeypatch.setattr(
        runtime_memory, "runtime_memory_snapshot",
        lambda _device: _snapshot(ram_total=16 * gib, ram_available=8 * gib),
    )
    batch, snapshot = runtime_memory.memory_aware_batch_size(
        128, "cpu", estimated_host_bytes_per_sample=256 * 1024 ** 2,
        ram_target_fraction=0.90,
    )
    assert batch == 25
    assert snapshot["ram_available_bytes"] == 8 * gib


def test_cuda_batch_uses_target_fraction_and_keeps_reserve(monkeypatch) -> None:
    gib = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runtime_memory, "runtime_memory_snapshot",
        lambda _device: _snapshot(
            ram_total=32 * gib, ram_available=24 * gib,
            gpu_total=12 * gib, gpu_free=10 * gib,
        ),
    )
    batch, snapshot = runtime_memory.memory_aware_batch_size(
        128, "cuda:0", estimated_peak_bytes_per_sample=gib,
        gpu_target_fraction=0.90,
    )
    # 2 GiB is already occupied; the 90% target leaves 8.8 GiB headroom.
    assert batch == 8
    assert snapshot["gpu_target_bytes"] == int(12 * gib * 0.90)


def test_empirical_cuda_batch_grows_at_most_twofold(monkeypatch) -> None:
    gib = 1024 ** 3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _index: (10 * gib, 12 * gib))
    grown = runtime_memory.empirical_cuda_batch_size(
        128, 8, "cuda:0", baseline_allocated_bytes=2 * gib,
        peak_allocated_bytes=3 * gib, processed_samples=8,
    )
    assert grown == 16


def test_system_memory_snapshot_is_well_formed() -> None:
    snapshot = runtime_memory.system_memory_snapshot()
    assert snapshot["ram_total_bytes"] >= snapshot["ram_available_bytes"] >= 0
    assert snapshot["ram_used_bytes"] >= 0
