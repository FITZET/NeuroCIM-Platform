from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt.neurosim_adapter import (
    export_differential_traces,
    parse_neurosim_output,
)


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.fc = nn.Linear(8, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(value)
        return self.fc(value.flatten(1))


def test_export_uses_four_path_differential_dimensions(tmp_path: Path) -> None:
    model = TinyNetwork()
    vendor = ROOT / "vendor/neurosim"
    network, pairs = export_differential_traces(
        model,
        torch.randn(1, 1, 4, 4),
        tmp_path,
        input_scales={"conv": 0.02, "fc": 0.03},
        vendor_root=vendor,
    )
    rows = np.loadtxt(network, delimiter=",", dtype=int)
    assert rows.tolist() == [
        [4, 4, 2, 3, 3, 4, 0, 1],
        [1, 1, 16, 1, 1, 6, 0, 1],
    ]
    conv_weight = np.loadtxt(pairs[0][0], delimiter=",")
    assert conv_weight.shape == (18, 4)
    assert np.all((conv_weight >= 0) & (conv_weight <= 1))
    conv_input = np.loadtxt(pairs[0][1], delimiter=",")
    assert conv_input.shape == (18, 4 * 8)
    fc_input = np.loadtxt(pairs[1][1], delimiter=",")
    assert fc_input.shape == (16, 8)
    manifest = json.loads((tmp_path / "trace_manifest.json").read_text(encoding="utf-8"))
    assert manifest["excluded_non_mac_ops"] == []


def test_parse_neurosim_metrics() -> None:
    output = """
ChipArea : 123.5um^2
Chip layer-by-layer readLatency (per image) is: 45.25ns
Chip total readDynamicEnergy is: 67.75pJ
layer1's readLatency is: 10ns
Energy Efficiency TOPS/W (Layer-by-Layer Process): 8.5
"""
    metrics, layers = parse_neurosim_output(output)
    assert metrics["chip_area_um2"] == 123.5
    assert metrics["read_latency_ns"] == 45.25
    assert metrics["read_dynamic_energy_pj"] == 67.75
    assert metrics["energy_efficiency_tops_per_w"] == 8.5
    assert layers["read_latency_ns"] == [10.0]


def test_parse_neurosim_pipelined_metrics() -> None:
    output = """
Chip pipeline-system-clock-cycle (per image) is: 22134.4ns
Chip pipeline-system readDynamicEnergy (per image) is: 230426pJ
Chip pipeline-system leakage Energy (per image) is: 1774.1pJ
Chip pipeline-system leakage Power (per image) is: 80.1516uW
Energy Efficiency TOPS/W (Pipelined Process): 19.2858
Throughput TOPS (Pipelined Process): 0.246827
Throughput FPS (Pipelined Process): 45178.6
Compute efficiency TOPS/mm^2 (Pipelined Process): 0.052254
"""
    metrics, _ = parse_neurosim_output(output)
    assert metrics["read_latency_ns"] == 22134.4
    assert metrics["read_dynamic_energy_pj"] == 230426
    assert metrics["leakage_energy_pj"] == 1774.1
    assert metrics["leakage_power_uw"] == 80.1516
    assert metrics["energy_efficiency_tops_per_w"] == 19.2858
    assert metrics["throughput_fps"] == 45178.6
