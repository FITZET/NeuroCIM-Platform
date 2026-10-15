from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


UPSTREAM_URL = "https://github.com/neurosim/NeuroSim"
UPSTREAM_BRANCH = "2DInferenceV1.4"
UPSTREAM_COMMIT = "8a88abf85844c0e1ba17cc771ea535fff6040456"
UPSTREAM_ARCHIVE_SHA256 = "102cac5f02c34592087f1a602bfe28a5545f5f0ee76d3173f946b1be7a0433e2"


@dataclass(frozen=True)
class NeuroSimConfig:
    weight_bits: int = 8
    activation_bits: int = 8
    subarray_rows: int = 128
    parallel_rows: int = 128

    def validate(self) -> None:
        if self.weight_bits <= 0 or self.activation_bits <= 0:
            raise ValueError("weight_bits and activation_bits must be positive")
        if self.subarray_rows <= 0:
            raise ValueError("subarray_rows must be positive")
        if not 0 < self.parallel_rows <= self.subarray_rows:
            raise ValueError("parallel_rows must be in [1, subarray_rows]")


@dataclass(frozen=True)
class NeuroSimRunResult:
    command: list[str]
    returncode: int
    metrics: dict[str, float]
    layer_metrics: dict[str, list[float]]
    stdout_path: str
    network_path: str
    trace_directory: str
    config: dict[str, Any]
    upstream: dict[str, str]


def _upstream_hook_module(vendor_root: Path):
    wrapper = vendor_root / "Inference_pytorch"
    if not (wrapper / "utee" / "hook.py").is_file():
        raise FileNotFoundError(f"NeuroSim hook module not found under {wrapper}")
    if str(wrapper) not in sys.path:
        sys.path.insert(0, str(wrapper))
    from utee import hook

    return hook


def _unsigned_codes(value: Tensor, scale: Tensor) -> np.ndarray:
    quantized = torch.clamp(torch.round(value.detach().abs() / scale), 0, 255)
    return quantized.cpu().numpy().astype(np.uint16)


def _write_unsigned_bitplanes(matrix: np.ndarray, bits: int, path: Path) -> None:
    if bits != 8:
        raise NotImplementedError("The measured multiplication model currently requires 8-bit codes")
    values = np.asarray(matrix, dtype=np.uint16)
    if values.ndim != 2:
        raise ValueError(f"bit-plane matrix must be two-dimensional, got {values.shape}")
    if values.size and values.max() > 255:
        raise ValueError("activation code exceeds uint8")
    planes = ((values[:, :, None] >> np.arange(bits - 1, -1, -1)) & 1).reshape(values.shape[0], -1)
    np.savetxt(path, planes, delimiter=",", fmt="%d")


def _expanded_differential_weight(module: nn.Conv2d | nn.Linear) -> np.ndarray:
    weight = module.weight.detach()
    flattened = weight.reshape(weight.shape[0], -1)
    scale = flattened.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps) / 255.0
    shape = [-1] + [1] * (weight.ndim - 1)
    codes = _unsigned_codes(weight, scale.view(*shape)) / 255.0
    positive = np.where(weight.cpu().numpy() > 0, codes, 0.0)
    negative = np.where(weight.cpu().numpy() < 0, codes, 0.0)
    # Two physical input branches and two physical weight branches. NeuroSim sees
    # all four non-negative arrays; signed subtraction is outside its analog core.
    positive_inputs = np.concatenate([positive, positive], axis=1)
    negative_inputs = np.concatenate([negative, negative], axis=1)
    return np.concatenate([positive_inputs, negative_inputs], axis=0)


def export_differential_traces(
    model: nn.Module,
    sample: Tensor,
    output_dir: str | Path,
    *,
    input_scales: dict[str, float],
    vendor_root: str | Path,
    pooling_after: set[str] | None = None,
    config: NeuroSimConfig | None = None,
    network_name: str = "network_differential",
    excluded_non_mac_ops: list[str] | None = None,
) -> tuple[Path, list[tuple[Path, Path]]]:
    """Export exact model traces using NeuroSim's own layout helper."""

    config = config or NeuroSimConfig()
    config.validate()
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    hook = _upstream_hook_module(Path(vendor_root).expanduser().resolve())
    pooling_after = pooling_after or set()
    records: list[tuple[str, nn.Module, Tensor]] = []
    handles = []

    def capture(name: str):
        def callback(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            records.append((name, module, inputs[0].detach().cpu()))
        return callback

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name not in input_scales:
                raise KeyError(f"Missing input scale for layer {name}")
            handles.append(module.register_forward_pre_hook(capture(name)))
    try:
        model.eval()
        with torch.inference_mode():
            model(sample.to(next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    if not records:
        raise RuntimeError("No Conv2d or Linear layers were captured")

    network_rows: list[list[int]] = []
    trace_pairs: list[tuple[Path, Path]] = []
    for index, (name, module, layer_input) in enumerate(records, start=1):
        weight_path = target / f"layer_{index:02d}_{name}_weight.csv"
        input_path = target / f"layer_{index:02d}_{name}_input.csv"
        expanded_weight = _expanded_differential_weight(module)
        hook.write_matrix_weight(expanded_weight, str(weight_path))

        scale = float(input_scales[name])
        positive_codes = _unsigned_codes(torch.clamp(layer_input, min=0), torch.tensor(scale))
        negative_codes = _unsigned_codes(torch.clamp(-layer_input, min=0), torch.tensor(scale))
        if isinstance(module, nn.Conv2d):
            if module.groups != 1 or module.dilation != (1, 1):
                raise NotImplementedError("NeuroSim adapter supports groups=1 and dilation=1")
            signed_codes = np.concatenate([positive_codes, negative_codes], axis=1)
            stretched = hook.stretch_input(
                signed_codes,
                window_size=module.kernel_size[0],
                padding=module.padding,
                stride=module.stride,
            )
            # NeuroSim's CopyInput indexes [weight-row][spatial-position * bit].
            # Its upstream hook therefore writes im2col as [K*K*Cin, N*bits].
            _write_unsigned_bitplanes(stretched[0].T, config.activation_bits, input_path)
            network_rows.append(
                [
                    int(layer_input.shape[-2]), int(layer_input.shape[-1]),
                    int(module.in_channels * 2), int(module.kernel_size[0]),
                    int(module.kernel_size[1]), int(module.out_channels * 2),
                    int(name in pooling_after), int(module.stride[0]),
                ]
            )
        else:
            signed_codes = np.concatenate([positive_codes, negative_codes], axis=1)
            # Fully connected input uses one column group per input vector.
            _write_unsigned_bitplanes(signed_codes[:1].T, config.activation_bits, input_path)
            network_rows.append(
                [1, 1, int(module.in_features * 2), 1, 1, int(module.out_features * 2), 0, 1]
            )
        trace_pairs.append((weight_path, input_path))

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", network_name).strip("._")
    if not safe_name:
        raise ValueError("network_name must contain at least one safe character")
    network_path = target / f"{safe_name}.csv"
    np.savetxt(network_path, np.asarray(network_rows, dtype=np.int64), delimiter=",", fmt="%d")
    manifest = {
        "mapping": "four non-negative paths: X+/X- rows by W+/W- columns",
        "digital_combination": "Y = X+W+ - X+W- - X-W+ + X-W-; not modeled by NeuroSim core",
        "excluded_non_mac_ops": excluded_non_mac_ops or [],
        "network": str(network_path),
        "traces": [[str(weight), str(inputs)] for weight, inputs in trace_pairs],
        "config": asdict(config),
        "upstream": {
            "url": UPSTREAM_URL,
            "branch": UPSTREAM_BRANCH,
            "commit": UPSTREAM_COMMIT,
            "archive_sha256": UPSTREAM_ARCHIVE_SHA256,
        },
    }
    (target / "trace_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return network_path, trace_pairs


_METRICS = {
    "chip_area_um2": r"ChipArea\s*:\s*([0-9.eE+-]+)um\^2",
    "read_latency_ns": r"Chip (?:layer-by-layer readLatency|pipeline-system-clock-cycle) \(per image\) is:\s*([0-9.eE+-]+)ns",
    "read_dynamic_energy_pj": r"Chip (?:total|pipeline-system) readDynamicEnergy(?: \(per image\))? is:\s*([0-9.eE+-]+)pJ",
    "leakage_energy_pj": r"Chip (?:total|pipeline-system) leakage Energy(?: \(per image\))? is:\s*([0-9.eE+-]+)pJ",
    "leakage_power_uw": r"Chip (?:total|pipeline-system) leakage Power(?: \(per image\))? is:\s*([0-9.eE+-]+)uW",
    "energy_efficiency_tops_per_w": r"Energy Efficiency TOPS/W \((?:Layer-by-Layer|Pipelined) Process\):\s*([0-9.eE+-]+)",
    "throughput_tops": r"Throughput TOPS \((?:Layer-by-Layer|Pipelined) Process\):\s*([0-9.eE+-]+)",
    "throughput_fps": r"Throughput FPS \((?:Layer-by-Layer|Pipelined) Process\):\s*([0-9.eE+-]+)",
    "compute_efficiency_tops_per_mm2": r"Compute efficiency TOPS/mm\^2 \((?:Layer-by-Layer|Pipelined) Process\):\s*([0-9.eE+-]+)",
    "runtime_seconds": r"Total Run-time of NeuroSim:\s*([0-9.eE+-]+) seconds",
}


def parse_neurosim_output(output: str) -> tuple[dict[str, float], dict[str, list[float]]]:
    metrics: dict[str, float] = {}
    for key, pattern in _METRICS.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    layer_metrics: dict[str, list[float]] = {}
    for label, key in (
        ("readLatency", "read_latency_ns"),
        ("readDynamicEnergy", "read_dynamic_energy_pj"),
        ("leakagePower", "leakage_power_uw"),
    ):
        matches = re.findall(rf"layer\d+'s {label} is:\s*([0-9.eE+-]+)", output)
        if matches:
            layer_metrics[key] = [float(value) for value in matches]
    return metrics, layer_metrics


def run_neurosim(
    binary: str | Path,
    network_path: str | Path,
    trace_pairs: list[tuple[Path, Path]],
    output_dir: str | Path,
    *,
    config: NeuroSimConfig | None = None,
    conda_prefix: str | Path | None = None,
    timeout_seconds: int = 300,
) -> NeuroSimRunResult:
    config = config or NeuroSimConfig()
    config.validate()
    executable = Path(binary).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    command = [
        str(executable), str(Path(network_path).resolve()),
        str(config.weight_bits), str(config.activation_bits),
        str(config.subarray_rows), str(config.parallel_rows),
    ]
    for weight, inputs in trace_pairs:
        command.extend([str(weight.resolve()), str(inputs.resolve())])
    if conda_prefix is not None:
        command = ["conda", "run", "-p", str(Path(conda_prefix).resolve()), *command]
    completed = subprocess.run(
        command,
        cwd=executable.parent,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    stdout_path = target / "neurosim_stdout.txt"
    combined = completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else "")
    stdout_path.write_text(combined, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"NeuroSim exited with {completed.returncode}; inspect {stdout_path}"
        )
    metrics, layer_metrics = parse_neurosim_output(combined)
    required = {"chip_area_um2", "read_latency_ns", "read_dynamic_energy_pj"}
    if missing := required.difference(metrics):
        raise RuntimeError(f"NeuroSim output is missing metrics {sorted(missing)}; inspect {stdout_path}")
    result = NeuroSimRunResult(
        command=command,
        returncode=completed.returncode,
        metrics=metrics,
        layer_metrics=layer_metrics,
        stdout_path=str(stdout_path),
        network_path=str(Path(network_path).resolve()),
        trace_directory=str(Path(network_path).resolve().parent),
        config=asdict(config),
        upstream={
            "url": UPSTREAM_URL,
            "branch": UPSTREAM_BRANCH,
            "commit": UPSTREAM_COMMIT,
            "license": "CC BY-NC 4.0",
        },
    )
    (target / "neurosim_result.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
