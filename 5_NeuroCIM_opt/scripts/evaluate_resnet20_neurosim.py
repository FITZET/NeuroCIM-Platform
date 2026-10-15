from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLATFORM_ROOT.parent
sys.path[:0] = [str(PLATFORM_ROOT), str(PROJECT_ROOT / "3_models/Resnet20")]

from neurocim_opt import HardwareErrorModel
from neurocim_opt.paths import data_root, neurosim_binary, output_root, require_path, toolchain_root
from neurocim_opt.neurosim_adapter import NeuroSimConfig, export_differential_traces, run_neurosim
from resnet20.data import create_cifar10_loaders
from resnet20.engine import differential_copy
from resnet20.model import resnet20


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ResNet-20 MAC layers through official NeuroSim")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-scales", type=Path, required=True)
    parser.add_argument("--error-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=data_root("cifar10"))
    parser.add_argument("--output-dir", type=Path, default=output_root() / "neurosim/resnet20_01_NMOS")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subarray", type=int, default=128)
    parser.add_argument("--parallel-rows", type=int, default=128)
    args = parser.parse_args()
    require_path(args.dataset_root, "CIFAR-10 dataset root")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = resnet20()
    model.load_state_dict(state["model_state_dict"])
    scales = json.loads(args.input_scales.read_text(encoding="utf-8"))
    error = HardwareErrorModel.load(args.error_model)
    model = differential_copy(model, error.effective_lut(1), scales).to(args.device).eval()
    _train, _validation, test = create_cifar10_loaders(
        args.dataset_root, batch_size=1, eval_batch_size=1
    )
    sample, _target = next(iter(test))
    config = NeuroSimConfig(subarray_rows=args.subarray, parallel_rows=args.parallel_rows)
    network, pairs = export_differential_traces(
        model, sample, args.output_dir / "traces", input_scales=scales,
        vendor_root=PLATFORM_ROOT / "vendor/neurosim", config=config,
        network_name="network_resnet20_differential",
        excluded_non_mac_ops=[
            "BatchNorm2d", "ReLU", "residual element-wise addition", "adaptive average pooling"
        ],
    )
    result = run_neurosim(
        neurosim_binary(), network, pairs, args.output_dir,
        config=config, conda_prefix=toolchain_root(), timeout_seconds=600,
    )
    print(json.dumps({"layers": len(pairs), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
