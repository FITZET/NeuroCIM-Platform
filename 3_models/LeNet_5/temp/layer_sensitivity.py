from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR))

from lenet5.config import ExperimentConfig
from lenet5.data import create_mnist_loaders
from lenet5.engine import evaluate, load_checkpoint
from lenet5.model import InferenceMode, LeNet5
from lenet5.quantization import (
    configure_lut_inference,
    ideal_multiplication_lut,
    load_fitting_lut,
    load_quantization_parameters,
)

CHECKPOINT = MODEL_DIR / "chekpoint" / "lenet5_float.pt"
QUANT_PARAMS = MODEL_DIR / "chekpoint" / "quantization.json"
RESULT_PATH = SCRIPT_DIR / "layer_sensitivity.json"
FIGURE_PATH = SCRIPT_DIR / "layer_sensitivity.png"
DEVICE_LUTS = {
    "NMOS": PROJECT_ROOT / "1_data_fitting" / "output" / "01_NMOS" / "hardware_multiplication_lut.csv",
    "CFET": PROJECT_ROOT / "1_data_fitting" / "output" / "02_CFET" / "hardware_multiplication_lut.csv",
    "UOS_1": PROJECT_ROOT / "1_data_fitting" / "output" / "03_UOS_1" / "hardware_multiplication_lut.csv",
    "UOS_2": PROJECT_ROOT / "1_data_fitting" / "output" / "04_UOS_2" / "hardware_multiplication_lut.csv",
}


def set_layer_luts(model, parameters, device_lut, fault_layers: set[str]) -> None:
    ideal = ideal_multiplication_lut()
    for name, layer in model.fault_layers().items():
        item = parameters[name]
        table = device_lut if name in fault_layers else ideal
        layer.configure_hardware_lut(
            table,
            input_scale=item.input_scale,
            input_zero_point=item.input_zero_point,
            weight_scale=torch.tensor(item.weight_scale),
            weight_zero_point=torch.tensor(item.weight_zero_point),
        )


def create_plot(payload: dict) -> None:
    layer_names = payload["layer_order"]
    devices = list(payload["devices"])
    colors = ["#2878B5", "#C82423", "#2F9E44", "#F2A104"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for device, color in zip(devices, colors):
        isolated = payload["devices"][device]["isolated"]
        cumulative = payload["devices"][device]["cumulative"]
        axes[0].plot(layer_names, [100 * isolated[name]["accuracy_drop"] for name in layer_names], "o-", color=color, lw=1.2, ms=3, label=device)
        axes[1].plot(layer_names, [100 * cumulative[name]["accuracy"] for name in layer_names], "o-", color=color, lw=1.2, ms=3, label=device)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set(xlabel="Faulty layer only", ylabel="Accuracy drop (percentage points)", title="(a) Isolated layer sensitivity")
    axes[1].axhline(100 * payload["quantized_baseline"]["accuracy"], color="black", lw=0.8, ls="--", label="INT8 baseline")
    axes[1].set(xlabel="Faults accumulated through layer", ylabel="MNIST accuracy (%)", title="(b) Cumulative fault propagation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, ncol=2)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    config = ExperimentConfig()
    _train_loader, test_loader = create_mnist_loaders(config, download=False)
    model = LeNet5()
    load_checkpoint(model, CHECKPOINT)
    parameters = load_quantization_parameters(QUANT_PARAMS)
    configure_lut_inference(model, parameters)
    baseline = evaluate(model, test_loader, device=config.device, mode=InferenceMode.QUANTIZED)
    layer_names = list(model.fault_layers())
    payload = {
        "method": {
            "isolated": "Only the named layer uses the device LUT; all other MAC layers use ideal LUT.",
            "cumulative": "Device LUT is enabled from conv1 through the named layer; later layers use ideal LUT.",
        },
        "layer_order": layer_names,
        "quantized_baseline": asdict(baseline),
        "devices": {},
    }
    for device_name, lut_path in DEVICE_LUTS.items():
        device_lut = torch.as_tensor(load_fitting_lut(lut_path))
        isolated = {}
        cumulative = {}
        for index, layer_name in enumerate(layer_names):
            set_layer_luts(model, parameters, device_lut, {layer_name})
            result = evaluate(model, test_loader, device=config.device, mode=InferenceMode.FAULT)
            isolated[layer_name] = {
                "accuracy": result.accuracy,
                "accuracy_drop": baseline.accuracy - result.accuracy,
                "loss": result.loss,
            }
            set_layer_luts(model, parameters, device_lut, set(layer_names[: index + 1]))
            result = evaluate(model, test_loader, device=config.device, mode=InferenceMode.FAULT)
            cumulative[layer_name] = {
                "accuracy": result.accuracy,
                "accuracy_drop": baseline.accuracy - result.accuracy,
                "loss": result.loss,
            }
            print(f"{device_name:5s} {layer_name:5s}: isolated={isolated[layer_name]['accuracy']:.4f}, cumulative={cumulative[layer_name]['accuracy']:.4f}")
        payload["devices"][device_name] = {"isolated": isolated, "cumulative": cumulative}
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    create_plot(payload)
    print(f"Saved {RESULT_PATH}")
    print(f"Saved {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
