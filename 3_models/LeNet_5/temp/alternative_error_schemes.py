"""Disposable experiments for physically interpretable error attenuation.

No production module imports this file. It can be deleted with the temp folder.
"""
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

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR))

from lenet5.config import ExperimentConfig
from lenet5.data import create_mnist_loaders
from lenet5.engine import evaluate, load_checkpoint
from lenet5.model import InferenceMode, LeNet5
from lenet5.quantization import ideal_multiplication_lut, load_fitting_lut, load_quantization_parameters

DEVICE_LUTS = {
    "NMOS": PROJECT_ROOT / "1_data_fitting/output/01_NMOS/hardware_multiplication_lut.csv",
    "CFET": PROJECT_ROOT / "1_data_fitting/output/02_CFET/hardware_multiplication_lut.csv",
    "AOS_1": PROJECT_ROOT / "1_data_fitting/output/03_AOS_1/hardware_multiplication_lut.csv",
    "AOS_2": PROJECT_ROOT / "1_data_fitting/output/04_AOS_2/hardware_multiplication_lut.csv",
}
STRENGTHS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def configure(model, params, lut_by_layer) -> None:
    for name, layer in model.fault_layers().items():
        item = params[name]
        layer.configure_hardware_lut(
            lut_by_layer[name], input_scale=item.input_scale,
            input_zero_point=item.input_zero_point,
            weight_scale=torch.tensor(item.weight_scale),
            weight_zero_point=torch.tensor(item.weight_zero_point),
        )


def main() -> int:
    config = ExperimentConfig()
    _, test_loader = create_mnist_loaders(config, download=False)
    model = LeNet5()
    load_checkpoint(model, MODEL_DIR / "chekpoint/lenet5_float.pt")
    params = load_quantization_parameters(MODEL_DIR / "chekpoint/quantization.json")
    ideal = ideal_multiplication_lut().float()
    layer_names = list(model.fault_layers())
    payload = {
        "definition": "L_eff = round(L_ideal + lambda * (L_device - L_ideal)); lambda is residual hardware-error strength after calibration/redundancy.",
        "strengths": list(STRENGTHS), "devices": {}, "layer_aware": {},
    }
    for device_name, path in DEVICE_LUTS.items():
        device = torch.as_tensor(load_fitting_lut(path)).float()
        sweep = {}
        for strength in STRENGTHS:
            effective = torch.round(ideal + strength * (device - ideal))
            configure(model, params, {name: effective for name in layer_names})
            result = evaluate(model, test_loader, device=config.device, mode=InferenceMode.FAULT)
            sweep[str(strength)] = {"accuracy": result.accuracy, "loss": result.loss}
            print(f"{device_name:5s} lambda={strength:.2f}: {result.accuracy:.4f}")
        payload["devices"][device_name] = sweep

        # Sensitivity-guided hybrid mapping: convolution/FC1 stay digital/ideal;
        # only the empirically robust FC2 and FC3 use the physical device LUT.
        tables = {name: (device if name in {"fc2", "fc3"} else ideal) for name in layer_names}
        configure(model, params, tables)
        hybrid = evaluate(model, test_loader, device=config.device, mode=InferenceMode.FAULT)
        payload["layer_aware"][device_name] = {"device_layers": ["fc2", "fc3"], **asdict(hybrid)}
        print(f"{device_name:5s} layer-aware(fc2,fc3): {hybrid.accuracy:.4f}")

    (HERE / "alternative_error_schemes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    colors = ["#2878B5", "#C82423", "#2F9E44", "#F2A104"]
    for (device_name, sweep), color in zip(payload["devices"].items(), colors):
        ax.plot(STRENGTHS, [100 * sweep[str(s)]["accuracy"] for s in STRENGTHS], "o-", lw=1.3, ms=3.5, color=color, label=device_name)
    ax.set(xlabel="Residual hardware-error strength, $\\lambda$", ylabel="MNIST accuracy (%)", title="Accuracy degradation under attenuated device error")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(HERE / "alternative_error_schemes.png", bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
