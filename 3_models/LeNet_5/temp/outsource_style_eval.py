"""Reproduce the contractor report's positive/negative differential LUT path.

Assumptions made explicit because the report does not ship LeNet calibration
tables or an ADC configuration: sigma=0, rho=0, ADC disabled. Both raw and
per-output-channel statistically calibrated results are exported.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR))

from lenet5.config import ExperimentConfig
from lenet5.data import create_mnist_loaders
from lenet5.engine import load_checkpoint
from lenet5.model import LeNet5
from lenet5.quantization import load_fitting_lut

DEVICE_LUTS = {
    "NMOS": PROJECT_ROOT / "1_data_fitting/output/01_NMOS/hardware_multiplication_lut.csv",
    "CFET": PROJECT_ROOT / "1_data_fitting/output/02_CFET/hardware_multiplication_lut.csv",
    "AOS_1": PROJECT_ROOT / "1_data_fitting/output/03_AOS_1/hardware_multiplication_lut.csv",
    "AOS_2": PROJECT_ROOT / "1_data_fitting/output/04_AOS_2/hardware_multiplication_lut.csv",
}


def lut_effective_gain(lut: Tensor) -> Tensor:
    codes = torch.arange(256, dtype=torch.float64, device=lut.device)
    product = codes[:, None] * codes[None, :]  # [X, W]
    # Input CSV is [W, X]; transpose to the contractor's lut[X, W].
    table_xw = lut.T.to(torch.float64)
    return (table_xw[1:, 1:] * product[1:, 1:]).sum() / product[1:, 1:].square().sum()


def quantize_input_per_sample(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flat = x.reshape(x.shape[0], -1)
    xmin = flat.min(dim=1).values
    xmax = flat.max(dim=1).values
    scale = ((xmax - xmin) / 255.0).clamp_min(torch.finfo(x.dtype).eps)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    codes = torch.clamp(torch.round((x - xmin.view(shape)) / scale.view(shape)), 0, 255).long()
    return codes, scale, xmin


def quantize_weight_pos_neg(weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flat = weight.reshape(weight.shape[0], -1)
    scale = (flat.abs().max(dim=1).values / 255.0).clamp_min(torch.finfo(weight.dtype).eps)
    shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    wp = torch.clamp(torch.round(weight.clamp_min(0) / scale.view(shape)), 0, 255).long()
    wn = torch.clamp(torch.round((-weight).clamp_min(0) / scale.view(shape)), 0, 255).long()
    return wp, wn, scale


def differential_products(lut_xw: Tensor, xcodes: Tensor, wp: Tensor, wn: Tensor) -> Tensor:
    return lut_xw[xcodes, wp] - lut_xw[xcodes, wn]


def cim_linear_raw(x: Tensor, layer, lut_wx: Tensor, chunk: int = 32) -> Tensor:
    original_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    xc, sx, xmin = quantize_input_per_sample(x)
    wp, wn, sw = quantize_weight_pos_neg(layer.weight)
    lut_xw = lut_wx.T
    gain = lut_effective_gain(lut_wx)
    acc = torch.zeros((x.shape[0], layer.out_features), dtype=torch.float64, device=x.device)
    for start in range(0, layer.in_features, chunk):
        stop = min(start + chunk, layer.in_features)
        acc += differential_products(
            lut_xw, xc[:, None, start:stop], wp[None, :, start:stop], wn[None, :, start:stop]
        ).to(torch.float64).sum(dim=-1)
    sum_w = (wp - wn).sum(dim=1).to(x.dtype)
    y = (sx[:, None] * sw[None, :] / gain.to(x.dtype)) * acc.to(x.dtype)
    y += xmin[:, None] * sw[None, :] * sum_w[None, :]
    if layer.bias is not None:
        y += layer.bias
    return y.reshape(*original_shape, layer.out_features)


def cim_conv_raw(x: Tensor, layer, lut_wx: Tensor, chunk: int = 32) -> Tensor:
    if layer.groups != 1 or layer.padding != (0, 0):
        raise NotImplementedError("Diagnostic currently targets LeNet conv layers (groups=1, padding=0)")
    xc, sx, xmin = quantize_input_per_sample(x)
    patches = F.unfold(xc.to(x.dtype), layer.kernel_size, dilation=layer.dilation, stride=layer.stride).long()
    wp, wn, sw = quantize_weight_pos_neg(layer.weight)
    wp, wn = wp.reshape(layer.out_channels, -1), wn.reshape(layer.out_channels, -1)
    lut_xw = lut_wx.T
    gain = lut_effective_gain(lut_wx)
    n, k, locations = patches.shape
    acc = torch.zeros((n, layer.out_channels, locations), dtype=torch.float64, device=x.device)
    for start in range(0, k, chunk):
        stop = min(start + chunk, k)
        xpart = patches[:, start:stop, :].permute(0, 2, 1).unsqueeze(1)
        acc += differential_products(
            lut_xw, xpart, wp[None, :, None, start:stop], wn[None, :, None, start:stop]
        ).to(torch.float64).sum(dim=-1)
    sum_w = (wp - wn).sum(dim=1).to(x.dtype)
    y = (sx[:, None, None] * sw[None, :, None] / gain.to(x.dtype)) * acc.to(x.dtype)
    y += xmin[:, None, None] * sw[None, :, None] * sum_w[None, :, None]
    if layer.bias is not None:
        y += layer.bias[None, :, None]
    ho = (x.shape[-2] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    wo = (x.shape[-1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
    return y.reshape(n, layer.out_channels, ho, wo)


def channel_stats(values: list[Tensor]) -> tuple[Tensor, Tensor]:
    # Conv: channel dim 1; Linear: last dim. Flatten all other dimensions.
    merged = []
    for value in values:
        if value.ndim == 2:
            merged.append(value)
        else:
            merged.append(value.movedim(1, -1).reshape(-1, value.shape[1]))
    data = torch.cat(merged, dim=0).float()
    return data.mean(dim=0), data.std(dim=0, unbiased=False).clamp_min(1e-8)


def apply_calibration(y: Tensor, calibration: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    cim_mean, cim_std, ideal_mean, ideal_std = calibration
    shape = (1, -1) + ((1,) * (y.ndim - 2) if y.ndim > 2 else ())
    return (y - cim_mean.view(shape)) / cim_std.view(shape) * ideal_std.view(shape) + ideal_mean.view(shape)


def layer_forward(name: str, x: Tensor, model: LeNet5, lut: Tensor) -> tuple[Tensor, Tensor]:
    layer = model.fault_layers()[name]
    if name.startswith("conv"):
        return F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding), cim_conv_raw(x, layer, lut)
    return F.linear(x, layer.weight, layer.bias), cim_linear_raw(x, layer, lut)


@torch.inference_mode()
def build_calibration(model: LeNet5, loader, lut: Tensor, device: str, batches: int = 20):
    collected = {name: {"ideal": [], "cim": []} for name in model.fault_layers()}
    for batch_index, (x, _) in enumerate(loader):
        if batch_index >= batches:
            break
        x = x.to(device)
        for name in model.fault_layers():
            if name == "fc1":
                x = torch.flatten(x, 1)
            ideal, raw = layer_forward(name, x, model, lut)
            collected[name]["ideal"].append(ideal.cpu())
            collected[name]["cim"].append(raw.cpu())
            x = ideal
            if name in ("conv1", "conv2"):
                x = model.pool(model.activation(x))
            elif name in ("fc1", "fc2"):
                x = model.activation(x)
    calibration = {}
    for name, item in collected.items():
        im, ist = channel_stats(item["ideal"])
        cm, cst = channel_stats(item["cim"])
        calibration[name] = (cm.to(device), cst.to(device), im.to(device), ist.to(device))
    return calibration


@torch.inference_mode()
def evaluate_style(model, loader, lut, device, calibration=None):
    correct = total = 0
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        for name in model.fault_layers():
            if name == "fc1":
                x = torch.flatten(x, 1)
            _, x = layer_forward(name, x, model, lut)
            if calibration is not None:
                x = apply_calibration(x, calibration[name])
            if name in ("conv1", "conv2"):
                x = model.pool(model.activation(x))
            elif name in ("fc1", "fc2"):
                x = model.activation(x)
        correct += int((x.argmax(1) == target).sum())
        total += target.numel()
    return correct / total


def main() -> int:
    config = ExperimentConfig()
    torch.manual_seed(config.seed)
    train_loader, test_loader = create_mnist_loaders(config, download=False)
    model = LeNet5().to(config.device).eval()
    load_checkpoint(model, MODEL_DIR / "chekpoint/lenet5_float.pt", map_location=config.device)
    results = {
        "assumptions": {"input_quantization": "dynamic uint8 per sample", "weight_quantization": "Wp/Wn shared uint8 scale per output channel", "calibration": "per-output-channel mean/std, 20 train batches", "sigma": 0.0, "rho": 0.0, "adc": "disabled because report does not state LeNet adc_bits"},
        "devices": {},
    }
    codes = torch.arange(256, dtype=torch.int64, device=config.device)
    ideal_lut = codes[:, None] * codes[None, :]  # [W, X]
    ideal_accuracy = evaluate_style(model, test_loader, ideal_lut, config.device)
    results["ideal_lut_accuracy"] = ideal_accuracy
    print(f"IDEAL: {ideal_accuracy:.4f}")
    for name, path in DEVICE_LUTS.items():
        lut = torch.as_tensor(load_fitting_lut(path), dtype=torch.int64, device=config.device)
        calibration = build_calibration(model, train_loader, lut, config.device)
        raw = evaluate_style(model, test_loader, lut, config.device)
        calibrated = evaluate_style(model, test_loader, lut, config.device, calibration)
        results["devices"][name] = {"raw_accuracy": raw, "calibrated_accuracy": calibrated}
        print(f"{name}: raw={raw:.4f}, calibrated={calibrated:.4f}")
    path = HERE / "outsource_style_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
