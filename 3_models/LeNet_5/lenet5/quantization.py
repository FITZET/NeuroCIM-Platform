from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor, nn

from .model import LeNet5


@dataclass(frozen=True)
class LayerQuantizationParameters:
    input_scale: float
    input_zero_point: int
    weight_scale: list[float]
    weight_zero_point: list[int]


def ideal_multiplication_lut(device: torch.device | str = "cpu") -> Tensor:
    codes = torch.arange(256, dtype=torch.int64, device=device)
    return codes[:, None] * codes[None, :]  # LUT convention: [W, X]


def _affine_parameters(minimum: Tensor, maximum: Tensor) -> tuple[Tensor, Tensor]:
    minimum = torch.minimum(minimum, torch.zeros_like(minimum))
    maximum = torch.maximum(maximum, torch.zeros_like(maximum))
    scale = torch.clamp((maximum - minimum) / 255.0, min=torch.finfo(torch.float32).eps)
    zero_point = torch.clamp(torch.round(-minimum / scale), 0, 255).to(torch.int64)
    return scale, zero_point


class MinMaxInputObserver:
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def __call__(self, _module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        value = inputs[0].detach()
        self.minimum = min(self.minimum, float(value.min()))
        self.maximum = max(self.maximum, float(value.max()))


@torch.inference_mode()
def calibrate(
    model: LeNet5,
    data_loader,
    *,
    device: torch.device | str,
    max_batches: int = 20,
) -> dict[str, LayerQuantizationParameters]:
    """Collect per-layer input ranges and per-output-channel weight ranges."""

    model.set_inference_mode("float").eval().to(device)
    observers = {name: MinMaxInputObserver() for name in model.fault_layers()}
    handles = [model.fault_layers()[name].register_forward_pre_hook(observer) for name, observer in observers.items()]
    try:
        for batch_index, (inputs, _targets) in enumerate(data_loader):
            if batch_index >= max_batches:
                break
            model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()

    parameters: dict[str, LayerQuantizationParameters] = {}
    for name, layer in model.fault_layers().items():
        observer = observers[name]
        if not np.isfinite(observer.minimum) or not np.isfinite(observer.maximum):
            raise RuntimeError(f"No calibration samples reached layer {name}")
        input_scale, input_zp = _affine_parameters(
            torch.tensor(observer.minimum), torch.tensor(observer.maximum)
        )
        weight = layer.weight.detach().reshape(layer.weight.shape[0], -1)
        weight_scale, weight_zp = _affine_parameters(weight.min(dim=1).values, weight.max(dim=1).values)
        parameters[name] = LayerQuantizationParameters(
            input_scale=float(input_scale),
            input_zero_point=int(input_zp),
            weight_scale=weight_scale.cpu().tolist(),
            weight_zero_point=weight_zp.cpu().tolist(),
        )
    return parameters


def configure_lut_inference(
    model: LeNet5,
    parameters: dict[str, LayerQuantizationParameters],
    *,
    lut: Tensor | np.ndarray | None = None,
    error_strength: float = 1.0,
) -> LeNet5:
    """Configure every MAC layer with an ideal or attenuated device LUT.

    ``error_strength`` is lambda in
    ``L_eff = L_ideal + lambda * (L_device - L_ideal)``. It is ignored when
    ``lut`` is omitted because that path is the ideal INT8 baseline.
    """

    if not 0.0 <= error_strength <= 1.0:
        raise ValueError("error_strength (lambda) must be in [0, 1]")
    ideal = ideal_multiplication_lut()
    if lut is None:
        table = ideal
    else:
        device = torch.as_tensor(lut, dtype=torch.float64)
        if device.shape != (256, 256):
            raise ValueError(f"Expected LUT shape (256, 256), got {tuple(device.shape)}")
        table = torch.round(
            ideal.to(torch.float64) + error_strength * (device - ideal.to(torch.float64))
        ).to(torch.int64)
    layers = model.fault_layers()
    missing = set(layers).difference(parameters)
    if missing:
        raise KeyError(f"Missing quantization parameters for layers: {sorted(missing)}")
    for name, layer in layers.items():
        item = parameters[name]
        layer.configure_hardware_lut(
            table,
            input_scale=item.input_scale,
            input_zero_point=item.input_zero_point,
            weight_scale=torch.tensor(item.weight_scale),
            weight_zero_point=torch.tensor(item.weight_zero_point),
        )
    return model


def load_fitting_lut(path: str | Path) -> np.ndarray:
    """Read ``hardware_multiplication_lut.csv`` from the fitting backend."""

    table = np.loadtxt(Path(path), delimiter=",", skiprows=1, usecols=range(1, 257))
    if table.shape != (256, 256):
        raise ValueError(f"Expected LUT shape (256, 256), got {table.shape}")
    return table


def save_quantization_parameters(
    parameters: dict[str, LayerQuantizationParameters], path: str | Path
) -> None:
    Path(path).write_text(
        json.dumps({name: asdict(value) for name, value in parameters.items()}, indent=2),
        encoding="utf-8",
    )


def load_quantization_parameters(path: str | Path) -> dict[str, LayerQuantizationParameters]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: LayerQuantizationParameters(**value) for name, value in raw.items()}
