from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lenet5.model import InferenceMode, LeNet5
from lenet5.quantization import LayerQuantizationParameters, configure_lut_inference


def dummy_parameters(model: LeNet5) -> dict[str, LayerQuantizationParameters]:
    return {
        name: LayerQuantizationParameters(0.05, 128, [0.01] * layer.weight.shape[0], [128] * layer.weight.shape[0])
        for name, layer in model.fault_layers().items()
    }


def test_float_shape_and_layer_inventory() -> None:
    model = LeNet5()
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 10)
    assert list(model.fault_layers()) == ["conv1", "conv2", "fc1", "fc2", "fc3"]


def test_quantized_framework_forward() -> None:
    model = LeNet5()
    configure_lut_inference(model, dummy_parameters(model))
    model.set_inference_mode(InferenceMode.QUANTIZED)
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 10)


def test_quantized_mode_requires_configuration() -> None:
    model = LeNet5()
    try:
        model.set_inference_mode(InferenceMode.QUANTIZED)
    except RuntimeError as error:
        assert "not configured" in str(error)
    else:
        raise AssertionError("Expected missing LUT configuration to fail")


def test_error_strength_endpoints_and_interpolation() -> None:
    model = LeNet5()
    params = dummy_parameters(model)
    ideal = torch.arange(256, dtype=torch.int64)[:, None] * torch.arange(256, dtype=torch.int64)[None, :]
    device = ideal + 100
    configure_lut_inference(model, params, lut=device, error_strength=0.0)
    assert torch.equal(model.conv1.hardware_lut.cpu(), ideal)
    configure_lut_inference(model, params, lut=device, error_strength=0.25)
    assert torch.equal(model.conv1.hardware_lut.cpu(), ideal + 25)
    configure_lut_inference(model, params, lut=device, error_strength=1.0)
    assert torch.equal(model.conv1.hardware_lut.cpu(), device)


def test_error_strength_range_is_validated() -> None:
    model = LeNet5()
    try:
        configure_lut_inference(model, dummy_parameters(model), lut=torch.zeros(256, 256), error_strength=1.1)
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("Expected an invalid lambda to fail")
