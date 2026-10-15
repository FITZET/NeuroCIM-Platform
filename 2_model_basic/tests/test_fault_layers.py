from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fault_layers import FaultConv2d, FaultLinear


def ideal_lut() -> torch.Tensor:
    codes = torch.arange(256, dtype=torch.int64)
    return codes[:, None] * codes[None, :]  # [W, X]


class AddConstant(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        return output + self.value


class LearnableScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        return output * self.scale


@pytest.mark.parametrize(
    ("original", "sample", "fault_type"),
    [
        (nn.Conv2d(4, 6, 3, stride=2, padding=1, groups=2), torch.randn(2, 4, 11, 13), FaultConv2d),
        (nn.Linear(9, 5), torch.randn(3, 7, 9), FaultLinear),
    ],
)
def test_conversion_preserves_normal_computation(original, sample, fault_type) -> None:
    original.eval()
    converted = fault_type.from_module(original)
    assert converted.training is False
    assert torch.equal(original(sample), converted(sample))


@pytest.mark.parametrize(
    ("layer", "sample"),
    [
        (FaultConv2d(3, 4, 3, padding=1), torch.randn(2, 3, 8, 8)),
        (FaultLinear(6, 2), torch.randn(4, 6)),
    ],
)
def test_fault_is_applied_only_when_enabled(layer, sample) -> None:
    normal = layer(sample)
    layer.set_error_model(AddConstant(0.25))
    assert torch.equal(layer(sample), normal)
    layer.enable_fault()
    assert torch.allclose(layer(sample), normal + 0.25)
    layer.disable_fault()
    assert torch.equal(layer(sample), normal)


def test_plain_callable_and_gradient_flow() -> None:
    layer = FaultLinear(3, 2, error_model=lambda output: output * 2, fault_enable=True)
    sample = torch.randn(4, 3, requires_grad=True)
    layer(sample).sum().backward()
    assert sample.grad is not None


def test_strict_shape_contract() -> None:
    layer = FaultLinear(3, 2, error_model=lambda output: output[..., :1], fault_enable=True)
    with pytest.raises(ValueError, match="changed output shape"):
        layer(torch.randn(4, 3))


def test_conversion_accepts_stateful_error_module() -> None:
    converted = FaultLinear.from_module(nn.Linear(3, 2), error_model=LearnableScale())
    assert "error_model.scale" in converted.state_dict()


def test_linear_ideal_lut_matches_affine_quantized_reference() -> None:
    torch.manual_seed(4)
    layer = FaultLinear(7, 5, bias=True)
    sample = torch.randn(3, 2, 7)
    x_scale, zx = 0.025, 121
    w_scale = torch.linspace(0.006, 0.012, 5)
    zw = torch.tensor([128, 126, 130, 125, 129])
    layer.configure_hardware_lut(
        ideal_lut(), input_scale=x_scale, input_zero_point=zx,
        weight_scale=w_scale, weight_zero_point=zw,
    ).enable_fault()
    actual = layer(sample)
    qx = torch.clamp(torch.round(sample / x_scale + zx), 0, 255)
    qw = torch.clamp(torch.round(layer.weight / w_scale[:, None] + zw[:, None]), 0, 255)
    reference = F.linear(
        (qx - zx) * x_scale,
        (qw - zw[:, None]) * w_scale[:, None],
        layer.bias,
    )
    assert torch.allclose(actual, reference, atol=2e-6, rtol=1e-6)


def test_grouped_conv_ideal_lut_matches_affine_quantized_reference() -> None:
    torch.manual_seed(5)
    layer = FaultConv2d(4, 6, 3, stride=2, padding=1, groups=2, bias=True)
    sample = torch.randn(2, 4, 9, 11)
    x_scale, zx = 0.03, 119
    w_scale = torch.linspace(0.005, 0.01, 6)
    zw = torch.tensor([127, 128, 126, 129, 125, 130])
    layer.configure_hardware_lut(
        ideal_lut(), input_scale=x_scale, input_zero_point=zx,
        weight_scale=w_scale, weight_zero_point=zw,
    ).enable_fault()
    actual = layer(sample)
    qx = torch.clamp(torch.round(sample / x_scale + zx), 0, 255)
    qw = torch.clamp(
        torch.round(layer.weight / w_scale[:, None, None, None] + zw[:, None, None, None]),
        0,
        255,
    )
    reference = F.conv2d(
        (qx - zx) * x_scale,
        (qw - zw[:, None, None, None]) * w_scale[:, None, None, None],
        layer.bias,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
    )
    assert torch.allclose(actual, reference, atol=3e-6, rtol=1e-6)
