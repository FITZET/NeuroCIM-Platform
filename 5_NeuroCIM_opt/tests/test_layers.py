from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt import DifferentialConv2d, DifferentialLinear, adapt_module


def ideal_lut() -> torch.Tensor:
    code = torch.arange(256, dtype=torch.int64)
    return code[:, None] * code[None, :]


def asymmetric_lut() -> torch.Tensor:
    w = torch.arange(256, dtype=torch.int64)[:, None]
    x = torch.arange(256, dtype=torch.int64)[None, :]
    # Deliberately asymmetric while retaining zero interaction axes.
    return w * x + (w * x * x) // 255


def explicit_linear(layer: DifferentialLinear, sample: torch.Tensor) -> torch.Tensor:
    result = torch.empty(sample.shape[0], layer.out_features)
    for batch in range(sample.shape[0]):
        for output in range(layer.out_features):
            total = 0
            for item in range(layer.in_features):
                qx = min(255, round(abs(float(sample[batch, item])) / float(layer.input_scale)))
                qw = min(255, round(abs(float(layer.weight[output, item])) / float(layer.weight_scale[output])))
                sign = (1 if sample[batch, item] > 0 else -1 if sample[batch, item] < 0 else 0)
                sign *= (1 if layer.weight[output, item] > 0 else -1 if layer.weight[output, item] < 0 else 0)
                total += int(layer.hardware_lut[qw, qx]) * sign
            value = total * float(layer.input_scale) * float(layer.weight_scale[output])
            result[batch, output] = value + float(layer.bias[output])
    return result


def test_signed_linear_ideal_lut_matches_symmetric_quantized_reference() -> None:
    torch.manual_seed(2)
    layer = DifferentialLinear(
        7, 4, lut=ideal_lut(), input_scale=0.02,
        weight_scale=torch.linspace(0.003, 0.008, 4), bias=True,
    )
    sample = torch.randn(3, 7)
    qx = torch.clamp(torch.round(sample.abs() / 0.02), 0, 255) * sample.sign() * 0.02
    ws = layer.weight_scale.float().view(-1, 1)
    qw = torch.clamp(torch.round(layer.weight.abs() / ws), 0, 255) * layer.weight.sign() * ws
    assert torch.allclose(layer(sample), F.linear(qx, qw, layer.bias), atol=2e-6, rtol=1e-6)


def test_asymmetric_lut_axis_order_and_explicit_signed_branches() -> None:
    layer = DifferentialLinear(
        3, 2, lut=asymmetric_lut(), input_scale=0.1,
        weight_scale=torch.tensor([0.05, 0.08]), bias=True,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.4, -0.7, 0.0], [-0.2, 0.56, -0.8]]))
        layer.bias.copy_(torch.tensor([0.1, -0.2]))
    sample = torch.tensor([[0.6, -0.3, 0.0], [-0.5, 0.2, -0.9]])
    assert torch.allclose(layer(sample), explicit_linear(layer, sample), atol=1e-6)


def test_half_linear_scales_integer_accumulator_in_float32() -> None:
    """A large integer MAC must not overflow before its small scales apply."""
    layer = DifferentialLinear(
        64, 2, lut=ideal_lut(), input_scale=1.0 / 255.0,
        weight_scale=torch.full((2,), 1.0 / 255.0), bias=False,
        dtype=torch.float16,
    )
    with torch.no_grad():
        layer.weight.fill_(1.0)
    result = layer(torch.ones(1, 64, dtype=torch.float16))
    assert torch.isfinite(result).all()
    assert torch.allclose(result.float(), torch.full((1, 2), 64.0), atol=0.05)


def test_half_chained_linear_does_not_create_invalid_lut_indices() -> None:
    first = DifferentialLinear(
        64, 4, lut=ideal_lut(), input_scale=1.0 / 255.0,
        weight_scale=torch.full((4,), 1.0 / 255.0), bias=False,
        dtype=torch.float16,
    )
    second = DifferentialLinear(
        4, 1, lut=ideal_lut(), input_scale=64.0 / 255.0,
        weight_scale=torch.tensor([1.0 / 255.0]), bias=False,
        dtype=torch.float16,
    )
    with torch.no_grad():
        first.weight.fill_(1.0)
        second.weight.fill_(1.0)
    result = second(first(torch.ones(1, 64, dtype=torch.float16)))
    assert torch.isfinite(result).all()
    assert result.item() == pytest.approx(256.0, abs=0.5)


def test_conv_matches_unfolded_linear_reference_with_padding() -> None:
    torch.manual_seed(3)
    layer = DifferentialConv2d(
        2, 3, 3, lut=asymmetric_lut(), input_scale=0.04,
        weight_scale=torch.tensor([0.01, 0.012, 0.014]), padding=1, bias=True,
    )
    sample = torch.randn(1, 2, 4, 5)
    actual = layer(sample)
    patches = F.unfold(sample, 3, padding=1).transpose(1, 2).reshape(-1, 18)
    linear = DifferentialLinear(
        18, 3, lut=layer.hardware_lut, input_scale=float(layer.input_scale),
        weight_scale=layer.weight_scale, bias=True,
    )
    with torch.no_grad():
        linear.weight.copy_(layer.weight.reshape(3, -1))
        linear.bias.copy_(layer.bias)
    reference = linear(patches).reshape(1, 4 * 5, 3).permute(0, 2, 1).reshape_as(actual)
    assert torch.allclose(actual, reference, atol=1e-6)


@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_ste_has_exact_forward_and_updates_weights(kind: str) -> None:
    torch.manual_seed(7)
    if kind == "linear":
        layer = DifferentialLinear(
            4, 2, lut=asymmetric_lut(), input_scale=0.05,
            weight_scale=torch.tensor([0.01, 0.02]), ste=True,
        )
        sample = torch.randn(3, 4, requires_grad=True)
    else:
        layer = DifferentialConv2d(
            1, 2, 3, lut=asymmetric_lut(), input_scale=0.05,
            weight_scale=torch.tensor([0.01, 0.02]), ste=True,
        )
        sample = torch.randn(2, 1, 4, 4, requires_grad=True)
    exact_before = layer(sample).detach()
    layer.ste = False
    assert torch.equal(exact_before, layer(sample.detach()))
    layer.ste = True
    loss = layer(sample).square().mean()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum() > 0
    old = layer.weight.detach().clone()
    torch.optim.SGD(layer.parameters(), lr=1e-3).step()
    assert not torch.equal(old, layer.weight)


def test_adapt_module_requires_all_scales() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
    with pytest.raises(KeyError, match="2"):
        adapt_module(model, lut=ideal_lut(), input_scales={"0": 0.1})
