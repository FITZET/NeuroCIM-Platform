from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resnet20.model import resnet20
from resnet20.engine import calibrate_input_scales, differential_copy, set_deterministic_seed


def test_resnet20_shape_and_parameter_count() -> None:
    model = resnet20()
    output = model(torch.randn(2, 3, 32, 32))
    parameters = sum(item.numel() for item in model.parameters())
    assert output.shape == (2, 10)
    assert 260_000 < parameters < 280_000


def test_calibration_and_differential_conversion() -> None:
    model = resnet20()
    batch = [(torch.randn(1, 3, 32, 32), torch.tensor([0]))]
    scales = calibrate_input_scales(model, batch, device="cpu")
    assert len(scales) == 22
    codes = torch.arange(256, dtype=torch.int64)
    converted = differential_copy(model, codes[:, None] * codes[None, :], scales)
    assert converted.conv1.__class__.__name__ == "DifferentialConv2d"


def test_seed_reproduces_model_initialization() -> None:
    set_deterministic_seed(7)
    first = resnet20().conv1.weight.detach().clone()
    set_deterministic_seed(7)
    second = resnet20().conv1.weight.detach().clone()
    assert torch.equal(first, second)
