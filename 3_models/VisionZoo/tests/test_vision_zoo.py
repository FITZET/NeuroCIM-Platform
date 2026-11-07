from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT / "5_NeuroCIM_opt")]

from neurocim_opt.layers import DifferentialConv2d, DifferentialLinear
from vision_zoo import FACTORIES, create_model
from vision_zoo.checkpoint import load_checkpoint, load_state_dict_safe, sha256_file
from vision_zoo.data import _split_indices
from vision_zoo.engine import calibrate_input_scales, differential_copy


EXPECTED_PARAMETERS = {
    "resnet8-mnist": 75_002,
    "mobilenetv2-cifar10": 2_236_682,
    "vgg11-bn-cifar10": 9_756_426,
    "squeezenet1_1-cifar10": 732_010,
}


@pytest.mark.parametrize("model_id", tuple(FACTORIES))
def test_factory_shapes_and_parameter_counts(model_id: str) -> None:
    torch.manual_seed(1)
    model = create_model(model_id).eval()
    channels, size = (1, 28) if model_id == "resnet8-mnist" else (3, 32)
    with torch.inference_mode():
        output = model(torch.randn(2, channels, size, size))
    assert output.shape == (2, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS[model_id]


def test_resnet8_is_explicit_6n_plus_2_n1() -> None:
    model = create_model("resnet8-mnist")
    assert model.depth == 8 and model.blocks_per_stage == 1
    assert sum(isinstance(module, nn.Conv2d) for module in model.modules()) == 7
    assert sum(isinstance(module, nn.Linear) for module in model.modules()) == 1
    assert model.conv1.in_channels == 1


def test_official_weights_have_expected_hash_and_load_strictly() -> None:
    metadata = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    required = [
        ROOT / "checkpoint" / metadata["weights"][model_id]["filename"]
        for model_id in ("mobilenetv2-cifar10", "vgg11-bn-cifar10")
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("official weights are external release resources; run download_pretrained.py")
    for model_id in ("mobilenetv2-cifar10", "vgg11-bn-cifar10"):
        record = metadata["weights"][model_id]
        checkpoint = ROOT / "checkpoint" / record["filename"]
        assert sha256_file(checkpoint) == record["sha256"]
        model = load_checkpoint(
            create_model(model_id), checkpoint, expected_sha256=record["sha256"]
        )
        assert isinstance(model, nn.Module)
    assert metadata["weights"]["resnet8-mnist"]["test_accuracy"] == pytest.approx(0.9936)
    squeeze = metadata["weights"]["squeezenet1_1-cifar10"]
    assert squeeze["test_accuracy"] == pytest.approx(0.9168)
    assert squeeze["test_samples"] == 10_000
    assert sha256_file(ROOT / "checkpoint" / squeeze["filename"]) == squeeze["sha256"]


def test_safe_checkpoint_loader_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_state_dict_safe(path, expected_sha256="0" * 64)


def test_calibration_and_differential_copy_cover_every_mac_layer() -> None:
    model = create_model("resnet8-mnist")
    loader = DataLoader(
        TensorDataset(torch.randn(2, 1, 28, 28), torch.zeros(2, dtype=torch.long)),
        batch_size=2,
    )
    scales = calibrate_input_scales(model, loader, device="cpu")
    expected = {
        name for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    assert set(scales) == expected
    codes = np.arange(256, dtype=np.int64)
    converted = differential_copy(model, codes[:, None] * codes[None, :], scales)
    assert sum(isinstance(module, DifferentialConv2d) for module in converted.modules()) == 7
    assert sum(isinstance(module, DifferentialLinear) for module in converted.modules()) == 1


def test_dataset_split_is_disjoint_and_reproducible() -> None:
    first_train, first_validation = _split_indices(100, 20, 42)
    second_train, second_validation = _split_indices(100, 20, 42)
    assert (first_train, first_validation) == (second_train, second_validation)
    assert len(first_train) == 80 and len(first_validation) == 20
    assert set(first_train).isdisjoint(first_validation)
    assert set(first_train) | set(first_validation) == set(range(100))
