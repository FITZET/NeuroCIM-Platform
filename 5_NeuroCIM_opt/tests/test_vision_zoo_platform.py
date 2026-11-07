from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT), str(PROJECT / "3_models/VisionZoo")]

from neurocim_opt.application import PlatformService
from neurocim_opt.model_registry import builtin_registry
from neurocim_opt.training_service import (
    ErrorAwareTrainingService, TrainingRequest, VISION_ZOO_DIFFERENTIAL_BATCH_LIMIT,
)


def _resources() -> tuple[Path, Path, Path]:
    checkpoint = PROJECT / "3_models/VisionZoo/checkpoint/resnet8_mnist.pt"
    auxiliary = PROJECT / "3_models/VisionZoo/checkpoint/resnet8-mnist_input_scales.json"
    error_model = ROOT / "outputs/error_models/03_UOS_1/error_model.json"
    required = (
        checkpoint, auxiliary, error_model,
        PROJECT / "data/mnist/MNIST/raw/train-images-idx3-ubyte",
        PROJECT / "data/mnist/MNIST/raw/t10k-images-idx3-ubyte",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("formal ResNet-8/MNIST integration resources are unavailable")
    return checkpoint, auxiliary, error_model


def test_vision_zoo_registry_tracks_real_resource_state_and_preprocessing() -> None:
    registry = builtin_registry(PROJECT)
    expected = {
        "resnet8-mnist", "mobilenetv2-cifar10", "vgg11-bn-cifar10",
        "squeezenet1_1-cifar10",
    }
    assert expected <= {item.model_id for item in registry.list()}
    assert registry.get("resnet8-mnist").checkpoint_available
    assert registry.get("mobilenetv2-cifar10").checkpoint_available
    assert registry.get("vgg11-bn-cifar10").checkpoint_available
    assert registry.get("squeezenet1_1-cifar10").checkpoint_available
    for model_id in expected - {"resnet8-mnist"}:
        assert registry.get(model_id).normalization_std == (0.2023, 0.1994, 0.2010)


def test_resnet8_single_inference_runs_all_three_modes() -> None:
    checkpoint, auxiliary, error_model = _resources()
    outcome = PlatformService.infer_registered_single(
        PROJECT, "resnet8-mnist", 0,
        dataset_root=PROJECT / "data/mnist", checkpoint_path=checkpoint,
        error_model_path=error_model, auxiliary_path=auxiliary, device="cpu",
    )
    assert outcome.true_label == 7
    assert outcome.input_shape == (1, 28, 28)
    assert all(0.0 <= value <= 1.0 for value in outcome.input_image)
    assert [item.mode for item in outcome.predictions[:2]] == [
        "float", "ideal_differential",
    ]
    assert len(outcome.predictions) == 3
    assert all(len(item.top5) == 5 for item in outcome.predictions)


def test_resnet8_training_prepare_uses_shared_factory_and_adaptive_batches(tmp_path: Path) -> None:
    checkpoint, auxiliary, error_model = _resources()
    request = TrainingRequest(
        PROJECT, error_model, tmp_path / "training", model_id="resnet8-mnist",
        dataset_root=PROJECT / "data/mnist", checkpoint_path=checkpoint,
        auxiliary_path=auxiliary,
        batch_size=128, device="cpu", max_train_batches=1, max_eval_batches=1,
    )
    service = ErrorAwareTrainingService()
    resources = service.validate(request)
    loaders, base, training_converter, evaluation_converter, state = service._prepare(
        resources, request,
    )
    assert loaders[0].batch_size == request.batch_size
    assert loaders[2].batch_size == request.batch_size
    assert base().__class__.__name__ == "ResNet8MNIST"
    assert training_converter is evaluation_converter
    assert state
