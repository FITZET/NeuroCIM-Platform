from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from neurocim_opt.application import PlatformService
from neurocim_opt.batch_inference import BatchInferenceRequest, run_grouped_batch_inference
from neurocim_opt.inference_runtime import (
    clear_lenet_inference_cache,
    get_bnn_inference_resources,
    mnist_required_files,
)


def _resources() -> tuple[Path, Path]:
    checkpoint = PROJECT_ROOT / "3_models/BNN/checkpoint/bnn_mnist.pt"
    error_model = ROOT / "outputs/error_models/01_NMOS/error_model.json"
    if not checkpoint.is_file() or not error_model.is_file():
        pytest.skip("real BNN inference resources are not installed")
    return checkpoint, error_model


def _synthetic_mnist_root(tmp_path: Path) -> Path:
    for path in mnist_required_files(tmp_path, include_training=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-test-marker")
    return tmp_path


def test_real_bnn_single_and_grouped_differential_inference(tmp_path: Path, monkeypatch) -> None:
    import torch
    from torchvision import datasets

    checkpoint, error_model = _resources()
    dataset_root = _synthetic_mnist_root(tmp_path)

    class SyntheticMnist:
        def __init__(self, root, train, transform, download):
            self.train = train

        def __len__(self) -> int:
            return 5_100 if self.train else 10_000

        def __getitem__(self, index: int):
            image = torch.linspace(-1.0, 1.0, 28 * 28).reshape(1, 28, 28)
            return image, index % 10

    monkeypatch.setattr(datasets, "MNIST", SyntheticMnist)
    clear_lenet_inference_cache()
    outcome = PlatformService.infer_registered_single(
        PROJECT_ROOT, "bnn-mnist", 4,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        error_model_path=error_model,
        device="cpu",
    )
    assert outcome.true_label == 4
    assert outcome.input_shape == (1, 28, 28)
    assert [prediction.mode for prediction in outcome.predictions] == [
        "native_bnn", "ideal_differential", "device_01_NMOS",
    ]
    assert all(len(prediction.top5) == 5 for prediction in outcome.predictions)

    grouped = run_grouped_batch_inference(BatchInferenceRequest(
        project_root=PROJECT_ROOT,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        quantization_path=tmp_path / "bnn-does-not-require-auxiliary.json",
        error_model_path=error_model,
        model_id="bnn-mnist",
        device="cpu",
        sample_count=5,
        group_size=2,
        inference_batch_size=2,
    ))
    assert grouped.cache_hit is True
    assert grouped.model_id == "bnn-mnist"
    assert [group.sample_count for group in grouped.groups] == [2, 2, 1]
    assert set(grouped.modes) == {"float", "ideal_differential", "device_differential"}
    clear_lenet_inference_cache()


def test_bnn_diagnostic_requires_train_and_test_mnist(tmp_path: Path) -> None:
    checkpoint, error_model = _resources()
    with pytest.raises(FileNotFoundError, match="MNIST dataset is incomplete") as captured:
        get_bnn_inference_resources(
            model_root=PROJECT_ROOT / "3_models/BNN",
            dataset_root=tmp_path,
            checkpoint_path=checkpoint,
            error_model_path=error_model,
            device_name="cpu",
        )
    message = str(captured.value)
    assert "train-images-idx3-ubyte" in message and "t10k-images-idx3-ubyte" in message


def test_bnn_inference_prefers_best_error_aware_state(tmp_path: Path, monkeypatch) -> None:
    import torch
    import bnn.data
    from bnn.model import BinaryMNIST

    _checkpoint, error_model = _resources()
    dataset_root = _synthetic_mnist_root(tmp_path / "dataset")
    checkpoint = tmp_path / "eat_checkpoint.pt"
    final = {"marker": torch.tensor(1)}
    best = {"marker": torch.tensor(2)}
    torch.save({"model_state_dict": final, "best_model_state_dict": best}, checkpoint)
    captured = {}

    class LoadedExpectedState(RuntimeError):
        pass

    def capture_state(_self, state, *args, **kwargs):
        captured.update(state)
        raise LoadedExpectedState

    monkeypatch.setattr(BinaryMNIST, "load_state_dict", capture_state)
    # This test stops at checkpoint loading, so no real IDX payload is needed.
    monkeypatch.setattr(bnn.data, "create_loaders", lambda *args, **kwargs: (None, None, None))
    clear_lenet_inference_cache()
    with pytest.raises(LoadedExpectedState):
        get_bnn_inference_resources(
            model_root=PROJECT_ROOT / "3_models/BNN",
            dataset_root=dataset_root,
            checkpoint_path=checkpoint,
            error_model_path=error_model,
            device_name="cpu",
        )
    assert int(captured["marker"]) == 2
