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
    cifar10_required_files,
    clear_lenet_inference_cache,
    get_resnet20_inference_resources,
)


def _real_resources() -> tuple[Path, Path, Path]:
    checkpoint = PROJECT_ROOT / "3_models/Resnet20/checkpoint/resnet20_cifar10.pt"
    scales = PROJECT_ROOT / "3_models/Resnet20/output/formal/input_scales.json"
    error_model = ROOT / "outputs/error_models/01_NMOS/error_model.json"
    if not all(path.is_file() for path in (checkpoint, scales, error_model)):
        pytest.skip("real ResNet-20 inference resources are not installed")
    return checkpoint, scales, error_model


def _synthetic_cifar_root(tmp_path: Path) -> Path:
    for path in cifar10_required_files(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-test-marker")
    return tmp_path


def test_real_resnet_models_single_sample_top5_and_image(tmp_path: Path, monkeypatch) -> None:
    import torch
    from torchvision import datasets

    checkpoint, scales, error_model = _real_resources()
    dataset_root = _synthetic_cifar_root(tmp_path)

    class SyntheticCifar10:
        def __init__(self, root, train, transform, download):
            self.transform = transform

        def __len__(self) -> int:
            return 10_000

        def __getitem__(self, index: int):
            image = torch.linspace(-1.0, 1.0, 3 * 32 * 32).reshape(3, 32, 32)
            return image, index % 10

    monkeypatch.setattr(datasets, "CIFAR10", SyntheticCifar10)
    clear_lenet_inference_cache()
    outcome = PlatformService.infer_registered_single(
        PROJECT_ROOT, "resnet20-cifar10", 3,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        auxiliary_path=scales,
        error_model_path=error_model,
        device="cpu",
    )
    assert outcome.true_label == 3
    assert outcome.input_shape == (3, 32, 32)
    assert len(outcome.input_image) == 3 * 32 * 32
    assert [prediction.mode for prediction in outcome.predictions] == [
        "float", "ideal_differential", "device_01_NMOS",
    ]
    assert all(len(prediction.top5) == 5 for prediction in outcome.predictions)
    assert all(
        0.0 <= confidence <= 1.0
        for prediction in outcome.predictions
        for _label, confidence in prediction.top5
    )
    grouped = run_grouped_batch_inference(BatchInferenceRequest(
        project_root=PROJECT_ROOT,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint,
        quantization_path=scales,
        error_model_path=error_model,
        model_id="resnet20-cifar10",
        device="cpu",
        sample_count=5,
        group_size=2,
        inference_batch_size=2,
        output_path=tmp_path / "resnet_batch.json",
    ))
    assert grouped.cache_hit is True
    assert grouped.dataset == "CIFAR-10/test"
    assert [group.sample_count for group in grouped.groups] == [2, 2, 1]
    assert set(grouped.modes) == {"float", "ideal_differential", "device_differential"}
    assert (tmp_path / "resnet_batch.json").is_file()
    clear_lenet_inference_cache()


def test_resnet_diagnostic_lists_incomplete_cifar_files(tmp_path: Path) -> None:
    checkpoint, scales, error_model = _real_resources()
    with pytest.raises(FileNotFoundError, match="dataset is incomplete") as captured:
        get_resnet20_inference_resources(
            model_root=PROJECT_ROOT / "3_models/Resnet20",
            dataset_root=tmp_path,
            checkpoint_path=checkpoint,
            input_scales_path=scales,
            error_model_path=error_model,
            device_name="cpu",
        )
    message = str(captured.value)
    assert "batches.meta" in message and "test_batch" in message and "data_batch_5" in message
