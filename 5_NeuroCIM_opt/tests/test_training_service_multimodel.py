from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt.training_service import (
    ErrorAwareTrainingService, TrainingCancelled, TrainingRequest,
    _configure_tinytransformer_head_finetuning, canonical_model_id,
)


@pytest.mark.parametrize("alias, expected", [
    ("lenet5_mnist", "lenet5-mnist"),
    ("BNN", "bnn-mnist"),
    ("resnet-20", "resnet20-cifar10"),
    ("CCT-7", "tinytransformer-cifar10"),
])
def test_canonical_model_aliases(alias: str, expected: str) -> None:
    assert canonical_model_id(alias) == expected


def _file(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("model_id,dataset_name,checkpoint,auxiliary", [
    ("lenet5-mnist", "mnist", "3_models/LeNet_5/chekpoint/model.pt",
     "3_models/LeNet_5/chekpoint/quantization.json"),
    ("bnn-mnist", "mnist", "3_models/BNN/checkpoint/model.pt", None),
    ("resnet20-cifar10", "cifar10", "3_models/Resnet20/checkpoint/model.pt",
     "3_models/Resnet20/output/input_scales.json"),
    ("tinytransformer-cifar10", "cifar10",
     "3_models/TinyTransformer/checkpoint/model.pth",
     "3_models/TinyTransformer/checkpoint/input_scales.json"),
])
def test_validation_resolves_each_model_and_dataset(
    tmp_path: Path, model_id: str, dataset_name: str,
    checkpoint: str, auxiliary: str | None,
) -> None:
    model_folder = {
        "lenet5-mnist": "LeNet_5", "bnn-mnist": "BNN",
        "resnet20-cifar10": "Resnet20",
        "tinytransformer-cifar10": "TinyTransformer",
    }[model_id]
    (tmp_path / "3_models" / model_folder).mkdir(parents=True)
    checkpoint_path = _file(tmp_path / checkpoint)
    auxiliary_path = _file(tmp_path / auxiliary, "{}") if auxiliary else None
    error_path = _file(tmp_path / "fit/error_model.json", "{}")
    dataset = tmp_path / "datasets"
    if dataset_name == "mnist":
        _file(dataset / "MNIST/raw/train-images-idx3-ubyte")
    else:
        _file(dataset / "cifar-10-batches-py/test_batch")
    resources = ErrorAwareTrainingService().validate(TrainingRequest(
        tmp_path, error_path, tmp_path / "out", model_id=model_id,
        dataset_root=dataset, checkpoint_path=checkpoint_path,
        auxiliary_path=auxiliary_path,
    ))
    assert resources.model_id == model_id
    assert resources.dataset_name == dataset_name
    assert resources.checkpoint == checkpoint_path.resolve()
    assert resources.error_model == error_path.resolve()


def test_validation_reports_model_specific_missing_cifar(tmp_path: Path) -> None:
    model_root = tmp_path / "3_models/Resnet20"
    checkpoint = _file(model_root / "checkpoint/model.pt")
    scales = _file(model_root / "output/input_scales.json", "{}")
    error = _file(tmp_path / "fit/error_model.json", "{}")
    with pytest.raises(FileNotFoundError, match=r"CIFAR10.*test_batch"):
        ErrorAwareTrainingService().validate(TrainingRequest(
            tmp_path, error, tmp_path / "out", model_id="resnet20-cifar10",
            dataset_root=tmp_path / "missing-cifar", checkpoint_path=checkpoint,
            auxiliary_path=scales,
        ))


def test_cifar_raw_file_selection_is_normalised_to_torchvision_root(tmp_path: Path) -> None:
    model_root = tmp_path / "3_models/Resnet20"
    checkpoint = _file(model_root / "checkpoint/model.pt")
    scales = _file(model_root / "output/input_scales.json", "{}")
    error = _file(tmp_path / "fit/error_model.json", "{}")
    selected = _file(tmp_path / "datasets/cifar-10-batches-py/test_batch")
    resources = ErrorAwareTrainingService().validate(TrainingRequest(
        tmp_path, error, tmp_path / "out", model_id="resnet20-cifar10",
        dataset_root=selected, checkpoint_path=checkpoint, auxiliary_path=scales,
    ))
    assert resources.dataset_root == selected.parent.parent.resolve()


def test_evaluation_honours_cancel_and_batch_limit() -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.randn(6, 2), torch.zeros(6, dtype=torch.long)), batch_size=2)
    result = ErrorAwareTrainingService()._evaluate(
        torch.nn.Linear(2, 10), loader, torch.device("cpu"), "float", None, 1
    )
    assert result["samples"] == 2
    with pytest.raises(TrainingCancelled):
        ErrorAwareTrainingService()._evaluate(
            torch.nn.Linear(2, 10), loader, torch.device("cpu"), "float", lambda: True
        )


def test_tinytransformer_finetuning_freezes_backbone() -> None:
    project = ROOT.parent
    sys.path.insert(0, str(project / "3_models/TinyTransformer"))
    from tiny_transformer import cct_7_3x1_32

    model = cct_7_3x1_32()
    trainable = _configure_tinytransformer_head_finetuning(model)
    assert set(trainable) == {
        "classifier.attention_pool.weight", "classifier.attention_pool.bias",
        "classifier.fc.weight", "classifier.fc.bias",
    }
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 2_827
