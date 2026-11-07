from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt.training_service import (
    ErrorAwareTrainingService, TrainingCancelled, TrainingRequest, TrainingResources,
    TINYTRANSFORMER_EVAL_BATCH_LIMIT,
    _configure_tinytransformer_head_finetuning, canonical_model_id,
)


@pytest.mark.parametrize("alias, expected", [
    ("lenet5_mnist", "lenet5-mnist"),
    ("BNN", "bnn-mnist"),
    ("resnet-20", "resnet20-cifar10"),
    ("CCT-7", "tinytransformer-cifar10"),
    ("ResNet-8", "resnet8-mnist"),
    ("MobileNet-V2", "mobilenetv2-cifar10"),
    ("VGG11-BN", "vgg11-bn-cifar10"),
    ("SqueezeNet1.1", "squeezenet1_1-cifar10"),
    ("GPT-2-Small", "gpt2-small"),
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
    ("resnet8-mnist", "mnist", "3_models/VisionZoo/checkpoint/resnet8.pt",
     "3_models/VisionZoo/checkpoint/resnet8-mnist_input_scales.json"),
    ("mobilenetv2-cifar10", "cifar10", "3_models/VisionZoo/checkpoint/mobile.pt",
     "3_models/VisionZoo/checkpoint/mobilenetv2-cifar10_input_scales.json"),
    ("vgg11-bn-cifar10", "cifar10", "3_models/VisionZoo/checkpoint/vgg.pt",
     "3_models/VisionZoo/checkpoint/vgg11-bn-cifar10_input_scales.json"),
    ("squeezenet1_1-cifar10", "cifar10", "3_models/VisionZoo/checkpoint/squeeze.pt",
     "3_models/VisionZoo/checkpoint/squeezenet1_1-cifar10_input_scales.json"),
])
def test_validation_resolves_each_model_and_dataset(
    tmp_path: Path, model_id: str, dataset_name: str,
    checkpoint: str, auxiliary: str | None,
) -> None:
    model_folder = {
        "lenet5-mnist": "LeNet_5", "bnn-mnist": "BNN",
        "resnet20-cifar10": "Resnet20",
        "tinytransformer-cifar10": "TinyTransformer",
        "resnet8-mnist": "VisionZoo", "mobilenetv2-cifar10": "VisionZoo",
        "vgg11-bn-cifar10": "VisionZoo", "squeezenet1_1-cifar10": "VisionZoo",
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


def test_gpt2_small_validation_uses_wikitext_and_rejects_large(tmp_path: Path) -> None:
    model_root = tmp_path / "3_models/GPT2"
    checkpoint = _file(model_root / "checkpoint/gpt2_nanogpt.pt")
    scales = _file(model_root / "checkpoint/input_scales.json", "{}")
    (model_root / "checkpoint/tokenizer").mkdir(parents=True)
    corpus = model_root / "checkpoint/gpt2-large/wikitext-2-raw-v1"
    for name in ("wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"):
        _file(corpus / name, "enough text")
    error = _file(tmp_path / "fit/error_model.json", "{}")
    resources = ErrorAwareTrainingService().validate(TrainingRequest(
        tmp_path, error, tmp_path / "out", model_id="gpt2-small",
        dataset_root=model_root / "checkpoint/tokenizer",
        checkpoint_path=checkpoint, auxiliary_path=scales,
        batch_size=1, sequence_length=8,
    ))
    assert resources.dataset_name == "wikitext2"
    assert resources.dataset_root == corpus.resolve()
    with pytest.raises(ValueError, match="Unsupported training model"):
        ErrorAwareTrainingService().validate(TrainingRequest(
            tmp_path, error, tmp_path / "out", model_id="gpt2-large",
        ))


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


def test_tinytransformer_training_uses_two_heads_but_evaluation_uses_full_graph(
    tmp_path: Path,
) -> None:
    import torch
    from neurocim_opt.layers import DifferentialConv2d, DifferentialLinear

    project = ROOT.parent
    model_root = project / "3_models/TinyTransformer"
    checkpoint = model_root / "checkpoint/cct_7_3x1_32_cifar10.pth"
    scales = model_root / "checkpoint/input_scales.json"
    dataset = project / "data/cifar10"
    if not all(path.exists() for path in (checkpoint, scales, dataset / "cifar-10-batches-py/test_batch")):
        pytest.skip("TinyTransformer resources are not installed")
    resources = TrainingResources(
        "tinytransformer-cifar10", "cifar10", model_root, checkpoint, scales,
        tmp_path / "unused-error-model.json", dataset, torch.device("cpu"),
    )
    request = TrainingRequest(
        project, resources.error_model, tmp_path / "out", model_id=resources.model_id,
        dataset_root=dataset, checkpoint_path=checkpoint, auxiliary_path=scales,
    )
    loaders, base, training_converter, evaluation_converter, _state = (
        ErrorAwareTrainingService()._prepare(resources, request)
    )
    codes = torch.arange(256, dtype=torch.int64)
    lut = codes[:, None] * codes[None, :]
    training_model = training_converter(base(), lut, True)
    evaluation_model = evaluation_converter(base(), lut, False)
    differential_types = (DifferentialConv2d, DifferentialLinear)
    training_names = {
        name for name, module in training_model.named_modules()
        if isinstance(module, differential_types)
    }
    evaluation_names = {
        name for name, module in evaluation_model.named_modules()
        if isinstance(module, differential_types)
    }
    all_mac_names = {
        name for name, module in base().named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    }
    assert training_names == {"classifier.attention_pool", "classifier.fc"}
    assert evaluation_names == all_mac_names
    assert len(evaluation_names) == 31
    # Keep a large host loader batch; CUDA execution is independently split
    # and adapted from measured peaks inside the training service.
    assert loaders[2].batch_size == request.batch_size
