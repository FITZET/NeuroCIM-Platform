from pathlib import Path
import hashlib
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "3_models/LeNet_5"))
sys.path.insert(0, str(PROJECT / "3_models/Resnet20"))
sys.path.insert(0, str(PROJECT / "3_models/TinyTransformer"))
sys.path.insert(0, str(PROJECT / "3_models/VisionZoo"))

from neurocim_opt import ModelRegistry, ModelSpec, builtin_registry


def test_builtin_registry_filters_and_builds_models() -> None:
    registry = builtin_registry(PROJECT)
    assert [item.model_id for item in registry.list(dataset="cifar10")] == [
        "mobilenetv2-cifar10", "resnet20-cifar10", "squeezenet1_1-cifar10",
        "tinytransformer-cifar10", "vgg11-bn-cifar10",
    ]
    assert registry.get("lenet5-mnist").checkpoint_available
    assert registry.get("resnet20-cifar10").create_model().__class__.__name__ == "ResNetCifar"
    assert registry.get("tinytransformer-cifar10").create_model().__class__.__name__ == "CCT"
    assert registry.get("resnet8-mnist").create_model().__class__.__name__ == "ResNet8MNIST"
    assert [item.model_id for item in registry.list(task="text_generation")] == [
        "gpt2-large", "gpt2-small",
    ]
    assert registry.get("gpt2-small").checkpoint_available


def test_registry_rejects_duplicate_ids() -> None:
    registry = ModelRegistry()
    spec = ModelSpec("x", "X", "task", ("data",), "resnet20.model:resnet20", None,
                     (3, 32, 32), ("accuracy",), "test")
    registry.register(spec)
    with pytest.raises(KeyError, match="Duplicate"):
        registry.register(spec)


def test_builtin_specs_expose_resources_capabilities_and_provenance() -> None:
    registry = builtin_registry(PROJECT)
    image_specs = registry.list(capability="error_aware_training")
    assert [item.model_id for item in image_specs] == [
        "bnn-mnist", "gpt2-small", "lenet5-mnist", "mobilenetv2-cifar10",
        "resnet20-cifar10", "resnet8-mnist", "squeezenet1_1-cifar10",
        "tinytransformer-cifar10", "vgg11-bn-cifar10",
    ]
    for spec in registry.list():
        assert spec.model_root is not None and spec.model_root.is_dir()
        if spec.model_id == "gpt2-large":
            assert spec.datasets == ("wikitext2",)
            assert spec.supports("language_model_evaluation")
            assert spec.weight_sha256 == "9008e91f3c248005e27aa0a258f334029cccdd32f700a9c459b143b3eb1eccc3"
            # A .partial download must never make the registry claim readiness.
            if not spec.checkpoint_available:
                assert not spec.resources_available
            continue
        assert spec.auxiliary_resources_available
        assert spec.source_url.startswith("https://github.com/")
        assert spec.source_version
        assert spec.checkpoint_available
        assert spec.weight_sha256 is not None and len(spec.weight_sha256) == 64
        with spec.checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == spec.weight_sha256

    lenet = registry.get("lenet5-mnist")
    assert lenet.dataset_root == PROJECT / "data/mnist"
    assert lenet.class_names == tuple(str(index) for index in range(10))
    assert lenet.normalization_mean == (0.1307,)
    assert lenet.normalization_std == (0.3081,)

    tiny = registry.get("tinytransformer-cifar10")
    assert tiny.dataset_root == PROJECT / "data/cifar10"
    assert tiny.class_names[0] == "airplane" and tiny.class_names[-1] == "truck"
    assert tiny.supports("single_inference")

    mobile = registry.get("mobilenetv2-cifar10")
    assert mobile.normalization_std == (0.2023, 0.1994, 0.2010)
    assert mobile.checkpoint_available
    squeeze = registry.get("squeezenet1_1-cifar10")
    assert squeeze.checkpoint_available
    assert squeeze.weight_sha256 == "5fe061ba144cc2b9890b47b3a395d36c7195051aea894bbaed3a245694fde196"

    gpt2 = registry.get("gpt2-small")
    assert gpt2.dataset_root is None
    assert gpt2.class_names == ()
    assert gpt2.supports("text_generation")
    assert gpt2.supports("error_aware_training")
    large = registry.get("gpt2-large")
    assert large.dataset_root == PROJECT / "3_models/GPT2/checkpoint/gpt2-large/wikitext-2-raw-v1"
    assert large.checkpoint.name == "model.fp16.safetensors"


def test_original_model_spec_positional_constructor_remains_compatible() -> None:
    spec = ModelSpec(
        "legacy", "Legacy", "task", ("data",), "resnet20.model:resnet20",
        None, (3, 32, 32), ("accuracy",), "test",
    )
    assert spec.model_root is None
    assert spec.auxiliary_resources == ()
    assert spec.capabilities == ()


def test_frontend_report_metadata_matches_delivery_table() -> None:
    registry = builtin_registry(PROJECT)
    expected = {
        "resnet8-mnist": ("手写数字识别", "MNIST", "95.00%", "残差 CNN"),
        "mobilenetv2-cifar10": ("图像分类", "CIFAR-10", "84.00%", "轻量 CNN"),
        "vgg11-bn-cifar10": ("图像分类", "CIFAR-10", "85.00%", "高算力 CNN"),
        "squeezenet1_1-cifar10": ("图像分类", "CIFAR-10", "81.00%", "超轻量 CNN"),
        "bnn-mnist": ("手写数字识别", "MNIST", "99.00%", "二值网络"),
        "tinytransformer-cifar10": ("情感分类", "合成数据", "99.60%", "轻量语言模型"),
        "gpt2-large": ("语言建模", "WikiText-2", "PPL=30.69", "大语言模型"),
    }
    for model_id, row in expected.items():
        spec = registry.get(model_id)
        assert (spec.report_task, spec.report_dataset, spec.report_baseline, spec.model_family) == row


def test_resource_manifest_matches_registry() -> None:
    registry = builtin_registry(PROJECT)
    manifest = json.loads((ROOT / "model_resource_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    rows = {row["model_id"]: row for row in manifest["models"]}
    assert set(rows) == {item.model_id for item in registry.list()}
    for spec in registry.list():
        row = rows[spec.model_id]
        assert row["sha256"] == spec.weight_sha256
        assert row["source"] == spec.source_url
        assert row["version"] == spec.source_version
        assert (PROJECT / row["checkpoint"]).resolve() == spec.checkpoint
        assert tuple((PROJECT / path).resolve() for path in row["auxiliary"]) == (
            spec.auxiliary_resources
        )
        expected_dataset = (
            str(spec.dataset_root.relative_to(PROJECT)).replace("\\", "/")
            if spec.dataset_root is not None else None
        )
        assert row["dataset"] == expected_dataset
