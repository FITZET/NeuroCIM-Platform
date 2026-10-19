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
        "bnn-mnist", "lenet5-mnist", "mobilenetv2-cifar10",
        "resnet20-cifar10", "resnet8-mnist", "squeezenet1_1-cifar10",
        "tinytransformer-cifar10", "vgg11-bn-cifar10",
    ]
    for spec in registry.list():
        assert spec.model_root is not None and spec.model_root.is_dir()
        if spec.model_id == "gpt2-large":
            assert spec.datasets == ("wikitext2",)
            assert spec.supports("language_model_evaluation")
            assert spec.weight_sha256 == "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce"
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
    assert squeeze.weight_sha256 == "df967fd502b9d8fe2e7dea77d8363ad7fe14df04efcc173df1f94fc05f7cfe4c"

    gpt2 = registry.get("gpt2-small")
    assert gpt2.dataset_root is None
    assert gpt2.class_names == ()
    assert gpt2.supports("text_generation")
    assert not gpt2.supports("error_aware_training")
    large = registry.get("gpt2-large")
    assert large.dataset_root == PROJECT / "3_models/GPT2/checkpoint/gpt2-large/wikitext-2-raw-v1"
    assert large.checkpoint.name == "model.safetensors"


def test_original_model_spec_positional_constructor_remains_compatible() -> None:
    spec = ModelSpec(
        "legacy", "Legacy", "task", ("data",), "resnet20.model:resnet20",
        None, (3, 32, 32), ("accuracy",), "test",
    )
    assert spec.model_root is None
    assert spec.auxiliary_resources == ()
    assert spec.capabilities == ()


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
