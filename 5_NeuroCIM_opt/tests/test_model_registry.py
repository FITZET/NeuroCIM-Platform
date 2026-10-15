from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "3_models/LeNet_5"))
sys.path.insert(0, str(PROJECT / "3_models/Resnet20"))
sys.path.insert(0, str(PROJECT / "3_models/TinyTransformer"))

from neurocim_opt import ModelRegistry, ModelSpec, builtin_registry


def test_builtin_registry_filters_and_builds_models() -> None:
    registry = builtin_registry(PROJECT)
    assert [item.model_id for item in registry.list(dataset="cifar10")] == [
        "resnet20-cifar10", "tinytransformer-cifar10",
    ]
    assert registry.get("lenet5-mnist").checkpoint_available
    assert registry.get("resnet20-cifar10").create_model().__class__.__name__ == "ResNetCifar"
    assert registry.get("tinytransformer-cifar10").create_model().__class__.__name__ == "CCT"
    assert [item.model_id for item in registry.list(task="text_generation")] == ["gpt2-small"]
    assert registry.get("gpt2-small").checkpoint_available


def test_registry_rejects_duplicate_ids() -> None:
    registry = ModelRegistry()
    spec = ModelSpec("x", "X", "task", ("data",), "resnet20.model:resnet20", None,
                     (3, 32, 32), ("accuracy",), "test")
    registry.register(spec)
    with pytest.raises(KeyError, match="Duplicate"):
        registry.register(spec)
