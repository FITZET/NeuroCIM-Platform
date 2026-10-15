from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Callable

from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    task: str
    datasets: tuple[str, ...]
    factory: str
    checkpoint: Path | None
    input_shape: tuple[int, ...]
    metrics: tuple[str, ...]
    license: str

    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint is not None and self.checkpoint.is_file()

    def create_model(self) -> nn.Module:
        module_name, separator, attribute = self.factory.partition(":")
        if not separator:
            raise ValueError(f"Invalid factory path: {self.factory}")
        factory: Callable[[], nn.Module] = getattr(importlib.import_module(module_name), attribute)
        return factory()


class ModelRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        if spec.model_id in self._items:
            raise KeyError(f"Duplicate model id: {spec.model_id}")
        self._items[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._items[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model {model_id!r}; available: {sorted(self._items)}") from exc

    def list(self, *, dataset: str | None = None, task: str | None = None) -> tuple[ModelSpec, ...]:
        items = self._items.values()
        if dataset is not None:
            items = (item for item in items if dataset in item.datasets)
        if task is not None:
            items = (item for item in items if item.task == task)
        return tuple(sorted(items, key=lambda item: item.model_id))


def builtin_registry(project_root: str | Path) -> ModelRegistry:
    root = Path(project_root).expanduser().resolve()
    for model_root in (
        root / "3_models/LeNet_5", root / "3_models/Resnet20", root / "3_models/BNN",
        root / "3_models/TinyTransformer", root / "3_models/GPT2",
    ):
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            model_id="lenet5-mnist", display_name="LeNet-5 / MNIST",
            task="image_classification", datasets=("mnist",),
            factory="lenet5.model:LeNet5",
            checkpoint=root / "3_models/LeNet_5/chekpoint/lenet5_float.pt",
            input_shape=(1, 32, 32), metrics=("accuracy", "loss"),
            license="project model; MNIST dataset terms apply",
        )
    )
    registry.register(
        ModelSpec(
            model_id="resnet20-cifar10", display_name="ResNet-20 / CIFAR-10",
            task="image_classification", datasets=("cifar10",),
            factory="resnet20.model:resnet20",
            checkpoint=root / "3_models/Resnet20/checkpoint/resnet20_cifar10.pt",
            input_shape=(3, 32, 32), metrics=("accuracy", "loss"),
            license="project implementation; CIFAR-10 dataset terms apply",
        )
    )
    registry.register(
        ModelSpec(
            model_id="bnn-mnist", display_name="Binary CNN / MNIST",
            task="image_classification", datasets=("mnist",),
            factory="bnn.model:BinaryMNIST",
            checkpoint=root / "3_models/BNN/checkpoint/bnn_mnist.pt",
            input_shape=(1, 28, 28), metrics=("accuracy", "loss"),
            license="project implementation; MNIST dataset terms apply",
        )
    )
    registry.register(
        ModelSpec(
            model_id="tinytransformer-cifar10",
            display_name="TinyTransformer (CCT-7/3x1) / CIFAR-10",
            task="image_classification", datasets=("cifar10",),
            factory="tiny_transformer:cct_7_3x1_32",
            checkpoint=root / "3_models/TinyTransformer/checkpoint/cct_7_3x1_32_cifar10.pth",
            input_shape=(3, 32, 32), metrics=("accuracy", "loss"),
            license="Apache-2.0; CIFAR-10 dataset terms apply",
        )
    )
    registry.register(
        ModelSpec(
            model_id="gpt2-small", display_name="GPT-2 Small / Text Generation",
            task="text_generation", datasets=("prompt_text",),
            factory="gpt2_cim:gpt2_small",
            checkpoint=root / "3_models/GPT2/checkpoint/gpt2_nanogpt.pt",
            input_shape=(1024,), metrics=("loss", "perplexity", "token_agreement"),
            license="nanoGPT MIT; OpenAI GPT-2 modified MIT",
        )
    )
    return registry
