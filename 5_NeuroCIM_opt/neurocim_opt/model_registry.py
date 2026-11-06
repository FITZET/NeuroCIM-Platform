from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
    # Resource and provenance metadata.  These fields intentionally follow the
    # original constructor arguments so existing positional callers remain
    # compatible.
    model_root: Path | None = None
    auxiliary_resources: tuple[Path, ...] = ()
    dataset_root: Path | None = None
    class_names: tuple[str, ...] = ()
    normalization_mean: tuple[float, ...] = ()
    normalization_std: tuple[float, ...] = ()
    capabilities: tuple[str, ...] = ()
    source_url: str = ""
    source_version: str = ""
    weight_sha256: str | None = None

    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint is not None and self.checkpoint.is_file()

    @property
    def auxiliary_resources_available(self) -> bool:
        return all(path.exists() for path in self.auxiliary_resources)

    @property
    def resources_available(self) -> bool:
        """Whether every artifact required to execute this combination exists."""
        dataset_ready = self.dataset_root is None or self.dataset_root.exists()
        return self.checkpoint_available and self.auxiliary_resources_available and dataset_ready

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

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

    def list(
        self, *, dataset: str | None = None, task: str | None = None,
        capability: str | None = None,
    ) -> tuple[ModelSpec, ...]:
        items = self._items.values()
        if dataset is not None:
            items = (item for item in items if dataset in item.datasets)
        if task is not None:
            items = (item for item in items if item.task == task)
        if capability is not None:
            items = (item for item in items if item.supports(capability))
        return tuple(sorted(items, key=lambda item: item.model_id))


def builtin_registry(project_root: str | Path) -> ModelRegistry:
    root = Path(project_root).expanduser().resolve()
    lenet_root = root / "3_models/LeNet_5"
    resnet_root = root / "3_models/Resnet20"
    bnn_root = root / "3_models/BNN"
    tiny_root = root / "3_models/TinyTransformer"
    gpt2_root = root / "3_models/GPT2"
    vision_root = root / "3_models/VisionZoo"
    mnist_root = root / "data/mnist"
    cifar10_root = root / "data/cifar10"
    mnist_classes = tuple(str(index) for index in range(10))
    cifar10_classes = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )
    image_capabilities = (
        "single_inference", "batch_inference", "error_aware_training",
    )
    for model_root in (lenet_root, resnet_root, bnn_root, tiny_root, gpt2_root, vision_root):
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            model_id="lenet5-mnist", display_name="LeNet-5",
            task="image_classification", datasets=("mnist",),
            factory="lenet5.model:LeNet5",
            checkpoint=lenet_root / "chekpoint/lenet5_float.pt",
            input_shape=(1, 32, 32), metrics=("accuracy", "loss"),
            license="project model; MNIST dataset terms apply",
            model_root=lenet_root,
            auxiliary_resources=(lenet_root / "chekpoint/quantization.json",),
            dataset_root=mnist_root, class_names=mnist_classes,
            normalization_mean=(0.1307,), normalization_std=(0.3081,),
            capabilities=image_capabilities,
            source_url="https://github.com/FITZET/NeuroCIM-Platform",
            source_version="project-lenet5-v1",
            weight_sha256="48c47fbbdeb4d8e4b71762d421d78d4e45496a5fbc97d6dbe4410900ace37825",
        )
    )
    vision_models = (
        (
            "resnet8-mnist", "ResNet-8", "mnist", "resnet8_mnist",
            vision_root / "checkpoint/resnet8_mnist.pt",
            vision_root / "checkpoint/resnet8-mnist_input_scales.json",
            (1, 28, 28), mnist_classes, (0.1307,), (0.3081,),
            "Project ResNet-8 implementation; MNIST dataset terms apply",
            "https://github.com/FITZET/NeuroCIM-Platform", "project-resnet8-mnist-v1",
            "b5130a0b3234d947dd0cbabb90f11631c1e25f23289c55cbc4a41b7ee0a986ed",
        ),
        (
            "mobilenetv2-cifar10", "MobileNetV2 x1.0", "cifar10",
            "mobilenetv2_x1_0_cifar10",
            vision_root / "checkpoint/cifar10_mobilenetv2_x1_0-fe6a5b48.pt",
            vision_root / "checkpoint/mobilenetv2-cifar10_input_scales.json",
            (3, 32, 32), cifar10_classes,
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
            "BSD-3-Clause; CIFAR-10 dataset terms apply",
            "https://github.com/chenyaofo/pytorch-cifar-models",
            "786c16252c0fc58ee9adac063f8337cc4a7a497a",
            "fe6a5b4808c9b1b7a440c02fa19a10dc3d8ea5c63e38381b2f60b1937958735e",
        ),
        (
            "vgg11-bn-cifar10", "VGG11-BN", "cifar10", "vgg11_bn_cifar10",
            vision_root / "checkpoint/cifar10_vgg11_bn-eaeebf42.pt",
            vision_root / "checkpoint/vgg11-bn-cifar10_input_scales.json",
            (3, 32, 32), cifar10_classes,
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
            "BSD-3-Clause; CIFAR-10 dataset terms apply",
            "https://github.com/chenyaofo/pytorch-cifar-models",
            "786c16252c0fc58ee9adac063f8337cc4a7a497a",
            "eaeebf42370c92fdfbb5dbe8eba7c27d4eb7366a1dca0b5be435364ccdf54378",
        ),
        (
            "squeezenet1_1-cifar10", "SqueezeNet 1.1", "cifar10",
            "squeezenet1_1_cifar10",
            vision_root / "checkpoint/squeezenet1_1_cifar10.pt",
            vision_root / "checkpoint/squeezenet1_1-cifar10_input_scales.json",
            (3, 32, 32), cifar10_classes,
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
            "torchvision BSD-3-Clause; CIFAR-10 dataset terms apply",
            "https://github.com/pytorch/vision", "v0.19.0-squeezenet1_1",
            "df967fd502b9d8fe2e7dea77d8363ad7fe14df04efcc173df1f94fc05f7cfe4c",
        ),
    )
    for (
        model_id, display_name, dataset, factory_name, checkpoint, input_scales,
        shape, classes, mean, std,
        license_name, source, version, expected_sha256,
    ) in vision_models:
        actual_sha256 = expected_sha256
        if checkpoint.is_file() and actual_sha256 is None:
            with checkpoint.open("rb") as stream:
                actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        registry.register(ModelSpec(
            model_id=model_id, display_name=display_name,
            task="image_classification", datasets=(dataset,),
            factory=f"vision_zoo.models:{factory_name}",
            checkpoint=checkpoint, input_shape=shape,
            metrics=("accuracy", "loss"), license=license_name,
            model_root=vision_root, dataset_root=mnist_root if dataset == "mnist" else cifar10_root,
            auxiliary_resources=(input_scales,),
            class_names=classes, normalization_mean=mean, normalization_std=std,
            capabilities=image_capabilities, source_url=source,
            source_version=version, weight_sha256=actual_sha256,
        ))
    registry.register(
        ModelSpec(
            model_id="resnet20-cifar10", display_name="ResNet-20",
            task="image_classification", datasets=("cifar10",),
            factory="resnet20.model:resnet20",
            checkpoint=resnet_root / "checkpoint/resnet20_cifar10.pt",
            input_shape=(3, 32, 32), metrics=("accuracy", "loss"),
            license="project implementation; CIFAR-10 dataset terms apply",
            model_root=resnet_root,
            auxiliary_resources=(resnet_root / "output/formal/input_scales.json",),
            dataset_root=cifar10_root, class_names=cifar10_classes,
            normalization_mean=(0.4914, 0.4822, 0.4465),
            normalization_std=(0.2470, 0.2435, 0.2616),
            capabilities=image_capabilities,
            source_url="https://github.com/FITZET/NeuroCIM-Platform",
            source_version="canonical-cifar-resnet20-v1",
            weight_sha256="e6d5b6d6a49dfee5e6dd36de508a6fbbb89fe75aa1ca77803cc55ffd15fd816e",
        )
    )
    registry.register(
        ModelSpec(
            model_id="bnn-mnist", display_name="BNN (BinaryNet)",
            task="image_classification", datasets=("mnist",),
            factory="bnn.model:BinaryMNIST",
            checkpoint=bnn_root / "checkpoint/bnn_mnist.pt",
            input_shape=(1, 28, 28), metrics=("accuracy", "loss"),
            license="project implementation; MNIST dataset terms apply",
            model_root=bnn_root, dataset_root=mnist_root,
            class_names=mnist_classes,
            normalization_mean=(0.1307,), normalization_std=(0.3081,),
            capabilities=image_capabilities,
            source_url="https://github.com/FITZET/NeuroCIM-Platform",
            source_version="binary-mnist-v1",
            weight_sha256="c7b9f3447864c8848ada6d46b65d129e90ac48440462530bb5d3a148636e695c",
        )
    )
    registry.register(
        ModelSpec(
            model_id="tinytransformer-cifar10",
            display_name="TinyTransformer (CCT-7/3x1)",
            task="image_classification", datasets=("cifar10",),
            factory="tiny_transformer:cct_7_3x1_32",
            checkpoint=tiny_root / "checkpoint/cct_7_3x1_32_cifar10.pth",
            input_shape=(3, 32, 32), metrics=("accuracy", "loss"),
            license="Apache-2.0; CIFAR-10 dataset terms apply",
            model_root=tiny_root,
            auxiliary_resources=(tiny_root / "checkpoint/input_scales.json",),
            dataset_root=cifar10_root, class_names=cifar10_classes,
            normalization_mean=(0.4914, 0.4822, 0.4465),
            normalization_std=(0.2470, 0.2435, 0.2616),
            capabilities=image_capabilities,
            source_url="https://github.com/SHI-Labs/Compact-Transformers",
            source_version="cct_7_3x1_32",
            weight_sha256="8abfcd3d98674cd5b6fe07361f37001f7d2ebe57f513e5cc724ccdc8790204fe",
        )
    )
    registry.register(
        ModelSpec(
            model_id="gpt2-small", display_name="GPT-2 Small",
            task="text_generation", datasets=("prompt_text",),
            factory="gpt2_cim:gpt2_small",
            checkpoint=gpt2_root / "checkpoint/gpt2_nanogpt.pt",
            input_shape=(1024,), metrics=("loss", "perplexity", "token_agreement"),
            license="nanoGPT MIT; OpenAI GPT-2 modified MIT",
            model_root=gpt2_root,
            auxiliary_resources=(
                gpt2_root / "checkpoint/input_scales.json",
                gpt2_root / "checkpoint/tokenizer",
            ),
            capabilities=(
                "single_inference", "batch_inference", "text_generation",
                "error_aware_training",
            ),
            source_url="https://github.com/karpathy/nanoGPT",
            source_version="gpt2-124M",
            weight_sha256="095c16e3c0dcd897b40046139828e77aed24bb64985e4358357e529616c5e534",
        )
    )
    gpt2_large_root = gpt2_root / "checkpoint/gpt2-large"
    wikitext2_root = gpt2_large_root / "wikitext-2-raw-v1"
    registry.register(
        ModelSpec(
            model_id="gpt2-large", display_name="GPT-2 Large",
            task="text_generation", datasets=("wikitext2",),
            factory="gpt2_cim:gpt2_large",
            checkpoint=gpt2_large_root / "model.safetensors",
            input_shape=(1024,), metrics=("loss", "perplexity", "token_agreement"),
            license="OpenAI GPT-2 modified MIT; WikiText CC BY-SA 3.0/GFDL",
            model_root=gpt2_root,
            auxiliary_resources=(
                gpt2_large_root / "config.json",
                gpt2_large_root / "merges.txt",
                gpt2_large_root / "tokenizer_config.json",
                gpt2_large_root / "vocab.json",
                wikitext2_root / "wiki.train.raw",
                wikitext2_root / "wiki.valid.raw",
                wikitext2_root / "wiki.test.raw",
            ),
            dataset_root=wikitext2_root,
            capabilities=(
                "single_inference", "batch_inference", "text_generation",
                "language_model_evaluation",
            ),
            source_url="https://github.com/openai/gpt-2",
            source_version="HF-32b71b12589c2f8d625668d2335a01cac3249519",
            weight_sha256="5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce",
        )
    )
    return registry
