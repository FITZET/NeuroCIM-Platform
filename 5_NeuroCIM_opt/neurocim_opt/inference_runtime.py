"""Cached resources for real LeNet-5/MNIST inference."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import sys
from threading import RLock
from typing import Any


VISION_ZOO_MODEL_IDS = frozenset({
    "resnet8-mnist", "mobilenetv2-cifar10", "vgg11-bn-cifar10",
    "squeezenet1_1-cifar10",
})


@dataclass(frozen=True)
class ResourceSignature:
    path: str
    size: int
    mtime_ns: int


@dataclass
class LeNetInferenceResources:
    dataset: Any
    quantization: Any
    device_lut: Any
    float_model: Any
    ideal_model: Any
    device_model: Any
    device: Any


# All registered image classifiers expose the same inference surface.  Keep the
# historical LeNet name as an alias-compatible dataclass for existing callers.
RegisteredInferenceResources = LeNetInferenceResources


@dataclass(frozen=True)
class InferenceCacheInfo:
    hits: int
    misses: int
    invalidations: int
    entries: int


_LOCK = RLock()
_CACHE: dict[tuple[Any, ...], LeNetInferenceResources] = {}
_REGISTERED_CACHE: dict[tuple[Any, ...], RegisteredInferenceResources] = {}
_HITS = 0
_MISSES = 0
_INVALIDATIONS = 0


def _signature(path: Path, purpose: str) -> ResourceSignature:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required {purpose} is missing: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Required {purpose} is not a file: {path}")
    return ResourceSignature(str(path), stat.st_size, stat.st_mtime_ns)


def _mnist_test_files(dataset_root: Path) -> tuple[Path, Path]:
    raw = dataset_root / "MNIST" / "raw"
    return raw / "t10k-images-idx3-ubyte", raw / "t10k-labels-idx1-ubyte"


def mnist_required_files(dataset_root: Path, *, include_training: bool = False) -> tuple[Path, ...]:
    raw = dataset_root / "MNIST" / "raw"
    files = list(_mnist_test_files(dataset_root))
    if include_training:
        files.extend((raw / "train-images-idx3-ubyte", raw / "train-labels-idx1-ubyte"))
    return tuple(files)


def cifar10_required_files(dataset_root: Path) -> tuple[Path, ...]:
    """Files required by torchvision's extracted CIFAR-10 Python dataset."""
    root = dataset_root / "cifar-10-batches-py"
    names = ("batches.meta", "test_batch", *(f"data_batch_{index}" for index in range(1, 6)))
    return tuple(root / name for name in names)


def _logical_key(
    dataset_root: Path, checkpoint: Path, quantization: Path, error_model: Path, device: str
) -> tuple[str, ...]:
    return tuple(str(path) for path in (dataset_root, checkpoint, quantization, error_model)) + (device,)


def _cache_key(
    dataset_root: Path, checkpoint: Path, quantization: Path, error_model: Path, device: str
) -> tuple[Any, ...]:
    images, labels = _mnist_test_files(dataset_root)
    signatures = (
        _signature(images, "MNIST test images"),
        _signature(labels, "MNIST test labels"),
        _signature(checkpoint, "LeNet-5 checkpoint"),
        _signature(quantization, "quantization parameters"),
        _signature(error_model, "device error model"),
    )
    return (_logical_key(dataset_root, checkpoint, quantization, error_model, device), signatures)


def clear_lenet_inference_cache() -> None:
    """Drop cached datasets and models; useful for benchmarks and long-lived GUIs."""
    global _HITS, _MISSES, _INVALIDATIONS
    with _LOCK:
        _CACHE.clear()
        _REGISTERED_CACHE.clear()
        _HITS = _MISSES = _INVALIDATIONS = 0


def lenet_inference_cache_info() -> InferenceCacheInfo:
    with _LOCK:
        return InferenceCacheInfo(_HITS, _MISSES, _INVALIDATIONS, len(_CACHE))


def get_lenet_inference_resources(
    *,
    model_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    quantization_path: Path,
    error_model_path: Path,
    device_name: str,
) -> tuple[LeNetInferenceResources, bool]:
    """Return resources and whether this lookup hit the cache.

    Every lookup re-stats source files. A missing or changed source evicts the
    matching prior entry, so a live GUI never silently uses stale weights/LUTs.
    """
    global _HITS, _MISSES, _INVALIDATIONS
    logical = _logical_key(
        dataset_root, checkpoint_path, quantization_path, error_model_path, device_name
    )
    with _LOCK:
        try:
            key = _cache_key(
                dataset_root, checkpoint_path, quantization_path, error_model_path, device_name
            )
        except FileNotFoundError:
            stale = [key for key in _CACHE if key[0] == logical]
            for old_key in stale:
                del _CACHE[old_key]
            _INVALIDATIONS += len(stale)
            raise
        cached = _CACHE.get(key)
        if cached is not None:
            _HITS += 1
            return cached, True

        stale = [old_key for old_key in _CACHE if old_key[0] == logical]
        for old_key in stale:
            del _CACHE[old_key]
        _INVALIDATIONS += len(stale)
        _MISSES += 1

        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        import torch
        from torchvision.datasets import MNIST

        from lenet5.data import mnist_transform
        from lenet5.differential import configure_differential_model
        from lenet5.engine import load_checkpoint
        from lenet5.model import LeNet5
        from lenet5.quantization import ideal_multiplication_lut, load_quantization_parameters
        from .error_model import HardwareErrorModel

        try:
            dataset = MNIST(
                root=dataset_root, train=False, transform=mnist_transform(), download=False
            )
        except RuntimeError as exc:
            raise FileNotFoundError(
                f"MNIST test data is unavailable under {dataset_root}; install it before inference"
            ) from exc
        parameters = load_quantization_parameters(quantization_path)
        device_lut = HardwareErrorModel.load(error_model_path).lut
        float_model = LeNet5()
        load_checkpoint(float_model, checkpoint_path)
        ideal_model = configure_differential_model(
            deepcopy(float_model),
            parameters,
            ideal_multiplication_lut(),
            ste=False,
        )
        device_model = configure_differential_model(
            deepcopy(float_model), parameters, device_lut, ste=False
        )
        device = torch.device(device_name)
        for model in (float_model, ideal_model, device_model):
            model.eval().to(device)
        resources = LeNetInferenceResources(
            dataset=dataset,
            quantization=parameters,
            device_lut=device_lut,
            float_model=float_model,
            ideal_model=ideal_model,
            device_model=device_model,
            device=device,
        )
        _CACHE[key] = resources
        return resources, False


def _registered_key(
    model_id: str, dataset_root: Path, checkpoint: Path, auxiliary: Path | None,
    error_model: Path, device_name: str,
) -> tuple[Any, ...]:
    if model_id in {"resnet20-cifar10", "tinytransformer-cifar10"} or (
        model_id in VISION_ZOO_MODEL_IDS and model_id.endswith("cifar10")
    ):
        data_files = cifar10_required_files(dataset_root)
        purposes = tuple(f"CIFAR-10 {path.name}" for path in data_files)
        missing = [path for path in data_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"CIFAR-10 dataset is incomplete under {dataset_root}. Missing extracted files:\n"
                + "\n".join(str(path) for path in missing)
            )
    elif model_id in {"bnn-mnist", "resnet8-mnist"}:
        data_files = mnist_required_files(dataset_root, include_training=True)
        purposes = tuple(f"MNIST {path.name}" for path in data_files)
        missing = [path for path in data_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"MNIST dataset is incomplete under {dataset_root}. Missing raw files:\n"
                + "\n".join(str(path) for path in missing)
            )
    else:
        raise ValueError(f"Unsupported registered inference model: {model_id}")
    signatures = tuple(_signature(path, purpose) for path, purpose in zip(data_files, purposes))
    signatures += (
        _signature(checkpoint, f"{model_id} checkpoint"),
        _signature(error_model, "device error model"),
    )
    if auxiliary is not None:
        signatures += (_signature(auxiliary, f"{model_id} auxiliary parameters"),)
    logical = (model_id, str(dataset_root), str(checkpoint), str(auxiliary), str(error_model), device_name)
    return logical, signatures


def get_vision_zoo_inference_resources(
    *, model_id: str, model_root: Path, dataset_root: Path,
    checkpoint_path: Path, input_scales_path: Path | None,
    error_model_path: Path, device_name: str,
) -> tuple[RegisteredInferenceResources, bool]:
    """Load any VisionZoo classifier through its shared factory/data/engine API."""
    if model_id not in VISION_ZOO_MODEL_IDS:
        raise ValueError(f"Unsupported VisionZoo inference model: {model_id}")
    global _HITS, _MISSES, _INVALIDATIONS
    logical = (
        model_id, str(dataset_root), str(checkpoint_path), str(input_scales_path),
        str(error_model_path), device_name,
    )
    with _LOCK:
        try:
            key = _registered_key(
                model_id, dataset_root, checkpoint_path, input_scales_path,
                error_model_path, device_name,
            )
        except FileNotFoundError:
            stale = [item for item in _REGISTERED_CACHE if item[0] == logical]
            for old_key in stale:
                del _REGISTERED_CACHE[old_key]
            _INVALIDATIONS += len(stale)
            raise
        cached = _REGISTERED_CACHE.get(key)
        if cached is not None:
            _HITS += 1
            return cached, True
        stale = [item for item in _REGISTERED_CACHE if item[0] == logical]
        for old_key in stale:
            del _REGISTERED_CACHE[old_key]
        _INVALIDATIONS += len(stale)
        _MISSES += 1

        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        import json
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        from vision_zoo.checkpoint import load_checkpoint
        from vision_zoo.data import CIFAR10_MEAN, CIFAR10_STD, MNIST_MEAN, MNIST_STD
        from vision_zoo.engine import calibrate_input_scales, differential_copy
        from vision_zoo.models import create_model
        from .error_model import HardwareErrorModel

        is_cifar = model_id.endswith("cifar10")
        mean, std = (CIFAR10_MEAN, CIFAR10_STD) if is_cifar else (MNIST_MEAN, MNIST_STD)
        transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean, std)))
        dataset_class = datasets.CIFAR10 if is_cifar else datasets.MNIST
        test_dataset = dataset_class(
            root=dataset_root, train=False, transform=transform, download=False
        )
        float_model = load_checkpoint(create_model(model_id), checkpoint_path)
        if input_scales_path is not None:
            input_scales = json.loads(input_scales_path.read_text(encoding="utf-8"))
            if not isinstance(input_scales, dict) or not input_scales:
                raise ValueError(
                    f"{model_id} input scales must be a non-empty JSON object: "
                    f"{input_scales_path}"
                )
            input_scales = {str(name): float(value) for name, value in input_scales.items()}
        else:
            calibration_dataset = dataset_class(
                root=dataset_root, train=True, transform=transform, download=False
            )
            calibration = DataLoader(
                calibration_dataset, batch_size=16, shuffle=False, num_workers=0
            )
            input_scales = calibrate_input_scales(
                float_model, calibration, device=device_name, max_batches=1,
            )
        codes = np.arange(256, dtype=np.int64)
        ideal_lut = codes[:, None] * codes[None, :]
        device_lut = HardwareErrorModel.load(error_model_path).lut
        ideal_model = differential_copy(float_model, ideal_lut, input_scales)
        device_model = differential_copy(float_model, device_lut, input_scales)
        device = torch.device(device_name)
        for model in (float_model, ideal_model, device_model):
            model.eval().to(device)
        resources = RegisteredInferenceResources(
            dataset=test_dataset, quantization=input_scales, device_lut=device_lut,
            float_model=float_model, ideal_model=ideal_model,
            device_model=device_model, device=device,
        )
        _REGISTERED_CACHE[key] = resources
        return resources, False


def get_resnet20_inference_resources(
    *,
    model_root: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    input_scales_path: Path,
    error_model_path: Path,
    device_name: str,
    _model_id: str = "resnet20-cifar10",
) -> tuple[RegisteredInferenceResources, bool]:
    """Load/cache a CIFAR-10 model with signed-differential Conv/Linear layers."""
    global _HITS, _MISSES, _INVALIDATIONS
    model_id = _model_id
    logical = (model_id, str(dataset_root), str(checkpoint_path), str(input_scales_path),
               str(error_model_path), device_name)
    with _LOCK:
        try:
            key = _registered_key(model_id, dataset_root, checkpoint_path, input_scales_path,
                                  error_model_path, device_name)
        except FileNotFoundError:
            stale = [item for item in _REGISTERED_CACHE if item[0] == logical]
            for old_key in stale:
                del _REGISTERED_CACHE[old_key]
            _INVALIDATIONS += len(stale)
            raise
        cached = _REGISTERED_CACHE.get(key)
        if cached is not None:
            _HITS += 1
            return cached, True
        stale = [item for item in _REGISTERED_CACHE if item[0] == logical]
        for old_key in stale:
            del _REGISTERED_CACHE[old_key]
        _INVALIDATIONS += len(stale)
        _MISSES += 1

        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        import json
        import numpy as np
        import torch
        from torchvision import datasets, transforms

        if model_id == "tinytransformer-cifar10":
            from tiny_transformer import cct_7_3x1_32 as model_factory
            from tiny_transformer.data import CIFAR10_MEAN, CIFAR10_STD
            from tiny_transformer.engine import differential_copy
        else:
            from resnet20.data import CIFAR10_MEAN, CIFAR10_STD
            from resnet20.engine import differential_copy
            from resnet20.model import resnet20 as model_factory
        from .error_model import HardwareErrorModel

        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        try:
            dataset = datasets.CIFAR10(
                root=dataset_root, train=False, transform=transform, download=False
            )
        except RuntimeError as exc:
            expected = "\n".join(str(path) for path in cifar10_required_files(dataset_root))
            raise FileNotFoundError(
                f"CIFAR-10 Python dataset is incomplete under {dataset_root}. "
                f"Required extracted files:\n{expected}"
            ) from exc
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("best_model_state_dict", payload.get("model_state_dict", payload))
        float_model = model_factory()
        float_model.load_state_dict(state)
        input_scales = json.loads(input_scales_path.read_text(encoding="utf-8"))
        if not isinstance(input_scales, dict) or not input_scales:
            raise ValueError(f"{model_id} input scales must be a non-empty JSON object: {input_scales_path}")
        input_scales = {str(name): float(value) for name, value in input_scales.items()}
        codes = np.arange(256, dtype=np.int64)
        ideal_lut = codes[:, None] * codes[None, :]
        device_lut = HardwareErrorModel.load(error_model_path).lut
        ideal_model = differential_copy(float_model, ideal_lut, input_scales)
        device_model = differential_copy(float_model, device_lut, input_scales)
        device = torch.device(device_name)
        for model in (float_model, ideal_model, device_model):
            model.eval().to(device)
        resources = RegisteredInferenceResources(
            dataset=dataset,
            quantization=input_scales,
            device_lut=device_lut,
            float_model=float_model,
            ideal_model=ideal_model,
            device_model=device_model,
            device=device,
        )
        _REGISTERED_CACHE[key] = resources
        return resources, False


def get_tinytransformer_inference_resources(
    *, model_root: Path, dataset_root: Path, checkpoint_path: Path,
    input_scales_path: Path, error_model_path: Path, device_name: str,
) -> tuple[RegisteredInferenceResources, bool]:
    return get_resnet20_inference_resources(
        model_root=model_root, dataset_root=dataset_root,
        checkpoint_path=checkpoint_path, input_scales_path=input_scales_path,
        error_model_path=error_model_path, device_name=device_name,
        _model_id="tinytransformer-cifar10",
    )


def get_bnn_inference_resources(
    *, model_root: Path, dataset_root: Path, checkpoint_path: Path,
    error_model_path: Path, device_name: str,
) -> tuple[RegisteredInferenceResources, bool]:
    """Load/cache frozen BNN plus calibrated ideal and device differential copies."""
    global _HITS, _MISSES, _INVALIDATIONS
    model_id = "bnn-mnist"
    logical = (model_id, str(dataset_root), str(checkpoint_path), str(None),
               str(error_model_path), device_name)
    with _LOCK:
        try:
            key = _registered_key(
                model_id, dataset_root, checkpoint_path, None, error_model_path, device_name
            )
        except FileNotFoundError:
            stale = [item for item in _REGISTERED_CACHE if item[0] == logical]
            for old_key in stale:
                del _REGISTERED_CACHE[old_key]
            _INVALIDATIONS += len(stale)
            raise
        cached = _REGISTERED_CACHE.get(key)
        if cached is not None:
            _HITS += 1
            return cached, True
        stale = [item for item in _REGISTERED_CACHE if item[0] == logical]
        for old_key in stale:
            del _REGISTERED_CACHE[old_key]
        _INVALIDATIONS += len(stale)
        _MISSES += 1

        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        import torch
        from bnn.data import create_loaders
        from bnn.differential import collect_input_scales, differential_models
        from bnn.model import BinaryMNIST
        from .error_model import HardwareErrorModel

        try:
            calibration, _validation, test = create_loaders(
                dataset_root, 128, 256, 42, num_workers=0
            )
        except RuntimeError as exc:
            expected = "\n".join(
                str(path) for path in mnist_required_files(dataset_root, include_training=True)
            )
            raise FileNotFoundError(
                f"MNIST training/test data is incomplete under {dataset_root}. "
                f"Required raw files:\n{expected}"
            ) from exc
        latent = BinaryMNIST()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("best_model_state_dict", payload.get("model_state_dict", payload))
        latent.load_state_dict(state)
        frozen = latent.frozen().eval()
        input_scales = collect_input_scales(
            frozen, calibration, device_name, batches=2
        )
        ideal_model, device_model = differential_models(
            frozen.cpu(), input_scales, error_model_path
        )
        device = torch.device(device_name)
        for model in (frozen, ideal_model, device_model):
            model.eval().to(device)
        resources = RegisteredInferenceResources(
            dataset=test.dataset,
            quantization=input_scales,
            device_lut=HardwareErrorModel.load(error_model_path).lut,
            float_model=frozen,
            ideal_model=ideal_model,
            device_model=device_model,
            device=device,
        )
        _REGISTERED_CACHE[key] = resources
        return resources, False


def get_registered_inference_resources(
    *, model_id: str, model_root: Path, dataset_root: Path, checkpoint_path: Path,
    auxiliary_path: Path | None, error_model_path: Path, device_name: str,
) -> tuple[RegisteredInferenceResources, bool]:
    """Dispatch to a cache-backed registered image-classifier runtime."""
    if model_id == "lenet5-mnist":
        return get_lenet_inference_resources(
            model_root=model_root, dataset_root=dataset_root,
            checkpoint_path=checkpoint_path, quantization_path=auxiliary_path,
            error_model_path=error_model_path, device_name=device_name,
        )
    if model_id == "resnet20-cifar10":
        if auxiliary_path is None:
            raise FileNotFoundError("ResNet-20 input-scales JSON was not selected")
        return get_resnet20_inference_resources(
            model_root=model_root, dataset_root=dataset_root,
            checkpoint_path=checkpoint_path, input_scales_path=auxiliary_path,
            error_model_path=error_model_path, device_name=device_name,
        )
    if model_id == "tinytransformer-cifar10":
        if auxiliary_path is None:
            raise FileNotFoundError("TinyTransformer input-scales JSON was not selected")
        return get_tinytransformer_inference_resources(
            model_root=model_root, dataset_root=dataset_root,
            checkpoint_path=checkpoint_path, input_scales_path=auxiliary_path,
            error_model_path=error_model_path, device_name=device_name,
        )
    if model_id == "bnn-mnist":
        return get_bnn_inference_resources(
            model_root=model_root, dataset_root=dataset_root,
            checkpoint_path=checkpoint_path, error_model_path=error_model_path,
            device_name=device_name,
        )
    if model_id in VISION_ZOO_MODEL_IDS:
        return get_vision_zoo_inference_resources(
            model_id=model_id, model_root=model_root, dataset_root=dataset_root,
            checkpoint_path=checkpoint_path, input_scales_path=auxiliary_path,
            error_model_path=error_model_path, device_name=device_name,
        )
    raise ValueError(f"Unsupported registered inference model: {model_id}")
