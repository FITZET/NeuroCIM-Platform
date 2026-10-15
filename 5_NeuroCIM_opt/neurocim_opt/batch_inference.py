"""UI-neutral grouped inference for float, ideal and device-error models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable


ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


class BatchInferenceCancelled(RuntimeError):
    """Raised when a caller cancels a running batch inference."""


@dataclass(frozen=True)
class BatchInferenceRequest:
    project_root: Path
    dataset_root: Path
    checkpoint_path: Path
    quantization_path: Path
    error_model_path: Path
    model_id: str = "lenet5_mnist"
    device: str = "auto"
    sample_count: int = 10_000
    group_size: int = 100
    inference_batch_size: int = 128
    output_path: Path | None = None


@dataclass(frozen=True)
class AccuracyGroup:
    start_index: int
    end_index: int
    sample_count: int
    accuracy: dict[str, float]


@dataclass(frozen=True)
class AccuracySummary:
    correct: int
    sample_count: int
    accuracy: float
    elapsed_ms: float


@dataclass(frozen=True)
class GroupedBatchInferenceOutcome:
    model_id: str
    dataset: str
    device: str
    sample_count: int
    group_size: int
    inference_batch_size: int
    cache_hit: bool
    preparation_ms: float
    total_elapsed_ms: float
    modes: dict[str, AccuracySummary]
    groups: tuple[AccuracyGroup, ...]
    resources: dict[str, str]
    output_path: Path | None = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_path"] = str(self.output_path) if self.output_path else None
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
        return target


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _canonical_model_id(model_id: str) -> str:
    normalized = model_id.strip().lower()
    aliases = {
        "lenet5": "lenet5-mnist", "lenet-5": "lenet5-mnist",
        "lenet5_mnist": "lenet5-mnist", "lenet-5/mnist": "lenet5-mnist",
        "lenet5-mnist": "lenet5-mnist",
        "resnet20": "resnet20-cifar10", "resnet-20": "resnet20-cifar10",
        "resnet20_cifar10": "resnet20-cifar10", "resnet-20/cifar-10": "resnet20-cifar10",
        "resnet20-cifar10": "resnet20-cifar10",
        "bnn": "bnn-mnist", "binary-cnn": "bnn-mnist", "bnn_mnist": "bnn-mnist",
        "bnn/mnist": "bnn-mnist", "bnn-mnist": "bnn-mnist",
        "tinytransformer": "tinytransformer-cifar10", "tiny-transformer": "tinytransformer-cifar10",
        "cct": "tinytransformer-cifar10", "cct-7": "tinytransformer-cifar10",
        "tinytransformer-cifar10": "tinytransformer-cifar10",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Grouped inference supports LeNet-5/MNIST, BNN/MNIST, ResNet-20/CIFAR-10 "
            "and TinyTransformer/CIFAR-10; "
            f"got {model_id!r}"
        ) from exc


def _validate(request: BatchInferenceRequest) -> tuple[str, Path, Path, Path, Path, Path]:
    model_id = _canonical_model_id(request.model_id)
    if request.sample_count <= 0 or request.sample_count > 10_000:
        raise ValueError("sample_count must be in [1, 10000] for the selected test dataset")
    if request.group_size <= 0:
        raise ValueError("group_size must be positive")
    if request.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    project = _resolved(request.project_root)
    dataset = _resolved(request.dataset_root)
    checkpoint = _resolved(request.checkpoint_path)
    quantization = _resolved(request.quantization_path)
    error_model = _resolved(request.error_model_path)
    if model_id in {"lenet5-mnist", "bnn-mnist"}:
        from .inference_runtime import mnist_required_files
        dataset_files = tuple(
            (f"MNIST {path.name}", path)
            for path in mnist_required_files(dataset, include_training=model_id == "bnn-mnist")
        )
        auxiliary_purpose = "quantization" if model_id == "lenet5-mnist" else None
    else:
        from .inference_runtime import cifar10_required_files
        dataset_files = tuple(
            (f"CIFAR-10 {path.name}", path) for path in cifar10_required_files(dataset)
        )
        auxiliary_purpose = "input scales"
    missing = [
        f"{purpose}: {path}"
        for purpose, path in (*dataset_files,
            ("checkpoint", checkpoint),
            ("error model", error_model),
        ) + (() if auxiliary_purpose is None else ((auxiliary_purpose, quantization),))
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing batch inference resources:\n" + "\n".join(missing))
    return model_id, project, dataset, checkpoint, quantization, error_model


def run_grouped_batch_inference(
    request: BatchInferenceRequest,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> GroupedBatchInferenceOutcome:
    """Evaluate contiguous registered test samples and aggregate every N samples.

    Accuracy values are fractions in ``[0, 1]``. The final group may contain
    fewer than ``group_size`` samples. Models and data are obtained from the
    same cache used by single-sample inference.
    """
    total_started = perf_counter()
    model_id, project, dataset, checkpoint, quantization, error_model = _validate(request)

    import torch

    from .inference_runtime import get_registered_inference_resources

    requested_device = request.device.strip().lower()
    if requested_device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cpu", "cuda"}:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is not available")
        device_name = requested_device
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")

    model_directory = {
        "lenet5-mnist": "LeNet_5", "bnn-mnist": "BNN", "resnet20-cifar10": "Resnet20",
        "tinytransformer-cifar10": "TinyTransformer",
    }[model_id]
    model_root = project / "3_models" / model_directory
    resources, cache_hit = get_registered_inference_resources(
        model_id=model_id,
        model_root=model_root,
        dataset_root=dataset,
        checkpoint_path=checkpoint,
        auxiliary_path=quantization if model_id != "bnn-mnist" else None,
        error_model_path=error_model,
        device_name=device_name,
    )
    if request.sample_count > len(resources.dataset):
        raise ValueError(
            f"sample_count must be in [1, {len(resources.dataset)}] for the selected test dataset"
        )
    preparation_ms = (perf_counter() - total_started) * 1_000.0
    mode_models = (
        ("float", resources.float_model),
        ("ideal_differential", resources.ideal_model),
        ("device_differential", resources.device_model),
    )
    correct_by_mode: dict[str, list[bool]] = {name: [] for name, _ in mode_models}
    elapsed_by_mode = {name: 0.0 for name, _ in mode_models}

    if progress:
        progress(0, request.sample_count)
    for start in range(0, request.sample_count, request.inference_batch_size):
        if cancelled and cancelled():
            raise BatchInferenceCancelled(
                f"Batch inference cancelled after {start} of {request.sample_count} samples"
            )
        end = min(start + request.inference_batch_size, request.sample_count)
        samples = [resources.dataset[index] for index in range(start, end)]
        inputs = torch.stack([sample[0] for sample in samples]).to(resources.device)
        labels = torch.tensor(
            [int(sample[1]) for sample in samples], device=resources.device, dtype=torch.long
        )
        for mode, model in mode_models:
            if resources.device.type == "cuda":
                torch.cuda.synchronize()
            started = perf_counter()
            with torch.inference_mode():
                predicted = model(inputs).argmax(dim=1)
            if resources.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_by_mode[mode] += (perf_counter() - started) * 1_000.0
            correct_by_mode[mode].extend(
                bool(value) for value in predicted.eq(labels).detach().cpu().tolist()
            )
        if progress:
            progress(end, request.sample_count)

    summaries = {
        mode: AccuracySummary(
            correct=sum(values),
            sample_count=request.sample_count,
            accuracy=sum(values) / request.sample_count,
            elapsed_ms=elapsed_by_mode[mode],
        )
        for mode, values in correct_by_mode.items()
    }
    groups = []
    for start in range(0, request.sample_count, request.group_size):
        end = min(start + request.group_size, request.sample_count)
        groups.append(
            AccuracyGroup(
                start_index=start,
                end_index=end - 1,
                sample_count=end - start,
                accuracy={
                    mode: sum(values[start:end]) / (end - start)
                    for mode, values in correct_by_mode.items()
                },
            )
        )
    output = _resolved(request.output_path) if request.output_path else None
    resource_paths = {
        "dataset_root": str(dataset),
        "checkpoint": str(checkpoint),
        "error_model": str(error_model),
    }
    if model_id == "bnn-mnist":
        resource_paths["input_scales"] = "runtime calibration from MNIST validation samples"
    else:
        resource_paths["quantization" if model_id == "lenet5-mnist" else "input_scales"] = str(quantization)
    outcome = GroupedBatchInferenceOutcome(
        model_id=model_id,
        dataset="CIFAR-10/test" if model_id.endswith("cifar10") else "MNIST/test",
        device=str(resources.device),
        sample_count=request.sample_count,
        group_size=request.group_size,
        inference_batch_size=request.inference_batch_size,
        cache_hit=cache_hit,
        preparation_ms=preparation_ms,
        total_elapsed_ms=(perf_counter() - total_started) * 1_000.0,
        modes=summaries,
        groups=tuple(groups),
        resources=resource_paths,
        output_path=output,
    )
    if output:
        outcome.save(output)
    return outcome
