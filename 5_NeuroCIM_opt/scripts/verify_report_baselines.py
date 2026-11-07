from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from neurocim_opt.model_registry import builtin_registry


TARGETS = {
    "resnet8-mnist": 0.95,
    "mobilenetv2-cifar10": 0.84,
    "vgg11-bn-cifar10": 0.85,
    "squeezenet1_1-cifar10": 0.81,
    "bnn-mnist": 0.99,
    # The installed CCT checkpoint is retained in the audit until the report's
    # sentiment/synthetic TinyTransformer is backed by a compatible artifact.
    "tinytransformer-cifar10": 0.996,
}


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    for key in ("best_model_state_dict", "model_state_dict", "state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload


def evaluate_image_model(model_id: str, device: torch.device, batch_size: int) -> dict:
    spec = builtin_registry(WORKSPACE).get(model_id)
    if spec.checkpoint is None or spec.dataset_root is None:
        raise FileNotFoundError(f"Incomplete resources for {model_id}")
    model = spec.create_model()
    model.load_state_dict(load_state(spec.checkpoint))
    model.eval().to(device)
    transform = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize(spec.normalization_mean, spec.normalization_std),
    ))
    dataset_cls = datasets.CIFAR10 if "cifar10" in spec.datasets else datasets.MNIST
    dataset = dataset_cls(spec.dataset_root, train=False, transform=transform, download=False)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    correct = samples = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            correct += int(model(inputs).argmax(1).eq(labels).sum())
            samples += labels.numel()
    accuracy = correct / samples
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model_id": model_id,
        "registered_task": spec.task,
        "registered_dataset": spec.datasets[0],
        "report_task": spec.report_task or spec.task,
        "report_dataset": spec.report_dataset or spec.datasets[0],
        "samples": samples,
        "correct": correct,
        "accuracy": accuracy,
        "target": TARGETS[model_id],
        "passed": accuracy >= TARGETS[model_id],
        "checkpoint": str(spec.checkpoint),
        "source_url": spec.source_url,
        "source_version": spec.source_version,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify report baselines on complete test sets")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--models", nargs="*", default=list(TARGETS))
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "outputs" / "acceptance" / "report_baselines.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [evaluate_image_model(item, device, args.batch_size) for item in args.models]
    payload = {
        "schema_version": 1,
        "evaluation_scope": "complete official test split",
        "device": str(device),
        "results": results,
        "all_passed": all(item["passed"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
