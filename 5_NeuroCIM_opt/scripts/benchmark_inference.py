"""Measure real cold, cached, and batched LeNet inference on CPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLATFORM_ROOT.parent
sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.application import PlatformService
from neurocim_opt.inference_runtime import clear_lenet_inference_cache, lenet_inference_cache_info
from neurocim_opt.paths import data_root


def prediction_values(outcome) -> dict[str, dict[str, list[float | int]]]:
    return {
        item.mode: {
            "labels": list(item.predicted_labels),
            "confidences": list(item.confidences),
        }
        for item in outcome.predictions
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PLATFORM_ROOT / "outputs/acceptance/performance/inference_benchmark.json",
    )
    args = parser.parse_args()
    dataset = (args.dataset_root or data_root("mnist")).resolve()
    clear_lenet_inference_cache()
    cold = PlatformService.infer_lenet5_mnist_batch(
        args.project_root, (0,), dataset_root=dataset, device="cpu"
    )
    after_cold = asdict(lenet_inference_cache_info())
    hot = PlatformService.infer_lenet5_mnist_batch(
        args.project_root, (0,), dataset_root=dataset, device="cpu"
    )
    after_hot = asdict(lenet_inference_cache_info())
    batch = PlatformService.infer_lenet5_mnist_batch(
        args.project_root, tuple(range(10)), dataset_root=dataset, device="cpu"
    )
    after_batch = asdict(lenet_inference_cache_info())
    if prediction_values(cold) != prediction_values(hot):
        raise RuntimeError("cached predictions differ from cold predictions")
    payload = {
        "kind": "real_lenet5_mnist_cpu_inference",
        "dataset_root": str(dataset),
        "cold_single": cold.as_dict(),
        "hot_single": hot.as_dict(),
        "hot_batch_10": batch.as_dict(),
        "cache_info": {
            "after_cold": after_cold,
            "after_hot": after_hot,
            "after_batch": after_batch,
        },
        "prediction_consistency": prediction_values(cold) == prediction_values(hot),
        "targets_ms": {"hot_single": 1000.0, "hot_batch_10": 10000.0},
        "targets_met": {
            "hot_single": hot.total_elapsed_ms <= 1000.0,
            "hot_batch_10": batch.total_elapsed_ms <= 10000.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "cold_ms": cold.total_elapsed_ms,
        "hot_ms": hot.total_elapsed_ms,
        "batch_10_ms": batch.total_elapsed_ms,
        "cache": after_batch,
        "targets_met": payload["targets_met"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0 if all(payload["targets_met"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
