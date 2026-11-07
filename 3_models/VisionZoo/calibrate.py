from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vision_zoo.checkpoint import load_checkpoint
from vision_zoo.data import CIFAR10_MEAN, CIFAR10_STD, MNIST_MEAN, MNIST_STD
from vision_zoo.engine import calibrate_input_scales
from vision_zoo.models import FACTORIES, create_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate static VisionZoo input scales")
    parser.add_argument("model_id", choices=tuple(FACTORIES))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    is_cifar = args.model_id.endswith("cifar10")
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if is_cifar else (MNIST_MEAN, MNIST_STD)
    dataset_cls = datasets.CIFAR10 if is_cifar else datasets.MNIST
    dataset = dataset_cls(
        args.dataset_root, train=True,
        transform=transforms.Compose((transforms.ToTensor(), transforms.Normalize(mean, std))),
        download=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_checkpoint(create_model(args.model_id), args.checkpoint)
    scales = calibrate_input_scales(
        model, loader, device=args.device, max_batches=args.max_batches,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(scales, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {len(scales)} calibrated layer scales to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
