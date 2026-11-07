from __future__ import annotations

from copy import copy
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _transform(training: bool) -> transforms.Compose:
    operations = []
    if training:
        operations.extend((transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()))
    operations.extend((transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)))
    return transforms.Compose(operations)


def _seed_worker(worker_id: int) -> None:
    random.seed(torch.initial_seed() % (2**32))


def create_cifar10_loaders(
    root: str | Path, *, batch_size: int = 128, eval_batch_size: int = 256,
    validation_size: int = 5000, seed: int = 42, num_workers: int = 0,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(root).expanduser().resolve()
    augmented = datasets.CIFAR10(root=root, train=True, transform=_transform(True), download=download)
    # Share the 146 MiB raw training array; only the transform differs.
    evaluation = copy(augmented)
    evaluation.transform = _transform(False)
    indices = torch.randperm(len(augmented), generator=torch.Generator().manual_seed(seed)).tolist()
    validation_indices, training_indices = indices[:validation_size], indices[validation_size:]
    test = datasets.CIFAR10(root=root, train=False, transform=_transform(False), download=False)
    return (
        DataLoader(Subset(augmented, training_indices), batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed), num_workers=num_workers,
                   worker_init_fn=_seed_worker, pin_memory=torch.cuda.is_available(),
                   persistent_workers=num_workers > 0),
        DataLoader(Subset(evaluation, validation_indices), batch_size=eval_batch_size,
                   shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                   persistent_workers=num_workers > 0),
        DataLoader(test, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0),
    )
