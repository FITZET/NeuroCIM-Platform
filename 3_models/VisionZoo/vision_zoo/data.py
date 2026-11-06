from __future__ import annotations

from copy import copy
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# Pinned chenyaofo training/evaluation configuration for the bundled
# MobileNetV2 and VGG11-BN checkpoints.
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)


def _split_indices(length: int, validation_size: int, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < validation_size < length:
        raise ValueError("validation_size must be between 1 and the training-set size")
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def create_mnist_loaders(
    root: str | Path, *, batch_size: int = 128, eval_batch_size: int = 256,
    validation_size: int = 5_000, seed: int = 42, num_workers: int = 0,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)))
    root = Path(root).expanduser().resolve()
    full = datasets.MNIST(root, train=True, transform=transform, download=download)
    test = datasets.MNIST(root, train=False, transform=transform, download=download)
    training, validation = _split_indices(len(full), validation_size, seed)
    common = {"num_workers": num_workers, "worker_init_fn": _seed_worker}
    return (
        DataLoader(Subset(full, training), batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed), **common),
        DataLoader(Subset(full, validation), batch_size=eval_batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=eval_batch_size, shuffle=False, **common),
    )


def create_cifar10_loaders(
    root: str | Path, *, batch_size: int = 128, eval_batch_size: int = 256,
    validation_size: int = 5_000, seed: int = 42, num_workers: int = 0,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(root).expanduser().resolve()
    training_transform = transforms.Compose((
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ))
    evaluation_transform = transforms.Compose((
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ))
    augmented = datasets.CIFAR10(root, train=True, transform=training_transform, download=download)
    # Keep two transform views without allocating a second full CIFAR array.
    evaluation = copy(augmented)
    evaluation.transform = evaluation_transform
    test = datasets.CIFAR10(root, train=False, transform=evaluation_transform, download=False)
    training, validation = _split_indices(len(augmented), validation_size, seed)
    common = {"num_workers": num_workers, "worker_init_fn": _seed_worker}
    return (
        DataLoader(Subset(augmented, training), batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed), **common),
        DataLoader(Subset(evaluation, validation), batch_size=eval_batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=eval_batch_size, shuffle=False, **common),
    )


def create_loaders(model_id: str, root: str | Path, **kwargs):
    if model_id == "resnet8-mnist":
        return create_mnist_loaders(root, **kwargs)
    if model_id in {"mobilenetv2-cifar10", "vgg11-bn-cifar10", "squeezenet1_1-cifar10"}:
        return create_cifar10_loaders(root, **kwargs)
    raise KeyError(f"Unknown VisionZoo model: {model_id}")
