from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import ExperimentConfig


def mnist_transform() -> transforms.Compose:
    # Original LeNet-5 consumes 32x32 inputs. Padding keeps MNIST content intact.
    return transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def create_mnist_loaders(
    config: ExperimentConfig, *, download: bool = False
) -> tuple[DataLoader, DataLoader]:
    root = config.mnist_root
    train_dataset = datasets.MNIST(root=root, train=True, transform=mnist_transform(), download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=mnist_transform(), download=download)
    common = {
        "num_workers": config.num_workers,
        "pin_memory": config.device.startswith("cuda"),
    }
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **common)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False, **common)
    return train_loader, test_loader


def create_mnist_train_val_test_loaders(
    config: ExperimentConfig,
    *,
    download: bool = False,
    validation_size: int = 5000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create a deterministic train/validation/test split for model selection."""

    full_train = datasets.MNIST(
        root=config.mnist_root, train=True, transform=mnist_transform(), download=download
    )
    if not 0 < validation_size < len(full_train):
        raise ValueError("validation_size must be between 1 and the training set size")
    train_dataset, validation_dataset = random_split(
        full_train,
        [len(full_train) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_dataset = datasets.MNIST(
        root=config.mnist_root, train=False, transform=mnist_transform(), download=download
    )
    common = {
        "num_workers": config.num_workers,
        "pin_memory": config.device.startswith("cuda"),
    }
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed), **common,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.eval_batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.eval_batch_size, shuffle=False, **common
    )
    return train_loader, validation_loader, test_loader
