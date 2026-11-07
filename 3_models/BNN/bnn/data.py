from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def create_loaders(root: Path, batch_size: int, eval_batch_size: int, seed: int,
                   num_workers: int = 0, validation_size: int = 5000):
    # The published checkpoint was trained directly on MNIST pixels in [0, 1].
    transform = transforms.ToTensor()
    full = datasets.MNIST(root=root, train=True, transform=transform, download=False)
    train, validation = random_split(
        full, [len(full) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    test = datasets.MNIST(root=root, train=False, transform=transform, download=False)
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed), **common),
        DataLoader(validation, batch_size=eval_batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=eval_batch_size, shuffle=False, **common),
    )
