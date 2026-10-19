from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_dict_safe(
    path: str | Path, *, expected_sha256: str | None = None,
) -> dict[str, Tensor]:
    checkpoint = Path(path).expanduser().resolve(strict=True)
    if expected_sha256 is not None:
        actual = sha256_file(checkpoint)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(f"Checkpoint SHA256 mismatch for {checkpoint}: {actual}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping: {checkpoint}")
    for key in ("best_model_state_dict", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not payload or not all(isinstance(key, str) and isinstance(value, Tensor)
                              for key, value in payload.items()):
        raise TypeError(f"Checkpoint does not contain a tensor state_dict: {checkpoint}")
    return dict(payload)


def load_checkpoint(
    model: nn.Module, path: str | Path, *, expected_sha256: str | None = None,
    strict: bool = True,
) -> nn.Module:
    model.load_state_dict(
        load_state_dict_safe(path, expected_sha256=expected_sha256), strict=strict
    )
    return model
