from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .model import InferenceMode, LeNet5

ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class EvaluationResult:
    mode: str
    loss: float
    accuracy: float
    correct: int
    samples: int
    per_class_accuracy: list[float]
    confusion_matrix: list[list[int]]
    error_strength: float | None = None


def train_float(
    model: LeNet5,
    train_loader,
    *,
    device: torch.device | str,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    progress: ProgressCallback | None = None,
) -> list[dict[str, float]]:
    notify = progress or (lambda _percent, _message: None)
    model.set_inference_mode(InferenceMode.FLOAT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * targets.numel()
            correct += int((logits.argmax(dim=1) == targets).sum())
            samples += targets.numel()
        record = {"epoch": epoch + 1, "loss": running_loss / samples, "accuracy": correct / samples}
        history.append(record)
        notify(round(100 * (epoch + 1) / epochs), f"Epoch {epoch + 1}/{epochs}")
    return history


@torch.inference_mode()
def evaluate(
    model: LeNet5,
    data_loader,
    *,
    device: torch.device | str,
    mode: InferenceMode | str,
    error_strength: float | None = None,
) -> EvaluationResult:
    selected = InferenceMode(mode)
    model.set_inference_mode(selected).eval().to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = samples = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += float(criterion(logits, targets))
        predictions = logits.argmax(dim=1)
        correct += int((predictions == targets).sum())
        indices = (targets * 10 + predictions).detach().cpu()
        confusion += torch.bincount(indices, minlength=100).reshape(10, 10)
        samples += targets.numel()
    class_totals = confusion.sum(dim=1)
    per_class = confusion.diag().float() / class_totals.clamp_min(1)
    return EvaluationResult(
        selected.value,
        total_loss / samples,
        correct / samples,
        correct,
        samples,
        per_class.tolist(),
        confusion.tolist(),
        error_strength,
    )


def save_checkpoint(model: LeNet5, path: str | Path, **metadata) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, target)


def load_checkpoint(model: LeNet5, path: str | Path, *, map_location="cpu") -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("metadata", {})
