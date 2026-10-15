from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from .model import LeNet5
from .quantization import LayerQuantizationParameters

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "5_NeuroCIM_opt"
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.layers import adapt_module


def symmetric_input_scales(
    parameters: dict[str, LayerQuantizationParameters],
) -> dict[str, float]:
    """Convert saved affine ranges to signed-magnitude uint8 scales."""

    result: dict[str, float] = {}
    for name, item in parameters.items():
        minimum = -item.input_zero_point * item.input_scale
        maximum = (255 - item.input_zero_point) * item.input_scale
        result[name] = max(abs(minimum), abs(maximum)) / 255.0
    return result


def configure_differential_model(
    model: LeNet5,
    parameters: dict[str, LayerQuantizationParameters],
    lut: np.ndarray | torch.Tensor,
    *,
    ste: bool = False,
) -> LeNet5:
    """Replace every LeNet MAC layer with a signed differential LUT layer."""

    adapt_module(
        model,
        lut=torch.as_tensor(lut),
        input_scales=symmetric_input_scales(parameters),
        ste=ste,
    )
    return model


def train_error_aware(
    model: LeNet5,
    train_loader,
    validation_loader,
    *,
    device: str | torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    optimizer_state: dict | None = None,
    start_epoch: int = 0,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict]:
    """Fine-tune with exact LUT forward values and floating STE gradients."""

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        correct = samples = 0
        for batch_index, (inputs, targets) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            if not any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and bool(torch.any(parameter.grad != 0))
                for parameter in model.parameters()
            ):
                raise RuntimeError("No finite nonzero parameter gradient in error-aware training")
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            correct += int((logits.argmax(1) == targets).sum())
            samples += targets.numel()
        if samples == 0:
            raise RuntimeError("No training samples were processed")

        model.eval()
        validation_correct = validation_samples = 0
        validation_loss = 0.0
        with torch.inference_mode():
            for batch_index, (inputs, targets) in enumerate(validation_loader):
                if max_validation_batches is not None and batch_index >= max_validation_batches:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                validation_loss += float(criterion(logits, targets)) * targets.numel()
                validation_correct += int((logits.argmax(1) == targets).sum())
                validation_samples += targets.numel()
        if validation_samples == 0:
            raise RuntimeError("No validation samples were processed")
        validation_accuracy = validation_correct / validation_samples
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / samples,
                "train_accuracy": correct / samples,
                "validation_loss": validation_loss / validation_samples,
                "validation_accuracy": validation_accuracy,
                "train_samples": samples,
                "validation_samples": validation_samples,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = deepcopy(model.state_dict())
    return history, best_state, optimizer.state_dict()
