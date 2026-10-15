from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "5_NeuroCIM_opt"
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.layers import adapt_module


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float
    samples: int


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model: nn.Module, loader, *, device: str, max_batches: int | None = None) -> Evaluation:
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = samples = 0
    with torch.inference_mode():
        for index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss += float(criterion(logits, targets)) * targets.numel()
            correct += int((logits.argmax(1) == targets).sum())
            samples += targets.numel()
    if samples == 0:
        raise RuntimeError("No evaluation samples were processed")
    return Evaluation(loss / samples, correct / samples, samples)


def train_batches(
    model: nn.Module,
    loader,
    *,
    device: str,
    learning_rate: float = 0.1,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = samples = 0
    for index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * targets.numel()
        correct += int((logits.argmax(1) == targets).sum())
        samples += targets.numel()
    if samples == 0:
        raise RuntimeError("No training samples were processed")
    return {"loss": total_loss / samples, "accuracy": correct / samples, "samples": samples}


def train_epoch(
    model: nn.Module,
    loader,
    *,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.to(device).train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = samples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * targets.numel()
        correct += int((logits.argmax(1) == targets).sum())
        samples += targets.numel()
    if samples == 0:
        raise RuntimeError("No training samples were processed")
    return {"loss": total_loss / samples, "accuracy": correct / samples, "samples": samples}


def calibrate_input_scales(
    model: nn.Module,
    loader,
    *,
    device: str,
    max_batches: int = 1,
) -> dict[str, float]:
    maxima: dict[str, float] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            def capture(_module, inputs, layer_name=name):
                maximum = float(inputs[0].detach().abs().max())
                maxima[layer_name] = max(maxima.get(layer_name, 0.0), maximum)
            handles.append(module.register_forward_pre_hook(capture))
    try:
        model.to(device).eval()
        with torch.inference_mode():
            for index, (inputs, _targets) in enumerate(loader):
                if index >= max_batches:
                    break
                model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()
    expected = {name for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))}
    if missing := expected.difference(maxima):
        raise RuntimeError(f"Calibration did not execute layers: {sorted(missing)}")
    epsilon = np.finfo(np.float32).eps
    return {name: max(value, epsilon) / 255.0 for name, value in maxima.items()}


def differential_copy(
    model: nn.Module,
    lut: np.ndarray | torch.Tensor,
    input_scales: dict[str, float],
    *,
    ste: bool = False,
) -> nn.Module:
    converted = deepcopy(model)
    adapt_module(converted, lut=torch.as_tensor(lut), input_scales=input_scales, ste=ste)
    return converted
