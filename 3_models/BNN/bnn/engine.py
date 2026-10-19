from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import BinaryMNIST


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    """Capture process-global RNGs used by this training backend."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Any) -> None:
    """Restore a current RNG bundle; accept legacy torch-only checkpoints."""
    if isinstance(state, torch.Tensor):
        torch.set_rng_state(state)
        return
    if not isinstance(state, dict):
        raise ValueError("checkpoint RNG state is missing or invalid")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])


@dataclass
class Metrics:
    loss: float
    accuracy: float
    samples: int


def evaluate(model: nn.Module, loader, device: str, max_samples: int | None = None) -> Metrics:
    model.to(device).eval(); criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0; correct = samples = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            if max_samples is not None:
                remaining = max_samples - samples
                if remaining <= 0: break
                inputs, targets = inputs[:remaining], targets[:remaining]
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs); loss += float(criterion(logits, targets))
            correct += int((logits.argmax(1) == targets).sum()); samples += targets.numel()
    return Metrics(loss / samples, correct / samples, samples)


def checkpoint_payload(model, optimizer, scheduler, history, epoch, best_accuracy, config,
                       train_loader_generator: torch.Generator | None = None) -> dict[str, Any]:
    return {"format_version": 2, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "history": history, "epoch": epoch, "best_validation_accuracy": best_accuracy,
            "config": config, "rng_state": capture_rng_state(),
            "train_loader_generator_state": (
                train_loader_generator.get_state() if train_loader_generator is not None else None)}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None, device="cpu",
                    train_loader_generator: torch.Generator | None = None) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    # Error-aware training checkpoints retain both the last epoch and the
    # validation-selected best weights. Inference and resumed model state must
    # use the same best-weight convention as the other supported networks.
    state = payload.get("best_model_state_dict", payload["model_state_dict"])
    model.load_state_dict(state)
    if optimizer is not None: optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None: scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_rng_state(payload["rng_state"])
    generator_state = payload.get("train_loader_generator_state")
    if train_loader_generator is not None and generator_state is not None:
        train_loader_generator.set_state(generator_state.cpu())
    return payload


def train(model: BinaryMNIST, train_loader, validation_loader, *, device: str, epochs: int,
          learning_rate: float, output_dir: Path, config: dict, resume: Path | None = None):
    model.to(device); criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict] = []; start_epoch = 0; best_accuracy = -1.0
    if resume:
        payload = load_checkpoint(
            resume, model, optimizer, scheduler, device,
            train_loader_generator=getattr(train_loader, "generator", None),
        )
        history = payload["history"]; start_epoch = payload["epoch"]; best_accuracy = payload["best_validation_accuracy"]
    best_state = None
    for epoch in range(start_epoch, epochs):
        model.train(); total_loss = correct = samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(inputs)
            loss = criterion(logits, targets); loss.backward(); optimizer.step(); model.clamp_latent_weights_()
            total_loss += float(loss.detach()) * targets.numel()
            correct += int((logits.argmax(1) == targets).sum()); samples += targets.numel()
        validation = evaluate(model, validation_loader, device)
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": total_loss / samples,
               "train_accuracy": correct / samples, "validation_loss": validation.loss,
               "validation_accuracy": validation.accuracy,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(row, flush=True)
        payload = checkpoint_payload(model, optimizer, scheduler, history, epoch + 1,
                                     max(best_accuracy, validation.accuracy), config,
                                     getattr(train_loader, "generator", None))
        save_checkpoint(output_dir / "latest.pt", payload)
        if validation.accuracy > best_accuracy:
            best_accuracy = validation.accuracy; best_state = deepcopy(model.state_dict())
            save_checkpoint(output_dir / "best.pt", payload)
    if best_state is not None: model.load_state_dict(best_state)
    elif (output_dir / "best.pt").is_file(): load_checkpoint(output_dir / "best.pt", model, device=device)
    return history
