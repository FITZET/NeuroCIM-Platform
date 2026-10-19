from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from vision_zoo import FACTORIES, create_model
from vision_zoo.checkpoint import load_checkpoint, sha256_file
from vision_zoo.data import create_loaders


ROOT = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = samples = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss += float(criterion(logits, targets)) * targets.numel()
            correct += int(logits.argmax(1).eq(targets).sum())
            samples += targets.numel()
    return {"loss": loss / samples, "accuracy": correct / samples, "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible VisionZoo training")
    parser.add_argument("model_id", choices=tuple(FACTORIES))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-sha256")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 30 if args.model_id == "resnet8-mnist" else 200
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    model = create_model(args.model_id)
    if args.initial_checkpoint:
        load_checkpoint(model, args.initial_checkpoint, expected_sha256=args.initial_sha256)
    model.to(device)
    train_loader, validation_loader, test_loader = create_loaders(
        args.model_id, args.dataset_root, batch_size=args.batch_size,
        eval_batch_size=max(args.batch_size, 256), seed=args.seed,
        num_workers=0, download=args.download,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss = 0.0
        correct = samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            batch_loss = criterion(logits, targets)
            batch_loss.backward()
            optimizer.step()
            loss += float(batch_loss.detach()) * targets.numel()
            correct += int(logits.argmax(1).eq(targets).sum())
            samples += targets.numel()
        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss / samples, "train_accuracy": correct / samples,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = deepcopy(model.state_dict())
        scheduler.step()

    assert best_state is not None
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    target = args.output_dir / args.model_id
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "best_state_dict.pt"
    temporary = checkpoint.with_suffix(".pt.part")
    torch.save(best_state, temporary)
    temporary.replace(checkpoint)
    report = {
        "schema_version": 1, "model_id": args.model_id, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": "SGD", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "momentum": 0.9, "nesterov": True,
        "scheduler": "CosineAnnealingLR",
        "dataset_root": str(args.dataset_root.resolve()),
        "selection": "best validation accuracy; test evaluated once after selection",
        "best_validation_accuracy": best_accuracy, "test": test,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "elapsed_seconds": time.time() - started, "history": history,
    }
    (target / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
