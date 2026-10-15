from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PLATFORM_ROOT = Path(__file__).resolve().parents[2] / "5_NeuroCIM_opt"
if str(PLATFORM_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PLATFORM_ROOT))
from neurocim_opt.error_model import HardwareErrorModel
from neurocim_opt.paths import data_root, output_root, require_path

from resnet20.data import create_cifar10_loaders
from resnet20.engine import (
    calibrate_input_scales, differential_copy, evaluate, set_deterministic_seed,
    train_batches, train_epoch,
)
from resnet20.model import resnet20


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _checkpoint_payload(model, optimizer, scheduler, *, epoch, best_accuracy, history, args):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_accuracy": best_accuracy,
        "history": history,
        "seed": args.seed,
        "smoke": False,
        "configuration": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "momentum": args.momentum,
            "weight_decay": args.weight_decay, "scheduler": "cosine_annealing",
        },
    }


def _differential_evidence(model, test_loader, scales, error_manifest: Path, device: str) -> dict:
    codes = np.arange(256, dtype=np.int64)
    ideal_lut = codes[:, None] * codes[None, :]
    hardware = HardwareErrorModel.load(error_manifest)
    results, predictions = {}, {}
    fixed_loader = DataLoader(Subset(test_loader.dataset, list(range(8))), batch_size=1, shuffle=False)
    for name, lut in (("ideal", ideal_lut), ("01_NMOS", hardware.effective_lut())):
        converted = differential_copy(model, lut, scales).to(device)
        loss, correct, count = 0.0, 0, 0
        target_values, predicted_values = [], []
        with torch.inference_mode():
            for inputs, targets in fixed_loader:
                logits = converted(inputs.to(device))
                predicted = logits.argmax(1).cpu()
                loss += float(torch.nn.functional.cross_entropy(logits, targets.to(device)))
                correct += int((predicted == targets).sum())
                count += targets.numel()
                target_values.extend(targets.tolist())
                predicted_values.extend(predicted.tolist())
        results[name] = {"loss": loss / count, "accuracy": correct / count, "samples": count}
        predictions[name] = {"targets": target_values, "predictions": predicted_values}
        del converted
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    ideal_predictions = predictions["ideal"]["predictions"]
    device_predictions = predictions["01_NMOS"]["predictions"]
    return {
        "method": "exact signed positive/negative differential LUT MAC",
        "fixed_test_indices": list(range(8)), "fixed_samples": results["ideal"]["samples"],
        "results": results,
        "prediction_agreement": sum(a == b for a, b in zip(ideal_predictions, device_predictions))
        / len(ideal_predictions),
        "predictions": predictions,
        "error_model_manifest": str(error_manifest.resolve()),
    }


def run_smoke(args, output_dir: Path) -> dict:
    train, validation, _ = create_cifar10_loaders(
        args.dataset_root, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        seed=args.seed, num_workers=args.num_workers,
    )
    model = resnet20()
    before = evaluate(model, validation, device=args.device, max_batches=args.eval_batches)
    trained = train_batches(
        model, train, device=args.device, max_batches=args.train_batches, learning_rate=0.01
    )
    after = evaluate(model, validation, device=args.device, max_batches=args.eval_batches)
    scales = calibrate_input_scales(model, validation, device=args.device, max_batches=1)
    checkpoint = output_dir / "resnet20_smoke.pt"
    torch.save({"model_state_dict": model.state_dict(), "seed": args.seed, "smoke": True}, checkpoint)
    _write_json(output_dir / "input_scales.json", scales)
    return {
        "kind": "smoke_only", "smoke": True, "device": args.device,
        "before": asdict(before), "training": trained, "after": asdict(after),
        "calibrated_layers": len(scales), "checkpoint": str(checkpoint.resolve()),
    }


def run_formal(args, output_dir: Path) -> dict:
    train, validation, test = create_cifar10_loaders(
        args.dataset_root, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        seed=args.seed, num_workers=args.num_workers,
    )
    model = resnet20().to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    latest_path, best_path = output_dir / "latest.pt", output_dir / "best.pt"
    history: list[dict] = []
    start_epoch, best_accuracy = 0, -1.0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"])
        best_accuracy = float(state["best_validation_accuracy"])
        history = list(state["history"])

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        training = train_epoch(model, train, device=args.device, optimizer=optimizer)
        validation_result = evaluate(model, validation, device=args.device)
        scheduler.step()
        row = {
            "epoch": epoch + 1, "learning_rate": learning_rate,
            "training": training, "validation": asdict(validation_result),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        improved = validation_result.accuracy > best_accuracy
        if improved:
            best_accuracy = validation_result.accuracy
        payload = _checkpoint_payload(
            model, optimizer, scheduler, epoch=epoch + 1,
            best_accuracy=best_accuracy, history=history, args=args,
        )
        torch.save(payload, latest_path)
        if improved:
            torch.save(payload, best_path)
        _write_json(output_dir / "history.json", history)
        print(json.dumps(row), flush=True)

    best = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    full_test = evaluate(model, test, device=args.device)
    scales = calibrate_input_scales(
        model, validation, device=args.device, max_batches=args.calibration_batches
    )
    _write_json(output_dir / "input_scales.json", scales)
    differential = _differential_evidence(model, test, scales, args.error_model, args.device)
    _write_json(output_dir / "differential_evaluation.json", differential)
    return {
        "kind": "formal_training", "smoke": False, "device": args.device,
        "seed": args.seed, "cudnn_deterministic": True,
        "epochs_requested": args.epochs, "epochs_completed": len(history),
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
        "optimizer": "SGD(momentum, nesterov, weight_decay)",
        "scheduler": "CosineAnnealingLR", "best_validation_accuracy": best_accuracy,
        "full_test_10000": asdict(full_test), "calibrated_layers": len(scales),
        "differential": differential, "best_checkpoint": str(best_path.resolve()),
        "latest_checkpoint": str(latest_path.resolve()),
        "history": str((output_dir / "history.json").resolve()),
        "training_seconds": sum(float(row["seconds"]) for row in history),
        "final_evaluation_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible ResNet-20 CIFAR-10 training")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--dataset-root", type=Path, default=data_root("cifar10"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument(
        "--error-model", type=Path,
        default=output_root() / "error_models/01_NMOS/error_model.json",
    )
    parser.add_argument("--train-batches", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    require_path(args.dataset_root, "CIFAR-10 dataset root")
    if args.mode == "formal":
        require_path(args.error_model, "01_NMOS error model")
    set_deterministic_seed(args.seed)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "output" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    report = run_formal(args, output_dir) if args.mode == "formal" else run_smoke(args, output_dir)
    report["started_utc"] = started.isoformat()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
