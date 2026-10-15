from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PLATFORM_ROOT = PROJECT_ROOT / "5_NeuroCIM_opt"
if str(PLATFORM_ROOT) not in sys.path: sys.path.insert(0, str(PLATFORM_ROOT))
from neurocim_opt.paths import data_root

from bnn.data import create_loaders
from bnn.differential import collect_input_scales, differential_models
from bnn.engine import evaluate, load_checkpoint, seed_everything, train
from bnn.model import BinaryMNIST


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Formal binary-weight/binary-activation MNIST experiment")
    p.add_argument("command", choices=("train", "evaluate", "differential"))
    p.add_argument("--dataset-root", type=Path,
                   default=Path(os.environ.get("NEUROCIM_MNIST_ROOT", data_root("mnist"))))
    p.add_argument("--output-dir", type=Path, default=HERE / "output" / "formal")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=15); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--error-model", type=Path,
                   default=PLATFORM_ROOT / "outputs/error_models/01_NMOS/error_model.json")
    p.add_argument("--differential-samples", type=int, default=4)
    return p


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parser().parse_args(); seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, validation_loader, test_loader = create_loaders(
        args.dataset_root, args.batch_size, args.eval_batch_size, args.seed, args.num_workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = BinaryMNIST()
    checkpoint = args.checkpoint or args.output_dir / "best.pt"
    config = {"dataset_root": str(args.dataset_root.resolve()), "epochs": args.epochs,
              "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
              "learning_rate": args.lr, "seed": args.seed, "device": device,
              "definition": {"weights": "sign(w) for every Conv2d/Linear",
                             "activations": "sign(BN(x)) for every hidden stage; real first input and logits",
                             "ste": "identity gradient for |x| <= 1, zero otherwise"}}
    started = time.perf_counter()
    if args.command == "train":
        resume = args.output_dir / "latest.pt" if args.resume else None
        history = train(model, train_loader, validation_loader, device=device, epochs=args.epochs,
                        learning_rate=args.lr, output_dir=args.output_dir, config=config,
                        resume=resume if resume and resume.is_file() else None)
        load_checkpoint(args.output_dir / "best.pt", model, device=device)
        metrics = evaluate(model, test_loader, device)
        result = {"status": "usable" if metrics.accuracy >= .97 else "target_not_met",
                  "accuracy_target": .97, "history": history, "test": asdict(metrics)}
    elif args.command == "evaluate":
        loaded = load_checkpoint(checkpoint, model, device=device); metrics = evaluate(model, test_loader, device)
        config = loaded["config"]
        result = {"status": "usable" if metrics.accuracy >= .97 else "target_not_met",
                  "accuracy_target": .97, "history": loaded["history"],
                  "completed_epochs": loaded["epoch"],
                  "best_validation_accuracy": loaded["best_validation_accuracy"],
                  "test": asdict(metrics)}
    else:
        loaded = load_checkpoint(checkpoint, model, device=device); config = loaded["config"]
        frozen = model.frozen().eval()
        native = evaluate(frozen, test_loader, device, args.differential_samples)
        scales = collect_input_scales(frozen, validation_loader, device)
        ideal, hardware = differential_models(frozen.cpu(), scales, args.error_model)
        ideal_metrics = evaluate(ideal, test_loader, device, args.differential_samples)
        hardware_metrics = evaluate(hardware, test_loader, device, args.differential_samples)
        result = {"status": "bounded_mechanism_validation",
                  "warning": "Fixed leading samples only; these values are not full-test-set statistics.",
                  "sample_count": args.differential_samples, "native": asdict(native),
                  "ideal_differential": asdict(ideal_metrics), "device_01_NMOS": asdict(hardware_metrics),
                  "input_scales": scales, "error_model": str(args.error_model.resolve())}
    evaluation_config = {"command": args.command, "eval_batch_size": args.eval_batch_size,
                         "differential_samples": args.differential_samples if args.command == "differential" else None,
                         "device": device, "seed": args.seed}
    result.update({"created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                   "evaluation_config": evaluation_config,
                   "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
                   "elapsed_seconds": time.perf_counter() - started})
    target = args.output_dir / ("differential_evaluation.json" if args.command == "differential" else "evaluation.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False)); print(f"saved: {target}")


if __name__ == "__main__": main()
