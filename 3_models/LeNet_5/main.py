from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from lenet5.config import ExperimentConfig
from lenet5.data import create_mnist_loaders
from lenet5.engine import evaluate, load_checkpoint, save_checkpoint, train_float
from lenet5.model import InferenceMode, LeNet5
from lenet5.quantization import (
    calibrate,
    configure_lut_inference,
    load_fitting_lut,
    load_quantization_parameters,
    save_quantization_parameters,
)
from lenet5.reporting import save_comparison, save_evaluation, save_history
from neurocim_opt.paths import require_path

BASE_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST LeNet-5 CIM experiment backend")
    parser.add_argument("command", choices=("train", "float-eval", "calibrate", "quant-eval", "fault-eval", "run-all"))
    parser.add_argument("--checkpoint", type=Path, default=BASE_DIR / "chekpoint" / "lenet5_float.pt")
    parser.add_argument("--quant-params", type=Path, default=BASE_DIR / "chekpoint" / "quantization.json")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output")
    parser.add_argument("--lut", type=Path, action="append", help="Repeat for multiple device LUTs in run-all")
    parser.add_argument(
        "--lambda", dest="error_strength", type=float, default=1.0,
        help="Residual device-error strength in [0, 1] for fault inference",
    )
    parser.add_argument("--dataset-root", type=Path, default=ExperimentConfig().dataset_root)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.error_strength <= 1.0:
        raise SystemExit("--lambda must be in [0, 1]")
    if not args.download:
        require_path(args.dataset_root, "MNIST dataset root")
    config = ExperimentConfig(dataset_root=args.dataset_root, epochs=args.epochs, device=args.device)
    torch.manual_seed(config.seed)
    train_loader, test_loader = create_mnist_loaders(config, download=args.download)
    model = LeNet5()

    if args.command in ("train", "run-all"):
        history = train_float(
            model, train_loader, device=config.device, epochs=config.epochs,
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            progress=lambda percent, message: print(f"[{percent:3d}%] {message}"),
        )
        save_checkpoint(model, args.checkpoint, config=config.as_dict(), history=history)
        save_history(history, args.output_dir)
        if args.command == "train":
            print(json.dumps(history[-1], indent=2))
            return 0

    if args.command != "run-all":
        load_checkpoint(model, args.checkpoint)
    if args.command == "float-eval":
        result = evaluate(model, test_loader, device=config.device, mode=InferenceMode.FLOAT)
    elif args.command == "calibrate":
        parameters = calibrate(
            model, train_loader, device=config.device, max_batches=config.calibration_batches
        )
        args.quant_params.parent.mkdir(parents=True, exist_ok=True)
        save_quantization_parameters(parameters, args.quant_params)
        print(f"Saved calibration parameters to {args.quant_params.resolve()}")
        return 0
    elif args.command == "run-all":
        results = {"float": evaluate(model, test_loader, device=config.device, mode=InferenceMode.FLOAT)}
        parameters = calibrate(model, train_loader, device=config.device, max_batches=config.calibration_batches)
        args.quant_params.parent.mkdir(parents=True, exist_ok=True)
        save_quantization_parameters(parameters, args.quant_params)
        configure_lut_inference(model, parameters)
        results["quantized"] = evaluate(model, test_loader, device=config.device, mode=InferenceMode.QUANTIZED)
        for lut_path in args.lut or []:
            configure_lut_inference(
                model, parameters, lut=load_fitting_lut(lut_path),
                error_strength=args.error_strength,
            )
            label = f"fault_{lut_path.parent.name}"
            results[label] = evaluate(
                model, test_loader, device=config.device, mode=InferenceMode.FAULT,
                error_strength=args.error_strength,
            )
        for name, item in results.items():
            save_evaluation(item, args.output_dir, name)
        save_comparison(results, args.output_dir)
        print(json.dumps({name: asdict(item) | {"confusion_matrix": "saved"} for name, item in results.items()}, indent=2))
        return 0
    else:
        parameters = load_quantization_parameters(args.quant_params)
        if args.command == "quant-eval":
            configure_lut_inference(model, parameters)  # ideal X*W LUT
            mode = InferenceMode.QUANTIZED
        else:
            if not args.lut:
                raise SystemExit("--lut is required for fault-eval")
            configure_lut_inference(
                model, parameters, lut=load_fitting_lut(args.lut[0]),
                error_strength=args.error_strength,
            )
            mode = InferenceMode.FAULT
        result = evaluate(
            model, test_loader, device=config.device, mode=mode,
            error_strength=args.error_strength if mode is InferenceMode.FAULT else None,
        )
    save_evaluation(result, args.output_dir, args.command.replace("-eval", ""))
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
