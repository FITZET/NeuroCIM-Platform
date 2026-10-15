from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "3_models" / "LeNet_5"))

from neurocim_opt import HardwareErrorModel, MacErrorModel, builtin_registry
from neurocim_opt.paths import data_root, neurosim_binary, output_root, require_path, toolchain_root
from neurocim_opt.neurosim_adapter import (
    NeuroSimConfig,
    export_differential_traces,
    run_neurosim,
)
from lenet5.config import ExperimentConfig
from lenet5.data import create_mnist_train_val_test_loaders
from lenet5.differential import (
    configure_differential_model,
    symmetric_input_scales,
    train_error_aware,
)
from lenet5.engine import evaluate, load_checkpoint
from lenet5.model import InferenceMode, LeNet5
from lenet5.quantization import load_quantization_parameters


def _limited(loader: DataLoader, samples: int) -> DataLoader:
    count = min(samples, len(loader.dataset))
    return DataLoader(
        Subset(loader.dataset, range(count)),
        batch_size=min(loader.batch_size or count, count),
        shuffle=False,
        num_workers=0,
    )


def _column_mapping(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        logical, separator, source = item.partition("=")
        if not separator or not logical.strip() or not source.strip():
            raise SystemExit(f"Invalid --map {item!r}; expected logical=source_column")
        mapping[logical.strip()] = source.strip()
    return mapping


def _load_model(args, lut, *, ste: bool) -> LeNet5:
    model = LeNet5()
    load_checkpoint(model, args.checkpoint)
    parameters = load_quantization_parameters(args.quant_params)
    return configure_differential_model(model, parameters, lut, ste=ste)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Device-error differential CIM workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-models")
    build = subparsers.add_parser("build-error-model")
    build.add_argument("input", type=Path)
    build.add_argument("--input-kind", choices=("raw", "lut"), required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    mac = subparsers.add_parser("fit-mac-model")
    mac.add_argument("input", type=Path)
    mac.add_argument("--map", action="append", default=[], metavar="LOGICAL=COLUMN")
    mac.add_argument("--encoding")
    mac.add_argument("--delimiter", choices=("comma", "semicolon"))
    mac.add_argument("--missing", choices=("error", "drop", "interpolate"), default="error")
    mac.add_argument("--degree", type=int, choices=(1, 2, 3), default=2)
    mac.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser("smoke-lenet")
    smoke.add_argument("--error-model", type=Path, required=True)
    smoke.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "3_models/LeNet_5/chekpoint/lenet5_float.pt")
    smoke.add_argument("--quant-params", type=Path, default=PROJECT_ROOT / "3_models/LeNet_5/chekpoint/quantization.json")
    smoke.add_argument("--dataset-root", type=Path, default=data_root("mnist"))
    smoke.add_argument("--samples", type=int, default=32)
    smoke.add_argument("--lambda", dest="error_strength", type=float, default=1.0)
    smoke.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    smoke.add_argument("--output", type=Path, default=output_root() / "smoke/report.json")
    smoke.add_argument("--download", action="store_true")

    train = subparsers.add_parser("train-eat")
    for action in smoke._actions[1:]:
        if action.dest in {"samples", "output"}:
            continue
        kwargs = {"default": action.default, "required": action.required, "help": action.help}
        if action.type:
            kwargs["type"] = action.type
        if action.option_strings == ["--download"]:
            train.add_argument("--download", action="store_true")
        elif action.option_strings:
            train.add_argument(*action.option_strings, dest=action.dest, **kwargs)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--max-train-batches", type=int)
    train.add_argument("--max-validation-batches", type=int)
    train.add_argument("--eval-samples", type=int, default=32)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--training-lut", choices=("device", "ideal"), default="device",
        help="Use device LUT for EAT or ideal LUT for the equal-budget QAT control",
    )
    train.add_argument("--output-dir", type=Path, default=output_root() / "eat")

    ns = subparsers.add_parser("neurosim-eval")
    ns.add_argument("--error-model", type=Path, required=True)
    ns.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "3_models/LeNet_5/chekpoint/lenet5_float.pt")
    ns.add_argument("--eat-checkpoint", type=Path, default=BASE_DIR / "outputs/eat/01_NMOS/eat_checkpoint.pt")
    ns.add_argument("--quant-params", type=Path, default=PROJECT_ROOT / "3_models/LeNet_5/chekpoint/quantization.json")
    ns.add_argument("--dataset-root", type=Path, default=data_root("mnist"))
    ns.add_argument("--lambda", dest="error_strength", type=float, default=1.0)
    ns.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ns.add_argument("--subarray", type=int, default=128)
    ns.add_argument("--parallel-rows", type=int, default=128)
    ns.add_argument("--output-dir", type=Path, default=output_root() / "neurosim/01_NMOS")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list-models":
        registry = builtin_registry(PROJECT_ROOT)
        print(json.dumps([
            {
                "model_id": item.model_id,
                "display_name": item.display_name,
                "task": item.task,
                "datasets": item.datasets,
                "input_shape": item.input_shape,
                "metrics": item.metrics,
                "checkpoint": str(item.checkpoint) if item.checkpoint else None,
                "checkpoint_available": item.checkpoint_available,
                "license": item.license,
            }
            for item in registry.list()
        ], indent=2, ensure_ascii=False))
        return 0
    if args.command == "build-error-model":
        model = (
            HardwareErrorModel.from_raw_csv(args.input)
            if args.input_kind == "raw"
            else HardwareErrorModel.from_lut_csv(args.input)
        )
        print(model.save(args.output_dir))
        return 0

    if args.command == "fit-mac-model":
        model = MacErrorModel.from_csv(
            args.input,
            column_mapping=_column_mapping(args.map) or None,
            encoding=args.encoding,
            delimiter={"comma": ",", "semicolon": ";"}.get(args.delimiter),
            missing=args.missing,
            polynomial_degree=args.degree,
        )
        path = model.save(args.output_dir)
        print(json.dumps({
            "model": str(path),
            "groups": [
                {
                    "key": item.key,
                    "samples": item.samples,
                    "r2": item.linear_metrics.r2,
                    "nonlinearity_percent_fs": item.linear_metrics.nonlinearity_percent_fs,
                }
                for item in model.groups
            ],
        }, indent=2, ensure_ascii=False))
        return 0

    if not 0 <= args.error_strength <= 1:
        raise SystemExit("--lambda must be in [0, 1]")
    error_model = HardwareErrorModel.load(args.error_model)
    device_lut = error_model.effective_lut(args.error_strength)
    if not getattr(args, "download", False):
        require_path(args.dataset_root, "MNIST dataset root")
    config = ExperimentConfig(dataset_root=args.dataset_root, device=args.device)
    train_loader, validation_loader, test_loader = create_mnist_train_val_test_loaders(
        config, download=getattr(args, "download", False)
    )

    if args.command == "neurosim-eval":
        parameters = load_quantization_parameters(args.quant_params)
        model = _load_model(args, device_lut, ste=False)
        checkpoint = torch.load(args.eat_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("best_model_state_dict", checkpoint["model_state_dict"])
        model.load_state_dict(state)
        model.to(args.device)
        sample, _target = next(iter(test_loader))
        sample = sample[:1]
        traces = args.output_dir / "traces"
        vendor = BASE_DIR / "vendor/neurosim"
        ns_config = NeuroSimConfig(
            subarray_rows=args.subarray,
            parallel_rows=args.parallel_rows,
        )
        network, pairs = export_differential_traces(
            model,
            sample,
            traces,
            input_scales=symmetric_input_scales(parameters),
            vendor_root=vendor,
            pooling_after={"conv1", "conv2"},
            config=ns_config,
        )
        result = run_neurosim(
            neurosim_binary(),
            network,
            pairs,
            args.output_dir,
            config=ns_config,
            conda_prefix=toolchain_root(),
        )
        print(json.dumps(asdict(result), indent=2))
        return 0

    if args.command == "smoke-lenet":
        if args.samples <= 0:
            raise SystemExit("--samples must be positive")
        test_loader = _limited(test_loader, args.samples)
        started = time.perf_counter()
        float_model = LeNet5()
        load_checkpoint(float_model, args.checkpoint)
        float_result = evaluate(float_model, test_loader, device=args.device, mode=InferenceMode.FLOAT)
        ideal_model = _load_model(args, error_model.effective_lut(0), ste=False)
        ideal_result = evaluate(ideal_model, test_loader, device=args.device, mode=InferenceMode.QUANTIZED)
        device_model = _load_model(args, device_lut, ste=False)
        device_result = evaluate(
            device_model, test_loader, device=args.device, mode=InferenceMode.FAULT,
            error_strength=args.error_strength,
        )
        report = {
            "kind": "smoke",
            "samples": args.samples,
            "elapsed_seconds": time.perf_counter() - started,
            "error_model": str(args.error_model.resolve()),
            "lambda": args.error_strength,
            "results": {
                "float": asdict(float_result),
                "ideal_differential": asdict(ideal_result),
                "device_differential": asdict(device_result),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    training_lut = device_lut if args.training_lut == "device" else error_model.effective_lut(0)
    model = _load_model(args, training_lut, ste=True)
    prior_history: list[dict] = []
    optimizer_state = None
    start_epoch = 0
    if args.resume:
        resume = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(resume["model_state_dict"])
        prior_history = resume.get("metadata", {}).get("history", [])
        start_epoch = len(prior_history)
        optimizer_state = resume.get("optimizer_state_dict")
    history, best_state, optimizer_state = train_error_aware(
        model, train_loader, validation_loader,
        device=args.device, epochs=args.epochs, learning_rate=args.learning_rate,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        optimizer_state=optimizer_state, start_epoch=start_epoch,
    )
    history = prior_history + history
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_state = model.state_dict()
    torch.save(
        {
            "model_state_dict": current_state,
            "best_model_state_dict": best_state,
            "optimizer_state_dict": optimizer_state,
            "metadata": {
                "training": (
                    "error-aware exact device-LUT forward with float STE backward"
                    if args.training_lut == "device"
                    else "ideal-differential QAT control with float STE backward"
                ),
                "training_lut": args.training_lut,
                "lambda": args.error_strength,
                "error_model": str(args.error_model.resolve()),
                "history": history,
            },
        },
        args.output_dir / "eat_checkpoint.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    evaluation = None
    if args.eval_samples > 0:
        evaluation_loader = _limited(test_loader, args.eval_samples)
        device_model = _load_model(args, device_lut, ste=False)
        device_model.load_state_dict(best_state)
        device_result = evaluate(
            device_model, evaluation_loader, device=args.device, mode=InferenceMode.FAULT,
            error_strength=args.error_strength,
        )
        ideal_model = _load_model(args, error_model.effective_lut(0), ste=False)
        ideal_model.load_state_dict(best_state)
        ideal_result = evaluate(
            ideal_model, evaluation_loader, device=args.device, mode=InferenceMode.QUANTIZED
        )
        evaluation = {
            "samples": args.eval_samples,
            "trained_device_differential": asdict(device_result),
            "trained_ideal_differential": asdict(ideal_result),
        }
        (args.output_dir / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
    print(json.dumps({"history": history, "evaluation": evaluation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
