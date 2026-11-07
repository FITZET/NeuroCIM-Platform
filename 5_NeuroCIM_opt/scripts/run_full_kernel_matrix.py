from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import subprocess
import sys
import traceback


PLATFORM = Path(__file__).resolve().parents[1]
PROJECT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

DEVICES = {
    "01_NMOS": {"file": "01_NMOS.csv", "output": "Ianalog"},
    "02_CFET": {"file": "02_CFET.csv", "output": "Iout"},
    "03_UOS_1": {"file": "03_UOS_1.csv", "output": "Iout"},
    "04_UOS_2": {"file": "04_UOS_2.csv", "output": "I"},
    "05_ideal_multiplication": {
        "file": "05_ideal_multiplication.csv", "output": "ideal_product_dec",
        "x": "input_code_dec", "w": "weight_code_dec",
    },
}
TRAINING_DEVICES = ("01_NMOS", "03_UOS_1")
VARIANTS = (
    ("empirical_interaction", "实测"),
    ("smooth_interaction_order1", "低阶"),
    ("smooth_interaction_order2", "高阶"),
)
NETWORKS = (
    ("bnn-mnist", "BNN", "MNIST"),
    ("lenet5-mnist", "LeNet-5", "MNIST"),
    ("resnet20-cifar10", "ResNet-20", "CIFAR-10"),
    ("tinytransformer-cifar10", "TinyTransformer", "CIFAR-10"),
)
METRICS = (
    ("r2_to_ideal_lut", "R2"),
    ("rmse_codes_to_ideal_lut", "RMSE"),
    ("mae_codes_to_ideal_lut", "MAE"),
    ("max_abs_codes_to_ideal_lut", "最大误差"),
    ("nl_percent_fs_to_ideal_lut", "非线性百分比FS"),
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fit_devices(root: Path) -> dict[str, dict[str, dict]]:
    from neurocim_opt.application import CellImportOptions, PlatformService
    from neurocim_opt.workflow import ErrorModelRegistry

    data = PROJECT / "0_data"
    registry = ErrorModelRegistry(root / "error_model_registry.json")
    collected: dict[str, dict[str, dict]] = {}
    for device, config in DEVICES.items():
        target = root / "fitting" / device
        summary = target / "fitting" / "fitting_summary.json"
        manifest = target / "error_model" / "error_model.json"
        if not summary.is_file() or not manifest.is_file():
            print(f"FIT {device}", flush=True)
            x = config.get("x", "VD_Digital")
            w = config.get("w", "VG_Digital")
            PlatformService().fit_cell(
                data / config["file"], target,
                CellImportOptions(
                    column_mapping={"X": x, "W": w, "output": config["output"]},
                    encoding="utf-8-sig", delimiter=",", missing_policy="error",
                    code_bits=8,
                    code_format="decimal" if device.startswith("05_") else "binary",
                    units={},
                ),
            )
        registered = registry.register_fitting_run(
            error_model_path=manifest, fitting_summary_path=summary,
        )
        variants = {
            item["method"]: item for item in registered["models"] if item.get("selectable")
        }
        collected[device] = variants
    return collected


def export_device_matrix(root: Path, fitted: dict[str, dict[str, dict]]) -> Path:
    fields = ["器件"] + [f"{label}_{title}" for _, label in VARIANTS for _, title in METRICS]
    rows = []
    for device in DEVICES:
        row = {"器件": device}
        for method, label in VARIANTS:
            metrics = fitted[device][method]["metrics"]
            for key, title in METRICS:
                row[f"{label}_{title}"] = metrics[key]
        rows.append(row)
    path = root / "device_error_model_matrix.csv"
    write_csv(path, fields, rows)
    return path


def worker(request_path: Path) -> int:
    from neurocim_opt.training_service import ErrorAwareTrainingService, TrainingRequest

    raw = json.loads(request_path.read_text(encoding="utf-8"))
    raw.update({key: Path(raw[key]) for key in (
        "project_root", "error_model_path", "output_dir",
    )})
    progress_path = raw["output_dir"] / "progress.jsonl"

    def progress(percent: int, message: str, metrics: dict | None) -> None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"percent": percent, "message": message, "metrics": metrics}) + "\n")
        if percent in {1, 10, 75, 78, 85, 92, 99, 100}:
            print(f"{percent}% {message}", flush=True)

    try:
        ErrorAwareTrainingService().run(TrainingRequest(**raw), progress=progress)
        return 0
    except Exception:
        raw["output_dir"].mkdir(parents=True, exist_ok=True)
        (raw["output_dir"] / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_training_jobs(root: Path, fitted: dict[str, dict[str, dict]], args) -> None:
    jobs = []
    for device in TRAINING_DEVICES:
        for method, _label in VARIANTS:
            for model_id, network, dataset in NETWORKS:
                jobs.append((device, method, model_id, network, dataset))
    for index, (device, method, model_id, network, dataset) in enumerate(jobs, 1):
        output = root / "training" / device / method / model_id
        result = output / "training_result.json"
        if result.is_file() and json.loads(result.read_text(encoding="utf-8")).get("status") == "completed":
            print(f"SKIP {index}/24 {device} {method} {network}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        request = {
            "project_root": str(PROJECT),
            "error_model_path": fitted[device][method]["path"],
            "output_dir": str(output),
            "error_strength": 1.0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "dataset_root": None,
            "learning_rate": args.learning_rate,
            "model_id": model_id,
            "checkpoint_path": None,
            "auxiliary_path": None,
            "seed": args.seed,
            "max_train_batches": None,
            "max_eval_batches": None,
        }
        request_path = output / "request.json"
        log_path = output / "console.log"
        print(f"RUN {index}/24 {device} {method} {network}", flush=True)
        while True:
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            failure_path = output / "failure.txt"
            failure_path.unlink(missing_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
                    cwd=PLATFORM, stdout=log, stderr=subprocess.STDOUT,
                )
            if completed.returncode == 0:
                print(
                    f"DONE {index}/24 {device} {method} {network} "
                    f"batch={request['batch_size']}", flush=True,
                )
                break
            failure = failure_path.read_text(encoding="utf-8") if failure_path.is_file() else ""
            if "out of memory" in failure.lower() and request["batch_size"] > 4:
                request["batch_size"] = max(4, request["batch_size"] // 2)
                print(
                    f"RETRY {device} {method} {network} after CUDA OOM: "
                    f"batch={request['batch_size']}", flush=True,
                )
                continue
            print(f"FAILED {device} {method} {network}; see {log_path}", flush=True)
            break
        export_training_matrices(root)


def export_training_matrices(root: Path) -> list[Path]:
    modes = {
        "error_aware_training_accuracy_matrix.csv": "device_differential",
        "ideal_differential_accuracy_matrix.csv": "ideal_differential",
        "float_accuracy_matrix.csv": "float",
    }
    common_fields = ["网络", "数据集"]
    result_paths = []
    for filename, mode in modes.items():
        fields = list(common_fields)
        for device in TRAINING_DEVICES:
            for _method, label in VARIANTS:
                fields.extend((f"{device}_{label}_训练前", f"{device}_{label}_训练后"))
        rows = []
        for model_id, network, dataset in NETWORKS:
            row = {"网络": network, "数据集": dataset}
            for device in TRAINING_DEVICES:
                for method, label in VARIANTS:
                    path = root / "training" / device / method / model_id / "training_result.json"
                    if not path.is_file():
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    row[f"{device}_{label}_训练前"] = payload["evaluations"]["before"][mode]["accuracy"]
                    row[f"{device}_{label}_训练后"] = payload["evaluations"]["after"][mode]["accuracy"]
            rows.append(row)
        path = root / filename
        write_csv(path, fields, rows)
        result_paths.append(path)

    fields = ["网络", "数据集"]
    for device in TRAINING_DEVICES:
        for _method, label in VARIANTS:
            fields.append(f"{device}_{label}_提升")
    rows = []
    for model_id, network, dataset in NETWORKS:
        row = {"网络": network, "数据集": dataset}
        for device in TRAINING_DEVICES:
            for method, label in VARIANTS:
                path = root / "training" / device / method / model_id / "training_result.json"
                if path.is_file():
                    p = json.loads(path.read_text(encoding="utf-8"))["evaluations"]
                    row[f"{device}_{label}_提升"] = (
                        p["after"]["device_differential"]["accuracy"]
                        - p["before"]["device_differential"]["accuracy"]
                    )
        rows.append(row)
    path = root / "accuracy_improvement_matrix.csv"
    write_csv(path, fields, rows)
    result_paths.append(path)
    return result_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLATFORM / "outputs" / "full_kernel_experiment")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker.resolve())
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    fitted = fit_devices(root)
    print(f"WROTE {export_device_matrix(root, fitted)}", flush=True)
    if args.fit_only:
        return 0
    run_training_jobs(root, fitted, args)
    for path in export_training_matrices(root):
        print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
