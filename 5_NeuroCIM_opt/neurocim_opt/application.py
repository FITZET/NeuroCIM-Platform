from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import pandas as pd
import numpy as np

from .mac_error_model import MacErrorModel, load_mac_csv
from .paths import data_root, neurosim_binary, toolchain_root


def _device_mode_name(error_model: Path) -> str:
    try:
        payload = json.loads(error_model.read_text(encoding="utf-8"))
        dataset = payload.get("metadata", {}).get("fitting_summary", {}).get("dataset")
        return f"device_{dataset}" if dataset else "device_differential"
    except (OSError, json.JSONDecodeError):
        return "device_differential"


@dataclass(frozen=True)
class MacImportOptions:
    column_mapping: dict[str, str]
    encoding: str | None = None
    delimiter: str | None = None
    missing: str = "error"
    polynomial_degree: int = 2


@dataclass(frozen=True)
class DataPreview:
    rows: int
    columns: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MacFitOutcome:
    model_path: Path
    summary: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CellImportOptions:
    column_mapping: dict[str, str]
    encoding: str = "auto"
    delimiter: str = "auto"
    missing_policy: str = "error"
    code_bits: int = 8
    code_format: str = "auto"
    units: dict[str, str] | None = None


@dataclass(frozen=True)
class CellFitOutcome:
    error_model_path: Path
    summary_path: Path
    plot_paths: tuple[Path, ...]
    metrics: dict[str, float]
    models: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InferencePrediction:
    mode: str
    predicted_label: int
    confidence: float
    elapsed_ms: float
    top5: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class SingleInferenceOutcome:
    sample_index: int
    true_label: int
    device: str
    preparation_ms: float
    total_elapsed_ms: float
    predictions: tuple[InferencePrediction, ...]
    input_image: tuple[float, ...] = ()
    input_shape: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "true_label": self.true_label,
            "device": self.device,
            "preparation_ms": self.preparation_ms,
            "total_elapsed_ms": self.total_elapsed_ms,
            "predictions": [
                {
                    "mode": item.mode,
                    "predicted_label": item.predicted_label,
                    "confidence": item.confidence,
                    "elapsed_ms": item.elapsed_ms,
                    "top5": [
                        {"label": label, "confidence": confidence}
                        for label, confidence in item.top5
                    ],
                }
                for item in self.predictions
            ],
            "input_image": list(self.input_image),
            "input_shape": list(self.input_shape),
        }


@dataclass(frozen=True)
class BatchInferencePrediction:
    mode: str
    predicted_labels: tuple[int, ...]
    confidences: tuple[float, ...]
    elapsed_ms: float
    top5_labels: tuple[tuple[int, ...], ...] = ()
    top5_confidences: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class BatchInferenceOutcome:
    sample_indices: tuple[int, ...]
    true_labels: tuple[int, ...]
    device: str
    cache_hit: bool
    preparation_ms: float
    total_elapsed_ms: float
    predictions: tuple[BatchInferencePrediction, ...]
    input_images: tuple[tuple[float, ...], ...] = ()
    input_shapes: tuple[tuple[int, ...], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PpaOutcome:
    model_id: str
    checkpoint_path: str
    error_model_path: str
    result_path: str
    log_path: str
    trace_directory: str
    trace_input: str
    metrics: dict[str, float]
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlatformService:
    """UI-neutral application use cases shared by CLI and desktop front ends."""

    def preview_mac(self, path: str | Path, options: MacImportOptions, limit: int = 20) -> DataPreview:
        frame, metadata = load_mac_csv(
            path,
            column_mapping=options.column_mapping or None,
            encoding=options.encoding,
            delimiter=options.delimiter,
            missing=options.missing,
        )
        records = tuple(frame.head(limit).to_dict(orient="records"))
        return DataPreview(len(frame), tuple(frame.columns), records, metadata)

    def fit_mac(
        self, path: str | Path, output_dir: str | Path, options: MacImportOptions
    ) -> MacFitOutcome:
        model = MacErrorModel.from_csv(
            path,
            column_mapping=options.column_mapping or None,
            encoding=options.encoding,
            delimiter=options.delimiter,
            missing=options.missing,
            polynomial_degree=options.polynomial_degree,
        )
        model_path = model.save(output_dir)
        summary = tuple(
            {
                "array": f"{item.array_rows}x{item.array_cols}",
                "bit_width": item.bit_width,
                "group": item.group,
                "samples": item.samples,
                "r2": item.linear_metrics.r2,
                "nonlinearity_percent_fs": item.linear_metrics.nonlinearity_percent_fs,
                "rmse": item.linear_metrics.rmse,
                "polynomial_r2": item.polynomial_metrics.r2,
            }
            for item in model.groups
        )
        return MacFitOutcome(model_path, summary)

    @staticmethod
    def _fitting_api() -> tuple[Any, Any]:
        fitting_root = Path(__file__).resolve().parents[2] / "1_data_fitting"
        if str(fitting_root) not in sys.path:
            sys.path.insert(0, str(fitting_root))
        from data_fitting import DataFittingService, FittingConfig
        return DataFittingService, FittingConfig

    def preview_cell(self, path: str | Path, options: CellImportOptions, limit: int = 20) -> DataPreview:
        DataFittingService, FittingConfig = self._fitting_api()
        dataset = DataFittingService().load(
            path,
            config=FittingConfig(
                column_mapping=options.column_mapping or None,
                encoding=options.encoding,
                delimiter=options.delimiter,
                missing_policy=options.missing_policy,
                code_bits=options.code_bits,
                code_format=options.code_format,
                units=options.units or {},
            ),
        )
        return DataPreview(
            rows=len(dataset.frame),
            columns=tuple(dataset.frame.columns),
            records=tuple(dataset.frame.head(limit).to_dict(orient="records")),
            metadata={
                **dataset.import_metadata,
                "code_bits": dataset.acquisition_bits,
                "code_range": [0, dataset.code_max],
                "units": dataset.units,
                "column_mapping": dataset.column_mapping,
            },
        )

    def inspect_cell_columns(
        self, path: str | Path, *, encoding: str = "auto", delimiter: str = "auto"
    ) -> tuple[tuple[str, ...], str, str]:
        DataFittingService, _FittingConfig = self._fitting_api()
        return DataFittingService().inspect_columns(
            path, encoding=encoding, delimiter=delimiter
        )

    def fit_cell(
        self,
        path: str | Path,
        output_dir: str | Path,
        options: CellImportOptions,
        progress: Any | None = None,
    ) -> CellFitOutcome:
        DataFittingService, FittingConfig = self._fitting_api()
        from .error_model import HardwareErrorModel

        config = FittingConfig(
            generate_plots=True,
            export_point_data=True,
            column_mapping=options.column_mapping or None,
            encoding=options.encoding,
            delimiter=options.delimiter,
            missing_policy=options.missing_policy,
            code_bits=options.code_bits,
            code_format=options.code_format,
            units=options.units or {},
        )
        target = Path(output_dir).expanduser().resolve()
        result = DataFittingService().fit(path, output_dir=target / "fitting", config=config, progress=progress)
        model = HardwareErrorModel.from_lut_csv(
            result.artifacts["lut"], fitting_summary=result.artifacts["summary"]
        )
        error_model_path = model.save(target / "error_model")
        models = tuple(
            {
                "model": name,
                "equation": fit.equation,
                "r2": fit.metrics.r2,
                "rmse": fit.metrics.rmse,
                "nl_percent_fs": fit.metrics.nl_percent_fs,
            }
            for name, fit in result.fits.items()
        )
        plot_paths = tuple(
            path for key, path in result.artifacts.items()
            if key not in {"lut", "interaction_current", "point_data", "summary"}
        )
        return CellFitOutcome(
            error_model_path=error_model_path,
            summary_path=result.artifacts["summary"],
            plot_paths=plot_paths,
            metrics=result.interaction_metrics.as_dict(),
            models=models,
            metadata=result.summary()["acquisition"],
        )

    @staticmethod
    def infer_lenet5_mnist_batch(
        project_root: str | Path,
        sample_indices: list[int] | tuple[int, ...],
        *,
        dataset_root: str | Path | None = None,
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
        quantization_path: str | Path | None = None,
        error_model_path: str | Path | None = None,
        adc_bits: int | None = None,
        noise_sigma: float = 0.0,
        noise_rho: float = 0.0,
        noise_seed: int = 0,
    ) -> BatchInferenceOutcome:
        """Run a real MNIST batch through float, ideal and device-LUT paths."""

        total_started = perf_counter()
        root = Path(project_root).expanduser().resolve()
        model_root = root / "3_models" / "LeNet_5"
        platform_root = root / "5_NeuroCIM_opt"
        checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve() if checkpoint_path
            else model_root / "chekpoint" / "lenet5_float.pt"
        )
        quantization_path = (
            Path(quantization_path).expanduser().resolve() if quantization_path
            else model_root / "chekpoint" / "quantization.json"
        )
        error_model_path = (
            Path(error_model_path).expanduser().resolve() if error_model_path
            else platform_root / "outputs" / "error_models" / "01_NMOS" / "error_model.json"
        )
        import torch
        from .inference_runtime import get_lenet_inference_resources

        data_path = (
            Path(dataset_root).expanduser().resolve()
            if dataset_root is not None
            else data_root("mnist")
        )
        indices = tuple(int(index) for index in sample_indices)
        if not indices:
            raise ValueError("sample_indices must contain at least one index")
        if any(index < 0 or index >= 10_000 for index in indices):
            raise IndexError("MNIST test sample indices must be in [0, 9999]")
        device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resources, cache_hit = get_lenet_inference_resources(
                model_root=model_root,
                dataset_root=data_path,
                checkpoint_path=checkpoint_path,
                quantization_path=quantization_path,
                error_model_path=error_model_path,
                device_name=device_name,
            )
        except FileNotFoundError:
            required = (checkpoint_path, quantization_path, error_model_path)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Required inference resources are missing:\n" + "\n".join(missing)
                ) from None
            raise
        samples = [resources.dataset[index] for index in indices]
        inputs = torch.stack([sample[0] for sample in samples]).to(resources.device)
        true_labels = tuple(int(sample[1]) for sample in samples)
        execution_device = resources.device
        if execution_device.type == "cuda":
            torch.cuda.synchronize()
        preparation_ms = (perf_counter() - total_started) * 1_000.0

        from .layers import configure_model_nonidealities
        configure_model_nonidealities(resources.ideal_model, adc_bits=adc_bits, noise_seed=noise_seed)
        configure_model_nonidealities(
            resources.device_model, adc_bits=adc_bits, noise_sigma=noise_sigma,
            noise_rho=noise_rho, noise_seed=noise_seed,
        )
        predictions: list[BatchInferencePrediction] = []
        for mode, model in (
            ("float", resources.float_model),
            ("ideal_differential", resources.ideal_model),
            (_device_mode_name(error_model_path), resources.device_model),
        ):
            if execution_device.type == "cuda":
                torch.cuda.synchronize()
            started = perf_counter()
            with torch.inference_mode():
                logits = model(inputs)
                probabilities = torch.softmax(logits, dim=1)
            if execution_device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (perf_counter() - started) * 1_000.0
            confidence, predicted = probabilities.max(dim=1)
            top_count = min(5, int(probabilities.shape[1]))
            top_confidence, top_labels = probabilities.topk(top_count, dim=1)
            predictions.append(
                BatchInferencePrediction(
                    mode=mode,
                    predicted_labels=tuple(int(value) for value in predicted.cpu().tolist()),
                    confidences=tuple(float(value) for value in confidence.cpu().tolist()),
                    elapsed_ms=elapsed_ms,
                    top5_labels=tuple(tuple(int(v) for v in row) for row in top_labels.cpu().tolist()),
                    top5_confidences=tuple(tuple(float(v) for v in row) for row in top_confidence.cpu().tolist()),
                )
            )
        return BatchInferenceOutcome(
            sample_indices=indices,
            true_labels=true_labels,
            device=str(execution_device),
            cache_hit=cache_hit,
            preparation_ms=preparation_ms,
            total_elapsed_ms=(perf_counter() - total_started) * 1_000.0,
            predictions=tuple(predictions),
            input_images=tuple(tuple(float(v) for v in sample[0].detach().cpu().flatten().tolist()) for sample in samples),
            input_shapes=tuple(tuple(int(v) for v in sample[0].shape) for sample in samples),
        )

    @staticmethod
    def infer_lenet5_mnist(
        project_root: str | Path,
        sample_index: int,
        *,
        dataset_root: str | Path | None = None,
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
        quantization_path: str | Path | None = None,
        error_model_path: str | Path | None = None,
        adc_bits: int | None = None,
        noise_sigma: float = 0.0,
        noise_rho: float = 0.0,
        noise_seed: int = 0,
    ) -> SingleInferenceOutcome:
        """Run one real sample; cached calls avoid rebuilding inference resources."""
        batch = PlatformService.infer_lenet5_mnist_batch(
            project_root, (sample_index,), dataset_root=dataset_root, device=device,
            checkpoint_path=checkpoint_path, quantization_path=quantization_path,
            error_model_path=error_model_path,
            adc_bits=adc_bits, noise_sigma=noise_sigma, noise_rho=noise_rho,
            noise_seed=noise_seed,
        )
        return SingleInferenceOutcome(
            sample_index=sample_index,
            true_label=batch.true_labels[0],
            device=batch.device,
            preparation_ms=batch.preparation_ms,
            total_elapsed_ms=batch.total_elapsed_ms,
            predictions=tuple(
                InferencePrediction(
                    mode=item.mode,
                    predicted_label=item.predicted_labels[0],
                    confidence=item.confidences[0],
                    elapsed_ms=item.elapsed_ms,
                    top5=tuple(zip(item.top5_labels[0], item.top5_confidences[0])),
                )
                for item in batch.predictions
            ),
            input_image=batch.input_images[0],
            input_shape=batch.input_shapes[0],
        )

    @staticmethod
    def infer_registered_single(
        project_root: str | Path,
        model_id: str,
        sample_index: int,
        *,
        dataset_root: str | Path,
        checkpoint_path: str | Path,
        error_model_path: str | Path,
        auxiliary_path: str | Path | None = None,
        device: str | None = None,
        adc_bits: int | None = None,
        noise_sigma: float = 0.0,
        noise_rho: float = 0.0,
        noise_seed: int = 0,
    ) -> SingleInferenceOutcome:
        """Run native, ideal-differential, and device-differential inference."""

        import torch

        root = Path(project_root).expanduser().resolve()
        dataset = Path(dataset_root).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        error_model = Path(error_model_path).expanduser().resolve()
        execution_device = device or "auto"
        if execution_device == "auto":
            execution_device = "cuda" if torch.cuda.is_available() else "cpu"
        if execution_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is unavailable")

        if model_id == "lenet5-mnist":
            if auxiliary_path is None:
                raise FileNotFoundError("LeNet-5 quantization parameters were not selected")
            return PlatformService.infer_lenet5_mnist(
                root, sample_index, dataset_root=dataset, device=execution_device,
                checkpoint_path=checkpoint, quantization_path=auxiliary_path,
                error_model_path=error_model,
                adc_bits=adc_bits, noise_sigma=noise_sigma, noise_rho=noise_rho,
                noise_seed=noise_seed,
            )

        for path, purpose in ((checkpoint, "checkpoint"), (error_model, "device error model")):
            if not path.is_file():
                raise FileNotFoundError(f"Required {purpose} is missing: {path}")
        started = perf_counter()

        def evaluate_modes(sample, target: int, modes, display_sample=None) -> SingleInferenceOutcome:
            preparation_ms = (perf_counter() - started) * 1_000.0
            inputs = sample.unsqueeze(0).to(execution_device)
            predictions: list[InferencePrediction] = []
            for name, model in modes:
                from .layers import configure_model_nonidealities
                if name == "ideal_differential":
                    configure_model_nonidealities(model, adc_bits=adc_bits, noise_seed=noise_seed)
                elif name not in {"float", "native_bnn"}:
                    configure_model_nonidealities(
                        model, adc_bits=adc_bits, noise_sigma=noise_sigma,
                        noise_rho=noise_rho, noise_seed=noise_seed,
                    )
                model.eval().to(execution_device)
                if execution_device == "cuda":
                    torch.cuda.synchronize()
                mode_started = perf_counter()
                with torch.inference_mode():
                    probabilities = torch.softmax(model(inputs), dim=1)
                if execution_device == "cuda":
                    torch.cuda.synchronize()
                confidence, predicted = probabilities.max(dim=1)
                top_count = min(5, int(probabilities.shape[1]))
                top_confidence, top_labels = probabilities.topk(top_count, dim=1)
                predictions.append(InferencePrediction(
                    name, int(predicted.item()), float(confidence.item()),
                    (perf_counter() - mode_started) * 1_000.0,
                    tuple(zip(
                        (int(value) for value in top_labels[0].cpu().tolist()),
                        (float(value) for value in top_confidence[0].cpu().tolist()),
                    )),
                ))
            shown = display_sample if display_sample is not None else sample
            return SingleInferenceOutcome(
                sample_index, int(target), execution_device, preparation_ms,
                (perf_counter() - started) * 1_000.0, tuple(predictions),
                tuple(float(value) for value in shown.detach().cpu().flatten().tolist()),
                tuple(int(value) for value in shown.shape),
            )

        if model_id == "bnn-mnist":
            model_root = root / "3_models" / "BNN"
            from .inference_runtime import get_bnn_inference_resources

            resources, _cache_hit = get_bnn_inference_resources(
                model_root=model_root, dataset_root=dataset,
                checkpoint_path=checkpoint, error_model_path=error_model,
                device_name=execution_device,
            )
            if not 0 <= sample_index < len(resources.dataset):
                raise IndexError(f"MNIST test sample index must be in [0, {len(resources.dataset)-1}]")
            sample, target = resources.dataset[sample_index]
            return evaluate_modes(sample, target, (
                ("native_bnn", resources.float_model),
                ("ideal_differential", resources.ideal_model),
                (_device_mode_name(error_model), resources.device_model),
            ))

        if model_id in {"resnet20-cifar10", "tinytransformer-cifar10"}:
            auxiliary = Path(auxiliary_path).expanduser().resolve() if auxiliary_path else None
            if auxiliary is None or not auxiliary.is_file():
                raise FileNotFoundError(f"{model_id} input-scales JSON was not selected or is missing")
            is_tiny = model_id == "tinytransformer-cifar10"
            model_root = root / "3_models" / ("TinyTransformer" if is_tiny else "Resnet20")
            from .inference_runtime import (
                get_resnet20_inference_resources, get_tinytransformer_inference_resources,
            )

            resource_loader = (
                get_tinytransformer_inference_resources if is_tiny
                else get_resnet20_inference_resources
            )
            resources, _cache_hit = resource_loader(
                model_root=model_root, dataset_root=dataset,
                checkpoint_path=checkpoint, input_scales_path=auxiliary,
                error_model_path=error_model, device_name=execution_device,
            )
            if not 0 <= sample_index < len(resources.dataset):
                raise IndexError(f"CIFAR-10 test sample index must be in [0, {len(resources.dataset)-1}]")
            sample, target = resources.dataset[sample_index]
            mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=sample.dtype).view(3, 1, 1)
            std = torch.tensor((0.2470, 0.2435, 0.2616), dtype=sample.dtype).view(3, 1, 1)
            display_sample = (sample.cpu() * std + mean).clamp(0.0, 1.0)
            return evaluate_modes(sample, target, (
                ("float", resources.float_model),
                ("ideal_differential", resources.ideal_model),
                (_device_mode_name(error_model), resources.device_model),
            ), display_sample)

        from .inference_runtime import VISION_ZOO_MODEL_IDS
        if model_id in VISION_ZOO_MODEL_IDS:
            auxiliary = Path(auxiliary_path).expanduser().resolve() if auxiliary_path else None
            if auxiliary is not None and not auxiliary.is_file():
                raise FileNotFoundError(f"{model_id} input-scales JSON is missing: {auxiliary}")
            from .inference_runtime import get_registered_inference_resources

            resources, _cache_hit = get_registered_inference_resources(
                model_id=model_id, model_root=root / "3_models/VisionZoo",
                dataset_root=dataset, checkpoint_path=checkpoint,
                auxiliary_path=auxiliary, error_model_path=error_model,
                device_name=execution_device,
            )
            dataset_name = "CIFAR-10" if model_id.endswith("cifar10") else "MNIST"
            if not 0 <= sample_index < len(resources.dataset):
                raise IndexError(
                    f"{dataset_name} test sample index must be in "
                    f"[0, {len(resources.dataset)-1}]"
                )
            sample, target = resources.dataset[sample_index]
            if model_id.endswith("cifar10"):
                mean_values = (0.4914, 0.4822, 0.4465)
                std_values = (0.2023, 0.1994, 0.2010)
            else:
                mean_values, std_values = (0.1307,), (0.3081,)
            channels = len(mean_values)
            mean = torch.tensor(mean_values, dtype=sample.dtype).view(channels, 1, 1)
            std = torch.tensor(std_values, dtype=sample.dtype).view(channels, 1, 1)
            display_sample = (sample.cpu() * std + mean).clamp(0.0, 1.0)
            return evaluate_modes(sample, target, (
                ("float", resources.float_model),
                ("ideal_differential", resources.ideal_model),
                (_device_mode_name(error_model), resources.device_model),
            ), display_sample)

        raise ValueError(f"Unsupported registered model for single-sample inference: {model_id}")

    @staticmethod
    def evaluate_lenet5_ppa(
        project_root: str | Path,
        checkpoint_path: str | Path,
        error_model_path: str | Path,
        output_dir: str | Path,
        *,
        subarray_rows: int = 128,
        parallel_rows: int = 128,
    ) -> PpaOutcome:
        """Export a real signed-differential LeNet-5 trace and run official NeuroSim."""

        root = Path(project_root).expanduser().resolve()
        platform_root = root / "5_NeuroCIM_opt"
        model_root = root / "3_models" / "LeNet_5"
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        error_path = Path(error_model_path).expanduser().resolve()
        quantization_path = model_root / "chekpoint" / "quantization.json"
        vendor_root = platform_root / "vendor" / "neurosim"
        binary = neurosim_binary()
        required = (checkpoint, error_path, quantization_path, binary)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required NeuroSim resources are missing:\n" + "\n".join(missing))

        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        import torch
        from lenet5.differential import symmetric_input_scales
        from lenet5.model import LeNet5
        from lenet5.quantization import load_quantization_parameters
        from .error_model import HardwareErrorModel
        from .neurosim_adapter import NeuroSimConfig, export_differential_traces, run_neurosim

        # Loading the error model is deliberate validation: PPA uses the same
        # four-path signed mapping as error-aware inference, while NeuroSim
        # models the physical arrays rather than numerical LUT perturbations.
        HardwareErrorModel.load(error_path)
        model = LeNet5()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("best_model_state_dict", payload.get("model_state_dict", payload))
        model.load_state_dict(state)
        parameters = load_quantization_parameters(quantization_path)
        config = NeuroSimConfig(subarray_rows=subarray_rows, parallel_rows=parallel_rows)
        config.validate()
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        sample = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        network, pairs = export_differential_traces(
            model,
            sample,
            target / "traces",
            input_scales=symmetric_input_scales(parameters),
            vendor_root=vendor_root,
            pooling_after={"conv1", "conv2"},
            config=config,
            network_name="lenet5_differential",
        )
        prefix = toolchain_root()
        result = run_neurosim(
            binary,
            network,
            pairs,
            target,
            config=config,
            conda_prefix=prefix if prefix.is_dir() else None,
        )
        outcome = PpaOutcome(
            model_id="lenet5-mnist",
            checkpoint_path=str(checkpoint),
            error_model_path=str(error_path),
            result_path=str(target / "gui_ppa_result.json"),
            log_path=result.stdout_path,
            trace_directory=result.trace_directory,
            trace_input="deterministic zero tensor, shape [1, 1, 32, 32]",
            metrics=result.metrics,
            config=result.config,
        )
        Path(outcome.result_path).write_text(
            json.dumps(outcome.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return outcome

    @staticmethod
    def export_table(records: list[dict[str, Any]], path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(records)
        if target.suffix.lower() == ".csv":
            frame.to_csv(target, index=False)
        elif target.suffix.lower() == ".xlsx":
            frame.to_excel(target, index=False)
        else:
            raise ValueError("Export path must end with .csv or .xlsx")
        return target
