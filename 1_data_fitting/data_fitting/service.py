from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .fitting import (
    build_interaction_lut,
    fit_code_voltage,
    fit_full_surface,
    fit_linear_product,
    fit_raw_product_output,
    fit_quadratic_surface,
)
from .loader import inspect_csv_columns, load_dataset
from .models import DeviceDataset, FittingConfig, FittingResult
from .plotting import generate_all_plots

ProgressCallback = Callable[[int, str], None]


class DataFittingService:
    """Stateless facade intended for direct use by a GUI worker thread."""

    def load(self, path: str | Path, *, config: FittingConfig | None = None) -> DeviceDataset:
        return load_dataset(path, config=config)

    def inspect_columns(
        self, path: str | Path, *, encoding: str = "auto", delimiter: str = "auto"
    ) -> tuple[tuple[str, ...], str, str]:
        return inspect_csv_columns(path, encoding=encoding, delimiter=delimiter)

    def fit(
        self,
        dataset_or_path: DeviceDataset | str | Path,
        *,
        output_dir: str | Path | None = None,
        config: FittingConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> FittingResult:
        config = config or FittingConfig()
        notify = progress or (lambda _percent, _message: None)
        notify(5, "Loading and validating data")
        dataset = dataset_or_path if isinstance(dataset_or_path, DeviceDataset) else self.load(dataset_or_path, config=config)
        notify(20, "Fitting code-voltage calibration")
        fits = fit_code_voltage(dataset)
        notify(35, "Fitting linear product model")
        fits["linear_product"] = fit_linear_product(dataset)
        fits["raw_product_output"] = fit_raw_product_output(dataset)
        notify(50, "Fitting quadratic response surface")
        fits["quadratic_surface"] = fit_quadratic_surface(dataset)
        notify(65, "Fitting complete transfer surface")
        fits["full_surface"] = fit_full_surface(dataset)
        notify(75, "Extracting interaction current and generating LUT")
        interaction, lut, gain, interaction_metrics = build_interaction_lut(
            dataset, config.lut_min, config.lut_max
        )
        result = FittingResult(dataset, fits, interaction, lut, gain, interaction_metrics)
        if output_dir is not None:
            notify(82, "Exporting fitting results")
            self.export(result, output_dir, config)
        notify(100, "Fitting completed")
        return result

    def export(self, result: FittingResult, output_dir: str | Path, config: FittingConfig | None = None) -> FittingResult:
        config = config or FittingConfig()
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        result.output_dir = target

        lut_path = target / "hardware_multiplication_lut.csv"
        pd.DataFrame(result.lut, index=range(256), columns=range(256)).rename_axis("W").rename_axis("X", axis=1).to_csv(lut_path)
        interaction_path = target / "interaction_current.csv"
        pd.DataFrame(result.interaction_current, index=range(256), columns=range(256)).rename_axis("W").rename_axis("X", axis=1).to_csv(interaction_path)
        result.artifacts.update({"lut": lut_path, "interaction_current": interaction_path})

        if config.export_point_data:
            points = result.dataset.frame.copy()
            for key, fit in result.fits.items():
                points[f"{key}_predicted"] = fit.predicted
                points[f"{key}_residual"] = fit.residual
            points_path = target / "fitting_points.csv"
            points.to_csv(points_path, index=False)
            result.artifacts["point_data"] = points_path

        if config.generate_plots:
            result.artifacts.update(generate_all_plots(result, target, config))

        summary_path = target / "fitting_summary.json"
        result.artifacts["summary"] = summary_path
        summary_path.write_text(json.dumps(result.summary(), indent=2, ensure_ascii=False), encoding="utf-8")
        return result
