from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DatasetKind(str, Enum):
    DEVICE = "device"
    IDEAL = "ideal"


@dataclass(frozen=True)
class DeviceDataset:
    """Canonical in-memory representation used by all fitters."""

    source_path: Path
    name: str
    kind: DatasetKind
    frame: pd.DataFrame
    source_hash: str
    column_mapping: dict[str, str]
    acquisition_bits: int = 8
    code_max: int = 255
    units: dict[str, str] = field(default_factory=dict)
    import_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitMetrics:
    r2: float
    rmse: float
    mae: float
    max_abs_error: float
    full_scale: float
    nl_percent_fs: float

    def as_dict(self) -> dict[str, float]:
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "full_scale": self.full_scale,
            "nl_percent_fs": self.nl_percent_fs,
        }


@dataclass
class ModelFit:
    name: str
    equation: str
    coefficients: dict[str, float]
    metrics: FitMetrics
    predicted: np.ndarray
    residual: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FittingConfig:
    """Options suitable for construction from future GUI controls."""

    generate_plots: bool = True
    export_point_data: bool = True
    plot_dpi: int = 300
    image_format: str = "png"
    lut_min: int = 0
    lut_max: int = 65535
    style: str = "paper"
    column_mapping: dict[str, str] | None = None
    encoding: str = "auto"
    delimiter: str = "auto"
    missing_policy: str = "error"
    code_bits: int = 8
    code_format: str = "auto"
    units: dict[str, str] = field(default_factory=dict)


@dataclass
class FittingResult:
    dataset: DeviceDataset
    fits: dict[str, ModelFit]
    interaction_current: np.ndarray
    lut: np.ndarray
    lut_gain: float
    interaction_metrics: FitMetrics
    output_dir: Path | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.name,
            "kind": self.dataset.kind.value,
            "source": str(self.dataset.source_path),
            "sha256": self.dataset.source_hash,
            "acquisition": {
                "code_bits": self.dataset.acquisition_bits,
                "code_range": [0, self.dataset.code_max],
                "grid_semantics": "native acquisition grid",
                "units": self.dataset.units,
                **self.dataset.import_metadata,
            },
            "column_mapping": self.dataset.column_mapping,
            "lut_gain": self.lut_gain,
            "interaction_metrics": self.interaction_metrics.as_dict(),
            "models": {
                key: {
                    "equation": fit.equation,
                    "coefficients": fit.coefficients,
                    "metrics": fit.metrics.as_dict(),
                    "metadata": fit.metadata,
                }
                for key, fit in self.fits.items()
            },
            "artifacts": {key: str(path) for key, path in self.artifacts.items()},
        }
