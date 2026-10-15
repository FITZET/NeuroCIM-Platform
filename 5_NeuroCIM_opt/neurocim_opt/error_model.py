from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_lut(lut: np.ndarray) -> np.ndarray:
    table = np.asarray(lut)
    if table.shape != (256, 256):
        raise ValueError(f"LUT must have shape (256, 256), got {table.shape}")
    if not np.issubdtype(table.dtype, np.number) or not np.isfinite(table).all():
        raise ValueError("LUT must contain only finite numeric values")
    rounded = np.rint(table)
    if not np.allclose(table, rounded):
        raise ValueError("LUT entries must be integer-valued multiplication codes")
    result = rounded.astype(np.int64)
    if result.min() < 0 or result.max() > 65535:
        raise ValueError("LUT entries must be in uint16 range [0, 65535]")
    # The fitting backend has already removed row/column baselines. Requiring zero
    # axes prevents a raw-current surface from being compensated a second time by
    # signed branch arithmetic.
    if np.any(result[0, :] != 0) or np.any(result[:, 0] != 0):
        raise ValueError(
            "LUT is not an interaction-product table: W=0 and X=0 axes must be zero. "
            "Fit raw device data with 1_data_fitting before importing it."
        )
    return result


def _read_lut_csv(path: Path) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for skiprows, usecols in ((1, range(1, 257)), (0, None)):
        try:
            candidates.append(np.loadtxt(path, delimiter=",", skiprows=skiprows, usecols=usecols))
        except (ValueError, IndexError):
            continue
    for table in candidates:
        if table.shape == (256, 256):
            return table
    shapes = [item.shape for item in candidates]
    raise ValueError(f"Could not read a 256x256 LUT from {path}; parsed shapes: {shapes}")


@dataclass(frozen=True)
class HardwareErrorModel:
    """Versioned deterministic multiplication model using the `lut[W, X]` convention."""

    lut: np.ndarray = field(repr=False)
    source_path: str
    source_sha256: str
    source_kind: str
    fitting_method: str = "measured interaction current with least-squares gain"
    gain: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "lut", _validate_lut(self.lut))
        if self.source_kind not in {"fitted_lut", "raw_device_csv", "model_bundle"}:
            raise ValueError(f"Unsupported source_kind: {self.source_kind}")

    @classmethod
    def from_lut_csv(
        cls,
        path: str | Path,
        *,
        fitting_summary: str | Path | None = None,
    ) -> "HardwareErrorModel":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        summary: dict[str, Any] = {}
        summary_path = Path(fitting_summary).resolve() if fitting_summary else source.with_name("fitting_summary.json")
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return cls(
            lut=_read_lut_csv(source),
            source_path=str(source),
            source_sha256=_sha256(source),
            source_kind="fitted_lut",
            gain=summary.get("lut_gain"),
            metadata={"fitting_summary": summary},
        )

    @classmethod
    def from_raw_csv(
        cls,
        path: str | Path,
        *,
        column_mapping: dict[str, str] | None = None,
        encoding: str = "auto",
        delimiter: str = "auto",
        missing_policy: str = "error",
        code_bits: int = 8,
        code_format: str = "auto",
        units: dict[str, str] | None = None,
    ) -> "HardwareErrorModel":
        """Fit a raw device grid by calling the existing public fitting service."""

        source = Path(path).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[2]
        fitting_root = project_root / "1_data_fitting"
        if str(fitting_root) not in sys.path:
            sys.path.insert(0, str(fitting_root))
        from data_fitting import DataFittingService, FittingConfig

        result = DataFittingService().fit(
            source,
            config=FittingConfig(
                generate_plots=False,
                export_point_data=False,
                column_mapping=column_mapping,
                encoding=encoding,
                delimiter=delimiter,
                missing_policy=missing_policy,
                code_bits=code_bits,
                code_format=code_format,
                units=units or {},
            ),
        )
        return cls(
            lut=result.lut,
            source_path=str(source),
            source_sha256=result.dataset.source_hash,
            source_kind="raw_device_csv",
            gain=result.lut_gain,
            metadata={
                "dataset": result.dataset.name,
                "column_mapping": result.dataset.column_mapping,
                "acquisition": {
                    "code_bits": result.dataset.acquisition_bits,
                    "code_range": [0, result.dataset.code_max],
                    "grid_semantics": "native acquisition grid",
                    "units": result.dataset.units,
                    **result.dataset.import_metadata,
                },
                "interaction_metrics": result.interaction_metrics.as_dict(),
                "fit_models": {
                    name: {
                        "equation": fit.equation,
                        "coefficients": fit.coefficients,
                        "metrics": fit.metrics.as_dict(),
                    }
                    for name, fit in result.fits.items()
                },
            },
        )

    def effective_lut(self, error_strength: float = 1.0) -> np.ndarray:
        if not 0.0 <= error_strength <= 1.0:
            raise ValueError("error_strength (lambda) must be in [0, 1]")
        codes = np.arange(256, dtype=np.int64)
        ideal = codes[:, None] * codes[None, :]
        return np.rint(ideal + error_strength * (self.lut - ideal)).astype(np.int64)

    def save(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        lut_path = target / "multiplication_lut.npy"
        np.save(lut_path, self.lut, allow_pickle=False)
        manifest = {
            "schema_version": self.schema_version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": self.source_path,
                "sha256": self.source_sha256,
                "kind": self.source_kind,
            },
            "lut": {
                "file": lut_path.name,
                "sha256": _sha256(lut_path),
                "axis_order": ["W", "X"],
                "shape": [256, 256],
                "code_bits": 8,
                "value_bits": 16,
                "rounding": "round-to-nearest (NumPy rint)",
                "clipping": [0, 65535],
                "baseline_semantics": "interaction LUT; zero X/W axes",
            },
            "error_definition": "E(X,W) = H(X,W) - X*W",
            "effective_lut": "H_lambda = round(X*W + lambda*(H-X*W))",
            "fitting": {"method": self.fitting_method, "gain": self.gain},
            "metadata": self.metadata,
        }
        manifest_path = target / "error_model.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest_path

    @classmethod
    def load(cls, manifest_path: str | Path) -> "HardwareErrorModel":
        source = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {manifest.get('schema_version')}")
        lut_path = source.parent / manifest["lut"]["file"]
        if _sha256(lut_path) != manifest["lut"]["sha256"]:
            raise ValueError("Saved LUT hash does not match the manifest")
        lut = np.load(lut_path, allow_pickle=False)
        return cls(
            lut=lut,
            source_path=manifest["source"]["path"],
            source_sha256=manifest["source"]["sha256"],
            source_kind="model_bundle",
            fitting_method=manifest["fitting"]["method"],
            gain=manifest["fitting"].get("gain"),
            metadata=manifest.get("metadata", {}),
            schema_version=manifest["schema_version"],
        )
