from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


MAC_SCHEMA_VERSION = "1.0"
ALLOWED_BITS = {1, 2, 4, 8}


class MacDataValidationError(ValueError):
    """Raised when array-MAC data cannot be fitted without changing its meaning."""


@dataclass(frozen=True)
class MacFitMetrics:
    r2: float
    rmse: float
    mae: float
    max_abs_error: float
    full_scale: float
    nonlinearity_percent_fs: float


@dataclass(frozen=True)
class MacGroupModel:
    array_rows: int
    array_cols: int
    bit_width: int
    group: str
    samples: int
    ideal_min: float
    ideal_max: float
    linear_coefficients: tuple[float, float]
    polynomial_coefficients: tuple[float, ...]
    linear_metrics: MacFitMetrics
    polynomial_metrics: MacFitMetrics

    @property
    def key(self) -> tuple[int, int, int, str]:
        return self.array_rows, self.array_cols, self.bit_width, self.group


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_text_sample(path: Path, encoding: str | None) -> tuple[str, str]:
    candidates = [encoding] if encoding else ["utf-8-sig", "utf-8", "gb18030"]
    raw = path.read_bytes()[:65536]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return raw.decode(candidate), candidate
        except UnicodeDecodeError:
            pass
    raise MacDataValidationError("CSV encoding is not UTF-8/UTF-8-BOM/GBK compatible")


def _detect_delimiter(sample: str, delimiter: str | None) -> str:
    if delimiter is not None:
        if delimiter not in {",", ";"}:
            raise MacDataValidationError("delimiter must be ',' or ';'")
        return delimiter
    try:
        detected = csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except csv.Error as exc:
        raise MacDataValidationError("Could not detect CSV delimiter") from exc
    return detected


_ALIASES = {
    "array_rows": ("array_rows", "rows", "m", "M"),
    "array_cols": ("array_cols", "cols", "n", "N"),
    "array_size": ("array_size", "array", "阵列规模", "加法规模"),
    "bit_width": ("bit_width", "bits", "n_bit", "位宽"),
    "ideal_mac": ("ideal_mac", "ideal", "mac_input", "理想乘累加"),
    "measured_output": ("measured_output", "output", "iout", "vout", "乘累加输出"),
    "group": ("group", "device", "experiment", "实验组"),
}


def _resolve_columns(columns: Iterable[str], mapping: dict[str, str] | None) -> dict[str, str]:
    available = set(columns)
    resolved = dict(mapping or {})
    for logical, aliases in _ALIASES.items():
        if logical not in resolved:
            match = next((name for name in aliases if name in available), None)
            if match:
                resolved[logical] = match
    required = {"bit_width", "ideal_mac", "measured_output"}
    missing = required.difference(resolved)
    if missing:
        raise MacDataValidationError(f"Missing column mappings: {sorted(missing)}")
    if "array_size" not in resolved and not {"array_rows", "array_cols"}.issubset(resolved):
        raise MacDataValidationError("Map array_size or both array_rows and array_cols")
    unknown = [name for name in resolved.values() if name not in available]
    if unknown:
        raise MacDataValidationError(f"Mapped columns not found: {unknown}")
    return resolved


def _parse_array_size(value: Any) -> tuple[int, int]:
    text = str(value).strip().lower().replace("×", "x")
    match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    try:
        size = int(float(text))
    except ValueError as exc:
        raise MacDataValidationError(f"Invalid array_size value: {value!r}") from exc
    return size, size


def load_mac_csv(
    path: str | Path,
    *,
    column_mapping: dict[str, str] | None = None,
    encoding: str | None = None,
    delimiter: str | None = None,
    missing: str = "error",
    array_rows: int | None = None,
    array_cols: int | None = None,
    bit_width: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load array-MAC samples into a validated canonical table."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".csv":
        raise MacDataValidationError("Array-MAC input must be a CSV file")
    if not source.is_file():
        raise FileNotFoundError(source)
    sample, actual_encoding = _read_text_sample(source, encoding)
    actual_delimiter = _detect_delimiter(sample, delimiter)
    raw = pd.read_csv(source, encoding=actual_encoding, sep=actual_delimiter)
    mapping = _resolve_columns(raw.columns, column_mapping)

    selected = list(dict.fromkeys(mapping.values()))
    frame = raw[selected].copy()
    if missing not in {"error", "drop", "interpolate"}:
        raise MacDataValidationError("missing must be error, drop, or interpolate")
    if frame.isna().any().any():
        if missing == "error":
            counts = frame.isna().sum()
            raise MacDataValidationError(f"Missing values: {counts[counts > 0].to_dict()}")
        if missing == "drop":
            frame = frame.dropna()
        else:
            value_columns = [mapping["ideal_mac"], mapping["measured_output"]]
            frame[value_columns] = frame[value_columns].apply(pd.to_numeric, errors="coerce").interpolate(
                limit_direction="both"
            )
            frame = frame.dropna()

    canonical = pd.DataFrame()
    if "array_size" in mapping:
        sizes = frame[mapping["array_size"]].map(_parse_array_size)
        canonical["array_rows"] = sizes.map(lambda item: item[0])
        canonical["array_cols"] = sizes.map(lambda item: item[1])
    else:
        canonical["array_rows"] = pd.to_numeric(frame[mapping["array_rows"]], errors="raise")
        canonical["array_cols"] = pd.to_numeric(frame[mapping["array_cols"]], errors="raise")
    canonical["bit_width"] = pd.to_numeric(frame[mapping["bit_width"]], errors="raise")
    canonical["ideal_mac"] = pd.to_numeric(frame[mapping["ideal_mac"]], errors="raise")
    canonical["measured_output"] = pd.to_numeric(frame[mapping["measured_output"]], errors="raise")
    canonical["group"] = (
        frame[mapping["group"]].astype(str) if "group" in mapping else source.stem
    )

    numeric = canonical.drop(columns="group").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise MacDataValidationError("Array-MAC data contain NaN or infinite values")
    for column in ("array_rows", "array_cols", "bit_width"):
        values = canonical[column].to_numpy(dtype=float)
        if not np.all(values == np.floor(values)):
            raise MacDataValidationError(f"{column} must contain integers")
        canonical[column] = values.astype(np.int64)
    if (canonical[["array_rows", "array_cols"]] <= 0).any().any():
        raise MacDataValidationError("Array dimensions must be positive")
    invalid_bits = sorted(set(canonical["bit_width"]) - ALLOWED_BITS)
    if invalid_bits:
        raise MacDataValidationError(f"Unsupported bit widths: {invalid_bits}; use 1/2/4/8")
    if canonical.empty:
        raise MacDataValidationError("No usable rows remain after missing-value handling")
    selection = {
        "array_rows": array_rows,
        "array_cols": array_cols,
        "bit_width": bit_width,
    }
    for name, value in selection.items():
        if value is None:
            continue
        if not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise MacDataValidationError(f"{name} selection must be a positive integer")
    if bit_width is not None and int(bit_width) not in ALLOWED_BITS:
        raise MacDataValidationError("bit_width selection must be one of 1/2/4/8")
    for name, value in selection.items():
        if value is not None:
            canonical = canonical[canonical[name] == int(value)]
    if canonical.empty:
        requested = {name: int(value) for name, value in selection.items() if value is not None}
        raise MacDataValidationError(f"No rows match array configuration {requested}")
    metadata = {
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "encoding": actual_encoding,
        "delimiter": actual_delimiter,
        "column_mapping": mapping,
        "missing_policy": missing,
        "rows": len(canonical),
        "selection": {name: int(value) for name, value in selection.items() if value is not None},
    }
    return canonical.reset_index(drop=True), metadata


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> MacFitMetrics:
    residual = actual - predicted
    ss_res = float(np.sum(residual**2))
    centered = actual - actual.mean()
    ss_total = float(np.sum(centered**2))
    r2 = 1.0 if ss_total == 0 and ss_res == 0 else (0.0 if ss_total == 0 else 1 - ss_res / ss_total)
    full_scale = float(actual.max() - actual.min())
    max_abs = float(np.max(np.abs(residual)))
    return MacFitMetrics(
        r2=r2,
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(np.abs(residual))),
        max_abs_error=max_abs,
        full_scale=full_scale,
        nonlinearity_percent_fs=0.0 if full_scale == 0 else 100 * max_abs / full_scale,
    )


@dataclass(frozen=True)
class MacErrorModel:
    groups: tuple[MacGroupModel, ...]
    source: dict[str, Any]
    polynomial_degree: int = 2
    schema_version: str = MAC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("MacErrorModel requires at least one fitted group")
        keys = [model.key for model in self.groups]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate MAC model group keys")

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        source: dict[str, Any],
        polynomial_degree: int = 2,
    ) -> "MacErrorModel":
        if polynomial_degree not in {1, 2, 3}:
            raise ValueError("polynomial_degree must be 1, 2, or 3")
        models: list[MacGroupModel] = []
        keys = ["array_rows", "array_cols", "bit_width", "group"]
        for key, part in frame.groupby(keys, sort=True, dropna=False):
            x = part["ideal_mac"].to_numpy(dtype=float)
            y = part["measured_output"].to_numpy(dtype=float)
            unique = np.unique(x).size
            if len(part) < max(3, polynomial_degree + 1) or unique < polynomial_degree + 1:
                raise MacDataValidationError(
                    f"Group {key} needs at least {max(3, polynomial_degree + 1)} rows and "
                    f"{polynomial_degree + 1} distinct ideal_mac values"
                )
            linear = np.polyfit(x, y, 1)
            polynomial = np.polyfit(x, y, polynomial_degree)
            models.append(
                MacGroupModel(
                    array_rows=int(key[0]), array_cols=int(key[1]), bit_width=int(key[2]),
                    group=str(key[3]), samples=len(part), ideal_min=float(x.min()), ideal_max=float(x.max()),
                    linear_coefficients=tuple(float(v) for v in linear),
                    polynomial_coefficients=tuple(float(v) for v in polynomial),
                    linear_metrics=_metrics(y, np.polyval(linear, x)),
                    polynomial_metrics=_metrics(y, np.polyval(polynomial, x)),
                )
            )
        return cls(tuple(models), dict(source), polynomial_degree)

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> "MacErrorModel":
        degree = int(kwargs.pop("polynomial_degree", 2))
        frame, source = load_mac_csv(path, **kwargs)
        return cls.fit(frame, source=source, polynomial_degree=degree)

    def select(self, array_rows: int, array_cols: int, bit_width: int, group: str) -> MacGroupModel:
        key = (array_rows, array_cols, bit_width, group)
        match = next((item for item in self.groups if item.key == key), None)
        if match is None:
            raise KeyError(f"No fitted MAC group for {key}; available: {[item.key for item in self.groups]}")
        return match

    def predict(
        self,
        ideal_mac: np.ndarray | float,
        *,
        array_rows: int,
        array_cols: int,
        bit_width: int,
        group: str,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        model = self.select(array_rows, array_cols, bit_width, group)
        values = np.asarray(ideal_mac, dtype=float)
        if not allow_extrapolation and (
            np.any(values < model.ideal_min) or np.any(values > model.ideal_max)
        ):
            raise ValueError(
                f"ideal_mac is outside fitted range [{model.ideal_min}, {model.ideal_max}]"
            )
        return np.polyval(model.polynomial_coefficients, values)

    def save(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "polynomial_degree": self.polynomial_degree,
            "groups": [asdict(item) for item in self.groups],
        }
        payload["integrity_sha256"] = _canonical_hash(payload)
        path = target / "mac_error_model.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MacErrorModel":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        integrity = payload.pop("integrity_sha256", None)
        if integrity != _canonical_hash(payload):
            raise ValueError("MAC model integrity hash does not match")
        if payload.get("schema_version") != MAC_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MAC schema: {payload.get('schema_version')}")
        groups = []
        for item in payload["groups"]:
            item["linear_coefficients"] = tuple(item["linear_coefficients"])
            item["polynomial_coefficients"] = tuple(item["polynomial_coefficients"])
            item["linear_metrics"] = MacFitMetrics(**item["linear_metrics"])
            item["polynomial_metrics"] = MacFitMetrics(**item["polynomial_metrics"])
            groups.append(MacGroupModel(**item))
        return cls(
            tuple(groups), payload["source"], int(payload["polynomial_degree"]), payload["schema_version"]
        )
