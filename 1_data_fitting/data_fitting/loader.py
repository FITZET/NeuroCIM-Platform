from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .models import DatasetKind, DeviceDataset, FittingConfig


class DataValidationError(ValueError):
    """Raised when imported data cannot safely be used for fitting."""


_DEVICE_OUTPUT_COLUMNS = ("Iout", "Ianalog", "I", "Iout_Analog")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_code(series: pd.Series, column: str, code_max: int, code_format: str, *, allow_missing: bool) -> np.ndarray:
    values: list[float] = []
    for row, raw in enumerate(series, start=2):
        text = str(raw).strip()
        if pd.isna(raw) or text.lower() in {"", "<na>", "nan", "none"}:
            if allow_missing:
                values.append(float("nan"))
                continue
            raise DataValidationError(f"{column} row {row} is missing")
        if text.endswith(".0"):
            text = text[:-2]
        binary = code_format == "binary" or (
            code_format == "auto" and any(char not in "0123456789" for char in text) is False
            and set(text) <= {"0", "1"} and (len(text) > 1 and text.startswith("0"))
        )
        try:
            value = int(text, 2 if binary else 10)
        except ValueError as exc:
            raise DataValidationError(f"{column} row {row} is not a valid {code_format} code: {raw!r}") from exc
        if not 0 <= value <= code_max:
            raise DataValidationError(f"{column} row {row} is outside acquisition range [0, {code_max}]: {value}")
        values.append(value)
    return np.asarray(values, dtype=float)


def _validate_grid(frame: pd.DataFrame, levels: int, *, require_complete: bool) -> None:
    if require_complete and len(frame) != levels * levels:
        raise DataValidationError(
            f"Expected {levels * levels:,} rows for a complete {levels}-level acquisition grid, got {len(frame):,}."
        )
    if frame[["X", "W"]].duplicated().any():
        raise DataValidationError("Duplicate (X, W) code combinations were found.")
    expected = pd.MultiIndex.from_product([range(levels), range(levels)])
    actual = pd.MultiIndex.from_frame(frame[["X", "W"]])
    missing = expected.difference(actual)
    if require_complete and len(missing):
        raise DataValidationError(f"Missing {len(missing)} (X, W) code combinations.")
    if not np.isfinite(frame[["X", "W", "VD", "VG", "output"]].to_numpy(dtype=float)).all():
        raise DataValidationError("NaN or infinite numeric values were found.")


def _read_csv(
    source: Path, encoding: str, delimiter: str, *, nrows: int | None = None
) -> tuple[pd.DataFrame, str, str]:
    encodings = ("utf-8-sig", "gbk") if encoding == "auto" else (encoding,)
    separators = (",", ";") if delimiter == "auto" else (delimiter,)
    errors: list[str] = []
    for candidate_encoding in encodings:
        for separator in separators:
            try:
                frame = pd.read_csv(
                    source, encoding=candidate_encoding, sep=separator,
                    dtype="string", nrows=nrows,
                )
                if len(frame.columns) > 1:
                    return frame, candidate_encoding, separator
                errors.append(
                    f"{candidate_encoding} with separator {separator!r} produced "
                    f"{len(frame.columns)} column"
                )
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                errors.append(str(exc))
    detail = errors[-2:] if errors else ["no valid multi-column CSV header was found"]
    raise DataValidationError(f"Could not parse CSV using requested encoding/delimiter: {detail}")


def inspect_csv_columns(
    path: str | Path, *, encoding: str = "auto", delimiter: str = "auto"
) -> tuple[tuple[str, ...], str, str]:
    """Read only the CSV header for GUI column-mapping controls."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame, used_encoding, used_delimiter = _read_csv(
        source, encoding, delimiter, nrows=0
    )
    return tuple(str(column) for column in frame.columns), used_encoding, used_delimiter


def load_dataset(path: str | Path, *, config: FittingConfig | None = None) -> DeviceDataset:
    """Load either a device CSV or the ideal multiplication reference."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    config = config or FittingConfig()
    if config.code_bits not in {1, 2, 4, 8}:
        raise DataValidationError("code_bits must be one of 1, 2, 4, or 8")
    if config.missing_policy not in {"error", "drop", "interpolate"}:
        raise DataValidationError("missing_policy must be 'error', 'drop', or 'interpolate'")
    if config.code_format not in {"auto", "binary", "decimal"}:
        raise DataValidationError("code_format must be 'auto', 'binary', or 'decimal'")
    levels = 1 << config.code_bits
    code_max = levels - 1
    raw, used_encoding, used_delimiter = _read_csv(source, config.encoding, config.delimiter)
    mapping: dict[str, str] = {}

    ideal_columns = {"input_code_dec", "weight_code_dec", "ideal_product_dec"}
    custom = config.column_mapping or {}
    if custom:
        required_keys = {"X", "W", "output"}
        missing_columns = set(custom.values()).difference(raw.columns)
        if missing_columns:
            raise DataValidationError(f"Mapped columns do not exist: {sorted(missing_columns)}")
        aliases = {"x": "X", "w": "W", "vd": "VD", "vg": "VG", "output": "output"}
        mapping = {aliases.get(key.lower(), key): value for key, value in custom.items()}
        missing_keys = required_keys.difference(mapping)
        if missing_keys:
            raise DataValidationError(f"column_mapping lacks canonical keys: {sorted(missing_keys)}")
        kind = DatasetKind.DEVICE
    elif ideal_columns.issubset(raw.columns):
        mapping = {
            "X": "input_code_dec",
            "W": "weight_code_dec",
            "output": "ideal_product_dec",
        }
        kind = DatasetKind.IDEAL
    else:
        required = {"VD_Digital", "VG_Digital", "VD", "VG"}
        missing = required.difference(raw.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {sorted(missing)}")
        output_column = next((c for c in _DEVICE_OUTPUT_COLUMNS if c in raw), None)
        if output_column is None:
            raise DataValidationError(
                f"No output column found; supported names: {_DEVICE_OUTPUT_COLUMNS}"
            )
        mapping = {
            "X": "VD_Digital",
            "W": "VG_Digital",
            "VD": "VD",
            "VG": "VG",
            "output": output_column,
        }
        kind = DatasetKind.DEVICE

    forced_format = config.code_format
    if not custom and kind is DatasetKind.DEVICE:
        forced_format = "binary"
    frame = pd.DataFrame({
        "X": _parse_code(raw[mapping["X"]], mapping["X"], code_max, forced_format, allow_missing=config.missing_policy == "drop"),
        "W": _parse_code(raw[mapping["W"]], mapping["W"], code_max, forced_format, allow_missing=config.missing_policy == "drop"),
        "output": pd.to_numeric(raw[mapping["output"]], errors="coerce"),
    })
    frame["VD"] = pd.to_numeric(raw[mapping["VD"]], errors="coerce") if "VD" in mapping else frame["X"].astype(float)
    frame["VG"] = pd.to_numeric(raw[mapping["VG"]], errors="coerce") if "VG" in mapping else frame["W"].astype(float)
    numeric = ["X", "W", "VD", "VG", "output"]
    missing_rows = frame[numeric].isna().any(axis=1)
    missing_count = int(missing_rows.sum())
    if missing_count and config.missing_policy == "error":
        raise DataValidationError(f"Found {missing_count} rows with missing or non-numeric values")
    if missing_count and config.missing_policy == "drop":
        frame = frame.loc[~missing_rows].copy()
    if missing_count and config.missing_policy == "interpolate":
        frame[["VD", "VG", "output"]] = frame[["VD", "VG", "output"]].interpolate(limit_direction="both")
        if frame[numeric].isna().any().any():
            raise DataValidationError("X and W codes cannot be interpolated")
    frame[["X", "W"]] = frame[["X", "W"]].astype(np.uint16)
    if not frame["X"].between(0, code_max).all() or not frame["W"].between(0, code_max).all():
        raise DataValidationError(f"X and W must be acquisition codes in [0, {code_max}].")
    frame = frame.sort_values(["W", "X"], kind="stable").reset_index(drop=True)
    _validate_grid(frame, levels, require_complete=config.missing_policy == "error")
    return DeviceDataset(
        source_path=source,
        name=source.stem,
        kind=kind,
        frame=frame,
        source_hash=_sha256(source),
        column_mapping=mapping,
        acquisition_bits=config.code_bits,
        code_max=code_max,
        units={"VD": "V", "VG": "V", "output": "code" if kind is DatasetKind.IDEAL else "A", **config.units},
        import_metadata={
            "encoding": used_encoding,
            "delimiter": used_delimiter,
            "missing_policy": config.missing_policy,
            "missing_rows": missing_count,
        },
    )
