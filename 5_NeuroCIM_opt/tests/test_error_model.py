from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt import HardwareErrorModel


def ideal_lut() -> np.ndarray:
    code = np.arange(256, dtype=np.int64)
    return code[:, None] * code[None, :]


def test_model_round_trip_and_lambda_endpoints(tmp_path: Path) -> None:
    table = ideal_lut()
    table[1:, 1:] += 3
    model = HardwareErrorModel(table, "synthetic", "abc", "fitted_lut")
    manifest = model.save(tmp_path)
    loaded = HardwareErrorModel.load(manifest)
    assert np.array_equal(loaded.lut, table)
    assert np.array_equal(loaded.effective_lut(0), ideal_lut())
    assert np.array_equal(loaded.effective_lut(1), table)


def test_rejects_raw_surface_with_nonzero_baseline() -> None:
    with pytest.raises(ValueError, match="zero"):
        HardwareErrorModel(np.ones((256, 256)), "synthetic", "abc", "fitted_lut")


def test_rejects_invalid_lambda() -> None:
    model = HardwareErrorModel(ideal_lut(), "synthetic", "abc", "fitted_lut")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.effective_lut(1.01)


def test_low_bit_raw_csv_bundle_round_trip_preserves_acquisition_metadata(tmp_path: Path) -> None:
    source = tmp_path / "coarse.csv"
    pd.DataFrame(
        [
            {"x": x, "w": w, "sense": float(x * w)}
            for w in range(2) for x in range(2)
        ]
    ).to_csv(source, index=False)
    model = HardwareErrorModel.from_raw_csv(
        source,
        column_mapping={"X": "x", "W": "w", "output": "sense"},
        code_bits=1,
        code_format="decimal",
        units={"output": "mV"},
    )
    manifest = model.save(tmp_path / "bundle")
    loaded = HardwareErrorModel.load(manifest)
    assert np.array_equal(loaded.lut, model.lut)
    assert loaded.metadata["acquisition"]["code_bits"] == 1
    assert loaded.metadata["acquisition"]["code_range"] == [0, 1]
    assert loaded.metadata["acquisition"]["units"]["output"] == "mV"
