from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt import MacDataValidationError, MacErrorModel, load_mac_csv


def _write_mac_csv(path: Path, *, separator: str = ",", encoding: str = "utf-8") -> None:
    x = np.linspace(0, 100, 21)
    pd.DataFrame(
        {
            "阵列": ["16x32"] * len(x),
            "位宽": [4] * len(x),
            "理想值": x,
            "测量值": 3 + 2 * x + 0.01 * x**2,
            "批次": ["sample-a"] * len(x),
        }
    ).to_csv(path, index=False, sep=separator, encoding=encoding)


MAPPING = {
    "array_size": "阵列",
    "bit_width": "位宽",
    "ideal_mac": "理想值",
    "measured_output": "测量值",
    "group": "批次",
}


def test_mac_model_fit_predict_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "mac.csv"
    _write_mac_csv(source)
    model = MacErrorModel.from_csv(source, column_mapping=MAPPING)
    group = model.groups[0]
    assert group.key == (16, 32, 4, "sample-a")
    assert group.polynomial_metrics.r2 == pytest.approx(1.0)
    assert group.linear_metrics.nonlinearity_percent_fs > 0
    predicted = model.predict(
        np.array([0.0, 50.0, 100.0]),
        array_rows=16, array_cols=32, bit_width=4, group="sample-a",
    )
    assert np.allclose(predicted, [3.0, 128.0, 303.0])
    loaded = MacErrorModel.load(model.save(tmp_path / "bundle"))
    assert loaded.groups[0] == model.groups[0]


def test_semicolon_gbk_and_missing_drop(tmp_path: Path) -> None:
    source = tmp_path / "mac_gbk.csv"
    _write_mac_csv(source, separator=";", encoding="gb18030")
    frame, metadata = load_mac_csv(source, column_mapping=MAPPING, missing="drop")
    assert len(frame) == 21
    assert metadata["delimiter"] == ";"
    assert metadata["encoding"] == "gb18030"


def test_rejects_invalid_bit_width(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    _write_mac_csv(source)
    frame = pd.read_csv(source)
    frame["位宽"] = 3
    frame.to_csv(source, index=False)
    with pytest.raises(MacDataValidationError, match="bit widths"):
        load_mac_csv(source, column_mapping=MAPPING)


def test_rejects_prediction_outside_fitted_range(tmp_path: Path) -> None:
    source = tmp_path / "mac.csv"
    _write_mac_csv(source)
    model = MacErrorModel.from_csv(source, column_mapping=MAPPING)
    with pytest.raises(ValueError, match="outside fitted range"):
        model.predict(101, array_rows=16, array_cols=32, bit_width=4, group="sample-a")


def test_detects_modified_model_bundle(tmp_path: Path) -> None:
    source = tmp_path / "mac.csv"
    _write_mac_csv(source)
    path = MacErrorModel.from_csv(source, column_mapping=MAPPING).save(tmp_path / "bundle")
    payload = path.read_text(encoding="utf-8").replace('"polynomial_degree": 2', '"polynomial_degree": 3')
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        MacErrorModel.load(path)
