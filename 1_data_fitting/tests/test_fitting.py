from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from data_fitting import DataFittingService, FittingConfig
from data_fitting.loader import DataValidationError


def test_ideal_reference_is_exact(tmp_path: Path) -> None:
    result = DataFittingService().fit(
        PROJECT_ROOT / "0_data" / "05_ideal_multiplication.csv",
        output_dir=tmp_path,
        config=FittingConfig(generate_plots=False),
    )
    assert np.array_equal(result.lut, np.multiply.outer(np.arange(256), np.arange(256)).T)
    assert result.fits["linear_product"].metrics.r2 == 1.0
    assert result.fits["linear_product"].metrics.nl_percent_fs < 1e-10


def test_nmos_report_regression() -> None:
    result = DataFittingService().fit(PROJECT_ROOT / "0_data" / "01_NMOS.csv")
    assert abs(result.fits["linear_product"].metrics.r2 - 0.943194) < 1e-5
    assert abs(result.fits["quadratic_surface"].metrics.r2 - 0.999698) < 1e-5
    assert abs(result.fits["raw_product_output"].metrics.r2 - 0.943194) < 1e-5
    assert result.lut.shape == (256, 256)


def test_normalized_visualizations_are_exported(tmp_path: Path) -> None:
    result = DataFittingService().fit(
        PROJECT_ROOT / "0_data" / "03_UOS_1.csv",
        output_dir=tmp_path,
        config=FittingConfig(plot_dpi=80, export_point_data=False),
    )
    expected = {
        "normalized_linear_plot": "normalized_linear.png",
        "normalized_surface_plot": "normalized_surface.png",
        "normalized_lut_plot": "normalized_lut.png",
    }
    for key, filename in expected.items():
        assert result.artifacts[key] == tmp_path / filename
        assert result.artifacts[key].stat().st_size > 1_000


def _coarse_csv(path: Path, *, missing_output: bool = False) -> None:
    rows = []
    for w in range(4):
        for x in range(4):
            output = float(x * w)
            if missing_output and (x, w) == (2, 2):
                output = np.nan
            rows.append({"输入码": x, "权重码": w, "输出": output, "漏压": x / 3, "栅压": w / 3})
    pd.DataFrame(rows).to_csv(path, sep=";", index=False, encoding="gbk")


def test_gbk_semicolon_custom_mapping_and_low_bit_grid(tmp_path: Path) -> None:
    source = tmp_path / "device.csv"
    _coarse_csv(source)
    config = FittingConfig(
        generate_plots=False,
        code_bits=2,
        code_format="decimal",
        column_mapping={"x": "输入码", "w": "权重码", "output": "输出", "vd": "漏压", "vg": "栅压"},
        units={"output": "uA"},
    )
    result = DataFittingService().fit(source, config=config)
    assert result.dataset.frame[["X", "W"]].to_numpy().max() == 3
    assert result.dataset.acquisition_bits == 2
    assert result.dataset.units["output"] == "uA"
    assert result.dataset.import_metadata["encoding"] == "gbk"
    assert result.dataset.import_metadata["delimiter"] == ";"
    assert result.lut.shape == (256, 256)
    assert result.lut[-1, -1] == 255 * 255


@pytest.mark.parametrize("bits", [0, 3, 16])
def test_rejects_unsupported_acquisition_bit_width(tmp_path: Path, bits: int) -> None:
    source = tmp_path / "device.csv"
    _coarse_csv(source)
    with pytest.raises(DataValidationError, match="1, 2, 4, or 8"):
        DataFittingService().load(source, config=FittingConfig(code_bits=bits))


def test_rejects_code_outside_declared_acquisition_range(tmp_path: Path) -> None:
    source = tmp_path / "device.csv"
    _coarse_csv(source)
    frame = pd.read_csv(source, sep=";", encoding="gbk")
    frame.loc[0, "输入码"] = 4
    frame.to_csv(source, sep=";", index=False, encoding="gbk")
    with pytest.raises(DataValidationError, match="outside acquisition range"):
        DataFittingService().load(source, config=FittingConfig(
            code_bits=2, code_format="decimal",
            column_mapping={"X": "输入码", "W": "权重码", "output": "输出"},
        ))


def test_missing_value_policies(tmp_path: Path) -> None:
    source = tmp_path / "device.csv"
    _coarse_csv(source, missing_output=True)
    base = dict(code_bits=2, code_format="decimal", column_mapping={"X": "输入码", "W": "权重码", "output": "输出"})
    with pytest.raises(DataValidationError, match="missing"):
        DataFittingService().load(source, config=FittingConfig(**base, missing_policy="error"))
    dropped = DataFittingService().load(source, config=FittingConfig(**base, missing_policy="drop"))
    assert len(dropped.frame) == 15
    interpolated = DataFittingService().load(source, config=FittingConfig(**base, missing_policy="interpolate"))
    assert len(interpolated.frame) == 16
    assert np.isfinite(interpolated.frame["output"]).all()
