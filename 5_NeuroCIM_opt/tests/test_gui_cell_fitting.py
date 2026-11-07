from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from neurocim_opt.application import CellImportOptions, PlatformService
from neurocim_opt.error_model import HardwareErrorModel
from neurocim_opt.gui import CellFittingWidget


MAPPING = {"X": "xin", "W": "weight", "VD": "vin", "VG": "gate", "output": "sense"}


def _custom_csv(path: Path, *, missing: bool = False) -> Path:
    rows = []
    for weight in range(4):
        for x in range(4):
            output = 1e-6 * (1 + x + weight + x * weight)
            rows.append({"xin": x, "weight": weight, "vin": 0.1 * x,
                         "gate": 0.2 * weight, "sense": output})
    if missing:
        rows[6]["sense"] = None
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _options(**changes: object) -> CellImportOptions:
    values = dict(column_mapping=MAPPING, code_bits=2, code_format="decimal",
                  units={"VD": "V", "VG": "V", "output": "uA"})
    values.update(changes)
    return CellImportOptions(**values)


def test_custom_columns_fit_and_saved_error_model_loads(tmp_path: Path) -> None:
    source = _custom_csv(tmp_path / "custom.csv")
    outcome = PlatformService().fit_cell(source, tmp_path / "result", _options())
    loaded = HardwareErrorModel.load(outcome.error_model_path)
    assert loaded.lut.shape == (256, 256)
    assert outcome.summary_path.is_file()
    assert len(outcome.plot_paths) == 8
    assert all(path.is_file() for path in outcome.plot_paths)
    assert outcome.metadata["code_bits"] == 2
    assert outcome.metadata["units"]["output"] == "uA"
    assert outcome.metrics["r2"] == pytest.approx(1.0)


def test_bad_column_bad_width_and_missing_value_are_reported(tmp_path: Path) -> None:
    source = _custom_csv(tmp_path / "custom.csv", missing=True)
    service = PlatformService()
    with pytest.raises(ValueError, match="Mapped columns do not exist"):
        service.preview_cell(source, _options(column_mapping={**MAPPING, "output": "absent"}))
    with pytest.raises(ValueError, match="code_bits must be one of"):
        service.preview_cell(source, _options(code_bits=3))
    with pytest.raises(ValueError, match="missing or non-numeric"):
        service.preview_cell(source, _options(missing_policy="error"))
    preview = service.preview_cell(source, _options(missing_policy="interpolate"))
    assert preview.rows == 16


def test_01_nmos_metric_regression_has_no_drift(tmp_path: Path) -> None:
    outcome = PlatformService().fit_cell(
        PROJECT_ROOT / "0_data" / "01_NMOS.csv", tmp_path / "nmos", CellImportOptions({})
    )
    assert outcome.metrics["r2"] == pytest.approx(0.9908762695039321, abs=1e-12)
    assert outcome.metrics["nl_percent_fs"] == pytest.approx(6.578497847717723, abs=1e-10)


def test_widget_background_fit_error_feedback_and_settings(tmp_path: Path) -> None:
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    settings = QSettings("DRAM-CIM", "NeuroSimPlatform")
    settings.clear()
    settings.sync()
    source = _custom_csv(tmp_path / "custom.csv")
    app = QApplication.instance() or QApplication([])
    widget = CellFittingWidget(ROOT)
    widget.path_edit.setText(str(source))
    for key, column in MAPPING.items():
        widget.mapping_edits[key].setCurrentText(column)
    widget.bits_combo.setCurrentIndex(widget.bits_combo.findData(2))
    widget.code_format_combo.setCurrentIndex(widget.code_format_combo.findData("decimal"))
    widget.output_unit_edit.setText("uA")
    widget.preview()
    assert widget.table.rowCount() == 16
    widget.start_fit(tmp_path / "gui_result")
    assert widget.worker is not None and widget.worker.wait(30_000)
    app.processEvents()
    assert widget.last_outcome is not None, widget.log.toPlainText()
    assert HardwareErrorModel.load(widget.last_outcome.error_model_path).lut.shape == (256, 256)
    widget._persist_settings()
    restored = CellFittingWidget(ROOT)
    assert restored.path_edit.text() == str(source)
    assert restored.bits_combo.currentData() == 2
    assert restored.mapping_edits["output"].currentText() == "sense"
    restored.path_edit.setText(str(tmp_path / "does-not-exist.csv"))
    restored.preview()
    assert "预览失败" in restored.summary.text()
    widget.close(); restored.close()


def test_csv_headers_populate_mapping_dropdowns(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    source = _custom_csv(tmp_path / "mapping_headers.csv")
    widget = CellFittingWidget(ROOT)
    widget.path_edit.setText(str(source))
    widget._refresh_mapping_columns()
    app.processEvents()

    expected = {"xin", "weight", "vin", "gate", "sense"}
    for combo in widget.mapping_edits.values():
        choices = {str(combo.itemData(index)) for index in range(1, combo.count())}
        assert expected.issubset(choices)
        assert combo.itemData(0) == ""
    widget.mapping_edits["X"].setCurrentText("xin")
    widget.mapping_edits["W"].setCurrentText("weight")
    widget.mapping_edits["output"].setCurrentText("sense")
    assert widget._options().column_mapping["output"] == "sense"

    standard = tmp_path / "standard.csv"
    pd.DataFrame({
        "VD_Digital": ["00"], "VG_Digital": ["00"], "VD": [0.0],
        "VG": [0.0], "Iout": [0.0], "extra": [1],
    }).to_csv(standard, index=False)
    widget.path_edit.setText(str(standard))
    widget._refresh_mapping_columns()
    assert widget._options().column_mapping == {
        "X": "VD_Digital", "W": "VG_Digital", "VD": "VD", "VG": "VG", "output": "Iout",
    }
    assert "已应用标准列映射" in widget.log.toPlainText()
    widget.close()


def test_uos2_standard_mapping_and_bit_width_are_detected() -> None:
    app = QApplication.instance() or QApplication([])
    widget = CellFittingWidget(ROOT)
    widget.path_edit.setText(str(PROJECT_ROOT / "0_data" / "04_UOS_2.csv"))
    widget._refresh_mapping_columns()
    app.processEvents()

    assert widget._options().column_mapping == {
        "X": "VD_Digital", "W": "VG_Digital", "VD": "VD", "VG": "VG", "output": "I",
    }
    assert widget.bits_combo.currentData() == 8
    assert widget.code_format_combo.currentData() == "binary"
    widget.preview()
    assert widget.table.rowCount() == 20
    assert "校验通过：65536 行" in widget.summary.text()
    widget.close()
