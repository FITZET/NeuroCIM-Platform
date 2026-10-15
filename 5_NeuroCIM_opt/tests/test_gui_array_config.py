from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from neurocim_opt.gui import MacAnalysisWidget
from neurocim_opt.mac_error_model import MacDataValidationError, MacErrorModel, load_mac_csv


def _dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-10.0, 10.0, 41)
    disturbed = 3.0 + 2.0 * x + 0.01 * x**2
    pd.concat([
        pd.DataFrame({"array_rows": 16, "array_cols": 32, "bit_width": 4,
                      "ideal_mac": x, "measured_output": 3.0 + 2.0 * x, "group": "linear"}),
        pd.DataFrame({"array_rows": 16, "array_cols": 32, "bit_width": 4,
                      "ideal_mac": x, "measured_output": disturbed, "group": "quadratic"}),
        pd.DataFrame({"array_rows": 32, "array_cols": 32, "bit_width": 8,
                      "ideal_mac": x, "measured_output": x, "group": "other"}),
    ], ignore_index=True).to_csv(path, index=False)
    return x, disturbed


def test_selected_configuration_and_theoretical_metrics(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.csv"
    x, disturbed = _dataset(source)
    frame, metadata = load_mac_csv(source, array_rows=16, array_cols=32, bit_width=4)
    assert len(frame) == 82
    assert metadata["selection"] == {"array_rows": 16, "array_cols": 32, "bit_width": 4}
    model = MacErrorModel.fit(frame, source=metadata, polynomial_degree=2)
    linear = model.select(16, 32, 4, "linear").linear_metrics
    assert f"{linear.r2:.4f}" == "1.0000"
    assert linear.nonlinearity_percent_fs < 1e-10

    # Independent closed-form ordinary least squares and metric calculation.
    slope = float(np.sum((x - x.mean()) * (disturbed - disturbed.mean())) / np.sum((x - x.mean()) ** 2))
    intercept = float(disturbed.mean() - slope * x.mean())
    residual = disturbed - (slope * x + intercept)
    expected_rmse = float(np.sqrt(np.mean(residual**2)))
    expected_r2 = float(1.0 - np.sum(residual**2) / np.sum((disturbed - disturbed.mean())**2))
    expected_nl = float(100.0 * np.max(np.abs(residual)) / np.ptp(disturbed))
    measured = model.select(16, 32, 4, "quadratic").linear_metrics
    assert measured.rmse == pytest.approx(expected_rmse, abs=1e-12)
    assert measured.r2 == pytest.approx(expected_r2, abs=1e-12)
    assert measured.nonlinearity_percent_fs == pytest.approx(expected_nl, abs=1e-12)

    with pytest.raises(MacDataValidationError, match="No rows match"):
        load_mac_csv(source, array_rows=64, array_cols=64, bit_width=1)
    with pytest.raises(MacDataValidationError, match="1/2/4/8"):
        load_mac_csv(source, array_rows=16, array_cols=32, bit_width=3)


def test_array_controls_drive_preview_fit_settings_and_evidence(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.csv"
    _dataset(source)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    app = QApplication.instance() or QApplication([])
    settings = QSettings("DRAM-CIM", "NeuroSimPlatform")
    settings.clear()
    settings.sync()
    widget = MacAnalysisWidget(ROOT)
    widget.path_edit.setText(str(source))
    widget.array_rows.setValue(16); widget.array_cols.setValue(32)
    widget.bit_width.setCurrentIndex(widget.bit_width.findData(4))
    widget.resize(1050, 760); widget.show(); widget.preview(); app.processEvents()
    assert widget.table.rowCount() == 20
    assert "'array_rows': 16" in widget.log.toPlainText()

    output = tmp_path / "fit"
    widget.start_fit(output)
    assert widget.worker is not None and widget.worker.wait(10_000)
    app.processEvents()
    assert {row["group"] for row in widget.last_results} == {"linear", "quadratic"}
    linear = next(row for row in widget.last_results if row["group"] == "linear")
    assert f"{linear['r2']:.4f}" == "1.0000"
    assert "R²: 1.0000" in widget.metrics_summary.text()
    assert (output / "mac_error_model.json").is_file()

    evidence = ROOT / "outputs" / "acceptance" / "gui_array_config"
    evidence.mkdir(parents=True, exist_ok=True)
    payload = {"selection": {"array_rows": 16, "array_cols": 32, "bit_width": 4},
               "groups": widget.last_results, "metrics_label": widget.metrics_summary.text()}
    (evidence / "array_config_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    assert widget.grab().save(str(evidence / "array_config_fit.png"))
    widget.close()

    restored = MacAnalysisWidget(ROOT)
    assert (restored.array_rows.value(), restored.array_cols.value(), restored.bit_width.currentData()) == (16, 32, 4)
    restored.close()
