"""Record a real UOS1 cell-fitting GUI walkthrough as an MP4 file."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLATFORM_ROOT.parent
sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.gui import CellFittingWidget, MainWindow, configure_windows_font


FPS = 12


def frame_from(window: MainWindow) -> np.ndarray:
    image = window.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    width, height = image.width(), image.height()
    rgba = np.frombuffer(image.bits(), dtype=np.uint8, count=width * height * 4)
    rgba = rgba.reshape((height, width, 4)).copy()
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def main() -> int:
    source = (PROJECT_ROOT / "0_data" / "03_UOS_1.csv").resolve()
    output = (PLATFORM_ROOT / "outputs" / "demo" / "uos1_gui").resolve()
    fit_output = output / "fit_result"
    video_path = output / "UOS1_GUI_demo.mp4"
    metadata_path = output / "UOS1_GUI_demo.json"
    output.mkdir(parents=True, exist_ok=True)

    application = QApplication.instance() or QApplication([])
    configure_windows_font(application)
    window = MainWindow(PLATFORM_ROOT)
    window.resize(1280, 900)
    window.show()
    window.raise_()
    window.activateWindow()
    application.processEvents()
    cell = window.findChild(CellFittingWidget)
    if cell is None:
        raise RuntimeError("Cell fitting page is unavailable")

    first = frame_from(window)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the MP4 output")

    def record(seconds: float) -> None:
        for _ in range(max(1, round(seconds * FPS))):
            application.processEvents()
            writer.write(frame_from(window))
            time.sleep(1 / FPS)

    try:
        window.statusBar().showMessage("UOS1 demo 1/4 - Open device error fitting")
        record(2.0)

        # UOS1 uses the standard aliases VD_Digital/VD/VG_Digital/VG/Iout.
        cell.path_edit.setText(str(source))
        cell.output_edit.setText(str(fit_output))
        for combo in cell.mapping_edits.values():
            combo.setCurrentIndex(0)
        for combo, value in (
            (cell.encoding_combo, "auto"),
            (cell.delimiter_combo, "auto"),
            (cell.missing_combo, "error"),
            (cell.bits_combo, 8),
            (cell.code_format_combo, "auto"),
        ):
            combo.setCurrentIndex(combo.findData(value))
        cell.vd_unit_edit.setText("V")
        cell.vg_unit_edit.setText("V")
        cell.output_unit_edit.setText("A")
        window.statusBar().showMessage("UOS1 demo 2/4 - 8-bit standard columns, units V/V/A")
        record(3.0)

        cell.preview()
        application.processEvents()
        if cell.table.rowCount() == 0:
            raise RuntimeError(cell.summary.text())
        window.statusBar().showMessage("UOS1 demo 3/4 - Preview validated; start background fitting")
        record(3.0)

        cell.start_fit(fit_output)
        deadline = time.monotonic() + 180.0
        while cell.worker is not None and cell.worker.isRunning():
            if time.monotonic() > deadline:
                raise TimeoutError("UOS1 fitting exceeded 180 seconds")
            application.processEvents()
            writer.write(frame_from(window))
            time.sleep(1 / FPS)
        application.processEvents()
        if cell.last_outcome is None:
            raise RuntimeError(cell.summary.text())
        window.statusBar().showMessage("UOS1 demo 4/4 - Fitting complete; model, metrics and plots saved")
        record(6.0)

        outcome = cell.last_outcome
        metadata = {
            "source": str(source),
            "rows": 65536,
            "code_bits": 8,
            "video_path": str(video_path),
            "error_model": str(outcome.error_model_path),
            "summary": str(outcome.summary_path),
            "plots": [str(path) for path in outcome.plot_paths],
            "interaction_metrics": outcome.metrics,
            "models": list(outcome.models),
            "video_spec": {"fps": FPS, "width": first.shape[1], "height": first.shape[0]},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        writer.release()
        window.close()

    print(json.dumps({"video": str(video_path), "metadata": str(metadata_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
