"""Open the fitting page, run AOS1, and keep all generated plot tabs visible."""
from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QTabWidget


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLATFORM_ROOT.parent
sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.gui import CellFittingWidget, MainWindow, configure_windows_font


def main() -> int:
    application = QApplication.instance() or QApplication([])
    configure_windows_font(application)
    window = MainWindow(PLATFORM_ROOT)
    window.resize(1420, 980)
    cell = window.findChild(CellFittingWidget)
    if cell is None:
        raise RuntimeError("Device fitting page is unavailable")

    root_tabs = window.centralWidget()
    if isinstance(root_tabs, QTabWidget):
        root_tabs.setCurrentIndex(0)
        analysis_tabs = root_tabs.currentWidget()
        if isinstance(analysis_tabs, QTabWidget):
            for index in range(analysis_tabs.count()):
                page = analysis_tabs.widget(index)
                if page is cell or page.isAncestorOf(cell):
                    analysis_tabs.setCurrentIndex(index)
                    break

    cell.path_edit.setText(str((PROJECT_ROOT / "0_data" / "03_AOS_1.csv").resolve()))
    cell.output_edit.setText(str((PLATFORM_ROOT / "outputs" / "demo" / "aos1_gui" / "fit_result").resolve()))
    cell._refresh_mapping_columns()
    for combo in cell.mapping_edits.values():
        combo.setCurrentIndex(0)
    cell.encoding_combo.setCurrentIndex(cell.encoding_combo.findData("auto"))
    cell.delimiter_combo.setCurrentIndex(cell.delimiter_combo.findData("auto"))
    cell.missing_combo.setCurrentIndex(cell.missing_combo.findData("error"))
    cell.bits_combo.setCurrentIndex(cell.bits_combo.findData(8))
    cell.code_format_combo.setCurrentIndex(cell.code_format_combo.findData("auto"))
    cell.vd_unit_edit.setText("V")
    cell.vg_unit_edit.setText("V")
    cell.output_unit_edit.setText("A")

    window.show()
    window.raise_()
    window.activateWindow()
    QTimer.singleShot(400, cell.start_fit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
