"""Launch the GUI and run the complete UOS1 LeNet-5 workflow once."""
from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QTabWidget


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLATFORM_ROOT.parent
sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.gui import ErrorAwareTrainingWidget, MainWindow, configure_windows_font


def main() -> int:
    error_model = (
        PLATFORM_ROOT
        / "outputs/demo/uos1_gui/fit_result/error_model/error_model.json"
    ).resolve()
    dataset_root = (PROJECT_ROOT / "data").resolve()
    output_dir = (PLATFORM_ROOT / "outputs/uos1_workflow").resolve()

    application = QApplication.instance() or QApplication([])
    application.setApplicationName("DRAM-CIM NeuroSim Platform")
    configure_windows_font(application)
    window = MainWindow(PLATFORM_ROOT)
    window.resize(1280, 900)

    training = window.findChild(ErrorAwareTrainingWidget)
    if training is None:
        raise RuntimeError("Error-aware training page is unavailable")
    tabs = window.centralWidget()
    if isinstance(tabs, QTabWidget):
        for index in range(tabs.count()):
            page = tabs.widget(index)
            if page is training or page.isAncestorOf(training):
                tabs.setCurrentIndex(index)
                break

    training.error_model_edit.setText(str(error_model))
    training.dataset_edit.setText(str(dataset_root))
    training.output_edit.setText(str(output_dir))
    training.lambda_spin.setValue(1.0)
    training.epoch_spin.setValue(1)
    training.batch_spin.setValue(128)
    training.device_combo.setCurrentText("cuda")

    window.statusBar().showMessage(
        "UOS1: baseline inference → signed-differential error inference → "
        "error-aware training → post-training evaluation"
    )
    window.show()
    window.raise_()
    window.activateWindow()
    QTimer.singleShot(500, lambda: training.start_training(output_dir))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
