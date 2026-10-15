from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from PySide6.QtCore import QEvent, QObject, QSize, QSettings, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox as _QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from .application import (
    CellFitOutcome, CellImportOptions, MacFitOutcome, MacImportOptions, PlatformService,
    PpaOutcome, SingleInferenceOutcome,
)
from .batch_inference import (
    BatchInferenceCancelled, BatchInferenceRequest, GroupedBatchInferenceOutcome,
    run_grouped_batch_inference,
)
from .model_registry import builtin_registry
from .gpt2_service import (
    GPT2GenerationCancelled, GPT2GenerationOutcome, GPT2GenerationRequest,
    GPT2GenerationService,
)
from .mac_error_model import MacErrorModel, load_mac_csv
from .paths import data_root
from .training_service import (
    ErrorAwareTrainingService, TrainingCancelled, TrainingOutcome, TrainingRequest,
)
from .workflow import ErrorModelRegistry, WorkflowContext


def _high_dpi_pixmap(
    path: str | Path,
    logical_size: QSize,
    widget: QWidget,
    *,
    smooth: bool = True,
) -> QPixmap:
    """Scale a source image at the monitor's physical pixel density.

    Qt widget sizes are expressed in logical pixels. Scaling directly to that
    size produces a visibly soft image on 125%-200% Windows displays, so the
    pixmap is rendered at the corresponding physical size and tagged with its
    device-pixel ratio before it is assigned to a label.
    """

    source = QPixmap(str(path))
    if source.isNull():
        return source
    ratio = max(1.0, float(widget.devicePixelRatioF()))
    physical = QSize(
        max(1, round(logical_size.width() * ratio)),
        max(1, round(logical_size.height() * ratio)),
    )
    mode = (
        Qt.TransformationMode.SmoothTransformation
        if smooth
        else Qt.TransformationMode.FastTransformation
    )
    rendered = source.scaled(physical, Qt.AspectRatioMode.KeepAspectRatio, mode)
    rendered.setDevicePixelRatio(ratio)
    return rendered


class QComboBox(_QComboBox):
    """A click-open combo box whose selection never changes by wheel scrolling."""

    def setEditable(self, editable: bool) -> None:
        old_editor = self.lineEdit()
        if old_editor is not None:
            old_editor.removeEventFilter(self)
        super().setEditable(editable)
        editor = self.lineEdit()
        if editor is not None:
            editor.installEventFilter(self)

    def wheelEvent(self, event: Any) -> None:
        # Let the surrounding page handle the wheel instead of changing a
        # parameter merely because the pointer is over its selector.
        event.ignore()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.showPopup()
            event.accept()
            return
        super().mousePressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.lineEdit():
            if event.type() == QEvent.Type.Wheel:
                event.ignore()
                return False
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton and self.isEnabled()):
                self.setFocus(Qt.FocusReason.MouseFocusReason)
                QTimer.singleShot(0, self.showPopup)
        return super().eventFilter(watched, event)


LOGICAL_COLUMNS = (
    ("array_size", "阵列规模列"),
    ("array_rows", "阵列行数列"),
    ("array_cols", "阵列列数列"),
    ("bit_width", "位宽列"),
    ("ideal_mac", "理想 MAC 列"),
    ("measured_output", "测量输出列"),
    ("group", "实验组列（可选）"),
)


class WorkflowCoordinator(QObject):
    """Share the fitted-error and inference context between all GUI stages."""

    models_changed = Signal()
    inference_changed = Signal()
    selection_changed = Signal(str)
    network_changed = Signal(str)

    def __init__(self, platform_root: Path):
        super().__init__()
        state_root = Path(platform_root).resolve() / "outputs" / "workflow"
        self.registry = ErrorModelRegistry(state_root / "error_model_registry.json")
        self.context = WorkflowContext(state_root / "workflow_context.json")
        self.model_registry = builtin_registry(Path(platform_root).resolve().parent)

    def models(self) -> tuple[dict[str, Any], ...]:
        return self.registry.models(selectable_only=True)

    def register_fit(self, outcome: CellFitOutcome) -> dict[str, Any]:
        result = self.registry.register_fitting_run(
            error_model_path=outcome.error_model_path,
            fitting_summary_path=outcome.summary_path,
        )
        selectable = [item for item in result["models"] if item.get("selectable")]
        if selectable:
            empirical = next(
                (item for item in selectable if item.get("method") == "empirical_interaction"),
                selectable[0],
            )
            self.context.select_error_model(self.registry.path, empirical["id"])
        self.models_changed.emit()
        if selectable:
            self.selection_changed.emit(str(empirical["id"]))
        self.inference_changed.emit()
        return result

    def selected_error_model_id(self) -> str | None:
        fitting = self.context.load().get("fitting")
        return str(fitting["selected_model_id"]) if fitting else None

    def select_error_model(self, model_id: str) -> dict[str, Any]:
        result = self.context.select_error_model(self.registry.path, model_id)
        self.selection_changed.emit(model_id)
        self.inference_changed.emit()
        return result

    def record_inference(self, **values: Any) -> dict[str, Any]:
        result = self.context.record_inference(**values)
        self.inference_changed.emit()
        return result

    def selected_network_model_id(self) -> str:
        selected = self.context.selected_network_model()
        if selected:
            try:
                self.model_registry.get(str(selected["model_id"]))
                return str(selected["model_id"])
            except KeyError:
                pass
        ready = [item for item in self.model_registry.list() if item.checkpoint_available]
        return (ready[0] if ready else self.model_registry.list()[0]).model_id

    def select_network_model(self, model_id: str) -> dict[str, Any]:
        spec = self.model_registry.get(model_id)
        if not spec.checkpoint_available:
            raise FileNotFoundError(f"模型 checkpoint 未安装：{spec.checkpoint}")
        result = self.context.select_network_model(
            model_id=spec.model_id, task=spec.task,
            dataset_name=spec.datasets[0] if spec.datasets else "",
        )
        self.network_changed.emit(spec.model_id)
        self.inference_changed.emit()
        return result


def _fill_error_model_combo(
    combo: QComboBox, coordinator: WorkflowCoordinator, preferred_id: str | None = None
) -> None:
    selected = preferred_id or coordinator.selected_error_model_id()
    combo.blockSignals(True)
    combo.clear()
    models = coordinator.models()
    if not models:
        combo.addItem("请先完成误差拟合", None)
        combo.setEnabled(False)
    else:
        combo.setEnabled(True)
        for item in models:
            metrics = item.get("metrics", {})
            r2 = metrics.get("r2_to_ideal_lut")
            suffix = f" | 相对理想 R²={r2:.6f}" if isinstance(r2, (int, float)) else ""
            combo.addItem(f"{item['name']}{suffix}", item["id"])
        index = combo.findData(selected)
        combo.setCurrentIndex(index if index >= 0 else 0)
    combo.blockSignals(False)


def _selected_error_model(
    combo: QComboBox, coordinator: WorkflowCoordinator
) -> tuple[str, Path]:
    model_id = combo.currentData()
    if not model_id:
        raise RuntimeError("请先完成误差拟合，并选择一种误差模型")
    model = coordinator.registry.resolve(str(model_id))
    return str(model_id), Path(model["path"]).resolve()


def configure_windows_font(application: QApplication) -> None:
    """Load a CJK-capable system font even with Qt's offscreen Windows plugin."""
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            application.setFont(QFont(families[0], 9))


APPLICATION_STYLE_SHEET = """
QMainWindow, QWidget {
    background-color: #F4F7FB;
    color: #172033;
}
QWidget#pageContent, QScrollArea#pageScroll > QWidget > QWidget {
    background-color: #F4F7FB;
}
QLabel {
    background: transparent;
}
QTabWidget::pane {
    border: 1px solid #D7DFEA;
    border-radius: 7px;
    background: #FFFFFF;
    top: -1px;
}
QTabBar::tab {
    min-height: 32px;
    padding: 0 15px;
    margin-right: 3px;
    border: 1px solid #D7DFEA;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    background: #EAF0F7;
    color: #526077;
}
QTabBar::tab:hover {
    background: #E0EAF7;
    color: #1D4ED8;
}
QTabBar::tab:selected {
    background: #FFFFFF;
    color: #1D4ED8;
    font-weight: 600;
    border-top: 2px solid #2563EB;
}
QTabWidget#analysisTabs QTabBar::tab {
    min-height: 29px;
    padding: 0 13px;
    background: #F4F7FB;
}
QGroupBox {
    margin-top: 13px;
    padding: 13px 10px 10px 10px;
    border: 1px solid #D7DFEA;
    border-radius: 8px;
    background-color: #FFFFFF;
    font-weight: 600;
    color: #26344D;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #31517D;
    background-color: #FFFFFF;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    min-height: 28px;
    padding: 2px 8px;
    border: 1px solid #C8D2E0;
    border-radius: 5px;
    background-color: #FFFFFF;
    selection-background-color: #BFDBFE;
    selection-color: #172033;
}
QPlainTextEdit {
    padding: 7px;
    font-family: "Cascadia Mono", "Consolas", "Microsoft YaHei UI";
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border: 1px solid #3B82F6;
}
QLineEdit:read-only {
    background-color: #F0F4F8;
    color: #536176;
}
QComboBox::drop-down {
    width: 26px;
    border: none;
    border-left: 1px solid #E1E7EF;
}
QPushButton {
    min-height: 29px;
    padding: 2px 13px;
    border: 1px solid #B9C6D8;
    border-radius: 5px;
    background-color: #FFFFFF;
    color: #26344D;
}
QPushButton:hover {
    background-color: #EFF6FF;
    border-color: #7EACEC;
    color: #1D4ED8;
}
QPushButton:pressed {
    background-color: #DBEAFE;
}
QPushButton[role="primary"] {
    background-color: #2563EB;
    border-color: #2563EB;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background-color: #1D4ED8;
    border-color: #1D4ED8;
    color: #FFFFFF;
}
QPushButton[role="danger"] {
    color: #B42318;
    border-color: #F0B5AE;
    background-color: #FFF7F6;
}
QPushButton:disabled, QLineEdit:disabled, QComboBox:disabled {
    color: #9AA6B6;
    background-color: #EEF2F6;
    border-color: #D9E0E8;
}
QTableWidget {
    alternate-background-color: #F6F9FD;
    background-color: #FFFFFF;
    border: 1px solid #D5DEEA;
    border-radius: 6px;
    gridline-color: #E5EBF2;
    selection-background-color: #DBEAFE;
    selection-color: #172033;
}
QHeaderView::section {
    min-height: 31px;
    padding: 4px 7px;
    border: none;
    border-right: 1px solid #D7DFEA;
    border-bottom: 1px solid #CAD5E3;
    background-color: #EAF0F7;
    color: #34445F;
    font-weight: 600;
}
QProgressBar {
    min-height: 15px;
    border: 1px solid #CFD8E5;
    border-radius: 7px;
    background: #E8EEF5;
    text-align: center;
    color: #34445F;
}
QProgressBar::chunk {
    border-radius: 6px;
    background-color: #2F80ED;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    width: 12px;
    margin: 2px;
    background: transparent;
}
QScrollBar::handle:vertical {
    min-height: 32px;
    border-radius: 5px;
    background: #B8C5D6;
}
QScrollBar::handle:vertical:hover {
    background: #8FA3BC;
}
QScrollBar:horizontal {
    height: 12px;
    margin: 2px;
    background: transparent;
}
QScrollBar::handle:horizontal {
    min-width: 32px;
    border-radius: 5px;
    background: #B8C5D6;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}
QStatusBar {
    border-top: 1px solid #D7DFEA;
    background: #EAF0F7;
    color: #526077;
}
QSplitter::handle {
    background-color: #D7DFEA;
}
QSplitter::handle:hover {
    background-color: #93B4DE;
}
QToolTip {
    padding: 5px 7px;
    border: 1px solid #94A3B8;
    background-color: #1E293B;
    color: #FFFFFF;
}
"""


def apply_application_style(application: QApplication) -> None:
    """Apply the shared light desktop theme without replacing any widgets."""
    application.setStyle("Fusion")
    application.setStyleSheet(APPLICATION_STYLE_SHEET)


class MacFitThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, service: PlatformService, source: Path, output: Path, options: MacImportOptions,
                 array_rows: int | None = None, array_cols: int | None = None,
                 bit_width: int | None = None):
        super().__init__()
        self.service = service
        self.source = source
        self.output = output
        self.options = options
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.bit_width = bit_width

    def run(self) -> None:
        try:
            frame, source = load_mac_csv(
                self.source, column_mapping=self.options.column_mapping or None,
                encoding=self.options.encoding, delimiter=self.options.delimiter,
                missing=self.options.missing, array_rows=self.array_rows,
                array_cols=self.array_cols, bit_width=self.bit_width,
            )
            model = MacErrorModel.fit(
                frame, source=source, polynomial_degree=self.options.polynomial_degree
            )
            model_path = model.save(self.output)
            summary = tuple({
                "array": f"{item.array_rows}x{item.array_cols}",
                "bit_width": item.bit_width, "group": item.group, "samples": item.samples,
                "r2": item.linear_metrics.r2,
                "nonlinearity_percent_fs": item.linear_metrics.nonlinearity_percent_fs,
                "rmse": item.linear_metrics.rmse,
                "polynomial_r2": item.polynomial_metrics.r2,
            } for item in model.groups)
            self.succeeded.emit(MacFitOutcome(model_path, summary))
        except Exception:
            self.failed.emit(traceback.format_exc())


class CellFitThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progressed = Signal(int, str)

    def __init__(self, service: PlatformService, source: Path, output: Path, options: CellImportOptions):
        super().__init__()
        self.service = service
        self.source = source
        self.output = output
        self.options = options

    def run(self) -> None:
        try:
            self.succeeded.emit(self.service.fit_cell(
                self.source, self.output, self.options,
                progress=lambda percent, message: self.progressed.emit(percent, message),
            ))
        except Exception:
            self.failed.emit(traceback.format_exc())


class InferenceThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, service: PlatformService, project_root: Path, model_id: str,
                 sample_index: int, dataset_root: Path, checkpoint: Path,
                 auxiliary: Path | None, error_model: Path, device: str):
        super().__init__()
        self.service = service
        self.project_root = project_root
        self.model_id = model_id
        self.sample_index = sample_index
        self.dataset_root = dataset_root
        self.checkpoint = checkpoint
        self.auxiliary = auxiliary
        self.error_model = error_model
        self.device = device

    def run(self) -> None:
        try:
            self.succeeded.emit(
                self.service.infer_registered_single(
                    self.project_root, self.model_id, self.sample_index,
                    dataset_root=self.dataset_root, checkpoint_path=self.checkpoint,
                    auxiliary_path=self.auxiliary, error_model_path=self.error_model,
                    device=self.device,
                )
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class BatchInferenceThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    progressed = Signal(int, int)

    def __init__(self, request: BatchInferenceRequest):
        super().__init__()
        self.request = request
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            outcome = run_grouped_batch_inference(
                self.request,
                progress=lambda done, total: self.progressed.emit(done, total),
                cancelled=lambda: self._cancel_requested,
            )
            self.succeeded.emit(outcome)
        except BatchInferenceCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PpaThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, service: PlatformService, project_root: Path, checkpoint: Path,
                 error_model: Path, output_dir: Path, subarray_rows: int, parallel_rows: int):
        super().__init__()
        self.service = service
        self.project_root = project_root
        self.checkpoint = checkpoint
        self.error_model = error_model
        self.output_dir = output_dir
        self.subarray_rows = subarray_rows
        self.parallel_rows = parallel_rows

    def run(self) -> None:
        try:
            self.succeeded.emit(self.service.evaluate_lenet5_ppa(
                self.project_root, self.checkpoint, self.error_model, self.output_dir,
                subarray_rows=self.subarray_rows, parallel_rows=self.parallel_rows,
            ))
        except Exception:
            self.failed.emit(traceback.format_exc())


class TrainingThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    progressed = Signal(int, str, object)

    def __init__(self, service: ErrorAwareTrainingService, request: TrainingRequest):
        super().__init__()
        self.service = service
        self.request = request
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            outcome = self.service.run(
                self.request,
                progress=lambda percent, message, metrics=None: self.progressed.emit(
                    percent, message, metrics
                ),
                cancel=lambda: self._cancel_requested,
            )
            self.succeeded.emit(outcome)
        except TrainingCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())


class GPT2GenerationThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    progressed = Signal(int, str)

    def __init__(self, service: GPT2GenerationService, request: GPT2GenerationRequest):
        super().__init__()
        self.service = service
        self.request = request
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            outcome = self.service.run(
                self.request,
                progress=lambda percent, message: self.progressed.emit(percent, message),
                cancelled=lambda: self._cancel_requested,
            )
            self.succeeded.emit(outcome)
        except GPT2GenerationCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())


class GPT2BatchThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    progressed = Signal(int, str)

    def __init__(self, service: GPT2GenerationService,
                 requests: tuple[GPT2GenerationRequest, ...]):
        super().__init__()
        self.service = service
        self.requests = requests
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            outcomes = []
            total = len(self.requests)
            for index, request in enumerate(self.requests):
                if self._cancel_requested:
                    raise GPT2GenerationCancelled("GPT-2 批量推理已取消")
                outcome = self.service.run(
                    request,
                    progress=lambda percent, message, i=index: self.progressed.emit(
                        round(100 * (i + percent / 100) / total),
                        f"提示词 {i + 1}/{total}：{message}",
                    ),
                    cancelled=lambda: self._cancel_requested,
                )
                outcomes.append(outcome)
            self.succeeded.emit(tuple(outcomes))
        except GPT2GenerationCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())


class CellFittingWidget(QWidget):
    """Import transistor measurements, fit them, and create a reusable error model."""

    MAPPING_FIELDS = (
        ("X", "输入数字码（X）"), ("W", "权重数字码（W）"),
        ("VD", "输入电压（VD）"), ("VG", "栅极电压（VG）"),
        ("output", "实测输出"),
    )

    def __init__(self, platform_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None):
        super().__init__(parent)
        application = QApplication.instance()
        if application is not None:
            configure_windows_font(application)
        self.platform_root = Path(platform_root).resolve()
        self.coordinator = coordinator
        self.service = PlatformService()
        self.settings = QSettings("DRAM-CIM", "NeuroSimPlatform")
        self.worker: CellFitThread | None = None
        self.last_outcome: CellFitOutcome | None = None
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.path_edit = QLineEdit()
        self.path_edit.editingFinished.connect(self._refresh_mapping_columns)
        source_row = QHBoxLayout(); source_row.addWidget(self.path_edit)
        browse = QPushButton("选择 CSV"); browse.clicked.connect(self._browse_source)
        source_row.addWidget(browse); form.addRow("测量数据 CSV", source_row)
        self.encoding_combo = QComboBox()
        for label, value in (("自动识别", "auto"), ("UTF-8", "utf-8-sig"), ("GBK", "gbk")):
            self.encoding_combo.addItem(label, value)
        form.addRow("文件编码", self.encoding_combo)
        self.delimiter_combo = QComboBox()
        for label, value in (("自动识别", "auto"), ("逗号（,）", ","), ("分号（;）", ";")):
            self.delimiter_combo.addItem(label, value)
        form.addRow("分隔符", self.delimiter_combo)
        self.encoding_combo.currentIndexChanged.connect(self._refresh_mapping_columns)
        self.delimiter_combo.currentIndexChanged.connect(self._refresh_mapping_columns)
        self.missing_combo = QComboBox()
        for label, value in (("遇到缺失值时报错", "error"), ("删除不完整行", "drop"), ("插值补全", "interpolate")):
            self.missing_combo.addItem(label, value)
        form.addRow("缺失值处理", self.missing_combo)
        self.bits_combo = QComboBox()
        for bits in (1, 2, 4, 8): self.bits_combo.addItem(f"{bits} 位（{1 << bits} 级）", bits)
        self.bits_combo.setCurrentIndex(3); form.addRow("采集位宽", self.bits_combo)
        self.code_format_combo = QComboBox()
        for label, value in (("自动识别", "auto"), ("十进制", "decimal"), ("二进制", "binary")):
            self.code_format_combo.addItem(label, value)
        form.addRow("数字码格式", self.code_format_combo); layout.addLayout(form)

        mapping_box = QGroupBox("CSV 列映射（从表头选择，或使用自动识别）")
        mapping_form = QFormLayout(mapping_box); self.mapping_edits: dict[str, QComboBox] = {}
        for key, label in self.MAPPING_FIELDS:
            combo = QComboBox(); combo.setEditable(True)
            combo.addItem("自动识别／未指定", "")
            combo.lineEdit().setPlaceholderText(key)
            self.mapping_edits[key] = combo
            mapping_form.addRow(label, combo)
        layout.addWidget(mapping_box)
        units = QHBoxLayout()
        self.vd_unit_edit = QLineEdit("V"); self.vg_unit_edit = QLineEdit("V"); self.output_unit_edit = QLineEdit("A")
        units.addWidget(QLabel("VD 单位")); units.addWidget(self.vd_unit_edit)
        units.addWidget(QLabel("VG 单位")); units.addWidget(self.vg_unit_edit)
        units.addWidget(QLabel("输出单位")); units.addWidget(self.output_unit_edit); layout.addLayout(units)
        self.output_edit = QLineEdit(); output_row = QHBoxLayout(); output_row.addWidget(self.output_edit)
        output_button = QPushButton("选择目录"); output_button.clicked.connect(self._browse_output); output_row.addWidget(output_button)
        output_form = QFormLayout(); output_form.addRow("输出目录", output_row); layout.addLayout(output_form)
        controls = QHBoxLayout(); self.preview_button = QPushButton("预览并校验")
        self.preview_button.clicked.connect(self.preview); self.fit_button = QPushButton("开始拟合并生成误差模型")
        self.fit_button.clicked.connect(self.start_fit); controls.addWidget(self.preview_button); controls.addWidget(self.fit_button)
        controls.addStretch(); layout.addLayout(controls)
        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.summary = QLabel("就绪"); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        splitter = QSplitter(); self.table = QTableWidget(); self.table.setAlternatingRowColors(True)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); splitter.addWidget(self.table); splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 2); layout.addWidget(splitter, 1)
        self.plot_tabs = QTabWidget(); self.plot_tabs.setMinimumHeight(300); self.plot_tabs.setVisible(False)
        layout.addWidget(self.plot_tabs)

    def _options(self) -> CellImportOptions:
        mapping = {}
        for key, combo in self.mapping_edits.items():
            value = self._mapping_value(combo)
            if value:
                mapping[key] = value
        return CellImportOptions(mapping, str(self.encoding_combo.currentData()), str(self.delimiter_combo.currentData()),
            str(self.missing_combo.currentData()), int(self.bits_combo.currentData()), str(self.code_format_combo.currentData()),
            {"VD": self.vd_unit_edit.text().strip() or "V", "VG": self.vg_unit_edit.text().strip() or "V",
             "output": self.output_unit_edit.text().strip() or "A"})

    def _source(self) -> Path:
        if not self.path_edit.text().strip(): raise ValueError("请选择测量数据 CSV")
        return Path(self.path_edit.text().strip()).expanduser().resolve()

    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择测量数据 CSV", self.path_edit.text(), "CSV 文件 (*.csv)")
        if path:
            self.path_edit.setText(path)
            self._refresh_mapping_columns()

    @staticmethod
    def _mapping_value(combo: QComboBox) -> str:
        text = combo.currentText().strip()
        return "" if text == "自动识别／未指定" else text

    @staticmethod
    def _set_mapping_value(combo: QComboBox, value: str) -> None:
        value = value.strip()
        if not value:
            combo.setCurrentIndex(0)
            return
        index = combo.findData(value)
        if index < 0:
            # Preserve a manually typed value only before a CSV header has
            # been loaded. Once choices exist, an absent value is stale.
            if combo.count() == 1:
                combo.setEditText(value)
            else:
                combo.setCurrentIndex(0)
            return
        combo.setCurrentIndex(index)

    def _refresh_mapping_columns(self) -> None:
        text = self.path_edit.text().strip()
        if not text or not Path(text).expanduser().is_file():
            return
        source = Path(text).expanduser().resolve()
        source_changed = str(source) != getattr(self, "_mapping_source", "")
        previous = {key: self._mapping_value(combo) for key, combo in self.mapping_edits.items()}
        try:
            columns, encoding, delimiter = self.service.inspect_cell_columns(
                text,
                encoding=str(self.encoding_combo.currentData()),
                delimiter=str(self.delimiter_combo.currentData()),
            )
        except Exception as exc:
            self.log.appendPlainText("无法读取 CSV 表头：" + str(exc))
            return
        suggestions = {
            "X": next((name for name in ("VD_Digital", "input_code_dec", "X", "input_code") if name in columns), ""),
            "W": next((name for name in ("VG_Digital", "weight_code_dec", "W", "weight_code") if name in columns), ""),
            "VD": next((name for name in ("VD", "drain_voltage") if name in columns), ""),
            "VG": next((name for name in ("VG", "gate_voltage") if name in columns), ""),
            "output": next((name for name in ("Iout", "Ianalog", "I", "Iout_Analog", "ideal_product_dec", "output") if name in columns), ""),
        }
        reset_fields = []
        available = set(columns)
        for key, combo in self.mapping_edits.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("自动识别／未指定", "")
            for column in columns:
                combo.addItem(column, column)
            value = suggestions[key] if source_changed and suggestions[key] else (
                previous[key] if previous[key] in available else ""
            )
            if previous[key] and not value:
                reset_fields.append(key)
            self._set_mapping_value(combo, value)
            combo.blockSignals(False)
        shown_delimiter = "逗号" if delimiter == "," else "分号" if delimiter == ";" else delimiter
        self.log.appendPlainText(
            f"已读取 CSV 表头（{len(columns)} 列，编码 {encoding}，{shown_delimiter}分隔）"
        )
        if reset_fields:
            self.log.appendPlainText(
                "以下旧列映射已恢复为自动识别：" + ", ".join(reset_fields)
            )
        if source_changed and suggestions["X"] and suggestions["W"] and suggestions["output"]:
            bits = self._infer_code_bits(
                source, suggestions["X"], suggestions["W"], encoding, delimiter
            )
            if bits is not None:
                self.bits_combo.setCurrentIndex(self.bits_combo.findData(bits))
                code_format = "binary" if suggestions["X"].endswith("_Digital") else "decimal"
                self.code_format_combo.setCurrentIndex(self.code_format_combo.findData(code_format))
            self.log.appendPlainText(
                "已应用标准列映射"
                + (f"，并识别为 {bits} 位数字码" if bits is not None else "")
            )
            self._standard_mapping_applied = True
        elif source_changed:
            self._standard_mapping_applied = False
        self._mapping_source = str(source)

    @staticmethod
    def _infer_code_bits(
        source: Path, x_column: str, w_column: str, encoding: str, delimiter: str
    ) -> int | None:
        """Infer 1/2/4/8-bit codes from the two selected CSV code columns."""
        allowed = (1, 2, 4, 8)
        binary = x_column.endswith("_Digital") or w_column.endswith("_Digital")
        maximum = 0
        max_width = 0
        try:
            with source.open("r", encoding=encoding, newline="") as stream:
                reader = csv.DictReader(stream, delimiter=delimiter)
                for row in reader:
                    for column in (x_column, w_column):
                        value = str(row[column]).strip()
                        if binary:
                            if not value or any(char not in "01" for char in value):
                                return None
                            max_width = max(max_width, len(value))
                        else:
                            maximum = max(maximum, int(float(value)))
        except (OSError, KeyError, ValueError, csv.Error):
            return None
        required = max_width if binary else max(1, maximum.bit_length())
        return next((bits for bits in allowed if required <= bits), None)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path: self.output_edit.setText(path)

    def _show_records(self, records: list[dict[str, Any]]) -> None:
        columns = list(records[0]) if records else []; self.table.setColumnCount(len(columns)); self.table.setRowCount(len(records))
        header_names = {
            "model": "模型／方法", "equation": "拟合方程", "r2": "决定系数 R²",
            "rmse": "均方根误差", "nl_percent_fs": "满量程非线性度（%）",
            "X": "输入数字码 X", "W": "权重数字码 W", "VD": "输入电压 VD",
            "VG": "栅极电压 VG", "output": "实测输出",
        }
        self.table.setHorizontalHeaderLabels([header_names.get(column, column) for column in columns])
        for row, record in enumerate(records):
            for column, key in enumerate(columns): self.table.setItem(row, column, QTableWidgetItem(str(record[key])))
        self.table.resizeColumnsToContents()

    def preview(self) -> None:
        try:
            self._persist_settings(); result = self.service.preview_cell(self._source(), self._options())
            self._show_records(list(result.records)); self.summary.setText(
                f"校验通过：{result.rows} 行｜{result.metadata['code_bits']} 位｜编码 {result.metadata['encoding']}｜分隔符 {result.metadata['delimiter']}")
            self.log.appendPlainText("列映射：" + str(result.metadata["column_mapping"])); self.progress.setValue(20)
        except Exception as exc: self._show_error("预览失败", exc)

    def start_fit(self, output_dir: Path | None = None) -> None:
        if self.worker is not None and self.worker.isRunning(): return
        try:
            self._persist_settings(); source = self._source()
            target = Path(output_dir).resolve() if output_dir else (Path(self.output_edit.text()).expanduser().resolve()
                if self.output_edit.text().strip() else self.platform_root / "outputs" / "gui_runs" /
                datetime.now().strftime("%Y%m%d-%H%M%S") / "cell_fit")
            self.output_edit.setText(str(target)); self.last_outcome = None
            self.fit_button.setEnabled(False); self.preview_button.setEnabled(False); self.progress.setValue(0)
            self.summary.setText("正在后台拟合……"); self.worker = CellFitThread(self.service, source, target, self._options())
            self.worker.progressed.connect(self._fit_progressed); self.worker.succeeded.connect(self._fit_succeeded)
            self.worker.failed.connect(self._fit_failed); self.worker.finished.connect(self._fit_finished); self.worker.start()
        except Exception as exc: self._show_error("无法开始拟合", exc)

    def _fit_progressed(self, percent: int, message: str) -> None:
        translations = {
            "Loading and validating data": "正在加载并校验数据",
            "Fitting code-voltage calibration": "正在拟合数字码—电压标定关系",
            "Fitting linear product model": "正在拟合线性乘积模型",
            "Fitting quadratic response surface": "正在拟合二次响应曲面",
            "Fitting complete transfer surface": "正在拟合完整传输曲面",
            "Extracting interaction current and generating LUT": "正在提取交互电流并生成查找表",
            "Exporting fitting results": "正在导出拟合结果",
            "Fitting completed": "拟合完成",
        }
        self.progress.setValue(percent)
        self.log.appendPlainText(f"{percent}% {translations.get(message, message)}")

    def _fit_succeeded(self, outcome: CellFitOutcome) -> None:
        self.last_outcome = outcome
        shown_models = list(outcome.models)
        self.summary.setText(f"误差模型：{outcome.error_model_path}｜交互项 R²={outcome.metrics['r2']:.6f}｜非线性度={outcome.metrics['nl_percent_fs']:.6f}% FS")
        self.log.appendPlainText("拟合摘要：" + str(outcome.summary_path))
        if self.coordinator is not None:
            registered = self.coordinator.register_fit(outcome)
            selectable = [item for item in registered["models"] if item.get("selectable")]
            for item in selectable:
                metrics = item.get("metrics", {})
                shown_models.append({
                    "model": item["name"],
                    "equation": "可部署正负差分 LUT（后续页面下拉可选）",
                    "r2": metrics.get("r2_to_ideal_lut"),
                    "rmse": metrics.get("rmse_codes_to_ideal_lut"),
                    "nl_percent_fs": metrics.get("nl_percent_fs_to_ideal_lut"),
                })
                self.log.appendPlainText(f"可选误差模型：{item['method']} → {item['path']}")
            self.log.appendPlainText(
                f"已生成 {len(selectable)} 种可选误差模型；注册表："
                + str(registered["registry_path"])
            )
            self.summary.setText(
                f"拟合完成：已生成 {len(selectable)} 种可选误差模型。后续推理页面可通过下拉框选择。"
            )
        self._show_records(shown_models)
        self.plot_tabs.clear()
        plot_names = {
            "surface_overview": "响应曲面总览", "fit_quality": "拟合质量",
            "ideal_product_vs_device": "理想乘积与器件输出", "transfer_slices": "传输切片",
            "lut_error": "查找表误差", "normalized_linear": "归一化线性拟合",
            "normalized_surface": "归一化曲面拟合", "normalized_lut": "归一化查找表",
        }
        for path in outcome.plot_paths:
            self.log.appendPlainText("图表：" + str(path))
            pixmap = _high_dpi_pixmap(path, QSize(920, 285), self)
            if not pixmap.isNull():
                label = QLabel(); label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setPixmap(pixmap)
                label.setToolTip(str(path)); self.plot_tabs.addTab(label, plot_names.get(path.stem, path.stem.replace("_", " ")))
        self.plot_tabs.setVisible(self.plot_tabs.count() > 0)

    def _fit_failed(self, details: str) -> None:
        self.log.appendPlainText(details); self.summary.setText("拟合失败：" + details.splitlines()[-1])

    def _fit_finished(self) -> None:
        self.fit_button.setEnabled(True); self.preview_button.setEnabled(True); self.progress.setValue(100 if self.last_outcome else 0)

    def _show_error(self, title: str, error: Exception) -> None:
        message = f"{title}: {error}"; self.summary.setText(message); self.log.appendPlainText(message)

    def _restore_settings(self) -> None:
        saved_source = str(self.settings.value("cell/source", "")).strip()
        bundled_source = self.platform_root.parent / "0_data" / "03_AOS_1.csv"
        if (not saved_source or not Path(saved_source).expanduser().is_file()) and bundled_source.is_file():
            saved_source = str(bundled_source)
        self.path_edit.setText(saved_source); self.output_edit.setText(str(self.settings.value("cell/output", "")))
        for combo, key, fallback in ((self.encoding_combo, "cell/encoding", "auto"), (self.delimiter_combo, "cell/delimiter", "auto"),
                                     (self.missing_combo, "cell/missing", "error"), (self.code_format_combo, "cell/code_format", "auto")):
            combo.setCurrentIndex(max(0, combo.findData(self.settings.value(key, fallback))))
        self.bits_combo.setCurrentIndex(max(0, self.bits_combo.findData(int(self.settings.value("cell/bits", 8)))))
        self._refresh_mapping_columns()
        if not getattr(self, "_standard_mapping_applied", False):
            for key, combo in self.mapping_edits.items():
                self._set_mapping_value(combo, str(self.settings.value(f"cell/mapping/{key}", "")))
        self.vd_unit_edit.setText(str(self.settings.value("cell/unit/VD", "V"))); self.vg_unit_edit.setText(str(self.settings.value("cell/unit/VG", "V")))
        self.output_unit_edit.setText(str(self.settings.value("cell/unit/output", "A")))

    def _persist_settings(self) -> None:
        self.settings.setValue("cell/source", self.path_edit.text().strip()); self.settings.setValue("cell/output", self.output_edit.text().strip())
        self.settings.setValue("cell/encoding", self.encoding_combo.currentData()); self.settings.setValue("cell/delimiter", self.delimiter_combo.currentData())
        self.settings.setValue("cell/missing", self.missing_combo.currentData()); self.settings.setValue("cell/bits", self.bits_combo.currentData())
        self.settings.setValue("cell/code_format", self.code_format_combo.currentData())
        for key, combo in self.mapping_edits.items():
            self.settings.setValue(f"cell/mapping/{key}", self._mapping_value(combo))
        self.settings.setValue("cell/unit/VD", self.vd_unit_edit.text().strip()); self.settings.setValue("cell/unit/VG", self.vg_unit_edit.text().strip())
        self.settings.setValue("cell/unit/output", self.output_unit_edit.text().strip())


class MacAnalysisWidget(QWidget):
    """Import, select and fit one physical array configuration."""

    def __init__(self, platform_root: Path, parent: QWidget | None = None):
        super().__init__(parent)
        application = QApplication.instance()
        if application is not None:
            configure_windows_font(application)
        self.platform_root = Path(platform_root)
        self.service = PlatformService()
        self.settings = QSettings("DRAM-CIM", "NeuroSimPlatform")
        self.settings.setFallbacksEnabled(False)
        self.worker: MacFitThread | None = None
        self.last_results: list[dict[str, Any]] = []
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        source_group = QGroupBox("Array MAC data")
        source_layout = QGridLayout(source_group)
        self.path_edit = QLineEdit()
        self.path_edit.textChanged.connect(self._source_changed)
        browse = QPushButton("Select CSV")
        browse.clicked.connect(self._browse_source)
        source_layout.addWidget(QLabel("File"), 0, 0)
        source_layout.addWidget(self.path_edit, 0, 1)
        source_layout.addWidget(browse, 0, 2)
        self.encoding = QComboBox()
        for label, value in (("Auto", None), ("UTF-8", "utf-8-sig"), ("GBK", "gb18030")):
            self.encoding.addItem(label, value)
        self.delimiter = QComboBox()
        for label, value in (("Auto", None), ("Comma", ","), ("Semicolon", ";")):
            self.delimiter.addItem(label, value)
        self.missing = QComboBox()
        for label, value in (("Report error", "error"), ("Drop rows", "drop"), ("Interpolate", "interpolate")):
            self.missing.addItem(label, value)
        source_layout.addWidget(QLabel("Encoding"), 1, 0)
        source_layout.addWidget(self.encoding, 1, 1)
        source_layout.addWidget(QLabel("Delimiter"), 2, 0)
        source_layout.addWidget(self.delimiter, 2, 1)
        source_layout.addWidget(QLabel("Missing values"), 3, 0)
        source_layout.addWidget(self.missing, 3, 1)
        root.addWidget(source_group)

        config_group = QGroupBox("Array configuration filter")
        config = QHBoxLayout(config_group)
        self.array_rows = QSpinBox(); self.array_rows.setObjectName("macArrayRows")
        self.array_rows.setRange(1, 8192); self.array_rows.setValue(16)
        self.array_cols = QSpinBox(); self.array_cols.setObjectName("macArrayCols")
        self.array_cols.setRange(1, 8192); self.array_cols.setValue(16)
        self.bit_width = QComboBox(); self.bit_width.setObjectName("macBitWidth")
        for bits in (1, 2, 4, 8): self.bit_width.addItem(f"{bits}-bit", bits)
        self.bit_width.setCurrentIndex(1)
        for label, widget in (("Rows", self.array_rows), ("Columns", self.array_cols), ("Precision", self.bit_width)):
            config.addWidget(QLabel(label)); config.addWidget(widget)
        config.addStretch(); root.addWidget(config_group)

        mapping_group = QGroupBox("Column mapping (blank means automatic)")
        mapping_layout = QFormLayout(mapping_group)
        self.mapping_edits: dict[str, QLineEdit] = {}
        for logical, label in LOGICAL_COLUMNS:
            edit = QLineEdit(); edit.setPlaceholderText(logical)
            self.mapping_edits[logical] = edit; mapping_layout.addRow(label, edit)
        root.addWidget(mapping_group)

        controls = QHBoxLayout()
        self.preview_button = QPushButton("Preview and validate"); self.preview_button.clicked.connect(self.preview)
        self.degree = QSpinBox(); self.degree.setRange(1, 3); self.degree.setValue(2)
        self.fit_button = QPushButton("Fit selected configuration"); self.fit_button.clicked.connect(self.start_fit)
        self.export_button = QPushButton("Export results"); self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        self.clear_log_button = QPushButton("Clear log")
        self.export_log_button = QPushButton("Export log")
        for widget in (self.preview_button, QLabel("Polynomial degree"), self.degree, self.fit_button,
                       self.export_button, self.clear_log_button, self.export_log_button): controls.addWidget(widget)
        controls.addStretch(); root.addLayout(controls)

        self.metrics_summary = QLabel("R\u00b2: --    Full-scale nonlinearity: --    RMSE: --")
        self.metrics_summary.setObjectName("macMetricsSummary"); root.addWidget(self.metrics_summary)
        splitter = QSplitter(); self.table = QTableWidget(); self.table.setAlternatingRowColors(True)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.clear_log_button.clicked.connect(self.log.clear); self.export_log_button.clicked.connect(self.export_log)
        splitter.addWidget(self.table); splitter.addWidget(self.log); splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2); root.addWidget(splitter, 1)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); root.addWidget(self.progress)

    def _options(self) -> MacImportOptions:
        mapping = {key: edit.text().strip() for key, edit in self.mapping_edits.items() if edit.text().strip()}
        return MacImportOptions(mapping, self.encoding.currentData(), self.delimiter.currentData(),
                                str(self.missing.currentData()), self.degree.value())

    def _selection(self) -> dict[str, int]:
        rows, cols, bits = self.array_rows.value(), self.array_cols.value(), int(self.bit_width.currentData())
        if rows <= 0 or cols <= 0 or bits not in (1, 2, 4, 8):
            raise ValueError("Rows/columns must be positive and precision must be 1/2/4/8-bit")
        return {"array_rows": rows, "array_cols": cols, "bit_width": bits}

    def _source(self) -> Path:
        text = self.path_edit.text().strip()
        if not text: raise ValueError("Select an array MAC CSV file")
        return Path(text).expanduser().resolve()

    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select array MAC data", self.path_edit.text(), "CSV (*.csv)")
        if path: self.path_edit.setText(path); self.settings.setValue("mac/source", path)

    def _source_changed(self, text: str) -> None:
        """Initialize controls from a newly selected file; users can then choose another group."""
        source = Path(text).expanduser()
        if not source.is_file():
            return
        try:
            options = self._options()
            frame, _ = load_mac_csv(
                source, column_mapping=options.column_mapping or None,
                encoding=options.encoding, delimiter=options.delimiter, missing=options.missing,
            )
            first = frame.iloc[0]
            self.array_rows.setValue(int(first["array_rows"]))
            self.array_cols.setValue(int(first["array_cols"]))
            index = self.bit_width.findData(int(first["bit_width"]))
            if index >= 0: self.bit_width.setCurrentIndex(index)
        except Exception:
            pass

    def _show_records(self, records: list[dict[str, Any]]) -> None:
        columns = list(records[0]) if records else []
        self.table.setRowCount(len(records)); self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        for row, record in enumerate(records):
            for column, name in enumerate(columns): self.table.setItem(row, column, QTableWidgetItem(str(record[name])))
        self.table.resizeColumnsToContents()

    def preview(self) -> None:
        try:
            self._persist_settings(); options = self._options(); selection = self._selection()
            frame, metadata = load_mac_csv(self._source(), column_mapping=options.column_mapping or None,
                encoding=options.encoding, delimiter=options.delimiter, missing=options.missing, **selection)
            self._show_records(frame.head(20).to_dict(orient="records"))
            self.log.appendPlainText(f"\u9884\u89c8\u6210\u529f: {len(frame)} rows; selection={metadata['selection']}")
            self.progress.setValue(20)
        except Exception as exc: self._show_error("Data preview failed", str(exc))

    def start_fit(self, output_dir: Path | None = None) -> None:
        try:
            self._persist_settings(); source = self._source(); selection = self._selection()
            output = Path(output_dir) if output_dir else self.platform_root / "outputs" / "gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "mac_model"
            self.fit_button.setEnabled(False); self.preview_button.setEnabled(False); self.progress.setRange(0, 0)
            self.log.appendPlainText(f"Fitting {source}; selection={selection}")
            self.worker = MacFitThread(self.service, source, output, self._options(), **selection)
            self.worker.succeeded.connect(self._fit_succeeded); self.worker.failed.connect(self._fit_failed)
            self.worker.finished.connect(self._fit_finished); self.worker.start()
        except Exception as exc: self._show_error("Cannot start fitting", str(exc))

    def _fit_succeeded(self, outcome: MacFitOutcome) -> None:
        self.last_results = list(outcome.summary); self._show_records(self.last_results)
        self.export_button.setEnabled(True)
        if self.last_results:
            result = self.last_results[0]
            self.metrics_summary.setText(f"R\u00b2: {result['r2']:.4f}    Full-scale nonlinearity: {result['nonlinearity_percent_fs']:.6f}%    RMSE: {result['rmse']:.6g}")
        self.log.appendPlainText(f"Fit complete: {outcome.model_path}")

    def _fit_failed(self, details: str) -> None:
        self.log.appendPlainText(details); self._show_error("Fit failed", details.splitlines()[-1])

    def _fit_finished(self) -> None:
        self.progress.setRange(0, 100); self.progress.setValue(100 if self.last_results else 0)
        self.fit_button.setEnabled(True); self.preview_button.setEnabled(True)

    def export_results(self) -> None:
        if not self.last_results: return
        path, selected = QFileDialog.getSaveFileName(self, "Export linearity results", "mac-linearity-results.xlsx", "Excel (*.xlsx);;CSV (*.csv)")
        if not path: return
        if not Path(path).suffix: path += ".xlsx" if "Excel" in selected else ".csv"
        try: self.log.appendPlainText(f"Results exported: {self.service.export_table(self.last_results, path)}")
        except Exception as exc: self._show_error("Export failed", str(exc))

    def export_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export run log", "dram-cim-log.txt", "Text (*.txt)")
        if path:
            try:
                target = Path(path).expanduser().resolve(); target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(self.log.toPlainText(), encoding="utf-8")
            except Exception as exc: self._show_error("Log export failed", str(exc))

    def _show_error(self, title: str, message: str) -> None:
        self.log.appendPlainText(f"{title}: {message}")
        if self.isVisible(): QMessageBox.critical(self, title, message)

    def _restore_settings(self) -> None:
        self.path_edit.setText(str(self.settings.value("mac/source", "")))
        for combo, key, fallback in ((self.encoding, "mac/encoding", None), (self.delimiter, "mac/delimiter", None),
                                     (self.missing, "mac/missing", "error")):
            index = combo.findData(self.settings.value(key, fallback)); combo.setCurrentIndex(max(0, index))
        self.degree.setValue(int(self.settings.value("mac/degree", 2)))
        self.array_rows.setValue(int(self.settings.value("mac/array_rows", 16)))
        self.array_cols.setValue(int(self.settings.value("mac/array_cols", 16)))
        self.bit_width.setCurrentIndex(max(0, self.bit_width.findData(int(self.settings.value("mac/bit_width", 2)))))
        for key, edit in self.mapping_edits.items(): edit.setText(str(self.settings.value(f"mac/mapping/{key}", "")))

    def _persist_settings(self) -> None:
        self.settings.setValue("mac/source", self.path_edit.text().strip())
        self.settings.setValue("mac/encoding", self.encoding.currentData()); self.settings.setValue("mac/delimiter", self.delimiter.currentData())
        self.settings.setValue("mac/missing", self.missing.currentData()); self.settings.setValue("mac/degree", self.degree.value())
        self.settings.setValue("mac/array_rows", self.array_rows.value()); self.settings.setValue("mac/array_cols", self.array_cols.value())
        self.settings.setValue("mac/bit_width", self.bit_width.currentData())
        for key, edit in self.mapping_edits.items(): self.settings.setValue(f"mac/mapping/{key}", edit.text().strip())

class SingleInferenceWidget(QWidget):
    """Selectable single-sample inference with image and Top-5 evidence."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.service = PlatformService()
        self.coordinator = coordinator
        self.require_workflow = require_workflow
        self.settings = QSettings("DRAM-CIM", "NeuroSimPlatform")
        self.worker: InferenceThread | None = None
        self.last_outcome: SingleInferenceOutcome | None = None
        self._active_model_id = ""
        layout = QVBoxLayout(self)
        description = QLabel(
            "选择网络、数据集、checkpoint 和器件误差模型，比较原生、理想差分和器件差分结果。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.model_combo = QComboBox()
        for spec in builtin_registry(self.project_root).list(task="image_classification"):
            self.model_combo.addItem(spec.display_name, spec.model_id)
        form.addRow("网络模型", self.model_combo)
        self.dataset_name = QLineEdit(); self.dataset_name.setReadOnly(True)
        form.addRow("数据集", self.dataset_name)
        self.dataset_edit = QLineEdit(); dataset_row = QHBoxLayout(); dataset_row.addWidget(self.dataset_edit)
        dataset_button = QPushButton("选择目录"); dataset_button.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(dataset_button); form.addRow("数据集根目录", dataset_row)
        self.checkpoint_edit = QLineEdit(); checkpoint_row = QHBoxLayout(); checkpoint_row.addWidget(self.checkpoint_edit)
        checkpoint_button = QPushButton("选择文件"); checkpoint_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_button); form.addRow("Checkpoint", checkpoint_row)
        self.auxiliary_label = QLabel("量化参数")
        self.auxiliary_edit = QLineEdit(); auxiliary_row = QHBoxLayout(); auxiliary_row.addWidget(self.auxiliary_edit)
        auxiliary_button = QPushButton("选择文件"); auxiliary_button.clicked.connect(self._browse_auxiliary)
        auxiliary_row.addWidget(auxiliary_button); form.addRow(self.auxiliary_label, auxiliary_row)
        self.error_model_combo = QComboBox()
        form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit = QLineEdit(); self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout(); error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("选择文件"); error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button); form.addRow("器件误差模型", error_row)
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("运行设备", self.device_combo)
        self.sample_index = QSpinBox(); self.sample_index.setRange(0, 9999)
        form.addRow("测试样本索引", self.sample_index)
        layout.addLayout(form)

        self.resource_status = QLabel()
        self.resource_status.setWordWrap(True)
        layout.addWidget(self.resource_status)
        for edit in (self.dataset_edit, self.checkpoint_edit, self.auxiliary_edit, self.error_model_edit):
            edit.textChanged.connect(self._update_resource_status)

        controls = QHBoxLayout(); self.run_button = QPushButton("运行单样本推理")
        self.run_button.clicked.connect(self.start_inference)
        controls.addWidget(self.run_button); controls.addStretch(); layout.addLayout(controls)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); layout.addWidget(self.progress)
        self.summary = QLabel("尚未运行"); self.summary.setWordWrap(True); layout.addWidget(self.summary)

        result_area = QHBoxLayout()
        image_group = QGroupBox("输入与最终结果")
        image_layout = QVBoxLayout(image_group)
        self.input_image = QLabel("运行后显示输入图片")
        self.input_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_image.setMinimumSize(220, 220)
        self.input_image.setStyleSheet(
            "QLabel { background: #172033; color: #DCE7F5; border: 1px solid #60738D; "
            "border-radius: 8px; padding: 8px; }"
        )
        image_layout.addWidget(self.input_image)
        self.final_prediction = QLabel("真实标签：-    最终预测：-")
        self.final_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.final_prediction)
        result_area.addWidget(image_group, 1)

        tables = QVBoxLayout()
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["模式", "预测标签", "置信度", "耗时 (ms)"])
        tables.addWidget(self.result_table)
        self.top5_table = QTableWidget(0, 4)
        self.top5_table.setHorizontalHeaderLabels(["模式", "排名", "类别", "置信度"])
        tables.addWidget(self.top5_table)
        result_area.addLayout(tables, 3)
        layout.addLayout(result_area)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100)
        layout.addWidget(self.log)

        requested_model = str(self.settings.value("inference/model", "lenet5-mnist"))
        saved_model = requested_model
        if not self._model_resources_ready(saved_model):
            candidates = ["lenet5-mnist"] + [
                str(self.model_combo.itemData(i)) for i in range(self.model_combo.count())
            ]
            saved_model = next(
                (model_id for model_id in candidates if self._model_resources_ready(model_id)),
                "lenet5-mnist",
            )
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(saved_model)))
        self._load_model_settings(saved_model)
        self._active_model_id = saved_model
        self.device_combo.setCurrentText(str(self.settings.value("inference/device", "auto")))
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        if self.coordinator is not None:
            _fill_error_model_combo(self.error_model_combo, self.coordinator)
            self.error_model_combo.currentIndexChanged.connect(self._workflow_error_changed)
            self.coordinator.models_changed.connect(self._refresh_error_models)
            self.coordinator.selection_changed.connect(self._sync_workflow_selection)
            self._sync_workflow_error_path()
        else:
            self.error_model_combo.addItem("手动选择误差模型文件", None)
            self.error_model_combo.setEnabled(False)
        self._update_resource_status()
        if requested_model != saved_model:
            self.resource_status.setText(
                f"保存的模型 {requested_model} 缺少资源，已自动切换到 {saved_model}。\n"
                + self.resource_status.text()
            )

    def _model_defaults(self, model_id: str) -> dict[str, str]:
        platform = self.project_root / "5_NeuroCIM_opt"
        error_model = platform / "outputs/error_models/01_NMOS/error_model.json"
        if model_id == "lenet5-mnist":
            return {"dataset": "MNIST", "root": str(self._dataset_root("mnist")),
                    "checkpoint": str(self.project_root / "3_models/LeNet_5/chekpoint/lenet5_float.pt"),
                    "auxiliary": str(self.project_root / "3_models/LeNet_5/chekpoint/quantization.json"),
                    "auxiliary_label": "量化参数", "error_model": str(error_model)}
        if model_id == "bnn-mnist":
            return {"dataset": "MNIST", "root": str(self._dataset_root("mnist")),
                    "checkpoint": str(self.project_root / "3_models/BNN/checkpoint/bnn_mnist.pt"),
                    "auxiliary": "", "auxiliary_label": "辅助参数（无需）", "error_model": str(error_model)}
        if model_id == "tinytransformer-cifar10":
            return {"dataset": "CIFAR-10", "root": str(self._dataset_root("cifar10")),
                    "checkpoint": str(self.project_root / "3_models/TinyTransformer/checkpoint/cct_7_3x1_32_cifar10.pth"),
                    "auxiliary": str(self.project_root / "3_models/TinyTransformer/checkpoint/input_scales.json"),
                    "auxiliary_label": "输入尺度 JSON", "error_model": str(error_model)}
        return {"dataset": "CIFAR-10", "root": str(self._dataset_root("cifar10")),
                "checkpoint": str(self.project_root / "3_models/Resnet20/checkpoint/resnet20_cifar10.pt"),
                "auxiliary": str(self.project_root / "3_models/Resnet20/output/formal/input_scales.json"),
                "auxiliary_label": "输入尺度 JSON", "error_model": str(error_model)}

    def _dataset_root(self, dataset: str) -> Path:
        configured = data_root(dataset)
        candidates = [configured, Path(r"E:\datasets"), self.project_root / "data" / dataset]
        marker = Path("MNIST/raw/t10k-images-idx3-ubyte") if dataset == "mnist" else Path("cifar-10-batches-py/test_batch")
        return next((path for path in candidates if (path / marker).is_file()), configured)

    def _model_configuration(self, model_id: str) -> dict[str, str]:
        values = self._model_defaults(model_id)
        for name in ("dataset_root", "checkpoint", "auxiliary", "error_model"):
            saved = str(self.settings.value(f"inference/{model_id}/{name}", ""))
            if saved:
                values["root" if name == "dataset_root" else name] = saved
        return values

    def _configuration_problems(self, model_id: str, values: dict[str, str] | None = None) -> list[str]:
        values = values or self._model_configuration(model_id)
        dataset = Path(values["root"]).expanduser()
        if model_id.endswith("mnist"):
            dataset_files = [dataset / "MNIST/raw/t10k-images-idx3-ubyte",
                             dataset / "MNIST/raw/t10k-labels-idx1-ubyte"]
        else:
            from .inference_runtime import cifar10_required_files
            dataset_files = list(cifar10_required_files(dataset))
        checks = [(path, "数据集") for path in dataset_files]
        checks += [(Path(values["checkpoint"]), "checkpoint"),
                  (Path(values["error_model"]), "误差模型")]
        if model_id != "bnn-mnist":
            checks.append((Path(values["auxiliary"]), "辅助参数"))
        problems = [f"{purpose}: {path}" for path, purpose in checks if not path.is_file()]
        if self.require_workflow and (
            self.coordinator is None or not self.error_model_combo.currentData()
        ):
            problems.insert(0, "尚未完成误差拟合，或没有选择误差建模方法")
        return problems

    def _model_resources_ready(self, model_id: str) -> bool:
        return self.model_combo.findData(model_id) >= 0 and not self._configuration_problems(model_id)

    def _save_model_settings(self, model_id: str) -> None:
        if not model_id:
            return
        for name, edit in (("dataset_root", self.dataset_edit), ("checkpoint", self.checkpoint_edit),
                           ("auxiliary", self.auxiliary_edit), ("error_model", self.error_model_edit)):
            self.settings.setValue(f"inference/{model_id}/{name}", edit.text().strip())

    def _load_model_settings(self, model_id: str) -> None:
        values = self._model_configuration(model_id)
        self.dataset_name.setText(values["dataset"]); self.dataset_edit.setText(values["root"])
        self.checkpoint_edit.setText(values["checkpoint"]); self.auxiliary_edit.setText(values["auxiliary"])
        self.auxiliary_label.setText(values["auxiliary_label"]); self.error_model_edit.setText(values["error_model"])
        self.auxiliary_edit.setEnabled(model_id != "bnn-mnist")
        if self.require_workflow:
            self._sync_workflow_error_path()

    def _model_changed(self, _index: int) -> None:
        self._save_model_settings(self._active_model_id)
        model_id = str(self.model_combo.currentData())
        self._load_model_settings(model_id)
        self._active_model_id = model_id
        self._update_resource_status()

    def _refresh_error_models(self) -> None:
        if self.coordinator is None: return
        preferred = self.coordinator.selected_error_model_id()
        _fill_error_model_combo(self.error_model_combo, self.coordinator, preferred)
        self._sync_workflow_error_path(); self._update_resource_status()

    def _sync_workflow_selection(self, model_id: str) -> None:
        index = self.error_model_combo.findData(model_id)
        if index < 0: return
        self.error_model_combo.blockSignals(True); self.error_model_combo.setCurrentIndex(index)
        self.error_model_combo.blockSignals(False)
        self._sync_workflow_error_path(); self._update_resource_status()

    def _sync_workflow_error_path(self) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData(): return
        model = self.coordinator.registry.resolve(str(self.error_model_combo.currentData()))
        self.error_model_edit.setText(str(Path(model["path"]).resolve()))

    def _workflow_error_changed(self, _index: int) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData():
            self._update_resource_status(); return
        self._sync_workflow_error_path()
        self.coordinator.select_error_model(str(self.error_model_combo.currentData()))
        self._update_resource_status()

    def _update_resource_status(self, *_args: object) -> None:
        if not hasattr(self, "resource_status"):
            return
        model_id = str(self.model_combo.currentData())
        values = {"root": self.dataset_edit.text().strip(), "checkpoint": self.checkpoint_edit.text().strip(),
                  "auxiliary": self.auxiliary_edit.text().strip(), "error_model": self.error_model_edit.text().strip()}
        problems = self._configuration_problems(model_id, values)
        if problems:
            self.resource_status.setText("资源状态：未就绪\n" + "\n".join(problems))
            self.resource_status.setStyleSheet("color: #b3432d;")
            self.run_button.setEnabled(False)
        else:
            self.resource_status.setText("资源状态：已就绪，可以运行")
            self.resource_status.setStyleSheet("color: #26734d;")
            self.run_button.setEnabled(self.worker is None or not self.worker.isRunning())

    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集根目录", self.dataset_edit.text())
        if path: self.dataset_edit.setText(path)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 checkpoint", self.checkpoint_edit.text(), "PyTorch (*.pt *.pth);;All files (*)")
        if path: self.checkpoint_edit.setText(path)

    def _browse_auxiliary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择辅助参数", self.auxiliary_edit.text(), "JSON (*.json);;All files (*)")
        if path: self.auxiliary_edit.setText(path)

    def _browse_error_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择器件误差模型", self.error_model_edit.text(), "JSON (*.json)")
        if path: self.error_model_edit.setText(path)

    def _validate_resources(self) -> tuple[str, Path, Path, Path | None, Path, str]:
        model_id = str(self.model_combo.currentData())
        values = {"root": self.dataset_edit.text().strip(), "checkpoint": self.checkpoint_edit.text().strip(),
                  "auxiliary": self.auxiliary_edit.text().strip(), "error_model": self.error_model_edit.text().strip()}
        problems = self._configuration_problems(model_id, values)
        if problems:
            raise FileNotFoundError("缺少推理资源：\n" + "\n".join(problems))
        auxiliary = Path(values["auxiliary"]).expanduser().resolve() if values["auxiliary"] else None
        return (model_id, Path(values["root"]).expanduser().resolve(),
                Path(values["checkpoint"]).expanduser().resolve(), auxiliary,
                Path(values["error_model"]).expanduser().resolve(), self.device_combo.currentText())

    def start_inference(self) -> None:
        if self.worker is not None and self.worker.isRunning(): return
        try:
            if self.coordinator is not None and self.error_model_combo.currentData():
                self.coordinator.select_error_model(str(self.error_model_combo.currentData()))
            model_id, dataset, checkpoint, auxiliary, error_model, device = self._validate_resources()
        except Exception as exc:
            message = str(exc); self.summary.setText("推理配置错误：" + message.splitlines()[0])
            self.log.appendPlainText(message); return
        self.last_outcome = None; self.run_button.setEnabled(False); self.progress.setRange(0, 0)
        self.log.appendPlainText(f"开始推理：{model_id}，样本 {self.sample_index.value()}")
        self._save_model_settings(model_id)
        self.settings.setValue("inference/model", model_id); self.settings.setValue("inference/device", device)
        self.worker = InferenceThread(self.service, self.project_root, model_id, self.sample_index.value(),
                                      dataset, checkpoint, auxiliary, error_model, device)
        self.worker.succeeded.connect(self._inference_succeeded); self.worker.failed.connect(self._inference_failed)
        self.worker.finished.connect(self._inference_finished); self.worker.start()

    @staticmethod
    def _display_name(model_id: str, label: int) -> str:
        if model_id in {"resnet20-cifar10", "tinytransformer-cifar10"}:
            names = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
            return f"{label} ({names[label]})" if 0 <= label < len(names) else str(label)
        return str(label)

    def _show_input_image(self, outcome: SingleInferenceOutcome) -> None:
        if not outcome.input_image or not outcome.input_shape:
            self.input_image.setText("输入图片数据不可用")
            return
        array = np.asarray(outcome.input_image, dtype=np.float32).reshape(outcome.input_shape)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
        low, high = float(array.min()), float(array.max())
        is_rgb_unit_range = array.ndim == 3 and array.shape[-1] == 3 and low >= 0.0 and high <= 1.0
        if not is_rgb_unit_range:
            array = (array - low) / (high - low) if high > low else np.zeros_like(array)
        pixels = np.ascontiguousarray(np.clip(array * 255, 0, 255).astype(np.uint8))
        if pixels.ndim == 2 or pixels.shape[-1] == 1:
            pixels = pixels.reshape(pixels.shape[0], pixels.shape[1])
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_Grayscale8).copy()
        else:
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.input_image.setPixmap(pixmap)

    def _inference_succeeded(self, outcome: SingleInferenceOutcome) -> None:
        self.last_outcome = outcome
        model_id = str(self.model_combo.currentData())
        self.summary.setText(
            f"样本索引：{outcome.sample_index}　真实标签：{self._display_name(model_id, outcome.true_label)}　"
            f"设备：{outcome.device}　准备：{outcome.preparation_ms:.1f} ms　总耗时：{outcome.total_elapsed_ms:.1f} ms"
        )
        self._show_input_image(outcome)
        final = outcome.predictions[-1]
        self.final_prediction.setText(
            f"真实标签：{self._display_name(model_id, outcome.true_label)}    "
            f"器件误差预测：{self._display_name(model_id, final.predicted_label)} "
            f"({final.confidence:.2%})"
        )
        self.result_table.setRowCount(len(outcome.predictions))
        top_rows = sum(len(prediction.top5) for prediction in outcome.predictions)
        self.top5_table.setRowCount(top_rows)
        top_row = 0
        for row, prediction in enumerate(outcome.predictions):
            values = (prediction.mode, self._display_name(model_id, prediction.predicted_label),
                      f"{prediction.confidence:.2%}", f"{prediction.elapsed_ms:.3f}")
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(value))
            for rank, (label, confidence) in enumerate(prediction.top5, 1):
                for column, value in enumerate((prediction.mode, str(rank), self._display_name(model_id, label), f"{confidence:.2%}")):
                    self.top5_table.setItem(top_row, column, QTableWidgetItem(value))
                top_row += 1
        self.result_table.resizeColumnsToContents(); self.top5_table.resizeColumnsToContents()
        self.log.appendPlainText("单样本推理完成")
        target = self.project_root / "5_NeuroCIM_opt/outputs/gui_inference"; target.mkdir(parents=True, exist_ok=True)
        result_path = target / f"{model_id}_sample_{outcome.sample_index}.json"
        result_path.write_text(json.dumps(outcome.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self.log.appendPlainText("结果文件：" + str(result_path))
        if self.coordinator is not None and self.error_model_combo.currentData():
            try:
                self.coordinator.record_inference(
                    model_id=str(self.error_model_combo.currentData()),
                    dataset_name=self.dataset_name.text(), dataset_root=self.dataset_edit.text(),
                    network_model=model_id, checkpoint_path=self.checkpoint_edit.text(),
                    auxiliary_path=self.auxiliary_edit.text() or None,
                    device=outcome.device, result=outcome.as_dict(),
                )
                self.log.appendPlainText("已将模型、数据集和误差建模方法传递到误差感知训练")
            except Exception as exc:
                self.log.appendPlainText("工作流记录失败：" + str(exc))
                self.summary.setText("推理完成，但无法传递到误差感知训练：" + str(exc))

    def _inference_failed(self, details: str) -> None:
        self.log.appendPlainText(details); self.summary.setText("推理失败：" + details.splitlines()[-1])

    def _inference_finished(self) -> None:
        self.progress.setRange(0, 100); self.progress.setValue(100 if self.last_outcome is not None else 0)
        self._update_resource_status()


class BatchInferenceWidget(QWidget):
    """Grouped inference for every registered network with a fitted error model."""

    MODE_NAMES = {
        "float": "浮点模型",
        "ideal_differential": "理想差分",
        "device_differential": "器件误差差分",
    }
    CHART_MODE_NAMES = {
        "float": "Float",
        "ideal_differential": "Ideal differential",
        "device_differential": "Device-error differential",
    }

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.coordinator = coordinator
        self.require_workflow = require_workflow
        self.worker: BatchInferenceThread | None = None
        self.last_outcome: GroupedBatchInferenceOutcome | None = None
        platform = self.project_root / "5_NeuroCIM_opt"
        layout = QVBoxLayout(self)
        description = QLabel(
            "选择误差建模方法和网络后批量运行对应测试集，对比浮点、理想正负权重差分和器件误差差分结果；"
            "柱状图按每 N 个连续样本统计平均准确率，最后不足 N 个的样本单独成组。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.model_combo = QComboBox()
        for spec in builtin_registry(self.project_root).list(task="image_classification"):
            self.model_combo.addItem(spec.display_name, spec.model_id)
        form.addRow("网络模型", self.model_combo)

        self.dataset_name = QLineEdit(); self.dataset_name.setReadOnly(True)
        form.addRow("数据集", self.dataset_name)
        self.dataset_edit = QLineEdit()
        dataset_row = QHBoxLayout(); dataset_row.addWidget(self.dataset_edit)
        dataset_button = QPushButton("选择目录"); dataset_button.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(dataset_button); form.addRow("数据集根目录", dataset_row)

        self.checkpoint_edit = QLineEdit()
        checkpoint_row = QHBoxLayout(); checkpoint_row.addWidget(self.checkpoint_edit)
        checkpoint_button = QPushButton("选择文件"); checkpoint_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_button); form.addRow("Checkpoint", checkpoint_row)

        self.auxiliary_label = QLabel("量化参数")
        self.quantization_edit = QLineEdit()
        quantization_row = QHBoxLayout(); quantization_row.addWidget(self.quantization_edit)
        quantization_button = QPushButton("选择文件"); quantization_button.clicked.connect(self._browse_quantization)
        quantization_row.addWidget(quantization_button); form.addRow(self.auxiliary_label, quantization_row)

        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit = QLineEdit(str(platform / "outputs/error_models/01_NMOS/error_model.json"))
        self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout(); error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("选择文件"); error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button); form.addRow("器件误差模型", error_row)

        parameter_row = QHBoxLayout()
        self.sample_count = QSpinBox(); self.sample_count.setRange(1, 10_000); self.sample_count.setValue(1_000)
        self.group_size = QSpinBox(); self.group_size.setRange(1, 10_000); self.group_size.setValue(100)
        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 2_048); self.batch_size.setValue(128)
        parameter_row.addWidget(QLabel("样本数")); parameter_row.addWidget(self.sample_count)
        parameter_row.addWidget(QLabel("每组 N")); parameter_row.addWidget(self.group_size)
        parameter_row.addWidget(QLabel("推理 batch")); parameter_row.addWidget(self.batch_size)
        parameter_row.addStretch(); form.addRow("统计设置", parameter_row)

        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("运行设备", self.device_combo)
        default_output = platform / "outputs/gui_batch_inference" / "batch_result.json"
        self.output_edit = QLineEdit(str(default_output))
        output_row = QHBoxLayout(); output_row.addWidget(self.output_edit)
        output_button = QPushButton("选择文件"); output_button.clicked.connect(self._browse_output)
        output_row.addWidget(output_button); form.addRow("结果 JSON", output_row)
        layout.addLayout(form)

        controls = QHBoxLayout()
        self.run_button = QPushButton("运行带误差批量推理"); self.run_button.clicked.connect(self.start_inference)
        self.cancel_button = QPushButton("取消"); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_inference)
        controls.addWidget(self.run_button); controls.addWidget(self.cancel_button); controls.addStretch()
        layout.addLayout(controls)
        self.summary = QLabel("尚未运行"); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); layout.addWidget(self.progress)

        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["模式", "正确数", "样本数", "总体准确率", "耗时 (ms)"])
        layout.addWidget(self.result_table)
        chart_group = QGroupBox("每 N 个样本的平均准确率柱状图")
        chart_layout = QVBoxLayout(chart_group)
        self.chart_scroll = QScrollArea(); self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chart_scroll.setMinimumHeight(270); self.chart_scroll.setMaximumHeight(350)
        self.chart_label = QLabel("运行后显示分组准确率柱状图")
        self.chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.chart_label.setMinimumHeight(250)
        self.chart_scroll.setWidget(self.chart_label); chart_layout.addWidget(self.chart_scroll)
        layout.addWidget(chart_group)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(150)
        layout.addWidget(self.log)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData("lenet5-mnist")))
        self._model_changed(self.model_combo.currentIndex())
        if self.coordinator is not None:
            _fill_error_model_combo(self.error_model_combo, self.coordinator)
            self.error_model_combo.currentIndexChanged.connect(self._workflow_error_changed)
            self.coordinator.models_changed.connect(self._refresh_error_models)
            self.coordinator.selection_changed.connect(self._sync_workflow_selection)
            self._sync_workflow_error_path()
        else:
            self.error_model_combo.addItem("手动选择误差模型文件", None)
            self.error_model_combo.setEnabled(False)

    def _dataset_root(self, dataset: str) -> Path:
        candidates = [data_root(dataset), Path(r"E:\datasets"), self.project_root / "data" / dataset]
        marker = (Path("MNIST/raw/t10k-images-idx3-ubyte") if dataset == "mnist"
                  else Path("cifar-10-batches-py/test_batch"))
        return next((path for path in candidates if (path / marker).is_file()), candidates[0])

    def _model_defaults(self, model_id: str) -> dict[str, str]:
        if model_id == "lenet5-mnist":
            model = self.project_root / "3_models/LeNet_5/chekpoint"
            return {"dataset": "MNIST", "root": str(self._dataset_root("mnist")),
                    "checkpoint": str(model / "lenet5_float.pt"),
                    "auxiliary": str(model / "quantization.json"), "auxiliary_label": "量化参数"}
        if model_id == "bnn-mnist":
            return {"dataset": "MNIST", "root": str(self._dataset_root("mnist")),
                    "checkpoint": str(self.project_root / "3_models/BNN/checkpoint/bnn_mnist.pt"),
                    "auxiliary": "", "auxiliary_label": "辅助参数（自动校准）"}
        if model_id == "tinytransformer-cifar10":
            return {"dataset": "CIFAR-10", "root": str(self._dataset_root("cifar10")),
                    "checkpoint": str(self.project_root / "3_models/TinyTransformer/checkpoint/cct_7_3x1_32_cifar10.pth"),
                    "auxiliary": str(self.project_root / "3_models/TinyTransformer/checkpoint/input_scales.json"),
                    "auxiliary_label": "输入尺度 JSON"}
        return {"dataset": "CIFAR-10", "root": str(self._dataset_root("cifar10")),
                "checkpoint": str(self.project_root / "3_models/Resnet20/checkpoint/resnet20_cifar10.pt"),
                "auxiliary": str(self.project_root / "3_models/Resnet20/output/formal/input_scales.json"),
                "auxiliary_label": "输入尺度 JSON"}

    def _model_changed(self, _index: int) -> None:
        values = self._model_defaults(str(self.model_combo.currentData()))
        self.dataset_name.setText(values["dataset"]); self.dataset_edit.setText(values["root"])
        self.checkpoint_edit.setText(values["checkpoint"]); self.quantization_edit.setText(values["auxiliary"])
        self.auxiliary_label.setText(values["auxiliary_label"])
        self.quantization_edit.setEnabled(bool(values["auxiliary"]))
        if str(self.model_combo.currentData()) == "tinytransformer-cifar10" and self.batch_size.value() > 4:
            self.batch_size.setValue(4)

    def _refresh_error_models(self) -> None:
        if self.coordinator is None: return
        preferred = self.coordinator.selected_error_model_id()
        _fill_error_model_combo(self.error_model_combo, self.coordinator, preferred)
        self._sync_workflow_error_path()

    def _sync_workflow_selection(self, model_id: str) -> None:
        index = self.error_model_combo.findData(model_id)
        if index < 0: return
        self.error_model_combo.blockSignals(True); self.error_model_combo.setCurrentIndex(index)
        self.error_model_combo.blockSignals(False); self._sync_workflow_error_path()

    def _sync_workflow_error_path(self) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData(): return
        model = self.coordinator.registry.resolve(str(self.error_model_combo.currentData()))
        self.error_model_edit.setText(str(Path(model["path"]).resolve()))

    def _workflow_error_changed(self, _index: int) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData(): return
        self._sync_workflow_error_path()
        self.coordinator.select_error_model(str(self.error_model_combo.currentData()))

    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集根目录", self.dataset_edit.text())
        if path: self.dataset_edit.setText(path)

    def _select_file(self, edit: QLineEdit, title: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, edit.text(), file_filter)
        if path: edit.setText(path)

    def _browse_checkpoint(self) -> None:
        self._select_file(self.checkpoint_edit, "选择 checkpoint", "PyTorch (*.pt *.pth);;All files (*)")

    def _browse_quantization(self) -> None:
        self._select_file(self.quantization_edit, "选择量化参数", "JSON (*.json)")

    def _browse_error_model(self) -> None:
        self._select_file(self.error_model_edit, "选择器件误差模型", "JSON (*.json)")

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存批量推理结果", self.output_edit.text(), "JSON (*.json)")
        if path: self.output_edit.setText(path)

    def _request(self) -> BatchInferenceRequest:
        auxiliary = self.quantization_edit.text().strip() or self.checkpoint_edit.text().strip()
        return BatchInferenceRequest(
            project_root=self.project_root,
            dataset_root=Path(self.dataset_edit.text().strip()),
            checkpoint_path=Path(self.checkpoint_edit.text().strip()),
            quantization_path=Path(auxiliary),
            error_model_path=Path(self.error_model_edit.text().strip()),
            model_id=str(self.model_combo.currentData()), device=self.device_combo.currentText(),
            sample_count=self.sample_count.value(), group_size=self.group_size.value(),
            inference_batch_size=self.batch_size.value(), output_path=Path(self.output_edit.text().strip()),
        )

    def start_inference(self) -> None:
        if self.worker is not None and self.worker.isRunning(): return
        if self.require_workflow and (
            self.coordinator is None or not self.error_model_combo.currentData()
        ):
            message = "请先完成误差拟合，并从下拉框选择一种误差建模方法"
            self.summary.setText(message); self.log.appendPlainText(message); return
        if self.coordinator is not None and self.error_model_combo.currentData():
            self.coordinator.select_error_model(str(self.error_model_combo.currentData()))
        request = self._request()
        model_id = str(self.model_combo.currentData())
        if model_id.endswith("mnist"):
            dataset_files = (
                (request.dataset_root / "MNIST/raw/t10k-images-idx3-ubyte", "数据集"),
                (request.dataset_root / "MNIST/raw/t10k-labels-idx1-ubyte", "数据集标签"),
            )
        else:
            from .inference_runtime import cifar10_required_files
            dataset_files = tuple((path, "CIFAR-10 数据") for path in cifar10_required_files(request.dataset_root))
        required = list(dataset_files) + [
            (request.checkpoint_path, "checkpoint"), (request.error_model_path, "误差模型")
        ]
        if model_id != "bnn-mnist": required.append((request.quantization_path, "辅助参数"))
        missing = [f"{name}: {path}" for path, name in required if not path.is_file()]
        if missing:
            message = "缺少批量推理资源：\n" + "\n".join(missing)
            self.summary.setText(message); self.log.appendPlainText(message); return
        self.last_outcome = None; self.result_table.setRowCount(0); self.progress.setValue(0)
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.summary.setText("批量推理运行中")
        self.log.appendPlainText(
            f"开始：样本数={request.sample_count}，每组 N={request.group_size}，误差模型={request.error_model_path}"
        )
        self.worker = BatchInferenceThread(request)
        self.worker.progressed.connect(self._progressed); self.worker.succeeded.connect(self._succeeded)
        self.worker.failed.connect(self._failed); self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._finished); self.worker.start()

    def cancel_inference(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel(); self.cancel_button.setEnabled(False)
            self.summary.setText("已请求取消，等待当前 batch 完成")

    def _progressed(self, done: int, total: int) -> None:
        self.progress.setValue(round(done * 100 / total) if total else 0)
        self.summary.setText(f"批量推理运行中：{done}/{total}")

    def _succeeded(self, outcome: GroupedBatchInferenceOutcome) -> None:
        self.last_outcome = outcome
        self.result_table.setRowCount(len(outcome.modes))
        for row, (mode, result) in enumerate(outcome.modes.items()):
            values = (self.MODE_NAMES.get(mode, mode), result.correct, result.sample_count,
                      f"{result.accuracy:.2%}", f"{result.elapsed_ms:.3f}")
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.result_table.resizeColumnsToContents()
        chart_path = self._save_chart(outcome)
        target = self.chart_scroll.viewport().size()
        target.setWidth(max(640, target.width() - 12)); target.setHeight(320)
        self.chart_label.setPixmap(_high_dpi_pixmap(chart_path, target, self.chart_label))
        device_accuracy = outcome.modes["device_differential"].accuracy
        self.summary.setText(
            f"完成：{outcome.sample_count} 个样本，共 {len(outcome.groups)} 组，"
            f"器件误差准确率 {device_accuracy:.2%}，设备 {outcome.device}"
        )
        self.log.appendPlainText(f"结果 JSON：{outcome.output_path}")
        self.log.appendPlainText(f"准确率柱状图：{chart_path}")
        if self.coordinator is not None and self.error_model_combo.currentData():
            auxiliary = self.quantization_edit.text().strip() or None
            try:
                self.coordinator.record_inference(
                    model_id=str(self.error_model_combo.currentData()),
                    dataset_name=self.dataset_name.text(), dataset_root=self.dataset_edit.text(),
                    network_model=str(self.model_combo.currentData()),
                    checkpoint_path=self.checkpoint_edit.text(), auxiliary_path=auxiliary,
                    device=outcome.device, result=outcome.as_dict(),
                )
                self.log.appendPlainText("已将本次批量推理配置传递到误差感知训练")
            except Exception as exc:
                self.log.appendPlainText("工作流记录失败：" + str(exc))
                self.summary.setText("推理完成，但无法传递到误差感知训练：" + str(exc))

    def _save_chart(self, outcome: GroupedBatchInferenceOutcome) -> Path:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties

        output = Path(outcome.output_path or self.output_edit.text()).resolve()
        chart_path = output.with_name(output.stem + "_accuracy.png")
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
        chinese_font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
        group_count = len(outcome.groups); width = max(7.2, min(11.0, group_count * 0.42))
        figure = Figure(figsize=(width, 3.6), tight_layout=True, facecolor="#FFFFFF")
        FigureCanvasAgg(figure); axes = figure.add_subplot(111)
        axes.set_facecolor("#FFFFFF")
        positions = np.arange(group_count); bar_width = 0.25
        colors = ("#94A3B8", "#2F80ED", "#F97316")
        for offset, (mode, color) in enumerate(zip(
            ("float", "ideal_differential", "device_differential"), colors
        )):
            values = [group.accuracy[mode] * 100 for group in outcome.groups]
            axes.bar(positions + (offset - 1) * bar_width, values, bar_width,
                     label=self.CHART_MODE_NAMES[mode], color=color)
        labels = [f"{group.start_index}-{group.end_index}" for group in outcome.groups]
        label_step = max(1, int(np.ceil(group_count / 12)))
        shown = positions[::label_step]
        axes.set_xticks(shown, [labels[index] for index in range(0, group_count, label_step)],
                        rotation=35, ha="right")
        axes.set_ylim(0, 105)
        axes.set_ylabel("准确率（%）", fontproperties=chinese_font)
        axes.set_xlabel("样本索引范围", fontproperties=chinese_font)
        axes.set_title("分组准确率对比", color="#26344D", pad=10,
                       fontproperties=chinese_font)
        axes.grid(axis="y", color="#D9E2EE", linewidth=0.8, alpha=0.85)
        axes.set_axisbelow(True)
        axes.spines[["top", "right"]].set_visible(False)
        axes.spines[["left", "bottom"]].set_color("#9AAAC0")
        axes.tick_params(colors="#526077")
        axes.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3,
                    frameon=False, prop=chinese_font)
        figure.savefig(
            chart_path,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            metadata={"Software": "NeuroCIM", "Description": "High-resolution batch inference chart"},
        )
        return chart_path

    def _failed(self, details: str) -> None:
        self.log.appendPlainText(details); self.summary.setText("批量推理失败：" + details.splitlines()[-1])

    def _cancelled(self, message: str) -> None:
        self.log.appendPlainText(message); self.summary.setText("批量推理已取消")

    def _finished(self) -> None:
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        if self.last_outcome is not None: self.progress.setValue(100)


class ErrorAwareTrainingWidget(QWidget):
    """Error-aware training that inherits the successful step-two network/dataset."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        application = QApplication.instance()
        if application is not None:
            configure_windows_font(application)
        self.project_root = Path(project_root).resolve()
        self.coordinator = coordinator
        self.require_workflow = require_workflow
        self.service = ErrorAwareTrainingService()
        self.worker: TrainingThread | None = None
        self.last_outcome: TrainingOutcome | None = None
        platform_root = self.project_root / "5_NeuroCIM_opt"
        layout = QVBoxLayout(self)
        description = QLabel(
            "网络类型和数据集自动继承最近一次成功推理；误差方法可从已生成的拟合结果中选择。"
            "训练使用正负权重差分器件 LUT 前向计算和 STE 梯度，并比较训练前后的三种推理模式。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        self.model_edit = QLineEdit("lenet5-mnist"); self.model_edit.setReadOnly(True)
        form.addRow("继承的网络模型", self.model_edit)
        self.dataset_name = QLineEdit("MNIST"); self.dataset_name.setReadOnly(True)
        form.addRow("继承的数据集", self.dataset_name)
        self.checkpoint_edit = QLineEdit(str(
            self.project_root / "3_models/LeNet_5/chekpoint/lenet5_float.pt"
        )); self.checkpoint_edit.setReadOnly(require_workflow)
        form.addRow("继承的 Checkpoint", self.checkpoint_edit)
        self.auxiliary_edit = QLineEdit(str(
            self.project_root / "3_models/LeNet_5/chekpoint/quantization.json"
        )); self.auxiliary_edit.setReadOnly(require_workflow)
        form.addRow("继承的辅助参数", self.auxiliary_edit)
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit = QLineEdit(str(
            platform_root / "outputs" / "error_models" / "01_NMOS" / "error_model.json"
        ))
        self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout()
        error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("Browse")
        error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button)
        form.addRow("误差模型文件", error_row)
        self.dataset_edit = QLineEdit(str(data_root("mnist")))
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_edit)
        dataset_button = QPushButton("Browse")
        dataset_button.clicked.connect(self._browse_dataset)
        dataset_button.setEnabled(not require_workflow)
        dataset_row.addWidget(dataset_button)
        form.addRow("继承的数据集根目录", dataset_row)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 1.0)
        self.lambda_spin.setSingleStep(0.05)
        self.lambda_spin.setValue(1.0)
        form.addRow("Error strength (lambda)", self.lambda_spin)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100)
        self.epoch_spin.setValue(1)
        form.addRow("Epochs", self.epoch_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(128)
        form.addRow("Batch size", self.batch_spin)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("Device", self.device_combo)
        self.output_edit = QLineEdit(str(
            platform_root / "outputs" / "gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "training"
        ))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self._browse_output)
        output_row.addWidget(output_button)
        form.addRow("Output directory", output_row)
        layout.addLayout(form)
        controls = QHBoxLayout()
        self.run_button = QPushButton("开始误差感知训练")
        self.run_button.clicked.connect(self.start_training)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        self.export_button = QPushButton("导出结果 JSON")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        controls.addWidget(self.run_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.export_button)
        controls.addStretch()
        layout.addLayout(controls)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.summary = QLabel("Not run")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Stage", "Mode", "Accuracy", "Loss"])
        layout.addWidget(self.metrics_table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        if self.coordinator is not None:
            _fill_error_model_combo(self.error_model_combo, self.coordinator)
            self.error_model_combo.currentIndexChanged.connect(self._training_error_changed)
            self.coordinator.models_changed.connect(self._refresh_training_models)
            self.coordinator.inference_changed.connect(self._load_inherited_context)
            self._sync_training_error_path(); self._load_inherited_context()
        else:
            self.error_model_combo.addItem("手动选择误差模型文件", None)
            self.error_model_combo.setEnabled(False)

    def _refresh_training_models(self) -> None:
        if self.coordinator is None: return
        preferred = self.coordinator.selected_error_model_id()
        _fill_error_model_combo(self.error_model_combo, self.coordinator, preferred)
        self._sync_training_error_path(); self._load_inherited_context()

    def _sync_training_error_path(self) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData(): return
        model = self.coordinator.registry.resolve(str(self.error_model_combo.currentData()))
        self.error_model_edit.setText(str(Path(model["path"]).resolve()))

    def _training_error_changed(self, _index: int) -> None:
        self._sync_training_error_path()

    def _load_inherited_context(self) -> None:
        if self.coordinator is None: return
        try:
            values = self.coordinator.context.training_inputs()
        except Exception as exc:
            self.summary.setText("等待推理配置：请先成功完成单样本或批量带误差推理。" + str(exc))
            if self.require_workflow: self.run_button.setEnabled(False)
            return
        self.model_edit.setText(str(values["network_model"]))
        if str(values["network_model"]) == "gpt2-small":
            self.dataset_name.setText(str(values["dataset_name"]))
            self.dataset_edit.setText(str(values["dataset_root"]))
            self.checkpoint_edit.setText(str(values["checkpoint_path"]))
            self.auxiliary_edit.setText(str(values["auxiliary_path"] or ""))
            self.summary.setText(
                "已继承 GPT-2 文本生成配置。当前训练服务只支持图像分类模型，"
                "GPT-2 误差感知微调需要文本训练语料，暂不可启动。"
            )
            self.run_button.setEnabled(False)
            return
        if str(values["network_model"]) == "tinytransformer-cifar10" and self.batch_spin.value() > 128:
            self.batch_spin.setValue(128)
        self.dataset_name.setText(str(values["dataset_name"]))
        self.dataset_edit.setText(str(values["dataset_root"]))
        self.checkpoint_edit.setText(str(values["checkpoint_path"]))
        self.auxiliary_edit.setText(str(values["auxiliary_path"] or ""))
        self.device_combo.setCurrentText(str(values["device"]))
        index = self.error_model_combo.findData(values["error_model_id"])
        if index >= 0:
            self.error_model_combo.blockSignals(True); self.error_model_combo.setCurrentIndex(index)
            self.error_model_combo.blockSignals(False); self._sync_training_error_path()
        self.summary.setText(
            f"已继承最近一次推理配置：{values['network_model']} / {values['dataset_name']}。请选择误差方法后训练。"
        )
        if str(values["network_model"]) == "tinytransformer-cifar10":
            self.summary.setText(
                self.summary.text()
                + "\nTinyTransformer 将冻结浮点骨干，仅对序列池化头和分类头进行器件差分微调。"
            )
        self.run_button.setEnabled(bool(self.error_model_combo.currentData()))

    def _browse_error_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select device error model", self.error_model_edit.text(), "JSON (*.json)"
        )
        if path:
            self.error_model_edit.setText(path)

    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select MNIST root", self.dataset_edit.text())
        if path:
            self.dataset_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _request(self, output_dir: Path | None = None) -> TrainingRequest:
        return TrainingRequest(
            project_root=self.project_root,
            error_model_path=Path(self.error_model_edit.text().strip()),
            output_dir=Path(output_dir) if output_dir else Path(self.output_edit.text().strip()),
            error_strength=self.lambda_spin.value(), epochs=self.epoch_spin.value(),
            batch_size=self.batch_spin.value(), device=self.device_combo.currentText(),
            dataset_root=Path(self.dataset_edit.text().strip()),
            model_id=self.model_edit.text().strip(),
            checkpoint_path=Path(self.checkpoint_edit.text().strip()),
            auxiliary_path=(Path(self.auxiliary_edit.text().strip())
                            if self.auxiliary_edit.text().strip() else None),
        )

    def start_training(self, output_dir: Path | None = None) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.last_outcome = None
        self.metrics_table.setRowCount(0)
        self.progress.setValue(0)
        try:
            if self.require_workflow:
                if self.coordinator is None: raise RuntimeError("工作流上下文不可用")
                self.coordinator.context.training_inputs()
                if not self.error_model_combo.currentData():
                    raise RuntimeError("请先完成误差拟合并选择误差模型")
            request = self._request(output_dir)
            self.service.validate(request)
        except Exception as exc:
            self.summary.setText("Training configuration error: " + str(exc).splitlines()[0])
            self.log.appendPlainText(traceback.format_exc())
            self._restore_controls()
            return
        self.output_edit.setText(str(request.output_dir))
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.summary.setText("Training in progress")
        self.log.appendPlainText(f"Starting full training job: {request.output_dir}")
        if request.model_id == "tinytransformer-cifar10":
            self.log.appendPlainText(
                "TinyTransformer fine-tuning: frozen backbone, trainable attention-pooling "
                "and classifier heads; the frozen backbone uses float forward."
            )
        self.worker = TrainingThread(self.service, request)
        self.worker.progressed.connect(self._training_progressed)
        self.worker.succeeded.connect(self._training_succeeded)
        self.worker.failed.connect(self._training_failed)
        self.worker.cancelled.connect(self._training_cancelled)
        self.worker.finished.connect(self._restore_controls)
        self.worker.start()

    def cancel_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.cancel_button.setEnabled(False)
            self.summary.setText("Cancellation requested; waiting for the current batch")
            self.log.appendPlainText("Cancellation requested")
            self.worker.request_cancel()

    def _training_progressed(self, percent: int, message: str, metrics: object) -> None:
        self.progress.setValue(percent)
        detail = ""
        if isinstance(metrics, dict) and "train_loss" in metrics:
            detail = f" | loss={metrics['train_loss']:.6f}, accuracy={metrics['train_accuracy']:.4f}"
        self.log.appendPlainText(message + detail)

    def _training_succeeded(self, outcome: TrainingOutcome) -> None:
        self.last_outcome = outcome
        rows = []
        for stage in ("before", "after"):
            for mode, result in outcome.evaluations[stage].items():
                rows.append((stage, mode, result["accuracy"], result["loss"]))
        self.metrics_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                shown = f"{value:.6f}" if isinstance(value, float) else str(value)
                self.metrics_table.setItem(row, column, QTableWidgetItem(shown))
        self.metrics_table.resizeColumnsToContents()
        update = outcome.weight_update
        self.summary.setText(
            f"Completed in {outcome.elapsed_seconds:.1f} s on {outcome.device}; "
            f"weight L2 delta={update['l2_delta']:.6g}; result={outcome.result_path}"
        )
        self.log.appendPlainText("Checkpoint: " + outcome.checkpoint_path)
        self.log.appendPlainText("History: " + outcome.history_path)
        self.log.appendPlainText("Result: " + outcome.result_path)
        self.progress.setValue(100)
        self.export_button.setEnabled(True)
        if self.coordinator is not None:
            self.coordinator.context.record_training(outcome.as_dict())

    def _training_failed(self, details: str) -> None:
        self.log.appendPlainText(details)
        self.summary.setText("Training failed: " + details.splitlines()[-1])

    def _training_cancelled(self, message: str) -> None:
        self.log.appendPlainText(message)
        self.summary.setText(message)

    def _restore_controls(self) -> None:
        if self.require_workflow and self.coordinator is not None:
            try:
                self.coordinator.context.training_inputs()
                ready = bool(self.error_model_combo.currentData())
            except Exception:
                ready = False
            self.run_button.setEnabled(ready)
        else:
            self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def export_result(self) -> None:
        if self.last_outcome is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export training result", "training_result.json", "JSON (*.json)"
        )
        if path:
            Path(path).write_text(
                Path(self.last_outcome.result_path).read_text(encoding="utf-8"), encoding="utf-8"
            )
            self.log.appendPlainText("Exported result: " + str(Path(path).resolve()))


class PpaEvaluationWidget(QWidget):
    """Official NeuroSim PPA evaluation in a worker thread."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        application = QApplication.instance()
        if application is not None:
            configure_windows_font(application)
        self.project_root = Path(project_root).resolve()
        self.coordinator = coordinator
        self.require_workflow = require_workflow
        self.service = PlatformService()
        self.worker: PpaThread | None = None
        self.last_outcome: PpaOutcome | None = None
        platform_root = self.project_root / "5_NeuroCIM_opt"
        layout = QVBoxLayout(self)
        description = QLabel(
            "Export signed differential traces and run the bundled official NeuroSim core. "
            "Metrics are read from NeuroSim stdout; the selected error file is validated and recorded."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItem("LeNet-5 / MNIST", "lenet5-mnist")
        form.addRow("Model", self.model_combo)
        self.checkpoint_edit = QLineEdit(
            str(platform_root / "outputs" / "eat" / "01_NMOS" / "eat_checkpoint.pt")
        )
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint_edit)
        checkpoint_button = QPushButton("Browse")
        checkpoint_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_button)
        form.addRow("Checkpoint", checkpoint_row)
        self.error_model_edit = QLineEdit(
            str(platform_root / "outputs" / "error_models" / "01_NMOS" / "error_model.json")
        )
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout()
        error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("Browse")
        error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button)
        form.addRow("Error model", error_row)
        self.subarray_rows = QSpinBox()
        self.subarray_rows.setRange(1, 4096)
        self.subarray_rows.setValue(128)
        form.addRow("Subarray rows", self.subarray_rows)
        self.parallel_rows = QSpinBox()
        self.parallel_rows.setRange(1, 4096)
        self.parallel_rows.setValue(128)
        form.addRow("Parallel rows", self.parallel_rows)
        layout.addLayout(form)

        self.run_button = QPushButton("Run official NeuroSim PPA")
        self.run_button.clicked.connect(self.start_evaluation)
        layout.addWidget(self.run_button)
        self.summary = QLabel("Not run")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["Metric", "Value"])
        layout.addWidget(self.result_table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        if self.coordinator is not None:
            _fill_error_model_combo(self.error_model_combo, self.coordinator)
            self.error_model_combo.currentIndexChanged.connect(self._ppa_error_changed)
            self.coordinator.models_changed.connect(self._refresh_ppa_models)
            self.coordinator.selection_changed.connect(self._sync_ppa_selection)
            self._sync_ppa_error_path()
        else:
            self.error_model_combo.addItem("手动选择误差模型文件", None)
            self.error_model_combo.setEnabled(False)

    def _refresh_ppa_models(self) -> None:
        if self.coordinator is None: return
        preferred = self.coordinator.selected_error_model_id()
        _fill_error_model_combo(self.error_model_combo, self.coordinator, preferred)
        self._sync_ppa_error_path()

    def _sync_ppa_selection(self, model_id: str) -> None:
        index = self.error_model_combo.findData(model_id)
        if index < 0: return
        self.error_model_combo.blockSignals(True); self.error_model_combo.setCurrentIndex(index)
        self.error_model_combo.blockSignals(False); self._sync_ppa_error_path()

    def _sync_ppa_error_path(self) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData(): return
        model = self.coordinator.registry.resolve(str(self.error_model_combo.currentData()))
        self.error_model_edit.setText(str(Path(model["path"]).resolve()))

    def _ppa_error_changed(self, _index: int) -> None:
        self._sync_ppa_error_path()

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select checkpoint", self.checkpoint_edit.text())
        if path:
            self.checkpoint_edit.setText(path)

    def _browse_error_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select error model", self.error_model_edit.text(), "JSON (*.json)"
        )
        if path:
            self.error_model_edit.setText(path)

    def start_evaluation(self, output_dir: Path | None = None) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if self.require_workflow and not self.error_model_combo.currentData():
            self.summary.setText("请先完成误差拟合并选择一种误差建模方法")
            return
        self.last_outcome = None
        self.result_table.setRowCount(0)
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)
        target = output_dir or (
            self.project_root / "5_NeuroCIM_opt" / "outputs" / "gui_runs"
            / datetime.now().strftime("%Y%m%d-%H%M%S") / "ppa"
        )
        self.log.appendPlainText(f"Starting NeuroSim PPA: {target}")
        self.worker = PpaThread(
            self.service, self.project_root, Path(self.checkpoint_edit.text()),
            Path(self.error_model_edit.text()), Path(target), self.subarray_rows.value(),
            self.parallel_rows.value(),
        )
        self.worker.succeeded.connect(self._ppa_succeeded)
        self.worker.failed.connect(self._ppa_failed)
        self.worker.finished.connect(self._ppa_finished)
        self.worker.start()

    def _ppa_succeeded(self, outcome: PpaOutcome) -> None:
        self.last_outcome = outcome
        display = [
            ("Area", outcome.metrics.get("chip_area_um2"), "um^2"),
            ("Latency", outcome.metrics.get("read_latency_ns"), "ns/image"),
            ("Dynamic energy", outcome.metrics.get("read_dynamic_energy_pj"), "pJ/image"),
            ("Energy efficiency", outcome.metrics.get("energy_efficiency_tops_per_w"), "TOPS/W"),
            ("Throughput", outcome.metrics.get("throughput_fps"), "FPS"),
        ]
        self.result_table.setRowCount(len(display))
        for row, (name, value, unit) in enumerate(display):
            shown = "unavailable" if value is None else f"{value:.6g} {unit}"
            self.result_table.setItem(row, 0, QTableWidgetItem(name))
            self.result_table.setItem(row, 1, QTableWidgetItem(shown))
        self.result_table.resizeColumnsToContents()
        self.summary.setText(
            f"Completed with official NeuroSim. JSON: {outcome.result_path} | Log: {outcome.log_path}"
        )
        self.log.appendPlainText(f"Trace directory: {outcome.trace_directory}")
        self.log.appendPlainText(f"Trace input: {outcome.trace_input}")
        self.log.appendPlainText(f"Result JSON: {outcome.result_path}")
        self.log.appendPlainText(f"NeuroSim log: {outcome.log_path}")

    def _ppa_failed(self, details: str) -> None:
        self.log.appendPlainText(details)
        self.summary.setText("PPA evaluation failed: " + details.splitlines()[-1])

    def _ppa_finished(self) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if self.last_outcome is not None else 0)
        self.run_button.setEnabled(True)


class GPT2GenerationWidget(QWidget):
    """Prompt-based GPT-2 comparison for float, ideal and fitted-device modes."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.coordinator = coordinator
        self.require_workflow = require_workflow
        self.service = GPT2GenerationService()
        self.worker: GPT2GenerationThread | None = None
        self.last_outcome: GPT2GenerationOutcome | None = None
        model_root = self.project_root / "3_models/GPT2/checkpoint"
        platform_root = self.project_root / "5_NeuroCIM_opt"

        layout = QVBoxLayout(self)
        description = QLabel(
            "输入文本提示词，对比 GPT-2 Small 的浮点、理想差分和所选器件误差差分生成结果。"
            "差分计算映射到最后一个 Transformer Block 的注意力输出投影与 MLP 输出投影。"
        )
        description.setWordWrap(True); layout.addWidget(description)
        form = QFormLayout()
        self.checkpoint_edit = QLineEdit(str(model_root / "gpt2_nanogpt.pt")); self.checkpoint_edit.setReadOnly(True)
        form.addRow("GPT-2 Checkpoint", self.checkpoint_edit)
        self.tokenizer_edit = QLineEdit(str(model_root / "tokenizer")); self.tokenizer_edit.setReadOnly(True)
        form.addRow("GPT-2 分词器", self.tokenizer_edit)
        self.scales_edit = QLineEdit(str(model_root / "input_scales.json")); self.scales_edit.setReadOnly(True)
        form.addRow("输入尺度 JSON", self.scales_edit)
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit = QLineEdit(str(platform_root / "outputs/error_models/01_NMOS/error_model.json"))
        self.error_model_edit.setReadOnly(require_workflow); form.addRow("误差模型文件", self.error_model_edit)
        self.max_tokens = QSpinBox(); self.max_tokens.setRange(1, 128); self.max_tokens.setValue(24)
        form.addRow("生成 token 数", self.max_tokens)
        self.lambda_spin = QDoubleSpinBox(); self.lambda_spin.setRange(0.0, 1.0)
        self.lambda_spin.setSingleStep(0.05); self.lambda_spin.setValue(1.0)
        form.addRow("误差强度 lambda", self.lambda_spin)
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("运行设备", self.device_combo)
        layout.addLayout(form)

        prompt_group = QGroupBox("文本提示词")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText("Computing in memory is")
        self.prompt_edit.setMaximumHeight(100); prompt_layout.addWidget(self.prompt_edit)
        layout.addWidget(prompt_group)

        controls = QHBoxLayout()
        self.run_button = QPushButton("运行 GPT-2 带误差生成"); self.run_button.clicked.connect(self.start_generation)
        self.cancel_button = QPushButton("取消"); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        controls.addWidget(self.run_button); controls.addWidget(self.cancel_button); controls.addStretch()
        layout.addLayout(controls)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); layout.addWidget(self.progress)
        self.summary = QLabel("尚未运行"); self.summary.setWordWrap(True); layout.addWidget(self.summary)

        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["模式", "生成 token", "耗时 (ms)", "token/s"])
        layout.addWidget(self.result_table)
        self.output_text = QPlainTextEdit(); self.output_text.setReadOnly(True); self.output_text.setMinimumHeight(240)
        layout.addWidget(self.output_text)
        self.top5_table = QTableWidget(0, 4)
        self.top5_table.setHorizontalHeaderLabels(["模式", "排名", "最后一步候选 token", "置信度"])
        layout.addWidget(self.top5_table)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

        if coordinator is not None:
            _fill_error_model_combo(self.error_model_combo, coordinator)
            self.error_model_combo.currentIndexChanged.connect(self._error_changed)
            coordinator.models_changed.connect(self._refresh_models)
            coordinator.selection_changed.connect(self._sync_selection)
            self._sync_error_path()
        else:
            self.error_model_combo.addItem("手动误差模型", None); self.error_model_combo.setEnabled(False)

    def _refresh_models(self) -> None:
        if self.coordinator is None: return
        _fill_error_model_combo(self.error_model_combo, self.coordinator,
                                self.coordinator.selected_error_model_id())
        self._sync_error_path()

    def _sync_selection(self, model_id: str) -> None:
        index = self.error_model_combo.findData(model_id)
        if index >= 0:
            self.error_model_combo.blockSignals(True); self.error_model_combo.setCurrentIndex(index)
            self.error_model_combo.blockSignals(False); self._sync_error_path()

    def _error_changed(self, _index: int) -> None:
        if self.coordinator is not None and self.error_model_combo.currentData():
            self.coordinator.select_error_model(str(self.error_model_combo.currentData()))
        self._sync_error_path()

    def _sync_error_path(self) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData(): return
        model = self.coordinator.registry.resolve(str(self.error_model_combo.currentData()))
        self.error_model_edit.setText(str(Path(model["path"]).resolve()))

    def _request(self) -> GPT2GenerationRequest:
        return GPT2GenerationRequest(
            project_root=self.project_root, prompt=self.prompt_edit.toPlainText(),
            checkpoint_path=Path(self.checkpoint_edit.text()),
            tokenizer_path=Path(self.tokenizer_edit.text()),
            input_scales_path=Path(self.scales_edit.text()),
            error_model_path=Path(self.error_model_edit.text()),
            max_new_tokens=self.max_tokens.value(), error_strength=self.lambda_spin.value(),
            device=self.device_combo.currentText(),
        )

    def start_generation(self) -> None:
        if self.worker is not None and self.worker.isRunning(): return
        if self.require_workflow and (
            self.coordinator is None or not self.error_model_combo.currentData()
        ):
            self.summary.setText("请先完成误差拟合并选择一种误差建模方法"); return
        request = self._request()
        try:
            self.service.validate(request)
        except Exception as exc:
            self.summary.setText("GPT-2 配置错误：" + str(exc).splitlines()[0]); return
        self.last_outcome = None; self.output_text.clear(); self.result_table.setRowCount(0)
        self.top5_table.setRowCount(0); self.progress.setValue(0)
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.summary.setText("GPT-2 三模式生成运行中")
        self.worker = GPT2GenerationThread(self.service, request)
        self.worker.progressed.connect(self._progressed); self.worker.succeeded.connect(self._succeeded)
        self.worker.failed.connect(self._failed); self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._finished); self.worker.start()

    def cancel_generation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel(); self.cancel_button.setEnabled(False)

    def _progressed(self, percent: int, message: str) -> None:
        self.progress.setValue(percent); self.summary.setText(message)

    def _succeeded(self, outcome: GPT2GenerationOutcome) -> None:
        self.last_outcome = outcome
        self.result_table.setRowCount(len(outcome.modes))
        top_rows = sum(len(mode.final_top5) for mode in outcome.modes)
        self.top5_table.setRowCount(top_rows)
        top_row = 0; sections = []
        for row, mode in enumerate(outcome.modes):
            values = (mode.mode, len(mode.generated_token_ids), f"{mode.elapsed_ms:.2f}",
                      f"{mode.tokens_per_second:.2f}")
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(str(value)))
            sections.append(f"[{mode.mode}]\n{mode.text}")
            for rank, (token, confidence) in enumerate(mode.final_top5, 1):
                for column, value in enumerate((mode.mode, rank, repr(token), f"{confidence:.2%}")):
                    self.top5_table.setItem(top_row, column, QTableWidgetItem(str(value)))
                top_row += 1
        self.output_text.setPlainText("\n\n".join(sections))
        self.result_table.resizeColumnsToContents(); self.top5_table.resizeColumnsToContents()
        output = self.project_root / "5_NeuroCIM_opt/outputs/gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "gpt2"
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "generation_result.json"
        result_path.write_text(json.dumps(outcome.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.summary.setText(
            f"完成：提示词 {outcome.prompt_token_count} tokens，三模式总耗时 {outcome.total_elapsed_ms:.1f} ms"
        )
        self.log.appendPlainText(f"结果：{result_path}")
        if self.coordinator is not None and self.error_model_combo.currentData():
            try:
                self.coordinator.record_inference(
                    model_id=str(self.error_model_combo.currentData()), dataset_name="prompt_text",
                    dataset_root=self.tokenizer_edit.text(), network_model="gpt2-small",
                    checkpoint_path=self.checkpoint_edit.text(), auxiliary_path=self.scales_edit.text(),
                    device=outcome.device, result=outcome.as_dict(),
                )
                self.log.appendPlainText("已记录当前 GPT-2 模型、文本任务和误差模型")
            except Exception as exc:
                self.log.appendPlainText("工作流记录失败：" + str(exc))

    def _failed(self, trace: str) -> None:
        self.summary.setText("GPT-2 生成失败"); self.log.appendPlainText(trace)

    def _cancelled(self, message: str) -> None:
        self.summary.setText(message); self.log.appendPlainText(message)

    def _finished(self) -> None:
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)


class GPT2BatchInferenceWidget(QWidget):
    """Batch prompt evaluation selected through the shared model catalog."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.coordinator = coordinator
        self.require_workflow = require_workflow
        self.service = GPT2GenerationService()
        self.worker: GPT2BatchThread | None = None
        checkpoint_root = self.project_root / "3_models/GPT2/checkpoint"
        platform_root = self.project_root / "5_NeuroCIM_opt"
        layout = QVBoxLayout(self)
        description = QLabel(
            "对多条提示词执行浮点、理想差分和器件误差差分生成，统计器件输出相对浮点输出的 token 一致率。"
        )
        description.setWordWrap(True); layout.addWidget(description)
        form = QFormLayout()
        self.checkpoint_edit = QLineEdit(str(checkpoint_root / "gpt2_nanogpt.pt")); self.checkpoint_edit.setReadOnly(True)
        self.tokenizer_edit = QLineEdit(str(checkpoint_root / "tokenizer")); self.tokenizer_edit.setReadOnly(True)
        self.scales_edit = QLineEdit(str(checkpoint_root / "input_scales.json")); self.scales_edit.setReadOnly(True)
        form.addRow("继承的模型", QLabel("GPT-2 Small + prompt_text"))
        form.addRow("Checkpoint", self.checkpoint_edit)
        form.addRow("分词器", self.tokenizer_edit)
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit = QLineEdit(str(platform_root / "outputs/error_models/01_NMOS/error_model.json"))
        self.error_model_edit.setReadOnly(require_workflow); form.addRow("误差模型", self.error_model_edit)
        self.max_tokens = QSpinBox(); self.max_tokens.setRange(1, 64); self.max_tokens.setValue(8)
        form.addRow("每条生成 token 数", self.max_tokens)
        self.lambda_spin = QDoubleSpinBox(); self.lambda_spin.setRange(0.0, 1.0)
        self.lambda_spin.setSingleStep(0.05); self.lambda_spin.setValue(1.0)
        form.addRow("误差强度 lambda", self.lambda_spin)
        self.device_combo = QComboBox(); self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("运行设备", self.device_combo); layout.addLayout(form)
        self.prompts_edit = QPlainTextEdit()
        self.prompts_edit.setPlainText(
            "Computing in memory is\nArtificial intelligence can\nThe future of hardware"
        )
        self.prompts_edit.setMaximumHeight(130)
        layout.addWidget(QLabel("批量提示词（每行一条，最多 20 条）")); layout.addWidget(self.prompts_edit)
        controls = QHBoxLayout()
        self.run_button = QPushButton("运行 GPT-2 批量带误差推理"); self.run_button.clicked.connect(self.start_inference)
        self.cancel_button = QPushButton("取消"); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_inference)
        controls.addWidget(self.run_button); controls.addWidget(self.cancel_button); controls.addStretch()
        layout.addLayout(controls)
        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.summary = QLabel("尚未运行"); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(
            ["提示词", "浮点生成", "器件误差生成", "Token 一致率", "三模式总耗时 (ms)"]
        )
        layout.addWidget(self.result_table)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100)
        layout.addWidget(self.log)
        if coordinator is not None:
            _fill_error_model_combo(self.error_model_combo, coordinator)
            self.error_model_combo.currentIndexChanged.connect(self._error_changed)
            coordinator.models_changed.connect(self._refresh_models)
            coordinator.selection_changed.connect(self._sync_error_selection)
            self._sync_error_path()
        else:
            self.error_model_combo.addItem("手动误差模型", None); self.error_model_combo.setEnabled(False)

    def _refresh_models(self) -> None:
        if self.coordinator is None: return
        _fill_error_model_combo(self.error_model_combo, self.coordinator,
                                self.coordinator.selected_error_model_id())
        self._sync_error_path()

    def _sync_error_selection(self, model_id: str) -> None:
        index = self.error_model_combo.findData(model_id)
        if index >= 0:
            self.error_model_combo.blockSignals(True); self.error_model_combo.setCurrentIndex(index)
            self.error_model_combo.blockSignals(False); self._sync_error_path()

    def _error_changed(self, _index: int) -> None:
        if self.coordinator is not None and self.error_model_combo.currentData():
            self.coordinator.select_error_model(str(self.error_model_combo.currentData()))
        self._sync_error_path()

    def _sync_error_path(self) -> None:
        if self.coordinator is None or not self.error_model_combo.currentData(): return
        model = self.coordinator.registry.resolve(str(self.error_model_combo.currentData()))
        self.error_model_edit.setText(str(Path(model["path"]).resolve()))

    def _requests(self) -> tuple[GPT2GenerationRequest, ...]:
        prompts = tuple(line.strip() for line in self.prompts_edit.toPlainText().splitlines() if line.strip())
        if not prompts:
            raise ValueError("至少输入一条提示词")
        if len(prompts) > 20:
            raise ValueError("批量提示词不能超过 20 条")
        return tuple(GPT2GenerationRequest(
            project_root=self.project_root, prompt=prompt,
            checkpoint_path=Path(self.checkpoint_edit.text()),
            tokenizer_path=Path(self.tokenizer_edit.text()),
            input_scales_path=Path(self.scales_edit.text()),
            error_model_path=Path(self.error_model_edit.text()),
            max_new_tokens=self.max_tokens.value(), error_strength=self.lambda_spin.value(),
            device=self.device_combo.currentText(),
        ) for prompt in prompts)

    def start_inference(self) -> None:
        if self.worker is not None and self.worker.isRunning(): return
        try:
            if self.require_workflow and (self.coordinator is None or not self.error_model_combo.currentData()):
                raise RuntimeError("请先完成误差拟合并选择一种误差模型")
            requests = self._requests()
            for request in requests: self.service.validate(request)
        except Exception as exc:
            self.summary.setText("批量推理配置错误：" + str(exc).splitlines()[0]); return
        self.result_table.setRowCount(0); self.progress.setValue(0)
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.summary.setText(f"正在处理 {len(requests)} 条提示词")
        self.worker = GPT2BatchThread(self.service, requests)
        self.worker.progressed.connect(lambda percent, message: (self.progress.setValue(percent), self.summary.setText(message)))
        self.worker.succeeded.connect(self._succeeded); self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled); self.worker.finished.connect(self._finished)
        self.worker.start()

    def cancel_inference(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel(); self.cancel_button.setEnabled(False)

    @staticmethod
    def _agreement(reference: tuple[int, ...], candidate: tuple[int, ...]) -> float:
        count = max(len(reference), len(candidate), 1)
        return sum(a == b for a, b in zip(reference, candidate)) / count

    def _succeeded(self, outcomes: tuple[GPT2GenerationOutcome, ...]) -> None:
        self.result_table.setRowCount(len(outcomes)); agreements = []
        payload = []
        for row, outcome in enumerate(outcomes):
            modes = {item.mode: item for item in outcome.modes}
            floating = modes["float"]; device = modes["device_differential"]
            agreement = self._agreement(floating.generated_token_ids, device.generated_token_ids)
            agreements.append(agreement)
            values = (outcome.prompt, floating.text, device.text, f"{agreement:.2%}",
                      f"{outcome.total_elapsed_ms:.1f}")
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(str(value)))
            payload.append(outcome.as_dict())
        self.result_table.resizeColumnsToContents()
        average = sum(agreements) / len(agreements)
        output = self.project_root / "5_NeuroCIM_opt/outputs/gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "gpt2_batch"
        output.mkdir(parents=True, exist_ok=True); result_path = output / "batch_result.json"
        result_path.write_text(json.dumps({"average_device_token_agreement": average,
                                           "samples": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.summary.setText(f"完成：{len(outcomes)} 条提示词，器件/浮点平均 token 一致率 {average:.2%}")
        self.log.appendPlainText("结果：" + str(result_path))
        if self.coordinator is not None and self.error_model_combo.currentData():
            try:
                self.coordinator.record_inference(
                    model_id=str(self.error_model_combo.currentData()), dataset_name="prompt_text",
                    dataset_root=self.tokenizer_edit.text(), network_model="gpt2-small",
                    checkpoint_path=self.checkpoint_edit.text(), auxiliary_path=self.scales_edit.text(),
                    device=outcomes[0].device,
                    result={"average_device_token_agreement": average, "samples": payload},
                )
            except Exception as exc:
                self.log.appendPlainText("工作流记录失败：" + str(exc))

    def _failed(self, trace: str) -> None:
        self.summary.setText("GPT-2 批量推理失败"); self.log.appendPlainText(trace)

    def _cancelled(self, message: str) -> None:
        self.summary.setText(message); self.log.appendPlainText(message)

    def _finished(self) -> None:
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        if self.result_table.rowCount(): self.progress.setValue(100)


class UnifiedSingleInferenceWidget(QWidget):
    """Switch inference input/output by the model selected in the catalog."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        if coordinator is None:
            raise ValueError("Unified inference requires a workflow coordinator")
        self.coordinator = coordinator
        layout = QVBoxLayout(self)
        self.selection = QLabel(); self.selection.setWordWrap(True); layout.addWidget(self.selection)
        self.stack = QStackedWidget(); layout.addWidget(self.stack)
        self.image_widget = SingleInferenceWidget(
            project_root, coordinator=coordinator, require_workflow=require_workflow
        )
        self.image_widget.model_combo.setEnabled(False)
        self.gpt_widget = GPT2GenerationWidget(
            project_root, coordinator=coordinator, require_workflow=require_workflow
        )
        self.stack.addWidget(self.image_widget); self.stack.addWidget(self.gpt_widget)
        coordinator.network_changed.connect(self.apply_model_selection)
        self.apply_model_selection(coordinator.selected_network_model_id())

    def apply_model_selection(self, model_id: str) -> None:
        spec = self.coordinator.model_registry.get(model_id)
        self.selection.setText(
            f"继承“网络与数据集”配置：{spec.display_name} / {', '.join(spec.datasets)}"
        )
        if spec.task == "text_generation":
            self.stack.setCurrentWidget(self.gpt_widget)
        else:
            index = self.image_widget.model_combo.findData(model_id)
            if index >= 0: self.image_widget.model_combo.setCurrentIndex(index)
            self.stack.setCurrentWidget(self.image_widget)


class UnifiedBatchInferenceWidget(QWidget):
    """Switch grouped image or prompt-batch inference from the shared selection."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        if coordinator is None:
            raise ValueError("Unified inference requires a workflow coordinator")
        self.coordinator = coordinator
        layout = QVBoxLayout(self)
        self.selection = QLabel(); self.selection.setWordWrap(True); layout.addWidget(self.selection)
        self.stack = QStackedWidget(); layout.addWidget(self.stack)
        self.image_widget = BatchInferenceWidget(
            project_root, coordinator=coordinator, require_workflow=require_workflow
        )
        self.image_widget.model_combo.setEnabled(False)
        self.gpt_widget = GPT2BatchInferenceWidget(
            project_root, coordinator=coordinator, require_workflow=require_workflow
        )
        self.stack.addWidget(self.image_widget); self.stack.addWidget(self.gpt_widget)
        coordinator.network_changed.connect(self.apply_model_selection)
        self.apply_model_selection(coordinator.selected_network_model_id())

    def apply_model_selection(self, model_id: str) -> None:
        spec = self.coordinator.model_registry.get(model_id)
        self.selection.setText(
            f"继承“网络与数据集”配置：{spec.display_name} / {', '.join(spec.datasets)}"
        )
        if spec.task == "text_generation":
            self.stack.setCurrentWidget(self.gpt_widget)
        else:
            index = self.image_widget.model_combo.findData(model_id)
            if index >= 0: self.image_widget.model_combo.setCurrentIndex(index)
            self.stack.setCurrentWidget(self.image_widget)


class MainWindow(QMainWindow):
    def __init__(self, platform_root: Path):
        super().__init__()
        application = QApplication.instance()
        if application is not None:
            configure_windows_font(application)
            apply_application_style(application)
        self.setWindowTitle("DRAM 存算仿真平台 — NeuroSim")
        self.setMinimumSize(720, 420)
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(1180, int(available.width() * 0.92)), min(780, int(available.height() * 0.88)))
        else:
            self.resize(1180, 780)

        self.workflow = WorkflowCoordinator(platform_root)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.analysis_tabs = QTabWidget(self.tabs)
        self.analysis_tabs.setObjectName("analysisTabs")
        self.analysis_tabs.setDocumentMode(True)
        self.analysis_tabs.addTab(
            self._scrollable(CellFittingWidget(platform_root, coordinator=self.workflow)),
            "器件误差拟合",
        )
        self.analysis_tabs.addTab(self._scrollable(MacAnalysisWidget(platform_root)), "阵列 MAC 分析")
        self.tabs.addTab(self.analysis_tabs, "误差建模与阵列分析")
        self.tabs.addTab(
            self._scrollable(ModelCatalogWidget(platform_root.parent, coordinator=self.workflow)),
            "网络与数据集",
        )
        self.tabs.addTab(
            self._scrollable(UnifiedSingleInferenceWidget(
                platform_root.parent, coordinator=self.workflow, require_workflow=True
            )), "单样本推理"
        )
        self.tabs.addTab(
            self._scrollable(UnifiedBatchInferenceWidget(
                platform_root.parent, coordinator=self.workflow, require_workflow=True
            )), "批量推理"
        )
        self.tabs.addTab(
            self._scrollable(ErrorAwareTrainingWidget(
                platform_root.parent, coordinator=self.workflow, require_workflow=True
            )), "误差感知训练"
        )
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("就绪")
        self._apply_visual_roles()

    def _apply_visual_roles(self) -> None:
        """Assign visual roles and table defaults without changing behavior."""
        primary_prefixes = (
            "开始", "运行", "预览", "生成", "执行", "导出", "保存",
            "Start", "Run", "Preview", "Generate", "Export", "Save",
        )
        for button in self.findChildren(QPushButton):
            text = button.text().strip()
            if text.startswith("取消"):
                button.setProperty("role", "danger")
            elif text.startswith(primary_prefixes):
                button.setProperty("role", "primary")
            button.style().unpolish(button)
            button.style().polish(button)

        for table in self.findChildren(QTableWidget):
            table.setAlternatingRowColors(True)
            table.setShowGrid(True)
            table.verticalHeader().setDefaultSectionSize(30)
            table.horizontalHeader().setMinimumHeight(32)
            table.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """Keep every page usable when the Windows taskbar reduces available height."""
        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        page.setObjectName("pageContent")
        if page.layout() is not None:
            page.layout().setContentsMargins(18, 16, 18, 20)
            page.layout().setSpacing(10)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.verticalScrollBar().setSingleStep(24)
        scroll.horizontalScrollBar().setSingleStep(24)
        scroll.setWidget(page)
        return scroll


class ModelCatalogWidget(QWidget):
    """Live view of registered model/dataset combinations and resource readiness."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None):
        super().__init__(parent)
        self.coordinator = coordinator
        self.specs = builtin_registry(project_root).list()
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "在这里勾选一个模型与数据集组合。单样本推理、批量推理和后续训练将统一继承该选择；"
            "后续页面不再单独切换网络。没有 checkpoint 的组合不能勾选。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.table = QTableWidget()
        columns = ["使用", "模型 ID", "名称", "任务", "数据集", "输入", "指标", "Checkpoint"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(self.specs))
        self.selection_group = QButtonGroup(self)
        self.selection_group.setExclusive(True)
        selected = coordinator.selected_network_model_id() if coordinator else self.specs[0].model_id
        for row, spec in enumerate(self.specs):
            selector = QCheckBox()
            selector.setEnabled(spec.checkpoint_available)
            selector.setChecked(spec.model_id == selected)
            selector.setToolTip("选择后，推理与训练页面统一使用此组合")
            self.selection_group.addButton(selector, row)
            self.table.setCellWidget(row, 0, selector)
            selector.clicked.connect(lambda checked, model_id=spec.model_id:
                                     self._select_model(model_id) if checked else None)
            values = [
                spec.model_id, spec.display_name, spec.task, ", ".join(spec.datasets),
                "×".join(map(str, spec.input_shape)), ", ".join(spec.metrics),
                "可用" if spec.checkpoint_available else "未安装",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column + 1, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        self.selection_status = QLabel()
        self.selection_status.setWordWrap(True)
        layout.addWidget(self.selection_status)
        if coordinator is not None:
            coordinator.network_changed.connect(self._sync_selection)
            selected_spec = next(item for item in self.specs if item.model_id == selected)
            if (coordinator.context.selected_network_model() is None
                    and selected_spec.checkpoint_available):
                coordinator.select_network_model(selected)
            if selected_spec.checkpoint_available:
                self._sync_selection(coordinator.selected_network_model_id())
            else:
                self.selection_status.setText(
                    "尚未安装可用的模型 checkpoint。请按发布包中的《外部资源配置说明》"
                    "放置模型文件，然后重新启动软件。"
                )

    def _select_model(self, model_id: str) -> None:
        if self.coordinator is None:
            self.selection_status.setText(f"当前组合：{model_id}")
            return
        try:
            self.coordinator.select_network_model(model_id)
        except Exception as exc:
            self.selection_status.setText("选择失败：" + str(exc))
            self._sync_selection(self.coordinator.selected_network_model_id())

    def _sync_selection(self, model_id: str) -> None:
        for row, spec in enumerate(self.specs):
            widget = self.table.cellWidget(row, 0)
            if isinstance(widget, QCheckBox):
                widget.blockSignals(True)
                widget.setChecked(spec.model_id == model_id)
                widget.blockSignals(False)
        spec = next(item for item in self.specs if item.model_id == model_id)
        dataset = ", ".join(spec.datasets)
        self.selection_status.setText(
            f"当前统一配置：{spec.display_name}　|　数据集：{dataset}　|　任务：{spec.task}"
        )


def run_gui(platform_root: str | Path) -> int:
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("DRAM-CIM NeuroSim Platform")
    configure_windows_font(application)
    apply_application_style(application)
    resolved_platform = Path(platform_root).expanduser().resolve()
    gpt_smoke_result = os.environ.get("NEUROCIM_RELEASE_GPT_SMOKE")
    if gpt_smoke_result:
        target = Path(gpt_smoke_result).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            from transformers import GPT2Tokenizer
            tokenizer_root = resolved_platform.parent / "3_models/GPT2/checkpoint/tokenizer"
            tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_root))
            token_ids = tokenizer.encode("NeuroCIM release smoke")
            if not token_ids:
                raise RuntimeError("GPT-2 tokenizer produced no token IDs")
            target.write_text(f"ok: {len(token_ids)} token IDs\n", encoding="utf-8")
            return 0
        except Exception:
            target.write_text(traceback.format_exc(), encoding="utf-8")
            return 3
    window = MainWindow(resolved_platform)
    window.show()
    smoke_screenshot = os.environ.get("NEUROCIM_RELEASE_SMOKE_SCREENSHOT")
    if smoke_screenshot:
        application.processEvents()
        target = Path(smoke_screenshot).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        saved = window.grab().save(str(target))
        window.close()
        return 0 if saved else 2
    return application.exec()
