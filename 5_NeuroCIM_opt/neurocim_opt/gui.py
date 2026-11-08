from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
from time import monotonic
from typing import Any

import numpy as np

from PySide6.QtCore import QEvent, QObject, QSize, QSettings, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap, QPixmapCache
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox as _QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
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
from .gpt2_evaluation import (
    WikiText2EvaluationCancelled, WikiText2EvaluationOutcome,
    WikiText2EvaluationRequest, WikiText2EvaluationService,
)
from .mac_error_model import MacErrorModel, load_mac_csv
from .paths import data_root
from .training_service import (
    ErrorAwareTrainingService, TrainingCancelled, TrainingOutcome, TrainingRequest,
)
from .workflow import ErrorModelRegistry, WorkflowContext
from .paths import output_root
from .release_config import ACCESS_PORTAL_URL
from .updater import (
    AvailableUpdate, CredentialStore, GitHubReleaseClient, UpdateError,
    installed_flavor, launch_installer,
)
from .version import __version__


VISION_ZOO_MODEL_IDS = frozenset({
    "resnet8-mnist", "mobilenetv2-cifar10", "vgg11-bn-cifar10",
    "squeezenet1_1-cifar10",
})
FRONTEND_MODEL_IDS = (
    "lenet5-mnist",
    "resnet8-mnist",
    "mobilenetv2-cifar10",
    "vgg11-bn-cifar10",
    "squeezenet1_1-cifar10",
    "bnn-mnist",
    "tinytransformer-cifar10",
    "gpt2-large",
)
FRONTEND_MODEL_ID_SET = frozenset(FRONTEND_MODEL_IDS)
RUNTIME_CALIBRATED_MODEL_IDS = frozenset({"bnn-mnist"})
NOISE_LEVELS = (
    ("档位 1：关闭附加随机噪声", (0.0, 0.0)),
    ("档位 2：弱独立噪声（σ=0.005）", (0.005, 0.0)),
    ("档位 3：中等独立噪声（σ=0.010）", (0.010, 0.0)),
    ("档位 4：较强独立噪声（σ=0.020）", (0.020, 0.0)),
    ("档位 5：强独立噪声（σ=0.040）", (0.040, 0.0)),
    ("档位 6：中等相关噪声（σ=0.010，ρ=0.05）", (0.010, 0.05)),
)


def _setup_noise_combo(combo: QComboBox) -> None:
    for label, value in NOISE_LEVELS:
        combo.addItem(label, value)


def _setup_adc_combo(combo: QComboBox) -> None:
    combo.addItem("理想 ADC（关闭量化）", None)
    for bits in (4, 6, 8, 10):
        combo.addItem(f"{bits} bit ADC", bits)


def _populate_image_model_combo(combo: _QComboBox, project_root: Path, capability: str):
    """Populate a selector from registry capability and actual checkpoint state."""
    registry = builtin_registry(project_root)
    for spec in _frontend_model_specs(
        registry, task="image_classification", capability=capability
    ):
        combo.addItem(spec.display_name, spec.model_id)
        item = combo.model().item(combo.count() - 1)
        if item is not None and not spec.checkpoint_available:
            item.setEnabled(False)
            item.setToolTip(f"正式 Checkpoint 尚未安装：{spec.checkpoint}")
    return registry


def _frontend_model_specs(registry, *, task: str | None = None,
                          capability: str | None = None):
    """Return the deliberately small, ordered model list exposed by the GUI."""

    specs = []
    for model_id in FRONTEND_MODEL_IDS:
        spec = registry.get(model_id)
        if task is not None and spec.task != task:
            continue
        if capability is not None and not spec.supports(capability):
            continue
        specs.append(spec)
    return tuple(specs)


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


class UpdateCheckThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, platform_root: Path):
        super().__init__()
        self.platform_root = platform_root

    def run(self) -> None:
        try:
            client = GitHubReleaseClient()
            update = client.latest_update(
                flavor=installed_flavor(self.platform_root), current=__version__
            )
            self.succeeded.emit(update)
        except Exception as exc:
            self.failed.emit(str(exc))


class UpdateDownloadThread(QThread):
    progressed = Signal(str, int)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, update: AvailableUpdate):
        super().__init__()
        self.update = update

    def run(self) -> None:
        try:
            destination = output_root().parent / "updates" / self.update.version
            client = GitHubReleaseClient()

            def report(name: str, done: int, total: int) -> None:
                percent = round(done * 100 / total) if total else 0
                self.progressed.emit(name, percent)

            installer = client.download_update_package(
                self.update, destination, progress=report
            )
            self.succeeded.emit(str(installer))
        except Exception as exc:
            self.failed.emit(str(exc))


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


class CollapsiblePanel(QWidget):
    """Compact native advanced-options panel; content is hidden by default."""

    def __init__(self, title: str = "高级选项", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(6)
        self.toggle = QPushButton(f"{title}  ▾")
        self.toggle.setCheckable(True); self.toggle.setProperty("advancedToggle", True)
        self.content = QScrollArea(); self.content.setObjectName("advancedContent")
        self.content.setWidgetResizable(True); self.content.setFrameShape(QScrollArea.Shape.NoFrame)
        self.content.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.content.setMaximumHeight(410)
        self.content_widget = QFrame(); self.content_widget.setObjectName("advancedContentBody")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(12, 10, 12, 10)
        self.content.setWidget(self.content_widget)
        self.content.hide(); layout.addWidget(self.toggle); layout.addWidget(self.content)
        self.toggle.toggled.connect(self.content.setVisible)
        self.toggle.toggled.connect(
            lambda checked: self.toggle.setText(f"{title}  {'▴' if checked else '▾'}")
        )


class ResponsivePixmapLabel(QLabel):
    """Keep a high-resolution chart fitted to the currently available card area."""

    def __init__(self, placeholder: str = "", parent: QWidget | None = None):
        super().__init__(placeholder, parent)
        self._image_path: Path | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_image(self, path: str | Path) -> None:
        self._image_path = Path(path)
        self.setToolTip(str(self._image_path))
        self._refresh_pixmap()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._image_path is not None:
            QTimer.singleShot(0, self._refresh_pixmap)

    def _refresh_pixmap(self) -> None:
        if self._image_path is None or not self._image_path.is_file():
            return
        target = QSize(max(240, self.width() - 8), max(150, self.height() - 8))
        self.setPixmap(_high_dpi_pixmap(self._image_path, target, self))


def _metric_card(title: str, initial: str = "--") -> tuple[QFrame, QLabel]:
    card = QFrame(); card.setProperty("metricCard", True)
    layout = QVBoxLayout(card); layout.setContentsMargins(13, 10, 13, 10); layout.setSpacing(2)
    caption = QLabel(title); caption.setProperty("metricCaption", True)
    value = QLabel(initial); value.setProperty("metricValue", True)
    layout.addWidget(caption); layout.addWidget(value)
    return card, value


def _add_independent_columns(
    host_layout: QVBoxLayout,
    left_card: QWidget,
    right_card: QWidget,
) -> tuple[QScrollArea, QScrollArea]:
    """Install 2:3 work columns with independent vertical scrolling."""
    columns = QWidget()
    columns.setObjectName("independentColumns")
    columns_layout = QHBoxLayout(columns)
    columns_layout.setContentsMargins(0, 0, 0, 0)
    columns_layout.setSpacing(12)

    scrolls: list[QScrollArea] = []
    for name, card, stretch in (
        ("leftColumnScroll", left_card, 2),
        ("rightColumnScroll", right_card, 3),
    ):
        card.setMinimumWidth(0)
        card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        for line_edit in card.findChildren(QLineEdit):
            line_edit.setMinimumWidth(0)
            line_edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for combo in card.findChildren(_QComboBox):
            combo.setMinimumWidth(0)
            combo.setMinimumContentsLength(8)
            combo.setSizeAdjustPolicy(_QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for form in card.findChildren(QFormLayout):
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        canvas = QWidget()
        canvas.setMinimumSize(0, 0)
        canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(card, 0, Qt.AlignmentFlag.AlignTop)
        canvas_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setObjectName(name)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.verticalScrollBar().setSingleStep(24)
        scroll.setWidget(canvas)
        columns_layout.addWidget(scroll, stretch)
        scrolls.append(scroll)
    host_layout.addWidget(columns, 1)
    return scrolls[0], scrolls[1]


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
        # Frozen applications can live below Program Files or another read-only
        # location.  Keep mutable workflow state in the per-user output root;
        # bundled datasets and checkpoints are still resolved beside the EXE.
        state_root = (
            output_root() / "workflow"
            if getattr(sys, "frozen", False)
            else Path(platform_root).resolve() / "outputs" / "workflow"
        )
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
                spec = self.model_registry.get(str(selected["model_id"]))
                if spec.model_id in FRONTEND_MODEL_ID_SET and spec.resources_available:
                    return spec.model_id
            except KeyError:
                pass
        visible = _frontend_model_specs(self.model_registry)
        ready = [item for item in visible if item.resources_available]
        return (ready[0] if ready else visible[0]).model_id

    def select_network_model(self, model_id: str) -> dict[str, Any]:
        spec = self.model_registry.get(model_id)
        if not spec.resources_available:
            raise FileNotFoundError(f"模型 Checkpoint 未安装：{spec.checkpoint}")
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
            application.setFont(QFont(families[0], 11))


APPLICATION_STYLE_SHEET = """
QMainWindow, QWidget {
    background-color: #F4F7FB;
    color: #172033;
}
QFrame#topBar { background-color: #172231; border: none; }
QLabel#brandTitle { color: #FFFFFF; font-size: 17px; font-weight: 700; }
QLabel#brandSubtitle, QLabel#topStatus { color: #AFC0D4; font-size: 10px; }
QLabel#brandMark { min-width: 40px; min-height: 40px; max-width: 40px; max-height: 40px; border-radius: 9px; background-color: #1976C9; color: #FFFFFF; font-size: 20px; font-weight: 700; }
QPushButton[topTool="true"] { min-height: 34px; border: 1px solid #3B4A5E; border-radius: 7px; background-color: #223044; color: #E6EDF5; }
QPushButton[topTool="true"]:hover { background-color: #2C3D54; border-color: #58708D; color: #FFFFFF; }
QFrame#sideBar { background-color: #FFFFFF; border-right: 1px solid #DCE4ED; }
QLabel#navCaption { color: #8795A8; font-size: 10px; font-weight: 600; }
QPushButton[nav="true"] { min-height: 43px; padding: 0 12px; border: none; border-left: 3px solid transparent; border-radius: 7px; background: transparent; color: #47586E; text-align: left; }
QPushButton[nav="true"]:hover { background-color: #F1F6FC; color: #126DBF; }
QPushButton[nav="true"]:checked { background-color: #E5F1FC; border-left: 3px solid #1976C9; color: #0867B9; font-weight: 700; }
QFrame#sideStatus { border: 1px solid #D7E2EE; border-radius: 9px; background-color: #F5F9FD; }
QLabel#homeTitle { color: #172231; font-size: 24px; font-weight: 700; }
QLabel#contentTitle { color: #172231; font-size: 22px; font-weight: 700; }
QLabel#contentSubtitle { color: #687A91; }
QLabel#homeHero { color: #126DBF; font-size: 13px; font-weight: 700; }
QLabel[muted="true"] { color: #687A91; }
QFrame[dashboardCard="true"] { border: 1px solid #D7E2EE; border-radius: 10px; background-color: #FFFFFF; }
QFrame#flowBar { background-color: #FFFFFF; border-bottom: 1px solid #D7E2EE; }
QLabel#flowTitle { color: #53657B; font-weight: 700; padding-right: 8px; }
QPushButton[flowStep="true"] { min-height: 34px; padding: 0 15px; border: 1px solid #D6E0EB; border-radius: 17px; background: #F7F9FC; color: #526477; }
QPushButton[flowStep="true"]:hover { border-color: #82AFE6; color: #126DBF; background: #F0F7FF; }
QPushButton[flowStep="true"]:checked { border-color: #1976C9; color: #126DBF; background: #E7F2FC; font-weight: 700; }
QPushButton[advancedToggle="true"] { min-height: 32px; border: none; background: transparent; color: #526477; }
QPushButton[advancedToggle="true"]:hover, QPushButton[advancedToggle="true"]:checked { color: #126DBF; background: #EEF6FE; }
QFrame#advancedNavigation { background: #F7FAFD; border-bottom: 1px solid #D7E2EE; }
QFrame#fitModePanel { background: #FFFFFF; border-right: 1px solid #D7E2EE; }
QPushButton[fitMode="true"] { min-height: 42px; border: 1px solid #D7E2EE; border-radius: 7px; background: #F7F9FC; color: #526477; text-align: left; padding-left: 12px; }
QPushButton[fitMode="true"]:checked { border-color: #1976C9; background: #E7F2FC; color: #126DBF; font-weight: 700; }
QPushButton[advancedNav="true"] { min-height: 30px; border: 1px solid #D6E0EB; border-radius: 6px; background: #FFFFFF; color: #526477; }
QPushButton[advancedNav="true"]:checked { border-color: #1976C9; color: #126DBF; background: #E7F2FC; }
QFrame#rememberedConfig { border: 1px solid #CFE3F6; border-radius: 8px; background: #EFF7FF; }
QFrame[entryCard="true"] { border: 1px solid #D7E2EE; border-radius: 11px; background-color: #FFFFFF; }
QFrame[entryCard="true"]:hover { border-color: #9EC2E8; }
QFrame[workCard="true"] { border: 1px solid #D7E2EE; border-radius: 11px; background-color: #FFFFFF; }
QScrollArea#advancedContent, QFrame#advancedContentBody { border: 1px solid #DCE5EF; border-radius: 8px; background: #F7FAFD; }
QFrame[metricCard="true"] { border: 1px solid #CFE0F2; border-radius: 9px; background: #F2F8FE; }
QLabel[metricCaption="true"] { color: #5E7189; font-size: 13px; font-weight: 600; }
QLabel[metricValue="true"] { color: #126DBF; font-size: 30px; font-weight: 750; }
QLabel[sectionTitle="true"] { color: #24364A; font-size: 16px; font-weight: 700; padding: 2px 0 7px 0; }
QLabel[sectionHint="true"] { color: #687A91; padding-bottom: 4px; }
QLabel#entryIcon { color: #1976C9; font-size: 30px; font-weight: 700; }
QLabel#entryTitle { color: #24364A; font-size: 20px; font-weight: 700; }
QLabel#entryMeta { color: #526477; background: #F5F8FB; border-radius: 6px; padding: 9px; }
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


def _fit_result_table_height(
    table: QTableWidget, *, max_visible_rows: int = 8, empty_rows: int = 1
) -> None:
    """Size short result tables to their content and scroll only long results."""
    row_count = table.rowCount()
    visible_rows = min(max_visible_rows, max(row_count, empty_rows))
    row_height = max(30, table.verticalHeader().defaultSectionSize())
    header_height = max(32, table.horizontalHeader().height())
    height = header_height + visible_rows * row_height + table.frameWidth() * 2 + 4
    table.setFixedHeight(height)
    table.setVerticalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
        if row_count > max_visible_rows else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )


def _mode_display_name(mode: str) -> str:
    return {
        "float": "浮点模型",
        "ideal_differential": "理想差分",
        "device_differential": "器件误差差分",
        "baseline float": "训练前浮点模型",
        "baseline ideal differential": "训练前理想差分",
        "baseline device differential": "训练前器件误差差分",
        "trained float": "训练后浮点模型",
        "trained ideal differential": "训练后理想差分",
        "trained device differential": "训练后器件误差差分",
    }.get(mode, mode)


def _stage_display_name(stage: str) -> str:
    return {"before": "训练前", "after": "训练后"}.get(stage, stage)


def _task_display_name(task: str) -> str:
    return {
        "image_classification": "图像分类",
        "text_generation": "文本生成",
        "language_modeling": "语言建模",
    }.get(task, task)


def _column_display_name(name: str) -> str:
    return {
        "model": "模型",
        "equation": "公式",
        "array": "阵列",
        "array_rows": "阵列行数",
        "array_cols": "阵列列数",
        "bit_width": "位宽",
        "group": "实验组",
        "samples": "样本数",
        "r2": "R²",
        "polynomial_r2": "多项式 R²",
        "rmse": "RMSE",
        "nonlinearity_percent_fs": "满量程非线性度（% FS）",
        "nl_percent_fs": "满量程非线性度（% FS）",
        "ideal_mac": "理想 MAC",
        "measured_output": "测量输出",
    }.get(name, name)


def _setup_device_combo(combo: QComboBox) -> None:
    combo.addItem("自动选择", "auto")
    combo.addItem("cuda", "cuda")
    combo.addItem("CPU", "cpu")


def _device_value(combo: QComboBox) -> str:
    return str(combo.currentData() or combo.currentText()).lower()


def _select_device(combo: QComboBox, value: str) -> None:
    index = combo.findData(str(value).lower())
    if index >= 0:
        combo.setCurrentIndex(index)


def _translate_runtime_message(message: str) -> str:
    """Translate service progress text while preserving model names and metrics."""
    exact = {
        "Training and evaluation completed": "训练与评估完成",
        "GPT-2 Small training and evaluation completed": "GPT-2 Small 训练与评估完成",
        "GPT-2 generation completed": "GPT-2 生成完成",
        "Calibrating GPT-2 Large input scales from the prompt": "正在根据提示词校准 GPT-2 Large 输入尺度",
    }
    if message in exact:
        return exact[message]
    translated = message
    for source in (
        "baseline ideal differential", "baseline device differential", "baseline float",
        "trained ideal differential", "trained device differential", "trained float",
        "ideal_differential", "device_differential", "float",
    ):
        translated = translated.replace(source, _mode_display_name(source))
    for source, target in (
        ("Evaluating trained ", "正在评估训练后的 "),
        ("Evaluating ", "正在评估 "),
        ("Preparing ", "正在准备 "),
        (" complete", " 完成"),
        (" batches", " 个 Batch"),
        (" tokens", " 个 Token"),
        ("Epoch ", "Epoch "),
    ):
        translated = translated.replace(source, target)
    return translated


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
                 auxiliary: Path | None, error_model: Path, device: str,
                 adc_bits: int | None = None, noise_sigma: float = 0.0,
                 noise_rho: float = 0.0,
                 input_image: str | Path | np.ndarray | None = None):
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
        self.adc_bits = adc_bits; self.noise_sigma = noise_sigma; self.noise_rho = noise_rho
        self.input_image = input_image

    def run(self) -> None:
        try:
            self.succeeded.emit(
                self.service.infer_registered_single(
                    self.project_root, self.model_id, self.sample_index,
                    dataset_root=self.dataset_root, checkpoint_path=self.checkpoint,
                    auxiliary_path=self.auxiliary, error_model_path=self.error_model,
                    device=self.device, adc_bits=self.adc_bits,
                    noise_sigma=self.noise_sigma, noise_rho=self.noise_rho,
                    input_image=self.input_image,
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


class WikiText2EvaluationThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    progressed = Signal(int, str)

    def __init__(self, service: WikiText2EvaluationService,
                 request: WikiText2EvaluationRequest):
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
        except WikiText2EvaluationCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())


def _retire_widget_worker(widget: QWidget) -> None:
    """Release a completed QThread instead of retaining it for the GUI lifetime."""

    worker = getattr(widget, "worker", None)
    if isinstance(worker, QThread) and not worker.isRunning():
        widget.worker = None
        worker.deleteLater()


class CellFittingWidget(QWidget):
    """Import transistor measurements, fit them, and create a reusable error model."""

    fit_mode_requested = Signal(int)

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
        self.setProperty("usesIndependentColumnScroll", True)
        left_card = QFrame(); left_card.setProperty("workCard", True)
        left = QVBoxLayout(left_card); left.setContentsMargins(18, 16, 18, 16); left.setSpacing(10)
        left_title = QLabel("误差数据与拟合配置"); left_title.setProperty("sectionTitle", True); left.addWidget(left_title)
        left_hint = QLabel("导入器件或阵列测量 CSV，完成列映射后生成多种可复用误差模型。")
        left_hint.setProperty("sectionHint", True); left_hint.setWordWrap(True); left.addWidget(left_hint)
        mode_row = QHBoxLayout(); mode_row.addWidget(QLabel("拟合对象"))
        self.device_mode_button = QPushButton("器件误差拟合"); self.device_mode_button.setCheckable(True)
        self.array_mode_button = QPushButton("阵列误差拟合"); self.array_mode_button.setCheckable(True)
        mode_group = QButtonGroup(self); mode_group.setExclusive(True)
        for button in (self.device_mode_button, self.array_mode_button):
            button.setProperty("fitMode", True); mode_group.addButton(button); mode_row.addWidget(button)
        self.device_mode_button.setChecked(True)
        self.device_mode_button.clicked.connect(lambda: self.fit_mode_requested.emit(0))
        self.array_mode_button.clicked.connect(lambda: self.fit_mode_requested.emit(1))
        mode_row.addStretch(); left.addLayout(mode_row)
        form = QFormLayout(); form.setSpacing(9)
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
        form.addRow("数字码格式", self.code_format_combo); left.addLayout(form)

        mapping_box = QGroupBox("CSV 列映射（从表头选择，或使用自动识别）")
        mapping_form = QFormLayout(mapping_box); self.mapping_edits: dict[str, QComboBox] = {}
        for key, label in self.MAPPING_FIELDS:
            combo = QComboBox(); combo.setEditable(True)
            combo.addItem("自动识别／未指定", "")
            combo.lineEdit().setPlaceholderText(key)
            self.mapping_edits[key] = combo
            mapping_form.addRow(label, combo)
        left.addWidget(mapping_box)
        units = QHBoxLayout()
        self.vd_unit_edit = QLineEdit("V"); self.vg_unit_edit = QLineEdit("V"); self.output_unit_edit = QLineEdit("A")
        units.addWidget(QLabel("VD 单位")); units.addWidget(self.vd_unit_edit)
        units.addWidget(QLabel("VG 单位")); units.addWidget(self.vg_unit_edit)
        units.addWidget(QLabel("输出单位")); units.addWidget(self.output_unit_edit); left.addLayout(units)
        self.output_edit = QLineEdit(); output_row = QHBoxLayout(); output_row.addWidget(self.output_edit)
        output_button = QPushButton("选择目录"); output_button.clicked.connect(self._browse_output); output_row.addWidget(output_button)
        output_form = QFormLayout(); output_form.addRow("输出目录", output_row); left.addLayout(output_form)
        controls = QHBoxLayout(); self.preview_button = QPushButton("预览并校验")
        self.preview_button.clicked.connect(self.preview); self.fit_button = QPushButton("开始拟合并生成误差模型")
        self.fit_button.clicked.connect(self.start_fit); controls.addWidget(self.preview_button); controls.addWidget(self.fit_button)
        controls.addStretch(); left.addLayout(controls)
        self.progress = QProgressBar(); left.addWidget(self.progress)
        self.summary = QLabel("就绪"); self.summary.setWordWrap(True); left.addWidget(self.summary)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(135)
        left.addWidget(QLabel("运行日志")); left.addWidget(self.log)

        right_card = QFrame(); right_card.setProperty("workCard", True)
        right = QVBoxLayout(right_card); right.setContentsMargins(18, 16, 18, 16); right.setSpacing(10)
        right_title = QLabel("数据预览与拟合结果"); right_title.setProperty("sectionTitle", True); right.addWidget(right_title)
        metrics = QHBoxLayout(); metrics.setSpacing(10)
        r2_card, self.r2_metric = _metric_card("决定系数 R²")
        nl_card, self.nl_metric = _metric_card("非线性误差")
        model_card, self.model_count_metric = _metric_card("可选误差模型")
        metrics.addWidget(r2_card); metrics.addWidget(nl_card); metrics.addWidget(model_card)
        right.addLayout(metrics)
        self.table = QTableWidget(); self.table.setAlternatingRowColors(True)
        self.table.setProperty("compactColumns", True)
        self.table.horizontalHeader().setStretchLastSection(False); right.addWidget(self.table)
        self.plot_tabs = QTabWidget(); self.plot_tabs.setMinimumHeight(390); self.plot_tabs.setMaximumHeight(560); self.plot_tabs.setVisible(False)
        right.addWidget(self.plot_tabs, 1)
        self.left_scroll, self.right_scroll = _add_independent_columns(layout, left_card, right_card)

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
        self.table.resizeColumnsToContents(); _fit_result_table_height(self.table, max_visible_rows=5)

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
                if self.output_edit.text().strip() else output_root() / "gui_runs" /
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
        self.r2_metric.setText(f"{outcome.metrics['r2']:.5f}")
        self.nl_metric.setText(f"{outcome.metrics['nl_percent_fs']:.3f}% FS")
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
            self.model_count_metric.setText(str(len(selectable)))
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
                label = ResponsivePixmapLabel(); label.setMinimumHeight(355)
                label.set_image(path)
                self.plot_tabs.addTab(label, plot_names.get(path.stem, path.stem.replace("_", " ")))
        self.plot_tabs.setVisible(self.plot_tabs.count() > 0)

    def _fit_failed(self, details: str) -> None:
        self.log.appendPlainText(details); self.summary.setText("拟合失败：" + details.splitlines()[-1])

    def _fit_finished(self) -> None:
        self.fit_button.setEnabled(True); self.preview_button.setEnabled(True); self.progress.setValue(100 if self.last_outcome else 0)
        _retire_widget_worker(self)

    def _show_error(self, title: str, error: Exception) -> None:
        message = f"{title}: {error}"; self.summary.setText(message); self.log.appendPlainText(message)

    def _restore_settings(self) -> None:
        saved_source = str(self.settings.value("cell/source", "")).strip()
        bundled_source = self.platform_root.parent / "0_data" / "03_UOS_1.csv"
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

    fit_mode_requested = Signal(int)

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
        mode_row = QHBoxLayout(); mode_row.addWidget(QLabel("拟合对象"))
        self.device_mode_button = QPushButton("器件误差拟合"); self.device_mode_button.setCheckable(True)
        self.array_mode_button = QPushButton("阵列误差拟合"); self.array_mode_button.setCheckable(True)
        mode_group = QButtonGroup(self); mode_group.setExclusive(True)
        for button in (self.device_mode_button, self.array_mode_button):
            button.setProperty("fitMode", True); mode_group.addButton(button); mode_row.addWidget(button)
        self.array_mode_button.setChecked(True)
        self.device_mode_button.clicked.connect(lambda: self.fit_mode_requested.emit(0))
        self.array_mode_button.clicked.connect(lambda: self.fit_mode_requested.emit(1))
        mode_row.addStretch(); root.addLayout(mode_row)
        source_group = QGroupBox("阵列 MAC 数据")
        source_layout = QGridLayout(source_group)
        self.path_edit = QLineEdit()
        self.path_edit.textChanged.connect(self._source_changed)
        browse = QPushButton("选择 CSV")
        browse.clicked.connect(self._browse_source)
        source_layout.addWidget(QLabel("文件"), 0, 0)
        source_layout.addWidget(self.path_edit, 0, 1)
        source_layout.addWidget(browse, 0, 2)
        self.encoding = QComboBox()
        for label, value in (("自动识别", None), ("UTF-8", "utf-8-sig"), ("GBK", "gb18030")):
            self.encoding.addItem(label, value)
        self.delimiter = QComboBox()
        for label, value in (("自动识别", None), ("逗号", ","), ("分号", ";")):
            self.delimiter.addItem(label, value)
        self.missing = QComboBox()
        for label, value in (("遇到缺失值时报错", "error"), ("删除缺失行", "drop"), ("插值填充", "interpolate")):
            self.missing.addItem(label, value)
        source_layout.addWidget(QLabel("文件编码"), 1, 0)
        source_layout.addWidget(self.encoding, 1, 1)
        source_layout.addWidget(QLabel("分隔符"), 2, 0)
        source_layout.addWidget(self.delimiter, 2, 1)
        source_layout.addWidget(QLabel("缺失值处理"), 3, 0)
        source_layout.addWidget(self.missing, 3, 1)
        root.addWidget(source_group)

        config_group = QGroupBox("阵列配置筛选")
        config = QHBoxLayout(config_group)
        self.array_rows = QSpinBox(); self.array_rows.setObjectName("macArrayRows")
        self.array_rows.setRange(1, 8192); self.array_rows.setValue(16)
        self.array_cols = QSpinBox(); self.array_cols.setObjectName("macArrayCols")
        self.array_cols.setRange(1, 8192); self.array_cols.setValue(16)
        self.bit_width = QComboBox(); self.bit_width.setObjectName("macBitWidth")
        for bits in (1, 2, 4, 8): self.bit_width.addItem(f"{bits}-bit", bits)
        self.bit_width.setCurrentIndex(1)
        for label, widget in (("行数", self.array_rows), ("列数", self.array_cols), ("位宽", self.bit_width)):
            config.addWidget(QLabel(label)); config.addWidget(widget)
        config.addStretch(); root.addWidget(config_group)

        mapping_group = QGroupBox("列映射（留空时自动识别）")
        mapping_layout = QFormLayout(mapping_group)
        self.mapping_edits: dict[str, QLineEdit] = {}
        for logical, label in LOGICAL_COLUMNS:
            edit = QLineEdit(); edit.setPlaceholderText(logical)
            self.mapping_edits[logical] = edit; mapping_layout.addRow(label, edit)
        root.addWidget(mapping_group)

        controls = QHBoxLayout()
        self.preview_button = QPushButton("预览并校验"); self.preview_button.clicked.connect(self.preview)
        self.degree = QSpinBox(); self.degree.setRange(1, 3); self.degree.setValue(2)
        self.fit_button = QPushButton("拟合选中配置"); self.fit_button.clicked.connect(self.start_fit)
        self.export_button = QPushButton("导出结果"); self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        self.clear_log_button = QPushButton("清空日志")
        self.export_log_button = QPushButton("导出日志")
        for widget in (self.preview_button, QLabel("多项式阶数"), self.degree, self.fit_button,
                       self.export_button, self.clear_log_button, self.export_log_button): controls.addWidget(widget)
        controls.addStretch(); root.addLayout(controls)

        self.metrics_summary = QLabel("R²：--    满量程非线性度：--    RMSE：--")
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
            raise ValueError("阵列行数和列数必须为正数，位宽必须为 1、2、4 或 8 bit")
        return {"array_rows": rows, "array_cols": cols, "bit_width": bits}

    def _source(self) -> Path:
        text = self.path_edit.text().strip()
        if not text: raise ValueError("请选择阵列 MAC CSV 文件")
        return Path(text).expanduser().resolve()

    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择阵列 MAC 数据", self.path_edit.text(), "CSV 文件 (*.csv)")
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
        self.table.setHorizontalHeaderLabels([_column_display_name(column) for column in columns])
        for row, record in enumerate(records):
            for column, name in enumerate(columns): self.table.setItem(row, column, QTableWidgetItem(str(record[name])))
        self.table.resizeColumnsToContents()

    def preview(self) -> None:
        try:
            self._persist_settings(); options = self._options(); selection = self._selection()
            frame, metadata = load_mac_csv(self._source(), column_mapping=options.column_mapping or None,
                encoding=options.encoding, delimiter=options.delimiter, missing=options.missing, **selection)
            self._show_records(frame.head(20).to_dict(orient="records"))
            self.log.appendPlainText(f"预览成功：{len(frame)} 行；选中配置={metadata['selection']}")
            self.progress.setValue(20)
        except Exception as exc: self._show_error("数据预览失败", str(exc))

    def start_fit(self, output_dir: Path | None = None) -> None:
        try:
            self._persist_settings(); source = self._source(); selection = self._selection()
            output = Path(output_dir) if output_dir else output_root() / "gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "mac_model"
            self.fit_button.setEnabled(False); self.preview_button.setEnabled(False); self.progress.setRange(0, 0)
            self.log.appendPlainText(f"正在拟合 {source}；选中配置={selection}")
            self.worker = MacFitThread(self.service, source, output, self._options(), **selection)
            self.worker.succeeded.connect(self._fit_succeeded); self.worker.failed.connect(self._fit_failed)
            self.worker.finished.connect(self._fit_finished); self.worker.start()
        except Exception as exc: self._show_error("无法开始拟合", str(exc))

    def _fit_succeeded(self, outcome: MacFitOutcome) -> None:
        self.last_results = list(outcome.summary); self._show_records(self.last_results)
        self.export_button.setEnabled(True)
        if self.last_results:
            result = self.last_results[0]
            self.metrics_summary.setText(f"R²：{result['r2']:.4f}    满量程非线性度：{result['nonlinearity_percent_fs']:.6f}%    RMSE：{result['rmse']:.6g}")
        self.log.appendPlainText(f"拟合完成：{outcome.model_path}")

    def _fit_failed(self, details: str) -> None:
        self.log.appendPlainText(details); self._show_error("拟合失败", details.splitlines()[-1])

    def _fit_finished(self) -> None:
        self.progress.setRange(0, 100); self.progress.setValue(100 if self.last_results else 0)
        self.fit_button.setEnabled(True); self.preview_button.setEnabled(True)
        _retire_widget_worker(self)

    def export_results(self) -> None:
        if not self.last_results: return
        path, selected = QFileDialog.getSaveFileName(self, "导出线性度结果", "mac-linearity-results.xlsx", "Excel (*.xlsx);;CSV (*.csv)")
        if not path: return
        if not Path(path).suffix: path += ".xlsx" if "Excel" in selected else ".csv"
        try: self.log.appendPlainText(f"结果已导出：{self.service.export_table(self.last_results, path)}")
        except Exception as exc: self._show_error("导出失败", str(exc))

    def export_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出运行日志", "dram-cim-log.txt", "文本文件 (*.txt)")
        if path:
            try:
                target = Path(path).expanduser().resolve(); target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(self.log.toPlainText(), encoding="utf-8")
            except Exception as exc: self._show_error("日志导出失败", str(exc))

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

class MnistDrawingCanvas(QWidget):
    """Black canvas with a white round pen, suitable for MNIST input."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(252, 252)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._image = QImage(252, 252, QImage.Format.Format_Grayscale8)
        self._image.fill(0)
        self._last_position = None

    def clear(self) -> None:
        self._image.fill(0)
        self._last_position = None
        self.update()
        self.changed.emit()

    def image_array(self) -> np.ndarray:
        image = self._image.convertToFormat(QImage.Format.Format_Grayscale8)
        view = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes())
        return view.reshape(image.height(), image.bytesPerLine())[:, :image.width()].copy()

    def has_ink(self) -> bool:
        return bool(np.any(self.image_array()))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._image)
        painter.setPen(QPen(QColor("#60738D"), 2))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_position = event.position()
            self._draw_to(event.position())

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._draw_to(event.position())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._draw_to(event.position())
            self._last_position = None

    def _draw_to(self, position) -> None:
        painter = QPainter(self._image)
        pen = QPen(QColor("white"), 18, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        start = self._last_position if self._last_position is not None else position
        painter.drawLine(start, position)
        painter.end()
        self._last_position = position
        self.update()
        self.changed.emit()


class CurrentPageStackedWidget(QStackedWidget):
    """Size a stack from its visible page instead of its largest page."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.currentChanged.connect(self._refresh_geometry)

    def sizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is None:
            return super().sizeHint()
        return QSize(super().sizeHint().width(), current.sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is None:
            return super().minimumSizeHint()
        return QSize(0, current.minimumSizeHint().height())

    def _refresh_geometry(self, _index: int) -> None:
        self.fitCurrentPage()

    def fitCurrentPage(self) -> None:
        """Hard-limit the stack to the active page's layout height."""

        current = self.currentWidget()
        if current is None:
            return
        height = max(1, current.sizeHint().height())
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()
        parent_layout = self.parentWidget().layout() if self.parentWidget() else None
        if parent_layout is not None:
            parent_layout.invalidate()


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
        self._active_input_description = "数据集样本"
        layout = QVBoxLayout(self)
        self.setProperty("usesIndependentColumnScroll", True)

        mode_bar = QFrame(); mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(0, 0, 0, 6); mode_layout.setSpacing(12)
        mode_title = QLabel("输入方式"); mode_title.setProperty("sectionTitle", True)
        mode_layout.addWidget(mode_title)
        self.input_mode_group = QButtonGroup(self); self.input_mode_group.setExclusive(True)
        self.dataset_mode_button = QPushButton("数据集样本")
        self.upload_mode_button = QPushButton("上传数据")
        self.handwriting_mode_button = QPushButton("MNIST 手写板")
        for mode_id, button in enumerate((self.dataset_mode_button, self.upload_mode_button,
                                          self.handwriting_mode_button)):
            button.setCheckable(True)
            button.setMinimumHeight(46)
            button.setStyleSheet(
                "QPushButton { padding: 8px 22px; border: 1px solid #CBD8E8; "
                "border-radius: 20px; background: #F6F9FC; color: #405A78; }"
                "QPushButton:checked { border: 2px solid #1674C8; background: #E4F1FC; "
                "color: #0969B7; font-weight: 700; }"
                "QPushButton:disabled { color: #9AA9BA; background: #EEF2F6; }"
            )
            self.input_mode_group.addButton(button, mode_id)
            mode_layout.addWidget(button)
        self.dataset_mode_button.setChecked(True)
        mode_layout.addStretch()
        layout.addWidget(mode_bar)
        left_card = QFrame(); left_card.setProperty("workCard", True)
        left = QVBoxLayout(left_card); left.setContentsMargins(18, 16, 18, 16); left.setSpacing(10)
        left_title = QLabel("推理配置"); left_title.setProperty("sectionTitle", True); left.addWidget(left_title)
        description = QLabel("选择网络、数据集与器件误差模型，比较三种计算模式。")
        description.setProperty("sectionHint", True); description.setWordWrap(True); left.addWidget(description)

        form = QFormLayout(); form.setSpacing(9)
        advanced = CollapsiblePanel(); advanced_form = QFormLayout(); advanced_form.setSpacing(8)
        self.model_combo = QComboBox()
        self.model_registry = _populate_image_model_combo(
            self.model_combo, self.project_root, "single_inference"
        )
        form.addRow("网络模型", self.model_combo)
        self.dataset_name = QLineEdit(); self.dataset_name.setReadOnly(True)
        form.addRow("数据集", self.dataset_name)
        self.dataset_edit = QLineEdit(); dataset_row = QHBoxLayout(); dataset_row.addWidget(self.dataset_edit)
        dataset_button = QPushButton("选择目录"); dataset_button.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(dataset_button); advanced_form.addRow("数据集根目录", dataset_row)
        self.checkpoint_edit = QLineEdit(); checkpoint_row = QHBoxLayout(); checkpoint_row.addWidget(self.checkpoint_edit)
        checkpoint_button = QPushButton("选择文件"); checkpoint_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_button); advanced_form.addRow("Checkpoint", checkpoint_row)
        self.auxiliary_label = QLabel("量化参数")
        self.auxiliary_edit = QLineEdit(); auxiliary_row = QHBoxLayout(); auxiliary_row.addWidget(self.auxiliary_edit)
        auxiliary_button = QPushButton("选择文件"); auxiliary_button.clicked.connect(self._browse_auxiliary)
        auxiliary_row.addWidget(auxiliary_button); advanced_form.addRow(self.auxiliary_label, auxiliary_row)
        self.error_model_combo = QComboBox()
        form.addRow("误差建模方法", self.error_model_combo)
        self.noise_combo = QComboBox(); _setup_noise_combo(self.noise_combo)
        form.addRow("附加随机噪声", self.noise_combo)
        self.adc_combo = QComboBox(); _setup_adc_combo(self.adc_combo)
        form.addRow("ADC 量化", self.adc_combo)
        self.error_model_edit = QLineEdit(); self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout(); error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("选择文件"); error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button); advanced_form.addRow("器件误差模型文件", error_row)
        self.device_combo = QComboBox(); _setup_device_combo(self.device_combo)
        advanced_form.addRow("运行设备", self.device_combo)
        left.addLayout(form)

        self.input_stack = CurrentPageStackedWidget()
        dataset_panel = QWidget(); dataset_form = QFormLayout(dataset_panel)
        dataset_form.setContentsMargins(0, 4, 0, 4)
        self.sample_index = QSpinBox(); self.sample_index.setRange(0, 9999)
        dataset_form.addRow("测试样本索引", self.sample_index)
        self.input_stack.addWidget(dataset_panel)

        upload_panel = QWidget(); upload_form = QFormLayout(upload_panel)
        upload_form.setContentsMargins(0, 4, 0, 4)
        self.upload_edit = QLineEdit(); self.upload_edit.setReadOnly(True)
        upload_row = QHBoxLayout(); upload_row.addWidget(self.upload_edit)
        self.upload_button = QPushButton("选择图片"); self.upload_button.clicked.connect(self._browse_input_image)
        upload_row.addWidget(self.upload_button)
        upload_form.addRow("待推理图片", upload_row)
        upload_hint = QLabel("支持 PNG、JPG、JPEG、BMP；将按当前模型自动调整尺寸、通道并归一化。")
        upload_hint.setWordWrap(True); upload_hint.setProperty("sectionHint", True)
        upload_form.addRow("", upload_hint)
        self.input_stack.addWidget(upload_panel)

        handwriting_panel = QWidget(); handwriting_layout = QVBoxLayout(handwriting_panel)
        handwriting_layout.setContentsMargins(0, 4, 0, 4)
        handwriting_hint = QLabel("在黑色区域内用鼠标书写白色数字。仅支持 MNIST 模型。")
        handwriting_hint.setWordWrap(True); handwriting_layout.addWidget(handwriting_hint)
        self.handwriting_canvas = MnistDrawingCanvas(); handwriting_layout.addWidget(
            self.handwriting_canvas, 0, Qt.AlignmentFlag.AlignHCenter)
        clear_row = QHBoxLayout(); clear_row.addStretch()
        clear_button = QPushButton("清空手写板"); clear_button.clicked.connect(self.handwriting_canvas.clear)
        clear_row.addWidget(clear_button); clear_row.addStretch(); handwriting_layout.addLayout(clear_row)
        self.input_stack.addWidget(handwriting_panel)
        self.input_stack.fitCurrentPage()
        left.addWidget(self.input_stack)
        advanced.content_layout.addLayout(advanced_form); left.addWidget(advanced)

        self.resource_status = QLabel()
        self.resource_status.setWordWrap(True)
        left.addWidget(self.resource_status)
        for edit in (self.dataset_edit, self.checkpoint_edit, self.auxiliary_edit, self.error_model_edit):
            edit.textChanged.connect(self._update_resource_status)

        controls = QHBoxLayout(); self.run_button = QPushButton("运行单样本推理")
        self.run_button.clicked.connect(self.start_inference)
        controls.addWidget(self.run_button); controls.addStretch(); left.addLayout(controls)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); left.addWidget(self.progress)
        self.summary = QLabel("尚未运行"); self.summary.setWordWrap(True); left.addWidget(self.summary)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100)
        left.addWidget(QLabel("运行日志")); left.addWidget(self.log)

        right_card = QFrame(); right_card.setProperty("workCard", True)
        right = QVBoxLayout(right_card); right.setContentsMargins(18, 16, 18, 16); right.setSpacing(10)
        right_title = QLabel("输入与推理结果"); right_title.setProperty("sectionTitle", True); right.addWidget(right_title)
        result_area = QHBoxLayout(); result_area.setSpacing(12)
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
        _fit_result_table_height(self.result_table, max_visible_rows=3)
        tables.addWidget(self.result_table)
        self.top5_table = QTableWidget(0, 4)
        self.top5_table.setHorizontalHeaderLabels(["模式", "排名", "类别", "置信度"])
        _fit_result_table_height(self.top5_table, max_visible_rows=8)
        tables.addWidget(self.top5_table)
        result_area.addLayout(tables, 3)
        right.addLayout(result_area)
        self.left_scroll, self.right_scroll = _add_independent_columns(layout, left_card, right_card)

        requested_model = str(self.settings.value("inference/model", "resnet8-mnist"))
        saved_model = requested_model
        if (saved_model not in FRONTEND_MODEL_ID_SET
                or not self._model_resources_ready(saved_model)):
            candidates = ["resnet8-mnist"] + [
                str(self.model_combo.itemData(i)) for i in range(self.model_combo.count())
            ]
            saved_model = next(
                (model_id for model_id in candidates if self._model_resources_ready(model_id)),
                "resnet8-mnist",
            )
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(saved_model)))
        self._load_model_settings(saved_model)
        self._active_model_id = saved_model
        _select_device(self.device_combo, str(self.settings.value("inference/device", "auto")))
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.input_mode_group.idClicked.connect(self._input_mode_changed)
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
        self._update_input_mode_availability()
        if requested_model != saved_model:
            self.resource_status.setText(
                f"保存的模型 {requested_model} 缺少资源，已自动切换到 {saved_model}。\n"
                + self.resource_status.text()
            )

    def _model_defaults(self, model_id: str) -> dict[str, str]:
        error_model = output_root() / "error_models/01_NMOS/error_model.json"
        spec = self.model_registry.get(model_id)
        dataset_id = spec.datasets[0]
        auxiliary = str(spec.auxiliary_resources[0]) if spec.auxiliary_resources else ""
        return {
            "dataset": "MNIST" if dataset_id == "mnist" else "CIFAR-10",
            "root": str(self._dataset_root(dataset_id)),
            "checkpoint": str(spec.checkpoint or ""),
            "auxiliary": auxiliary,
            "auxiliary_label": (
                "量化/输入尺度参数" if auxiliary else "输入尺度（运行时自动校准）"
            ),
            "error_model": str(error_model),
        }

    def _dataset_root(self, dataset: str) -> Path:
        configured = data_root(dataset)
        candidates = [configured, self.project_root / "data" / dataset]
        marker = Path("MNIST/raw/t10k-images-idx3-ubyte") if dataset == "mnist" else Path("cifar-10-batches-py/test_batch")
        return next((path for path in candidates if (path / marker).is_file()), configured)

    def _model_configuration(self, model_id: str) -> dict[str, str]:
        values = self._model_defaults(model_id)
        for name in ("dataset_root", "checkpoint", "auxiliary", "error_model"):
            setting_key = f"inference/{model_id}/{name}"
            if name == "auxiliary" and model_id in RUNTIME_CALIBRATED_MODEL_IDS:
                # These models calibrate input scales from the training split.
                # Ignore stale scale paths saved by an older portable install.
                self.settings.remove(setting_key)
                continue
            saved = str(self.settings.value(setting_key, ""))
            if not saved:
                continue
            value_key = "root" if name == "dataset_root" else name
            saved_path = Path(saved).expanduser()
            default_path = Path(values[value_key]).expanduser() if values[value_key] else None
            if name == "dataset_root":
                marker = (
                    Path("MNIST/raw/t10k-images-idx3-ubyte")
                    if model_id.endswith("mnist")
                    else Path("cifar-10-batches-py/test_batch")
                )
                saved_available = (saved_path / marker).is_file()
                default_available = bool(default_path and (default_path / marker).is_file())
            else:
                saved_available = saved_path.is_file()
                default_available = bool(default_path and default_path.is_file())
            if saved_available or not default_available:
                values[value_key] = saved
            else:
                # Absolute paths stored by an older portable/install location
                # must not override resources bundled beside the current EXE.
                self.settings.remove(setting_key)
        return values

    def _configuration_problems(self, model_id: str, values: dict[str, str] | None = None) -> list[str]:
        values = values or self._model_configuration(model_id)
        dataset = Path(values["root"]).expanduser()
        if model_id.endswith("mnist"):
            from .inference_runtime import mnist_required_files
            dataset_files = list(mnist_required_files(
                dataset, include_training=model_id in RUNTIME_CALIBRATED_MODEL_IDS,
            ))
        else:
            from .inference_runtime import cifar10_required_files
            dataset_files = list(cifar10_required_files(dataset))
        checks = [(path, "数据集") for path in dataset_files]
        checks += [(Path(values["checkpoint"]), "checkpoint"),
                  (Path(values["error_model"]), "误差模型")]
        if self.model_registry.get(model_id).auxiliary_resources:
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
        self.auxiliary_edit.setEnabled(bool(values["auxiliary"]))
        if self.require_workflow:
            self._sync_workflow_error_path()

    def _model_changed(self, _index: int) -> None:
        self._save_model_settings(self._active_model_id)
        model_id = str(self.model_combo.currentData())
        self._load_model_settings(model_id)
        self._active_model_id = model_id
        self._update_input_mode_availability()
        self._update_resource_status()

    def _update_input_mode_availability(self) -> None:
        is_mnist = str(self.model_combo.currentData()).endswith("mnist")
        self.handwriting_mode_button.setEnabled(is_mnist)
        self.handwriting_mode_button.setToolTip(
            "" if is_mnist else "MNIST 手写板仅适用于使用 MNIST 数据集的模型"
        )
        if not is_mnist and self.handwriting_mode_button.isChecked():
            self.dataset_mode_button.setChecked(True)
            self._input_mode_changed(0)

    def _input_mode_changed(self, mode_id: int) -> None:
        self.input_stack.setCurrentIndex(mode_id)
        labels = ("数据集样本", "上传数据", "MNIST 手写板")
        self.summary.setText(f"当前输入方式：{labels[mode_id]}，尚未运行")
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
        path, _ = QFileDialog.getOpenFileName(self, "选择 Checkpoint", self.checkpoint_edit.text(), "PyTorch (*.pt *.pth);;所有文件 (*)")
        if path: self.checkpoint_edit.setText(path)

    def _browse_auxiliary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择辅助参数", self.auxiliary_edit.text(), "JSON (*.json);;所有文件 (*)")
        if path: self.auxiliary_edit.setText(path)

    def _browse_error_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择器件误差模型", self.error_model_edit.text(), "JSON (*.json)")
        if path: self.error_model_edit.setText(path)

    def _browse_input_image(self) -> None:
        start = self.upload_edit.text().strip() or str(self.project_root)
        path, _ = QFileDialog.getOpenFileName(
            self, "选择待推理图片", start,
            "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
        )
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "图片不可用", "无法读取所选图片，请选择有效的 PNG、JPG 或 BMP 文件。")
            return
        self.upload_edit.setText(path)
        self.input_image.setPixmap(pixmap.scaled(
            220, 220, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self.final_prediction.setText("自定义输入，等待推理")

    def _selected_input(self) -> tuple[str | Path | np.ndarray | None, str]:
        mode_id = self.input_mode_group.checkedId()
        if mode_id == 0:
            return None, f"数据集样本 {self.sample_index.value()}"
        if mode_id == 1:
            path = Path(self.upload_edit.text().strip()).expanduser()
            if not path.is_file():
                raise FileNotFoundError("请先选择一张有效的待推理图片")
            return path.resolve(), f"上传图片 {path.name}"
        if not str(self.model_combo.currentData()).endswith("mnist"):
            raise ValueError("MNIST 手写板仅适用于 MNIST 模型")
        if not self.handwriting_canvas.has_ink():
            raise ValueError("请先在手写板中书写一个数字")
        return self.handwriting_canvas.image_array(), "MNIST 手写板"

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
                Path(values["error_model"]).expanduser().resolve(), _device_value(self.device_combo))

    def start_inference(self) -> None:
        if self.worker is not None and self.worker.isRunning(): return
        try:
            if self.coordinator is not None and self.error_model_combo.currentData():
                self.coordinator.select_error_model(str(self.error_model_combo.currentData()))
            model_id, dataset, checkpoint, auxiliary, error_model, device = self._validate_resources()
            input_image, input_description = self._selected_input()
        except Exception as exc:
            message = str(exc); self.summary.setText("推理配置错误：" + message.splitlines()[0])
            self.log.appendPlainText(message); return
        self.last_outcome = None; self.run_button.setEnabled(False); self.progress.setRange(0, 0)
        self._active_input_description = input_description
        self.log.appendPlainText(f"开始推理：{model_id}，输入：{input_description}")
        self._save_model_settings(model_id)
        self.settings.setValue("inference/model", model_id); self.settings.setValue("inference/device", device)
        sigma, rho = self.noise_combo.currentData()
        self.worker = InferenceThread(
            self.service, self.project_root, model_id, self.sample_index.value(),
            dataset, checkpoint, auxiliary, error_model, device,
            self.adc_combo.currentData(), float(sigma), float(rho),
            input_image,
        )
        self.worker.succeeded.connect(self._inference_succeeded); self.worker.failed.connect(self._inference_failed)
        self.worker.finished.connect(self._inference_finished); self.worker.start()

    @staticmethod
    def _display_name(model_id: str, label: int) -> str:
        if model_id.endswith("cifar10"):
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
        truth = (
            self._display_name(model_id, outcome.true_label)
            if outcome.true_label >= 0 else "未知（自定义输入）"
        )
        self.summary.setText(
            f"输入：{self._active_input_description}　真实标签：{truth}　"
            f"设备：{outcome.device}　准备：{outcome.preparation_ms:.1f} ms　总耗时：{outcome.total_elapsed_ms:.1f} ms"
        )
        self._show_input_image(outcome)
        final = outcome.predictions[-1]
        self.final_prediction.setText(
            f"真实标签：{truth}    "
            f"器件误差预测：{self._display_name(model_id, final.predicted_label)} "
            f"({final.confidence:.2%})"
        )
        self.result_table.setRowCount(len(outcome.predictions))
        top_rows = sum(len(prediction.top5) for prediction in outcome.predictions)
        self.top5_table.setRowCount(top_rows)
        top_row = 0
        for row, prediction in enumerate(outcome.predictions):
            values = (_mode_display_name(prediction.mode), self._display_name(model_id, prediction.predicted_label),
                      f"{prediction.confidence:.2%}", f"{prediction.elapsed_ms:.3f}")
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(value))
            for rank, (label, confidence) in enumerate(prediction.top5, 1):
                for column, value in enumerate((_mode_display_name(prediction.mode), str(rank), self._display_name(model_id, label), f"{confidence:.2%}")):
                    self.top5_table.setItem(top_row, column, QTableWidgetItem(value))
                top_row += 1
        self.result_table.resizeColumnsToContents(); self.top5_table.resizeColumnsToContents()
        _fit_result_table_height(self.result_table, max_visible_rows=3); _fit_result_table_height(self.top5_table, max_visible_rows=8)
        self.log.appendPlainText("单样本推理完成")
        target = output_root() / "gui_inference"; target.mkdir(parents=True, exist_ok=True)
        suffix = (
            f"sample_{outcome.sample_index}" if outcome.true_label >= 0
            else "custom_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        result_path = target / f"{model_id}_{suffix}.json"
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
        _retire_widget_worker(self)


class BatchInferenceWidget(QWidget):
    """Grouped inference for every registered network with a fitted error model."""

    MODE_NAMES = {
        "float": "浮点模型",
        "ideal_differential": "理想差分",
        "device_differential": "器件误差差分",
    }
    CHART_MODE_NAMES = {
        "float": "浮点模型",
        "ideal_differential": "理想差分",
        "device_differential": "器件误差差分",
    }

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.coordinator = coordinator
        self.require_workflow = require_workflow
        self.worker: BatchInferenceThread | None = None
        self.last_outcome: GroupedBatchInferenceOutcome | None = None
        platform = output_root()
        layout = QVBoxLayout(self)
        self.setProperty("usesIndependentColumnScroll", True)
        left_card = QFrame(); left_card.setProperty("workCard", True)
        left = QVBoxLayout(left_card); left.setContentsMargins(18, 16, 18, 16); left.setSpacing(10)
        left_title = QLabel("批量推理配置"); left_title.setProperty("sectionTitle", True); left.addWidget(left_title)
        description = QLabel("选择误差模型与网络，批量比较浮点、理想差分和器件误差差分。")
        description.setProperty("sectionHint", True); description.setWordWrap(True); left.addWidget(description)

        form = QFormLayout(); form.setSpacing(9)
        advanced = CollapsiblePanel(); advanced_form = QFormLayout(); advanced_form.setSpacing(8)
        self.model_combo = QComboBox()
        self.model_registry = _populate_image_model_combo(
            self.model_combo, self.project_root, "batch_inference"
        )
        form.addRow("网络模型", self.model_combo)

        self.dataset_name = QLineEdit(); self.dataset_name.setReadOnly(True)
        form.addRow("数据集", self.dataset_name)
        self.dataset_edit = QLineEdit()
        dataset_row = QHBoxLayout(); dataset_row.addWidget(self.dataset_edit)
        dataset_button = QPushButton("选择目录"); dataset_button.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(dataset_button); advanced_form.addRow("数据集根目录", dataset_row)

        self.checkpoint_edit = QLineEdit()
        checkpoint_row = QHBoxLayout(); checkpoint_row.addWidget(self.checkpoint_edit)
        checkpoint_button = QPushButton("选择文件"); checkpoint_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_button); advanced_form.addRow("Checkpoint", checkpoint_row)

        self.auxiliary_label = QLabel("量化参数")
        self.quantization_edit = QLineEdit()
        quantization_row = QHBoxLayout(); quantization_row.addWidget(self.quantization_edit)
        quantization_button = QPushButton("选择文件"); quantization_button.clicked.connect(self._browse_quantization)
        quantization_row.addWidget(quantization_button); advanced_form.addRow(self.auxiliary_label, quantization_row)

        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.noise_combo = QComboBox(); _setup_noise_combo(self.noise_combo)
        form.addRow("附加随机噪声", self.noise_combo)
        self.adc_combo = QComboBox(); _setup_adc_combo(self.adc_combo)
        form.addRow("ADC 量化", self.adc_combo)
        self.error_model_edit = QLineEdit(str(platform / "error_models/01_NMOS/error_model.json"))
        self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout(); error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("选择文件"); error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button); advanced_form.addRow("器件误差模型文件", error_row)

        parameter_row = QHBoxLayout()
        self.sample_count = QSpinBox(); self.sample_count.setRange(1, 10_000); self.sample_count.setValue(1_000)
        self.group_size = QSpinBox(); self.group_size.setRange(1, 10_000); self.group_size.setValue(100)
        self.batch_size = QSpinBox(); self.batch_size.setRange(1, 2_048); self.batch_size.setValue(128)
        parameter_row.addWidget(QLabel("样本数")); parameter_row.addWidget(self.sample_count)
        parameter_row.addWidget(QLabel("每组 N")); parameter_row.addWidget(self.group_size)
        parameter_row.addWidget(QLabel("推理 batch")); parameter_row.addWidget(self.batch_size)
        parameter_row.addStretch(); advanced_form.addRow("统计设置", parameter_row)

        self.device_combo = QComboBox(); _setup_device_combo(self.device_combo)
        advanced_form.addRow("运行设备", self.device_combo)
        default_output = platform / "gui_batch_inference" / "batch_result.json"
        self.output_edit = QLineEdit(str(default_output))
        output_row = QHBoxLayout(); output_row.addWidget(self.output_edit)
        output_button = QPushButton("选择文件"); output_button.clicked.connect(self._browse_output)
        output_row.addWidget(output_button); advanced_form.addRow("结果 JSON", output_row)
        left.addLayout(form)
        advanced.content_layout.addLayout(advanced_form); left.addWidget(advanced)

        controls = QHBoxLayout()
        self.run_button = QPushButton("运行带误差批量推理"); self.run_button.clicked.connect(self.start_inference)
        self.cancel_button = QPushButton("取消"); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_inference)
        controls.addWidget(self.run_button); controls.addWidget(self.cancel_button); controls.addStretch()
        left.addLayout(controls)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); left.addWidget(self.progress)
        self.summary = QLabel("尚未运行"); self.summary.setWordWrap(True); left.addWidget(self.summary)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(145)
        left.addWidget(QLabel("运行日志")); left.addWidget(self.log)

        right_card = QFrame(); right_card.setProperty("workCard", True)
        right = QVBoxLayout(right_card); right.setContentsMargins(18, 16, 18, 16); right.setSpacing(10)
        right_title = QLabel("批量推理结果"); right_title.setProperty("sectionTitle", True); right.addWidget(right_title)
        metrics = QHBoxLayout(); metrics.setSpacing(10)
        float_card, self.float_accuracy_metric = _metric_card("浮点准确率")
        ideal_card, self.ideal_accuracy_metric = _metric_card("理想差分准确率")
        device_card, self.device_accuracy_metric = _metric_card("器件误差准确率")
        metrics.addWidget(float_card); metrics.addWidget(ideal_card); metrics.addWidget(device_card)
        right.addLayout(metrics)
        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["模式", "正确数", "样本数", "总体准确率", "耗时 (ms)"])
        _fit_result_table_height(self.result_table, max_visible_rows=3)
        right.addWidget(self.result_table)
        chart_group = QGroupBox("每 N 个样本的平均准确率柱状图")
        chart_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        chart_group.setMaximumHeight(420)
        chart_layout = QVBoxLayout(chart_group)
        self.chart_scroll = QScrollArea(); self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chart_scroll.setMinimumHeight(300); self.chart_scroll.setMaximumHeight(365)
        self.chart_label = ResponsivePixmapLabel("运行后显示分组准确率柱状图")
        self.chart_label.setMinimumHeight(305)
        self.chart_scroll.setWidget(self.chart_label); chart_layout.addWidget(self.chart_scroll)
        right.addWidget(chart_group)
        self.left_scroll, self.right_scroll = _add_independent_columns(layout, left_card, right_card)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData("resnet8-mnist")))
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
        candidates = [data_root(dataset), self.project_root / "data" / dataset]
        marker = (Path("MNIST/raw/t10k-images-idx3-ubyte") if dataset == "mnist"
                  else Path("cifar-10-batches-py/test_batch"))
        return next((path for path in candidates if (path / marker).is_file()), candidates[0])

    def _model_defaults(self, model_id: str) -> dict[str, str]:
        spec = self.model_registry.get(model_id)
        dataset_id = spec.datasets[0]
        auxiliary = str(spec.auxiliary_resources[0]) if spec.auxiliary_resources else ""
        return {
            "dataset": "MNIST" if dataset_id == "mnist" else "CIFAR-10",
            "root": str(self._dataset_root(dataset_id)),
            "checkpoint": str(spec.checkpoint or ""),
            "auxiliary": auxiliary,
            "auxiliary_label": (
                "量化/输入尺度参数" if auxiliary else "输入尺度（运行时自动校准）"
            ),
        }

    def _model_changed(self, _index: int) -> None:
        values = self._model_defaults(str(self.model_combo.currentData()))
        self.dataset_name.setText(values["dataset"]); self.dataset_edit.setText(values["root"])
        self.checkpoint_edit.setText(values["checkpoint"]); self.quantization_edit.setText(values["auxiliary"])
        self.auxiliary_label.setText(values["auxiliary_label"])
        self.quantization_edit.setEnabled(bool(values["auxiliary"]))

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
        auxiliary = self.quantization_edit.text().strip()
        sigma, rho = self.noise_combo.currentData()
        return BatchInferenceRequest(
            project_root=self.project_root,
            dataset_root=Path(self.dataset_edit.text().strip()),
            checkpoint_path=Path(self.checkpoint_edit.text().strip()),
            quantization_path=Path(auxiliary) if auxiliary else None,
            error_model_path=Path(self.error_model_edit.text().strip()),
            model_id=str(self.model_combo.currentData()), device=_device_value(self.device_combo),
            sample_count=self.sample_count.value(), group_size=self.group_size.value(),
            inference_batch_size=self.batch_size.value(), output_path=Path(self.output_edit.text().strip()),
            adc_bits=self.adc_combo.currentData(), noise_sigma=float(sigma), noise_rho=float(rho),
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
        if self.model_registry.get(model_id).auxiliary_resources:
            required.append((request.quantization_path, "辅助参数"))
        missing = [f"{name}: {path}" for path, name in required if not path.is_file()]
        if missing:
            message = "缺少批量推理资源：\n" + "\n".join(missing)
            self.summary.setText(message); self.log.appendPlainText(message); return
        self.last_outcome = None; self.result_table.setRowCount(0); self.progress.setValue(0); _fit_result_table_height(self.result_table, max_visible_rows=3)
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
            self.summary.setText("已请求取消，等待当前 Batch 完成")

    def _progressed(self, done: int, total: int) -> None:
        self.progress.setValue(round(done * 100 / total) if total else 0)
        self.summary.setText(f"批量推理运行中：{done}/{total}")

    def _succeeded(self, outcome: GroupedBatchInferenceOutcome) -> None:
        self.last_outcome = outcome
        self.float_accuracy_metric.setText(f"{outcome.modes['float'].accuracy:.2%}")
        self.ideal_accuracy_metric.setText(f"{outcome.modes['ideal_differential'].accuracy:.2%}")
        self.device_accuracy_metric.setText(f"{outcome.modes['device_differential'].accuracy:.2%}")
        self.result_table.setRowCount(len(outcome.modes))
        for row, (mode, result) in enumerate(outcome.modes.items()):
            values = (self.MODE_NAMES.get(mode, mode), result.correct, result.sample_count,
                      f"{result.accuracy:.2%}", f"{result.elapsed_ms:.3f}")
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.result_table.resizeColumnsToContents(); _fit_result_table_height(self.result_table, max_visible_rows=3)
        chart_path = self._save_chart(outcome)
        self.chart_label.set_image(chart_path)
        device_accuracy = outcome.modes["device_differential"].accuracy
        self.summary.setText(
            f"完成：{outcome.sample_count} 个样本，共 {len(outcome.groups)} 组，"
            f"器件误差准确率 {device_accuracy:.2%}，设备 {outcome.device}，"
            f"有效 batch {outcome.inference_batch_size}"
        )
        self.log.appendPlainText(f"结果 JSON：{outcome.output_path}")
        self.log.appendPlainText(f"准确率柱状图：{chart_path}")
        free_bytes = int(outcome.resources.get("cuda_free_bytes_at_start", "0"))
        ram_available = int(outcome.resources.get("ram_available_bytes_at_start", "0"))
        self.log.appendPlainText(
            "资源自适应：请求 batch "
            f"{outcome.resources.get('requested_batch_size', outcome.inference_batch_size)}，"
            f"实际最小 batch {outcome.inference_batch_size}，"
            f"实际最大 batch {outcome.resources.get('effective_max_batch_size', outcome.inference_batch_size)}，"
            f"启动时可用显存 {free_bytes / 1024 ** 3:.2f} GiB，"
            f"可用 RAM {ram_available / 1024 ** 3:.2f} GiB，"
            f"OOM 自动重试 {outcome.resources.get('oom_retries', '0')} 次"
        )
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
        group_count = len(outcome.groups)
        figure = Figure(figsize=(10.0, 3.75), constrained_layout=True, facecolor="#FFFFFF")
        FigureCanvasAgg(figure); axes = figure.add_subplot(111)
        axes.set_facecolor("#FFFFFF")
        positions = np.arange(group_count); bar_width = min(0.25, 0.78 / 3)
        colors = ("#9AA9BC", "#347FD1", "#F28C45")
        for offset, (mode, color) in enumerate(zip(
            ("float", "ideal_differential", "device_differential"), colors
        )):
            values = [group.accuracy[mode] * 100 for group in outcome.groups]
            axes.bar(positions + (offset - 1) * bar_width, values, bar_width,
                     label=self.CHART_MODE_NAMES[mode], color=color)
        labels = [f"{group.start_index}-{group.end_index}" for group in outcome.groups]
        label_step = max(1, int(np.ceil(group_count / 8)))
        shown = positions[::label_step]
        axes.set_xticks(shown, [labels[index] for index in range(0, group_count, label_step)],
                        rotation=0, ha="center", fontsize=10)
        axes.set_ylim(0, 102)
        axes.set_ylabel("准确率（%）", fontproperties=chinese_font)
        axes.set_title("分组准确率对比", color="#26344D", pad=8, loc="left",
                       fontproperties=chinese_font, fontsize=13, fontweight="bold")
        axes.grid(axis="y", color="#D9E2EE", linewidth=1.1, alpha=0.85)
        axes.set_axisbelow(True)
        axes.spines[["top", "right"]].set_visible(False)
        axes.spines[["left", "bottom"]].set_color("#9AAAC0")
        axes.spines[["left", "bottom"]].set_linewidth(1.2)
        axes.tick_params(colors="#526077", labelsize=10, width=1.1)
        axes.margins(x=0.015)
        axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015), ncol=3,
                    frameon=False, prop=chinese_font, fontsize=10, handlelength=1.4,
                    columnspacing=1.0, borderaxespad=0.0)
        figure.savefig(
            chart_path,
            dpi=300,
            facecolor="white",
            bbox_inches="tight", pad_inches=0.05,
            metadata={"Software": "NeuroCIM", "Description": "High-resolution batch inference chart"},
        )
        figure.clear()
        return chart_path

    def _failed(self, details: str) -> None:
        self.log.appendPlainText(details); self.summary.setText("批量推理失败：" + details.splitlines()[-1])

    def _cancelled(self, message: str) -> None:
        self.log.appendPlainText(message); self.summary.setText("批量推理已取消")

    def _finished(self) -> None:
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        if self.last_outcome is not None: self.progress.setValue(100)
        _retire_widget_worker(self)


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
        platform_root = output_root()
        layout = QVBoxLayout(self)
        self.setProperty("usesIndependentColumnScroll", True)
        left_card = QFrame(); left_card.setProperty("workCard", True)
        left = QVBoxLayout(left_card); left.setContentsMargins(18, 16, 18, 16); left.setSpacing(10)
        left_title = QLabel("再训练配置"); left_title.setProperty("sectionTitle", True); left.addWidget(left_title)
        description = QLabel("自动继承最近一次成功推理的网络与数据集，并选择已拟合的误差模型。")
        description.setProperty("sectionHint", True); description.setWordWrap(True); left.addWidget(description)
        form = QFormLayout(); form.setSpacing(9)
        advanced = CollapsiblePanel(); advanced_form = QFormLayout(); advanced_form.setSpacing(8)
        self.model_edit = QLineEdit("resnet8-mnist"); self.model_edit.setReadOnly(True)
        form.addRow("继承的网络模型", self.model_edit)
        self.dataset_name = QLineEdit("MNIST"); self.dataset_name.setReadOnly(True)
        form.addRow("继承的数据集", self.dataset_name)
        self.checkpoint_edit = QLineEdit(str(
            self.project_root / "3_models/VisionZoo/checkpoint/resnet8_mnist.pt"
        )); self.checkpoint_edit.setReadOnly(require_workflow)
        advanced_form.addRow("继承的 Checkpoint", self.checkpoint_edit)
        self.auxiliary_edit = QLineEdit(str(
            self.project_root / "3_models/VisionZoo/checkpoint/resnet8-mnist_input_scales.json"
        )); self.auxiliary_edit.setReadOnly(require_workflow)
        advanced_form.addRow("继承的辅助参数", self.auxiliary_edit)
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.noise_combo = QComboBox(); _setup_noise_combo(self.noise_combo)
        form.addRow("附加随机噪声", self.noise_combo)
        self.adc_combo = QComboBox(); _setup_adc_combo(self.adc_combo)
        form.addRow("ADC 量化", self.adc_combo)
        self.error_model_edit = QLineEdit(str(
            platform_root / "error_models" / "01_NMOS" / "error_model.json"
        ))
        self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout()
        error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("选择文件")
        error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button)
        advanced_form.addRow("误差模型文件", error_row)
        self.dataset_edit = QLineEdit(str(data_root("mnist")))
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_edit)
        dataset_button = QPushButton("选择目录")
        dataset_button.clicked.connect(self._browse_dataset)
        dataset_button.setEnabled(not require_workflow)
        dataset_row.addWidget(dataset_button)
        advanced_form.addRow("继承的数据集根目录", dataset_row)
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 1.0)
        self.lambda_spin.setSingleStep(0.05)
        self.lambda_spin.setValue(1.0)
        form.addRow("误差强度（lambda）", self.lambda_spin)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100)
        self.epoch_spin.setValue(1)
        advanced_form.addRow("Epoch 数", self.epoch_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(128)
        advanced_form.addRow("Batch Size", self.batch_spin)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(7)
        self.learning_rate_spin.setRange(0.0000001, 0.01)
        self.learning_rate_spin.setSingleStep(0.00001)
        self.learning_rate_spin.setValue(0.0001)
        advanced_form.addRow("学习率", self.learning_rate_spin)
        self.sequence_length_spin = QSpinBox()
        self.sequence_length_spin.setRange(2, 1024)
        self.sequence_length_spin.setValue(64)
        advanced_form.addRow("文本序列长度", self.sequence_length_spin)
        self.gradient_accumulation_spin = QSpinBox()
        self.gradient_accumulation_spin.setRange(1, 128)
        self.gradient_accumulation_spin.setValue(1)
        advanced_form.addRow("梯度累积批数", self.gradient_accumulation_spin)
        self.max_train_batches_spin = QSpinBox()
        self.max_train_batches_spin.setRange(0, 1_000_000)
        self.max_train_batches_spin.setValue(0)
        self.max_train_batches_spin.setSpecialValueText("完整训练集")
        advanced_form.addRow("每轮最大训练批数", self.max_train_batches_spin)
        self.max_eval_batches_spin = QSpinBox()
        self.max_eval_batches_spin.setRange(0, 1_000_000)
        self.max_eval_batches_spin.setValue(0)
        self.max_eval_batches_spin.setSpecialValueText("完整评估集")
        advanced_form.addRow("每种模式最大评估批数", self.max_eval_batches_spin)
        self.device_combo = QComboBox()
        _setup_device_combo(self.device_combo)
        advanced_form.addRow("运行设备", self.device_combo)
        self.output_edit = QLineEdit(str(
            platform_root / "gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "training"
        ))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_button = QPushButton("选择目录")
        output_button.clicked.connect(self._browse_output)
        output_row.addWidget(output_button)
        advanced_form.addRow("输出目录", output_row)
        left.addLayout(form)
        advanced.content_layout.addLayout(advanced_form); left.addWidget(advanced)
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
        left.addLayout(controls)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left.addWidget(self.progress)
        self.summary = QLabel("尚未运行")
        self.summary.setWordWrap(True)
        left.addWidget(self.summary)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        left.addWidget(QLabel("训练日志")); left.addWidget(self.log)

        right_card = QFrame(); right_card.setProperty("workCard", True)
        right = QVBoxLayout(right_card); right.setContentsMargins(18, 16, 18, 16); right.setSpacing(10)
        right_title = QLabel("训练指标与评估结果"); right_title.setProperty("sectionTitle", True); right.addWidget(right_title)
        metrics = QHBoxLayout(); metrics.setSpacing(10)
        before_card, self.before_accuracy_metric = _metric_card("训练前器件准确率")
        after_card, self.after_accuracy_metric = _metric_card("训练后器件准确率")
        gain_card, self.accuracy_gain_metric = _metric_card("准确率变化")
        metrics.addWidget(before_card); metrics.addWidget(after_card); metrics.addWidget(gain_card)
        right.addLayout(metrics)
        self.metrics_table = QTableWidget(0, 5)
        self.metrics_table.setHorizontalHeaderLabels(
            ["阶段", "模式", "Accuracy / Token Accuracy", "Loss", "Perplexity"]
        )
        _fit_result_table_height(self.metrics_table, max_visible_rows=8)
        right.addWidget(self.metrics_table)
        result_hint = QLabel("训练完成后，这里显示训练前后在浮点、理想差分和器件误差差分模式下的指标。")
        result_hint.setProperty("sectionHint", True); result_hint.setWordWrap(True); right.addWidget(result_hint)
        chart_group = QGroupBox("训练前后准确率对比")
        chart_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        chart_group.setMaximumHeight(410)
        chart_layout = QVBoxLayout(chart_group)
        self.training_chart_label = ResponsivePixmapLabel("训练完成后显示准确率柱状图")
        self.training_chart_label.setMinimumHeight(310)
        chart_layout.addWidget(self.training_chart_label)
        right.addWidget(chart_group)
        self.left_scroll, self.right_scroll = _add_independent_columns(layout, left_card, right_card)
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
        if str(values["network_model"]) not in FRONTEND_MODEL_ID_SET:
            self.summary.setText("历史推理使用的模型已从当前前端清单移除，请先选择可见模型并重新推理。")
            self.run_button.setEnabled(False)
            return
        self.model_edit.setText(str(values["network_model"]))
        if str(values["network_model"]) == "gpt2-large":
            self.dataset_name.setText(str(values["dataset_name"]))
            self.dataset_edit.setText(str(values["dataset_root"]))
            self.checkpoint_edit.setText(str(values["checkpoint_path"]))
            self.auxiliary_edit.setText(str(values["auxiliary_path"] or ""))
            self.summary.setText(
                "已继承 GPT-2-large 配置。当前版本暂未开放该模型的误差感知训练。"
            )
            self.run_button.setEnabled(False)
            return
        if str(values["network_model"]) == "gpt2-small":
            corpus = self.project_root / "3_models/GPT2/checkpoint/gpt2-large/wikitext-2-raw-v1"
            self.dataset_name.setText("WikiText-2")
            self.dataset_edit.setText(str(corpus))
            self.checkpoint_edit.setText(str(values["checkpoint_path"]))
            self.auxiliary_edit.setText(str(values["auxiliary_path"] or ""))
            _select_device(self.device_combo, str(values["device"]))
            self.batch_spin.setValue(1)
            self.learning_rate_spin.setValue(0.00002)
            self.sequence_length_spin.setValue(64)
            self.gradient_accumulation_spin.setValue(8)
            self.max_train_batches_spin.setValue(100)
            self.max_eval_batches_spin.setValue(10)
            index = self.error_model_combo.findData(values["error_model_id"])
            if index >= 0:
                self.error_model_combo.blockSignals(True)
                self.error_model_combo.setCurrentIndex(index)
                self.error_model_combo.blockSignals(False)
                self._sync_training_error_path()
            self.summary.setText(
                "已继承 GPT-2 Small 配置。训练将冻结主干，只更新最后一个 Transformer "
                "Block 的 attn.c_proj 与 mlp.c_proj，并使用 WikiText-2 和所选器件 LUT。"
            )
            self.run_button.setEnabled(bool(self.error_model_combo.currentData()))
            return
        inherited_model = str(values["network_model"])
        self.dataset_name.setText(str(values["dataset_name"]))
        self.dataset_edit.setText(str(values["dataset_root"]))
        self.checkpoint_edit.setText(str(values["checkpoint_path"]))
        self.auxiliary_edit.setText(str(values["auxiliary_path"] or ""))
        _select_device(self.device_combo, str(values["device"]))
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
            self, "选择器件误差模型", self.error_model_edit.text(), "JSON (*.json)"
        )
        if path:
            self.error_model_edit.setText(path)

    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 MNIST 数据集根目录", self.dataset_edit.text())
        if path:
            self.dataset_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _request(self, output_dir: Path | None = None) -> TrainingRequest:
        sigma, rho = self.noise_combo.currentData()
        return TrainingRequest(
            project_root=self.project_root,
            error_model_path=Path(self.error_model_edit.text().strip()),
            output_dir=Path(output_dir) if output_dir else Path(self.output_edit.text().strip()),
            error_strength=self.lambda_spin.value(), epochs=self.epoch_spin.value(),
            batch_size=self.batch_spin.value(), device=_device_value(self.device_combo),
            dataset_root=Path(self.dataset_edit.text().strip()),
            model_id=self.model_edit.text().strip(),
            checkpoint_path=Path(self.checkpoint_edit.text().strip()),
            auxiliary_path=(Path(self.auxiliary_edit.text().strip())
                            if self.auxiliary_edit.text().strip() else None),
            learning_rate=self.learning_rate_spin.value(),
            sequence_length=self.sequence_length_spin.value(),
            gradient_accumulation_steps=self.gradient_accumulation_spin.value(),
            max_train_batches=(self.max_train_batches_spin.value() or None),
            max_eval_batches=(self.max_eval_batches_spin.value() or None),
            adc_bits=self.adc_combo.currentData(), noise_sigma=float(sigma), noise_rho=float(rho),
        )

    def start_training(self, output_dir: Path | None = None) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.last_outcome = None
        self.metrics_table.setRowCount(0)
        _fit_result_table_height(self.metrics_table, max_visible_rows=8)
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
            self.summary.setText("训练配置错误：" + str(exc).splitlines()[0])
            self.log.appendPlainText(traceback.format_exc())
            self._restore_controls()
            return
        self.output_edit.setText(str(request.output_dir))
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.summary.setText("正在进行误差感知训练")
        self.log.appendPlainText(f"开始完整训练任务：{request.output_dir}")
        if request.model_id == "tinytransformer-cifar10":
            self.log.appendPlainText(
                "TinyTransformer 微调：冻结骨干网络，仅训练注意力池化头和分类头；"
                "冻结部分使用浮点前向计算。"
            )
        elif request.model_id == "gpt2-small":
            self.log.appendPlainText(
                "GPT-2 Small 微调：冻结主干，只训练最后一个 Transformer Block 的两个 "
                "c_proj；器件 LUT 前向、STE 反向。"
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
            self.summary.setText("已请求取消，等待当前 Batch 完成")
            self.log.appendPlainText("已请求取消")
            self.worker.request_cancel()

    def _training_progressed(self, percent: int, message: str, metrics: object) -> None:
        self.progress.setValue(percent)
        detail = ""
        if isinstance(metrics, dict) and "train_loss" in metrics:
            detail = f" | loss={metrics['train_loss']:.6f}, accuracy={metrics['train_accuracy']:.4f}"
        shown_message = _translate_runtime_message(message)
        self.summary.setText(shown_message)
        self.log.appendPlainText(shown_message + detail)

    def _training_succeeded(self, outcome: TrainingOutcome) -> None:
        self.last_outcome = outcome
        before_device = float(outcome.evaluations["before"]["device_differential"]["accuracy"])
        after_device = float(outcome.evaluations["after"]["device_differential"]["accuracy"])
        self.before_accuracy_metric.setText(f"{before_device:.2%}")
        self.after_accuracy_metric.setText(f"{after_device:.2%}")
        self.accuracy_gain_metric.setText(f"{after_device - before_device:+.2%}")
        rows = []
        for stage in ("before", "after"):
            for mode, result in outcome.evaluations[stage].items():
                rows.append((
                    _stage_display_name(stage), _mode_display_name(mode), result["accuracy"], result["loss"],
                    result.get("perplexity", "-"),
                ))
        self.metrics_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                shown = f"{value:.6f}" if isinstance(value, float) else str(value)
                self.metrics_table.setItem(row, column, QTableWidgetItem(shown))
        self.metrics_table.resizeColumnsToContents(); _fit_result_table_height(self.metrics_table, max_visible_rows=8)
        update = outcome.weight_update
        self.summary.setText(
            f"训练完成：耗时 {outcome.elapsed_seconds:.1f} s，设备 {outcome.device}；"
            f"权重 L2 变化量={update['l2_delta']:.6g}；结果={outcome.result_path}"
        )
        self.log.appendPlainText("Checkpoint：" + outcome.checkpoint_path)
        self.log.appendPlainText("训练历史：" + outcome.history_path)
        self.log.appendPlainText("结果：" + outcome.result_path)
        chart_path = self._save_training_chart(outcome)
        self.training_chart_label.set_image(chart_path)
        self.log.appendPlainText("训练前后准确率柱状图：" + str(chart_path))
        self.progress.setValue(100)
        self.export_button.setEnabled(True)
        if self.coordinator is not None:
            self.coordinator.context.record_training(outcome.as_dict())

    @staticmethod
    def _save_training_chart(outcome: TrainingOutcome) -> Path:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties

        result_path = Path(outcome.result_path).resolve()
        chart_path = result_path.with_name("training_accuracy_comparison.png")
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
        chinese_font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
        modes = ("float", "ideal_differential", "device_differential")
        labels = ("浮点模型", "理想差分", "器件误差差分")
        before = [float(outcome.evaluations["before"][mode]["accuracy"]) * 100 for mode in modes]
        after = [float(outcome.evaluations["after"][mode]["accuracy"]) * 100 for mode in modes]
        positions = np.arange(len(modes))
        width = 0.32

        figure = Figure(figsize=(7.4, 3.6), constrained_layout=True, facecolor="#FFFFFF")
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(111)
        axes.set_facecolor("#FFFFFF")
        axes.bar(positions - width / 2, before, width, label="训练前", color="#9AA9BC")
        axes.bar(positions + width / 2, after, width, label="训练后", color="#347FD1")
        axes.set_xticks(positions, labels, fontproperties=chinese_font)
        axes.set_ylim(0, 102)
        axes.set_ylabel("准确率（%）", fontproperties=chinese_font)
        axes.set_title("训练前后准确率对比", loc="left", pad=7,
                       fontproperties=chinese_font, fontsize=13, fontweight="bold")
        axes.grid(axis="y", color="#D9E2EE", linewidth=1.1, alpha=0.85)
        axes.set_axisbelow(True)
        axes.spines[["top", "right"]].set_visible(False)
        axes.spines[["left", "bottom"]].set_color("#9AAAC0")
        axes.spines[["left", "bottom"]].set_linewidth(1.2)
        axes.tick_params(colors="#526077", labelsize=10, width=1.1)
        axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                    frameon=False, prop=chinese_font, fontsize=10)
        figure.savefig(chart_path, dpi=300, facecolor="white", bbox_inches="tight")
        figure.clear()
        return chart_path

    def _training_failed(self, details: str) -> None:
        self.log.appendPlainText(details)
        self.summary.setText("训练失败：" + details.splitlines()[-1])

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
        _retire_widget_worker(self)

    def export_result(self) -> None:
        if self.last_outcome is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出训练结果", "training_result.json", "JSON (*.json)"
        )
        if path:
            Path(path).write_text(
                Path(self.last_outcome.result_path).read_text(encoding="utf-8"), encoding="utf-8"
            )
            self.log.appendPlainText("结果已导出：" + str(Path(path).resolve()))


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
        platform_root = output_root()
        layout = QVBoxLayout(self)
        description = QLabel(
            "导出正负差分跟踪数据并运行随软件提供的官方 NeuroSim 内核。"
            "指标从 NeuroSim 标准输出中读取，同时校验并记录所选误差文件。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItem("LeNet-5", "lenet5-mnist")
        form.addRow("模型", self.model_combo)
        self.checkpoint_edit = QLineEdit(
            str(platform_root / "eat" / "01_NMOS" / "eat_checkpoint.pt")
        )
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint_edit)
        checkpoint_button = QPushButton("选择文件")
        checkpoint_button.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_button)
        form.addRow("Checkpoint", checkpoint_row)
        self.error_model_edit = QLineEdit(
            str(platform_root / "error_models" / "01_NMOS" / "error_model.json")
        )
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit.setReadOnly(require_workflow)
        error_row = QHBoxLayout()
        error_row.addWidget(self.error_model_edit)
        error_button = QPushButton("选择文件")
        error_button.clicked.connect(self._browse_error_model)
        error_button.setEnabled(not require_workflow)
        error_row.addWidget(error_button)
        form.addRow("误差模型", error_row)
        self.subarray_rows = QSpinBox()
        self.subarray_rows.setRange(1, 4096)
        self.subarray_rows.setValue(128)
        form.addRow("子阵列行数", self.subarray_rows)
        self.parallel_rows = QSpinBox()
        self.parallel_rows.setRange(1, 4096)
        self.parallel_rows.setValue(128)
        form.addRow("并行行数", self.parallel_rows)
        layout.addLayout(form)

        self.run_button = QPushButton("运行官方 NeuroSim PPA 评估")
        self.run_button.clicked.connect(self.start_evaluation)
        layout.addWidget(self.run_button)
        self.summary = QLabel("尚未运行")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["指标", "数值"])
        _fit_result_table_height(self.result_table, max_visible_rows=8)
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
        path, _ = QFileDialog.getOpenFileName(self, "选择 Checkpoint", self.checkpoint_edit.text())
        if path:
            self.checkpoint_edit.setText(path)

    def _browse_error_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择误差模型", self.error_model_edit.text(), "JSON (*.json)"
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
        _fit_result_table_height(self.result_table, max_visible_rows=8)
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)
        target = output_dir or (
            output_root() / "gui_runs"
            / datetime.now().strftime("%Y%m%d-%H%M%S") / "ppa"
        )
        self.log.appendPlainText(f"开始 NeuroSim PPA 评估：{target}")
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
            ("芯片面积", outcome.metrics.get("chip_area_um2"), "um^2"),
            ("延迟", outcome.metrics.get("read_latency_ns"), "ns/图像"),
            ("动态能耗", outcome.metrics.get("read_dynamic_energy_pj"), "pJ/图像"),
            ("能效", outcome.metrics.get("energy_efficiency_tops_per_w"), "TOPS/W"),
            ("吞吐率", outcome.metrics.get("throughput_fps"), "FPS"),
        ]
        self.result_table.setRowCount(len(display))
        for row, (name, value, unit) in enumerate(display):
            shown = "不可用" if value is None else f"{value:.6g} {unit}"
            self.result_table.setItem(row, 0, QTableWidgetItem(name))
            self.result_table.setItem(row, 1, QTableWidgetItem(shown))
        self.result_table.resizeColumnsToContents(); _fit_result_table_height(self.result_table, max_visible_rows=8)
        self.summary.setText(
            f"官方 NeuroSim 评估完成。JSON：{outcome.result_path}｜日志：{outcome.log_path}"
        )
        self.log.appendPlainText(f"跟踪数据目录：{outcome.trace_directory}")
        self.log.appendPlainText(f"跟踪输入：{outcome.trace_input}")
        self.log.appendPlainText(f"结果 JSON：{outcome.result_path}")
        self.log.appendPlainText(f"NeuroSim 日志：{outcome.log_path}")

    def _ppa_failed(self, details: str) -> None:
        self.log.appendPlainText(details)
        self.summary.setText("PPA 评估失败：" + details.splitlines()[-1])

    def _ppa_finished(self) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if self.last_outcome is not None else 0)
        self.run_button.setEnabled(True)
        _retire_widget_worker(self)


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
        self.model_variant = "gpt2-large"
        self.dataset_name = "wikitext2"
        self.dataset_root = model_root / "gpt2-large/wikitext-2-raw-v1"
        platform_root = output_root()

        layout = QVBoxLayout(self)
        description = QLabel(
            "输入文本提示词，对比 GPT-2-large 的浮点、理想差分和所选器件误差差分生成结果。"
            "差分计算映射到最后一个 Transformer Block 的注意力输出投影与 MLP 输出投影。"
        )
        description.setWordWrap(True); layout.addWidget(description)
        form = QFormLayout()
        self.model_label = QLabel("GPT-2-large / WikiText-2")
        form.addRow("继承的模型与数据集", self.model_label)
        self.checkpoint_edit = QLineEdit(str(model_root / "gpt2_nanogpt.pt")); self.checkpoint_edit.setReadOnly(True)
        form.addRow("GPT-2 Checkpoint", self.checkpoint_edit)
        self.tokenizer_edit = QLineEdit(str(model_root / "tokenizer")); self.tokenizer_edit.setReadOnly(True)
        form.addRow("GPT-2 分词器", self.tokenizer_edit)
        self.scales_edit = QLineEdit(str(model_root / "input_scales.json")); self.scales_edit.setReadOnly(True)
        form.addRow("输入尺度 JSON", self.scales_edit)
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit = QLineEdit(str(platform_root / "error_models/01_NMOS/error_model.json"))
        self.error_model_edit.setReadOnly(require_workflow); form.addRow("误差模型文件", self.error_model_edit)
        self.max_tokens = QSpinBox(); self.max_tokens.setRange(1, 128); self.max_tokens.setValue(24)
        form.addRow("生成 Token 数", self.max_tokens)
        self.lambda_spin = QDoubleSpinBox(); self.lambda_spin.setRange(0.0, 1.0)
        self.lambda_spin.setSingleStep(0.05); self.lambda_spin.setValue(1.0)
        form.addRow("误差强度 lambda", self.lambda_spin)
        self.device_combo = QComboBox(); _setup_device_combo(self.device_combo)
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
        self.result_table.setHorizontalHeaderLabels(["模式", "生成 Token", "耗时 (ms)", "Token/s"])
        _fit_result_table_height(self.result_table, max_visible_rows=3)
        layout.addWidget(self.result_table)
        self.output_text = QPlainTextEdit(); self.output_text.setReadOnly(True); self.output_text.setMinimumHeight(240)
        layout.addWidget(self.output_text)
        self.top5_table = QTableWidget(0, 4)
        self.top5_table.setHorizontalHeaderLabels(["模式", "排名", "最后一步候选 Token", "置信度"])
        _fit_result_table_height(self.top5_table, max_visible_rows=8)
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

    def apply_model_selection(self, spec) -> None:
        if spec.model_id not in {"gpt2-small", "gpt2-large"}:
            return
        self.model_variant = spec.model_id
        self.dataset_name = spec.datasets[0]
        checkpoint_root = self.project_root / "3_models/GPT2/checkpoint"
        if spec.model_id == "gpt2-large":
            resource_root = checkpoint_root / "gpt2-large"
            runtime = resource_root / "model.fp16.safetensors"
            self.checkpoint_edit.setText(str(runtime if runtime.is_file() else resource_root / "model.safetensors"))
            self.tokenizer_edit.setText(str(resource_root))
            self.scales_edit.setText(str(resource_root / "input_scales.json"))
            self.dataset_root = resource_root / "wikitext-2-raw-v1"
        else:
            self.checkpoint_edit.setText(str(checkpoint_root / "gpt2_nanogpt.pt"))
            self.tokenizer_edit.setText(str(checkpoint_root / "tokenizer"))
            self.scales_edit.setText(str(checkpoint_root / "input_scales.json"))
            self.dataset_root = checkpoint_root / "tokenizer"
        self.model_label.setText(f"{spec.display_name} / {self.dataset_name}")

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
            device=_device_value(self.device_combo), model_variant=self.model_variant,
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
        _fit_result_table_height(self.result_table, max_visible_rows=3); _fit_result_table_height(self.top5_table, max_visible_rows=8)
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.summary.setText("正在运行 GPT-2 三模式生成")
        self.worker = GPT2GenerationThread(self.service, request)
        self.worker.progressed.connect(self._progressed); self.worker.succeeded.connect(self._succeeded)
        self.worker.failed.connect(self._failed); self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._finished); self.worker.start()

    def cancel_generation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel(); self.cancel_button.setEnabled(False)

    def _progressed(self, percent: int, message: str) -> None:
        self.progress.setValue(percent); self.summary.setText(_translate_runtime_message(message))

    def _succeeded(self, outcome: GPT2GenerationOutcome) -> None:
        self.last_outcome = outcome
        self.result_table.setRowCount(len(outcome.modes))
        top_rows = sum(len(mode.final_top5) for mode in outcome.modes)
        self.top5_table.setRowCount(top_rows)
        top_row = 0; sections = []
        for row, mode in enumerate(outcome.modes):
            shown_mode = _mode_display_name(mode.mode)
            values = (shown_mode, len(mode.generated_token_ids), f"{mode.elapsed_ms:.2f}",
                      f"{mode.tokens_per_second:.2f}")
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(str(value)))
            sections.append(f"【{shown_mode}】\n{mode.text}")
            for rank, (token, confidence) in enumerate(mode.final_top5, 1):
                for column, value in enumerate((shown_mode, rank, repr(token), f"{confidence:.2%}")):
                    self.top5_table.setItem(top_row, column, QTableWidgetItem(str(value)))
                top_row += 1
        self.output_text.setPlainText("\n\n".join(sections))
        self.result_table.resizeColumnsToContents(); self.top5_table.resizeColumnsToContents(); _fit_result_table_height(self.result_table, max_visible_rows=3); _fit_result_table_height(self.top5_table, max_visible_rows=8)
        output = output_root() / "gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "gpt2"
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "generation_result.json"
        result_path.write_text(json.dumps(outcome.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.summary.setText(
            f"完成：提示词包含 {outcome.prompt_token_count} 个 Token，三模式总耗时 {outcome.total_elapsed_ms:.1f} ms"
        )
        self.log.appendPlainText(f"结果：{result_path}")
        if self.coordinator is not None and self.error_model_combo.currentData():
            try:
                self.coordinator.record_inference(
                    model_id=str(self.error_model_combo.currentData()), dataset_name=self.dataset_name,
                    dataset_root=str(self.dataset_root), network_model=outcome.model_variant,
                    checkpoint_path=self.checkpoint_edit.text(), auxiliary_path=self.scales_edit.text(),
                    device=outcome.device, result=outcome.as_dict(),
                )
                self.log.appendPlainText("已记录当前 GPT-2 模型、文本任务和误差模型")
            except Exception as exc:
                self.log.appendPlainText("工作流记录失败：" + str(exc))

    def _failed(self, trace: str) -> None:
        self.summary.setText("GPT-2 生成失败"); self.log.appendPlainText(trace)

    def _cancelled(self, message: str) -> None:
        shown = _translate_runtime_message(message); self.summary.setText(shown); self.log.appendPlainText(shown)

    def _finished(self) -> None:
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        _retire_widget_worker(self)


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
        self.model_variant = "gpt2-large"
        self.dataset_name = "prompt_text"
        self.dataset_root = checkpoint_root / "tokenizer"
        platform_root = output_root()
        layout = QVBoxLayout(self)
        description = QLabel(
            "对多条提示词执行浮点、理想差分和器件误差差分生成，统计器件输出相对浮点输出的 Token 一致率。"
        )
        description.setWordWrap(True); layout.addWidget(description)
        form = QFormLayout()
        self.model_label = QLabel("GPT-2-large / WikiText-2")
        form.addRow("继承的模型与数据集", self.model_label)
        self.checkpoint_edit = QLineEdit(str(checkpoint_root / "gpt2_nanogpt.pt")); self.checkpoint_edit.setReadOnly(True)
        self.tokenizer_edit = QLineEdit(str(checkpoint_root / "tokenizer")); self.tokenizer_edit.setReadOnly(True)
        self.scales_edit = QLineEdit(str(checkpoint_root / "input_scales.json")); self.scales_edit.setReadOnly(True)
        form.addRow("Checkpoint", self.checkpoint_edit)
        form.addRow("分词器", self.tokenizer_edit)
        self.error_model_combo = QComboBox(); form.addRow("误差建模方法", self.error_model_combo)
        self.error_model_edit = QLineEdit(str(platform_root / "error_models/01_NMOS/error_model.json"))
        self.error_model_edit.setReadOnly(require_workflow); form.addRow("误差模型", self.error_model_edit)
        self.max_tokens = QSpinBox(); self.max_tokens.setRange(1, 64); self.max_tokens.setValue(8)
        form.addRow("每条生成 Token 数", self.max_tokens)
        self.lambda_spin = QDoubleSpinBox(); self.lambda_spin.setRange(0.0, 1.0)
        self.lambda_spin.setSingleStep(0.05); self.lambda_spin.setValue(1.0)
        form.addRow("误差强度 lambda", self.lambda_spin)
        self.device_combo = QComboBox(); _setup_device_combo(self.device_combo)
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
        _fit_result_table_height(self.result_table, max_visible_rows=6)
        layout.addWidget(self.result_table)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100)
        layout.addWidget(self.log)
        self.evaluation_widget = WikiText2EvaluationWidget(self.project_root)
        self.evaluation_widget.setVisible(False)
        layout.addWidget(self.evaluation_widget)
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

    def apply_model_selection(self, spec) -> None:
        if spec.model_id not in {"gpt2-small", "gpt2-large"}:
            return
        self.model_variant = spec.model_id
        self.dataset_name = spec.datasets[0]
        checkpoint_root = self.project_root / "3_models/GPT2/checkpoint"
        if spec.model_id == "gpt2-large":
            resource_root = checkpoint_root / "gpt2-large"
            runtime = resource_root / "model.fp16.safetensors"
            self.checkpoint_edit.setText(str(runtime if runtime.is_file() else resource_root / "model.safetensors"))
            self.tokenizer_edit.setText(str(resource_root))
            self.scales_edit.setText(str(resource_root / "input_scales.json"))
            self.dataset_root = resource_root / "wikitext-2-raw-v1"
        else:
            self.checkpoint_edit.setText(str(checkpoint_root / "gpt2_nanogpt.pt"))
            self.tokenizer_edit.setText(str(checkpoint_root / "tokenizer"))
            self.scales_edit.setText(str(checkpoint_root / "input_scales.json"))
            self.dataset_root = checkpoint_root / "tokenizer"
        self.model_label.setText(f"{spec.display_name} / {self.dataset_name}")
        if hasattr(self, "evaluation_widget"):
            self.evaluation_widget.apply_model_selection(spec)

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
            device=_device_value(self.device_combo), model_variant=self.model_variant,
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
        self.result_table.setRowCount(0); self.progress.setValue(0); _fit_result_table_height(self.result_table, max_visible_rows=6)
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.summary.setText(f"正在处理 {len(requests)} 条提示词")
        self.worker = GPT2BatchThread(self.service, requests)
        self.worker.progressed.connect(lambda percent, message: (self.progress.setValue(percent), self.summary.setText(_translate_runtime_message(message))))
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
        self.result_table.resizeColumnsToContents(); _fit_result_table_height(self.result_table, max_visible_rows=6)
        average = sum(agreements) / len(agreements)
        output = output_root() / "gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S") / "gpt2_batch"
        output.mkdir(parents=True, exist_ok=True); result_path = output / "batch_result.json"
        result_path.write_text(json.dumps({"average_device_token_agreement": average,
                                           "samples": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.summary.setText(f"完成：{len(outcomes)} 条提示词，器件/浮点平均 token 一致率 {average:.2%}")
        self.log.appendPlainText("结果：" + str(result_path))
        if self.coordinator is not None and self.error_model_combo.currentData():
            try:
                self.coordinator.record_inference(
                    model_id=str(self.error_model_combo.currentData()), dataset_name=self.dataset_name,
                    dataset_root=str(self.dataset_root), network_model=outcomes[0].model_variant,
                    checkpoint_path=self.checkpoint_edit.text(), auxiliary_path=self.scales_edit.text(),
                    device=outcomes[0].device,
                    result={"average_device_token_agreement": average, "samples": payload},
                )
            except Exception as exc:
                self.log.appendPlainText("工作流记录失败：" + str(exc))

    def _failed(self, trace: str) -> None:
        self.summary.setText("GPT-2 批量推理失败"); self.log.appendPlainText(trace)

    def _cancelled(self, message: str) -> None:
        shown = _translate_runtime_message(message); self.summary.setText(shown); self.log.appendPlainText(shown)

    def _finished(self) -> None:
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        if self.result_table.rowCount(): self.progress.setValue(100)
        _retire_widget_worker(self)


class WikiText2EvaluationWidget(QWidget):
    """Offline WikiText-2 sliding-window loss/perplexity evaluation for GPT-2 Large."""

    DATASET_SHA256 = {
        "train": "57f3eb280cfe78d8cebe3aa76d7416607845416e22464dd9709ccb487f8d7b92",
        "validation": "fbf387067edf87fc0f11e8cc1708be8ff6c944f6eeb98c1e34140c8a95629a00",
        "test": "7952b062817620ac99b47296dc945942b55068d0e85baf1f95f300c7ad1480ef",
    }

    def __init__(self, project_root: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.service = WikiText2EvaluationService()
        self.worker: WikiText2EvaluationThread | None = None
        self.last_outcome: WikiText2EvaluationOutcome | None = None
        self.model_variant = "gpt2-large"
        resource_root = self.project_root / "3_models/GPT2/checkpoint/gpt2-large"

        layout = QVBoxLayout(self)
        group = QGroupBox("WikiText-2 语言模型评估（浮点基线）")
        group_layout = QVBoxLayout(group)
        note = QLabel(
            "使用离线 WikiText-2 原始数据 v1 计算滑窗负对数似然和困惑度。"
            "Batch Size 固定为 1；每个目标 Token 只计入一次。"
        )
        note.setWordWrap(True); group_layout.addWidget(note)
        form = QFormLayout()
        self.model_label = QLabel("GPT-2 Large")
        form.addRow("继承的模型", self.model_label)
        runtime = resource_root / "model.fp16.safetensors"
        self.checkpoint_edit = QLineEdit(str(runtime if runtime.is_file() else resource_root / "model.safetensors"))
        self.checkpoint_edit.setReadOnly(True); form.addRow("Checkpoint", self.checkpoint_edit)
        self.tokenizer_edit = QLineEdit(str(resource_root))
        self.tokenizer_edit.setReadOnly(True); form.addRow("分词器", self.tokenizer_edit)
        self.dataset_edit = QLineEdit(str(resource_root / "wikitext-2-raw-v1"))
        self.dataset_edit.setReadOnly(True); form.addRow("WikiText-2 原始数据", self.dataset_edit)
        self.split_combo = QComboBox()
        for shown, value in (("测试集", "test"), ("验证集", "validation"), ("训练集", "train")):
            self.split_combo.addItem(shown, value)
        form.addRow("数据切分", self.split_combo)
        self.context_spin = QSpinBox(); self.context_spin.setRange(2, 1024); self.context_spin.setValue(1024)
        form.addRow("上下文长度", self.context_spin)
        self.stride_spin = QSpinBox(); self.stride_spin.setRange(1, 1023); self.stride_spin.setValue(512)
        form.addRow("滑窗步长", self.stride_spin)
        self.max_tokens_spin = QSpinBox(); self.max_tokens_spin.setRange(0, 10_000_000)
        self.max_tokens_spin.setValue(4096); self.max_tokens_spin.setSpecialValueText("完整语料")
        form.addRow("最多评估 Token", self.max_tokens_spin)
        self.device_combo = QComboBox(); _setup_device_combo(self.device_combo)
        form.addRow("运行设备", self.device_combo)
        group_layout.addLayout(form)
        controls = QHBoxLayout()
        self.run_button = QPushButton("开始 WikiText-2 评估")
        self.run_button.clicked.connect(self.start_evaluation)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False); self.cancel_button.clicked.connect(self.cancel_evaluation)
        controls.addWidget(self.run_button); controls.addWidget(self.cancel_button); controls.addStretch()
        group_layout.addLayout(controls)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); group_layout.addWidget(self.progress)
        self.summary = QLabel("尚未运行"); self.summary.setWordWrap(True); group_layout.addWidget(self.summary)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["指标", "结果"])
        _fit_result_table_height(self.result_table, max_visible_rows=7); group_layout.addWidget(self.result_table)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(100)
        group_layout.addWidget(self.log)
        layout.addWidget(group)

    def apply_model_selection(self, spec) -> None:
        visible = spec.model_id == "gpt2-large"
        self.setVisible(visible)
        if not visible:
            return
        self.model_variant = spec.model_id
        resource_root = Path(spec.checkpoint).parent
        self.model_label.setText(f"{spec.display_name} / {spec.datasets[0]}")
        self.checkpoint_edit.setText(str(spec.checkpoint))
        self.tokenizer_edit.setText(str(resource_root))
        self.dataset_edit.setText(str(spec.dataset_root))

    def _request(self) -> WikiText2EvaluationRequest:
        split = str(self.split_combo.currentData())
        maximum = self.max_tokens_spin.value()
        return WikiText2EvaluationRequest(
            project_root=self.project_root, model_variant=self.model_variant,
            checkpoint_path=Path(self.checkpoint_edit.text()),
            tokenizer_path=Path(self.tokenizer_edit.text()),
            dataset_path=Path(self.dataset_edit.text()), split=split,
            device=_device_value(self.device_combo), stride=self.stride_spin.value(),
            context_length=self.context_spin.value(), batch_size=1,
            max_eval_tokens=maximum or None,
            expected_dataset_sha256=self.DATASET_SHA256[split],
        )

    def start_evaluation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        request = self._request()
        if request.stride >= request.context_length:
            self.summary.setText("滑窗步长必须小于上下文长度")
            return
        self.last_outcome = None; self.result_table.setRowCount(0); self.progress.setValue(0); _fit_result_table_height(self.result_table, max_visible_rows=7)
        self.run_button.setEnabled(False); self.cancel_button.setEnabled(True)
        self.summary.setText("正在评估 WikiText-2……")
        self.worker = WikiText2EvaluationThread(self.service, request)
        self.worker.progressed.connect(self._progressed)
        self.worker.succeeded.connect(self._succeeded)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def cancel_evaluation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_cancel(); self.cancel_button.setEnabled(False)

    def _progressed(self, percent: int, message: str) -> None:
        self.progress.setValue(percent); self.summary.setText(_translate_runtime_message(message))

    def _succeeded(self, outcome: WikiText2EvaluationOutcome) -> None:
        self.last_outcome = outcome
        values = (
            ("loss", f"{outcome.loss:.6f}"),
            ("perplexity", f"{outcome.perplexity:.6f}"),
            ("预测 token 数", str(outcome.predicted_token_count)),
            ("滑窗数", str(outcome.window_count)),
            ("是否截断", "是" if outcome.truncated else "否"),
            ("模型 SHA256", outcome.checkpoint_sha256),
            ("数据 SHA256", outcome.dataset_sha256),
            ("误差注入范围", outcome.injection_scope),
        )
        self.result_table.setRowCount(len(values))
        for row, (name, value) in enumerate(values):
            self.result_table.setItem(row, 0, QTableWidgetItem(name))
            self.result_table.setItem(row, 1, QTableWidgetItem(value))
        self.result_table.resizeColumnsToContents(); _fit_result_table_height(self.result_table, max_visible_rows=7)
        output = (output_root() / "gui_runs"
                  / datetime.now().strftime("%Y%m%d-%H%M%S") / "wikitext2")
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "evaluation_result.json"
        result_path.write_text(
            json.dumps(outcome.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        self.summary.setText(
            f"完成：loss={outcome.loss:.6f}，perplexity={outcome.perplexity:.6f}，"
            f"Token 数={outcome.predicted_token_count}"
        )
        self.log.appendPlainText(f"结果：{result_path}")

    def _failed(self, trace: str) -> None:
        self.summary.setText("WikiText-2 评估失败"); self.log.appendPlainText(trace)

    def _cancelled(self, message: str) -> None:
        shown = _translate_runtime_message(message); self.summary.setText(shown); self.log.appendPlainText(shown)

    def _finished(self) -> None:
        self.run_button.setEnabled(True); self.cancel_button.setEnabled(False)
        if self.last_outcome is not None:
            self.progress.setValue(100)
        _retire_widget_worker(self)


class UnifiedSingleInferenceWidget(QWidget):
    """Switch inference input/output by the model selected in the catalog."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None, require_workflow: bool = False):
        super().__init__(parent)
        if coordinator is None:
            raise ValueError("Unified inference requires a workflow coordinator")
        self.setProperty("usesIndependentColumnScroll", True)
        self.coordinator = coordinator
        layout = QVBoxLayout(self)
        self.selection = QLabel(); self.selection.setWordWrap(True); layout.addWidget(self.selection)
        self.stack = QStackedWidget()
        self.stack.setMinimumSize(0, 0)
        self.stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.stack)
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
            self.gpt_widget.apply_model_selection(spec)
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
        self.setProperty("usesIndependentColumnScroll", True)
        self.coordinator = coordinator
        layout = QVBoxLayout(self)
        self.selection = QLabel(); self.selection.setWordWrap(True); layout.addWidget(self.selection)
        self.stack = QStackedWidget()
        self.stack.setMinimumSize(0, 0)
        self.stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.stack)
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
            self.gpt_widget.apply_model_selection(spec)
            self.stack.setCurrentWidget(self.gpt_widget)
        else:
            index = self.image_widget.model_combo.findData(model_id)
            if index >= 0: self.image_widget.model_combo.setCurrentIndex(index)
            self.stack.setCurrentWidget(self.image_widget)


class MainWindow(QMainWindow):
    NAVIGATION = (
        ("01", "首页"), ("02", "器件误差建模"), ("03", "阵列 MAC 分析"),
        ("04", "模型与数据集"), ("05", "单样本推理"), ("06", "批量推理"),
        ("07", "误差感知训练"),
    )

    def __init__(self, platform_root: Path):
        super().__init__()
        self.platform_root = Path(platform_root).resolve()
        self.update_check_thread: UpdateCheckThread | None = None
        self.update_download_thread: UpdateDownloadThread | None = None
        self.update_progress: QProgressDialog | None = None
        application = QApplication.instance()
        if application is not None:
            configure_windows_font(application)
            apply_application_style(application)
        self.setWindowTitle(f"DRAM 存算仿真平台 — NeuroSim  v{__version__}")
        self.setMinimumSize(960, 600)
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(1440, int(available.width() * 0.94)), min(900, int(available.height() * 0.90)))
        else:
            self.resize(1366, 820)

        self.workflow = WorkflowCoordinator(platform_root)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.analysis_tabs = QTabWidget(self.tabs)
        self.analysis_tabs.setObjectName("analysisTabs")
        self.analysis_tabs.setDocumentMode(True)
        self.cell_fit_widget = CellFittingWidget(platform_root, coordinator=self.workflow)
        self.mac_analysis_widget = MacAnalysisWidget(platform_root)
        self.analysis_tabs.addTab(
            self._scrollable(self.cell_fit_widget, "器件误差建模", "导入器件测量 CSV，检查列映射并生成可供推理和训练选择的多种误差模型。"),
            "器件误差拟合",
        )
        self.analysis_tabs.addTab(self._scrollable(self.mac_analysis_widget, "阵列 MAC 分析", "拟合并分析不同阵列规模、位宽和实验组下的 MAC 输出误差。"), "阵列 MAC 分析")
        self.analysis_tabs.tabBar().hide()
        self.cell_fit_widget.fit_mode_requested.connect(self._set_analysis_fit_mode)
        self.mac_analysis_widget.fit_mode_requested.connect(self._set_analysis_fit_mode)
        self.analysis_tabs.currentChanged.connect(self._sync_analysis_fit_mode_buttons)
        self._sync_analysis_fit_mode_buttons(0)
        self.tabs.addTab(self.analysis_tabs, "误差建模与阵列分析")
        self.tabs.addTab(
            self._scrollable(ModelCatalogWidget(platform_root.parent, coordinator=self.workflow), "模型与数据集", "统一选择后续单样本推理、批量推理和误差感知训练使用的模型数据集组合。"),
            "网络与数据集",
        )
        self.tabs.addTab(
            self._scrollable(UnifiedSingleInferenceWidget(platform_root.parent, coordinator=self.workflow, require_workflow=True), "单样本带误差推理", "对同一样本比较浮点、理想差分和器件误差差分结果；文本模型显示生成结果。"), "单样本推理"
        )
        self.tabs.addTab(
            self._scrollable(UnifiedBatchInferenceWidget(platform_root.parent, coordinator=self.workflow, require_workflow=True), "批量带误差推理", "按样本分组汇总 Accuracy 或语言模型指标，比较三种计算模式。"), "批量推理"
        )
        self.tabs.addTab(
            self._scrollable(ErrorAwareTrainingWidget(platform_root.parent, coordinator=self.workflow, require_workflow=True), "误差感知训练", "继承已选择的网络、数据集和器件误差模型，执行可恢复的误差感知微调。"), "误差感知训练"
        )
        self.tabs.tabBar().hide()
        self.shell_stack = QStackedWidget(); self.home_page = self._build_home_page()
        self.shell_stack.addWidget(self.home_page); self.shell_stack.addWidget(self.tabs)
        self.setCentralWidget(self._build_navigation_shell())
        self.statusBar().showMessage("就绪")
        help_menu = self.menuBar().addMenu("帮助")
        update_action = QAction("检查更新", self)
        update_action.triggered.connect(lambda: self._check_for_updates(interactive=True))
        help_menu.addAction(update_action)
        help_menu.addSeparator()
        version_action = QAction(f"当前版本：{__version__}", self)
        version_action.setEnabled(False)
        help_menu.addAction(version_action)
        self.menuBar().hide()
        self._apply_visual_roles()
        self._configure_long_running_safety()
        self.workflow.models_changed.connect(self._refresh_home_summary)
        self.workflow.selection_changed.connect(lambda _value: self._refresh_home_summary())
        self.workflow.network_changed.connect(lambda _value: self._refresh_home_summary())
        self._show_navigation_page(0); self._refresh_home_summary()
        if CredentialStore.load():
            QTimer.singleShot(5000, lambda: self._check_for_updates(interactive=False))

    def _set_analysis_fit_mode(self, index: int) -> None:
        self.analysis_tabs.setCurrentIndex(index)
        self._sync_analysis_fit_mode_buttons(index)

    def _sync_analysis_fit_mode_buttons(self, index: int) -> None:
        device_selected = index == 0
        for widget in (self.cell_fit_widget, self.mac_analysis_widget):
            widget.device_mode_button.setChecked(device_selected)
            widget.array_mode_button.setChecked(not device_selected)

    def _configure_long_running_safety(self) -> None:
        """Bound GUI-owned memory and recover idle caches under pressure."""

        # QPixmapCache uses KiB. High-resolution charts otherwise accumulate
        # in the global Qt cache even after labels show a newer image.
        QPixmapCache.setCacheLimit(48 * 1024)
        for editor in self.findChildren(QPlainTextEdit):
            if editor.isReadOnly():
                editor.document().setMaximumBlockCount(1500)
        self._last_automatic_cache_cleanup = 0.0
        self._memory_watchdog = QTimer(self)
        self._memory_watchdog.setInterval(15_000)
        self._memory_watchdog.timeout.connect(self._memory_watchdog_tick)
        self._memory_watchdog.start()

    def _has_running_background_job(self) -> bool:
        for widget in self.findChildren(QWidget):
            worker = getattr(widget, "worker", None)
            if isinstance(worker, QThread) and worker.isRunning():
                return True
        for name in ("update_check_thread", "update_download_thread"):
            worker = getattr(self, name, None)
            if isinstance(worker, QThread) and worker.isRunning():
                return True
        return False

    def _memory_watchdog_tick(self) -> None:
        if self._has_running_background_job():
            return
        try:
            import torch
            from .runtime_memory import runtime_memory_snapshot

            device = "cuda" if torch.cuda.is_available() else "cpu"
            snapshot = runtime_memory_snapshot(device)
            ram_total = snapshot["ram_total_bytes"]
            ram_pressure = bool(
                ram_total and snapshot["ram_available_bytes"] / ram_total < 0.12
            )
            gpu_total = snapshot["total_bytes"]
            gpu_pressure = bool(
                gpu_total and snapshot["free_bytes"] / gpu_total < 0.08
            )
            now = monotonic()
            if ((ram_pressure or gpu_pressure)
                    and now - self._last_automatic_cache_cleanup >= 60.0):
                self._last_automatic_cache_cleanup = now
                self._clear_runtime_caches(interactive=False)
        except Exception:
            # A health monitor must never destabilize the GUI it protects.
            return

    def _clear_runtime_caches(self, _checked: bool = False, *, interactive: bool = True) -> None:
        import gc

        from .inference_runtime import clear_lenet_inference_cache

        clear_lenet_inference_cache()
        QPixmapCache.clear()
        gc.collect()
        cuda_released = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_released = True
        except Exception:
            pass
        detail = "推理模型、数据集和图片缓存已释放"
        if cuda_released:
            detail += "，CUDA 未使用缓存已释放"
        detail += "；运行结果文件未删除。"
        self.statusBar().showMessage(detail, 8000)
        if interactive:
            QMessageBox.information(self, "清空缓存", detail)

    def _build_navigation_shell(self) -> QWidget:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        top = QFrame()
        top.setObjectName("topBar")
        top.setFixedHeight(64)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(22, 7, 22, 7)
        top_layout.setSpacing(12)
        mark = QLabel("D")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(mark)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel("DRAM 存算仿真平台")
        title.setObjectName("brandTitle")
        subtitle = QLabel("DRAM-CIM Simulator")
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        top_layout.addLayout(brand)
        top_layout.addStretch()

        group = QButtonGroup(self)
        group.setExclusive(True)
        self.navigation_buttons = []
        for index, (_number, text) in enumerate(self.NAVIGATION):
            button = QPushButton(text, root)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked=False, value=index: self._show_navigation_page(value)
            )
            group.addButton(button)
            self.navigation_buttons.append(button)

        home_button = self.navigation_buttons[0]
        home_button.setText("首页")
        home_button.setProperty("topTool", True)
        top_layout.addWidget(home_button)
        output_button = QPushButton("运行日志")
        output_button.setProperty("topTool", True)
        output_button.clicked.connect(self._open_run_records)
        top_layout.addWidget(output_button)
        self.cache_button = QPushButton("清空缓存")
        self.cache_button.setObjectName("clearCacheButton")
        self.cache_button.setProperty("topTool", True)
        self.cache_button.setToolTip("释放推理模型、数据集、图片以及未使用的 CUDA 缓存；不会删除结果文件")
        self.cache_button.clicked.connect(self._clear_runtime_caches)
        top_layout.addWidget(self.cache_button)
        outer.addWidget(top)

        self.flow_bar = QFrame()
        self.flow_bar.setObjectName("flowBar")
        flow_layout = QHBoxLayout(self.flow_bar)
        flow_layout.setContentsMargins(24, 10, 24, 10)
        flow_layout.setSpacing(8)
        flow_title = QLabel("仿真流程")
        flow_title.setObjectName("flowTitle")
        flow_layout.addWidget(flow_title)
        for index, caption in ((1, "1  误差建模"), (3, "2  模型与数据集"), (5, "3  带误差批量推理"), (6, "4  误差感知再训练")):
            button = self.navigation_buttons[index]
            button.setText(caption)
            button.setProperty("flowStep", True)
            flow_layout.addWidget(button)
        flow_layout.addStretch()
        outer.addWidget(self.flow_bar)
        outer.addWidget(self.shell_stack, 1)

        self.sidebar_summary = QLabel(root)
        self.sidebar_summary.hide()
        return root

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 30, 40, 36)
        layout.setSpacing(18)
        title = QLabel("功能入口")
        title.setObjectName("homeTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "请选择需要进入的模块。仿真模拟包含误差建模、带误差批量推理和误差感知再训练；单数据推理用于快速查看一个样本的结果。"
        )
        subtitle.setProperty("muted", True)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        remembered = QFrame()
        remembered.setObjectName("rememberedConfig")
        remembered_layout = QVBoxLayout(remembered)
        remembered_layout.setContentsMargins(14, 10, 14, 10)
        self.home_experiment = QLabel()
        self.home_experiment.setObjectName("homeHero")
        self.home_experiment.setWordWrap(True)
        self.home_hint = QLabel()
        self.home_hint.setProperty("muted", True)
        self.home_hint.setWordWrap(True)
        remembered_layout.addWidget(self.home_experiment)
        remembered_layout.addWidget(self.home_hint)
        layout.addWidget(remembered)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        entries = (
            (
                "仿真模拟",
                "导入器件或阵列误差数据，完成多种误差模型拟合、带误差批量推理、误差感知再训练与结果导出。",
                "误差建模  →  带误差推理  →  误差感知再训练",
                1,
            ),
            (
                "单数据推理",
                "选择模型、数据集与待推理数据，展示输入图片或文本、预测结果、Top-5 置信度和推理耗时。",
                "统一支持图像分类模型与 GPT 文本模型",
                4,
            ),
        )
        for column, (name, description, metadata, target) in enumerate(entries):
            card = QFrame()
            card.setProperty("entryCard", True)
            card.setMinimumHeight(255)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(24, 22, 24, 20)
            card_layout.setSpacing(12)
            icon = QLabel("◫" if column == 0 else "◎")
            icon.setObjectName("entryIcon")
            name_label = QLabel(name)
            name_label.setObjectName("entryTitle")
            detail = QLabel(description)
            detail.setProperty("muted", True)
            detail.setWordWrap(True)
            meta = QLabel(metadata)
            meta.setObjectName("entryMeta")
            meta.setWordWrap(True)
            button = QPushButton(f"进入{name}")
            button.setProperty("role", "primary")
            button.clicked.connect(
                lambda _checked=False, value=target: self._show_navigation_page(value)
            )
            card_layout.addWidget(icon)
            card_layout.addWidget(name_label)
            card_layout.addWidget(detail)
            card_layout.addWidget(meta)
            card_layout.addStretch()
            card_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignRight)
            grid.addWidget(card, 0, column)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch()
        return page

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _show_navigation_page(self, index: int) -> None:
        if not hasattr(self, "navigation_buttons"): return
        index = max(0, min(index, len(self.navigation_buttons) - 1)); self.navigation_buttons[index].setChecked(True)
        if hasattr(self, "flow_bar"):
            self.flow_bar.setVisible(index not in (0, 4))
            if index in (0, 4) and hasattr(self, "advanced_navigation_panel"):
                self.advanced_navigation_panel.hide()
                self.advanced_flow_button.setChecked(False)
        if index == 0: self.shell_stack.setCurrentIndex(0)
        else:
            self.shell_stack.setCurrentIndex(1)
            if index in (1, 2): self.tabs.setCurrentIndex(0); self.analysis_tabs.setCurrentIndex(index - 1)
            else: self.tabs.setCurrentIndex(index - 2)

    def _refresh_home_summary(self) -> None:
        try:
            spec = self.workflow.model_registry.get(self.workflow.selected_network_model_id()); network = f"{spec.display_name} / {', '.join(spec.datasets)}"
        except Exception: network = "尚未选择网络与数据集"
        selected = self.workflow.selected_error_model_id()
        if selected:
            try:
                model = self.workflow.registry.resolve(selected); error_name = str(model.get("name") or model.get("method") or selected)
            except Exception: error_name = selected
        else: error_name = "尚未完成误差建模"
        self.home_experiment.setText(f"{error_name}  ×  {network}"); self.home_hint.setText("此配置会自动传递到单样本推理、批量推理与误差感知训练。")
        self.sidebar_summary.setText(f"{len(_frontend_model_specs(self.workflow.model_registry))} 个网络配置\n{len(self.workflow.models())} 个可选误差模型")

    def _open_run_records(self) -> None:
        from PySide6.QtCore import QUrl
        target = output_root(); target.mkdir(parents=True, exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _show_help(self) -> None:
        QMessageBox.information(self, "使用帮助", "建议流程：先完成器件误差建模，再选择网络与数据集，随后运行单样本或批量推理，最后执行误差感知训练。")

    def _check_for_updates(self, *, interactive: bool) -> None:
        if self.update_check_thread is not None and self.update_check_thread.isRunning():
            return
        if not CredentialStore.load():
            if interactive:
                box = QMessageBox(self)
                box.setWindowTitle("NeuroCIM更新")
                box.setIcon(QMessageBox.Icon.Information)
                box.setText("尚未登录获批的GitHub账号。")
                box.setInformativeText("请通过安装引导程序登录；未注册用户请先打开访问申请网页。")
                apply_button = box.addButton("打开申请网页", QMessageBox.ButtonRole.ActionRole)
                box.addButton(QMessageBox.StandardButton.Close)
                box.exec()
                if box.clickedButton() is apply_button:
                    from PySide6.QtCore import QUrl
                    QDesktopServices.openUrl(QUrl(ACCESS_PORTAL_URL))
            return
        self.statusBar().showMessage("正在检查NeuroCIM更新……")
        self.update_check_thread = UpdateCheckThread(self.platform_root)
        self.update_check_thread.succeeded.connect(
            lambda update, value=interactive: self._update_check_succeeded(update, value)
        )
        self.update_check_thread.failed.connect(
            lambda message, value=interactive: self._update_check_failed(message, value)
        )
        self.update_check_thread.finished.connect(self._update_check_finished)
        self.update_check_thread.start()

    def _update_check_succeeded(
        self, update: AvailableUpdate | None, interactive: bool
    ) -> None:
        if update is None:
            self.statusBar().showMessage(f"NeuroCIM {__version__} 已是最新版本", 5000)
            if interactive:
                QMessageBox.information(self, "NeuroCIM更新", "当前已经是最新版本。")
            return
        answer = QMessageBox.question(
            self,
            "发现NeuroCIM更新",
            f"发现版本 {update.version}（{update.flavor.upper()}版）。\n\n"
            "是否立即下载并安装？软件将在校验完成后关闭。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._download_update(update)

    def _update_check_failed(self, message: str, interactive: bool) -> None:
        self.statusBar().showMessage("更新检查失败", 5000)
        if interactive:
            QMessageBox.warning(self, "NeuroCIM更新", message)

    def _update_check_finished(self) -> None:
        if self.update_check_thread is not None:
            self.update_check_thread.deleteLater()
        self.update_check_thread = None

    def _download_update(self, update: AvailableUpdate) -> None:
        self.update_progress = QProgressDialog("准备下载更新……", "取消", 0, 100, self)
        self.update_progress.setWindowTitle("NeuroCIM在线更新")
        self.update_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.update_progress.setAutoClose(False)
        self.update_progress.setCancelButton(None)
        self.update_progress.setValue(0)
        self.update_download_thread = UpdateDownloadThread(update)

        def report(name: str, percent: int) -> None:
            if self.update_progress is not None:
                self.update_progress.setLabelText(f"正在下载 {name}")
                self.update_progress.setValue(percent)

        self.update_download_thread.progressed.connect(report)
        self.update_download_thread.succeeded.connect(self._update_download_succeeded)
        self.update_download_thread.failed.connect(self._update_download_failed)
        self.update_download_thread.finished.connect(self._update_download_finished)
        self.update_download_thread.start()

    def _update_download_succeeded(self, installer: str) -> None:
        if self.update_progress is not None:
            self.update_progress.setValue(100)
            self.update_progress.close()
        launch_installer(Path(installer))
        QApplication.quit()

    def _update_download_failed(self, message: str) -> None:
        if self.update_progress is not None:
            self.update_progress.close()
        QMessageBox.critical(self, "NeuroCIM更新失败", message)

    def _update_download_finished(self) -> None:
        if self.update_download_thread is not None:
            self.update_download_thread.deleteLater()
        self.update_download_thread = None
        self.update_progress = None

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
            table.horizontalHeader().setStretchLastSection(
                not bool(table.property("compactColumns"))
            )

    @staticmethod
    def _scrollable(page: QWidget, title: str | None = None, subtitle: str | None = None) -> QScrollArea:
        """Keep every page usable when the Windows taskbar reduces available height."""
        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        page.setObjectName("pageContent")
        if page.layout() is not None:
            page.layout().setContentsMargins(0, 0, 0, 0)
            page.layout().setSpacing(10)
        content = QWidget(); content.setObjectName("pageContent"); content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(26, 20, 26, 24); content_layout.setSpacing(12)
        if title:
            heading = QLabel(title); heading.setObjectName("contentTitle"); content_layout.addWidget(heading)
        if subtitle:
            detail = QLabel(subtitle); detail.setObjectName("contentSubtitle"); detail.setWordWrap(True); content_layout.addWidget(detail)
        independent_columns = bool(page.property("usesIndependentColumnScroll"))
        if independent_columns:
            page.setMinimumSize(0, 0)
            page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        content_layout.addWidget(page, 1 if independent_columns else 0)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            if independent_columns else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.verticalScrollBar().setSingleStep(24)
        scroll.horizontalScrollBar().setSingleStep(24)
        scroll.setWidget(content)
        return scroll


class ModelCatalogWidget(QWidget):
    """Live view of registered model/dataset combinations and resource readiness."""

    def __init__(self, project_root: Path, parent: QWidget | None = None,
                 coordinator: WorkflowCoordinator | None = None):
        super().__init__(parent)
        self.coordinator = coordinator
        self.specs = _frontend_model_specs(builtin_registry(project_root))
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "在这里勾选一个模型与数据集组合。单样本推理、批量推理和后续训练将统一继承该选择；"
            "后续页面不再单独切换网络。缺少 Checkpoint 的组合不能勾选。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.table = QTableWidget()
        columns = ["使用", "模型", "任务", "数据集", "浮点精度", "网络类别", "资源状态"]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(self.specs))
        self.selection_group = QButtonGroup(self)
        self.selection_group.setExclusive(True)
        selected = coordinator.selected_network_model_id() if coordinator else self.specs[0].model_id
        for row, spec in enumerate(self.specs):
            selector = QCheckBox()
            selector.setEnabled(spec.resources_available)
            selector.setChecked(spec.model_id == selected)
            selector.setToolTip("选择后，推理与训练页面统一使用此组合")
            self.selection_group.addButton(selector, row)
            self.table.setCellWidget(row, 0, selector)
            selector.clicked.connect(lambda checked, model_id=spec.model_id:
                                     self._select_model(model_id) if checked else None)
            values = [
                spec.display_name,
                spec.report_task or _task_display_name(spec.task),
                spec.report_dataset or ", ".join(spec.datasets),
                spec.float_metric or "-",
                spec.model_family or "-",
                "可用" if spec.resources_available else "未安装",
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
                    and selected_spec.resources_available):
                coordinator.select_network_model(selected)
            if selected_spec.resources_available:
                self._sync_selection(coordinator.selected_network_model_id())
            else:
                self.selection_status.setText(
                    "尚未安装可用的模型 Checkpoint。请按发布包中的《外部资源配置说明》"
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
        dataset = spec.report_dataset or ", ".join(spec.datasets)
        self.selection_status.setText(
            f"当前统一配置：{spec.display_name}　|　数据集：{dataset}　|　"
            f"任务：{spec.report_task or _task_display_name(spec.task)}　|　"
            f"浮点精度：{spec.float_metric or '-'}"
        )


def run_gui(platform_root: str | Path) -> int:
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("DRAM-CIM NeuroSim Platform")
    configure_windows_font(application)
    apply_application_style(application)
    resolved_platform = Path(platform_root).expanduser().resolve()
    version_smoke_result = os.environ.get("NEUROCIM_RELEASE_VERSION_RESULT")
    if version_smoke_result:
        target = Path(version_smoke_result).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(__version__ + "\n", encoding="utf-8")
        return 0
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
