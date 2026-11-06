from __future__ import annotations

import os
from pathlib import Path
import sys
import json

import pandas as pd
import pytest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea
from neurocim_opt.gui import (
    BatchInferenceWidget, CellFittingWidget, MacAnalysisWidget, MainWindow, ModelCatalogWidget,
    PpaEvaluationWidget, SingleInferenceWidget, ErrorAwareTrainingWidget, GPT2GenerationWidget,
    GPT2BatchInferenceWidget, WikiText2EvaluationWidget,
    UnifiedBatchInferenceWidget, UnifiedSingleInferenceWidget,
    WorkflowCoordinator, QComboBox,
)
from neurocim_opt.application import CellFitOutcome
from neurocim_opt.error_model import HardwareErrorModel
from neurocim_opt.paths import data_root
from neurocim_opt.model_registry import builtin_registry
from neurocim_opt.workflow import WorkflowContext


def test_gui_can_preview_real_mac_data(tmp_path: Path) -> None:
    source = tmp_path / "mac.csv"
    pd.DataFrame(
        {
            "array_size": ["16x16"] * 4,
            "bit_width": [2] * 4,
            "ideal_mac": [0, 1, 2, 3],
            "measured_output": [0, 1, 2.1, 3.2],
        }
    ).to_csv(source, index=False)
    application = QApplication.instance() or QApplication([])
    widget = MacAnalysisWidget(ROOT)
    widget.path_edit.setText(str(source))
    widget.preview()
    application.processEvents()
    assert widget.table.rowCount() == 4
    assert "预览成功" in widget.log.toPlainText()
    widget.close()


def test_main_window_shows_live_model_catalog() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(ROOT)
    assert len(window.navigation_buttons) == 7
    assert window.cache_button.text() == "清空缓存"
    assert "不会删除结果文件" in window.cache_button.toolTip()
    assert window.shell_stack.currentWidget() is window.home_page
    # The redesigned window wraps the established business-page tab container
    # in a top bar/sidebar shell.  Keep asserting the same page ownership via
    # its compatibility attribute.
    tabs = window.tabs
    assert tabs.count() == 5
    assert all(tabs.tabText(index) != "NeuroSim PPA" for index in range(tabs.count()))
    catalog = tabs.widget(1).findChild(ModelCatalogWidget)
    assert catalog is not None
    assert catalog.table.rowCount() == len(builtin_registry(ROOT).list())
    rows = {
        catalog.table.item(row, 1).text(): catalog.table.item(row, 7).text()
        for row in range(catalog.table.rowCount())
    }
    assert set(rows) == {item.model_id for item in builtin_registry(ROOT).list()}
    assert rows["squeezenet1_1-cifar10"] == rows["mobilenetv2-cifar10"]
    single = tabs.widget(2).findChild(UnifiedSingleInferenceWidget)
    batch = tabs.widget(3).findChild(UnifiedBatchInferenceWidget)
    assert single is not None and batch is not None
    gpt2 = single.findChild(GPT2GenerationWidget)
    assert gpt2 is not None
    assert Path(gpt2.checkpoint_edit.text()).is_file()
    assert Path(gpt2.tokenizer_edit.text()).is_dir()
    assert all(tabs.tabText(index) != "GPT-2 文本生成" for index in range(tabs.count()))
    window._show_navigation_page(4)
    assert window.shell_stack.currentWidget() is tabs
    assert tabs.currentIndex() == 2
    window._show_navigation_page(1)
    assert tabs.currentIndex() == 0
    assert window.analysis_tabs.currentIndex() == 0
    application.processEvents()
    window.close()


def test_main_pages_use_independent_2_to_3_column_scrolls() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(ROOT)
    window.resize(760, 430)
    window.show()
    window._show_navigation_page(5)
    application.processEvents()
    page_scroll = window.tabs.widget(3)
    assert isinstance(page_scroll, QScrollArea)
    left_scroll = page_scroll.findChild(QScrollArea, "leftColumnScroll")
    right_scroll = page_scroll.findChild(QScrollArea, "rightColumnScroll")
    assert left_scroll is not None and right_scroll is not None
    assert left_scroll.verticalScrollBar() is not right_scroll.verticalScrollBar()
    assert left_scroll.verticalScrollBar().maximum() > 0
    assert page_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert 0.58 < left_scroll.width() / right_scroll.width() < 0.75
    window.close()


def test_combo_boxes_ignore_wheel_and_open_from_the_full_control() -> None:
    application = QApplication.instance() or QApplication([])
    combo = QComboBox()
    combo.addItems(["one", "two", "three"])
    combo.setCurrentIndex(1)
    combo.resize(220, 36)
    combo.show()
    application.processEvents()

    wheel = QWheelEvent(
        QPointF(combo.rect().center()), QPointF(combo.mapToGlobal(combo.rect().center())),
        QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False,
    )
    QApplication.sendEvent(combo, wheel)
    assert combo.currentIndex() == 1

    QTest.mouseClick(combo, Qt.MouseButton.LeftButton, pos=QPoint(8, combo.height() // 2))
    application.processEvents()
    assert combo.view().isVisible()
    combo.hidePopup()

    combo.setEditable(True)
    QTest.mouseClick(combo.lineEdit(), Qt.MouseButton.LeftButton, pos=QPoint(8, combo.height() // 2))
    application.processEvents()
    assert combo.view().isVisible()
    combo.close()


def test_single_inference_progress_is_directly_below_run_controls() -> None:
    application = QApplication.instance() or QApplication([])
    widget = SingleInferenceWidget(ROOT.parent)
    layout = widget.run_button.parentWidget().layout()
    controls_index = next(
        index for index in range(layout.count())
        if layout.itemAt(index).layout() is not None
        and layout.itemAt(index).layout().indexOf(widget.run_button) >= 0
    )
    assert layout.indexOf(widget.progress) == controls_index + 1
    assert layout.indexOf(widget.progress) < layout.indexOf(widget.summary)
    widget.close()


def test_requested_progress_and_result_panel_layouts() -> None:
    application = QApplication.instance() or QApplication([])
    cell = CellFittingWidget(ROOT)
    training = ErrorAwareTrainingWidget(ROOT.parent)
    single = SingleInferenceWidget(ROOT.parent)
    batch = BatchInferenceWidget(ROOT.parent)

    for widget, button in ((cell, cell.fit_button), (training, training.run_button)):
        layout = button.parentWidget().layout()
        controls_index = next(
            index for index in range(layout.count())
            if layout.itemAt(index).layout() is not None
            and layout.itemAt(index).layout().indexOf(button) >= 0
        )
        assert layout.indexOf(widget.progress) == controls_index + 1
        assert layout.indexOf(widget.progress) < layout.indexOf(widget.summary)

    assert single.log.maximumHeight() == 100
    assert batch.chart_scroll.maximumHeight() == 365
    assert batch.chart_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    cell.close(); training.close(); single.close(); batch.close()


def test_fit_mode_highlight_tracks_the_visible_page() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow(ROOT)
    window._set_analysis_fit_mode(1)
    application.processEvents()
    assert window.analysis_tabs.currentIndex() == 1
    assert window.mac_analysis_widget.array_mode_button.isChecked()
    assert not window.mac_analysis_widget.device_mode_button.isChecked()
    window._set_analysis_fit_mode(0)
    application.processEvents()
    assert window.analysis_tabs.currentIndex() == 0
    assert window.cell_fit_widget.device_mode_button.isChecked()
    assert not window.cell_fit_widget.array_mode_button.isChecked()
    window.close()


def test_catalog_selection_drives_single_and_batch_task_views(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    coordinator = WorkflowCoordinator(ROOT)
    coordinator.context = WorkflowContext(tmp_path / "workflow_context.json")
    single = UnifiedSingleInferenceWidget(ROOT.parent, coordinator=coordinator)
    batch = UnifiedBatchInferenceWidget(ROOT.parent, coordinator=coordinator)
    coordinator.select_network_model("gpt2-small")
    application.processEvents()
    assert single.stack.currentWidget() is single.gpt_widget
    assert batch.stack.currentWidget() is batch.gpt_widget
    coordinator.select_network_model("resnet20-cifar10")
    application.processEvents()
    assert single.stack.currentWidget() is single.image_widget
    assert batch.stack.currentWidget() is batch.image_widget
    assert single.image_widget.model_combo.currentData() == "resnet20-cifar10"
    assert batch.image_widget.model_combo.currentData() == "resnet20-cifar10"
    assert not single.image_widget.model_combo.isEnabled()
    assert not batch.image_widget.model_combo.isEnabled()
    single.close(); batch.close()


def test_gpt2_large_selection_updates_generation_batch_and_wikitext_resources() -> None:
    application = QApplication.instance() or QApplication([])
    coordinator = WorkflowCoordinator(ROOT)
    large = coordinator.model_registry.get("gpt2-large")
    single = UnifiedSingleInferenceWidget(ROOT.parent, coordinator=coordinator)
    batch = UnifiedBatchInferenceWidget(ROOT.parent, coordinator=coordinator)
    single.apply_model_selection("gpt2-large")
    batch.apply_model_selection("gpt2-large")
    application.processEvents()

    assert single.stack.currentWidget() is single.gpt_widget
    assert single.gpt_widget._request().model_variant == "gpt2-large"
    assert Path(single.gpt_widget.checkpoint_edit.text()) == large.checkpoint
    assert Path(single.gpt_widget.tokenizer_edit.text()) == large.checkpoint.parent
    assert batch.stack.currentWidget() is batch.gpt_widget
    assert all(request.model_variant == "gpt2-large" for request in batch.gpt_widget._requests())
    evaluation = batch.findChild(WikiText2EvaluationWidget)
    assert evaluation is not None and not evaluation.isHidden()
    evaluation_request = evaluation._request()
    assert evaluation_request.model_variant == "gpt2-large"
    assert evaluation_request.batch_size == 1
    assert evaluation_request.dataset_path == large.dataset_root
    single.close(); batch.close()


def test_catalog_never_enables_partial_gpt2_large_download() -> None:
    application = QApplication.instance() or QApplication([])
    coordinator = WorkflowCoordinator(ROOT)
    widget = ModelCatalogWidget(ROOT.parent, coordinator=coordinator)
    large = coordinator.model_registry.get("gpt2-large")
    row = next(index for index, spec in enumerate(widget.specs) if spec.model_id == "gpt2-large")
    selector = widget.table.cellWidget(row, 0)
    assert selector.isEnabled() == large.resources_available
    partial = large.checkpoint.with_name(large.checkpoint.name + ".partial")
    if partial.exists() and not large.checkpoint.exists():
        assert not selector.isEnabled()
    widget.close()


def test_real_single_sample_inference_offscreen() -> None:
    application = QApplication.instance() or QApplication([])
    widget = SingleInferenceWidget(ROOT.parent)
    widget.model_combo.setCurrentIndex(widget.model_combo.findData("lenet5-mnist"))
    dataset_root = Path(widget.dataset_edit.text())
    if not (dataset_root / "MNIST").is_dir():
        pytest.skip("select or install an MNIST dataset root")
    if not widget.run_button.isEnabled():
        pytest.skip(
            "single-sample inference requires fitted error-model resources: "
            + widget.resource_status.text()
        )
    widget.resize(900, 560)
    widget.show()
    widget.sample_index.setValue(0)
    widget.start_inference()
    assert widget.worker is not None
    assert widget.worker.wait(30_000)
    application.processEvents()

    outcome = widget.last_outcome
    assert outcome is not None, widget.log.toPlainText()
    assert outcome.sample_index == 0
    assert outcome.true_label == 7
    modes = [item.mode for item in outcome.predictions]
    assert modes[:2] == ["float", "ideal_differential"]
    assert modes[2].startswith("device_")
    assert all(0.0 <= item.confidence <= 1.0 for item in outcome.predictions)
    assert all(item.elapsed_ms > 0.0 for item in outcome.predictions)
    assert outcome.input_shape in ((1, 28, 28), (1, 32, 32))
    assert len(outcome.input_image) > 0
    assert all(len(item.top5) == 5 for item in outcome.predictions)
    assert widget.result_table.rowCount() == 3
    assert widget.top5_table.rowCount() == 15
    assert widget.input_image.pixmap() is not None
    assert "真实标签" in widget.final_prediction.text()

    evidence = ROOT / "outputs" / "acceptance" / "gui_inference"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "sample_0_prediction.json").write_text(
        json.dumps(outcome.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert widget.grab().save(str(evidence / "single_sample_inference.png"))
    widget.close()


def test_single_inference_model_selection_updates_resources() -> None:
    application = QApplication.instance() or QApplication([])
    widget = SingleInferenceWidget(ROOT.parent)
    assert widget.model_combo.findData("gpt2-small") == -1
    widget.model_combo.setCurrentIndex(widget.model_combo.findData("bnn-mnist"))
    assert widget.dataset_name.text() == "MNIST"
    assert widget.auxiliary_edit.isEnabled() is False
    assert widget.checkpoint_edit.text().endswith("bnn_mnist.pt")
    widget.model_combo.setCurrentIndex(widget.model_combo.findData("resnet20-cifar10"))
    assert widget.dataset_name.text() == "CIFAR-10"
    assert widget.auxiliary_edit.isEnabled() is True
    assert widget.auxiliary_edit.text().endswith("input_scales.json")
    widget.model_combo.setCurrentIndex(widget.model_combo.findData("tinytransformer-cifar10"))
    assert widget.dataset_name.text() == "CIFAR-10"
    assert "TinyTransformer" in widget.checkpoint_edit.text()
    assert widget.checkpoint_edit.text().endswith("cct_7_3x1_32_cifar10.pth")
    assert widget.auxiliary_edit.text().endswith("input_scales.json")
    expected = {
        "resnet8-mnist": ("MNIST", "resnet8_mnist.pt", "resnet8-mnist_input_scales.json"),
        "mobilenetv2-cifar10": (
            "CIFAR-10", "cifar10_mobilenetv2_x1_0-fe6a5b48.pt",
            "mobilenetv2-cifar10_input_scales.json",
        ),
        "vgg11-bn-cifar10": (
            "CIFAR-10", "cifar10_vgg11_bn-eaeebf42.pt",
            "vgg11-bn-cifar10_input_scales.json",
        ),
        "squeezenet1_1-cifar10": (
            "CIFAR-10", "squeezenet1_1_cifar10.pt",
            "squeezenet1_1-cifar10_input_scales.json",
        ),
    }
    for model_id, (dataset, checkpoint_name, auxiliary_name) in expected.items():
        index = widget.model_combo.findData(model_id)
        assert index >= 0 and widget.model_combo.model().item(index).isEnabled()
        widget.model_combo.setCurrentIndex(index)
        assert widget.dataset_name.text() == dataset
        assert Path(widget.checkpoint_edit.text()).name == checkpoint_name
        assert Path(widget.auxiliary_edit.text()).name == auxiliary_name
        assert widget.auxiliary_edit.isEnabled() is True
        assert widget.auxiliary_label.text() == "量化/输入尺度参数"
    assert "资源状态" in widget.resource_status.text()
    if not (Path(widget.dataset_edit.text()) / "cifar-10-batches-py/test_batch").is_file():
        assert "未就绪" in widget.resource_status.text()
        assert widget.run_button.isEnabled() is False
    widget.close()


def test_single_inference_discards_stale_portable_resource_paths(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    widget = SingleInferenceWidget(ROOT.parent)
    model_id = "lenet5-mnist"
    defaults = widget._model_defaults(model_id)
    stale = tmp_path / "old-portable-location"
    settings = {
        "dataset_root": stale / "data" / "mnist",
        "checkpoint": stale / "3_models" / "LeNet_5" / "chekpoint" / "lenet5_float.pt",
        "auxiliary": stale / "3_models" / "LeNet_5" / "chekpoint" / "quantization.json",
    }
    try:
        for name, path in settings.items():
            widget.settings.setValue(f"inference/{model_id}/{name}", str(path))
        values = widget._model_configuration(model_id)
        assert values["root"] == defaults["root"]
        assert values["checkpoint"] == defaults["checkpoint"]
        assert values["auxiliary"] == defaults["auxiliary"]
        for name in settings:
            assert widget.settings.value(f"inference/{model_id}/{name}") is None
    finally:
        for name in settings:
            widget.settings.remove(f"inference/{model_id}/{name}")
        widget.close()


def test_real_grouped_batch_inference_chart_offscreen(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    widget = BatchInferenceWidget(ROOT.parent)
    if not (Path(widget.dataset_edit.text()) / "MNIST/raw/t10k-images-idx3-ubyte").is_file():
        pytest.skip("select or install an MNIST dataset root")
    required = [widget.checkpoint_edit.text(), widget.quantization_edit.text(), widget.error_model_edit.text()]
    if not all(Path(path).is_file() for path in required):
        pytest.skip("batch inference model resources are not installed")
    widget.sample_count.setValue(23)
    widget.group_size.setValue(10)
    widget.batch_size.setValue(8)
    widget.output_edit.setText(str(tmp_path / "batch_result.json"))
    widget.start_inference()
    assert widget.worker is not None
    assert widget.worker.wait(30_000)
    application.processEvents()

    outcome = widget.last_outcome
    assert outcome is not None, widget.log.toPlainText()
    assert [group.sample_count for group in outcome.groups] == [10, 10, 3]
    assert widget.result_table.rowCount() == 3
    assert (tmp_path / "batch_result.json").is_file()
    assert (tmp_path / "batch_result_accuracy.png").is_file()
    assert widget.chart_label.pixmap() is not None
    assert widget.chart_label.pixmap().width() <= 640
    assert widget.run_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    widget.close()


def test_error_selection_syncs_inference_pages_and_training_inherits_step_two(tmp_path: Path) -> None:
    source = tmp_path / "device.csv"; source.write_text("X,W,output\n", encoding="utf-8")
    codes = np.arange(256, dtype=np.int64)
    lut = codes[:, None] * codes[None, :]
    error = HardwareErrorModel(
        lut=lut, source_path=str(source), source_sha256="c" * 64,
        source_kind="raw_device_csv", fitting_method="measured interaction",
    ).save(tmp_path / "source_model")
    summary = tmp_path / "fitting_summary.json"
    summary.write_text(json.dumps({
        "dataset": "sync_test", "source": str(source), "sha256": "c" * 64,
        "lut_gain": 1.0, "interaction_metrics": {}, "models": {},
    }), encoding="utf-8")
    coordinator = WorkflowCoordinator(tmp_path / "platform")
    coordinator.register_fit(CellFitOutcome(error, summary, (), {}, (), {}))
    application = QApplication.instance() or QApplication([])
    single = SingleInferenceWidget(ROOT.parent, coordinator=coordinator, require_workflow=True)
    batch = BatchInferenceWidget(ROOT.parent, coordinator=coordinator, require_workflow=True)
    training = ErrorAwareTrainingWidget(ROOT.parent, coordinator=coordinator, require_workflow=True)
    order1 = next(item for item in coordinator.models() if item["method"] == "smooth_interaction_order1")
    single.error_model_combo.setCurrentIndex(single.error_model_combo.findData(order1["id"]))
    application.processEvents()
    assert batch.error_model_combo.currentData() == order1["id"]

    dataset = tmp_path / "dataset"; dataset.mkdir()
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"weights")
    auxiliary = tmp_path / "scales.json"; auxiliary.write_text("{}", encoding="utf-8")
    coordinator.record_inference(
        model_id=order1["id"], dataset_name="CIFAR-10", dataset_root=dataset,
        network_model="resnet20-cifar10", checkpoint_path=checkpoint,
        auxiliary_path=auxiliary, device="cpu", result={"accuracy": 0.5},
    )
    application.processEvents()
    assert training.model_edit.text() == "resnet20-cifar10"
    assert training.dataset_name.text() == "CIFAR-10"
    assert training.error_model_combo.currentData() == order1["id"]
    order2 = next(item for item in coordinator.models() if item["method"] == "smooth_interaction_order2")
    training.error_model_combo.setCurrentIndex(training.error_model_combo.findData(order2["id"]))
    assert coordinator.context.training_inputs()["error_model_id"] == order1["id"]
    single.close(); batch.close(); training.close()


def test_real_lenet_ppa_offscreen() -> None:
    checkpoint = ROOT / "outputs" / "eat" / "01_NMOS" / "eat_checkpoint.pt"
    error_model = ROOT / "outputs" / "error_models" / "01_NMOS" / "error_model.json"
    binary = ROOT / "outputs" / "neurosim" / "bin" / "neurosim.exe"
    if not all(path.is_file() for path in (checkpoint, error_model, binary)):
        pytest.skip("formal LeNet/NeuroSim resources are not installed")
    application = QApplication.instance() or QApplication([])
    widget = PpaEvaluationWidget(ROOT.parent)
    widget.resize(1000, 720)
    widget.show()
    evidence = ROOT / "outputs" / "acceptance" / "gui_ppa"
    widget.start_evaluation(evidence)
    assert widget.worker is not None
    assert widget.worker.wait(30_000)
    application.processEvents()

    outcome = widget.last_outcome
    assert outcome is not None, widget.log.toPlainText()
    assert outcome.model_id == "lenet5-mnist"
    for metric in (
        "chip_area_um2", "read_latency_ns", "read_dynamic_energy_pj",
        "energy_efficiency_tops_per_w", "throughput_fps",
    ):
        assert outcome.metrics[metric] > 0
    assert Path(outcome.result_path).is_file()
    assert Path(outcome.log_path).is_file()
    assert (evidence / "traces" / "trace_manifest.json").is_file()
    assert widget.result_table.rowCount() == 5
    assert widget.grab().save(str(evidence / "neurosim_ppa.png"))
    widget.close()


def test_ppa_worker_surfaces_missing_resource_error(tmp_path: Path) -> None:
    application = QApplication.instance() or QApplication([])
    widget = PpaEvaluationWidget(ROOT.parent)
    widget.checkpoint_edit.setText(str(tmp_path / "missing.pt"))
    widget.error_model_edit.setText(str(tmp_path / "missing.json"))
    widget.start_evaluation(tmp_path / "out")
    assert widget.worker is not None
    assert widget.worker.wait(10_000)
    application.processEvents()
    assert widget.last_outcome is None
    assert "Required NeuroSim resources are missing" in widget.log.toPlainText()
    assert "PPA 评估失败" in widget.summary.text()
    assert widget.run_button.isEnabled()
    widget.close()
