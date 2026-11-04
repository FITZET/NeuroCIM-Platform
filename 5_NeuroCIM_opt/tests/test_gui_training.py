from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import json

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from neurocim_opt.gui import ErrorAwareTrainingWidget
from neurocim_opt.training_service import (
    ErrorAwareTrainingService, TrainingCancelled, TrainingOutcome, TrainingRequest,
)


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_training_request_rejects_illegal_parameters(tmp_path: Path) -> None:
    service = ErrorAwareTrainingService()
    request = TrainingRequest(tmp_path, tmp_path / "missing.json", tmp_path / "out", error_strength=1.1)
    with pytest.raises(ValueError, match="lambda"):
        service.validate(request)
    request = TrainingRequest(tmp_path, tmp_path / "missing.json", tmp_path / "out", batch_size=0)
    with pytest.raises(ValueError, match="batch size"):
        service.validate(request)


def test_training_reports_missing_resources_and_restores_ui(tmp_path: Path) -> None:
    app = _application()
    widget = ErrorAwareTrainingWidget(tmp_path)
    widget.error_model_edit.setText(str(tmp_path / "missing.json"))
    widget.dataset_edit.setText(str(tmp_path / "missing-data"))
    widget.start_training(tmp_path / "out")
    app.processEvents()
    assert "Required training resources are missing" in widget.log.toPlainText()
    assert "训练配置错误" in widget.summary.text()
    assert widget.run_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    widget.close()


class _SlowService:
    def validate(self, _request):
        return None

    def run(self, _request, *, progress, cancel):
        for index in range(200):
            if cancel():
                raise TrainingCancelled("Training cancelled by user")
            progress(index // 2, f"batch {index}", None)
            time.sleep(0.005)
        raise AssertionError("test failed to cancel worker")


def test_cancel_is_cooperative_and_ui_recovers(tmp_path: Path) -> None:
    app = _application()
    widget = ErrorAwareTrainingWidget(ROOT.parent)
    widget.service = _SlowService()
    widget.error_model_edit.setText(str(tmp_path / "placeholder.json"))
    widget.dataset_edit.setText(str(tmp_path))
    widget.start_training(tmp_path / "out")
    time.sleep(0.03)
    widget.cancel_training()
    assert widget.worker is not None and widget.worker.wait(5_000)
    app.processEvents()
    assert widget.last_outcome is None
    assert "cancelled" in widget.summary.text().lower()
    assert widget.run_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    widget.close()


class _SuccessService:
    def validate(self, _request):
        return None

    def run(self, request, *, progress, cancel):
        assert not cancel()
        progress(50, "Epoch 1 complete", {"train_loss": 0.1, "train_accuracy": 0.9})
        evaluations = {
            stage: {
                mode: {"accuracy": accuracy, "loss": 0.1, "samples": 10_000}
                for mode, accuracy in (("float", .99), ("ideal_differential", .98),
                                       ("device_differential", .97))
            } for stage in ("before", "after")
        }
        return TrainingOutcome(
            str(request.output_dir / "eat_checkpoint.pt"), str(request.output_dir / "history.json"),
            str(request.output_dir / "training_result.json"), str(request.output_dir), "cpu", 1.0,
            ({"epoch": 1},), evaluations,
            {"l2_delta": 0.25, "max_abs_delta": 0.1, "changed_parameter_tensors": 10},
        )


def test_success_populates_metrics_and_restores_ui(tmp_path: Path) -> None:
    app = _application()
    widget = ErrorAwareTrainingWidget(ROOT.parent)
    widget.service = _SuccessService()
    widget.error_model_edit.setText(str(tmp_path / "placeholder.json"))
    widget.dataset_edit.setText(str(tmp_path))
    widget.start_training(tmp_path / "out")
    assert widget.worker is not None and widget.worker.wait(5_000)
    app.processEvents()
    assert widget.last_outcome is not None
    assert widget.metrics_table.rowCount() == 6
    assert "权重 L2 变化量=0.25" in widget.summary.text()
    assert widget.run_button.isEnabled()
    assert widget.export_button.isEnabled()
    widget.close()


@pytest.mark.skipif(
    os.environ.get("NEUROCIM_RUN_FULL_TRAINING") != "1",
    reason="set NEUROCIM_RUN_FULL_TRAINING=1 for the complete one-epoch GUI acceptance",
)
def test_real_full_epoch_training_from_gui() -> None:
    dataset = Path(os.environ["NEUROCIM_MNIST_ROOT"])
    evidence = ROOT / "outputs" / "acceptance" / "gui_training"
    app = _application()
    widget = ErrorAwareTrainingWidget(ROOT.parent)
    widget.resize(1100, 800)
    widget.show()
    widget.dataset_edit.setText(str(dataset))
    widget.error_model_edit.setText(str(
        ROOT / "outputs" / "error_models" / "01_NMOS" / "error_model.json"
    ))
    widget.lambda_spin.setValue(1.0)
    widget.epoch_spin.setValue(1)
    widget.batch_spin.setValue(128)
    widget.device_combo.setCurrentText("cuda")
    widget.start_training(evidence)
    assert widget.worker is not None
    assert widget.worker.wait(1_800_000), "full training exceeded 30 minutes"
    app.processEvents()
    outcome = widget.last_outcome
    assert outcome is not None, widget.log.toPlainText()
    assert len(outcome.history) == 1
    assert outcome.history[0]["train_samples"] == 55_000
    assert outcome.history[0]["validation_samples"] == 5_000
    assert outcome.weight_update["l2_delta"] > 0
    assert outcome.weight_update["changed_parameter_tensors"] > 0
    for stage in ("before", "after"):
        for mode in ("float", "ideal_differential", "device_differential"):
            assert outcome.evaluations[stage][mode]["samples"] == 10_000
    assert Path(outcome.checkpoint_path).is_file()
    assert Path(outcome.history_path).is_file()
    assert Path(outcome.result_path).is_file()
    (evidence / "gui_log.txt").write_text(widget.log.toPlainText(), encoding="utf-8")
    (evidence / "gui_acceptance.json").write_text(json.dumps({
        "workflow": "ErrorAwareTrainingWidget.start_training",
        "status": "completed",
        "summary": widget.summary.text(),
        "metrics_rows": widget.metrics_table.rowCount(),
        "outcome": outcome.as_dict(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    assert widget.grab().save(str(evidence / "error_aware_training.png"))
    widget.close()
