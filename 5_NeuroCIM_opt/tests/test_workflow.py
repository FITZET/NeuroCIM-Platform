from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt.workflow import ErrorModelRegistry, WorkflowContext
from neurocim_opt.error_model import HardwareErrorModel


def _fitting_files(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "device.csv"
    source.write_text("X,W,output\n", encoding="utf-8")
    summary = tmp_path / "fitting_summary.json"
    summary.write_text(
        json.dumps(
            {
                "dataset": "UOS1",
                "source": str(source),
                "sha256": "a" * 64,
                "acquisition": {"code_bits": 8},
                "column_mapping": {"X": "VD_Digital"},
                "lut_gain": 2.0,
                "interaction_metrics": {"r2": 0.99, "rmse": 0.01},
                "models": {
                    "linear_product": {
                        "equation": "y=a+b*x*w",
                        "coefficients": {"a": 0, "b": 1},
                        "metrics": {"r2": 0.9},
                        "metadata": {},
                    },
                    "full_surface": {
                        "equation": "y=a*v*w+b*v+c*w+d",
                        "coefficients": {},
                        "metrics": {"r2": 0.995},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    codes = np.arange(256, dtype=np.int64)
    lut = codes[:, None] * codes[None, :]
    lut = np.clip(
        np.rint(lut * (1.0 + 0.05 * (codes[:, None] / 255.0))), 0, 65535
    ).astype(np.int64)
    error = HardwareErrorModel(
        lut=lut,
        source_path=str(source),
        source_sha256="a" * 64,
        source_kind="raw_device_csv",
        fitting_method="measured interaction",
    ).save(tmp_path / "source_error_model")
    return error, summary


def test_registry_records_all_fits_and_only_lut_is_selectable(tmp_path: Path) -> None:
    error, summary = _fitting_files(tmp_path)
    registry = ErrorModelRegistry(tmp_path / "error_model_registry.json")
    result = registry.register_fitting_run(
        error_model_path=error, fitting_summary_path=summary
    )
    assert len(result["models"]) == 5
    assert {item["method"] for item in registry.models()} == {
        "linear_product", "full_surface", "empirical_interaction",
        "smooth_interaction_order1", "smooth_interaction_order2",
    }
    selectable = registry.models(selectable_only=True)
    assert len(selectable) == 3
    assert all(item["kind"] == "deployable_error_model" for item in selectable)
    assert all(Path(item["path"]).is_file() for item in selectable)
    assert all(np.all(HardwareErrorModel.load(item["path"]).lut[0, :] == 0) for item in selectable)
    assert all("r2_to_ideal_lut" in item["metrics"] for item in selectable)
    empirical = next(item for item in selectable if item["method"] == "empirical_interaction")
    assert empirical["metrics"]["r2_to_ideal_lut"] < 1.0
    assert "r2_to_empirical_lut" not in empirical["metrics"]
    assert empirical["name"] == "UOS1 原始实测误差表"
    with pytest.raises(ValueError, match="analytical only"):
        registry.resolve(next(item["id"] for item in registry.models() if not item["selectable"]))


def test_register_same_run_is_idempotent(tmp_path: Path) -> None:
    error, summary = _fitting_files(tmp_path)
    registry = ErrorModelRegistry(tmp_path / "registry.json")
    registry.register_fitting_run(error_model_path=error, fitting_summary_path=summary)
    registry.register_fitting_run(error_model_path=error, fitting_summary_path=summary)
    assert len(registry.load()["runs"]) == 1
    assert len(registry.models()) == 5


def test_different_fitting_runs_keep_independent_model_directories(tmp_path: Path) -> None:
    error, summary = _fitting_files(tmp_path)
    registry = ErrorModelRegistry(tmp_path / "registry.json")
    registry.register_fitting_run(error_model_path=error, fitting_summary_path=summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["dataset"] = "UOS2"
    payload["sha256"] = "b" * 64
    summary.write_text(json.dumps(payload), encoding="utf-8")
    registry.register_fitting_run(error_model_path=error, fitting_summary_path=summary)
    selectable = registry.models(selectable_only=True)
    assert len(selectable) == 6
    run_directories = {Path(item["path"]).parent.parent.name for item in selectable}
    assert len(run_directories) == 2


def test_workflow_handoff_and_upstream_change_invalidates_downstream(tmp_path: Path) -> None:
    error, summary = _fitting_files(tmp_path)
    registry = ErrorModelRegistry(tmp_path / "registry.json")
    registry.register_fitting_run(error_model_path=error, fitting_summary_path=summary)
    selected = registry.models(selectable_only=True)[0]
    context = WorkflowContext(tmp_path / "workflow_context.json")
    context.select_error_model(registry.path, selected["id"])
    dataset = tmp_path / "MNIST"
    dataset.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    auxiliary = tmp_path / "quantization.json"
    auxiliary.write_text("{}", encoding="utf-8")
    context.record_inference(
        model_id=selected["id"],
        dataset_name="MNIST",
        dataset_root=dataset,
        network_model="LeNet-5",
        checkpoint_path=checkpoint,
        auxiliary_path=auxiliary,
        device="cpu",
        result={"device_accuracy": 0.98},
    )
    inputs = context.training_inputs()
    assert inputs["network_model"] == "LeNet-5"
    assert inputs["error_model_path"] == str(Path(selected["path"]).resolve())
    context.record_training({"accuracy": 0.99})
    assert context.load()["training"]["completed"] is True
    context.select_error_model(registry.path, selected["id"])
    assert context.load()["inference"] is None
    with pytest.raises(RuntimeError, match="Successful inference"):
        context.training_inputs()


def test_network_selection_is_persistent_and_invalidates_dependent_results(tmp_path: Path) -> None:
    context = WorkflowContext(tmp_path / "workflow_context.json")
    context.select_network_model(
        model_id="lenet5-mnist", task="image_classification", dataset_name="MNIST"
    )
    assert context.selected_network_model()["model_id"] == "lenet5-mnist"
    context.select_network_model(
        model_id="gpt2-small", task="text_generation", dataset_name="prompt_text"
    )
    state = context.load()
    assert state["network_selection"]["dataset_name"] == "prompt_text"
    assert state["inference"] is None and state["training"] is None


def test_inference_must_match_selected_model(tmp_path: Path) -> None:
    error, summary = _fitting_files(tmp_path)
    registry = ErrorModelRegistry(tmp_path / "registry.json")
    registry.register_fitting_run(error_model_path=error, fitting_summary_path=summary)
    selected = registry.models(selectable_only=True)[0]
    context = WorkflowContext(tmp_path / "context.json")
    context.select_error_model(registry.path, selected["id"])
    with pytest.raises(ValueError, match="does not match"):
        context.record_inference(
            model_id="another-model",
            dataset_name="MNIST",
            dataset_root=tmp_path,
            network_model="LeNet-5",
            checkpoint_path=error,
            auxiliary_path=None,
            device="cpu",
            result={},
        )
