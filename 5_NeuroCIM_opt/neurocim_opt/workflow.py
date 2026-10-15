"""Persistent error-model registry and cross-step workflow context.

The fitting backend produces several analytical fits but only the interaction
LUT is a deployable multiplication model.  This module records both kinds and
marks deployable entries explicitly so later workflow steps cannot mistake a
diagnostic regression for an inference LUT.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import numpy as np


REGISTRY_SCHEMA = "1.0"
WORKFLOW_SCHEMA = "1.0"
_LOCK = RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _read_object(path: Path, purpose: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {purpose} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{purpose} must contain a JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _lut_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = reference.astype(float).ravel()
    estimate = predicted.astype(float).ravel()
    residual = actual - estimate
    centered = actual - actual.mean()
    ss_total = float(centered @ centered)
    ss_error = float(residual @ residual)
    full_scale = float(actual.max() - actual.min())
    maximum = float(np.max(np.abs(residual)))
    r2 = (1.0 if ss_error == 0 else 0.0) if ss_total == 0 else 1.0 - ss_error / ss_total
    return {
        "r2_to_empirical_lut": r2,
        "rmse_codes_to_empirical_lut": float(np.sqrt(np.mean(residual**2))),
        "mae_codes_to_empirical_lut": float(np.mean(np.abs(residual))),
        "max_abs_codes_to_empirical_lut": maximum,
        "nl_percent_fs_to_empirical_lut": 0.0 if full_scale == 0 else 100.0 * maximum / full_scale,
    }


def _ideal_lut_metrics(lut: np.ndarray) -> dict[str, float]:
    """Measure a deployable LUT against the ideal unsigned 8-bit product."""
    codes = np.arange(256, dtype=np.int64)
    ideal = codes[:, None] * codes[None, :]
    actual = ideal.astype(float).ravel()
    estimate = np.asarray(lut, dtype=float).ravel()
    residual = estimate - actual
    centered = actual - actual.mean()
    ss_total = float(centered @ centered)
    ss_error = float(residual @ residual)
    full_scale = float(actual.max() - actual.min())
    maximum = float(np.max(np.abs(residual)))
    return {
        "r2_to_ideal_lut": 1.0 - ss_error / ss_total,
        "rmse_codes_to_ideal_lut": float(np.sqrt(np.mean(residual**2))),
        "mae_codes_to_ideal_lut": float(np.mean(np.abs(residual))),
        "max_abs_codes_to_ideal_lut": maximum,
        "nl_percent_fs_to_ideal_lut": 0.0 if full_scale == 0 else 100.0 * maximum / full_scale,
    }


DEPLOYABLE_LABELS = {
    "empirical_interaction": "原始实测误差表",
    "smooth_interaction_order1": "低阶平滑误差模型",
    "smooth_interaction_order2": "高阶平滑误差模型",
}


def _smooth_interaction_lut(empirical: np.ndarray, order: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit H(X,W)=XW*g(X,W), which guarantees zero code axes."""
    if order not in {1, 2}:
        raise ValueError("interaction smoothing order must be 1 or 2")
    codes = np.arange(256, dtype=float)
    x, w = np.meshgrid(codes / 255.0, codes / 255.0)
    product = (x * 255.0) * (w * 255.0)
    basis = [np.ones_like(x), x, w]
    if order == 2:
        basis.extend((x**2, x * w, w**2))
    design = np.column_stack([(product * item).ravel() for item in basis])
    active = product.ravel() > 0
    coefficients, *_ = np.linalg.lstsq(
        design[active], empirical.astype(float).ravel()[active], rcond=None
    )
    fitted = (design @ coefficients).reshape(256, 256)
    fitted[0, :] = 0.0
    fitted[:, 0] = 0.0
    lut = np.clip(np.rint(fitted), 0, 65535).astype(np.int64)
    details = _lut_metrics(empirical, lut)
    details["polynomial_order"] = order
    details["coefficient_count"] = len(coefficients)
    return lut, {**details, "coefficients": [float(value) for value in coefficients]}


def _export_deployable_variants(
    error_path: Path, summary: Mapping[str, Any], target: Path
) -> list[dict[str, Any]]:
    """Write three independent, inference-compatible HardwareErrorModel bundles."""
    from .error_model import HardwareErrorModel

    empirical = HardwareErrorModel.load(error_path)
    target.mkdir(parents=True, exist_ok=True)
    variants: list[tuple[str, str, np.ndarray, dict[str, Any]]] = [
        (
            "empirical_interaction",
            DEPLOYABLE_LABELS["empirical_interaction"],
            empirical.lut,
            {
                **_ideal_lut_metrics(empirical.lut),
                "source_interaction_metrics": summary.get("interaction_metrics", {}),
            },
        )
    ]
    for order, slug, label in (
        (1, "smooth_interaction_order1", DEPLOYABLE_LABELS["smooth_interaction_order1"]),
        (2, "smooth_interaction_order2", DEPLOYABLE_LABELS["smooth_interaction_order2"]),
    ):
        lut, fit_metrics = _smooth_interaction_lut(empirical.lut, order)
        variants.append((slug, label, lut, {**_ideal_lut_metrics(lut), **fit_metrics}))

    exported = []
    for slug, label, lut, metrics in variants:
        model = HardwareErrorModel(
            lut=lut,
            source_path=empirical.source_path,
            source_sha256=empirical.source_sha256,
            source_kind="model_bundle",
            fitting_method=slug,
            gain=empirical.gain,
            metadata={
                "variant": slug,
                "label": label,
                "metrics": metrics,
                "physical_constraint": "H(X,W)=X*W*g(X,W); zero X/W axes",
                "parent_error_model": str(error_path),
            },
        )
        manifest = model.save(target / slug)
        exported.append(
            {"slug": slug, "label": label, "path": manifest, "metrics": metrics}
        )
    return exported


class ErrorModelRegistry:
    """Manifest of all fits from one or more fitting runs."""

    def __init__(self, path: str | Path):
        self.path = _path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": REGISTRY_SCHEMA,
                "updated_utc": None,
                "runs": [],
                "models": [],
            }
        payload = _read_object(self.path, "error-model registry")
        if payload.get("schema_version") != REGISTRY_SCHEMA:
            raise ValueError(
                f"Unsupported error-model registry schema: {payload.get('schema_version')}"
            )
        if not isinstance(payload.get("runs"), list) or not isinstance(payload.get("models"), list):
            raise ValueError("Error-model registry requires list fields: runs and models")
        return payload

    def register_fitting_run(
        self,
        *,
        error_model_path: str | Path,
        fitting_summary_path: str | Path,
    ) -> dict[str, Any]:
        """Register every analytical fit and the deployable interaction LUT."""
        error_path = _path(error_model_path)
        summary_path = _path(fitting_summary_path)
        error = _read_object(error_path, "error-model manifest")
        summary = _read_object(summary_path, "fitting summary")
        source = summary.get("source") or error.get("source", {}).get("path")
        source_hash = str(summary.get("sha256") or error.get("source", {}).get("sha256") or "")
        dataset = str(summary.get("dataset") or Path(str(source or "dataset")).stem)
        run_id = f"{dataset}:{source_hash[:12] or 'unknown'}"
        timestamp = str(error.get("created_utc") or _now())
        common = {
            "run_id": run_id,
            "dataset": dataset,
            "source": {"path": str(source or ""), "sha256": source_hash},
            "created_utc": timestamp,
        }
        entries: list[dict[str, Any]] = []
        fits = summary.get("models", {})
        if not isinstance(fits, dict):
            raise ValueError("Fitting summary field 'models' must be an object")
        for method, fit in fits.items():
            if not isinstance(fit, dict):
                continue
            entries.append(
                {
                    **common,
                    "id": f"{run_id}:fit:{method}",
                    "name": str(method),
                    "kind": "analytical_fit",
                    "method": str(method),
                    "selectable": False,
                    "path": str(summary_path),
                    "equation": fit.get("equation"),
                    "coefficients": fit.get("coefficients", {}),
                    "metrics": fit.get("metrics", {}),
                    "metadata": fit.get("metadata", {}),
                }
            )
        safe_dataset = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in dataset
        ).strip("_") or "dataset"
        run_directory = f"{safe_dataset}_{source_hash[:12] or 'unknown'}"
        variants = _export_deployable_variants(
            error_path, summary, self.path.parent / "error_models" / run_directory
        )
        for variant in variants:
            entries.append(
                {
                    **common,
                    "id": f"{run_id}:error:{variant['slug']}",
                    "name": f"{dataset} {variant['label']}",
                    "kind": "deployable_error_model",
                    "method": variant["slug"],
                    "selectable": True,
                    "path": str(variant["path"]),
                    "metrics": variant["metrics"],
                    "metadata": {
                        "lut_gain": summary.get("lut_gain"),
                        "acquisition": summary.get("acquisition", {}),
                        "column_mapping": summary.get("column_mapping", {}),
                        "parent_method": error.get("fitting", {}).get("method"),
                    },
                }
            )
        run = {
            "id": run_id,
            "dataset": dataset,
            "source": common["source"],
            "created_utc": timestamp,
            "fitting_summary_path": str(summary_path),
            "error_model_path": str(error_path),
            "model_ids": [entry["id"] for entry in entries],
        }
        with _LOCK:
            payload = self.load()
            payload["runs"] = [item for item in payload["runs"] if item.get("id") != run_id]
            payload["runs"].append(run)
            entry_ids = {entry["id"] for entry in entries}
            payload["models"] = [
                item for item in payload["models"]
                if item.get("run_id") != run_id and item.get("id") not in entry_ids
            ]
            payload["models"].extend(entries)
            payload["updated_utc"] = _now()
            _atomic_write(self.path, payload)
        return {"run": deepcopy(run), "models": deepcopy(entries), "registry_path": str(self.path)}

    def models(self, *, selectable_only: bool = False) -> tuple[dict[str, Any], ...]:
        values = self.load()["models"]
        if selectable_only:
            values = [item for item in values if item.get("selectable") is True]
        enriched = deepcopy(values)
        # Older registries used self-comparison for the empirical LUT, yielding the
        # uninformative R²=1. Recompute ideal-reference metrics at read time so
        # existing fitting runs immediately use the corrected interpretation.
        from .error_model import HardwareErrorModel
        for item in enriched:
            method = item.get("method")
            if item.get("selectable") is not True or method not in DEPLOYABLE_LABELS:
                continue
            model_path = Path(str(item.get("path", ""))).expanduser()
            if model_path.is_file():
                item.setdefault("metrics", {}).update(
                    _ideal_lut_metrics(HardwareErrorModel.load(model_path).lut)
                )
            item["name"] = f"{item.get('dataset', '')} {DEPLOYABLE_LABELS[method]}".strip()
        return tuple(enriched)

    def resolve(self, model_id: str, *, require_selectable: bool = True) -> dict[str, Any]:
        matches = [item for item in self.load()["models"] if item.get("id") == model_id]
        if not matches:
            raise KeyError(f"Unknown error model id: {model_id}")
        model = matches[0]
        if require_selectable and model.get("selectable") is not True:
            raise ValueError(f"Model is analytical only and cannot be used for inference: {model_id}")
        model_path = _path(model["path"])
        if not model_path.is_file():
            raise FileNotFoundError(f"Registered error model is missing: {model_path}")
        return deepcopy(model)


class WorkflowContext:
    """Atomic persistent state shared by fitting, inference and training pages."""

    def __init__(self, path: str | Path):
        self.path = _path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": WORKFLOW_SCHEMA,
                "updated_utc": None,
                "network_selection": None,
                "fitting": None,
                "inference": None,
                "training": None,
            }
        payload = _read_object(self.path, "workflow context")
        if payload.get("schema_version") != WORKFLOW_SCHEMA:
            raise ValueError(f"Unsupported workflow schema: {payload.get('schema_version')}")
        return payload

    def select_network_model(
        self, *, model_id: str, task: str, dataset_name: str,
    ) -> dict[str, Any]:
        """Persist the model/dataset combination chosen in the catalog."""
        if not model_id or not task or not dataset_name:
            raise ValueError("model_id, task and dataset_name are required")
        with _LOCK:
            payload = self.load()
            current = payload.get("network_selection") or {}
            changed = current.get("model_id") != model_id
            payload["network_selection"] = {
                "model_id": model_id,
                "task": task,
                "dataset_name": dataset_name,
                "selected_utc": _now(),
            }
            if changed:
                payload["inference"] = None
                payload["training"] = None
            return self._save(payload)

    def selected_network_model(self) -> dict[str, Any] | None:
        selected = self.load().get("network_selection")
        return deepcopy(selected) if isinstance(selected, dict) else None

    def _save(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["updated_utc"] = _now()
        _atomic_write(self.path, payload)
        return deepcopy(payload)

    def select_error_model(self, registry_path: str | Path, model_id: str) -> dict[str, Any]:
        registry = ErrorModelRegistry(registry_path)
        model = registry.resolve(model_id)
        with _LOCK:
            payload = self.load()
            payload["fitting"] = {
                "completed": True,
                "registry_path": str(registry.path),
                "selected_model_id": model_id,
                "error_model_path": str(_path(model["path"])),
                "dataset": model.get("dataset"),
                "source": model.get("source", {}),
                "metrics": model.get("metrics", {}),
                "selected_utc": _now(),
            }
            # A changed upstream error model invalidates dependent results.
            payload["inference"] = None
            payload["training"] = None
            return self._save(payload)

    def record_inference(
        self,
        *,
        model_id: str,
        dataset_name: str,
        dataset_root: str | Path,
        network_model: str,
        checkpoint_path: str | Path,
        auxiliary_path: str | Path | None,
        device: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record a successful inference and make its resources available to training."""
        with _LOCK:
            payload = self.load()
            fitting = payload.get("fitting")
            if not fitting or not fitting.get("completed"):
                raise RuntimeError("Select a fitted error model before recording inference")
            if fitting.get("selected_model_id") != model_id:
                raise ValueError("Inference model id does not match the selected error model")
            selected_network = payload.get("network_selection")
            if selected_network and selected_network.get("model_id") != network_model:
                raise ValueError(
                    "Inference network does not match the model/dataset selection page"
                )
            checkpoint = _path(checkpoint_path)
            dataset = _path(dataset_root)
            auxiliary = _path(auxiliary_path) if auxiliary_path else None
            missing = [
                f"checkpoint: {checkpoint}" if not checkpoint.is_file() else None,
                f"dataset root: {dataset}" if not dataset.exists() else None,
                f"auxiliary: {auxiliary}" if auxiliary and not auxiliary.is_file() else None,
            ]
            missing = [item for item in missing if item]
            if missing:
                raise FileNotFoundError("Inference resources are missing:\n" + "\n".join(missing))
            payload["inference"] = {
                "completed": True,
                "completed_utc": _now(),
                "error_model_id": model_id,
                "error_model_path": fitting["error_model_path"],
                "dataset_name": dataset_name,
                "dataset_root": str(dataset),
                "network_model": network_model,
                "checkpoint_path": str(checkpoint),
                "auxiliary_path": str(auxiliary) if auxiliary else None,
                "device": device,
                "result": dict(result),
            }
            payload["training"] = None
            return self._save(payload)

    def training_inputs(self) -> dict[str, Any]:
        payload = self.load()
        inference = payload.get("inference")
        if not inference or not inference.get("completed"):
            raise RuntimeError("Successful inference is required before error-aware training")
        return {
            key: deepcopy(inference[key])
            for key in (
                "error_model_id",
                "error_model_path",
                "dataset_name",
                "dataset_root",
                "network_model",
                "checkpoint_path",
                "auxiliary_path",
                "device",
            )
        }

    def record_training(self, result: Mapping[str, Any]) -> dict[str, Any]:
        with _LOCK:
            payload = self.load()
            inference = payload.get("inference")
            if not inference or not inference.get("completed"):
                raise RuntimeError("Successful inference is required before recording training")
            payload["training"] = {
                "completed": True,
                "completed_utc": _now(),
                "result": dict(result),
            }
            return self._save(payload)
