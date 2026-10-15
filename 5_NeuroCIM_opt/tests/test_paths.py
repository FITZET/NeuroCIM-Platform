from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neurocim_opt import paths


def test_environment_overrides_user_config(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_root": str(tmp_path / "configured")}), encoding="utf-8")
    monkeypatch.setenv("NEUROCIM_CONFIG", str(config))
    monkeypatch.setenv("NEUROCIM_DATA_ROOT", str(tmp_path / "environment"))
    assert paths.data_root() == (tmp_path / "environment").resolve()


def test_dataset_specific_config(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    target = tmp_path / "shared-mnist"
    config.write_text(json.dumps({"datasets": {"mnist": str(target)}}), encoding="utf-8")
    monkeypatch.setenv("NEUROCIM_CONFIG", str(config))
    monkeypatch.delenv("NEUROCIM_MNIST_ROOT", raising=False)
    monkeypatch.delenv("NEUROCIM_DATA_ROOT", raising=False)
    assert paths.data_root("mnist") == target.resolve()


def test_resolution_does_not_require_data_to_exist(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "not-installed"
    monkeypatch.setenv("NEUROCIM_DATA_ROOT", str(missing))
    assert paths.data_root() == missing.resolve()


def test_require_path_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NEUROCIM_.*_ROOT environment variable"):
        paths.require_path(tmp_path / "missing", "MNIST dataset")


def test_packaging_manifest_excludes_local_assets() -> None:
    manifest = json.loads(
        (paths.PLATFORM_ROOT / "packaging_manifest.json").read_text(encoding="utf-8-sig")
    )
    rules = "\n".join(manifest["exclude"])
    for marker in (".tools", "outputs", "data", "checkpoint", "*.pt"):
        assert marker in rules
