"""Portable filesystem locations for NeuroCIM.

Resolution order is: an explicit environment variable, the optional user JSON
configuration, then a repository-relative default.  Merely importing this
module never requires datasets, checkpoints, or compiler tools to exist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

PLATFORM_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1]
)
PROJECT_ROOT = PLATFORM_ROOT.parent


def _config_file() -> Path:
    configured = os.environ.get("NEUROCIM_CONFIG")
    if configured:
        return Path(configured).expanduser()
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / ".config"
    return base / "NeuroCIM" / "config.json"


def _user_config() -> dict[str, Any]:
    path = _config_file()
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read NeuroCIM configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"NeuroCIM configuration must contain a JSON object: {path}")
    return value


def _resolve(env_name: str, config_name: str, fallback: Path) -> Path:
    raw = os.environ.get(env_name)
    if raw is None:
        raw = _user_config().get(config_name)
    return (Path(raw).expanduser() if raw else fallback).resolve()


def data_root(dataset: str | None = None) -> Path:
    """Return a dataset root without checking that data has been installed."""
    key = dataset.lower() if dataset else None
    if key:
        env_name = f"NEUROCIM_{key.upper()}_ROOT"
        configured = os.environ.get(env_name)
        datasets = _user_config().get("datasets", {})
        if configured:
            return Path(configured).expanduser().resolve()
        if isinstance(datasets, dict) and datasets.get(key):
            return Path(datasets[key]).expanduser().resolve()
    base = _resolve("NEUROCIM_DATA_ROOT", "data_root", PROJECT_ROOT / "data")
    return base / key if key else base


def output_root() -> Path:
    if getattr(sys, "frozen", False):
        local_appdata = os.environ.get("LOCALAPPDATA")
        writable_default = (
            Path(local_appdata) / "NeuroCIM" / "outputs"
            if local_appdata
            else Path.home() / "AppData" / "Local" / "NeuroCIM" / "outputs"
        )
    else:
        writable_default = PLATFORM_ROOT / "outputs"
    return _resolve("NEUROCIM_OUTPUT_ROOT", "output_root", writable_default)


def neurosim_binary() -> Path:
    return _resolve(
        "NEUROCIM_NEUROSIM_BIN", "neurosim_binary",
        output_root() / "neurosim" / "bin" / "neurosim.exe",
    )


def toolchain_root() -> Path:
    return _resolve("NEUROCIM_TOOLCHAIN_ROOT", "toolchain_root", PLATFORM_ROOT / ".tools" / "mingw")


def require_path(path: Path, purpose: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{purpose} was not found at {path}. Supply the matching command-line "
            "option, set a NEUROCIM_*_ROOT environment variable, or configure "
            f"{_config_file()}."
        )
    return path
