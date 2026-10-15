"""Device-error modeling and differential CIM operators.

Public objects are loaded on first access so lightweight tools such as the
installer bootstrapper do not import the complete NumPy/PyTorch runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .version import __version__


_LAZY_EXPORTS = {
    "HardwareErrorModel": (".error_model", "HardwareErrorModel"),
    "MacDataValidationError": (".mac_error_model", "MacDataValidationError"),
    "MacErrorModel": (".mac_error_model", "MacErrorModel"),
    "load_mac_csv": (".mac_error_model", "load_mac_csv"),
    "DataPreview": (".application", "DataPreview"),
    "MacFitOutcome": (".application", "MacFitOutcome"),
    "MacImportOptions": (".application", "MacImportOptions"),
    "PlatformService": (".application", "PlatformService"),
    "ModelRegistry": (".model_registry", "ModelRegistry"),
    "ModelSpec": (".model_registry", "ModelSpec"),
    "builtin_registry": (".model_registry", "builtin_registry"),
    "DifferentialConv2d": (".layers", "DifferentialConv2d"),
    "DifferentialLinear": (".layers", "DifferentialLinear"),
    "adapt_module": (".layers", "adapt_module"),
    "NeuroSimConfig": (".neurosim_adapter", "NeuroSimConfig"),
    "export_differential_traces": (".neurosim_adapter", "export_differential_traces"),
    "run_neurosim": (".neurosim_adapter", "run_neurosim"),
}

__all__ = [*_LAZY_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
