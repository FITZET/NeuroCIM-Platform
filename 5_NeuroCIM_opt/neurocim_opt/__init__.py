"""Device-error modeling and differential CIM operators."""

from .version import __version__

from .error_model import HardwareErrorModel
from .mac_error_model import MacDataValidationError, MacErrorModel, load_mac_csv
from .application import DataPreview, MacFitOutcome, MacImportOptions, PlatformService
from .model_registry import ModelRegistry, ModelSpec, builtin_registry
from .layers import DifferentialConv2d, DifferentialLinear, adapt_module
from .neurosim_adapter import NeuroSimConfig, export_differential_traces, run_neurosim

__all__ = [
    "HardwareErrorModel",
    "MacDataValidationError",
    "MacErrorModel",
    "load_mac_csv",
    "DataPreview",
    "MacFitOutcome",
    "MacImportOptions",
    "PlatformService",
    "ModelRegistry",
    "ModelSpec",
    "builtin_registry",
    "DifferentialConv2d",
    "DifferentialLinear",
    "adapt_module",
    "NeuroSimConfig",
    "export_differential_traces",
    "__version__",
    "run_neurosim",
]
