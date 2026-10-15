from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "5_NeuroCIM_opt"
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.layers import DifferentialLinear, adapt_module


FINETUNE_HEADS = ("classifier.attention_pool", "classifier.fc")


def calibrate_input_scales(
    model: nn.Module, loader, *, device: str, max_batches: int = 1,
) -> dict[str, float]:
    maxima: dict[str, float] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            def capture(_module, inputs, layer_name=name):
                maximum = float(inputs[0].detach().abs().max())
                maxima[layer_name] = max(maxima.get(layer_name, 0.0), maximum)
            handles.append(module.register_forward_pre_hook(capture))
    try:
        model.to(device).eval()
        with torch.inference_mode():
            for index, (inputs, _targets) in enumerate(loader):
                if index >= max_batches:
                    break
                model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()
    expected = {name for name, layer in model.named_modules()
                if isinstance(layer, (nn.Conv2d, nn.Linear))}
    if missing := expected.difference(maxima):
        raise RuntimeError(f"Calibration did not execute layers: {sorted(missing)}")
    epsilon = np.finfo(np.float32).eps
    return {name: max(value, epsilon) / 255.0 for name, value in maxima.items()}


def differential_copy(
    model: nn.Module, lut: np.ndarray | torch.Tensor,
    input_scales: dict[str, float], *, ste: bool = False,
) -> nn.Module:
    converted = deepcopy(model)
    adapt_module(converted, lut=torch.as_tensor(lut), input_scales=input_scales, ste=ste)
    return converted


def differential_head_copy(
    model: nn.Module, lut: np.ndarray | torch.Tensor,
    input_scales: dict[str, float], *, ste: bool = False,
) -> nn.Module:
    """Convert only the two fine-tuned CCT heads to signed-differential layers."""
    converted = deepcopy(model)
    table = torch.as_tensor(lut)
    for full_name in FINETUNE_HEADS:
        parent_name, _, child_name = full_name.rpartition(".")
        parent = converted.get_submodule(parent_name)
        source = getattr(parent, child_name)
        if not isinstance(source, nn.Linear):
            raise TypeError(f"{full_name} must be nn.Linear, got {type(source).__name__}")
        if full_name not in input_scales:
            raise KeyError(f"Missing input scale for fine-tuning head: {full_name}")
        setattr(parent, child_name, DifferentialLinear.from_module(
            source, lut=table, input_scale=float(input_scales[full_name]), ste=ste,
        ))
    return converted
