from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "5_NeuroCIM_opt"
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt.layers import DifferentialLinear


DIFFERENTIAL_LAYERS = (
    "transformer.h.11.attn.c_proj",
    "transformer.h.11.mlp.c_proj",
)


def load_native_checkpoint(model: nn.Module, checkpoint: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("best_model_state_dict", payload.get("model_state_dict", payload))
    model.load_state_dict(state)
    return state


def calibrate_input_scales(
    model: nn.Module, token_batches: list[torch.Tensor], *, device: str,
) -> dict[str, float]:
    maxima: dict[str, float] = {}
    modules = dict(model.named_modules())
    handles = []
    for name in DIFFERENTIAL_LAYERS:
        layer = modules.get(name)
        if not isinstance(layer, nn.Linear):
            raise RuntimeError(f"GPT-2 calibration layer is missing: {name}")
        def capture(_module, inputs, layer_name=name):
            value = float(inputs[0].detach().abs().max())
            maxima[layer_name] = max(maxima.get(layer_name, 0.0), value)
        handles.append(layer.register_forward_pre_hook(capture))
    try:
        model.eval().to(device)
        with torch.inference_mode():
            for tokens in token_batches:
                model(tokens.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    epsilon = np.finfo(np.float32).eps
    return {name: max(maxima[name], epsilon) / 255.0 for name in DIFFERENTIAL_LAYERS}


def configure_differential_heads(
    model: nn.Module, lut: np.ndarray | torch.Tensor,
    input_scales: dict[str, float], *, ste: bool = False,
) -> nn.Module:
    """Map the final attention/MLP output projections to signed CIM arithmetic."""
    table = torch.as_tensor(lut)
    for full_name in DIFFERENTIAL_LAYERS:
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        source = getattr(parent, child_name)
        if not isinstance(source, nn.Linear):
            raise TypeError(f"{full_name} must be nn.Linear, got {type(source).__name__}")
        try:
            scale = float(input_scales[full_name])
        except KeyError as exc:
            raise KeyError(f"Missing GPT-2 input scale: {full_name}") from exc
        setattr(parent, child_name, DifferentialLinear.from_module(
            source, lut=table, input_scale=scale, ste=ste,
        ))
    return model
