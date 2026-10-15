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

from neurocim_opt.error_model import HardwareErrorModel
from neurocim_opt.layers import adapt_module


def collect_input_scales(model: nn.Module, loader, device: str, batches: int = 2) -> dict[str, float]:
    maxima: dict[str, float] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_pre_hook(
                lambda _m, args, n=name: maxima.__setitem__(
                    n, max(maxima.get(n, 0.0), float(args[0].detach().abs().max())))))
    model.to(device).eval()
    with torch.inference_mode():
        for index, (inputs, _) in enumerate(loader):
            if index >= batches: break
            model(inputs.to(device))
    for handle in handles: handle.remove()
    epsilon = torch.finfo(torch.float32).eps
    return {name: max(value, epsilon) / 255.0 for name, value in maxima.items()}


def differential_models(frozen: nn.Module, input_scales: dict[str, float], error_manifest: Path):
    codes = np.arange(256, dtype=np.int64)
    ideal_lut = codes[:, None] * codes[None, :]
    device_lut = HardwareErrorModel.load(error_manifest).lut
    ideal = adapt_module(deepcopy(frozen), lut=torch.from_numpy(ideal_lut),
                         input_scales=input_scales, ste=False)
    device = adapt_module(deepcopy(frozen), lut=torch.from_numpy(device_lut),
                          input_scales=input_scales, ste=False)
    return ideal, device
