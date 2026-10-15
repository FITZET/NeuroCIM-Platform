from __future__ import annotations

from enum import Enum
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_BASIC = PROJECT_ROOT / "2_model_basic"
if str(MODEL_BASIC) not in sys.path:
    sys.path.insert(0, str(MODEL_BASIC))

from fault_layers import FaultConv2d, FaultLinear


class InferenceMode(str, Enum):
    FLOAT = "float"
    QUANTIZED = "quantized"
    FAULT = "fault"


class LeNet5(nn.Module):
    """Classic 32x32 LeNet-5 built directly from reusable fault layers."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = FaultConv2d(1, 6, kernel_size=5)
        self.conv2 = FaultConv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activation = nn.Tanh()
        self.fc1 = FaultLinear(16 * 5 * 5, 120)
        self.fc2 = FaultLinear(120, 84)
        self.fc3 = FaultLinear(84, num_classes)
        self._inference_mode = InferenceMode.FLOAT

    @property
    def inference_mode(self) -> InferenceMode:
        return self._inference_mode

    def fault_layers(self) -> dict[str, FaultConv2d | FaultLinear]:
        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, (FaultConv2d, FaultLinear))
        }

    def set_inference_mode(self, mode: InferenceMode | str) -> LeNet5:
        selected = InferenceMode(mode)
        if selected is not InferenceMode.FLOAT:
            missing = [name for name, layer in self.fault_layers().items() if not layer.has_hardware_lut]
            if missing:
                raise RuntimeError(f"LUT/quantization parameters are not configured for: {missing}")
        for layer in self.fault_layers().values():
            layer.enable_fault(selected is not InferenceMode.FLOAT)
        self._inference_mode = selected
        return self

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)
