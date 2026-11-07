from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _BinarySignSTE(torch.autograd.Function):
    """Deterministic sign with the clipped-identity BNN gradient."""

    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        ctx.save_for_backward(value)
        return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

    @staticmethod
    def backward(ctx, gradient: Tensor) -> Tensor:
        (value,) = ctx.saved_tensors
        return gradient * (value.abs() <= 1).to(gradient.dtype)


def binary_sign_ste(value: Tensor) -> Tensor:
    return _BinarySignSTE.apply(value)


class BinaryConv2d(nn.Conv2d):
    def forward(self, value: Tensor) -> Tensor:
        return F.conv2d(value, binary_sign_ste(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BinaryLinear(nn.Linear):
    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, binary_sign_ste(self.weight), self.bias)


class BinaryMNIST(nn.Module):
    """RWTH-IDS binary LeNet-5 adapted to the platform interfaces."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = BinaryConv2d(1, 32, 5, padding=2, bias=False)
        self.conv2 = BinaryConv2d(32, 64, 5, bias=False)
        self.fc1 = BinaryLinear(64 * 5 * 5, 512, bias=False)
        self.fc2 = BinaryLinear(512, 10, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(10, affine=False)

    @staticmethod
    def _activation(value: Tensor) -> Tensor:
        return binary_sign_ste(F.hardtanh(value))

    def forward(self, value: Tensor) -> Tensor:
        value = self._activation(self.bn1(F.max_pool2d(self.conv1(value), 2)))
        value = self._activation(self.bn2(F.max_pool2d(self.conv2(value), 2)))
        value = self._activation(self.bn3(self.fc1(value.flatten(1))))
        return self.bn4(self.fc2(value))

    def clamp_latent_weights_(self) -> None:
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (BinaryConv2d, BinaryLinear)):
                    module.weight.clamp_(-1, 1)

    def frozen(self) -> "FrozenBinaryMNIST":
        result = FrozenBinaryMNIST()
        result.load_state_dict(self.state_dict())
        with torch.no_grad():
            for module in result.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.copy_(torch.where(module.weight >= 0, 1.0, -1.0))
        result.train(self.training)
        return result


class FrozenBinaryMNIST(nn.Module):
    """Deployment graph with fixed +/-1 weights for differential conversion."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.fc1 = nn.Linear(64 * 5 * 5, 512, bias=False)
        self.fc2 = nn.Linear(512, 10, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(10, affine=False)

    @staticmethod
    def _activation(value: Tensor) -> Tensor:
        return torch.where(F.hardtanh(value) >= 0, 1.0, -1.0)

    def forward(self, value: Tensor) -> Tensor:
        value = self._activation(self.bn1(F.max_pool2d(self.conv1(value), 2)))
        value = self._activation(self.bn2(F.max_pool2d(self.conv2(value), 2)))
        value = self._activation(self.bn3(self.fc1(value.flatten(1))))
        return self.bn4(self.fc2(value))
