from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _BinarySignSTE(torch.autograd.Function):
    """Sign in forward; clipped-identity gradient on [-1, 1] in backward."""

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


class BinaryActivation(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return binary_sign_ste(value)


class BinaryConv2d(nn.Conv2d):
    def forward(self, value: Tensor) -> Tensor:
        return F.conv2d(value, binary_sign_ste(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BinaryLinear(nn.Linear):
    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, binary_sign_ste(self.weight), self.bias)


class BinaryMNIST(nn.Module):
    """BNN with binary learned weights and binary hidden activations.

    MNIST pixels remain real at the first layer and class logits remain real.
    Every Conv2d/Linear uses sign(weight); every hidden stage uses sign(BN(x)).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            BinaryConv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), BinaryActivation(),
            BinaryConv2d(64, 64, 3, padding=1, bias=False), nn.MaxPool2d(2),
            nn.BatchNorm2d(64), BinaryActivation(),
            BinaryConv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), BinaryActivation(),
            BinaryConv2d(128, 128, 3, padding=1, bias=False), nn.MaxPool2d(2),
            nn.BatchNorm2d(128), BinaryActivation(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), BinaryLinear(128 * 7 * 7, 512, bias=False),
            nn.BatchNorm1d(512), BinaryActivation(), BinaryLinear(512, 10, bias=False),
            nn.BatchNorm1d(10),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.classifier(self.features(value))

    def clamp_latent_weights_(self) -> None:
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (BinaryConv2d, BinaryLinear)):
                    module.weight.clamp_(-1, 1)

    def frozen(self) -> "FrozenBinaryMNIST":
        result = FrozenBinaryMNIST()
        result.load_state_dict(self.state_dict())
        for module in result.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data.copy_(torch.where(module.weight >= 0, 1.0, -1.0))
        result.train(self.training)
        return result


class FrozenBinaryMNIST(nn.Module):
    """Deployment graph containing ordinary Conv2d/Linear with fixed +/-1 weights."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), BinaryActivation(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.MaxPool2d(2),
            nn.BatchNorm2d(64), BinaryActivation(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), BinaryActivation(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.MaxPool2d(2),
            nn.BatchNorm2d(128), BinaryActivation(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 7 * 7, 512, bias=False),
            nn.BatchNorm1d(512), BinaryActivation(), nn.Linear(512, 10, bias=False),
            nn.BatchNorm1d(10),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.classifier(self.features(value))
