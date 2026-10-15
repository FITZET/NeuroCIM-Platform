from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        residual = self.shortcut(value)
        value = F.relu(self.bn1(self.conv1(value)), inplace=False)
        value = self.bn2(self.conv2(value))
        return F.relu(value + residual, inplace=False)


class ResNetCifar(nn.Module):
    """Canonical 20-layer ResNet for 32×32 CIFAR images."""

    def __init__(self, blocks_per_stage: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._stage(16, blocks_per_stage, stride=1)
        self.stage2 = self._stage(32, blocks_per_stage, stride=2)
        self.stage3 = self._stage(64, blocks_per_stage, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self._initialize()

    def _stage(self, channels: int, count: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        blocks.extend(BasicBlock(channels, channels) for _ in range(1, count))
        return nn.Sequential(*blocks)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value: Tensor) -> Tensor:
        value = F.relu(self.bn1(self.conv1(value)), inplace=False)
        value = self.stage1(value)
        value = self.stage2(value)
        value = self.stage3(value)
        value = F.adaptive_avg_pool2d(value, 1).flatten(1)
        return self.fc(value)


def resnet20(num_classes: int = 10) -> ResNetCifar:
    return ResNetCifar(3, num_classes)
