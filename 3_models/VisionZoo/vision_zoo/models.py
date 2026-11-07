from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .upstream.mobilenetv2 import cifar10_mobilenetv2_x1_0
from .upstream.vgg import cifar10_vgg11_bn


class BasicBlock(nn.Module):
    """Two-convolution CIFAR-style residual block (option-A shortcut)."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _shortcut(self, value: Tensor) -> Tensor:
        if self.stride != 1:
            value = value[:, :, :: self.stride, :: self.stride]
        channel_padding = self.out_channels - self.in_channels
        if channel_padding:
            left = channel_padding // 2
            right = channel_padding - left
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, left, right))
        return value

    def forward(self, value: Tensor) -> Tensor:
        residual = self._shortcut(value)
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + residual)


class ResNet8MNIST(nn.Module):
    """ResNet with depth ``6n+2`` and ``n=1``, adapted to 1-channel MNIST.

    Depth counts the initial convolution, six block convolutions and the final
    fully connected classifier.  The three residual stages use 16/32/64
    channels and option-A parameter-free shortcuts.
    """

    depth = 8
    blocks_per_stage = 1

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = BasicBlock(16, 16)
        self.stage2 = BasicBlock(16, 32, stride=2)
        self.stage3 = BasicBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, value: Tensor) -> Tensor:
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.stage1(value)
        value = self.stage2(value)
        value = self.stage3(value)
        return self.fc(torch.flatten(self.pool(value), 1))


def resnet8_mnist(*, num_classes: int = 10) -> ResNet8MNIST:
    return ResNet8MNIST(num_classes=num_classes)


def mobilenetv2_x1_0_cifar10(*, num_classes: int = 10) -> nn.Module:
    return cifar10_mobilenetv2_x1_0(pretrained=False, num_classes=num_classes)


def vgg11_bn_cifar10(*, num_classes: int = 10) -> nn.Module:
    return cifar10_vgg11_bn(pretrained=False, num_classes=num_classes)


class CifarFire(nn.Module):
    """Fire block adapted for native 32x32 inputs with stable BN training."""

    def __init__(self, in_channels: int, squeeze_channels: int, expand_channels: int) -> None:
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1, bias=False),
            nn.BatchNorm2d(squeeze_channels), nn.ReLU(inplace=True),
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, 1, bias=False),
            nn.BatchNorm2d(expand_channels),
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(expand_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        value = self.squeeze(value)
        return self.activation(torch.cat((self.expand1x1(value), self.expand3x3(value)), 1))


class CifarSqueezeNet(nn.Module):
    """SqueezeNet topology specialized for CIFAR-10's native 32x32 grid."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            CifarFire(96, 16, 64), CifarFire(128, 16, 64),
            CifarFire(128, 32, 128), nn.MaxPool2d(2),
            CifarFire(256, 32, 128), CifarFire(256, 48, 192),
            CifarFire(384, 48, 192), CifarFire(384, 64, 256),
            nn.MaxPool2d(2), CifarFire(512, 64, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Conv2d(512, num_classes, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value: Tensor) -> Tensor:
        return torch.flatten(self.classifier(self.features(value)), 1)


def squeezenet1_1_cifar10(*, num_classes: int = 10) -> nn.Module:
    """Create the native-CIFAR SqueezeNet graph used by the platform."""

    return CifarSqueezeNet(num_classes=num_classes)


FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "resnet8-mnist": resnet8_mnist,
    "mobilenetv2-cifar10": mobilenetv2_x1_0_cifar10,
    "vgg11-bn-cifar10": vgg11_bn_cifar10,
    "squeezenet1_1-cifar10": squeezenet1_1_cifar10,
}


def create_model(model_id: str) -> nn.Module:
    try:
        factory = FACTORIES[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown VisionZoo model {model_id!r}; available: {sorted(FACTORIES)}") from exc
    return factory()
