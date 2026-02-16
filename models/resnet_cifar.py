"""
ResNet-18 for CIFAR with width multiplier.
- 3x3 conv stem, stride 1, padding 1
- No 7x7 conv, no maxpool
- 4 stages, 2 residual blocks each
- Global average pooling, linear head to num_classes
- Stage widths: [base_width, 2*base_width, 4*base_width, 8*base_width]
  with base_width = int(64 * width_multiplier)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def _conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet-18 for CIFAR with configurable width."""

    def __init__(
        self,
        width_multiplier: float = 1.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        base = max(1, int(64 * width_multiplier))
        self.in_planes = base
        # Stem: 3x3 conv, stride 1, padding 1 (no 7x7, no maxpool)
        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        # 4 stages, 2 blocks each
        self.layer1 = self._make_layer(base, 2, stride=1)   # 32x32
        self.layer2 = self._make_layer(base * 2, 2, stride=2)  # 16x16
        self.layer3 = self._make_layer(base * 4, 2, stride=2)  # 8x8
        self.layer4 = self._make_layer(base * 8, 2, stride=2)  # 4x4
        self.linear = nn.Linear(base * 8 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_model(
    width_multiplier: float = 1.0,
    num_classes: int = 10,
) -> ResNetCIFAR:
    """Create Wide ResNet-18 CIFAR model."""
    return ResNetCIFAR(width_multiplier=width_multiplier, num_classes=num_classes)
