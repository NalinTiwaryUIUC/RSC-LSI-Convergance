from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension at each (H, W) location."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C]
        out = x.permute(0, 2, 3, 1)
        out = self.ln(out)
        # back to [B, C, H, W]
        return out.permute(0, 3, 1, 2)


class BasicBlockLN(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.ln1 = ChannelLayerNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = ChannelLayerNorm(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SmallResNetCIFAR_LN(nn.Module):
    """
    Small ResNet-style network for CIFAR with LayerNorm.

    - CIFAR stem: 3x3 conv, stride 1, padding 1, no maxpool.
    - Single stage of BasicBlockLN blocks at 32x32 resolution.
    - Global average pooling and linear head.
    - Adjustable width via width_multiplier.
    """

    def __init__(
        self,
        width_multiplier: float = 1.0,
        num_classes: int = 10,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        base = max(1, int(64 * width_multiplier))
        self.in_planes = base

        # Stem: 3x3 conv, stride 1, padding 1 (no 7x7, no maxpool)
        self.conv1 = nn.Conv2d(
            3,
            base,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.ln1 = ChannelLayerNorm(base)
        self.relu = nn.ReLU(inplace=True)

        # Single stage of BasicBlockLN blocks at 32x32
        self.layer1 = self._make_layer(BasicBlockLN, base, num_blocks, stride=1)

        # Head: global average pooling and linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * BasicBlockLN.expansion, num_classes)

    def _make_layer(
        self,
        block: type[BasicBlockLN],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                ChannelLayerNorm(planes * block.expansion),
            )

        layers: list[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_small_resnet_ln(
    width_multiplier: float = 1.0,
    num_classes: int = 10,
    num_blocks: int = 2,
) -> SmallResNetCIFAR_LN:
    return SmallResNetCIFAR_LN(
        width_multiplier=width_multiplier,
        num_classes=num_classes,
        num_blocks=num_blocks,
    )

