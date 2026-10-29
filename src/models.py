"""Wide ResNet for CIFAR-10, with optional Squeeze-and-Excitation blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention with a reduction ratio of 16."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3))
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(scale), inplace=True)))
        return x * scale.view(scale.shape[0], -1, 1, 1)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout: float = 0.0,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.se = SqueezeExcitation(out_planes) if use_se else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        return F.relu(out + self.shortcut(x), inplace=True)


class WideResNet(nn.Module):
    """Wide ResNet (Zagoruyko & Komodakis, 2016) for 32x32 images.

    depth must satisfy (depth - 4) % 6 == 0.  The standard CIFAR model
    WRN-28-10 uses depth=28 and widen_factor=10.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        widen_factor: int = 10,
        dropout: float = 0.0,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("depth must satisfy (depth - 4) % 6 == 0")
        n_blocks = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(
            widths[0], widths[1], n_blocks, stride=1, dropout=dropout, use_se=use_se
        )
        self.layer2 = self._make_layer(
            widths[1], widths[2], n_blocks, stride=2, dropout=dropout, use_se=use_se
        )
        self.layer3 = self._make_layer(
            widths[2], widths[3], n_blocks, stride=2, dropout=dropout, use_se=use_se
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        self._initialize()

    @staticmethod
    def _make_layer(
        in_planes: int,
        out_planes: int,
        n_blocks: int,
        stride: int,
        dropout: float,
        use_se: bool,
    ) -> nn.Sequential:
        blocks = [
            BasicBlock(in_planes, out_planes, stride, dropout=dropout, use_se=use_se)
        ]
        blocks += [
            BasicBlock(out_planes, out_planes, 1, dropout=dropout, use_se=use_se)
            for _ in range(n_blocks - 1)
        ]
        return nn.Sequential(*blocks)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Start every residual branch at zero; this stabilises early training.
        for module in self.modules():
            if isinstance(module, BasicBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)


def build_model(args: object) -> WideResNet:
    return WideResNet(
        num_classes=10,
        depth=args.depth,
        widen_factor=args.width,
        dropout=args.dropout,
        use_se=args.se,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
