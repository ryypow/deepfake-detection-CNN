"""
Meso-XceptionNet: A MesoNet variant using depthwise separable convolutions.

Replaces standard convolutions in the Meso4 architecture with Xception-style
depthwise separable convolution blocks. Includes residual (skip) connections
where input and output dimensions match.

Architecture (256x256 input):
    Conv1   : standard Conv2d(3→8, 3x3)  + BN + ReLU + MaxPool(2)  → 128×128×8
    SepConv2: SepConv(8→8,   5x5) + BN + ReLU + MaxPool(2) + skip  →  64×64×8
    SepConv3: SepConv(8→16,  5x5) + BN + ReLU + MaxPool(2)         →  32×32×16
    SepConv4: SepConv(16→16, 5x5) + BN + ReLU + MaxPool(4) + skip  →   8×8×16
    Flatten : 16×8×8 = 1024
    FC1     : 1024 → 16  + LeakyReLU + Dropout(0.5)
    FC2     : 16   → 1   (raw logit)

Input : 256×256 RGB face crops
Output: single logit (use BCEWithLogitsLoss)

Author: TJ (TJ branch)

Reference:
  - MesoNet: Afchar et al., "MesoNet: a Compact Facial Video Forgery
    Detection Network", 2018 (arXiv:1809.00888)
  - Xception: Chollet, "Xception: Deep Learning with Depthwise Separable
    Convolutions", 2017 (arXiv:1610.02357)
"""

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SepConvBlock(nn.Module):
    """Separable conv -> BatchNorm -> ReLU -> MaxPool, with optional residual."""

    def __init__(self, in_ch, out_ch, kernel_size=5, pool_size=2):
        super().__init__()
        padding = kernel_size // 2  # same-padding

        self.sep_conv = SeparableConv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size, pool_size)

        # Residual shortcut when dims match (same channels, pool shrinks spatial)
        self.use_residual = (in_ch == out_ch)
        if self.use_residual:
            self.shortcut = nn.MaxPool2d(pool_size, pool_size)

    def forward(self, x):
        identity = x
        out = self.sep_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)

        if self.use_residual:
            out = out + self.shortcut(identity)

        return out


class MesoXceptionNet(nn.Module):
    """
    Meso-XceptionNet for binary deepfake detection.

    Replaces standard convolutions with Xception-style depthwise separable
    convolutions, reducing parameters while maintaining representational power.
    Residual connections are added where input/output channels match.
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # Block 1: standard conv (3 input channels is too few for separable)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256→128
        )

        # Blocks 2-4: depthwise separable convolutions
        self.sep2 = SepConvBlock(8, 8, kernel_size=5, pool_size=2)    # 128→64, residual
        self.sep3 = SepConvBlock(8, 16, kernel_size=5, pool_size=2)   # 64→32
        self.sep4 = SepConvBlock(16, 16, kernel_size=5, pool_size=4)  # 32→8,  residual

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)   # 256→128
        x = self.sep2(x)    # 128→64
        x = self.sep3(x)    # 64→32
        x = self.sep4(x)    # 32→8
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MesoXceptionNet()
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
