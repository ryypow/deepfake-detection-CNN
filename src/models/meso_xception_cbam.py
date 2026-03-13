"""
MesoXceptionNet + CBAM: Attention-enhanced deepfake detection.

Adds CBAM (channel + spatial attention) after BN+ReLU and before MaxPool
in every block of MesoXceptionNet. Keeps residual connections where
input/output channels match.

Architecture (256x256 input):
    Conv1   : Conv2d(3->8, 3x3) + BN + ReLU + CBAM(8) + MaxPool(2)          -> 128x128x8
    SepConv2: SepConv(8->8, 5x5) + BN + ReLU + CBAM(8) + MaxPool(2) + skip  ->  64x64x8
    SepConv3: SepConv(8->16, 5x5) + BN + ReLU + CBAM(16) + MaxPool(2)       ->  32x32x16
    SepConv4: SepConv(16->16, 5x5) + BN + ReLU + CBAM(16) + MaxPool(4)+skip ->   8x8x16
    Flatten : 16x8x8 = 1024
    FC1     : 1024 -> 16 + LeakyReLU + Dropout(0.5)
    FC2     : 16 -> 1 (raw logit)

Input : 256x256 RGB face crops
Output: single logit (use BCEWithLogitsLoss)

Author: TJ (TJ branch)
"""

import torch
import torch.nn as nn

from .meso_xception import SeparableConv2d
from .cbam import CBAM


class SepConvBlockCBAM(nn.Module):
    """Separable conv -> BN -> ReLU -> CBAM -> MaxPool, with optional residual."""

    def __init__(self, in_ch, out_ch, kernel_size=5, pool_size=2, reduction=2):
        super().__init__()
        padding = kernel_size // 2

        self.sep_conv = SeparableConv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_ch, reduction=reduction)
        self.pool = nn.MaxPool2d(pool_size, pool_size)

        self.use_residual = (in_ch == out_ch)
        if self.use_residual:
            self.shortcut = nn.MaxPool2d(pool_size, pool_size)

    def forward(self, x):
        identity = x
        out = self.sep_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.cbam(out)
        out = self.pool(out)

        if self.use_residual:
            out = out + self.shortcut(identity)

        return out


class MesoXceptionCBAM(nn.Module):
    """
    MesoXceptionNet with CBAM attention in every block.

    CBAM (channel + spatial attention) is inserted after BatchNorm+ReLU
    and before MaxPool in each block, allowing the model to focus on
    relevant channels and spatial regions before downsampling.

    ~19k parameters (≈6% overhead vs base MesoXceptionNet).
    """

    def __init__(self, num_classes=1, reduction=2):
        super().__init__()

        # Block 1: standard conv + CBAM
        self.conv1_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_relu = nn.ReLU(inplace=True)
        self.conv1_cbam = CBAM(8, reduction=reduction)
        self.conv1_pool = nn.MaxPool2d(2, 2)  # 256->128

        # Blocks 2-4: depthwise separable convolutions + CBAM
        self.sep2 = SepConvBlockCBAM(8, 8, kernel_size=5, pool_size=2, reduction=reduction)
        self.sep3 = SepConvBlockCBAM(8, 16, kernel_size=5, pool_size=2, reduction=reduction)
        self.sep4 = SepConvBlockCBAM(16, 16, kernel_size=5, pool_size=4, reduction=reduction)

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
        # Block 1
        x = self.conv1_conv(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.conv1_cbam(x)
        x = self.conv1_pool(x)  # 256->128

        # Blocks 2-4
        x = self.sep2(x)    # 128->64
        x = self.sep3(x)    # 64->32
        x = self.sep4(x)    # 32->8

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MesoXceptionCBAM()
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
