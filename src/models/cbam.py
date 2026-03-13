"""
Convolutional Block Attention Module (CBAM).

Applies channel attention followed by spatial attention to refine feature maps.
Uses a small reduction ratio (default=2) suited for lightweight models with
few channels (8-16).

This version is used by MesoXceptionNet variants (TJ branch).
For CBAM with Grad-CAM spatial-gate caching, see cbam_module.py.

Reference:
  Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
  https://arxiv.org/abs/1807.06521

Author: TJ (TJ branch)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention: squeeze spatial dims, excite channels.

    AdaptiveAvgPool + AdaptiveMaxPool -> shared MLP (Conv2d 1x1) -> sigmoid
    """

    def __init__(self, channels, reduction=2):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """
    Spatial attention: compress channels, attend to spatial locations.

    Channel-wise mean + max -> concat [B,2,H,W] -> Conv2d(2,1,7x7) -> sigmoid
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return attn * x


class CBAM(nn.Module):
    """Combined CBAM: channel attention followed by spatial attention."""

    def __init__(self, channels, reduction=2, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
