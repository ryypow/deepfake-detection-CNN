"""
This version is for experimenting with:
- CBAM placement
- channel progrsssion
- channel sizes
- dimensions
- regularization impact

Channel progression: 11/12/16/16 → 32/48/64/64
inception1 branches: 1/4/4/2 → 4/12/12/4 (symmetric, total 32)
inception2 branches: 2/4/4/2 → 8/16/16/8 (symmetric, total 48)
conv1: 12→16 → 48→64
conv2: 16→16 → 64→64

CBAM reduction: 4 → 8 everywhere

Bottleneck sizes with the new numbers:

inception1_cbam: 32//8 = 4 neurons
inception2_cbam: 48//8 = 6 neurons
conv1_cbam: 64//8 = 8 neurons
conv2_cbam: 64//8 = 8 neurons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam_module import CBAM

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, a, b, c, d):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_ch, a, kernel_size=1, padding=0, bias=False)
        self.branch2_1x1 = nn.Conv2d(in_ch, b, kernel_size=1, padding=0, bias=False)
        self.branch2_3x3 = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)
        self.branch3_1x1 = nn.Conv2d(in_ch, c, kernel_size=1, padding=0, bias=False)
        self.branch3_3x3 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, bias=False)
        self.branch4_1x1 = nn.Conv2d(in_ch, d, kernel_size=1, padding=0, bias=False)
        self.branch4_3x3 = nn.Conv2d(d, d, kernel_size=3, padding=3, dilation=3, bias=False)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        branch1_out = self.ReLU(self.branch1(x))
        branch2_out = self.ReLU(self.branch2_1x1(x))
        branch2_out = self.ReLU(self.branch2_3x3(branch2_out))
        branch3_out = self.ReLU(self.branch3_1x1(x))
        branch3_out = self.ReLU(self.branch3_3x3(branch3_out))
        branch4_out = self.ReLU(self.branch4_1x1(x))
        branch4_out = self.ReLU(self.branch4_3x3(branch4_out))
        output = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        return output

class MesoInception4_CBAM_experimental(nn.Module):
    def __init__(self):
        super(MesoInception4_CBAM_experimental, self).__init__()
        self.inception1 = InceptionBlock(3, a=4, b=12, c=12, d=4)
        self.inception1_bn = nn.BatchNorm2d(32)
        self.inception1_cbam = CBAM(channels=32, reduction=4)
        self.inception1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inception2 = InceptionBlock(32, a=8, b=16, c=16, d=8)
        self.inception2_bn = nn.BatchNorm2d(48)
        self.inception2_cbam = CBAM(channels=48, reduction=4)
        self.inception2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(48, 64, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_cbam = CBAM(channels=64, reduction=4)
        self.conv1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_cbam = CBAM(channels=64, reduction=4)
        self.conv2_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(4096, 16)
        self.fc2 = nn.Linear(16, 1)
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception1_bn(x)
        #x = self.inception1_cbam(x)
        x = self.inception1_pool(x)
        x = self.inception2(x)
        x = self.inception2_bn(x)
        x = self.inception2_cbam(x)
        x = self.inception2_pool(x)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.ReLU(x)
        x = self.conv1_cbam(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.ReLU(x)
        x = self.conv2_cbam(x)
        x = self.conv2_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leakyReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
