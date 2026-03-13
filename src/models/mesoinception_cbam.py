"""
MesoInception4 + CBAM(Convolutional Block Attention Module)
============================================================
Two modules:
-MesoInception4 + CBAM modules
    This is MesoInception4 enhanced with CBAM attention. CBAM helps the model
    learn WHERE deepfake artifacts appear (spatial attention) and WHICH features
    matter most for detecting them (channel attention).

CBAM integration:
- 2 inception blocks:  batch norm->CBAM->max pooling
- 2 standard conv layers: batch norm->CBAM->max pooling
- fully connected layers with dropout for classification

The CBAM sits between BN and pooling because:
    -BN stabilizes features first (good input for attention)
    -Pooling happens AFTER attention so CBAM can see full spatial resolution

CBAM paper: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
https://arxiv.org/abs/1807.06521

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam_module import CBAM

# This is the same InceptionBlock from the original paper.
# It captures features at multiple scales using different
# kernel sizes and dilation rates


#Inception Block
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, a, b, c, d):  # a/b/c/d are the output parameters
        super(InceptionBlock, self).__init__()

        #branch1: 1x1 conv only
        self.branch1 = nn.Conv2d(in_ch, a, kernel_size=1, padding=0, bias=False)

        #branch2: 1x1 bottleneck -> 3x3 conv
        self.branch2_1x1 = nn.Conv2d(in_ch, b, kernel_size=1, padding=0, bias=False)
        self.branch2_3x3 = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)

        #branch3: 1x1 bottleneck -> dilated 3x3 (dilation=2)
        self.branch3_1x1 = nn.Conv2d(in_ch, c, kernel_size=1, padding=0, bias=False)
        self.branch3_3x3 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, bias=False)

        #branch4: 1x1 bottleneck -> dilated 3x3 (dilation=3)
        self.branch4_1x1 = nn.Conv2d(in_ch, d, kernel_size=1, padding=0, bias=False)
        self.branch4_3x3 = nn.Conv2d(d, d, kernel_size=3, padding=3, dilation=3, bias=False)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        # branch 1: just 1x1
        branch1_out = self.ReLU(self.branch1(x))

        #branch 2: 1x1 -> ReLU -> 3x3
        branch2_out = self.ReLU(self.branch2_1x1(x))
        branch2_out = self.ReLU(self.branch2_3x3(branch2_out))

        #branch 3: 1x1 -> ReLU -> dilated 3x3 (dilation=2)
        branch3_out = self.ReLU(self.branch3_1x1(x))
        branch3_out = self.ReLU(self.branch3_3x3(branch3_out))

        #branch 4: 1x1 -> ReLU -> dilated 3x3 (dilation=3)
        branch4_out = self.ReLU(self.branch4_1x1(x))
        branch4_out = self.ReLU(self.branch4_3x3(branch4_out))

        #concat all branches
        output = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        return output

# MESOINCEPTION4 + CBAM
class MesoInception4_CBAM(nn.Module):
    """
    MesoInception4 with CBAM attention in every block
    -Original:  InceptionBlock -> BN -> MaxPool
    -With CBAM: InceptionBlock -> BN -> CBAM -> MaxPool

    """
    def __init__(self):
        super(MesoInception4_CBAM, self).__init__()

        #input: (batch, 3, 256, 256)  ->  3 RGB channels
        #InceptionBlock outputs a+b+c+d = 1+4+4+2 = 11 channels
        self.inception1 = InceptionBlock(3, a=1, b=4, c=4, d=2)
        self.inception1_bn = nn.BatchNorm2d(11)
        self.inception1_cbam = CBAM(channels=11, reduction=4)
        self.inception1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #output: (batch, 11, 128, 128)

        #second inception block
        #inceptionBlock outputs 2+4+4+2 = 12 channels
        self.inception2 = InceptionBlock(11, a=2, b=4, c=4, d=2)
        self.inception2_bn = nn.BatchNorm2d(12)
        self.inception2_cbam = CBAM(channels=12, reduction=4)
        self.inception2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # output: (batch, 12, 64, 64)

        #standard conv1
        #input: (batch, 12, 64, 64)
        self.conv1 = nn.Conv2d(12, 16, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv1_cbam = CBAM(channels=16, reduction=4)
        self.conv1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #output: (batch, 16, 32, 32)

        #standard conv2
        #input: (batch, 16, 32, 32)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_cbam = CBAM(channels=16, reduction=4)
        self.conv2_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        #output: (batch, 16, 8, 8)

        #dense layers
        #input: 16 channels * 8 height * 8 width = 1024
        self.fc1 = nn.Linear(1024, 16)
        self.fc2 = nn.Linear(16, 1)

        #activations/dropout
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.inception1(x) # multi-scale features
        x = self.inception1_bn(x) # normalize:
        #x = self.inception1_cbam(x) # attention: what channels + where spatially
        x = self.inception1_pool(x) # downsample 256->128

        x = self.inception2(x)
        x = self.inception2_bn(x)
        x = self.inception2_cbam(x)
        x = self.inception2_pool(x) # downsample 128->64

        #Conv Layers
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.ReLU(x)
        x = self.conv1_cbam(x)
        x = self.conv1_pool(x) # downsample 64->32

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.ReLU(x)
        x = self.conv2_cbam(x)
        x = self.conv2_pool(x)# downsample 32->8

        #classifier
        x = torch.flatten(x, start_dim=1) # (batch, 1024)

        x = self.dropout(x)
        x = self.fc1(x)# 1024 -> 16
        x = self.leakyReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)# 16 -> 1 (raw logit, no sigmoid)

        return x
