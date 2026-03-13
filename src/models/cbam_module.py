"""
CBAM - Convolutional Block Attention Module

What it does:
1. Channel Attention: looks at ALL channels and asks "which feature maps
     are most useful for detecting fakes?" Then it boosts useful channels
     and suppresses useless ones.
2. Spatial Attention: looks at ALL spatial positions and asks "where in this
     image should I focus?" Then it boosts important regions (like face
     boundaries where artifacts appear) and suppresses background.

Reduction ratio:
    - reduction factor of 4
    - reason: gentle compression, more expressive -> models subtler inter-channel dependencies

The output ispassed through sigmoid to get values [0,1] -> these become per-channel multipliers that 
scale the original feature map

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction_factor):
        super(ChannelAttention, self).__init__()
        #pools across spatial dimensions -> squeezes each channel to a single number
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #self.reduction_ratio = 4 
        self.reduction = max(in_ch // reduction_factor, 1)  # at least 1 neuron

        #the MLP bottleneck: 
        #first convolution squeezes: compresses 11 channel input into 2
        #Relu to add non-linearity between the convolution layers
        #second convolution: expands back to 11, producing one attention weight per channel
        self.fc = nn.Sequential(nn.Conv2d(in_ch, self.reduction, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(self.reduction, in_ch, 1, bias=False))

    def forward(self, x):
        """
        Args:
            x: input feature map, shape (batch, channels, height, width)
        Returns:
            refined feature map, same shape as input
        """
        #batch, channels, height, width = x.size()

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        gate = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1) values 0-1
        return x * gate  # scale each channel by its importance
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        #padding = kernel_size // 2
        #2 input channels (avg + max), 1 output channel (attentionmap)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Returns:
            refined feature map, same shape as input
        """
        #reminder: spatial attention pools across channel dimensions
        #squeezes all channels into one value per pixel
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        gate = self.sigmoid(self.conv1(combined))  # (B, 1, H, W) values 0-1
        self.last_gate = gate  # cached for AttentionCapture in gradcam.py
        return x * gate  # scale each pixel by its importance
    
class CBAM(nn.Module):
    """
    Combines Channel Attention + Spatial Attention in sequence.
    Order matters: channel first, then spatial. The paper tested both orders
    and found channel-first works slightly better.
    Source: CBAM paper Section 3, Figure 2 (the full module diagram)
    """
    def __init__(self, channels, reduction=4, spatial_kernel=7):
        """
        Args:
            channels: number of input channels
            reduction: reduction ratio for channel attention bottleneck
            spatial_kernel: kernel size for spatial attention conv
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        #Step 1: Channel attention - refine "what" to focus on
        x = self.channel_attention(x)
        #Step 2: Spatial attention - refine "where" to focus
        x = self.spatial_attention(x)
        return x
