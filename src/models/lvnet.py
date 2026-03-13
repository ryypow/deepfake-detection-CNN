"""
LVNet (Locate and Verify) — Two-Stream Network for Deepfake Detection
Implemented by: Hieu (extension/evaluate_lvnet.py + extension/source/Locate-and-Verify)

Original paper:
    Zhong et al., "Locate and Verify: A Two-Stream Network for Improved Deepfake Detection",
    ACM MM 2023. arXiv:2309.11131.  https://github.com/sccsok/Locate-and-Verify

Architecture overview:
    Two parallel Xception streams (RGB + SRM noise residuals) exchange features at each
    stage through CMCE and LFGA cross-modal fusion modules. Multi-scale patch features
    are aggregated via MPFF and fused with HdmProdBilinearFusion into a 4096-dim
    classification embedding. A segmentation head localises forgery regions, and a
    projection head produces contrastive features for semi-supervised patch-similarity
    learning.

    Input : (B, 3, 299, 299) — face crops, normalised to [-1, 1]
    Output: (B, 2) raw logits  [real_logit, fake_logit]

Integration with unified train.py / eval.py:
    Use the `LVNetWrapper` class (key: "lvnet" in MODEL_REGISTRY), which converts the
    2-class output to a single binary logit (B, 1) compatible with BCEWithLogitsLoss.

    LVNet is a complex two-stream model (~50M parameters) trained with both a
    classification loss and a segmentation loss.  Fine-tuning from scratch with the
    shared train.py requires a segmentation-annotated dataset (produced by hieu's
    prepare_data.py).  For inference from hieu's pre-trained checkpoint use eval.py
    with --model_arch lvnet.

Usage:
    # Inference from hieu's checkpoint
    python eval.py \\
      --manifest  ../manifests/FF++/test_frames.json \\
      --checkpoint ../checkpoints/lvnet_best.pth \\
      --model_arch lvnet \\
      --output_dir ../evaluation_results/lvnet_FF

    Note: LVNet's checkpoint format wraps weights under the key 'state_dict', e.g.:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
    The eval.py script handles both plain state-dict and wrapped {'state_dict': ...} formats
    automatically.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# SRM (Steganalysis Rich Model) noise-residual filters
# Source: components/srm_conv.py (Locate-and-Verify)
# =============================================================================

class SRMConv2d_simple(nn.Module):
    """Fixed 3-filter SRM noise-residual extractor (KB, KV, horizontal 2nd-order)."""

    def __init__(self, inc=3, learnable=False):
        super().__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)            # (3, inc, 5, 5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        return self.truc(out)

    @staticmethod
    def _build_kernel(inc):
        f1 = [[0,  0,  0,  0,  0],
              [0, -1,  2, -1,  0],
              [0,  2, -4,  2,  0],
              [0, -1,  2, -1,  0],
              [0,  0,  0,  0,  0]]
        f2 = [[-1,  2, -2,  2, -1],
              [ 2, -6,  8, -6,  2],
              [-2,  8,-12,  8, -2],
              [ 2, -6,  8, -6,  2],
              [-1,  2, -2,  2, -1]]
        f3 = [[0, 0,  0, 0, 0],
              [0, 0,  0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0,  0, 0, 0],
              [0, 0,  0, 0, 0]]
        filters = np.array([
            [np.array(f1) / 4.],
            [np.array(f2) / 12.],
            [np.array(f3) / 2.],
        ])                                          # (3, 1, 5, 5)
        filters = np.repeat(filters, inc, axis=1)  # (3, inc, 5, 5)
        return torch.FloatTensor(filters)


# =============================================================================
# Multi-scale Patch Feature Fusion (MPFF)
# Source: model/modules.py (Locate-and-Verify)
# =============================================================================

class MPFF(nn.Module):
    """Multi-scale Patch Feature Fusion: fuses fa (large map) with fb (19×19 anchor)."""

    def __init__(self, size=19):
        super().__init__()
        self.size = size

    def forward(self, fa, fb):
        b, c, h1, w1 = fa.size()
        b2, c2, h2, w2 = fb.size()
        assert b == b2 and c == c2 and self.size == h2

        padding = abs(h1 % self.size - self.size) % self.size
        if padding:
            pad = nn.ReplicationPad2d((
                padding // 2, (padding + 1) // 2,
                padding // 2, (padding + 1) // 2,
            )).to(fa.device)
            fa = pad(fa)
        b, c, h1, w1 = fa.size()

        window = h1 // self.size
        fb_up = fb.repeat_interleave(window, dim=2).repeat_interleave(window, dim=3)
        ff = torch.tanh(fa * fb_up)
        ff = torch.sum(ff, dim=1, keepdim=True)

        unfold = nn.Unfold(kernel_size=window, stride=window)
        ff = unfold(ff).view(b, -1, self.size, self.size)
        return ff


# =============================================================================
# Cross-Modal Channel Enhancement (CMCE)
# Source: model/modules.py (Locate-and-Verify)
# =============================================================================

class CMCE(nn.Module):
    """Cross-Modal Channel Enhancement via cosine-similarity gating."""

    def __init__(self, in_channel=64):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, fa, fb):
        cos_sim = F.cosine_similarity(fa, fb, dim=1).unsqueeze(1)
        fa = self.relu(fa + fb * cos_sim)
        fb = self.relu(fb + fa * cos_sim)
        return fa, fb


# =============================================================================
# Local Feature Global Attention (LFGA)
# Source: model/modules.py (Locate-and-Verify)
# =============================================================================

class LFGA(nn.Module):
    """Local Feature Global Attention: cross-attention from fb (guide) onto fa."""

    def __init__(self, in_channel=728, ratio=4):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channel, in_channel // ratio, 1)
        self.key_conv   = nn.Conv2d(in_channel, in_channel // ratio, 1)
        self.value_conv = nn.Conv2d(in_channel, in_channel, 1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)
        self.relu       = nn.ReLU()

    def forward(self, fa, fb):
        B, C, H, W = fa.size()
        q = self.query_conv(fb).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_conv(fb).view(B, -1, H * W)
        att = self.softmax(torch.bmm(q, k))
        v = self.value_conv(fa).view(B, -1, H * W)
        out = torch.bmm(v, att.permute(0, 2, 1)).view(B, C, H, W)
        return self.relu(self.gamma * out + fa)


# =============================================================================
# Hadamard-Product Bilinear Fusion
# Source: components/linear_fusion.py (Locate-and-Verify)
# =============================================================================

class HdmProdBilinearFusion(nn.Module):
    """Element-wise product fusion for two 4-D feature maps."""

    def __init__(self, dim1, dim2, hidden_dim=2048, output_dim=4096,
                 bili_dropout=0.5, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.Trans1   = nn.Linear(dim1, hidden_dim)
        self.Trans2   = nn.Linear(dim2, hidden_dim)
        self.OutTrans = nn.Linear(hidden_dim, output_dim)
        self.Dropout  = nn.Dropout(bili_dropout) if bili_dropout else nn.Identity()

    def forward(self, features1, features2):
        b, c1, h, w = features1.size()
        b, c2, _, _ = features2.size()
        f1 = features1.view(b, c1, -1).permute(0, 2, 1).reshape(-1, c1)
        f2 = features2.view(b, c2, -1).permute(0, 2, 1).reshape(-1, c2)
        prod = torch.tanh(self.Trans1(f1) * self.Trans2(f2))
        out  = self.OutTrans(self.Dropout(prod))
        return out.view(b, -1, self.output_dim).permute(0, 2, 1).view(b, self.output_dim, h, w)


# =============================================================================
# Xception backbone (feature-extraction only, no ImageNet head)
# Source: model/xception.py (Locate-and-Verify, originally from FaceForensics++)
# =============================================================================

class _SeparableConv2d(nn.Module):
    def __init__(self, in_c, out_c, ks=1, stride=1, pad=0, dil=1, bias=False):
        super().__init__()
        self.conv1     = nn.Conv2d(in_c, in_c, ks, stride, pad, dil, groups=in_c, bias=bias)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class _Block(nn.Module):
    def __init__(self, in_f, out_f, reps, strides=1,
                 start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = (nn.Sequential(
            nn.Conv2d(in_f, out_f, 1, stride=strides, bias=False),
            nn.BatchNorm2d(out_f),
        ) if (out_f != in_f or strides != 1) else None)

        relu = nn.ReLU(inplace=True)
        rep, filters = [], in_f
        if grow_first:
            rep += [nn.ReLU(inplace=False),
                    _SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_f)]
            filters = out_f
        for _ in range(reps - 1):
            rep += [relu,
                    _SeparableConv2d(filters, filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(filters)]
        if not grow_first:
            rep += [relu,
                    _SeparableConv2d(in_f, out_f, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_f)]
        if not start_with_relu:
            rep = rep[1:]
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        skip = self.skip(x) if self.skip else x
        return out + skip


class _Xception(nn.Module):
    """Xception backbone exposing staged feature-extraction methods used by LVNet."""

    def __init__(self, inc=3):
        super().__init__()
        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 1, bias=False);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False);   self.bn2 = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.block1  = _Block(64,  128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2  = _Block(128, 256, 2, 2, start_with_relu=True,  grow_first=True)
        self.block3  = _Block(256, 728, 2, 2, start_with_relu=True,  grow_first=True)
        # Middle flow
        self.block4  = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5  = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6  = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7  = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8  = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9  = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = _Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        # Exit flow
        self.block12 = _Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3   = _SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3     = nn.BatchNorm2d(1536)
        self.conv4   = _SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4     = nn.BatchNorm2d(2048)

    # -- Staged accessors (match names used by Two_Stream_Net) --

    def fea_part1_0(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x                                    # (B, 64, 150, 150)

    def fea_part1_1(self, x):
        return self.block1(x)                       # (B, 128, 75, 75)

    def fea_part1_2(self, x):
        return self.block2(x)                       # (B, 256, 38, 38)

    def fea_part1_3(self, x):
        return self.block3(x)                       # (B, 728, 19, 19)

    def fea_part2_0(self, x):
        return self.block7(self.block6(self.block5(self.block4(x))))   # (B, 728, 19, 19)

    def fea_part2_1(self, x):
        return self.block11(self.block10(self.block9(self.block8(x)))) # (B, 728, 19, 19)

    def fea_part3(self, x):
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x                                    # (B, 2048, 10, 10)


class _TransferModel(nn.Module):
    """Thin wrapper matching the API used by Two_Stream_Net (return_fea=True)."""

    def __init__(self, dropout=0.5, inc=3):
        super().__init__()
        self.model = _Xception(inc=inc)
        self._dropout = nn.Dropout(p=dropout)
        # Classifier head — not used in Two_Stream_Net but kept for checkpoint compat.
        self.last_linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 2),
        )


# =============================================================================
# BasicConv2d helper
# =============================================================================

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# =============================================================================
# Two_Stream_Net  (the core LVNet model)
# Source: model/LVNet_semi.py (Locate-and-Verify)
# =============================================================================

class Two_Stream_Net(nn.Module):
    """
    Two-stream Xception network with cross-modal fusion for deepfake detection.

    Forward signature: forward(x, mask=None) -> (cls_preds, seg_preds, pro_feas, mask)
        cls_preds : (B, 2) raw 2-class logits  [real, fake]
        seg_preds : (B, 2, 19, 19) segmentation logits
        pro_feas  : (B, 256, 19, 19) projection features
        mask      : downsampled mask (or None)
    """

    def __init__(self):
        super().__init__()
        self.output_dim  = 4096
        self.mid_channel = 512
        self.seg_size    = 19
        self.cls_size    = 10
        self.channels    = [64, 128, 256, 728, 728, 728]

        self.xception_rgb = _TransferModel(dropout=0.5, inc=3)
        self.xception_srm = _TransferModel(dropout=0.5, inc=3)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.relu      = nn.ReLU(inplace=True)

        # 1×1 channel-projection convs (one per xception stage)
        self.score0 = BasicConv2d(self.channels[0], self.mid_channel, kernel_size=1)
        self.score1 = BasicConv2d(self.channels[1], self.mid_channel, kernel_size=1)
        self.score2 = BasicConv2d(self.channels[2], self.mid_channel, kernel_size=1)
        self.score3 = BasicConv2d(self.channels[3], self.mid_channel, kernel_size=1)
        self.score4 = BasicConv2d(self.channels[4], self.mid_channel, kernel_size=1)
        self.score5 = BasicConv2d(self.channels[5], self.mid_channel, kernel_size=1)

        self.msff     = MPFF(size=self.seg_size)
        self.HBFusion = HdmProdBilinearFusion(
            dim1=(64 + 128 + 256 + 728 + 728),  # = 1904
            dim2=2048,
            hidden_dim=2048,
            output_dim=self.output_dim,
        )

        # Cross-modal modules
        self.cmc0 = CMCE(in_channel=64)
        self.cmc1 = CMCE(in_channel=128)
        self.cmc2 = CMCE(in_channel=256)
        self.lfe0 = LFGA(in_channel=728)
        self.lfe1 = LFGA(in_channel=728)
        self.lfe2 = LFGA(in_channel=728)

        # Heads
        self.cls_header = nn.Sequential(
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.output_dim, 2),
        )
        # 728 + 64(x0m) + 16(x1m) + 4(x2m) + 1(x3m) + 1(x4m) = 814
        self.seg_header = nn.Sequential(
            nn.BatchNorm2d(728 + 64 + 16 + 4 + 1 + 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(728 + 64 + 16 + 4 + 1 + 1, 2, kernel_size=1, bias=False),
        )
        self.pro_header = nn.Sequential(
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            nn.Conv2d(728, 256, kernel_size=1, bias=False),
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _pad_max_pool(self, x):
        """Pad-then-pool to a fixed cls_size × cls_size grid."""
        b, c, h, w = x.size()
        padding = abs(h % self.cls_size - self.cls_size) % self.cls_size
        if padding:
            p = (padding // 2, (padding + 1) // 2,
                 padding // 2, (padding + 1) // 2)
            x = nn.ReplicationPad2d(p).to(x.device)(x)
        _, _, h, _ = x.size()
        k = h // self.cls_size
        return nn.MaxPool2d(kernel_size=k, stride=k, padding=0)(x)

    def _get_mask(self, mask):
        b, c, h, w = mask.size()
        padding = abs(h % self.seg_size - self.seg_size) % self.seg_size
        if padding:
            p = (padding // 2, (padding + 1) // 2,
                 padding // 2, (padding + 1) // 2)
            mask = nn.ReplicationPad2d(p).to(mask.device)(mask)
        _, _, h, _ = mask.size()
        k = h // self.seg_size
        mask = nn.MaxPool2d(kernel_size=k, stride=k, padding=0)(mask)
        mask[mask > 0] = 1.0
        return mask

    # -----------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------

    def features(self, x):
        srm = self.srm_conv0(x)

        # Entry flow — cross-modal channel enhancement
        x0 = self.xception_rgb.model.fea_part1_0(x)
        y0 = self.xception_srm.model.fea_part1_0(srm)
        x0, y0 = self.cmc0(x0, y0)                                    # 64 ch, 150×150

        x1 = self.xception_rgb.model.fea_part1_1(x0)
        y1 = self.xception_srm.model.fea_part1_1(y0)
        x1, y1 = self.cmc1(x1, y1)                                    # 128 ch, 75×75

        x2 = self.xception_rgb.model.fea_part1_2(x1)
        y2 = self.xception_srm.model.fea_part1_2(y1)
        x2, y2 = self.cmc2(x2, y2)                                    # 256 ch, 38×38

        # Middle flow — local-feature global attention
        x3 = self.xception_rgb.model.fea_part1_3(x2 + y2)
        y3 = self.xception_srm.model.fea_part1_3(x2 + y2)
        y3 = self.lfe0(y3, x3)                                        # 728 ch, 19×19

        x4 = self.xception_rgb.model.fea_part2_0(x3)
        y4 = self.xception_srm.model.fea_part2_0(y3)
        y4 = self.lfe1(y4, x4)

        x5 = self.xception_rgb.model.fea_part2_1(x4)
        y5 = self.xception_srm.model.fea_part2_1(y4)
        y5 = self.lfe2(y5, x5)                                        # 728 ch, 19×19

        # Exit flow (2048 ch, 10×10)
        x6 = self.xception_rgb.model.fea_part3(x5)
        y6 = self.xception_srm.model.fea_part3(y5)

        # Multi-scale features → segmentation head input
        x0u = self.score0(x0)   # 512, 150×150
        x1u = self.score1(x1)   # 512, 75×75
        x2u = self.score2(x2)   # 512, 38×38
        x3u = self.score3(x3)   # 512, 19×19
        x4u = self.score4(x4)   # 512, 19×19
        x5u = self.score5(x5)   # 512, 19×19

        # MPFF compresses each to (B, window², 19, 19)
        x4m = self.msff(x4u, x5u)   # (B, 1, 19, 19)
        x3m = self.msff(x3u, x5u)   # (B, 1, 19, 19)
        x2m = self.msff(x2u, x5u)   # (B, 4, 19, 19)
        x1m = self.msff(x1u, x5u)   # (B, 16, 19, 19)
        x0m = self.msff(x0u, x5u)   # (B, 64, 19, 19)

        seg_feas = torch.cat((x0m, x1m, x2m, x3m, x4m, x5), dim=1)   # (B, 814, 19, 19)

        # SRM stream → classification head input
        y0m = self._pad_max_pool(y0)   # (B, 64, 10, 10)
        y1m = self._pad_max_pool(y1)   # (B, 128, 10, 10)
        y2m = self._pad_max_pool(y2)   # (B, 256, 10, 10)
        y3m = self._pad_max_pool(y3)   # (B, 728, 10, 10)
        y5m = self._pad_max_pool(y5)   # (B, 728, 10, 10)
        mul_feas = torch.cat((y0m, y1m, y2m, y3m, y5m), dim=1)        # (B, 1904, 10, 10)
        cls_feas = self.HBFusion(mul_feas, y6)                         # (B, 4096, 10, 10)

        return cls_feas, seg_feas, x5

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, x, mask=None):
        cls_feas, seg_feas, x5 = self.features(x)
        cls_preds = self.cls_header(cls_feas)               # (B, 2)
        seg_preds = self.seg_header(seg_feas)               # (B, 2, 19, 19)
        pro_feas  = self.pro_header(x5)                     # (B, 256, 19, 19)

        if mask is not None:
            if isinstance(mask, list):
                mask = [self._get_mask(m) for m in mask]
            else:
                mask = self._get_mask(mask)

        return cls_preds, seg_preds, pro_feas, mask


# =============================================================================
# LVNetWrapper — binary-logit adapter for the unified train.py / eval.py
# =============================================================================

class LVNetWrapper(nn.Module):
    """
    Wraps Two_Stream_Net to produce a single binary logit (B, 1) compatible with
    BCEWithLogitsLoss and the shared eval.py inference loop.

    Logit = cls_preds[:, 1] - cls_preds[:, 0]
          = log P(fake) - log P(real)   (under softmax interpretation)

    Checkpoint loading:
        LVNet checkpoints saved by hieu's training script wrap weights under 'state_dict':
            ckpt = torch.load(path)
            model.load_state_dict(ckpt['state_dict'])
        eval.py handles this automatically.
    """

    def __init__(self):
        super().__init__()
        self.net = Two_Stream_Net()

    def forward(self, x):
        cls_preds, _, _, _ = self.net(x, None)
        # Convert 2-class raw logits → binary logit
        return (cls_preds[:, 1] - cls_preds[:, 0]).unsqueeze(1)   # (B, 1)
