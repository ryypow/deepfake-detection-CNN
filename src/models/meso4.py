"""
Meso4 — Standard 4-block MesoNet for deepfake detection.

Original Keras implementation by Hieu Minh (hieu/baseline branch),
ported to PyTorch for compatibility with the shared train.py pipeline.

Architecture:
    Block 1: Conv2d(3→8,  3×3) + BN + ReLU + MaxPool(2)  →  128×128×8
    Block 2: Conv2d(8→8,  5×5) + BN + ReLU + MaxPool(2)  →   64×64×8
    Block 3: Conv2d(8→16, 5×5) + BN + ReLU + MaxPool(2)  →   32×32×16
    Block 4: Conv2d(16→16,5×5) + BN + ReLU + MaxPool(4)  →    8×8×16
    FC      : 1024 → 16 (LeakyReLU, Dropout) → 1 (raw logit)

Input : 256×256 RGB face crops
Output: single logit (use BCEWithLogitsLoss)

Reference:
    Afchar et al., "MesoNet: a Compact Facial Video Forgery Detection
    Network", WIFS 2018.  arXiv:1809.00888
"""

import torch
import torch.nn as nn


class Meso4(nn.Module):
    """Meso4: compact 4-block CNN baseline for deepfake detection."""

    def __init__(self, num_classes: int = 1):
        super().__init__()

        # Block 1 — 256→128
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2 — 128→64
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3 — 64→32
        self.block3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4 — 32→8
        self.block4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        # Classifier — 16×8×8 = 1024 → 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 8 * 8, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # 256→128
        x = self.block2(x)   # 128→64
        x = self.block3(x)   # 64→32
        x = self.block4(x)   # 32→8
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Meso4()
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
