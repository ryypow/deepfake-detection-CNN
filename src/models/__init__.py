"""
Model registry for all deepfake detection architectures.

Available models:
    Meso4                           — Standard 4-block MesoNet (Hieu, ported from Keras)
    MesoInception4                  — Inception-block MesoNet (rypow)
    MesoInception4_CBAM             — MesoInception4 + CBAM attention (rypow)
    MesoInception4_CBAM_experimental — Wider CBAM variant, more channels (rypow)
    MesoXceptionNet                 — MesoNet with depthwise separable convs (TJ)
    MesoXceptionCBAM                — MesoXceptionNet + CBAM attention (TJ)
    LVNetWrapper                    — Two-Stream LVNet (Hieu), see lvnet.py

Standard compact CNNs (meso4, mesoinception4*,  meso_xception*):
  - Input:  (B, 3, 256, 256)  [RGB face crops at 256×256, FF++ normalisation]
  - Output: (B, 1)  raw logit  -> use BCEWithLogitsLoss during training

LVNet:
  - Input:  (B, 3, 299, 299)  [RGB, normalised to [-1, 1]]
  - Output: (B, 1)  raw logit  (wrapper converts 2-class output to binary logit)
  - Note:   Full training requires segmentation labels from hieu's prepare_data.py.
            Inference from hieu's checkpoint works out-of-the-box with eval.py.

Usage:
    from models import MODEL_REGISTRY
    model = MODEL_REGISTRY["mesoinception4_cbam"]()
"""

from .meso4 import Meso4
from .meso_inception_base import MesoInception4
from .mesoinception_cbam import MesoInception4_CBAM
from .mesoinception_cbam_experimental import MesoInception4_CBAM_experimental
from .meso_xception import MesoXceptionNet
from .meso_xception_cbam import MesoXceptionCBAM
from .lvnet import LVNetWrapper

# Registry of all model architectures.
# To add a new model: import its class above, then add a key here.
MODEL_REGISTRY = {
    # ── Original MesoNet baseline ─────────────────────────────────────────────
    "meso4":                            Meso4,

    # ── MesoInception variants (rypow) ────────────────────────────────────────
    "mesoinception4":                   MesoInception4,
    "mesoinception4_cbam":              MesoInception4_CBAM,
    "mesoinception4_cbam_experimental": MesoInception4_CBAM_experimental,

    # ── MesoXception variants (TJ) ────────────────────────────────────────────
    "meso_xception":                    MesoXceptionNet,
    "meso_xception_cbam":               MesoXceptionCBAM,

    # ── LVNet two-stream network (Hieu) ───────────────────────────────────────
    "lvnet":                            LVNetWrapper,
}

# Default Grad-CAM target layer for each architecture.
# Should be the last convolutional stage before the classifier (flatten/FC).
# To find valid names for your model, run:
#   python -c "from models import MODEL_REGISTRY; m = MODEL_REGISTRY['yourarch'](); \
#              print([n for n, _ in m.named_modules() if n])"
GRAD_CAM_LAYERS = {
    "meso4":                            "block4.2",   # last ReLU before pool
    "mesoinception4":                   "conv2_bn",
    "mesoinception4_cbam":              "conv2_cbam",
    "mesoinception4_cbam_experimental": "conv2_cbam",
    "meso_xception":                    "sep4.sep_conv.pointwise",
    "meso_xception_cbam":               "sep4.sep_conv.pointwise",
    "lvnet":                            "net.xception_rgb.model.block12",
}
