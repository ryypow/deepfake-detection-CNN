"""
Model-agnostic Grad-CAM and attention map capture.

GradCAM
-------
Works with any nn.Module that:
  - accepts (B, 3, H, W) input tensors
  - outputs a single logit per sample — shape (B, 1) or (B,)

Usage:
    cam = GradCAM(model, layer_name="conv2_cbam")
    heatmap, prob = cam.generate(image_tensor)   # image_tensor: (1, 3, H, W)
    cam.remove_hooks()

AttentionCapture
----------------
Collects spatial attention gate maps from any module that caches `self.last_gate`
during its forward pass (shape B×1×H×W). Works automatically after a GradCAM
forward pass, or after any plain model(x) call.

Usage:
    _ = model(image_tensor)                      # forward pass populates last_gate
    maps = get_attention_maps(model, (256, 256)) # collect all cached gates

Custom models
----------------------
To enable attention map visualization in a custom model, add ONE line to any
attention module's forward():

    gate = sigmoid(...)      # your existing gate computation
    self.last_gate = gate    # <-- add this
    return x * gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Layer utilities

def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolve a dotted layer-name string to the actual nn.Module.

    Uses PyTorch's named_modules(), so any valid dotted path works:
        "conv2_cbam"              (top-level attribute)
        "layer4.1.conv2"          (nested, e.g. ResNet)
        "blocks.6.0.conv_pw"      (nested, e.g. EfficientNet)

    Raises ValueError with a list of all available names if not found.
    """
    modules = dict(model.named_modules())
    if layer_name not in modules:
        available = "\n  ".join(k for k in modules if k)
        raise ValueError(
            f"Layer '{layer_name}' not found in model.\n"
            f"Available layer names:\n  {available}"
        )
    return modules[layer_name]


def find_last_conv(model: nn.Module) -> str:
    """Auto-detect the last Conv2d layer name in the model.

    Used as a fallback when the model arch is not in GRAD_CAM_LAYERS and
    the user hasn't supplied --target_layer. The last Conv2d is usually a
    reasonable Grad-CAM target for any CNN.
    """
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
    if last_name is None:
        raise ValueError("No Conv2d layers found in model.")
    return last_name


def list_layer_names(model: nn.Module) -> list[str]:
    """Return all non-empty layer names — useful for exploring a new model."""
    return [name for name in dict(model.named_modules()).keys() if name]

# Grad-CAM

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Selselberger et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
    https://arxiv.org/abs/1610.02391

    Args:
        model: any nn.Module -> must output logit
        layer_name: Dotted name of the target convolutional layer.
                    Use list_layer_names(model) to discover valid names.
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self._layer_name = layer_name
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        target = get_layer(model, layer_name)
        self._fwd_hook = target.register_forward_hook(self._save_activation)
        self._bwd_hook = target.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    def _save_activation(self, module, input, output):
        self._activations = output.detach()   # (B, C, H, W)

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()  # (B, C, H, W)

    # ------------------------------------------------------------------
    def generate(self, image_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """Compute Grad-CAM heatmap for a single image.

        Args:
            image_tensor: (1, 3, H, W) on the model's device.
                          Must NOT be inside a torch.no_grad() context.

        Returns:
            heatmap: (H, W) float32 numpy array in [0, 1]
            prob:    sigmoid probability of the fake class (float)
        """
        self.model.eval()
        img = image_tensor.clone().requires_grad_(True)

        logit = self.model(img).squeeze()   # scalar
        prob  = torch.sigmoid(logit).item()

        self.model.zero_grad()
        logit.backward()                    # populates self._gradients

        # Importance weight: mean gradient over spatial dims for each channel
        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum across channels + ReLU (only keep positive contributions)
        cam = (alpha * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Upsample to input image size
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)

        return cam.squeeze().cpu().numpy().astype(np.float32), prob

    # ------------------------------------------------------------------
    def remove_hooks(self):
        """Always call this when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ──────────────────────────────────────────────
# Attention map capture
# ──────────────────────────────────────────────

def get_attention_maps(
    model: nn.Module,
    target_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Collect all spatial attention gate maps set during the last forward pass.

    Scans every module in the model for a `last_gate` attribute (set by
    SpatialAttention.forward() in cbam_module.py, or any partner attention
    module that follows the same convention).

    Args:
        model:       The model (must have had a forward pass already).
        target_size: (H, W) to upsample all maps to — typically (256, 256).

    Returns:
        Dict mapping layer_name -> (H, W) float32 numpy array in [0, 1].
        Empty dict if no attention modules cached their gates.
    """
    maps = {}
    for name, module in model.named_modules():
        gate = getattr(module, 'last_gate', None)
        if gate is None:
            continue
        gate_up = F.interpolate(
            gate.float(), size=target_size, mode='bilinear', align_corners=False
        )
        maps[name] = gate_up.squeeze().cpu().detach().numpy().astype(np.float32)
    return maps
