"""
Visualize Grad-CAM and/or CBAM spatial attention maps for any trained model.

Each sampled image is saved as a multi-panel PNG:
  [Original] [Grad-CAM overlay] [Attention stage 1] ... [Attention stage N]

Filenames encode the prediction outcome for easy browsing:
  0003_Deepfakes_FN_p0.31.png  <- false negative, model gave prob=0.31 for fake

Usage: from /src
  python visualize.py \\
    --manifest   ../manifests/FF++/test_frames.json \\
    --checkpoint checkpoints_xxx/best_model.pth \\
    --model_arch mesoinception4_cbam \\
    --output_dir ../eval_viz/run_name \\
    --n_samples  20 \\
    --mode       both

Optional flags:
  --target_layer  conv2_cbam      Override the default Grad-CAM layer
  --method        Deepfakes       Restrict to one generation method
  --seed          42
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM, find_last_conv, get_attention_maps, list_layer_names
from models import GRAD_CAM_LAYERS, MODEL_REGISTRY


# Image normalization (must match training / eval.py)
 #-> tailor for whichever dataset the model was trained on

#FF++ stats — for from-scratch MesoInception models
_FF_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5027, 0.3921, 0.3616],
                         std=[0.2643, 0.2193, 0.2178]),
])
#ImageNet stats — for pretrained Xception
_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
#ImageNet stats + resize — for pretrained ViT (requires 224x224)
_VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_eval_transform(model_arch):
    if model_arch == "vit_b16":
        return _VIT_TRANSFORM
    elif model_arch == "xception_pretrained":
        return _IMAGENET_TRANSFORM
    else:
        return _FF_TRANSFORM

_DENORM_STATS = {
    "vit_b16":             ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "xception_pretrained": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}
_FF_STATS = ([0.5027, 0.3921, 0.3616], [0.2643, 0.2193, 0.2178])

def get_denorm_stats(model_arch):
    return _DENORM_STATS.get(model_arch, _FF_STATS)


def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) uint8 numpy for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ──────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────

def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap over a uint8 RGB image using the jet colormap.

    Args:
        image_np: (H, W, 3) uint8
        heatmap:  (H, W) float32 in [0, 1]
        alpha:    heatmap opacity

    Returns:
        (H, W, 3) uint8 blended image
    """
    colormap   = plt.get_cmap('jet')
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended    = ((1 - alpha) * image_np + alpha * heatmap_rgb).astype(np.uint8)
    return blended


def outcome_label(label: int, prob: float) -> str:
    """TP / TN / FP / FN based on ground truth label and predicted probability."""
    pred = int(prob > 0.5)
    if pred == label:
        return "TP" if label == 1 else "TN"
    return "FN" if label == 1 else "FP"


def shorten_layer_name(name: str) -> str:
    """'inception2_cbam.spatial_attention' -> 'inception2_cbam' for plot titles."""
    parts = name.split(".")
    # Drop generic suffixes like 'spatial_attention' to keep titles compact
    skip = {"spatial_attention", "channel_attention"}
    parts = [p for p in parts if p not in skip]
    return ".".join(parts) if parts else name


# ──────────────────────────────────────────────
# Per-image figure
# ──────────────────────────────────────────────

def save_figure(
    image_np:    np.ndarray,
    gradcam_map: np.ndarray | None,
    attn_maps:   dict[str, np.ndarray],
    prob:        float,
    label:       int,
    method:      str,
    out_path:    str,
) -> None:
    """Save a single multi-panel visualization figure."""
    outcome = outcome_label(label, prob)

    # Build panel list: (title, base_image_or_None, heatmap_or_None)
    panels = [("Original", image_np, None)]

    if gradcam_map is not None:
        panels.append(("Grad-CAM", image_np, gradcam_map))

    for layer_name, amap in attn_maps.items():
        title = f"Attn: {shorten_layer_name(layer_name)}"
        panels.append((title, None, amap))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]

    label_str = "fake" if label == 1 else "real"
    pred_str  = "fake" if prob > 0.5 else "real"
    fig.suptitle(
        f"Method: {method}  |  Label: {label_str}  |  "
        f"Pred: {pred_str} ({prob:.3f})  |  Outcome: {outcome}",
        fontsize=10,
        y=1.01,
    )

    for ax, (title, img, heatmap) in zip(axes, panels):
        if img is not None and heatmap is not None:
            ax.imshow(overlay_heatmap(img, heatmap))
        elif img is not None:
            ax.imshow(img)
        else:
            # Standalone heatmap (attention map without image base)
            ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1)

        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Grad-CAM and attention maps for a trained deepfake model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest",     type=str, required=True,
                        help="Path to manifest JSON")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model_arch",   type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture key from MODEL_REGISTRY")
    parser.add_argument("--output_dir",   type=str, default="viz_output",
                        help="Directory to save figures")
    parser.add_argument("--n_samples",    type=int, default=20,
                        help="Number of images to visualize")
    parser.add_argument("--mode",         type=str, default="both",
                        choices=["gradcam", "attention", "both"],
                        help="What to visualize")
    parser.add_argument("--target_layer", type=str, default=None,
                        help="Grad-CAM target layer name (overrides GRAD_CAM_LAYERS default). "
                             "Run with --list_layers to see valid names.")
    parser.add_argument("--method",       type=str, default=None,
                        help="Filter manifest to a single generation method")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--list_layers",  action="store_true",
                        help="Print all layer names for this model arch and exit")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_transform = get_eval_transform(args.model_arch)
    denorm_mean, denorm_std = get_denorm_stats(args.model_arch)

    # Load model
    model = MODEL_REGISTRY[args.model_arch]().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded {args.model_arch} from {args.checkpoint}")
    print(f"Device: {device}")

    # --list_layers: just print names and exit
    if args.list_layers:
        print(f"\nLayer names for {args.model_arch}:")
        for name in list_layer_names(model):
            print(f"  {name}")
        return

    # Resolve Grad-CAM target layer
    target_layer_name = args.target_layer or GRAD_CAM_LAYERS.get(args.model_arch)
    if target_layer_name is None:
        target_layer_name = find_last_conv(model)
        print(f"Auto-detected Grad-CAM target layer: {target_layer_name}")
    else:
        print(f"Grad-CAM target layer: {target_layer_name}")

    # Set up Grad-CAM
    gradcam = None
    if args.mode in ("gradcam", "both"):
        gradcam = GradCAM(model, target_layer_name)

    # Load and optionally filter manifest
    with open(args.manifest) as f:
        data = json.load(f)

    if args.method:
        data = [e for e in data if e.get("method", "") == args.method]
        print(f"Filtered to method='{args.method}': {len(data)} samples")

    if not data:
        print("No samples found. Check --method or --manifest.")
        return

    n = min(args.n_samples, len(data))
    samples = random.sample(data, n)
    print(f"Visualizing {n} samples -> {args.output_dir}/\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, entry in enumerate(samples):
        image_path = entry["image_path"]
        label      = int(entry["label"])
        method     = entry.get("method", "unknown")

        image_raw = Image.open(image_path).convert("RGB")
        tensor    = eval_transform(image_raw).unsqueeze(0).to(device)
        image_np  = denormalize(tensor.squeeze(0), denorm_mean, denorm_std)
        img_size  = (tensor.shape[2], tensor.shape[3])

        # Grad-CAM (also runs a forward pass, which populates last_gate)
        gradcam_map = None
        prob        = None
        if gradcam is not None:
            gradcam_map, prob = gradcam.generate(tensor)

        # Attention maps — populated by the Grad-CAM pass above,
        # or by a plain forward pass if mode='attention'
        attn_maps = {}
        if args.mode in ("attention", "both"):
            if gradcam is None:
                # No Grad-CAM pass happened yet; run a plain forward to populate last_gate
                with torch.no_grad():
                    logit = model(tensor)
                prob = torch.sigmoid(logit).item()
            attn_maps = get_attention_maps(model, img_size)

        # Fallback: get prob if neither branch set it (shouldn't happen)
        if prob is None:
            with torch.no_grad():
                logit = model(tensor)
            prob = torch.sigmoid(logit).item()

        # Filename encodes outcome for easy visual browsing of results
        out_name = f"{i:04d}_{method}_{outcome_label(label, prob)}_p{prob:.2f}.png"
        out_path = os.path.join(args.output_dir, out_name)

        save_figure(image_np, gradcam_map, attn_maps, prob, label, method, out_path)
        print(f"  [{i+1:3d}/{n}] {out_name}")

    if gradcam is not None:
        gradcam.remove_hooks()

    print(f"\nDone. {n} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
