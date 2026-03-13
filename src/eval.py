"""
Evaluate a trained deepfake detection model on a manifest split.

NOTE: Ensure you select the *_TRANSFORM for whichever architecture you are evaluating

Usage: from /src
  python eval.py \
    --manifest  ../manifests/FF++/test_frames.json \
    --checkpoint ../checkpoints/path-to-model \
    --model_arch mesoinception4_cbam \
    --output_dir ../evaluation_results/run_name
"""

import argparse
import json
import os
import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score, roc_curve,
)
import matplotlib.pyplot as plt
from collections import defaultdict

from models.meso4 import Meso4
from models.meso_inception_base import MesoInception4
from models.mesoinception_cbam import MesoInception4_CBAM
from models.mesoinception_cbam_experimental import MesoInception4_CBAM_experimental
from models.meso_xception import MesoXceptionNet
from models.meso_xception_cbam import MesoXceptionCBAM
from models.lvnet import LVNetWrapper

MODEL_REGISTRY = {
    # ── From-scratch CNNs ─────────────────────────────────────────────────────
    "meso4":                            Meso4,
    "mesoinception4":                   MesoInception4,
    "mesoinception4_cbam":              MesoInception4_CBAM,
    "mesoinception4_cbam_experimental": MesoInception4_CBAM_experimental,
    "meso_xception":                    MesoXceptionNet,
    "meso_xception_cbam":               MesoXceptionCBAM,
    # ── LVNet two-stream network (Hieu) ───────────────────────────────────────
    "lvnet":               LVNetWrapper,
    # ── Pretrained foundation models ──────────────────────────────────────────
    "xception_pretrained": lambda: timm.create_model('xception', pretrained=False, num_classes=1),
    "vit_b16":             lambda: timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1),
}

# FF++ stats
_FF_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5027, 0.3921, 0.3616],
                         std=[0.2643, 0.2193, 0.2178]),
])
# ImageNet stats
_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# ImageNet stats + resize — for pretrained ViT (requires 224x224)
_VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# LVNet stats + resize — 299×299, normalised to [-1, 1]
_LVNET_TRANSFORM = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

def get_eval_transform(model_arch):
    if model_arch == "vit_b16":
        return _VIT_TRANSFORM
    elif model_arch == "xception_pretrained":
        return _IMAGENET_TRANSFORM
    elif model_arch == "lvnet":
        return _LVNET_TRANSFORM
    else:
        return _FF_TRANSFORM

# Data loader
class DeepfakeDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        with open(manifest_path) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        label = int(entry["label"])
        if self.transform:
            image = self.transform(image)
        method = entry.get("method", "unknown") #for per-method accuracy
        return image, label, method


#inference
@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    all_logits  = []
    all_labels  = []
    all_methods = []

    for images, labels, methods in dataloader:
        images = images.to(device)
        logits = model(images).view(-1)
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_methods.extend(list(methods))

    all_logits  = np.array(all_logits)
    all_labels  = np.array(all_labels, dtype=int)
    all_probs   = 1 / (1 + np.exp(-all_logits))   # sigmoid
    all_preds   = (all_probs > 0.5).astype(int)

    return all_labels, all_preds, all_probs, all_methods



# Metrics
def compute_overall_metrics(all_labels, all_preds, all_probs):
    return {
        "auc":       float(roc_auc_score(all_labels, all_probs)),
        "accuracy":  float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1":        float(f1_score(all_labels, all_preds, zero_division=0)),
        "n_samples": int(len(all_labels)),
        "n_real":    int((all_labels == 0).sum()),
        "n_fake":    int((all_labels == 1).sum()),
    }


def compute_per_method(all_labels, all_preds, all_methods):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for label, pred, method in zip(all_labels, all_preds, all_methods):
        stats[method]["total"] += 1
        if pred == label:
            stats[method]["correct"] += 1
    return {
        method: {
            "accuracy": float(s["correct"] / s["total"]),
            "correct":  int(s["correct"]),
            "total":    int(s["total"]),
        }
        for method, s in sorted(stats.items())
    }

#plots -> ROC/Confusion matrix
def plot_confusion_matrix(all_labels, all_preds, output_dir):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion_matrix.png")

def plot_roc_curve(all_labels, all_probs, auc, output_dir):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved roc_curve.png")

def plot_per_method_accuracy(method_breakdown, output_dir):
    methods = list(method_breakdown.keys())
    accs    = [method_breakdown[m]["accuracy"] for m in methods]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, [a * 100 for a in accs], color="steelblue", edgecolor="black")
    for bar, acc in zip(bars, accs):
        bar.set_color("green" if acc >= 0.8 else "orange" if acc >= 0.6 else "red")

    mean_acc = np.mean(accs)
    plt.axhline(mean_acc * 100, color="red", linestyle="--", label=f"Mean: {mean_acc*100:.1f}%")
    plt.title("Per-Method Accuracy (Test Set)")
    plt.ylabel("Accuracy (%)")
    plt.ylim([0, 105])
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_method_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per_method_accuracy.png")

#Main
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained deepfake detection model")
    parser.add_argument("--manifest",   type=str, required=True,
                        help="Path to manifest JSON (e.g. ../manifests/FF++/test_frames.json)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model_arch", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Load model
    model = MODEL_REGISTRY[args.model_arch]().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # LVNet checkpoints wrap weights under 'state_dict'; plain checkpoints are the dict itself
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    print(f"Loaded {args.model_arch} from {args.checkpoint}")

    #Dataset
    dataset = DeepfakeDataset(args.manifest, transform=get_eval_transform(args.model_arch))
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Evaluating on {len(dataset)} samples from {args.manifest}")

    #Run
    all_labels, all_preds, all_probs, all_methods = run_inference(model, loader, device)

    #Metrics
    metrics          = compute_overall_metrics(all_labels, all_preds, all_probs)
    method_breakdown = compute_per_method(all_labels, all_preds, all_methods)

    #Print
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples:   {metrics['n_samples']}  (real: {metrics['n_real']}, fake: {metrics['n_fake']})")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print("\nPer-method accuracy:")
    for method, s in method_breakdown.items():
        print(f"  {method:20s}: {s['accuracy']:.4f}  ({s['correct']}/{s['total']})")
    print("=" * 60)

    #Plots
    plot_confusion_matrix(all_labels, all_preds, args.output_dir)
    plot_roc_curve(all_labels, all_probs, metrics["auc"], args.output_dir)
    plot_per_method_accuracy(method_breakdown, args.output_dir)

    #Save JSON
    results = {
        "checkpoint":       args.checkpoint,
        "model_arch":       args.model_arch,
        "manifest":         args.manifest,
        "metrics":          metrics,
        "per_method":       method_breakdown,
    }
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
