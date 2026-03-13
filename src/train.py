"""
Loss: BCEWithLogits
Optimizer: AdamW
LR Scheduler: ReduceLROnPlateau

Usage: from /src
  python train.py \
    --train_manifest <manifests/FF++/train_frames.json> \
    --val_manifest   <manifests/FF++/val_frames.json> \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --output_dir <checkpoints/run_name> \
    --model mesoinception4_cbam

Data:
The processed data must be located in:
  - absolute: ProjectRoot/data
  - relative (used in train.py): ../data

Available models (--model flag):
  meso4                            Meso4 baseline (Hieu)
  mesoinception4                   MesoInception4 (rypow)
  mesoinception4_cbam              MesoInception4 + CBAM (rypow)  [default/best]
  mesoinception4_cbam_experimental Wider CBAM variant (rypow)
  meso_xception                    MesoXceptionNet (TJ)
  meso_xception_cbam               MesoXceptionNet + CBAM (TJ)
  lvnet                            LVNet two-stream network (Hieu) [299×299, [-1,1] norm]
  xception_pretrained              Pretrained Xception (timm)
  vit_b16                          Pretrained ViT-B/16 (timm)

Pretrained models:
- the pretrained foundational models expect ImageNet stats --> uncomment the ImageNet stats when finetuning
- ViT requires 224x224 ---> uncomment transforms.Resize(224)

ViT fine-tuning recommended settings (very different from CNN defaults):
  --lr 1e-4              (default 1e-3 is too high, destroys pretrained attention weights)
  --weight_decay 0.1     (default 1e-5 is too low for an 86M param model)
  --warmup_epochs 5      (ramp LR from near-zero to --lr over first N epochs)
  --patience 15          (ViT converges slower than CNNs, give it more time)

Augmentation:
    -Transforms will be different depending on the model being used:
        -For ViT: uncomment ViT section. Main difference = input requirements (224) + ImageNet Stats
        -For Xception: Uncomment Xception section. Requires ImageNet Stats
    
    Augmentation Strategy:
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    #     transforms.Lambda(lambda img: jpeg_compress(img, quality=random.randint(70, 95))),
    #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    #     transforms.ToTensor(),
    #     transforms.Normalize( TODO: depends on which model)
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
from collections import defaultdict
import io
import random
from models.meso4 import Meso4
from models.meso_inception_base import MesoInception4
from models.mesoinception_cbam import MesoInception4_CBAM
from models.mesoinception_cbam_experimental import MesoInception4_CBAM_experimental
from models.meso_xception import MesoXceptionNet
from models.meso_xception_cbam import MesoXceptionCBAM
from models.lvnet import LVNetWrapper

import timm  # for pretrained foundation models

# Model registry — to add a new model: import above, then add it here.
MODEL_REGISTRY = {
    # ── From-scratch CNNs ─────────────────────────────────────────────────────
    "meso4":                            Meso4,
    "mesoinception4":                   MesoInception4,
    "mesoinception4_cbam":              MesoInception4_CBAM,
    "mesoinception4_cbam_experimental": MesoInception4_CBAM_experimental,
    "meso_xception":                    MesoXceptionNet,
    "meso_xception_cbam":               MesoXceptionCBAM,
    # ── LVNet two-stream network (Hieu) ───────────────────────────────────────
    # Note: full LVNet training requires segmentation labels (see extension/prepare_data.py).
    # This entry allows fine-tuning the classification head only via the shared pipeline.
    "lvnet":               LVNetWrapper,
    # ── Pretrained foundation models (fine-tuned) ─────────────────────────────
    "xception_pretrained": lambda: timm.create_model('xception', pretrained=True, num_classes=1),
    "vit_b16":             lambda: timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1),
}

# ──────────────────
#      dataloader
# ──────────────────
# Reads a json manifest with columns: image_path, label, and method (forgery)
# label: 0 = real, 1 = fake

class DeepfakeDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        with open(manifest_path) as f:
            self.data = json.load(f) #list of {image_path, label}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        entry = self.data[idx]
        #label = 0 if entry["label"] == "real" else 1
        #gets frame and converts to RGB -> should already be in RGB but added as fail safe
        image = Image.open(entry["image_path"]).convert("RGB")
        method = entry.get("method", "unknown")
        label = int(entry["label"])

        if self.transform:
            image = self.transform(image)

        return image, label, method

#========================
#TRAINING LOOP
#=================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device).float()

        #Forward pass
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Track loss, correct and total predictions
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ──────────────────────────────────────────────
# Validation (per-epoch) & Final Evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    all_logits = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device).float()

        logits = model(images).view(-1) #finds dimension itself
        loss = criterion(logits, labels)

        losses.append(loss.item())
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy() #shape N
    all_labels = torch.cat(all_labels).numpy()#shape [N] values of 0 or 1

    #Overall metrics
    probs = 1 / (1 + np.exp(-all_logits)) # sigmoid - raw logits to binary
    preds = (probs > 0.5).astype(int) #0.5 threshold
    y_true = all_labels.astype(int)

    overall = {
        "loss": float(np.mean(losses)),
        "auc": float(roc_auc_score(y_true, probs)),
        "acc": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    return overall

@torch.no_grad()
def final_evaluation(model, dataloader, device, output_dir):
    """Run one pass, collect every prediction, plot confusion matrix."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_methods = []

    for images, labels, methods in dataloader:
        images = images.to(device)
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits) #sigmoid transformation
        preds = (probs > 0.5).int() #threshold probabilities
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_methods.extend(list(methods))
        
    #Final precision/recall/f1
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"FINAL EVAL | P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}")

    #Per-method accuracy
    method_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, method in enumerate(all_methods):
        method_stats[method]["total"] += 1
        if all_preds[i] == all_labels[i]:
            method_stats[method]["correct"] += 1

    print("Per-method accuracy:")
    for method, stats in sorted(method_stats.items()):
        acc = stats["correct"] / stats["total"]
        print(f"    {method:20s}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    #Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()

    #Build/plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Validation Set)")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

# ───────────────────────────────
#                   MAIN
# ───────────────────────────────

#takes image, compresses it using quality arg, reloads compressed version and returns
#simulates jpeg compression artifacts
def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to training CSV/json (columns: image_path, label, method)")
    parser.add_argument("--val_manifest", type=str, required=True,
                        help="Path to validation CSV (columns: image_path, label, method)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear LR warmup epochs (recommended: 5 for ViT fine-tuning)")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()),help="the model must first be added into MODEL_REGISTRY")
    args = parser.parse_args()

    #create output directory for best_model.pth and evaluation
    os.makedirs(args.output_dir, exist_ok=True)

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────────
    # To switch models: uncomment the matching block, comment out the active block.

    # FF++ stats (from-scratch models: mesoinception, mesoinception_cbam, xception)
    # train_transform = transforms.Compose([
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    #     transforms.Lambda(lambda img: jpeg_compress(img, quality=random.randint(70, 95))),
    #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5027, 0.3921, 0.3616], std=[0.2643, 0.2193, 0.2178]),
    # ])
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5027, 0.3921, 0.3616], std=[0.2643, 0.2193, 0.2178]),
    # ])

    # ViT (vit_b16) — ImageNet stats, resize to 224
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.Lambda(lambda img: jpeg_compress(img, quality=random.randint(70, 95))),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Xception pretrained (xception_pretrained) — ImageNet stats, 256 input ok (uses GlobalAvgPool)
    # train_transform = transforms.Compose([
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    #     transforms.Lambda(lambda img: jpeg_compress(img, quality=random.randint(70, 95))),
    #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    #Dataloaders
    train_dataset = DeepfakeDataset(args.train_manifest, transform=train_transform)
    val_dataset = DeepfakeDataset(args.val_manifest, transform=val_transform)

    #added persistent workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    #Model init
    model = MODEL_REGISTRY[args.model]().to(device)

    #Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    #LR scheduler
    # warmup_scheduler ramps LR from 1% -> 100% over warmup_epochs (no-op if warmup_epochs=0)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=max(args.warmup_epochs, 1))
    main_scheduler   = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    #Training results tracker(s)
    best_val_auc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_aucs = []

    #early stop trigger
    patience = 10
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        #train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        #test model on validation set + collect metrics
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_auc = val_metrics["auc"]
        val_precision = val_metrics["precision"]
        val_recall = val_metrics["recall"]
        val_f1 = val_metrics["f1"]

        #store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)

        #adjust LR
        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()          # ramp up during warmup
        else:
            main_scheduler.step(val_auc)     # reduce on plateau after warmup

        #stop timer
        elapsed = time.time() - start

        #Print epoch metrics
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")
        print(f"R: {val_recall:.4f} | F1: {val_f1:.4f} | P: {val_precision:.4f} | Train_ACC: {train_acc:.4f} | val_ACC: {val_acc:.4f}")
        
        #Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"------> Saved best model (val_auc={val_auc:.4f})")
            patience_counter = 0 #reset for improvements
        #save checkpoint (every 10 epoch)
        elif epoch % 10 == 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save(model.state_dict(), path)
            print(f"-----------> saved checkpoint (epoch={epoch})")
            patience_counter += 1
        #If no progress, increase patience counter
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    #Save final model
    path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), path)
    print(f"\nTRAINING COMPLETE: Best val AUC: {best_val_auc:.4f}")


    #loss Curve plot
    plt.figure(figsize=(10, 5)) #creates a blank canvas, 10 inches wide by 5 tall
    plt.plot(train_losses, label="Train Loss") #draws a line using loss values per epoch
    plt.plot(val_losses, label="Val Loss") #draws line for val loss per epoch
    plt.xlabel("Epoch") 
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend() 
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    #Accuracy Curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "acc_curve.png"), dpi=150)
    plt.close()

    #AUC Curve 
    plt.figure(figsize=(10, 5))
    plt.plot(val_aucs, label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "auc_curve.png"), dpi=150)
    plt.close()

    #FINAL EVALUATION: with best model on validation set
    best_path = os.path.join(args.output_dir, "best_model.pth") #loads best model before final eval
    model.load_state_dict(torch.load(best_path, map_location=device))
    final_evaluation(model, val_loader, device, args.output_dir)
    print("Plots saved to", args.output_dir)

if __name__ == "__main__":
    main()
