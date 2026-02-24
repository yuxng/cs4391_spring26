#!/usr/bin/env python3
"""
HW Part 2 — Train From Scratch vs Pretrained (with plots)

Runs training and saves:
  - predictions_val.csv
  - predictions_test.csv
  - training_loss.png
  - val_accuracy.png
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import matplotlib.pyplot as plt  # ---- NEW ----


# -------------------------
# Dataset
# -------------------------

class SplitDataset(Dataset):
    def __init__(self, root: Path, items, transform=None):
        self.root = root
        self.items = items  # list of {"path": "...", "y": int}
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r = self.items[idx]
        img_path = self.root / r["path"]
        y = int(r["y"])
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, y, r["path"]


# -------------------------
# Models
# -------------------------

class SmallCNN(nn.Module):
    """
    TODO:
    Implement a small CNN for image classification.

    Requirements:
      - Input: (B, 3, H, W)
      - Use at least 3 convolution layers
      - Use ReLU activations
      - Use MaxPool2d for downsampling
      - Output shape: (B, num_classes)

    Hint:
      If input size is 224x224 and you use 3 MaxPool(2),
      spatial size becomes:
        224 → 112 → 56 → 28
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # TODO: define convolutional layers
        # Example structure:
        # Conv → ReLU → MaxPool
        # Conv → ReLU → MaxPool
        # Conv → ReLU → MaxPool
        #
        # Store them in self.features

        raise NotImplementedError

        # TODO:
        # After convolution layers, reduce spatial dimension.
        # You may use:
        #   nn.AdaptiveAvgPool2d((1,1))
        #
        # Then define a Linear classifier.

    def forward(self, x):
        # TODO:
        # 1. Pass through feature extractor
        # 2. Pool
        # 3. Flatten
        # 4. Classify

        raise NotImplementedError


def build_model(num_classes: int, mode: str, use_imagenet_pretrained: bool, freeze_backbone: bool):
    if mode == "scratch":
        return SmallCNN(num_classes)

    if mode == "pretrained":
        weights = models.ResNet18_Weights.DEFAULT if use_imagenet_pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

        if freeze_backbone:
            for name, p in m.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False
        return m

    raise ValueError(f"Unknown mode: {mode}")


# -------------------------
# Train / Eval
# -------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total = 0.0, 0
    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

    return total_loss / max(1, total)


@torch.no_grad()
def eval_acc_and_dump(model, loader, device, out_csv: Path):
    model.eval()
    correct, total = 0, 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "y_true", "y_pred"])

        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

            pred_cpu = pred.cpu().tolist()
            y_cpu = y.cpu().tolist()
            for p, yt, yp in zip(paths, y_cpu, pred_cpu):
                w.writerow([p, yt, yp])

    return correct / max(1, total)


def save_curves(out_dir: Path, train_losses, val_accs):
    """
    Save a single figure with:
      - Training Loss (blue, left y-axis)
      - Validation Accuracy (red, right y-axis)
      - Legend included
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots()

    # ---- Training Loss (Left axis) ----
    line1, = ax1.plot(
        epochs,
        train_losses,
        color="blue",
        linewidth=2,
        label="Training Loss"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # ---- Validation Accuracy (Right axis) ----
    ax2 = ax1.twinx()
    line2, = ax2.plot(
        epochs,
        val_accs,
        color="red",
        linewidth=2,
        label="Validation Accuracy"
    )
    ax2.set_ylabel("Validation Accuracy", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # ---- Title ----
    plt.title("Training Loss and Validation Accuracy")

    # ---- Legend (combine both axes) ----
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, required=True, help="Path to split.json")

    ap.add_argument("--model", type=str, choices=["scratch", "pretrained"], required=True)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet weights (only for --model pretrained)")
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze ResNet backbone, train fc only")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--tag", type=str, default="", help="Extra tag for output folder name (optional)")
    ap.add_argument("--out_root", type=str, default="runs/part2")
    args = ap.parse_args()

    meta = json.loads(Path(args.split).read_text(encoding="utf-8"))
    root = Path(meta["root"])
    num_classes = int(meta["num_classes"])
    splits = meta["splits"]
    label_map = meta["label_map"]

    tag = args.tag if args.tag else f"{args.model}" + ("_imagenet" if args.pretrained else "")
    if args.freeze_backbone:
        tag += "_freeze"
    out_dir = Path(args.out_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "split_used.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = SplitDataset(root, splits["train"], transform=train_tf)
    val_ds = SplitDataset(root, splits["val"], transform=test_tf)
    test_ds = SplitDataset(root, splits["test"], transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        num_classes=num_classes,
        mode=args.model,
        use_imagenet_pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    # ---- NEW ----
    train_losses = []
    val_accs = []

    print(f"== Part 2: {tag} ==")
    print(f"Classes: {num_classes}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Device: {device}")
    print(f"Out: {out_dir}")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_csv = out_dir / "predictions_val.csv"
        val_acc = eval_acc_and_dump(model, val_loader, device, val_csv)

        # ---- NEW ----
        train_losses.append(loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs}  loss={loss:.4f}  val_acc={val_acc*100:.2f}%")

        torch.save({"model": model.state_dict(), "args": vars(args), "meta": meta}, last_path)
        if args.save_best and val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "meta": meta}, best_path)

        # ---- NEW ---- (update plots each epoch, so students can watch progress)
        save_curves(out_dir, train_losses, val_accs)

    # Evaluate test (use best if requested)
    if args.save_best and best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    test_csv = out_dir / "predictions_test.csv"
    test_acc = eval_acc_and_dump(model, test_loader, device, test_csv)
    print(f"TEST accuracy: {test_acc*100:.2f}%")
    print(f"Wrote: {test_csv}")
    print(f"Saved plot: {out_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
