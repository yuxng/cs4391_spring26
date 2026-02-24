#!/usr/bin/env python3
"""
Interactive visualization loop for test predictions.

After closing each figure window, another random batch
of test samples will be shown.

Stop with Ctrl+C.

Example:
  python visualize_test_predictions.py \
      --split split.json \
      --ckpt runs/part2/pretrained_ft/best.pt \
      --num_samples 8
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


# ------------------------------------------------
# Model Loading
# ------------------------------------------------

def load_model(ckpt_path, num_classes, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()
    return model


# ------------------------------------------------
# Denormalization
# ------------------------------------------------

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return torch.clamp(img, 0, 1)


# ------------------------------------------------
# Visualization Block
# ------------------------------------------------

def show_batch(samples, root, model, transform, label_map, device, img_size):

    cols = min(len(samples), 4)
    rows = (len(samples) + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, item in enumerate(samples):
        img_path = root / item["path"]
        y_true = int(item["y"])

        with Image.open(img_path) as im:
            im = im.convert("RGB")

        input_tensor = transform(im).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            y_pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, y_pred].item())

        class_name_true = label_map[y_true]
        class_name_pred = label_map[y_pred]

        img_vis = denormalize(input_tensor.squeeze(0))

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img_vis.permute(1, 2, 0))
        ax.axis("off")

        correct = (y_pred == y_true)
        color = "green" if correct else "red"

        ax.set_title(
            f"GT: {class_name_true}\n"
            f"Pred: {class_name_pred} ({confidence:.2f})",
            color=color,
            fontsize=9
        )

    plt.tight_layout()
    plt.show()   # <-- blocks until window closed


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    meta = json.loads(Path(args.split).read_text(encoding="utf-8"))

    root = Path(meta["root"])
    splits = meta["splits"]
    label_map = {int(k): v for k, v in meta["label_map"].items()}
    num_classes = int(meta["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, num_classes, device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_items = splits["test"]

    print("Press Ctrl+C to stop.\n")

    try:
        while True: 
            samples = random.sample(test_items, args.num_samples)

            show_batch(
                samples,
                root,
                model,
                transform,
                label_map,
                device,
                args.img_size,
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()