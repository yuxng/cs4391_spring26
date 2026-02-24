#!/usr/bin/env python3
"""
ResNet-18 inference from either:
  1) Live USB camera, or
  2) Saved image files.

Examples:
  # Live camera
  python infer_cam_or_images.py --split split.json --ckpt best.pt --source cam --cam_id 0

  # Folder of images
  python infer_cam_or_images.py --split split.json --ckpt best.pt --source images --images_dir saved_frames

  # Explicit list of images
  python infer_cam_or_images.py --split split.json --ckpt best.pt --source images --images_list a.jpg b.jpg c.png

Keys (live cam):
  q / ESC : quit
  s       : save current frame (if --save_dir set)

Keys (images mode):
  n / SPACE : next image
  p         : previous image
  q / ESC   : quit
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ------------------------------------------------
# Model
# ------------------------------------------------
def load_model(ckpt_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()
    return model


# ------------------------------------------------
# Utils
# ------------------------------------------------
def put_text(img_bgr: np.ndarray, text: str, org=(10, 30), scale=0.8, thickness=2):
    # black outline
    cv2.putText(
        img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    # white text
    cv2.putText(
        img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA
    )


def infer_on_bgr(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    label_map: dict,
    show_probs: bool = False,
) -> Tuple[int, str, float, Optional[List[Tuple[str, float]]]]:
    """Run inference on a BGR image and return (pred_id, pred_name, conf, topk_list)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)

    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())

    pred_name = label_map.get(pred_id, str(pred_id))

    topk_list = None
    if show_probs:
        topk = min(5, probs.numel())
        vals, idxs = torch.topk(probs, k=topk)
        topk_list = []
        for i in range(topk):
            cid = int(idxs[i].item())
            name = label_map.get(cid, str(cid))
            p = float(vals[i].item())
            topk_list.append((name, p))

    return pred_id, pred_name, conf, topk_list


def list_images_in_dir(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    paths = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


# ------------------------------------------------
# Modes
# ------------------------------------------------
def run_live_cam(
    args,
    model,
    transform,
    device,
    label_map,
):
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera cam_id={args.cam_id}. Try --cam_id 1,2,...")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    last_infer_t = 0.0
    ema_fps = None
    prev_t = time.time()

    print("Live cam started. Keys: q/ESC quit, s save frame (if --save_dir set)")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Warning: failed to read frame.")
                continue

            now = time.time()

            # Optional inference FPS limiting
            if args.max_fps > 0:
                min_dt = 1.0 / float(args.max_fps)
                if now - last_infer_t < min_dt:
                    cv2.imshow("Inference", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s") and save_dir is not None:
                        out = save_dir / f"frame_{int(time.time()*1000)}.jpg"
                        cv2.imwrite(str(out), frame_bgr)
                        print(f"Saved {out}")
                    continue

            last_infer_t = now

            pred_id, pred_name, conf, topk_list = infer_on_bgr(
                frame_bgr, model, transform, device, label_map, show_probs=args.show_probs
            )

            # FPS display
            dt = now - prev_t
            prev_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            put_text(frame_bgr, f"Pred: {pred_name} ({conf:.2f})", (10, 30), 0.8, 2)
            put_text(frame_bgr, f"FPS: {ema_fps:.1f}", (10, 60), 0.7, 2)

            if topk_list is not None:
                for i, (name, p) in enumerate(topk_list):
                    put_text(frame_bgr, f"{i+1}) {name}: {p:.2f}", (10, 95 + 25 * i), 0.6, 1)

            cv2.imshow("Inference", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and save_dir is not None:
                out = save_dir / f"frame_{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(out), frame_bgr)
                print(f"Saved {out}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_images(
    args,
    model,
    transform,
    device,
    label_map,
):
    # Build list of images
    img_paths: List[Path] = []
    if args.images_list:
        img_paths = [Path(p) for p in args.images_list]
    elif args.images_dir:
        img_paths = list_images_in_dir(Path(args.images_dir))
    else:
        raise ValueError("Images mode requires --images_dir or --images_list")

    img_paths = [p for p in img_paths if p.exists()]
    if not img_paths:
        raise RuntimeError("No valid image files found.")

    idx = 0
    print("Images mode started. Keys: n/SPACE next, p prev, q/ESC quit")

    while True:
        p = img_paths[idx]
        frame_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"Warning: failed to read {p}")
            # skip forward
            idx = (idx + 1) % len(img_paths)
            continue

        pred_id, pred_name, conf, topk_list = infer_on_bgr(
            frame_bgr, model, transform, device, label_map, show_probs=args.show_probs
        )

        disp = frame_bgr.copy()
        put_text(disp, f"{p.name}", (10, 30), 0.7, 2)
        put_text(disp, f"Pred: {pred_name} ({conf:.2f})", (10, 60), 0.8, 2)

        if topk_list is not None:
            for i, (name, pr) in enumerate(topk_list):
                put_text(disp, f"{i+1}) {name}: {pr:.2f}", (10, 95 + 25 * i), 0.6, 1)

        cv2.imshow("Inference", disp)
        key = cv2.waitKey(0) & 0xFF  # wait for keypress per image

        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord(" "), 83):  # 'n' or space (83 can be right arrow on some builds)
            idx = (idx + 1) % len(img_paths)
        elif key in (ord("p"), 81):  # 'p' (81 can be left arrow on some builds)
            idx = (idx - 1) % len(img_paths)
        else:
            # default: next
            idx = (idx + 1) % len(img_paths)

    cv2.destroyAllWindows()


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--source", type=str, choices=["cam", "images"], default="cam")

    # cam args
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_fps", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="")

    # images args
    parser.add_argument("--images_dir", type=str, default="")
    parser.add_argument("--images_list", type=str, nargs="*", default=[])

    parser.add_argument("--show_probs", action="store_true")
    args = parser.parse_args()

    meta = json.loads(Path(args.split).read_text(encoding="utf-8"))
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

    if args.source == "cam":
        run_live_cam(args, model, transform, device, label_map)
    else:
        run_images(args, model, transform, device, label_map)


if __name__ == "__main__":
    main()
