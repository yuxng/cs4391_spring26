#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

IMG_SUFFIX = "-color.jpg"


def read_class_name(folder: Path) -> str:
    f = folder / "name.txt"
    if f.exists():
        txt = f.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return folder.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help='Path to "real_objects" folder')
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--val", type=int, default=2)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="split.json")
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)

    folders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not folders:
        raise RuntimeError(f"No subfolders found under {root}")

    classes = []
    for folder in folders:
        images = sorted([p for p in folder.glob(f"*{IMG_SUFFIX}") if p.is_file()])
        if not images:
            continue
        classes.append(folder)

    # Assign class_id deterministically by sorted folder order
    label_map = {}
    splits = {"train": [], "val": [], "test": []}
    need = args.shots + args.val + args.test

    class_id = 0
    for folder in classes:
        imgs = sorted([p for p in folder.glob(f"*{IMG_SUFFIX}") if p.is_file()])
        if len(imgs) < need:
            continue

        name = read_class_name(folder)
        label_map[str(class_id)] = name

        imgs_rel = [str(p.relative_to(root)) for p in imgs]
        rng.shuffle(imgs_rel)

        train = imgs_rel[: args.shots]
        val = imgs_rel[args.shots : args.shots + args.val]
        test = imgs_rel[args.shots + args.val : args.shots + args.val + args.test]

        for r in train:
            splits["train"].append({"path": r, "y": class_id})
        for r in val:
            splits["val"].append({"path": r, "y": class_id})
        for r in test:
            splits["test"].append({"path": r, "y": class_id})

        class_id += 1

    out = {
        "root": str(root),
        "seed": args.seed,
        "shots": args.shots,
        "val": args.val,
        "test": args.test,
        "num_classes": class_id,
        "label_map": label_map,
        "splits": splits,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path.resolve()}")
    print(f"Classes: {class_id}")
    print(f"Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")


if __name__ == "__main__":
    main()
