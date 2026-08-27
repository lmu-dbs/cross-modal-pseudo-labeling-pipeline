#!/usr/bin/env python3
import argparse, random, os
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="e.g. /.../shared/DATASET/waste_dataset/iosb")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlink (not recommended for huge images).")
    args = ap.parse_args()

    images_dir = args.root / "images"
    masks_dir  = args.root / "masks"
    assert images_dir.exists(), f"Missing {images_dir}"
    assert masks_dir.exists(),  f"Missing {masks_dir}"

    out_train_imgs = args.root / "train_imgs"
    out_train_lbls = args.root / "train_lbls"
    out_val_imgs   = args.root / "val_imgs"
    out_val_lbls   = args.root / "val_lbls"
    for d in (out_train_imgs, out_train_lbls, out_val_imgs, out_val_lbls):
        d.mkdir(parents=True, exist_ok=True)

    # only keep pairs that exist in both
    img_files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    pairs = []
    for ip in img_files:
        mp = masks_dir / (ip.stem + ".png")
        if mp.exists():
            pairs.append((ip, mp))

    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_val = int(round(n * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def link_or_copy(src: Path, dst: Path):
        if dst.exists():
            return
        if args.copy:
            dst.write_bytes(src.read_bytes())
        else:
            os.symlink(src, dst)

    for ip, mp in train_pairs:
        link_or_copy(ip, out_train_imgs / ip.name)
        link_or_copy(mp, out_train_lbls / mp.name)

    for ip, mp in val_pairs:
        link_or_copy(ip, out_val_imgs / ip.name)
        link_or_copy(mp, out_val_lbls / mp.name)

    print(f"Total pairs: {n}")
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")
    print(f"Wrote folders under: {args.root}")

if __name__ == "__main__":
    main()
