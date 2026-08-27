# sam_clip_full/seg/compute_class_weights.py
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

IGNORE_INDEX = 255

def compute_class_weights(lbl_dir: str, num_classes: int, out_path: str):
    lbl_dir = Path(lbl_dir)
    counts = np.zeros(num_classes, dtype=np.float64)

    pngs = sorted([p for p in lbl_dir.glob("*.png")])
    print(f"Scanning {len(pngs)} label PNGs in {lbl_dir}")

    for p in pngs:
        arr = np.array(Image.open(p), dtype=np.int64)
        mask = (arr >= 0) & (arr < num_classes)
        vals, cnts = np.unique(arr[mask], return_counts=True)
        counts[vals] += cnts

    total = counts.sum()
    freq = counts / max(total, 1.0)
    print("Class frequencies:", freq)

    # inverse frequency, clipped
    eps = 1e-6
    inv = 1.0 / (freq + eps)
    inv = inv / inv.mean()  # normalize so mean weight ≈ 1

    print("Class weights (normalized):", inv)
    np.save(out_path, inv)
    print(f"Saved class weights to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbl_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    compute_class_weights(args.lbl_dir, args.num_classes, args.out)
