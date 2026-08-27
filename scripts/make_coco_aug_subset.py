import os
from pathlib import Path
import argparse

import cv2
import numpy as np
import albumentations as A


def build_transform():
    """
    'Fake target domain' style transform: fog/haze, color, blur, noise.
    You can tweak later to better match your real target.
    """
    return A.Compose([
        # Fog / haze
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6, alpha_coef=0.08, p=0.6),
        # Global color / contrast
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.7),
        # Blur / low visibility
        A.MotionBlur(blur_limit=5, p=0.4),
        A.GaussianBlur(blur_limit=3, p=0.4),
        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    ])


def load_list(list_path: Path):
    with open(list_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def augment_split(src_dir: Path, list_path: Path, dst_dir: Path, transform: A.Compose):
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = load_list(list_path)
    print(f"[INFO] Augmenting {len(files)} images from {src_dir} → {dst_dir}")

    for i, fname in enumerate(files):
        src_path = src_dir / fname
        dst_path = dst_dir / fname

        if not src_path.is_file():
            print(f"[WARN] Missing source image: {src_path}")
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            print(f"[WARN] Failed to read {src_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = transform(image=img)["image"]
        aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst_path), aug_bgr)
        if not ok:
            print(f"[WARN] Failed to write {dst_path}")

        if (i + 1) % 50 == 0:
            print(f"[INFO] {i+1}/{len(files)} done")

    print(f"[DONE] Finished {dst_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_train_imgs", type=str, required=True)
    parser.add_argument("--src_val_imgs", type=str, required=True)
    parser.add_argument("--train_list", type=str, required=True)
    parser.add_argument("--val_list", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    args = parser.parse_args()

    src_train = Path(args.src_train_imgs)
    src_val = Path(args.src_val_imgs)
    train_list = Path(args.train_list)
    val_list = Path(args.val_list)
    out_root = Path(args.out_root)

    train_dst = out_root / "train_imgs"
    val_dst = out_root / "val_imgs"

    transform = build_transform()

    augment_split(src_train, train_list, train_dst, transform)
    augment_split(src_val,   val_list,   val_dst,   transform)


if __name__ == "__main__":
    main()
