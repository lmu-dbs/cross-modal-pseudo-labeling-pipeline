#!/usr/bin/env python3
"""
Train/adapt DeepLab on Lobbe using EVA-CLIP-only pseudo-labels.

This script is a thin, explicit wrapper around:
    python -m seg.seg_main train ...

It starts from the IOSB source DeepLab checkpoint and trains on Lobbe images
supervised by EVA-only pseudo-label masks. Validation uses IOSB validation GT
so that Lobbe GT is not used during adaptation/model selection.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png")))


def parse_args() -> argparse.Namespace:
    project_root = Path("/home/wiss/shubhangi/da-seg/thesis_da_segmentation/sam_clip_full")
    data_root = Path("/home/wiss/shubhangi/home/scs_deal_projects_notapebackup/shared/DATASET/waste_dataset")
    run_root = project_root / "camera_ready_runs"

    ap = argparse.ArgumentParser("Train DeepLab with Lobbe EVA-only pseudo-labels")
    ap.add_argument("--project-root", type=Path, default=project_root)
    ap.add_argument("--data-root", type=Path, default=data_root)
    ap.add_argument("--run-root", type=Path, default=run_root)

    ap.add_argument("--source-ckpt", type=Path, default=run_root / "segmentation/deeplab_iosb_source/best.pth")
    ap.add_argument("--pseudo-lbls", type=Path, default=run_root / "pseudo_labels/lobbe_eva_only/lbls")
    ap.add_argument("--out-dir", type=Path, default=run_root / "segmentation/deeplab_lobbe_eva_only")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--num-classes", type=int, default=9)
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-poly-lr", action="store_true")
    ap.add_argument("--save-preds", action="store_true", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    iosb = args.data_root / "iosb_packform9"
    lobbe = args.data_root / "lobbe_fix"

    train_imgs = lobbe / "val_imgs"
    train_lbls = args.pseudo_lbls
    val_imgs = iosb / "val_imgs"
    val_lbls = iosb / "val_lbls"

    checks = {
        "project_root": args.project_root,
        "data_root": args.data_root,
        "source_ckpt": args.source_ckpt,
        "train_imgs_lobbe": train_imgs,
        "train_lbls_eva_only": train_lbls,
        "val_imgs_iosb": val_imgs,
        "val_lbls_iosb": val_lbls,
    }
    print("[CHECK] Paths", flush=True)
    for name, path in checks.items():
        print(f"  {name}: {path} exists={path.exists()}", flush=True)

    missing = [f"{name}: {path}" for name, path in checks.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    n_imgs = count_pngs(train_imgs)
    n_lbls = count_pngs(train_lbls)
    print(f"[DATA] Lobbe train images: {n_imgs}", flush=True)
    print(f"[DATA] EVA-only pseudo-labels: {n_lbls}", flush=True)
    if n_lbls == 0:
        raise RuntimeError("No EVA-only pseudo-labels found.")
    if n_imgs != n_lbls:
        print("[WARN] Image count and pseudo-label count differ. seg_main will use matched stems.", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "seg.seg_main",
        "train",
        "--train_imgs", str(train_imgs),
        "--train_lbls", str(train_lbls),
        "--val_imgs", str(val_imgs),
        "--val_lbls", str(val_lbls),
        "--num_classes", str(args.num_classes),
        "--ignore_index", str(args.ignore_index),
        "--epochs", str(args.epochs),
        "--bs", str(args.bs),
        "--workers", str(args.workers),
        "--crop_size", str(args.crop_size),
        "--lr", str(args.lr),
        "--wd", str(args.wd),
        "--init_ckpt", str(args.source_ckpt),
        "--out_dir", str(args.out_dir),
    ]

    if not args.no_amp:
        cmd.append("--amp")
    if not args.no_poly_lr:
        cmd.append("--poly_lr")
    if args.save_preds:
        cmd.append("--save_preds")

    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(args.project_root), check=True)
    print("[DONE] EVA-only DeepLab adaptation complete.", flush=True)
    print(f"[DONE] Output: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
