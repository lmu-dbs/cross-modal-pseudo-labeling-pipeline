#!/usr/bin/env python3
"""
evaluate_deeplab_lobbe.py

Evaluate DeepLab checkpoints on Lobbe ground-truth labels and save a paper-ready
JSON + CSV comparison table.

Default evaluation set:
  images: /home/wiss/shubhangi/home/scs_deal_projects_notapebackup/shared/DATASET/waste_dataset/lobbe_fix/val_imgs
  labels: /home/wiss/shubhangi/home/scs_deal_projects_notapebackup/shared/DATASET/waste_dataset/lobbe_fix/val_lbls_form9

Default checkpoints:
  source_only  -> camera_ready_runs/segmentation/deeplab_iosb_source/best.pth
  eva_only     -> camera_ready_runs/segmentation/deeplab_lobbe_eva_only/best.pth
  eva_blip     -> camera_ready_runs/segmentation/deeplab_lobbe_eva_blip/best.pth

Outputs:
  camera_ready_runs/final_metrics/lobbe_deeplab_comparison.json
  camera_ready_runs/final_metrics/lobbe_deeplab_comparison.csv
  camera_ready_runs/final_metrics/preds/<model_name>/*.png  (if --save-preds)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


CLASS_NAMES = [
    "background",
    "bottle",
    "bag_film",
    "cup_tray",
    "lid_cap",
    "carton",
    "can",
    "foam",
    "other_packaging",
]


def build_deeplab(num_classes: int) -> nn.Module:
    """Build DeepLabV3-ResNet50 without downloading pretrained weights."""
    from inspect import signature

    sig = signature(deeplabv3_resnet50)
    kwargs: Dict[str, Any] = {"num_classes": int(num_classes)}
    if "weights" in sig.parameters:
        kwargs["weights"] = None
    if "weights_backbone" in sig.parameters:
        kwargs["weights_backbone"] = None
    if "pretrained" in sig.parameters:
        kwargs["pretrained"] = False
    if "pretrained_backbone" in sig.parameters:
        kwargs["pretrained_backbone"] = False
    return deeplabv3_resnet50(**kwargs)


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing = model.load_state_dict(sd, strict=False)
    print(f"[CKPT] Loaded {ckpt_path}", flush=True)
    print(f"[CKPT] load_state_dict result: {missing}", flush=True)


def list_common_files(img_dir: Path, gt_dir: Path) -> List[str]:
    img_files = {p.stem: p for p in img_dir.glob("*.png")}
    img_files.update({p.stem: p for p in img_dir.glob("*.jpg")})
    img_files.update({p.stem: p for p in img_dir.glob("*.jpeg")})
    gt_files = {p.stem: p for p in gt_dir.glob("*.png")}
    common = sorted(set(img_files) & set(gt_files))
    return common


def find_image_path(img_dir: Path, stem: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image found for stem: {stem}")


def resize_max_side(img: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    if max_side <= 0:
        return img, 1.0
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img, 1.0
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR), scale


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    pred = pred.astype(np.int64)
    gt = gt.astype(np.int64)
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    if not np.any(valid):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = gt[valid] * num_classes + pred[valid]
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.astype(np.int64)


def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    cm = cm.astype(np.float64)
    diag = np.diag(cm)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    union = row + col - diag

    eps = 1e-8
    iou = (diag + eps) / (union + eps)
    dice = (2 * diag + eps) / (row + col + eps)
    pixel_accuracy = float((diag.sum() + eps) / (cm.sum() + eps))

    return {
        "miou": float(np.mean(iou)),
        "mdice": float(np.mean(dice)),
        "pixel_accuracy": pixel_accuracy,
        "class_iou": [float(x) for x in iou],
        "class_dice": [float(x) for x in dice],
        "confusion_matrix": cm.astype(np.int64).tolist(),
    }


def flatten_for_csv(model_name: str, ckpt_path: Path, metrics: Dict[str, Any], num_images: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "num_images": num_images,
        "miou": metrics["miou"],
        "mdice": metrics["mdice"],
        "pixel_accuracy": metrics["pixel_accuracy"],
    }
    for i, cname in enumerate(CLASS_NAMES):
        row[f"iou_{cname}"] = metrics["class_iou"][i]
        row[f"dice_{cname}"] = metrics["class_dice"][i]
    return row


@torch.no_grad()
def evaluate_checkpoint(
    model_name: str,
    ckpt_path: Path,
    img_dir: Path,
    gt_dir: Path,
    out_pred_dir: Path,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    max_side: int,
    save_preds: bool,
) -> Dict[str, Any]:
    print(f"\n[EVAL] {model_name}", flush=True)
    print(f"[EVAL] checkpoint: {ckpt_path}", flush=True)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {ckpt_path}")

    model = build_deeplab(num_classes).to(device)
    load_checkpoint(model, ckpt_path)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    stems = list_common_files(img_dir, gt_dir)
    print(f"[EVAL] common image/GT pairs: {len(stems)}", flush=True)
    if len(stems) == 0:
        raise RuntimeError("No common image/GT pairs found.")

    if save_preds:
        out_pred_dir.mkdir(parents=True, exist_ok=True)

    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for stem in tqdm(stems, desc=f"eval:{model_name}"):
        img_path = find_image_path(img_dir, stem)
        gt_path = gt_dir / f"{stem}.png"

        img = Image.open(img_path).convert("RGB")
        gt = np.array(Image.open(gt_path), dtype=np.int64)
        gt_h, gt_w = gt.shape[:2]

        img_eval, _ = resize_max_side(img, max_side=max_side)
        x = normalize(to_tensor(img_eval)).unsqueeze(0).to(device)

        logits = model(x)["out"]
        # Bring logits back to GT resolution before argmax.
        logits = F.interpolate(logits, size=(gt_h, gt_w), mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        total_cm += confusion_matrix(pred, gt, num_classes=num_classes, ignore_index=ignore_index)

        if save_preds:
            Image.fromarray(pred).save(out_pred_dir / f"{stem}.png")

    metrics = metrics_from_cm(total_cm)
    metrics["model_name"] = model_name
    metrics["checkpoint_path"] = str(ckpt_path)
    metrics["num_images"] = len(stems)
    metrics["class_names"] = CLASS_NAMES
    return metrics


def parse_args() -> argparse.Namespace:
    project_root = Path("/home/wiss/shubhangi/da-seg/thesis_da_segmentation/sam_clip_full")
    data_root = Path("/home/wiss/shubhangi/home/scs_deal_projects_notapebackup/shared/DATASET/waste_dataset")
    run_root = project_root / "camera_ready_runs"

    ap = argparse.ArgumentParser("Evaluate DeepLab checkpoints on Lobbe GT")
    ap.add_argument("--project-root", type=Path, default=project_root)
    ap.add_argument("--data-root", type=Path, default=data_root)
    ap.add_argument("--run-root", type=Path, default=run_root)
    ap.add_argument("--img-dir", type=Path, default=data_root / "lobbe_fix/val_imgs")
    ap.add_argument("--gt-dir", type=Path, default=data_root / "lobbe_fix/val_lbls_form9")
    ap.add_argument("--out-dir", type=Path, default=run_root / "final_metrics")

    ap.add_argument("--source-ckpt", type=Path, default=run_root / "segmentation/deeplab_iosb_source/best.pth")
    ap.add_argument("--eva-only-ckpt", type=Path, default=run_root / "segmentation/deeplab_lobbe_eva_only/best.pth")
    ap.add_argument("--eva-blip-ckpt", type=Path, default=run_root / "segmentation/deeplab_lobbe_eva_blip/best.pth")

    ap.add_argument("--num-classes", type=int, default=9)
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--max-side", type=int, default=1024, help="Resize long image side for inference; 0 = full resolution.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-preds", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    print("[DEVICE]", device, flush=True)
    print("[PATH] img_dir:", args.img_dir, "exists=", args.img_dir.exists(), flush=True)
    print("[PATH] gt_dir:", args.gt_dir, "exists=", args.gt_dir.exists(), flush=True)
    print("[PATH] out_dir:", args.out_dir, flush=True)

    if not args.img_dir.exists():
        raise FileNotFoundError(args.img_dir)
    if not args.gt_dir.exists():
        raise FileNotFoundError(args.gt_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = args.out_dir / "preds"

    ckpts = [
        ("source_only", args.source_ckpt),
        ("eva_clip_only", args.eva_only_ckpt),
        ("eva_clip_blip", args.eva_blip_ckpt),
    ]

    all_metrics: Dict[str, Any] = {
        "evaluation_dataset": "lobbe_fix/val_imgs + lobbe_fix/val_lbls_form9",
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "class_names": CLASS_NAMES,
        "max_side": args.max_side,
        "models": {},
    }
    csv_rows: List[Dict[str, Any]] = []

    for model_name, ckpt_path in ckpts:
        metrics = evaluate_checkpoint(
            model_name=model_name,
            ckpt_path=ckpt_path,
            img_dir=args.img_dir,
            gt_dir=args.gt_dir,
            out_pred_dir=pred_root / model_name,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            max_side=args.max_side,
            save_preds=args.save_preds,
        )
        all_metrics["models"][model_name] = metrics
        csv_rows.append(flatten_for_csv(model_name, ckpt_path, metrics, metrics["num_images"]))
        print(
            f"[RESULT] {model_name}: "
            f"mIoU={metrics['miou']:.4f}, "
            f"mDice={metrics['mdice']:.4f}, "
            f"pixAcc={metrics['pixel_accuracy']:.4f}",
            flush=True,
        )

    json_path = args.out_dir / "lobbe_deeplab_comparison.json"
    csv_path = args.out_dir / "lobbe_deeplab_comparison.csv"

    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n[DONE] Saved:", flush=True)
    print(json_path, flush=True)
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
