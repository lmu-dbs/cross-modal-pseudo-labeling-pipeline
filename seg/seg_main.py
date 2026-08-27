#!/usr/bin/env python3
"""
seg_main.py — DeepLabV3-ResNet50 only, robust joint resize + joint crop trainer/evaluator.

Key design goals:
- Works for GTA5/Cityscapes-style UDA pretraining (source supervised) and general folder datasets.
- Joint transforms: resize/scale/crop/flip applied identically to image & label.
- Label PNG is assumed to already be trainIds in [0..K-1] with ignore_index (default 255).
  (For GTA5 you typically map labels to Cityscapes trainId beforehand.)

Typical DA preprocessing:
- Source (GTA5) often resized to 1280x720; target (Cityscapes) to 1024x512 or 512x1024; then random crop 512. (You can set via CLI.)
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from torchvision.models.segmentation import deeplabv3_resnet50


# -----------------------
# Utilities
# -----------------------

def estimate_class_weights(lbl_dir: str, num_classes: int, ignore_index: int = 255, max_files: int = 2000):
    from pathlib import Path
    import numpy as np
    from PIL import Image

    lbl_dir = Path(lbl_dir)
    files = sorted(lbl_dir.glob("*.png"))
    if max_files is not None:
        files = files[:max_files]

    counts = np.zeros((num_classes,), dtype=np.float64)
    for p in files:
        y = np.array(Image.open(p), dtype=np.int64).reshape(-1)
        y = y[(y != ignore_index) & (y >= 0) & (y < num_classes)]
        if y.size:
            binc = np.bincount(y, minlength=num_classes).astype(np.float64)
            counts += binc

    # inverse frequency with smoothing, then normalize
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_hw(arg_list: Optional[List[int]]) -> Optional[Tuple[int, int]]:
    if arg_list is None:
        return None
    if len(arg_list) != 2:
        raise ValueError("Expected two ints for H W")
    h, w = int(arg_list[0]), int(arg_list[1])
    if h <= 0 or w <= 0:
        raise ValueError("H and W must be > 0")
    return (h, w)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _safe_to_uint8_mask(arr: np.ndarray) -> np.ndarray:
    # for saving preds; assumes classes < 256
    return arr.astype(np.uint8)

def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    """
    pred/target: [H,W] int64 tensors on CPU
    """
    k = (
        (target != ignore_index)
        & (target >= 0)
        & (target < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )
    if k.sum().item() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long)
    return torch.bincount(
        target[k] * num_classes + pred[k],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).to(torch.long)

def metrics_from_cm(cm: torch.Tensor) -> Dict[str, Any]:
    eps = 1e-8
    cm = cm.to(torch.float64)
    diag = torch.diag(cm)
    sum_row = cm.sum(dim=1)
    sum_col = cm.sum(dim=0)
    union = sum_row + sum_col - diag

    iou = (diag + eps) / (union + eps)
    dice = (2 * diag + eps) / (sum_row + sum_col + eps)

    miou = float(torch.nanmean(iou).item())
    mdice = float(torch.nanmean(dice).item())
    pixacc = float(((diag.sum() + eps) / (cm.sum() + eps)).item())

    return {
        "miou": miou,
        "mdice": mdice,
        "pixel_accuracy": pixacc,
        "class_iou": [None if torch.isnan(v) else float(v.item()) for v in iou],
        "class_dice": [None if torch.isnan(v) else float(v.item()) for v in dice],
        "confusion_matrix": cm.to(torch.int64).tolist(),
    }

def build_deeplabv3_r50(num_classes: int) -> nn.Module:
    """
    Robust to torchvision API differences:
    - Newer: deeplabv3_resnet50(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2, num_classes=K)
    - Older: deeplabv3_resnet50(pretrained=False, pretrained_backbone=True, num_classes=K)
    """
    from inspect import signature

    sig = signature(deeplabv3_resnet50)
    kwargs = {}

    # always start from scratch head
    if "weights" in sig.parameters:
        kwargs["weights"] = None
    if "pretrained" in sig.parameters:
        kwargs["pretrained"] = False

    # backbone init
    if "weights_backbone" in sig.parameters:
        # torchvision>=0.13 style
        try:
            from torchvision.models import ResNet50_Weights
            kwargs["weights_backbone"] = ResNet50_Weights.IMAGENET1K_V2
        except Exception:
            # fallback: sometimes strings are accepted, but if not, user can upgrade torchvision
            kwargs["weights_backbone"] = None
    elif "pretrained_backbone" in sig.parameters:
        kwargs["pretrained_backbone"] = True

    # num_classes
    kwargs["num_classes"] = int(num_classes)

    model = deeplabv3_resnet50(**kwargs)
    return model


# -----------------------
# Dataset
# -----------------------
class SemSegFolder(Dataset):
    """
    Paired (img, label) dataset from folders.
    - images: .png/.jpg/.jpeg
    - labels: .png with same stem
    """

    def __init__(
        self,
        img_dir: str,
        lbl_dir: str,
        crop_size: int = 512,
        augment: bool = False,
        ignore_index: int = 255,
        resize_hw: Optional[Tuple[int, int]] = None,     # fixed resize (H,W) before crop
        scale_min: float = 1.0,                          # random scale jitter (multiscale)
        scale_max: float = 1.0,
        hflip_p: float = 0.5,
        color_jitter: float = 0.0,                       # 0 disables; else strength
    ):
        self.img_dir = str(img_dir)
        self.lbl_dir = str(lbl_dir)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        self.ignore_index = int(ignore_index)

        self.resize_hw = resize_hw
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        assert self.scale_min > 0 and self.scale_max > 0 and self.scale_max >= self.scale_min

        self.hflip_p = float(hflip_p)
        self.color_jitter_strength = float(color_jitter)

        self.imgs = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

        if self.color_jitter_strength > 0:
            s = self.color_jitter_strength
            self.jitter = transforms.ColorJitter(
                brightness=0.2 * s,
                contrast=0.2 * s,
                saturation=0.2 * s,
                hue=0.05 * s,
            )
        else:
            self.jitter = None

    def __len__(self):
        return len(self.imgs)

    def _load_pair(self, fname: str):
        img_path = os.path.join(self.img_dir, fname)
        lbl_path = os.path.join(self.lbl_dir, Path(fname).with_suffix(".png").name)

        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path)  # keep as indexed
        return img, lbl

    def _joint_resize(self, img: Image.Image, lbl: Image.Image, hw: Tuple[int, int]):
        h, w = hw
        img = TF.resize(img, (h, w), interpolation=InterpolationMode.BILINEAR)
        lbl = TF.resize(lbl, (h, w), interpolation=InterpolationMode.NEAREST)
        return img, lbl

    def _joint_random_scale(self, img: Image.Image, lbl: Image.Image):
        if (not self.augment) or (self.scale_min == 1.0 and self.scale_max == 1.0):
            return img, lbl
        s = float(torch.empty(1).uniform_(self.scale_min, self.scale_max).item())
        w, h = img.size
        nh = max(1, int(round(h * s)))
        nw = max(1, int(round(w * s)))
        img = TF.resize(img, (nh, nw), interpolation=InterpolationMode.BILINEAR)
        lbl = TF.resize(lbl, (nh, nw), interpolation=InterpolationMode.NEAREST)
        return img, lbl

    def _joint_pad_if_needed(self, img: Image.Image, lbl: Image.Image):
        cs = self.crop_size
        w, h = img.size
        pad_w = max(cs - w, 0)
        pad_h = max(cs - h, 0)
        if pad_w > 0 or pad_h > 0:
            img = TF.pad(img, padding=(0, 0, pad_w, pad_h), fill=0)
            lbl = TF.pad(lbl, padding=(0, 0, pad_w, pad_h), fill=self.ignore_index)
        return img, lbl

    def _joint_crop(self, img: Image.Image, lbl: Image.Image):
        cs = self.crop_size
        w, h = img.size

        if not self.augment:
            top = max((h - cs) // 2, 0)
            left = max((w - cs) // 2, 0)
            img = TF.crop(img, top, left, cs, cs)
            lbl = TF.crop(lbl, top, left, cs, cs)
            return img, lbl

    # foreground-aware random crop
        tries = 10
        best = None
        best_fg = -1.0

        for _ in range(tries):
            top, left, th, tw = transforms.RandomCrop.get_params(img, output_size=(cs, cs))
            lbl_crop = TF.crop(lbl, top, left, th, tw)
            y = np.array(lbl_crop, dtype=np.int64)

        # fg = non-background and non-ignore
            fg = (y != 0) & (y != self.ignore_index)
            fg_frac = float(fg.mean())

            if fg_frac > best_fg:
                best_fg = fg_frac
                best = (top, left, th, tw)

            if fg_frac >= 0.05:   # 5% foreground is enough
                best = (top, left, th, tw)
                break

        top, left, th, tw = best
        img = TF.crop(img, top, left, th, tw)
        lbl = TF.crop(lbl, top, left, th, tw)
        return img, lbl


    def _joint_hflip(self, img: Image.Image, lbl: Image.Image):
        if self.augment and (torch.rand(1).item() < self.hflip_p):
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)
        return img, lbl

    def __getitem__(self, idx: int):
        fname = self.imgs[idx]
        img, lbl = self._load_pair(fname)

        # 1) fixed resize (if requested) — useful for GTA5/Cityscapes protocols
        if self.resize_hw is not None:
            img, lbl = self._joint_resize(img, lbl, self.resize_hw)

        # 2) random scale jitter (multiscale), optional
        img, lbl = self._joint_random_scale(img, lbl)

        # 3) pad then joint crop
        img, lbl = self._joint_pad_if_needed(img, lbl)
        img, lbl = self._joint_crop(img, lbl)

        # 4) joint flip
        img, lbl = self._joint_hflip(img, lbl)

        # 5) image-only jitter
        if self.jitter is not None and self.augment:
            img = self.jitter(img)

        x = self.normalize(self.to_tensor(img))
        y_np = np.array(lbl, dtype=np.int64)

# Keep IOSB labels as-is: 0..8 (background included).
# Only preserve ignore pixels (255) if they exist (e.g., from padding).
        ignore = self.ignore_index
        y_np = np.where(y_np == ignore, ignore, y_np)

        y = torch.from_numpy(y_np).long()
        return x, y, fname


# optional safety assert (remove later)
        if not np.all((y_np == ignore) | ((y_np >= 0) & (y_np < 8))):
            bad = np.unique(y_np[~((y_np == ignore) | ((y_np >= 0) & (y_np < 8)))])
            raise ValueError(f"Bad label values after remap: {bad}")


        y = torch.from_numpy(y_np).long()

        return x, y, fname


# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    max_norm: float = 0.0,
    lr_sched: Optional[callable] = None,
):
    model.train()
    running = 0.0
    n = 0

    for step, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x)["out"]
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)["out"]
            loss = criterion(logits, y)
            loss.backward()
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if lr_sched is not None:
            lr_sched()

        running += float(loss.item()) * x.size(0)
        n += int(x.size(0))

    return running / max(n, 1)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    if save_dir is not None:
        ensure_dir(Path(save_dir))

    for x, y, names in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)["out"]
        pred = logits.argmax(dim=1).cpu()
        y = y.cpu()

        for i in range(pred.size(0)):
            cm += confusion_matrix(pred[i], y[i], num_classes=num_classes, ignore_index=ignore_index)
            if save_dir is not None:
                out_path = Path(save_dir) / Path(names[i]).with_suffix(".png").name
                Image.fromarray(_safe_to_uint8_mask(pred[i].numpy())).save(out_path)

    return metrics_from_cm(cm)

@torch.no_grad()
def eval_from_folders(pred_dir: str, gt_dir: str, num_classes: int, ignore_index: int) -> Dict[str, Any]:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pr_files = {p.name: p for p in pred_dir.glob("*.png")}
    gt_files = {p.name: p for p in gt_dir.glob("*.png")}
    common = sorted(set(pr_files.keys()) & set(gt_files.keys()))
    print(f"common files: {len(common)}")

    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for fn in common:
        pr = np.array(Image.open(pr_files[fn]), dtype=np.int64)
        gt = np.array(Image.open(gt_files[fn]), dtype=np.int64)

        if pr.shape != gt.shape:
            pr = np.array(Image.fromarray(pr).resize((gt.shape[1], gt.shape[0]), resample=Image.NEAREST), dtype=np.int64)

        pr_t = torch.from_numpy(pr).long()
        gt_t = torch.from_numpy(gt).long()
        cm += confusion_matrix(pr_t, gt_t, num_classes=num_classes, ignore_index=ignore_index)

    return metrics_from_cm(cm)


# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser("DeepLabV3-R50 segmentation trainer/evaluator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train DeepLabV3-R50 on a folder dataset")
    tr.add_argument("--train_imgs", required=True)
    tr.add_argument("--train_lbls", required=True)
    tr.add_argument("--val_imgs", required=True)
    tr.add_argument("--val_lbls", required=True)

    tr.add_argument("--num_classes", type=int, required=True)
    tr.add_argument("--ignore_index", type=int, default=255)

    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--bs", type=int, default=4)
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--device", type=str, default="cuda")

    tr.add_argument("--crop_size", type=int, default=512)

    # fixed resize before crop (joint)
    tr.add_argument("--train_resize_hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="Fixed resize (H W) before crop for TRAIN (e.g., 720 1280 for GTA5)")
    tr.add_argument("--val_resize_hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="Fixed resize (H W) before crop for VAL (e.g., 512 1024 for Cityscapes)")

    # random scale jitter (multiscale) before crop
    tr.add_argument("--scale_min", type=float, default=1.0)
    tr.add_argument("--scale_max", type=float, default=1.0)

    tr.add_argument("--hflip_p", type=float, default=0.5)
    tr.add_argument("--color_jitter", type=float, default=0.0, help="0 disables; try 1.0 for mild jitter")

    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--wd", type=float, default=1e-2)
    tr.add_argument("--max_norm", type=float, default=0.0)

    tr.add_argument("--amp", action="store_true", help="use torch.cuda.amp")
    tr.add_argument("--poly_lr", action="store_true", help="poly LR schedule over total steps (common for seg)")
    tr.add_argument("--poly_power", type=float, default=0.9)

    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--save_preds", action="store_true")
    tr.add_argument("--resume", type=str, default="")
    tr.add_argument("--init_ckpt", type=str, default="")

    tr.add_argument("--seed", type=int, default=42)

    ev = sub.add_parser("eval", help="Compute metrics from predicted vs GT folders")
    ev.add_argument("--pred_dir", required=True)
    ev.add_argument("--gt_dir", required=True)
    ev.add_argument("--num_classes", type=int, required=True)
    ev.add_argument("--ignore_index", type=int, default=255)
    ev.add_argument("--out_json", type=str, default="")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.cmd == "eval":
        metrics = eval_from_folders(args.pred_dir, args.gt_dir, args.num_classes, args.ignore_index)
        print(json.dumps({k: (v if isinstance(v, (float, int)) else None) for k, v in metrics.items()}, indent=2))
        if args.out_json:
            save_json(Path(args.out_json), metrics)
        return 0

    # -------- train --------
    set_seed(args.seed)

    device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")
    print(f"Using device: {device}")

    train_resize_hw = parse_hw(args.train_resize_hw)
    val_resize_hw   = parse_hw(args.val_resize_hw)

    train_ds = SemSegFolder(
        args.train_imgs, args.train_lbls,
        crop_size=args.crop_size,
        augment=True,
        ignore_index=args.ignore_index,
        resize_hw=train_resize_hw,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        hflip_p=args.hflip_p,
        color_jitter=args.color_jitter,
    )

    val_ds = SemSegFolder(
        args.val_imgs, args.val_lbls,
        crop_size=args.crop_size,
        augment=False,
        ignore_index=args.ignore_index,
        resize_hw=val_resize_hw,
        scale_min=1.0,
        scale_max=1.0,
        hflip_p=0.0,
        color_jitter=0.0,
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    model = build_deeplabv3_r50(args.num_classes).to(device)

    # init / resume
    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        sd = torch.load(args.init_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        print(f"Loaded init checkpoint: {args.init_ckpt}")

    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from: {args.resume}")

    class_w = estimate_class_weights(args.train_lbls, args.num_classes, args.ignore_index, max_files=2000).to(device)
    print("class_weights:", class_w.detach().cpu().numpy().round(3).tolist())

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index, weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    best_path = out_dir / "best.pth"

    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # poly LR schedule over total steps (optional)
    total_steps = args.epochs * max(len(train_dl), 1)
    step_idx = 0

    def poly_step():
        nonlocal step_idx
        step_idx += 1
        t = min(step_idx / max(total_steps, 1), 1.0)
        lr = args.lr * ((1.0 - t) ** args.poly_power)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    lr_sched = poly_step if args.poly_lr else None

    best_miou = -1.0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_dl, device, criterion, optimizer,
            scaler=scaler, max_norm=args.max_norm, lr_sched=lr_sched
        )

        metrics = evaluate(
            model, val_dl, device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            save_dir=str(out_dir / "preds") if args.save_preds else None
        )

        print(
            f"Epoch {ep:02d} | loss {tr_loss:.4f} | "
            f"mIoU {metrics['miou']:.4f} | mDice {metrics['mdice']:.4f} | "
            f"pixAcc {metrics['pixel_accuracy']:.4f}"
        )
        print("class_iou:", [round(x, 4) if x is not None else None for x in metrics["class_iou"]])

        save_json(out_dir / "val_metrics.json", {"epoch": ep, "train_loss": tr_loss, **metrics})

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save(model.state_dict(), best_path)

    print(f"Best mIoU: {best_miou:.4f} @ {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# How to run it for GTA5 (10k subset) “DA-style”
#Log: /home/scs_deal_projects_notapebackup/user/shubhang/thesis/seg_runs/gta5_dlv3r50_10k/train.log

# Use fixed resize + random crop:

# python seg_main.py train \
#   --train_imgs /path/gta5_10k/train_imgs \
#   --train_lbls /path/gta5_10k/train_lbls \
#   --val_imgs   /path/gta5_val/imgs \
#   --val_lbls   /path/gta5_val/lbls \
#   --num_classes 19 \
#   --crop_size 512 \
#   --train_resize_hw 720 1280 \
#   --val_resize_hw   720 1280 \
#   --hflip_p 0.5 \
#   --color_jitter 1.0 \
#   --amp \
#   --poly_lr \
#   --epochs 20 \
#   --bs 4 \
#   --out_dir /path/seg_runs/gta5_10k_dlv3r50


# Epoch 01 | loss 1.5860 | mIoU 0.0760 | mDice 0.1103 | pixAcc 0.4758
# Epoch 02 | loss 1.5839 | mIoU 0.0586 | mDice 0.0798 | pixAcc 0.4689
# Epoch 03 | loss nan | mIoU 0.0588 | mDice 0.0801 | pixAcc 0.4686
# Epoch 04 | loss 1.4706 | mIoU 0.0584 | mDice 0.0796 | pixAcc 0.4669