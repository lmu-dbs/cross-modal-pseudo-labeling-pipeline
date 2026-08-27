#!/usr/bin/env python3
# seg/train_eval_deeplab.py

import os, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# -----------------------------
# Dataset
# -----------------------------
class SemSegFolder(Dataset):
    def __init__(self, img_dir: str, label_dir: str, size: int = 512):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.size = int(size)

        self.imgs = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        self.t = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        fn = self.imgs[i]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")

        # label filename must match image stem
        lab_path = os.path.join(self.label_dir, Path(fn).with_suffix(".png").name)
        lab_pil = Image.open(lab_path)

        # IMPORTANT: keep as PIL for resize; do NOT cast to int64 before Image.fromarray
        lab_pil = lab_pil.resize((self.size, self.size), resample=Image.NEAREST)
        lab = np.array(lab_pil, dtype=np.int64)

        return self.t(img), torch.from_numpy(lab)


# -----------------------------
# Metrics
# -----------------------------
def per_class_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> List[float]:
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        g = (target == c)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(float(inter / union))
    return ious

def pixel_accuracy(pred: np.ndarray, target: np.ndarray, ignore_index: int = 255) -> float:
    mask = (target != ignore_index)
    denom = mask.sum()
    if denom == 0:
        return float("nan")
    return float((pred[mask] == target[mask]).sum() / denom)

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, ignore_index: int = 255) -> Tuple[float, List[float], float]:
    model.eval()
    iou_list = []
    acc_list = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)["out"]  # BxCxhxw
        pred = out.argmax(1).cpu().numpy()
        y_np = y.cpu().numpy()

        for i in range(pred.shape[0]):
            iou_list.append(per_class_iou(pred[i], y_np[i], num_classes))
            acc_list.append(pixel_accuracy(pred[i], y_np[i], ignore_index=ignore_index))

    iou_arr = np.array(iou_list)  # NxC
    class_iou = np.nanmean(iou_arr, axis=0).tolist()
    miou = float(np.nanmean(class_iou))
    pix_acc = float(np.nanmean(np.array(acc_list, dtype=np.float64)))

    return miou, class_iou, pix_acc


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_imgs", required=True)
    ap.add_argument("--train_lbls", required=True)
    ap.add_argument("--val_imgs", required=True)
    ap.add_argument("--val_lbls", required=True)

    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--ignore_label", type=int, default=255)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_ckpt", default="", type=str, help="Optional .pth to initialize model weights")

    # ---- wandb options ----
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default="seg_pseudo", type=str)
    ap.add_argument("--wandb_entity", default="", type=str)
    ap.add_argument("--wandb_run_name", default="", type=str)
    ap.add_argument("--wandb_tags", default="", type=str, help="Comma-separated tags")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])

    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ---- optional wandb ----
    run = None
    if args.wandb and args.wandb_mode != "disabled":
        import wandb
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            tags=tags or None,
            mode=args.wandb_mode,
            config={
                "train_imgs": args.train_imgs,
                "train_lbls": args.train_lbls,
                "val_imgs": args.val_imgs,
                "val_lbls": args.val_lbls,
                "num_classes": args.num_classes,
                "ignore_label": args.ignore_label,
                "epochs": args.epochs,
                "bs": args.bs,
                "size": args.size,
                "lr": args.lr,
                "wd": args.wd,
                "init_ckpt": args.init_ckpt,
            },
        )

    # data
    train_ds = SemSegFolder(args.train_imgs, args.train_lbls, size=args.size)
    val_ds   = SemSegFolder(args.val_imgs, args.val_lbls, size=args.size)

    if len(train_ds) == 0:
        raise ValueError(f"Train dataset is empty (0 images). Check train_imgs={args.train_imgs}")
    if len(val_ds) == 0:
        raise ValueError(f"Val dataset is empty (0 images). Check val_imgs={args.val_imgs}")

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.segmentation.deeplabv3_resnet50(
        num_classes=args.num_classes,
        aux_loss=None,
        weights=None
    ).to(device)

    # ---- optional init from checkpoint ----
    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        sd = torch.load(args.init_ckpt, map_location="cpu")
        # common cases: either raw state_dict OR dict with a key
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Init] Loaded checkpoint: {args.init_ckpt}")
        print(f"[Init] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    else:
        print("[Init] Training from scratch (no --init_ckpt provided or file not found).")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss(ignore_index=int(args.ignore_label))

    best_miou = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)["out"]
            loss = criterion(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            steps += 1

        train_loss = running_loss / max(1, steps)

        miou, class_iou, pix_acc = evaluate(
            model, val_dl, device,
            num_classes=args.num_classes,
            ignore_index=int(args.ignore_label),
        )

        print(f"Epoch {ep}: train_loss={train_loss:.4f} | mIoU={miou:.4f} | pixAcc={pix_acc:.4f}")

        metrics = {
            "epoch": ep,
            "train_loss": train_loss,
            "miou": miou,
            "pixel_acc": pix_acc,
            "class_iou": class_iou,
        }
        with open(os.path.join(args.out_dir, "val_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        if run is not None:
            import wandb
            log_dict = {
                "epoch": ep,
                "train/loss": train_loss,
                "val/mIoU": miou,
                "val/pixel_acc": pix_acc,
            }
            # also log per-class IoU as separate scalars
            for c, v in enumerate(class_iou):
                if v == v:  # not NaN
                    log_dict[f"val/iou_class_{c:02d}"] = float(v)
            wandb.log(log_dict, step=ep)

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_deeplabv3.pth"))

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
