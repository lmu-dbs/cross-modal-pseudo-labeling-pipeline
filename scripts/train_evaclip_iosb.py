#!/usr/bin/env python3
"""
Train an IOSB fine-tuned EVA-CLIP checkpoint for the waste-sorting pipeline.

This script is extracted and cleaned from the IOSB section of
COCO_SAM_EVA_FineTune_and_Pipeline_v4.ipynb.

It trains EVA-CLIP on IOSB packaging-form labels 1..8, excluding background.
The saved checkpoint is intended for the later Lobbe pseudo-label generation
pipeline, where the EVA-CLIP classifier supervises SAM proposals.

Default output:
  <project-root>/camera_ready_runs/checkpoints/eva_clip_ft_iosb_packform9_FULL.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import open_clip


IOSB_CLASSES_9 = [
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
TRAIN_CLASSES_8 = IOSB_CLASSES_9[1:]

IOSB_SYNONYMS = {
    "bottle": [
        "a plastic bottle", "a PET bottle", "a water bottle", "a soda bottle"
    ],
    "bag_film": [
        "a plastic bag", "plastic film", "shrink wrap", "wrapping film", "plastic wrapper"
    ],
    "cup_tray": [
        "a plastic cup", "a food tray", "a plastic tray", "a takeaway container",
        "a food container", "a clamshell container"
    ],
    "lid_cap": [
        "a bottle cap", "a plastic cap", "a lid", "a container lid"
    ],
    "carton": [
        "a beverage carton", "a milk carton", "a juice carton", "a liquid carton (Tetra Pak)"
    ],
    "can": [
        "a metal can", "an aluminum can", "a soda can", "a tin can"
    ],
    "foam": [
        "foam packaging", "styrofoam", "expanded polystyrene foam", "foam tray"
    ],
    "other_packaging": [
        "other packaging", "miscellaneous packaging", "unknown packaging item"
    ],
}

IOSB_TEMPLATES = [
    "a waste sorting image containing {}",
    "an industrial waste stream showing {}",
    "a conveyor-belt scene with {}",
    "a waste item: {}",
    "an image of {} in a waste sorting setting",
    "a piece of {} packaging on a conveyor belt",
]


def parse_args() -> argparse.Namespace:
    default_project = Path.home() / "da-seg/thesis_da_segmentation/sam_clip_full"
    default_iosb = Path(
        "/home/wiss/shubhangi/home/scs_deal_projects_notapebackup/"
        "shared/DATASET/waste_dataset/iosb_packform9"
    )

    p = argparse.ArgumentParser(description="Fine-tune EVA-CLIP on IOSB waste crops.")
    p.add_argument("--project-root", type=Path, default=default_project)
    p.add_argument("--iosb-root", type=Path, default=default_iosb)
    p.add_argument("--img-dir-name", type=str, default="images")
    p.add_argument("--lbl-dir-name", type=str, default="masks")
    p.add_argument("--ckpt-name", type=str, default="eva_clip_ft_iosb_packform9_FULL.pt")

    p.add_argument("--model-name", type=str, default="EVA02-L-14")
    p.add_argument("--pretrained", type=str, default="merged2b_s4b_b131k")

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-bs", type=int, default=1)
    p.add_argument("--val-bs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--ctx-pad", type=float, default=0.40)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.2)

    p.add_argument("--max-train-samples", type=int, default=120_000)
    p.add_argument("--max-val-samples", type=int, default=20_000)
    p.add_argument("--fast-dev-run", action="store_true", help="Use tiny sample limits and one epoch.")

    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--samples-per-class-per-image", type=int, default=2)
    p.add_argument("--min-dom", type=float, default=0.55)
    p.add_argument("--min-pixels-per-class", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-rebuild-csv", action="store_true", help="Reuse existing CSVs instead of rebuilding them.")
    p.add_argument("--allow-cpu", action="store_true", help="Allow CPU training. Not recommended.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def list_common_stems(img_dir: Path, lbl_dir: Path, exts=(".png", ".jpg", ".jpeg")):
    imgs = {}
    for e in exts:
        for p in img_dir.glob(f"*{e}"):
            imgs[p.stem] = p
    lbls = {p.stem: p for p in lbl_dir.glob("*.png")}
    common = sorted(set(imgs.keys()) & set(lbls.keys()))
    return common, imgs, lbls


def clip_bbox(x0, y0, x1, y1, W, H):
    x0 = max(0, min(int(x0), W - 1))
    y0 = max(0, min(int(y0), H - 1))
    x1 = max(1, min(int(x1), W))
    y1 = max(1, min(int(y1), H))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def build_iosb_evaclip_csv(
    img_dir: Path,
    lbl_dir: Path,
    out_train_csv: Path,
    out_val_csv: Path,
    max_images: Optional[int] = None,
    val_frac: float = 0.10,
    samples_per_class_per_image: int = 2,
    seed: int = 42,
    crop_sizes=(96, 128, 160, 192),
    min_dom: float = 0.55,
    min_pixels_per_class: int = 500,
):
    rng = np.random.default_rng(seed)
    common, imgs_map, lbls_map = list_common_stems(img_dir, lbl_dir)
    print(f"[CSV] Common image/label pairs: {len(common)}", flush=True)

    if max_images is not None and len(common) > max_images:
        common = rng.choice(common, size=max_images, replace=False).tolist()
        common = sorted(common)

    rng.shuffle(common)
    n_val = int(round(len(common) * val_frac))
    val_stems = set(common[:n_val])

    rows_train, rows_val = [], []

    for k, stem in enumerate(common, start=1):
        if k % 200 == 0:
            print(f"[CSV] processed {k}/{len(common)} images", flush=True)

        img_path = imgs_map[stem]
        lbl_path = lbls_map[stem]
        gt = np.array(Image.open(lbl_path), dtype=np.int64)
        H, W = gt.shape[:2]

        present = [int(c) for c in np.unique(gt) if 1 <= int(c) <= 8]
        if not present:
            continue

        for cid in present:
            ys, xs = np.where(gt == cid)
            if len(xs) < min_pixels_per_class:
                continue

            for _ in range(samples_per_class_per_image):
                ok = False
                for _try in range(30):
                    j = rng.integers(0, len(xs))
                    cx, cy = int(xs[j]), int(ys[j])
                    s = int(rng.choice(crop_sizes))

                    bb = clip_bbox(cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2, W, H)
                    if bb is None:
                        continue
                    x0, y0, x1, y1 = bb

                    patch = gt[y0:y1, x0:x1]
                    dom = float((patch == cid).mean())
                    if dom >= min_dom:
                        ok = True
                        break

                if not ok:
                    continue

                row = {
                    "img_path": str(img_path),
                    "x1": int(x0), "y1": int(y0), "x2": int(x1), "y2": int(y1),
                    "class_id": int(cid - 1),
                    "class_name": TRAIN_CLASSES_8[int(cid - 1)],
                }
                if stem in val_stems:
                    rows_val.append(row)
                else:
                    rows_train.append(row)

    df_tr = pd.DataFrame(rows_train)
    df_va = pd.DataFrame(rows_val)
    out_train_csv.parent.mkdir(parents=True, exist_ok=True)
    df_tr.to_csv(out_train_csv, index=False)
    df_va.to_csv(out_val_csv, index=False)

    print(f"[CSV] Wrote {out_train_csv} rows={len(df_tr)}", flush=True)
    print(f"[CSV] Wrote {out_val_csv} rows={len(df_va)}", flush=True)
    if len(df_tr):
        print("[CSV] Train counts:\n", df_tr["class_name"].value_counts(), flush=True)
    if len(df_va):
        print("[CSV] Val counts:\n", df_va["class_name"].value_counts(), flush=True)


def pad_bbox_xyxy(xmin, ymin, xmax, ymax, pad_ratio, W, H):
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    pad = int(round(pad_ratio * max(bw, bh)))
    x0 = max(0, xmin - pad)
    y0 = max(0, ymin - pad)
    x1 = min(W - 1, xmax + pad)
    y1 = min(H - 1, ymax + pad)
    return x0, y0, x1, y1


class SegCropWithGlobalDataset(Dataset):
    """CSV columns: img_path, x1, y1, x2, y2, class_id, class_name."""

    def __init__(self, csv_path, crop_preprocess, global_preprocess=None, max_samples=None, ctx_pad=0.40):
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            raise ValueError(f"CSV has no rows: {csv_path}")
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        self.df = df
        self.crop_preprocess = crop_preprocess
        self.global_preprocess = global_preprocess if global_preprocess is not None else crop_preprocess
        self.ctx_pad = ctx_pad

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]
        img = Image.open(row["img_path"]).convert("RGB")
        W, H = img.size
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])

        crop_pil = img.crop((x1, y1, x2, y2))
        cx0, cy0, cx1, cy1 = pad_bbox_xyxy(x1, y1, x2, y2, self.ctx_pad, W, H)
        ctx_pil = img.crop((cx0, cy0, cx1, cy1))

        crop_t = self.crop_preprocess(crop_pil)
        ctx_t = self.global_preprocess(ctx_pil)
        return crop_t, ctx_t, int(row["class_id"]), row["class_name"]


def build_text_embeds_from_synonyms(model, class_names, synonyms_dict, templates, tokenizer, device, pool="mean"):
    model.eval()
    phrase_counts = {}
    class_vecs = []
    with torch.no_grad():
        for cname in class_names:
            phrases = synonyms_dict.get(cname) or [cname]
            phrase_counts[cname] = len(phrases)
            prompts = []
            for ph in phrases:
                prompts.extend([t.format(ph) for t in templates])
            tok = tokenizer(prompts).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                z = model.encode_text(tok)
                z = F.normalize(z, dim=-1)
            if pool == "max":
                z_cls = z.max(dim=0, keepdim=True).values
            else:
                z_cls = z.mean(dim=0, keepdim=True)
            z_cls = F.normalize(z_cls, dim=-1)
            class_vecs.append(z_cls)
        text_embeds = torch.cat(class_vecs, dim=0)
        text_embeds = F.normalize(text_embeds, dim=-1)
        text_feat_mean = F.normalize(text_embeds.mean(dim=0, keepdim=True), dim=-1)
    model.train()
    return text_embeds, text_feat_mean, phrase_counts


def disable_all_grad_checkpointing(model):
    """Best-effort disabling of checkpointing across open_clip/EVA variants."""
    if hasattr(model, "visual") and hasattr(model.visual, "trunk"):
        trunk = model.visual.trunk
        if hasattr(trunk, "set_grad_checkpointing"):
            try:
                trunk.set_grad_checkpointing(False)
            except TypeError:
                trunk.set_grad_checkpointing(enable=False)
        if hasattr(trunk, "grad_checkpointing"):
            trunk.grad_checkpointing = False
        for m in trunk.modules():
            for attr in ("use_checkpoint", "grad_checkpointing", "checkpoint", "use_ckpt"):
                if hasattr(m, attr):
                    try:
                        setattr(m, attr, False)
                    except Exception:
                        pass
    if hasattr(model, "visual"):
        for m in model.visual.modules():
            for attr in ("use_checkpoint", "grad_checkpointing", "checkpoint", "use_ckpt"):
                if hasattr(m, attr):
                    try:
                        setattr(m, attr, False)
                    except Exception:
                        pass
            if hasattr(m, "set_grad_checkpointing"):
                try:
                    m.set_grad_checkpointing(False)
                except TypeError:
                    m.set_grad_checkpointing(enable=False)
    print("[OK] Disabled grad checkpointing (best-effort).", flush=True)


@torch.no_grad()
def encode_global_cpu_batch_to_feats(model, global_t_cpu, device):
    feats = []
    for i in range(global_t_cpu.size(0)):
        g_i = global_t_cpu[i : i + 1].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = model.encode_image(g_i)
            f = F.normalize(f, dim=-1)
        feats.append(f.to(torch.float16 if device == "cuda" else torch.float32))
        del g_i
    return torch.cat(feats, dim=0)


@torch.no_grad()
def eval_top1_top5(model, loader, text_embeds, text_feat_mean, device):
    model.eval()
    top1 = top5 = total = 0
    for crop_t, global_t, y, _ in loader:
        crop_t = crop_t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            obj_feat = model.encode_image(crop_t)
            obj_feat = F.normalize(obj_feat, dim=-1)
        g_feat = encode_global_cpu_batch_to_feats(model, global_t, device).to(obj_feat.dtype)

        sim_og = (obj_feat * g_feat).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid(8.0 * (sim_og - 0.2))
        beta_dyn = 0.4 * gate
        scene_fused = F.normalize(obj_feat + beta_dyn * (g_feat - obj_feat), dim=-1)

        logits0 = scene_fused @ text_embeds.T
        conf = logits0.softmax(dim=-1).max(dim=-1, keepdim=True).values
        alpha_dyn = 0.2 * (1.0 - conf)

        text_mean_batch = text_feat_mean.to(scene_fused.device).expand_as(scene_fused)
        fused_feat = F.normalize(scene_fused + alpha_dyn * (text_mean_batch - scene_fused), dim=-1)
        logits = fused_feat @ text_embeds.T

        top1 += (logits.argmax(dim=-1) == y).sum().item()
        pred5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        top5 += (pred5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    model.train()
    return top1 / max(1, total), top5 / max(1, total)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_torch()

    if args.fast_dev_run:
        args.epochs = 1
        args.max_train_samples = min(args.max_train_samples, 200)
        args.max_val_samples = min(args.max_val_samples, 100)
        args.workers = min(args.workers, 2)
        print("[FAST_DEV_RUN] epochs=1 max_train=200 max_val=100", flush=True)

    project_root = args.project_root
    iosb_root = args.iosb_root
    img_dir = iosb_root / args.img_dir_name
    lbl_dir = iosb_root / args.lbl_dir_name
    csv_train = iosb_root / "iosb_evaclip_train.csv"
    csv_val = iosb_root / "iosb_evaclip_val.csv"

    run_root = project_root / "camera_ready_runs"
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / args.ckpt_name
    metrics_path = ckpt_dir / args.ckpt_name.replace(".pt", "_metrics.json")

    for name, p in {
        "project_root": project_root,
        "iosb_root": iosb_root,
        "img_dir": img_dir,
        "lbl_dir": lbl_dir,
        "ckpt_dir": ckpt_dir,
    }.items():
        print(f"[PATH] {name}: {p} exists={p.exists()}", flush=True)
    assert img_dir.exists(), img_dir
    assert lbl_dir.exists(), lbl_dir

    print("[TORCH]", torch.__version__, "cuda_build=", torch.version.cuda, flush=True)
    print("[CUDA] available=", torch.cuda.is_available(), "count=", torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Use a Slurm GPU job or pass --allow-cpu for a tiny smoke test.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.no_rebuild_csv or not (csv_train.exists() and csv_val.exists()):
        build_iosb_evaclip_csv(
            img_dir,
            lbl_dir,
            csv_train,
            csv_val,
            max_images=None,
            val_frac=args.val_frac,
            samples_per_class_per_image=args.samples_per_class_per_image,
            seed=args.seed,
            crop_sizes=(96, 128, 160, 192),
            min_dom=args.min_dom,
            min_pixels_per_class=args.min_pixels_per_class,
        )
    else:
        print(f"[CSV] Reusing existing CSVs: {csv_train}, {csv_val}", flush=True)

    print(f"[MODEL] Loading {args.model_name} pretrained={args.pretrained}", flush=True)
    clip_model, _, _preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    clip_model = clip_model.to(device)
    disable_all_grad_checkpointing(clip_model)

    print("[TEXT] Building text embeddings", flush=True)
    text_embeds, text_feat_mean, phrase_counts = build_text_embeds_from_synonyms(
        clip_model,
        class_names=TRAIN_CLASSES_8,
        synonyms_dict=IOSB_SYNONYMS,
        templates=IOSB_TEMPLATES,
        tokenizer=tokenizer,
        device=device,
        pool="mean",
    )
    print("[TEXT] text_embeds", tuple(text_embeds.shape), "phrase_counts", phrase_counts, flush=True)

    img_size = args.img_size
    train_preprocess = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_preprocess = train_preprocess

    train_ds = SegCropWithGlobalDataset(
        csv_train,
        train_preprocess,
        max_samples=args.max_train_samples,
        ctx_pad=args.ctx_pad,
    )
    val_ds = SegCropWithGlobalDataset(
        csv_val,
        val_preprocess,
        max_samples=args.max_val_samples,
        ctx_pad=args.ctx_pad,
    )
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)}", flush=True)

    pin_memory = device == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=args.train_bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.val_bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(args.workers > 0),
    )

    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_val_top1 = -1.0
    history = []

    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print("[TRAIN] Starting", flush=True)
    for epoch in range(1, args.epochs + 1):
        clip_model.train()
        running = 0.0
        n = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (crop_t, global_t, y, _) in enumerate(train_dl):
            crop_t = crop_t.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                obj_feat = clip_model.encode_image(crop_t)
                obj_feat = F.normalize(obj_feat, dim=-1)

            with torch.no_grad():
                g_feat = encode_global_cpu_batch_to_feats(clip_model, global_t, device).to(obj_feat.dtype)

            scene_fused = F.normalize(obj_feat + args.beta * (g_feat - obj_feat), dim=-1)
            text_mean_batch = text_feat_mean.to(scene_fused.device).expand_as(scene_fused)
            fused_feat = F.normalize(scene_fused + args.alpha * (text_mean_batch - scene_fused), dim=-1)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = fused_feat @ text_embeds.T
                loss = F.cross_entropy(logits, y) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += (loss.item() * args.grad_accum) * crop_t.size(0)
            n += crop_t.size(0)

            if (step + 1) % 100 == 0:
                print(f"[TRAIN] epoch={epoch} step={step+1}/{len(train_dl)} loss={running/max(1,n):.4f}", flush=True)
            if device == "cuda" and (step + 1) % 200 == 0:
                torch.cuda.empty_cache()

        if (step + 1) % args.grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = running / max(1, n)
        val_top1, val_top5 = eval_top1_top5(clip_model, val_dl, text_embeds, text_feat_mean, device)
        row = {"epoch": epoch, "train_loss": avg_loss, "val_top1": val_top1, "val_top5": val_top5}
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss {avg_loss:.4f} | "
            f"val_top1 {val_top1*100:.2f}% | val_top5 {val_top5*100:.2f}%",
            flush=True,
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(
                {
                    "model_state_dict": clip_model.state_dict(),
                    "epoch": epoch,
                    "val_top1": float(best_val_top1),
                    "classes": TRAIN_CLASSES_8,
                    "iosb_classes_9": IOSB_CLASSES_9,
                    "synonyms": IOSB_SYNONYMS,
                    "templates": IOSB_TEMPLATES,
                    "text_pool": "mean",
                    "alpha": float(args.alpha),
                    "beta": float(args.beta),
                    "train_csv": str(csv_train),
                    "val_csv": str(csv_val),
                    "img_sz": int(img_size),
                    "train_bs": int(args.train_bs),
                    "grad_accum": int(args.grad_accum),
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                    "note": "IOSB packform9 EVA-CLIP full finetune; background excluded; class_id 0..7",
                },
                ckpt_path,
            )
            print(f"[CKPT] saved best: {ckpt_path}", flush=True)

        metrics_path.write_text(json.dumps({"best_val_top1": best_val_top1, "history": history}, indent=2))

    print(f"[DONE] Best val top1={best_val_top1:.4f} ckpt={ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
