#!/usr/bin/env python3
"""
General pseudo-label generation with:
  - SAM automatic masks (via your sam_module)
  - EVA-CLIP finetuned checkpoint (RAW or MERGED)
  - Text labels + templates
  - Context-residual fusion with dynamic alpha/beta from box size
  - Outputs JSONL (one line per image)

Example (Cityscapes):
  python -m sam_clip_full.pseudo.pseudo_sam_evaclip_general \
    --img_root /path/to/leftImg8bit/val \
    --out_jsonl /path/to/out/pseudo_val.jsonl \
    --labels_txt sam_clip_full/configs/cityscapes_text_labels_19.txt \
    --templates_txt sam_clip_full/configs/cityscapes_templates.txt \
    --eva_ckpt /path/to/eva_clip_ft_gta5_ctxfusion_19cls.pt \
    --pattern "*leftImg8bit.png"
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import torch

# ---------------------------
# project imports (your codebase)
# ---------------------------
from sam_clip_full.clip_module import load_eva_clip, EVACLIPWrapper

from sam_clip_full.sam_module import load_sam_predictor, sam_predict_masks

from sam_clip_full.utils.context_fusion import (
    pad_bbox,
    crop_image,
    apply_soft_mask_weighting,
    mask_to_tight_bbox,
    dynamic_alpha_beta_from_box,   # make sure this exists in your repo
)

# ---------------------------
# helpers
# ---------------------------
def load_lines(filepath: Path) -> List[str]:
    with open(filepath, "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]

def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    """Convert mask to COCO RLE."""
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def iter_images(img_root: Path, pattern: str) -> List[Path]:
    # pattern can be "*.png", "*leftImg8bit.png", etc.
    return sorted(img_root.rglob(pattern))

# ---------------------------
# main
# ---------------------------
@torch.no_grad()
def main(
    img_root: Path,
    out_jsonl: Path,
    labels_txt: Path,
    templates_txt: Path,
    eva_ckpt: Path,
    device: str = "cuda",
    pattern: str = "*.png",
    max_images: int = 0,
    ctx_pad_ratio: float = 0.40,
    min_mask_area: int = 800,
    min_box_w: int = 24,
    min_box_h: int = 24,
    min_conf: float = 0.30,
    a_min: float = 0.05,
    a_max: float = 0.20,
    b_min: float = 0.05,
    b_max: float = 0.25,
):
    device = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"[Device] {device}")

    labels = load_lines(labels_txt)
    templates = load_lines(templates_txt)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_jsonl, "w")
    print(f"[Out] {out_jsonl}")

    # ---------------------------
    # SAM
    # ---------------------------
    SAM1_CKPT = "/home/scs_deal_projects_notapebackup/user/shubhang/thesis/third_party/checkpoints/sam_vit_h_4b8939.pth"

    sam = load_sam_predictor(
    device=device,
    backend="sam1",               # << force SAM1 (prevents auto->sam2)
    model_type="vit_h",           # SAM1 model type
    checkpoint=SAM1_CKPT,         # .pth checkpoint
                    )
    # ---------------------------
    # EVA-CLIP
    # ---------------------------
    print("[EVA] Loading base EVA-CLIP...")
    model, preprocess, tokenizer = load_eva_clip(
        device=device,
        model_name="EVA02-L-14",
        pretrained="merged2b_s4b_b131k",
    )

    print(f"[EVA] Loading finetuned ckpt: {eva_ckpt}")
    ckpt = torch.load(str(eva_ckpt), map_location="cpu", weights_only=False)
    missing = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("[EVA] load_state_dict missing/unexpected:", missing)

    eva = EVACLIPWrapper(
        clip_model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        combine="add",
    )

    print("[Text] Building text cache...")
    text_embeds = eva.build_text_cache(labels, templates, layout="mean")
    text_mean = eva.compute_text_mean(text_embeds)

    # ---------------------------
    # loop over images
    # ---------------------------
    img_paths = iter_images(img_root, pattern)
    print(f"[Data] Found {len(img_paths)} images in {img_root} matching pattern={pattern}")

    total = 0
    for img_path in img_paths:
        total += 1
        if max_images > 0 and total > max_images:
            break

        if total % 50 == 0:
            print(f"Processed {total} images...")

        img_pil = Image.open(img_path).convert("RGB")
        np_img = np.array(img_pil)
        H, W = np_img.shape[:2]

        masks = sam_predict_masks(sam, np_img)

        instances = []
        for m in masks:
            m = m.astype(np.uint8)
            area = int(m.sum())
            if area < min_mask_area:
                continue

            tight = mask_to_tight_bbox(m.astype(bool))
            if tight is None:
                continue
            xmin, ymin, xmax, ymax = map(int, tight)

            bw = xmax - xmin + 1
            bh = ymax - ymin + 1
            if bw < min_box_w or bh < min_box_h:
                continue

            # dynamic alpha/beta based on box size
            alpha_dyn, beta_dyn, fg_ratio = dynamic_alpha_beta_from_box(
                (xmin, ymin, xmax, ymax),
                img_w=W,
                img_h=H,
                a_min=a_min, a_max=a_max,
                b_min=b_min, b_max=b_max,
            )

            # pad bbox for crop + ctx
            px0, py0, px1, py1 = pad_bbox(xmin, ymin, xmax, ymax, ctx_pad_ratio, W, H)
            crop_pil = crop_image(img_pil, (px0, py0, px1, py1))
            cw, ch = crop_pil.size

# align mask into OBJECT crop coords and apply soft weighting
            mask_canvas = np.zeros((ch, cw), dtype=np.uint8)

            y0 = max(0, py0); x0 = max(0, px0)
            y1 = min(H, py1); x1 = min(W, px1)
            if y1 <= y0 or x1 <= x0:
                continue

            mask_crop = m[y0:y1, x0:x1]
            mask_canvas[:mask_crop.shape[0], :mask_crop.shape[1]] = mask_crop

            crop_weighted = apply_soft_mask_weighting(crop_pil, mask_canvas)

# -------------------------
# 2) CONTEXT CROP (bigger than object crop) -> NOT weighted
# -------------------------
            CTX_PAD_RATIO = min(0.90, ctx_pad_ratio + 0.35)   # 0.40 -> 0.75
            cx0, cy0, cx1, cy1 = pad_bbox(xmin, ymin, xmax, ymax, CTX_PAD_RATIO, W, H)
            ctx_pil = crop_image(img_pil, (cx0, cy0, cx1, cy1))

            # classify
            logits_by_class, probs, topk_dict, a_used, b_used = eva.classify_crop_with_context_residual_topk(
                crop_pil=crop_weighted,
                context_pil=ctx_pil,
                text_embeds=text_embeds,
                text_feat_mean=text_mean,
                alpha=alpha_dyn,
                beta=beta_dyn,
                cat_names=labels,
                templates=templates,
                topk_list=(1, 5),
            )

            pred_idx = int(torch.argmax(logits_by_class).item())
            pred_conf = float(probs[pred_idx].item())
            if pred_conf < min_conf:
                continue

            instances.append({
                "rle": encode_binary_mask(m),
                "pred_idx": pred_idx,
                "pred_name": labels[pred_idx] if 0 <= pred_idx < len(labels) else str(pred_idx),
                "pred_conf": pred_conf,
                "bbox": [xmin, ymin, xmax, ymax],
                "area": area,
                "alpha": float(alpha_dyn),
                "beta": float(beta_dyn),
            })

        # write JSONL
        fout.write(json.dumps({
            "file_name": str(img_path.relative_to(img_root)),
            "abs_path": str(img_path),
            "width": int(W),
            "height": int(H),
            "instances": instances,
        }) + "\n")

    fout.close()
    print(f"[DONE] Processed {min(total, len(img_paths))} images. Output: {out_jsonl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True, type=str, help="Folder containing images (recursively searched).")
    ap.add_argument("--out_jsonl", required=True, type=str)
    ap.add_argument("--labels_txt", required=True, type=str)
    ap.add_argument("--templates_txt", required=True, type=str)
    ap.add_argument("--eva_ckpt", required=True, type=str)

    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--pattern", default="*.png", type=str)
    ap.add_argument("--max_images", type=int, default=0)

    ap.add_argument("--ctx_pad_ratio", type=float, default=0.40)
    ap.add_argument("--min_mask_area", type=int, default=800)
    ap.add_argument("--min_box_w", type=int, default=24)
    ap.add_argument("--min_box_h", type=int, default=24)
    ap.add_argument("--min_conf", type=float, default=0.30)

    ap.add_argument("--a_min", type=float, default=0.05)
    ap.add_argument("--a_max", type=float, default=0.20)
    ap.add_argument("--b_min", type=float, default=0.05)
    ap.add_argument("--b_max", type=float, default=0.25)

    args = ap.parse_args()

    main(
        img_root=Path(args.img_root),
        out_jsonl=Path(args.out_jsonl),
        labels_txt=Path(args.labels_txt),
        templates_txt=Path(args.templates_txt),
        eva_ckpt=Path(args.eva_ckpt),
        device=args.device,
        pattern=args.pattern,
        max_images=args.max_images,
        ctx_pad_ratio=args.ctx_pad_ratio,
        min_mask_area=args.min_mask_area,
        min_box_w=args.min_box_w,
        min_box_h=args.min_box_h,
        min_conf=args.min_conf,
        a_min=args.a_min, a_max=args.a_max,
        b_min=args.b_min, b_max=args.b_max,
    )
