#!/usr/bin/env python3
"""
pseudo_lobbe_sam_evaclip_synonyms.py

Lobbe waste pseudo-label generation using:
  - SAM1 automatic masks (vit_h)
  - EVA-CLIP (EVA02-L-14) with a finetuned checkpoint
  - Context-residual fusion with dynamic alpha/beta from box size
  - Optional synonym-based prompting (recommended)
  - Outputs JSONL compatible with your existing convert_jsonl_to_semantic.py

JSONL format (per image):
{
  "file_name": "<relative path from img_root>",
  "abs_path": "<absolute path>",
  "width": W,
  "height": H,
  "instances": [
     {
       "rle": <COCO RLE dict>,
       "pred_idx": int,          # 0..C-1 (NOTE: when skip_background=True, this is 0..7 for 8 foreground classes)
       "pred_name": str,
       "pred_conf": float,
       "bbox": [xmin, ymin, xmax, ymax],
       "area": int,
       "alpha": float,
       "beta": float
     }, ...
  ]
}

Important:
- By default, we SKIP background in classification (because SAM instances are objects).
  So pred_idx corresponds to the 8 foreground classes in this order:
    bottle, bag_film, cup_tray, lid_cap, carton, can, foam, other_packaging
  i.e. num_classes=8 for the conversion script.

Example:
  python pseudo_lobbe_sam_evaclip_synonyms.py \
    --img_root /path/to/lobbe_train_imgs \
    --out_jsonl /path/to/out/pseudo_train.jsonl \
    --eva_ckpt /path/to/eva_clip_ft_iosb_packform9_FULL.pt \
    --use_synonyms \
    --device cuda \
    --pattern "*.png"
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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
    dynamic_alpha_beta_from_box,
)


def _get_clip_model(eva):
    # EVACLIPWrapper naming differs across repos; try common attribute names.
    for attr in ("clip_model", "model", "clip", "net"):
        if hasattr(eva, attr):
            m = getattr(eva, attr)
            if m is not None:
                return m
    raise AttributeError(
        "EVACLIPWrapper does not expose the underlying CLIP model under "
        "any of: clip_model, model, clip, net. Inspect EVACLIPWrapper to confirm."
    )


# ---------------------------
# Lobbe / IOSB label space
# ---------------------------
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


# ---------------------------
# helpers
# ---------------------------
def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    """Convert binary mask to COCO RLE dict (pycocotools)."""
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # json-serializable
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def iter_images(img_root: Path, pattern: str) -> List[Path]:
    return sorted(img_root.rglob(pattern))


def _tokenize_prompts(tokenizer, prompts: List[str], device: str) -> torch.Tensor:
    """
    EVA/OpenCLIP tokenizers are usually callable (tokenizer(prompts)).
    If your tokenizer differs, adjust here without touching the rest of the code.
    """
    tok = tokenizer(prompts)
    if isinstance(tok, torch.Tensor):
        return tok.to(device)
    raise TypeError(f"Tokenizer returned {type(tok)}; expected torch.Tensor. Adjust _tokenize_prompts().")


@torch.no_grad()
def build_text_cache_from_synonyms(
    eva: EVACLIPWrapper,
    class_names: List[str],                 # IOSB_CLASSES_9
    synonyms_dict: Dict[str, List[str]],
    templates: List[str],
    pool: str = "mean",                     # "mean" or "max"
    skip_background: bool = True,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Returns:
      labels_out: list[str] aligned with embeddings (length C)
      text_embeds: torch.Tensor (C, D)
      text_mean: torch.Tensor (1, D)
      phrase_counts: dict[class] -> number of synonym phrases used
    """
    device = eva.device
    clip_model = _get_clip_model(eva)
    clip_model.eval()


    labels_out: List[str] = []
    vecs: List[torch.Tensor] = []
    phrase_counts: Dict[str, int] = {}

    for cname in class_names:
        if skip_background and cname == "background":
            continue

        phrases = synonyms_dict.get(cname) or [cname]
        phrase_counts[cname] = len(phrases)

        prompts: List[str] = []
        for ph in phrases:
            prompts.extend([t.format(ph) for t in templates])

        tok = _tokenize_prompts(eva.tokenizer, prompts, device=device)
        z = clip_model.encode_text(tok)

        z = z / z.norm(dim=-1, keepdim=True)

        if pool == "max":
            z_cls = z.max(dim=0, keepdim=True).values
        else:
            z_cls = z.mean(dim=0, keepdim=True)

        z_cls = z_cls / z_cls.norm(dim=-1, keepdim=True)

        labels_out.append(cname)
        vecs.append(z_cls)

    if len(vecs) == 0:
        raise RuntimeError("No text vectors were built. Check class list / skip_background settings.")

    text_embeds = torch.cat(vecs, dim=0)  # (C,D)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    text_mean = text_embeds.mean(dim=0, keepdim=True)
    text_mean = text_mean / text_mean.norm(dim=-1, keepdim=True)

    clip_model.train()

    return labels_out, text_embeds, text_mean, phrase_counts


@torch.no_grad()
def build_text_cache_plain(
    eva: EVACLIPWrapper,
    labels: List[str],
    templates: List[str],
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Plain prompts (templates × label).
    Uses EVACLIPWrapper's helper.
    """
    text_embeds = eva.build_text_cache(labels, templates, layout="mean")
    text_mean = eva.compute_text_mean(text_embeds)
    return labels, text_embeds, text_mean


# ---------------------------
# main
# ---------------------------
@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = "cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    print(f"[Device] {device}")

    img_root = Path(args.img_root)
    out_jsonl = Path(args.out_jsonl)
    eva_ckpt = Path(args.eva_ckpt)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_jsonl, "w")
    print(f"[Out] {out_jsonl}")

    # ---------------------------
    # SAM (force SAM1 vit_h)
    # ---------------------------
    SAM1_CKPT = args.sam1_ckpt
    if not os.path.isfile(SAM1_CKPT):
        raise FileNotFoundError(f"SAM1 checkpoint not found: {SAM1_CKPT}")

    sam = load_sam_predictor(
        device=device,
        backend="sam1",
        model_type="vit_h",
        checkpoint=SAM1_CKPT,
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
    # your checkpoints use ckpt["model_state_dict"]
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing = model.load_state_dict(sd, strict=False)
    print("[EVA] load_state_dict missing/unexpected:", missing)

    eva = EVACLIPWrapper(
        clip_model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        combine="add",
    )

    # ---------------------------
    # Text cache
    # ---------------------------
    templates = IOSB_TEMPLATES

    # Note: for instance pseudo-labeling, background doesn't make sense, so skip_background=True
    if args.use_synonyms:
        print("[Text] Building synonym-pooled text cache (skip background)...")
        labels, text_embeds, text_mean, phrase_counts = build_text_cache_from_synonyms(
            eva,
            class_names=IOSB_CLASSES_9,
            synonyms_dict=IOSB_SYNONYMS,
            templates=templates,
            pool=args.pool,
            skip_background=True,
        )
        print("[Text] Labels:", labels)
        print("[Text] Phrases per class:", phrase_counts)
    else:
        print("[Text] Building plain label text cache (skip background)...")
        labels_plain = IOSB_CLASSES_9[1:]  # 8 foreground classes
        labels, text_embeds, text_mean = build_text_cache_plain(eva, labels_plain, templates)
        print("[Text] Labels:", labels)

    # ---------------------------
    # Images
    # ---------------------------
    img_paths = iter_images(img_root, args.pattern)
    print(f"[Data] Found {len(img_paths)} images under {img_root} pattern={args.pattern}")

    total = 0
    for img_path in img_paths:
        total += 1
        if args.max_images > 0 and total > args.max_images:
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
            if area < args.min_mask_area:
                continue

            tight = mask_to_tight_bbox(m.astype(bool))
            if tight is None:
                continue
            xmin, ymin, xmax, ymax = map(int, tight)

            bw = xmax - xmin + 1
            bh = ymax - ymin + 1
            if bw < args.min_box_w or bh < args.min_box_h:
                continue

            # dynamic alpha/beta based on box size
            alpha_dyn, beta_dyn, _fg_ratio = dynamic_alpha_beta_from_box(
                (xmin, ymin, xmax, ymax),
                img_w=W,
                img_h=H,
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max,
            )

            # 1) OBJECT crop (pad bbox) + soft mask weighting
            px0, py0, px1, py1 = pad_bbox(xmin, ymin, xmax, ymax, args.ctx_pad_ratio, W, H)
            crop_pil = crop_image(img_pil, (px0, py0, px1, py1))
            cw, ch = crop_pil.size

            mask_canvas = np.zeros((ch, cw), dtype=np.uint8)

            y0 = max(0, py0); x0 = max(0, px0)
            y1 = min(H, py1); x1 = min(W, px1)
            if y1 <= y0 or x1 <= x0:
                continue

            mask_crop = m[y0:y1, x0:x1]
            mask_canvas[:mask_crop.shape[0], :mask_crop.shape[1]] = mask_crop
            crop_weighted = apply_soft_mask_weighting(crop_pil, mask_canvas)

            # 2) CONTEXT crop (bigger) - not weighted
            ctx_ratio = min(0.90, args.ctx_pad_ratio + args.ctx_extra_pad)
            cx0, cy0, cx1, cy1 = pad_bbox(xmin, ymin, xmax, ymax, ctx_ratio, W, H)
            ctx_pil = crop_image(img_pil, (cx0, cy0, cx1, cy1))

            # classify
            logits_by_class, probs, _topk_dict, _a_used, _b_used = eva.classify_crop_with_context_residual_topk(
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
            if pred_conf < args.min_conf:
                continue

            instances.append({
                "rle": encode_binary_mask(m),
                "pred_idx": pred_idx,  # IMPORTANT: 0..(len(labels)-1), compatible with converter
                "pred_name": labels[pred_idx] if 0 <= pred_idx < len(labels) else str(pred_idx),
                "pred_conf": pred_conf,
                "bbox": [xmin, ymin, xmax, ymax],
                "area": area,
                "alpha": float(alpha_dyn),
                "beta": float(beta_dyn),
            })

        fout.write(json.dumps({
            "file_name": str(img_path.relative_to(img_root)),
            "abs_path": str(img_path),
            "width": int(W),
            "height": int(H),
            "instances": instances,
        }) + "\n")

    fout.close()
    print(f"[DONE] Processed {min(total, len(img_paths))} images. Output: {out_jsonl}")
    print(f"[INFO] Number of classes in JSONL pred_idx space: {len(labels)}")
    print(f"[INFO] Use convert_jsonl_to_semantic.py with --num_classes {len(labels)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Lobbe pseudo-label generation with SAM1 + EVA-CLIP + synonym prompts")

    ap.add_argument("--img_root", required=True, type=str, help="Folder containing images (recursively searched).")
    ap.add_argument("--out_jsonl", required=True, type=str, help="Output JSONL path.")
    ap.add_argument("--eva_ckpt", required=True, type=str, help="Finetuned EVA-CLIP checkpoint (.pt).")

    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--pattern", default="*.png", type=str)
    ap.add_argument("--max_images", type=int, default=0)

    ap.add_argument("--sam1_ckpt", type=str, default="/home/scs_deal_projects_notapebackup/user/shubhang/thesis/third_party/checkpoints/sam_vit_h_4b8939.pth")

    ap.add_argument("--use_synonyms", action="store_true", help="Use IOSB synonym prompts instead of plain labels.")
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "max"], help="How to pool synonyms into a class vector.")

    ap.add_argument("--ctx_pad_ratio", type=float, default=0.40, help="Pad ratio for object crop around bbox.")
    ap.add_argument("--ctx_extra_pad", type=float, default=0.35, help="Extra pad added to ctx_pad_ratio for context crop.")
    ap.add_argument("--min_mask_area", type=int, default=800)
    ap.add_argument("--min_box_w", type=int, default=24)
    ap.add_argument("--min_box_h", type=int, default=24)
    ap.add_argument("--min_conf", type=float, default=0.30)

    ap.add_argument("--a_min", type=float, default=0.05)
    ap.add_argument("--a_max", type=float, default=0.20)
    ap.add_argument("--b_min", type=float, default=0.05)
    ap.add_argument("--b_max", type=float, default=0.25)

    args = ap.parse_args()
    main(args)
