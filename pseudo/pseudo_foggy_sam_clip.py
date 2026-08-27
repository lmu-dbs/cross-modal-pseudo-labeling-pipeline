"""
Foggy Cityscapes Pseudo-Label Generation
Uses:
 - SAM AutomaticMaskGenerator
 - Finetuned EVA-CLIP (Cityscapes + context fusion)
 - Strong descriptive text labels + templates loaded from .txt files
 - Dynamic alpha/beta from bounding box size
 - Outputs JSONL (one line per image)

Run:
    python -m sam_clip_full.pseudo.pseudo_foggy_sam_clip --beta 0.005
    python -m sam_clip_full.pseudo.pseudo_foggy_sam_clip --beta 0.01
    python -m sam_clip_full.pseudo.pseudo_foggy_sam_clip --beta 0.02
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

import torch

# ---------------------------
# your project imports
# ---------------------------
from sam_clip_full.clip_module import (
    load_eva_clip,
    EVACLIPWrapper,
    dynamic_alpha_beta_from_box,   # <-- add this
    apply_soft_mask_weighting,     # if you moved it here
    pad_bbox,
    crop_image,
)


from sam_clip_full.sam_module import (
    load_sam_predictor,
    sam_predict_masks,
)
from sam_clip_full.utils.context_fusion import (
    pad_bbox,
    crop_image,
    apply_soft_mask_weighting,
    mask_to_tight_bbox,
    dynamic_alpha_from_box,
    dynamic_beta_from_box,
)

# ---------------------------
# helpers
# ---------------------------

def load_lines(filepath: Path) -> List[str]:
    with open(filepath, "r") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    return lines


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    """Convert mask to COCO RLE."""
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ---------------------------
# main
# ---------------------------

def main(beta: float, max_images: int = 0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Running Foggy pseudo-label generation for beta={beta} on device={device} ===")

    # paths
    CITY = Path("/home/scs_deal_projects_notapebackup/shared/DATASET/cityscapes")
    FOG_ROOT = CITY / "leftImg8bit_foggyDBF"

    CFG_DIR = Path("/home/scs_deal_projects_notapebackup/user/shubhang/thesis/sam_clip_full/configs")
    LABELS_FILE = CFG_DIR / "cityscapes_text_labels_19.txt"
    TEMPLATES_FILE = CFG_DIR / "cityscapes_templates.txt"

    labels = load_lines(LABELS_FILE)
    templates = load_lines(TEMPLATES_FILE)
    assert len(labels) == 19, "Expected 19 Cityscapes labels."

    OUT_DIR = CITY / f"pseudo_foggy_beta_{str(beta).replace('.', '')}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSONL = OUT_DIR / "pseudo_labels.jsonl"
    fout = open(OUT_JSONL, "w")

    print(f"Saving pseudo-labels to: {OUT_JSONL}")

    # ---------------------------
    # SAM
    # ---------------------------
    print("Loading SAM AutomaticMaskGenerator...")
    sam = load_sam_predictor(
        device=device,
        model_type="vit_h",
        checkpoint=None  # uses DEFAULT_SAM_CHECKPOINT
    )

    # ---------------------------
    # EVA-CLIP
    # ---------------------------
    print("Loading EVA-CLIP...")
    model, preprocess, tokenizer = load_eva_clip(
    device=device,
    model_name="EVA02-L-14",
    pretrained="merged2b_s4b_b131k",
)

# <<< NEW: load your *balanced* Cityscapes checkpoint >>>
    ckpt_path = "/home/scs_deal_projects_notapebackup/user/shubhang/thesis/third_party/checkpoints/eva_clip_ft_cityscapes_ctxfusion_19cls_balanced.pt"
    print(f"Loading finetuned checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("Missing keys from checkpoint:", missing)

    eva = EVACLIPWrapper(
        clip_model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        combine="add",
    )
    # Build text embeddings
    with torch.no_grad():
        text_embeds = eva.build_text_cache(
            labels,
            templates,
            layout="mean",
        )
        text_mean = eva.compute_text_mean(text_embeds)

    # ---------------------------
    # Loop over foggy images
    # ---------------------------
    total_imgs = 0

    for split in ["train", "val"]:
        root = FOG_ROOT / split
        print(f"Processing: {root}")

        # matches filenames ending in ...beta_0.005.png, etc.
        pattern = f"*beta_{beta}.png"

        for img_path in sorted(root.rglob(pattern)):
            total_imgs += 1
            if max_images > 0 and total_imgs > max_images:
                break

            if total_imgs % 100 == 0:
                print(f"Processed {total_imgs} images...")

            img_pil = Image.open(img_path).convert("RGB")
            np_img = np.array(img_pil)
            h, w = np_img.shape[:2]

            # get SAM masks
            masks = sam_predict_masks(sam, np_img)

            instances = []
            H, W = h, w
            MIN_MASK_AREA = 800      # consistent with sam_module
            MIN_BOX_W = 24
            MIN_BOX_H = 24
            MIN_CONF = 0.30 
            for m in masks:
                area = int(m.sum())
                if area < MIN_MASK_AREA:   # area filter (can tune)
                    continue

                tight = mask_to_tight_bbox(m.astype(bool))
                if tight is None:
                    continue

                xmin, ymin, xmax, ymax = tight
                bw = xmax - xmin + 1
                bh = ymax - ymin + 1
                if bw < MIN_BOX_W or bh < MIN_BOX_H:
                    # too small for EVA-CLIP to be meaningful
                    continue

                # dynamic α, β based on relative box size
                alpha_dyn, beta_dyn, fg_ratio = dynamic_alpha_beta_from_box(
                    (xmin, ymin, xmax, ymax),
                    img_w=w,
                    img_h=h,
                    a_min=0.05, a_max=0.20,
                    b_min=0.05, b_max=0.25,
                )
                

                # crop window with pad
                px0, py0, px1, py1 = pad_bbox(xmin, ymin, xmax, ymax, 0.40, w, h)
                crop_pil = crop_image(img_pil, (px0, py0, px1, py1))
                cw, ch = crop_pil.size  # (width, height)

                # align mask to crop
                mask_canvas = np.zeros((ch, cw), dtype=np.uint8)
                y0 = max(0, py0)
                x0 = max(0, px0)
                y1 = min(h, py1)
                x1 = min(w, px1)
                if y1 <= y0 or x1 <= x0:
                    continue

                mask_crop = m[y0:y1, x0:x1]
                mask_canvas[:mask_crop.shape[0], :mask_crop.shape[1]] = mask_crop

                crop_weighted = apply_soft_mask_weighting(crop_pil, mask_canvas)

                # context crop (same pad)
                cx0, cy0, cx1, cy1 = pad_bbox(xmin, ymin, xmax, ymax, 0.40, w, h)
                ctx_pil = crop_image(img_pil, (cx0, cy0, cx1, cy1))

                # classify with context-residual fusion
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

                if pred_conf < MIN_CONF:
                    # too uncertain → don't use as pseudo label
                    continue

                instances.append({
                    "rle": encode_binary_mask(m),
                    "pred_idx": pred_idx,
                    "pred_name": labels[pred_idx],
                    "pred_conf": pred_conf,
                    "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "area": area,
                })


                instances.append({
                    "rle": encode_binary_mask(m),
                    "pred_idx": pred_idx,
                    "pred_name": labels[pred_idx],
                    "pred_conf": pred_conf,
                    "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "area": area,
                })

            # write JSONL
            fout.write(json.dumps({
                "file_name": str(img_path.relative_to(FOG_ROOT)),
                "width": w,
                "height": h,
                "instances": instances,
            }) + "\n")

        if max_images > 0 and total_imgs >= max_images:
            break

    fout.close()
    print(f"=== DONE. Total foggy images processed: {total_imgs} ===")
    print(f"Output: {OUT_JSONL}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True,
                        help="Fog intensity: 0.005, 0.01, 0.02")
    parser.add_argument("--max_images", type=int, default=0,
                        help="If > 0, only process this many images (for quick tests).")
    args = parser.parse_args()
    main(args.beta, max_images=args.max_images)
