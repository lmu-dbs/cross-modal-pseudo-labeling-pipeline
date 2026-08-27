import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from pycocotools import mask as mask_utils

from sam_clip_full.clip_module import (
    DEFAULT_TEMPLATES,
    load_eva_clip,
    EVACLIPWrapper,
    load_finetuned_checkpoint,
)
from sam_clip_full.utils.context_fusion import (
    pad_bbox,
    crop_image,
    apply_soft_mask_weighting,
    mask_to_tight_bbox,
    dynamic_alpha_from_box,
    dynamic_beta_from_box,
)

from sam_clip_full.sam_module import load_sam_predictor, sam_predict_masks


def encode_binary_mask(m: np.ndarray) -> dict:
    """Encode a {0,1} mask (HxW) to JSON-safe COCO RLE."""
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def load_class_prompts_from_txt(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate SAM+EVA-CLIP pseudo labels from a generic image folder."
    )
    ap.add_argument("--img_root", type=str, required=True,
                    help="Folder with target-domain images (.jpg/.png).")
    ap.add_argument("--class_prompts", type=str, required=True,
                    help="Text file with one class name per line.")
    ap.add_argument("--out_jsonl", type=str, required=True,
                    help="Where to write the pseudo-label JSONL.")

    ap.add_argument("--device", type=str, default="cuda",
                    help="'cuda' or 'cpu'.")
    ap.add_argument("--model_name", type=str, default="EVA02-L-14")
    ap.add_argument("--pretrained", type=str, default="merged2b_s4b_b131k")
    ap.add_argument("--finetuned_ckpt", type=str, default="",
                    help="Optional finetuned EVA-CLIP checkpoint.")

    ap.add_argument("--max_images", type=int, default=-1,
                    help="Max images to process (-1 = all).")
    ap.add_argument("--max_instances_per_image", type=int, default=50)
    ap.add_argument("--min_area", type=int, default=100)
    ap.add_argument("--pad_ratio", type=float, default=0.40)
    ap.add_argument("--context_ratio", type=float, default=0.40)
    return ap.parse_args()


def main():
    args = parse_args()
    img_root = Path(args.img_root)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device

    # 1) EVA-CLIP
    model, preprocess, tokenizer = load_eva_clip(
        device=device,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )
    if args.finetuned_ckpt:
        _ = load_finetuned_checkpoint(model, args.finetuned_ckpt)
        print(f"Loaded finetuned EVA-CLIP checkpoint: {args.finetuned_ckpt}")

    eva = EVACLIPWrapper(
        clip_model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        combine="add",
    )

    # 2) Text prompts
    classes = load_class_prompts_from_txt(args.class_prompts)
    print(f"Loaded {len(classes)} class prompts.")
    text_embeds = eva.build_text_cache(classes, DEFAULT_TEMPLATES, layout="mean")
    text_feat_mean = eva.compute_text_mean(text_embeds)

    # 3) SAM automatic mask generator
    sam_predictor = load_sam_predictor(device=device)

    # 4) Image loop
    img_paths = sorted(
        list(img_root.glob("*.jpg")) +
        list(img_root.glob("*.jpeg")) +
        list(img_root.glob("*.png"))
    )
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    print(f"Found {len(img_paths)} images in {img_root}")

    with out_path.open("w") as f_out:
        for img_path in tqdm(img_paths, desc="Pseudo-labeling"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to load {img_path}: {e}")
                continue

            np_img = np.array(img)
            h, w = np_img.shape[:2]
            img_w, img_h = img.size

            try:
                masks = sam_predict_masks(sam_predictor, np_img)
            except Exception as e:
                print(f"[WARN] SAM failed on {img_path}: {e}")
                continue
            if not masks:
                continue

            per_image_instances: List[Dict[str, Any]] = []
            inst_count = 0

            for m in masks:
                if inst_count >= args.max_instances_per_image:
                    break
                if m is None or m.sum() < args.min_area:
                    continue

                tight = mask_to_tight_bbox(m.astype(bool))
                if tight is None:
                    continue
                xmin, ymin, xmax, ymax = map(int, tight)

                alpha_dyn = dynamic_alpha_from_box(
                    tight, img_w, img_h, a_min=0.05, a_max=0.20
                )
                beta_dyn = dynamic_beta_from_box(
                    tight, img_w, img_h, b_min=0.05, b_max=0.25
                )

                pxmin, pymin, pxmax, pymax = pad_bbox(
                    xmin, ymin, xmax, ymax, args.pad_ratio, img_w, img_h
                )
                crop_pil = crop_image(img, (pxmin, pymin, pxmax, pymax))
                cw, ch = crop_pil.size

                y0, y1 = max(pymin, 0), min(pymin + ch, h)
                x0, x1 = max(pxmin, 0), min(pxmin + cw, w)
                if (y1 - y0) <= 0 or (x1 - x0) <= 0:
                    continue
                mask_crop = m[y0:y1, x0:x1]
                mask_canvas = np.zeros((ch, cw), dtype=np.uint8)
                mask_canvas[0:(y1 - y0), 0:(x1 - x0)] = mask_crop

                crop_weighted = apply_soft_mask_weighting(crop_pil, mask_canvas)

                cxmin, cymin, cxmax, cymax = pad_bbox(
                    xmin, ymin, xmax, ymax, args.context_ratio, img_w, img_h
                )
                context_pil = crop_image(img, (cxmin, cymin, cxmax, cymax))

                logits_by_class, probs, topk_dict, a_used, b_used = \
                    eva.classify_crop_with_context_residual_topk(
                        crop_pil=crop_weighted,
                        context_pil=context_pil,
                        text_embeds=text_embeds,
                        text_feat_mean=text_feat_mean,
                        alpha=alpha_dyn,
                        beta=beta_dyn,
                        cat_names=classes,
                        templates=DEFAULT_TEMPLATES,
                        topk_list=(1,),
                    )

                pred_idx = int(torch.argmax(logits_by_class).item())
                pred_conf = float(probs[pred_idx].item())

                m_rle = encode_binary_mask(m)

                per_image_instances.append(
                    {
                        "rle": m_rle,
                        "pred_idx": int(pred_idx),
                        "pred_conf": pred_conf,
                        "area": int(m.sum()),
                    }
                )
                inst_count += 1

            if per_image_instances:
                rec = {
                    "file_name": img_path.name,
                    "width": int(img_w),
                    "height": int(img_h),
                    "instances": per_image_instances,
                }
                f_out.write(json.dumps(rec) + "\n")

    print(f"Done. Wrote pseudo labels to {out_path}")


if __name__ == "__main__":
    main()
