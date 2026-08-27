# runner_coco_eval.py
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils  # NEW: for RLE export

# === Your modules (adjust names if needed) ===
from sam_clip_full.clip_module import (
    DEFAULT_TEMPLATES,
    load_eva_clip,
    EVACLIPWrapper,
    load_finetuned_checkpoint,
)
from sam_clip_full.utils.context_fusion import (
    pad_bbox, crop_image, apply_soft_mask_weighting,
    mask_to_tight_bbox, dynamic_alpha_from_box, dynamic_beta_from_box
)

# Try to import your CSV writer; else fallback
try:
    from sam_clip_full.utils.io_utils import write_csv
except Exception:
    def write_csv(rows: List[Dict[str, Any]], path: str):
        import csv
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with open(path, "w", newline="") as f:
                pass
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def encode_binary_mask(m: np.ndarray) -> dict:
    """Encode a {0,1} mask (HxW) to JSON-safe COCO RLE."""
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # make JSON-serializable
    return rle


def bbox_to_mask(bbox: List[float], h: int, w: int) -> np.ndarray:
    """Simple bbox→binary mask (uint8 {0,1})."""
    x, y, bw, bh = bbox
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w, int(x + bw))
    y1 = min(h, int(y + bh))
    m = np.zeros((h, w), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 1
    return m


def load_labels_from_val(coco_val: COCO) -> List[str]:
    cats = coco_val.loadCats(coco_val.getCatIds())
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    return [c["name"] for c in cats_sorted]


def main(cfg_path: str):
    # -------------------------
    # 0) Load config
    # -------------------------
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # dataset
    val_imgs = cfg["dataset"]["val"]["images_dir"]
    val_ann  = cfg["dataset"]["val"]["ann_file"]
    test_imgs = cfg["dataset"]["test"]["images_dir"]
    test_ann  = cfg["dataset"]["test"]["ann_file"]

    # eval settings
    max_images = int(cfg["eval"].get("max_images", 0))
    max_inst   = int(cfg["eval"].get("max_instances_per_image", 50))
    min_area   = int(cfg["eval"].get("min_area", 100))
    use_bbox_fallback = bool(cfg["eval"].get("use_bbox_if_no_seg", True))
    pad_ratio     = float(cfg["eval"].get("pad_ratio", 0.40))
    context_ratio = float(cfg["eval"].get("context_ratio", 0.40))
    # topk list (e.g., [1,5] or [1,3,5])
    topk_list = tuple(sorted({int(k) for k in cfg["eval"].get("topk", [1, 3, 5]) if int(k) > 0}))

    # output
    out_csv  = cfg["output"]["pred_csv"]
    out_json = cfg["output"]["metrics_json"]
    pseudo_jsonl = cfg["output"].get("pseudo_jsonl", None)  # NEW
    pseudo_f = open(pseudo_jsonl, "w") if pseudo_jsonl else None

    # clip args
    clip_args = cfg["clip"]
    device = clip_args.get("device", "cuda")
    model_name = clip_args.get("model_name", "EVA02-L-14")
    pretrained_tag = clip_args.get("pretrained", "merged2b_s4b_b131k")
    ckpt_path = clip_args.get("finetuned_ckpt", None)
    combine = clip_args.get("combine", "add")
    text_layout = clip_args.get("text_layout", "mean")  # "mean" or "stack"

    # -------------------------
    # 1) COCO + labels
    # -------------------------
    coco_val  = COCO(val_ann)
    coco_test = COCO(test_ann)
    labels = load_labels_from_val(coco_val)
    print(f"Loaded {len(labels)} labels from val categories.")
    name_to_idx = {name: i for i, name in enumerate(labels)}
    classes = labels

    # -------------------------
    # 2) EVA-CLIP (+ optional finetuned ckpt)
    # -------------------------
    model, preprocess, tokenizer = load_eva_clip(
        device=device, model_name=model_name, pretrained=pretrained_tag
    )
    if ckpt_path:
        _ = load_finetuned_checkpoint(model, ckpt_path)
        print(f"Loaded finetuned EVA-CLIP checkpoint: {ckpt_path}")

    eva = EVACLIPWrapper(
        clip_model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        combine=combine,
    )

    # -------------------------
    # 3) Text embeddings (+ mean)
    # -------------------------
    text_embeds = eva.build_text_cache(classes, DEFAULT_TEMPLATES, layout=text_layout)
    text_feat_mean = eva.compute_text_mean(text_embeds)

    # -------------------------
    # 4) Eval loop
    # -------------------------
    img_ids = coco_test.getImgIds()
    if max_images > 0:
        img_ids = img_ids[:max_images]

    rows: List[Dict[str, Any]] = []
    y_true, y_pred = [], []
    per_k_hits: Dict[int, List[int]] = {k: [] for k in topk_list}

    for i, img_id in enumerate(img_ids, start=1):
        img_info = coco_test.loadImgs([img_id])[0]
        img_path = os.path.join(test_imgs, img_info["file_name"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue

        np_img = np.array(img)
        h, w = np_img.shape[:2]
        img_w, img_h = img.size

        ann_ids = coco_test.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco_test.loadAnns(ann_ids)

        # NEW: per-image collector for pseudo export
        per_image_instances: List[Dict[str, Any]] = []

        inst_count = 0
        for ann in anns:
            if inst_count >= max_inst:
                break

            cat_id = ann["category_id"]
            cat_name = coco_test.loadCats([cat_id])[0]["name"]
            if cat_name not in name_to_idx:
                continue
            target_idx = name_to_idx[cat_name]

            # mask (or bbox fallback)
            mask = coco_test.annToMask(ann)
            if mask is None or mask.sum() < min_area:
                if use_bbox_fallback and "bbox" in ann and ann["bbox"]:
                    mask = bbox_to_mask(ann["bbox"], h, w)
                    if mask.sum() < min_area:
                        continue
                else:
                    continue

            # tight bbox from mask; fallback to COCO bbox if degenerate
            tight = mask_to_tight_bbox(mask.astype(bool))
            if tight is None:
                tight = tuple(map(int, [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]))
            xmin, ymin, xmax, ymax = map(int, tight)

            # dynamic α/β from relative area
            alpha_dyn = dynamic_alpha_from_box(tight, img_w, img_h, a_min=0.05, a_max=0.20)
            beta_dyn  = dynamic_beta_from_box(tight,  img_w, img_h, b_min=0.05, b_max=0.25)

            # crop window (with context)
            pxmin, pymin, pxmax, pymax = pad_bbox(xmin, ymin, xmax, ymax, pad_ratio, img_w, img_h)
            crop_pil  = crop_image(img, (pxmin, pymin, pxmax, pymax))
            cw, ch = crop_pil.size  # PIL returns (width, height)

            # align mask to crop; paste into canvas of crop size to avoid mismatch
            y0, y1 = max(pymin, 0), min(pymin + ch, h)
            x0, x1 = max(pxmin, 0), min(pxmin + cw, w)
            if (y1 - y0) <= 0 or (x1 - x0) <= 0:
                continue
            mask_crop = mask[y0:y1, x0:x1]
            mask_canvas = np.zeros((ch, cw), dtype=np.uint8)
            mask_canvas[0:(y1 - y0), 0:(x1 - x0)] = mask_crop

            crop_weighted = apply_soft_mask_weighting(crop_pil, mask_canvas)

            # separate local context window (for z_ctx)
            cxmin, cymin, cxmax, cymax = pad_bbox(xmin, ymin, xmax, ymax, context_ratio, img_w, img_h)
            context_pil = crop_image(img, (cxmin, cymin, cxmax, cymax))

            # classify via residual fusion with top-k
            logits_by_class, probs, topk_dict, a_used, b_used = eva.classify_crop_with_context_residual_topk(
                crop_pil=crop_weighted,
                context_pil=context_pil,
                text_embeds=text_embeds,
                text_feat_mean=text_feat_mean,
                alpha=alpha_dyn,
                beta=beta_dyn,
                cat_names=classes,
                templates=DEFAULT_TEMPLATES,
                topk_list=topk_list,
            )

            pred_idx = int(torch.argmax(logits_by_class).item())

            # bookkeeping
            y_true.append(target_idx)
            y_pred.append(pred_idx)
            for k in topk_list:
                per_k_hits[k].append(1 if target_idx in topk_dict.get(k, []) else 0)

            rows.append({
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "gt_name": cat_name,
                "gt_idx": target_idx,
                "pred_idx": pred_idx,
                "pred_name": classes[pred_idx],
                "pred_conf": float(probs[pred_idx].item()),
                "pad_ratio": float(pad_ratio),
                "alpha_used": float(a_used),
                "beta_used": float(b_used),
                "source": "sam_eva_context_residual_topk"
            })

            # NEW: add to per-image pseudo instances
            if pseudo_f is not None:
                m_rle = encode_binary_mask(mask)
                per_image_instances.append({
                    "rle": m_rle,
                    "pred_idx": int(pred_idx),
                    "pred_conf": float(probs[pred_idx].item()),
                    "area": int(mask.sum()),
                })

            inst_count += 1

        # NEW: flush one JSON line per image
        if pseudo_f is not None and per_image_instances:
            rec = {
                "file_name": img_info["file_name"],
                "width": int(img_w),
                "height": int(img_h),
                "instances": per_image_instances,
            }
            pseudo_f.write(json.dumps(rec) + "\n")

        if i % 50 == 0:
            print(f"Processed {i}/{len(img_ids)} images...")

    # Close pseudo JSONL if open
    if pseudo_f is not None:
        pseudo_f.close()

    # -------------------------
    # 5) Metrics + save
    # -------------------------
    if not rows:
        print("No instances collected; check dataset paths/filters.")
        return

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)

    metrics: Dict[str, Any] = {
        "num_instances": int(len(y_true_np)),
    }
    for k in sorted(per_k_hits):
        metrics[f"top{k}_acc"] = float(np.mean(per_k_hits[k])) if per_k_hits[k] else 0.0

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="macro", zero_division=0
    )
    metrics.update({
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
    })

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("==== EVA-CLIP Context-Residual (GT masks, dynamic α/β) ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"Saved predictions: {out_csv}")
    print(f"Saved metrics:     {out_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to eval config yaml")
    args = parser.parse_args()
    main(args.config)
