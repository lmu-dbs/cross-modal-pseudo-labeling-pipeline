# sam_clip_full/pseudo/convert_jsonl_to_semantic.py
import os, json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

IGNORE_LABEL = 255
NUM_CLASSES = 19   # Cityscapes 19 trainIds


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = mask_utils.decode(rle)  # HxW uint8
    return m


def convert_jsonl_to_semantic(
    jsonl_path: str,
    out_dir: str,
    overlap_policy: str = "largest",   # or "conf"
    min_conf: float = 0.0,
    min_area: int = 0,
    max_images: int = 0,               # 0 = no limit
):
    os.makedirs(out_dir, exist_ok=True)

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f, start=1):
            if max_images > 0 and i > max_images:
                print(f"Reached max_images={max_images}, stopping early.")
                break

            rec = json.loads(line)
            fname = rec["file_name"]
            W, H = int(rec["width"]), int(rec["height"])
            insts = rec.get("instances", [])

            # filter by confidence / area
            insts = [
                d for d in insts
                if d.get("pred_conf", 1.0) >= min_conf and d.get("area", 0) >= min_area
            ]

            # sort by policy (decide overlap order)
            if overlap_policy == "largest":
                insts.sort(key=lambda d: d.get("area", 0), reverse=True)
            elif overlap_policy == "conf":
                insts.sort(key=lambda d: d.get("pred_conf", 0.0), reverse=True)

            # label image (Cityscapes classes are 0..18 => uint8 is enough)
                        # label image (Cityscapes classes are 0..18 => uint8 is enough)
            label = np.full((H, W), IGNORE_LABEL, dtype=np.uint8)
            conf_map = np.zeros((H, W), dtype=np.float32)

            # --- per-instance rasterization with confidence-aware overwrite ---
            for it in insts:
                m = decode_rle(it["rle"]).astype(bool)
                cls = int(it.get("pred_idx", -1))
                if cls < 0 or cls >= NUM_CLASSES:
                    continue

                prob = float(it.get("pred_conf", 0.0))

                # Only overwrite pixels where this instance is *more confident*
                overwrite = m & (prob > conf_map)

                label[overwrite]    = cls
                conf_map[overwrite] = prob
  # only write where SAM mask says there is something


            out_path = os.path.join(out_dir, Path(fname).with_suffix(".png").name)
            Image.fromarray(label).save(out_path)

            if i % 50 == 0:
                print(f"Processed {i} images...")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overlap_policy", default="largest", choices=["largest", "conf"])
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--min_area", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=0,
                    help="Optional limit on number of images to process (0 = all)")
    args = ap.parse_args()

    convert_jsonl_to_semantic(
        jsonl_path=args.jsonl,
        out_dir=args.out_dir,
        overlap_policy=args.overlap_policy,
        min_conf=args.min_conf,
        min_area=args.min_area,
        max_images=args.max_images,
    )
