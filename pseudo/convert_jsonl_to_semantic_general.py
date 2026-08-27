#!/usr/bin/env python3
import os, json
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle).astype(np.uint8)  # HxW

def convert_jsonl_to_semantic(
    jsonl_path: str,
    out_dir: str,
    num_classes: int,
    ignore_label: int = 255,
    overlap_policy: str = "conf",     # "largest" or "conf"
    min_conf: float = 0.0,
    min_area: int = 0,
    max_images: int = 0,
    out_ext: str = ".png",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f, start=1):
            if max_images > 0 and i > max_images:
                print(f"Reached max_images={max_images}, stopping.")
                break

            rec = json.loads(line)
            fname = rec["file_name"]
            W, H = int(rec["width"]), int(rec["height"])
            insts = rec.get("instances", [])

            # filter
            insts = [
                d for d in insts
                if float(d.get("pred_conf", 1.0)) >= float(min_conf)
                and int(d.get("area", 0)) >= int(min_area)
            ]

            # sort for overlap order
            if overlap_policy == "largest":
                insts.sort(key=lambda d: int(d.get("area", 0)), reverse=True)
            elif overlap_policy == "conf":
                insts.sort(key=lambda d: float(d.get("pred_conf", 0.0)), reverse=True)
            else:
                raise ValueError("overlap_policy must be 'largest' or 'conf'")

            label = np.full((H, W), int(ignore_label), dtype=np.uint8)
            conf_map = np.zeros((H, W), dtype=np.float32)

            for it in insts:
                cls = int(it.get("pred_idx", -1))
                if cls < 0 or cls >= int(num_classes):
                    continue

                prob = float(it.get("pred_conf", 0.0))
                m = decode_rle(it["rle"]).astype(bool)

                # confidence-aware overwrite (best practice)
                overwrite = m & (prob > conf_map)
                label[overwrite] = cls
                conf_map[overwrite] = prob

            out_path = out_dir / (Path(fname).stem + out_ext)
            Image.fromarray(label).save(out_path)

            if i % 50 == 0:
                print(f"Converted {i} images...")

    print(f"[DONE] Wrote semantic masks to: {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--ignore_label", type=int, default=255)
    ap.add_argument("--overlap_policy", default="conf", choices=["largest", "conf"])
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--min_area", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    convert_jsonl_to_semantic(
        jsonl_path=args.jsonl,
        out_dir=args.out_dir,
        num_classes=args.num_classes,
        ignore_label=args.ignore_label,
        overlap_policy=args.overlap_policy,
        min_conf=args.min_conf,
        min_area=args.min_area,
        max_images=args.max_images,
    )
