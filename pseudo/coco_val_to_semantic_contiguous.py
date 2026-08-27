#coco_val_to_semantic_contiguous.py
import os
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

def ann_to_mask(coco, ann, h, w):
    m = coco.annToMask(ann)
    if m is None:
        x,y,bw,bh = ann.get("bbox", [0,0,0,0])
        x0,y0 = int(x), int(y)
        x1,y1 = min(w, int(x+bw)), min(h, int(y+bh))
        m = np.zeros((h,w), dtype=np.uint8)
        if x1>x0 and y1>y0: m[y0:y1, x0:x1] = 1
    return m.astype(np.uint8)

def build_name_to_idx_from_val(coco_val: COCO):
    cats = coco_val.loadCats(coco_val.getCatIds())
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    labels = [c["name"] for c in cats_sorted]
    return {name: i for i, name in enumerate(labels)}

def to_semantic_png_contiguous(images_dir, ann_file, out_dir, min_area=0, overlap="largest"):
    os.makedirs(out_dir, exist_ok=True)
    coco = COCO(ann_file)
    name_to_idx = build_name_to_idx_from_val(coco)  # 0..79 contiguous to match your runner
    img_ids = coco.getImgIds()
    for i, img_id in enumerate(img_ids, 1):
        info = coco.loadImgs([img_id])[0]
        w, h = info["width"], info["height"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=None))
        keep = [a for a in anns if int(a.get("area",0)) >= min_area]
        if overlap == "largest":
            keep.sort(key=lambda a: a.get("area",0), reverse=True)
        label = np.zeros((h, w), dtype=np.uint16)
        for a in keep:
            cat = coco.loadCats([a["category_id"]])[0]
            cls = name_to_idx[cat["name"]]
            m = ann_to_mask(coco, a, h, w).astype(bool)
            label[m] = cls
        out_path = os.path.join(out_dir, Path(info["file_name"]).with_suffix(".png").name)
        Image.fromarray(label).save(out_path)
        if i % 200 == 0: print(f"Saved {i}/{len(img_ids)} GT masks")
    print("Done writing GT semantic maps.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ann_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_area", type=int, default=0)
    ap.add_argument("--overlap", type=str, default="largest")
    args = ap.parse_args()
    to_semantic_png_contiguous(args.images_dir, args.ann_file, args.out_dir, args.min_area, args.overlap)
