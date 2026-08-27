# pseudo/eval_pseudo_vs_gt.py
import os, json
from pathlib import Path
import numpy as np
from PIL import Image

def per_class_iou(pred, gt, num_classes):
    ious = []
    for c in range(num_classes):
        p = (pred == c); g = (gt == c)
        inter = np.logical_and(p,g).sum()
        union = np.logical_or(p,g).sum()
        if union == 0: ious.append(np.nan)
        else: ious.append(inter/union)
    return ious

def main(pred_dir, gt_dir, num_classes, out_json):
    names = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])
    all_ious = []
    for fn in names:
        p = np.array(Image.open(os.path.join(pred_dir, fn)), dtype=np.int64)
        g = np.array(Image.open(os.path.join(gt_dir, fn)), dtype=np.int64)
        if p.shape != g.shape: 
            continue
        all_ious.append(per_class_iou(p, g, num_classes))
    arr = np.array(all_ious)
    class_iou = np.nanmean(arr, axis=0).tolist()
    miou = float(np.nanmean(class_iou))
    with open(out_json, "w") as f:
        json.dump({"miou": miou, "class_iou": class_iou}, f, indent=2)
    print(f"Pseudo vs GT mIoU: {miou:.4f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    main(args.pred_dir, args.gt_dir, args.num_classes, args.out_json)
