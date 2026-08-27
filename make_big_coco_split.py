import os
import random
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# --------- CONFIG ---------
COCO_ROOT = "/home/scs_deal_projects_notapebackup/shared/DATASET/coco2014/coco2014"
ANN_DIR   = os.path.join(COCO_ROOT, "annotations")
TRAIN_IMGS_DIR = os.path.join(COCO_ROOT, "train2014")
VAL_IMGS_DIR   = os.path.join(COCO_ROOT, "val2014")

OUT_ROOT = "/home/scs_deal_projects_notapebackup/shared/DATASET/coco2014/thesis_outputs/big_20k"

# how big you want the split
N_TRAIN = 20000
N_VAL   = 4000

RANDOM_SEED = 42
# --------------------------


def build_cat_mapping(coco):
    """
    Map COCO category_ids (e.g. [1,2,3,5,7,...,90]) -> contiguous [1..K].
    Background will be 0.
    """
    cats = coco.loadCats(coco.getCatIds())
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i + 1 for i, c in enumerate(cats_sorted)}
    print(f"Num categories: {len(cat_id_to_idx)}")
    return cat_id_to_idx


def make_split(split_name, ann_file, img_dir, out_img_dir, out_lbl_dir, n_samples):
    print(f"\n=== Processing {split_name} ===")
    print("Annotations:", ann_file)
    print("Images dir:", img_dir)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    coco = COCO(ann_file)
    cat_id_to_idx = build_cat_mapping(coco)

    # get all image ids that have at least one segmentation annotation
    img_ids = coco.getImgIds()
    valid_img_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue
        # keep if at least one ann has segmentation
        if any(a.get("segmentation") for a in anns):
            valid_img_ids.append(img_id)

    print(f"Total images with segmentation: {len(valid_img_ids)}")

    random.seed(RANDOM_SEED)
    random.shuffle(valid_img_ids)

    # choose subset
    n_use = min(n_samples, len(valid_img_ids))
    chosen_ids = valid_img_ids[:n_use]
    print(f"Using {n_use} images for {split_name}")

    for i, img_id in enumerate(chosen_ids, 1):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]  # e.g. COCO_train2014_000000000009.jpg
        width, height = img_info["width"], img_info["height"]

        # load annotations
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # create empty mask: 0 = background
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue
            cat_id = ann["category_id"]
            cls_idx = cat_id_to_idx[cat_id]  # in [1..K]
            m = coco.annToMask(ann)  # binary mask
            # simple overwrite; last object wins for overlapping regions
            mask[m == 1] = cls_idx

        # save image (copy) and mask
        src_img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(src_img_path):
            print(f"WARNING: image not found: {src_img_path}, skipping")
            continue

        dst_img_path = os.path.join(out_img_dir, file_name)
        dst_lbl_name = os.path.splitext(file_name)[0] + ".png"
        dst_lbl_path = os.path.join(out_lbl_dir, dst_lbl_name)

        # copy image by re-saving (you could also use shutil.copy2)
        img = Image.open(src_img_path).convert("RGB")
        img.save(dst_img_path)

        lbl_img = Image.fromarray(mask)
        lbl_img.save(dst_lbl_path)

        if i % 500 == 0:
            print(f"{split_name}: processed {i}/{n_use} images")

    print(f"Done {split_name}.")


def main():
    # Train split
    train_ann = os.path.join(ANN_DIR, "instances_train2014.json")
    make_split(
        split_name="train",
        ann_file=train_ann,
        img_dir=TRAIN_IMGS_DIR,
        out_img_dir=os.path.join(OUT_ROOT, "train_imgs"),
        out_lbl_dir=os.path.join(OUT_ROOT, "train_lbls"),
        n_samples=N_TRAIN,
    )

    # Val split
    val_ann = os.path.join(ANN_DIR, "instances_val2014.json")
    make_split(
        split_name="val",
        ann_file=val_ann,
        img_dir=VAL_IMGS_DIR,
        out_img_dir=os.path.join(OUT_ROOT, "val_imgs"),
        out_lbl_dir=os.path.join(OUT_ROOT, "val_lbls"),
        n_samples=N_VAL,
    )

    print("\nAll done. Big split is at:", OUT_ROOT)


if __name__ == "__main__":
    main()
