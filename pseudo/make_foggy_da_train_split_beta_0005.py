# sam_clip_full/pseudo/make_foggy_da_train_split_beta_0005.py

import os, json, shutil
from pathlib import Path

def main():
    CITY = Path("/home/scs_deal_projects_notapebackup/shared/DATASET/cityscapes")
    FOG_ROOT = CITY / "leftImg8bit_foggyDBF"              # original foggy images (train/val)
    JSONL = CITY / "pseudo_foggy_beta_0005" / "pseudo_labels.jsonl"
    PSEUDO_SEM = CITY / "seg_data_foggy_beta_0005"        # semantic pseudo PNGs (flat)

    OUT_ROOT = CITY / "seg_da_foggy_beta_0005"
    IMG_OUT = OUT_ROOT / "train_imgs"
    LBL_OUT = OUT_ROOT / "train_lbls"
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    LBL_OUT.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_missing_img = 0
    n_missing_lbl = 0

    with open(JSONL, "r") as f:
        for line in f:
            rec = json.loads(line)
            fname_rel = rec["file_name"]  # e.g. "train/aachen/aachen_000000_000019_leftImg8bit_foggy_beta_0.005.png"
            base = Path(fname_rel).name

            src_img = FOG_ROOT / fname_rel
            src_lbl = PSEUDO_SEM / base  # convert_jsonl_to_semantic used just base name

            if not src_img.is_file():
                n_missing_img += 1
                continue
            if not src_lbl.is_file():
                n_missing_lbl += 1
                continue

            dst_img = IMG_OUT / base
            dst_lbl = LBL_OUT / base

            if not dst_img.is_file():
                shutil.copy2(src_img, dst_img)
            if not dst_lbl.is_file():
                shutil.copy2(src_lbl, dst_lbl)

            n_total += 1

    print("Foggy DA train split (beta=0.005)")
    print(f"  JSONL:           {JSONL}")
    print(f"  Images copied:   {n_total}")
    print(f"  Missing images:  {n_missing_img}")
    print(f"  Missing labels:  {n_missing_lbl}")
    print(f"  IMG_OUT:         {IMG_OUT}")
    print(f"  LBL_OUT:         {LBL_OUT}")

if __name__ == "__main__":
    main()
