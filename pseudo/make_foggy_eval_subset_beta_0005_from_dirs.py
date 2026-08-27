# sam_clip_full/pseudo/make_foggy_eval_subset_beta_0005_from_dirs.py

from pathlib import Path
import os
import shutil

CITY = Path("/home/scs_deal_projects_notapebackup/shared/DATASET/cityscapes")

# Pseudo-labels (your semantic PNGs from convert_jsonl_to_semantic)
PSEUDO_DIR = CITY / "seg_data_foggy_beta_0005"

# GT foggy labels (498 train + 52 val)
GT_ROOT = CITY / "seg_data_foggy"

OUT_ROOT = CITY / "seg_eval_foggy_beta_0005"
PRED_OUT = OUT_ROOT / "pred"
GT_OUT   = OUT_ROOT / "gt"

PRED_OUT.mkdir(parents=True, exist_ok=True)
GT_OUT.mkdir(parents=True, exist_ok=True)


def scene_id_from_gt(name: str) -> str:
    """
    Turn something like
      'frankfurt_000000_000576_gtFine_labelTrainIds.png'
    or
      'frankfurt_000000_000576_leftImg8bit_foggy.png'
    into
      'frankfurt_000000_000576'
    """
    if "_gtFine_labelTrainIds" in name:
        return name.split("_gtFine_labelTrainIds")[0]
    if "_leftImg8bit_foggy" in name:
        return name.split("_leftImg8bit_foggy")[0]
    # fallback: first 3 tokens (city + 2 indices)
    parts = name.split("_")
    return "_".join(parts[:3])


def scene_id_from_pseudo(name: str) -> str:
    """
    Turn something like
      'frankfurt_000000_000576_leftImg8bit_foggy_beta_0.005.png'
    into
      'frankfurt_000000_000576'
    """
    if "_leftImg8bit_foggy" in name:
        return name.split("_leftImg8bit_foggy")[0]
    # fallback: first 3 tokens
    parts = name.split("_")
    return "_".join(parts[:3])


def main():
    # 1) Build GT index by scene id
    gt_index = {}
    for p in GT_ROOT.rglob("*.png"):
        sid = scene_id_from_gt(p.name)
        gt_index[sid] = p
    print(f"Found {len(gt_index)} GT PNGs under {GT_ROOT}")

    # 2) Walk all pseudo PNGs and match via scene id
    n_pseudo = 0
    n_used = 0
    n_missing_gt = 0

    for pseudo_path in sorted(PSEUDO_DIR.glob("*.png")):
        n_pseudo += 1
        sid = scene_id_from_pseudo(pseudo_path.name)
        gt_path = gt_index.get(sid, None)
        if gt_path is None:
            n_missing_gt += 1
            # uncomment to debug:
            # print("[WARN] no GT for pseudo:", pseudo_path.name, "scene:", sid)
            continue

        # Use original pseudo filename so eval_pseudo_vs_gt can just match by name
        name = pseudo_path.name
        pred_dest = PRED_OUT / name
        gt_dest   = GT_OUT   / name

        try:
            if not pred_dest.exists():
                os.symlink(pseudo_path, pred_dest)
            if not gt_dest.exists():
                os.symlink(gt_path, gt_dest)
        except OSError:
            if not pred_dest.exists():
                shutil.copy2(pseudo_path, pred_dest)
            if not gt_dest.exists():
                shutil.copy2(gt_path, gt_dest)

        n_used += 1

    print("Total pseudo PNGs seen:    ", n_pseudo)
    print("Pairs used (pseudo + GT): ", n_used)
    print("Pseudo without GT match:  ", n_missing_gt)
    print("Pred dir:", PRED_OUT)
    print("GT dir:  ", GT_OUT)


if __name__ == "__main__":
    main()
