import numpy as np
from PIL import Image
from pathlib import Path

# ---- Cityscapes official mapping: labelId -> trainId ----
ID2TRAINID = {
    0: 255,   # unlabeled
    1: 255,   # ego vehicle
    2: 255,   # rectif. border
    3: 255,   # out of roi
    4: 255,   # static
    5: 255,   # dynamic
    6: 255,   # ground
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
}

def remap_to_trainid(mask: np.ndarray) -> np.ndarray:
    """Map Cityscapes labelIds (0–33) to trainIds (0–18, 255)."""
    out = np.full_like(mask, 255, dtype=np.uint8)
    for id_val, t_val in ID2TRAINID.items():
        out[mask == id_val] = t_val
    return out

def convert_dir(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(src_dir.glob("*.png"))
    print(f"Converting {len(paths)} files in {src_dir} -> {dst_dir}")
    for p in paths:
        mask = np.array(Image.open(p))
        mask_t = remap_to_trainid(mask)
        out_path = dst_dir / p.name
        Image.fromarray(mask_t).save(out_path)

def main():
    # absolute base for Cityscapes on your machine
    city_base = Path("/home/scs_deal_projects_notapebackup/shared/DATASET/cityscapes")

    src_root = city_base / "seg_data"
    src_train_lbls = src_root / "train_lbls"
    src_val_lbls   = src_root / "val_lbls"

    # new root for 19-class labels (will be created if missing)
    dst_root = city_base / "seg_data_trainId"
    dst_train_lbls = dst_root / "train_lbls"
    dst_val_lbls   = dst_root / "val_lbls"

    convert_dir(src_train_lbls, dst_train_lbls)
    convert_dir(src_val_lbls,   dst_val_lbls)

if __name__ == "__main__":
    main()
