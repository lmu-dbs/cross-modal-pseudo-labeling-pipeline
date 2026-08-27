import numpy as np
from PIL import Image
from pathlib import Path

ID2TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   8: 1,
    9: 255, 10: 255,
    11: 2,  12: 3,  13: 4,
    14: 255, 15: 255, 16: 255,
    17: 5,  18: 255,
    19: 6,  20: 7,
    21: 8,  22: 9,
    23: 10,
    24: 11, 25: 12,
    26: 13, 27: 14, 28: 15,
    29: 255, 30: 255,
    31: 16, 32: 17, 33: 18,
}

def remap_to_trainid(mask: np.ndarray) -> np.ndarray:
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
        Image.fromarray(mask_t).save(dst_dir / p.name)

def main():
    city_base = Path("/home/scs_deal_projects_notapebackup/shared/DATASET/cityscapes")
    src_root = city_base / "seg_data_foggy"

    src_train_lbls = src_root / "train_lbls"
    src_val_lbls   = src_root / "val_lbls"

    dst_root = city_base / "seg_data_foggy_trainId"
    dst_train_lbls = dst_root / "train_lbls"
    dst_val_lbls   = dst_root / "val_lbls"

    convert_dir(src_train_lbls, dst_train_lbls)
    convert_dir(src_val_lbls,   dst_val_lbls)

if __name__ == "__main__":
    main()
