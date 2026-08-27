#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Optional: needed for "label_studio_converter.brush.decode_rle" style masks
try:
    from label_studio_converter.brush import decode_rle as ls_decode_rle
except Exception:
    ls_decode_rle = None


# ----------------------------
# Label mapping (0..5)
# ----------------------------
CLS_BG = 0
CLS_PET = 1
CLS_PE  = 2
CLS_PP  = 3
CLS_PS  = 4
CLS_OTH = 5

PE_TYPES  = {"PE", "LDPE", "HDPE", "LLDPE"}
PS_TYPES  = {"PS", "EPS"}  # EPS is polystyrene family
PP_TYPES  = {"PP"}
PET_TYPES = {"PET"}


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def decode_napari_rle2mask_runs(runs) -> np.ndarray:
    """
    Napari rle2mask_runs format:
      [H, W, start0, len0, start1, len1, ...]
    start is flat index into row-major array (H*W).
    Returns boolean mask (H,W).
    """
    arr = np.asarray(runs, dtype=np.int64).ravel()
    if arr.size < 4:
        return np.zeros((1, 1), dtype=bool)

    H = int(arr[0])
    W = int(arr[1])
    pairs = arr[2:]

    if pairs.size % 2 != 0:
        pairs = pairs[:-1]

    total = H * W
    if total <= 0:
        return np.zeros((1, 1), dtype=bool)

    flat = np.zeros(total, dtype=np.uint8)

    starts = pairs[0::2]
    lens   = pairs[1::2]

    starts = np.clip(starts, 0, max(total - 1, 0))
    lens   = np.clip(lens, 0, total)

    for s, ln in zip(starts, lens):
        s = int(s)
        e = min(total, int(s + ln))
        if e > s:
            flat[s:e] = 1

    return flat.reshape((H, W)).astype(bool)


def decode_labelstudio_brush(rle_list, h: int, w: int) -> np.ndarray:
    """
    label_studio_converter.brush.decode_rle often returns:
      - flat RGBA uint8 of size h*w*4  -> use alpha channel > 0
      - OR flat mask of size h*w
    Returns boolean mask (h,w).
    """
    if ls_decode_rle is None:
        raise RuntimeError("label_studio_converter is not installed but JSON requests it.")

    flat = np.asarray(ls_decode_rle(rle_list))

    if flat.size == h * w * 4:
        rgba = flat.reshape((h, w, 4))
        return (rgba[..., 3] > 0)

    if flat.size == h * w:
        m = flat.reshape((h, w))
        return (m > 0)

    raise ValueError(f"Unexpected decode_rle output size {flat.size} (expected {h*w} or {h*w*4}).")


def decode_instance_mask(j: dict, inst: dict, H_img: int, W_img: int) -> np.ndarray | None:
    """
    Decode one instance mask to image-space boolean mask (H_img, W_img).
    Uses j['mask decoding'] to pick the decoder.
    """
    rle = inst.get("mask")
    if not isinstance(rle, list) or len(rle) < 4:
        return None

    mode = (j.get("mask decoding") or "").strip()

    if mode == "read_and_write.rle2mask_runs":
        m = decode_napari_rle2mask_runs(rle)

    elif mode == "label_studio_converter.brush.decode_rle":
        h = int(j.get("image_height", H_img))
        w = int(j.get("image_width",  W_img))
        m = decode_labelstudio_brush(rle, h=h, w=w)

    else:
        # Infer:
        # - Napari starts with [H,W,...] like [4096,4096,...]
        # - LabelStudio encoding often looks like bytes [0..255,...]
        a0, a1 = int(rle[0]), int(rle[1])
        if 64 <= a0 <= 10000 and 0 <= a1 <= 10000:
            m = decode_napari_rle2mask_runs(rle)
        else:
            h = int(j.get("image_height", H_img))
            w = int(j.get("image_width",  W_img))
            m = decode_labelstudio_brush(rle, h=h, w=w)

    # Resize to image space if needed
    if m.shape != (H_img, W_img):
        m_img = Image.fromarray((m.astype(np.uint8) * 255))
        m_img = m_img.resize((W_img, H_img), resample=Image.NEAREST)
        m = (np.array(m_img) > 0)

    return m


def extract_plastic_type(inst: dict) -> str | None:
    """
    inst['annotations'] is a list of dicts like:
      {'worker id': '...', 'label tree': {...}}

    We pick Plastic Type Primary > Plastic Type across any worker.
    Returns e.g. 'PP','PET','LDPE','HDPE','PS','EPS', or None.
    """
    ann_list = inst.get("annotations", [])
    if not isinstance(ann_list, list):
        return None

    candidates = []
    for a in ann_list:
        if not isinstance(a, dict):
            continue
        lt = a.get("label tree")
        if not isinstance(lt, dict):
            continue

        for key in ("Plastic Type Primary", "Plastic Type"):
            val = lt.get(key, [])
            if isinstance(val, list) and val:
                for v in val:
                    if isinstance(v, str) and v.strip():
                        candidates.append(v.strip().upper())

    for tset in (PET_TYPES, PP_TYPES, PS_TYPES, PE_TYPES):
        for c in candidates:
            if c in tset:
                return c

    return candidates[0] if candidates else None


def map_type_to_class(plastic_type: str | None) -> int:
    if plastic_type is None:
        return CLS_OTH

    t = plastic_type.upper()
    if t in PET_TYPES:
        return CLS_PET
    if t in PE_TYPES:
        return CLS_PE
    if t in PP_TYPES:
        return CLS_PP
    if t in PS_TYPES:
        return CLS_PS
    return CLS_OTH


def make_overlay(img_rgb: np.ndarray, sem: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Colored overlay per class for quick visual checking.
    sem is (H,W) uint8 in {0..5}.
    """
    colors = {
        CLS_PET: np.array([255,  80,  80], dtype=np.float32),
        CLS_PE:  np.array([ 80, 255,  80], dtype=np.float32),
        CLS_PP:  np.array([ 80,  80, 255], dtype=np.float32),
        CLS_PS:  np.array([255, 255,  80], dtype=np.float32),
        CLS_OTH: np.array([255,  80, 255], dtype=np.float32),
    }

    out = img_rgb.astype(np.float32).copy()
    for cls_id, col in colors.items():
        m = (sem == cls_id)
        if m.any():
            out[m] = out[m] * (1 - alpha) + col * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, type=Path)
    ap.add_argument("--ann_dir", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--prefix", required=True, type=str, help="Namespace prefix, e.g. Vollstrom6")
    ap.add_argument("--demo_n", type=int, default=20,
                    help="Save debug/overlay for first N *saved pairs* (all images+masks always saved).")
    ap.add_argument("--save_debug50", action="store_true")
    ap.add_argument("--save_overlay", action="store_true")
    args = ap.parse_args()

    out_images  = args.out_root / "images"
    out_masks   = args.out_root / "masks"
    out_debug50 = args.out_root / "debug50"
    out_overlay = args.out_root / "overlays"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    if args.save_debug50:
        out_debug50.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        out_overlay.mkdir(parents=True, exist_ok=True)

    img_map  = {p.stem: p for p in sorted(args.img_dir.glob("*.png"))}
    json_map = {p.stem: p for p in sorted(args.ann_dir.glob("*.json"))}

    common = sorted(set(img_map.keys()) & set(json_map.keys()))
    missing_json = sorted(set(img_map.keys()) - set(json_map.keys()))
    missing_img  = sorted(set(json_map.keys()) - set(img_map.keys()))

    print(f"Images in IMG_DIR: {len(img_map)}")
    print(f"JSONs in ANN_DIR:  {len(json_map)}")
    print(f"Common basenames:  {len(common)}")
    print(f"Images missing JSON: {len(missing_json)}")
    print(f"JSONs missing image: {len(missing_img)}")

    saved_pairs = 0
    demo_saved = 0
    empty_masks = 0
    decode_fail = 0
    no_instances = 0

    cls_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    for base in common:
        img_path = img_map[base]
        jp = json_map[base]

        img = np.array(Image.open(img_path).convert("RGB"))
        H_img, W_img = img.shape[:2]

        j = load_json(jp)
        masks = j.get("masks", [])
        if not isinstance(masks, list) or len(masks) == 0:
            empty_masks += 1
            continue

        sem = np.zeros((H_img, W_img), dtype=np.uint8)

        # policy: draw OTHER first, then PET/PE/PP/PS overwrite on top
        order = [CLS_OTH, CLS_PET, CLS_PE, CLS_PP, CLS_PS]

        decoded_instances = []
        for inst in masks:
            try:
                mb = decode_instance_mask(j, inst, H_img, W_img)
                if mb is None or not mb.any():
                    continue
                plastic_type = extract_plastic_type(inst)
                cls_id = map_type_to_class(plastic_type)
                decoded_instances.append((cls_id, mb))
            except Exception:
                decode_fail += 1
                continue

        if len(decoded_instances) == 0:
            no_instances += 1
            continue

        for cls_id in order:
            for cid, mb in decoded_instances:
                if cid == cls_id:
                    sem[mb] = np.uint8(cls_id)

        u, c = np.unique(sem, return_counts=True)
        for ui, ci in zip(u.tolist(), c.tolist()):
            cls_counts[int(ui)] += int(ci)

        out_base = f"{args.prefix}__{base}"

        # ALWAYS save training pair
        Image.fromarray(img).save(out_images / f"{out_base}.png")
        Image.fromarray(sem).save(out_masks / f"{out_base}.png")
        saved_pairs += 1

        # Save only first demo_n pairs for debug/overlay
        if args.demo_n > 0 and demo_saved < args.demo_n:
            if args.save_debug50:
                dbg = (sem.astype(np.uint16) * 50).clip(0, 255).astype(np.uint8)
                Image.fromarray(dbg).save(out_debug50 / f"{out_base}_debug50.png")
            if args.save_overlay:
                over = make_overlay(img, sem, alpha=0.45)
                Image.fromarray(over).save(out_overlay / f"{out_base}_overlay.png")
            demo_saved += 1

    print(f"Saved image+mask pairs: {saved_pairs}")
    print(f"Demo overlays/debug saved: {demo_saved} (limit={args.demo_n})")
    print(f"JSONs with empty 'masks' list: {empty_masks}")
    print(f"JSONs with masks but no decodable foreground instances: {no_instances}")
    print(f"Decode failures: {decode_fail}")
    print("Pixel counts by class (over processed images):")
    print(f"  0 bg:    {cls_counts[0]}")
    print(f"  1 PET:   {cls_counts[1]}")
    print(f"  2 PE:    {cls_counts[2]}")
    print(f"  3 PP:    {cls_counts[3]}")
    print(f"  4 PS:    {cls_counts[4]}")
    print(f"  5 other: {cls_counts[5]}")
    print(f"Output root: {args.out_root}")


if __name__ == "__main__":
    main()
