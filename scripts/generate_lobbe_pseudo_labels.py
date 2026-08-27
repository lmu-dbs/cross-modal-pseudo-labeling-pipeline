#!/usr/bin/env python3
"""
generate_lobbe_pseudo_labels.py

Generate Lobbe pseudo-labels for the waste segmentation pipeline in one run:
  1) EVA-CLIP only pseudo-labels
  2) EVA-CLIP + BLIP pseudo-labels for masks that EVA-CLIP is uncertain about

Default behavior is intentionally aligned with the paper/thesis setup:
  - Uses the IOSB fine-tuned EVA-CLIP checkpoint.
  - Uses only Lobbe target images, not Lobbe GT.
  - Limits generation to 500 Lobbe images by default.
  - Outputs 9-class masks: {0..8, 255}, where 0=background and 255=ignore.

Expected run from project root:
  python scripts/generate_lobbe_pseudo_labels.py

Expected output:
  camera_ready_runs/pseudo_labels/lobbe_eva_only/{imgs,lbls,meta}
  camera_ready_runs/pseudo_labels/lobbe_eva_blip/{imgs,lbls,meta}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Paths and imports
# -----------------------------------------------------------------------------

def infer_project_root() -> Path:
    """Infer project root when this file lives in <project>/scripts/."""
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parent.parent
    return Path.cwd().resolve()


def add_import_paths(project_root: Path) -> None:
    # Supports both import styles depending on how the repo is placed on PYTHONPATH.
    for p in [project_root, project_root.parent]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


PROJECT_ROOT_DEFAULT = Path("/home/wiss/shubhangi/da-seg/thesis_da_segmentation/sam_clip_full")
DATA_ROOT_DEFAULT = Path(
    "/home/wiss/shubhangi/home/scs_deal_projects_notapebackup/shared/DATASET/waste_dataset"
)
LOBBE_ROOT_DEFAULT = DATA_ROOT_DEFAULT / "lobbe_fix"
CKPT_DEFAULT = PROJECT_ROOT_DEFAULT / "camera_ready_runs/checkpoints/eva_clip_ft_iosb_packform9_FULL.pt"
OUT_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "camera_ready_runs/pseudo_labels"

SAM_CKPT_CANDIDATES = [
    PROJECT_ROOT_DEFAULT / "camera_ready_runs/checkpoints/sam_vit_h_4b8939.pth",
    PROJECT_ROOT_DEFAULT / "third_party/checkpoints/sam_vit_h_4b8939.pth",
    Path("/home/wiss/shubhangi/home/scs_deal_projects_notapebackup/user/shubhang/thesis/third_party/checkpoints/sam_vit_h_4b8939.pth"),
    Path("/home/scs_deal_projects_notapebackup/user/shubhang/thesis/third_party/checkpoints/sam_vit_h_4b8939.pth"),
]


# -----------------------------------------------------------------------------
# Ontology
# -----------------------------------------------------------------------------

CAT_NAMES = [
    "background",
    "bottle",
    "bag_film",
    "cup_tray",
    "lid_cap",
    "carton",
    "can",
    "foam",
    "other_packaging",
]
NUM_CLASSES = len(CAT_NAMES)
IGNORE_LABEL = 255

IOSB_SYNONYMS: Dict[str, List[str]] = {
    "background": ["background", "conveyor belt", "empty belt", "belt surface", "no object"],
    "bottle": ["a plastic bottle", "a PET bottle", "a water bottle", "a soda bottle"],
    "bag_film": ["a plastic bag", "plastic film", "shrink wrap", "wrapping film", "plastic wrapper"],
    "cup_tray": [
        "a plastic cup",
        "a food tray",
        "a plastic tray",
        "a takeaway container",
        "a food container",
        "a clamshell container",
    ],
    "lid_cap": ["a bottle cap", "a plastic cap", "a lid", "a container lid"],
    "carton": ["a beverage carton", "a milk carton", "a juice carton", "a liquid carton", "Tetra Pak carton"],
    "can": ["a metal can", "an aluminum can", "a soda can", "a tin can"],
    "foam": ["foam packaging", "styrofoam", "expanded polystyrene foam", "foam tray"],
    "other_packaging": ["other packaging", "miscellaneous packaging", "unknown packaging item"],
}

IOSB_TEMPLATES = [
    "a waste sorting image containing {}",
    "an industrial waste stream showing {}",
    "a conveyor-belt scene with {}",
    "a waste item: {}",
    "an image of {} in a waste sorting setting",
    "a piece of {} packaging on a conveyor belt",
]

DEFAULT_TEMPLATES_LOCAL = [
    "a photo of a {}.",
    "a cropped photo of a {}.",
    "a close-up of a {}.",
    "a bright photo of a {}.",
    "a low-resolution photo of a {}.",
]

BLIP_BG_PATTERNS = [
    r"\bconveyor\b",
    r"\bbelt\b",
    r"\bempty\b",
    r"\bbackground\b",
    r"\bno object\b",
]
BLIP_KEYWORDS = {
    "bottle": [r"\bbottle\b", r"\bpet\b"],
    "bag_film": [r"\bbag\b", r"\bfilm\b", r"\bwrap\b", r"\bwrapper\b"],
    "cup_tray": [r"\bcup\b", r"\btray\b", r"\bcontainer\b", r"\bclamshell\b"],
    "lid_cap": [r"\bcap\b", r"\blid\b"],
    "carton": [r"\bcarton\b", r"\btetra\b", r"\btetra pak\b"],
    "can": [r"\bcan\b", r"\baluminum\b", r"\btin\b", r"\bmetal\b"],
    "foam": [r"\bfoam\b", r"\bstyrofoam\b", r"\bpolystyrene\b", r"\beps\b"],
    "other_packaging": [r"\bpackaging\b", r"\bplastic\b"],
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def safe_stem(p: Path) -> str:
    return p.stem


def save_pseudo(out_dir: Path, img_path: Path, label_map: np.ndarray, meta: Dict[str, Any], copy_image: bool = True) -> None:
    stem = safe_stem(img_path)
    (out_dir / "imgs").mkdir(parents=True, exist_ok=True)
    (out_dir / "lbls").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    if copy_image:
        out_img = out_dir / "imgs" / f"{stem}.png"
        if not out_img.exists():
            shutil.copy2(img_path, out_img)

    out_lbl = out_dir / "lbls" / f"{stem}.png"
    Image.fromarray(label_map.astype(np.uint8)).save(out_lbl)

    out_meta = out_dir / "meta" / f"{stem}.json"
    with open(out_meta, "w") as f:
        json.dump(json_sanitize(meta), f, indent=2)


def sanitize_label_map(lbl: np.ndarray, num_classes: int = NUM_CLASSES, ignore: int = IGNORE_LABEL) -> np.ndarray:
    lbl = lbl.astype(np.int64)
    bad = (lbl != ignore) & ((lbl < 0) | (lbl >= num_classes))
    if bool(bad.any()):
        lbl[bad] = ignore
    return lbl.astype(np.uint8)


def mask_to_bbox(mask_bool: np.ndarray, pad: int = 4) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = mask_bool.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_image(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = map(int, bbox)
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    return pil_img.crop((x0, y0, x1, y1))


def bbox_area_frac(bbox: Tuple[int, int, int, int], H: int, W: int) -> float:
    x0, y0, x1, y1 = bbox
    return float(max(0, x1 - x0) * max(0, y1 - y0) / max(1, H * W))


def dynamic_alpha_beta_from_box_local(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    a_min: float = 0.05,
    a_max: float = 0.20,
    b_min: float = 0.05,
    b_max: float = 0.25,
) -> Tuple[float, float, float]:
    frac = bbox_area_frac(bbox, img_h, img_w)
    t = float(np.sqrt(np.clip(frac / 0.10, 0.0, 1.0)))
    alpha = a_max + (a_min - a_max) * t
    beta = b_max + (b_min - b_max) * t
    return float(alpha), float(beta), frac


def apply_soft_mask_gray(crop_pil: Image.Image, mask_crop: np.ndarray, bg_weight: float = 0.0) -> Image.Image:
    if mask_crop is None or mask_crop.size == 0:
        return crop_pil
    m = Image.fromarray((mask_crop.astype(np.uint8) * 255)).resize(crop_pil.size, Image.NEAREST)
    m_arr = np.array(m) > 127
    img = np.array(crop_pil).astype(np.float32)
    w = np.where(m_arr[..., None], 1.0, bg_weight)
    out = np.clip(img * w + 127.0 * (1.0 - w), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def make_blip_focus_crop(
    pil_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    mask_full_bool: np.ndarray,
    blur_radius: float = 4.0,
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    crop = crop_image(pil_img, bbox)
    mask_crop = mask_full_bool[y0:y1, x0:x1].astype(bool)
    if mask_crop.size == 0:
        return crop

    fg = np.array(crop).astype(np.uint8)
    bg = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bg_arr = np.array(bg).astype(np.float32)
    gray = np.full_like(bg_arr, 127.0)
    bg_arr = 0.5 * bg_arr + 0.5 * gray

    m = Image.fromarray((mask_crop.astype(np.uint8) * 255)).resize(crop.size, Image.NEAREST)
    m_arr = (np.array(m) > 127)[..., None]
    out = np.where(m_arr, fg, bg_arr).astype(np.uint8)
    return Image.fromarray(out)


def compute_uncertainty_from_logits(
    logits_np: np.ndarray,
    logit_scale_exp: float,
    temp_calib: float = 0.07,
) -> Tuple[int, int, float, float, float]:
    logits_np = np.asarray(logits_np, dtype=np.float64)
    cos = logits_np / max(float(logit_scale_exp), 1e-8)
    order = np.argsort(-cos)
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else int(order[0])
    margin_cos = float(cos[top1] - cos[top2])

    z = cos / max(float(temp_calib), 1e-8)
    z = z - z.max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)
    pcal_top1 = float(p[top1])
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return top1, top2, margin_cos, pcal_top1, entropy


def should_trigger_blip(
    margin_cos: float,
    pcal_top1: float,
    entropy: float,
    bbox_frac: float,
    min_bbox_frac: float,
    max_bbox_frac: float,
) -> bool:
    if bbox_frac > max_bbox_frac:
        return False
    if bbox_frac < min_bbox_frac:
        return False
    return (margin_cos <= 0.12) or (pcal_top1 <= 0.60) or (entropy >= 1.60)


def build_semantic_label_map(
    H: int,
    W: int,
    instances: List[Dict[str, Any]],
    ignore_label: int = IGNORE_LABEL,
    hard_ignore_masks: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    lbl = np.full((H, W), fill_value=ignore_label, dtype=np.uint8)
    score_map = np.full((H, W), fill_value=-1.0, dtype=np.float32)

    if hard_ignore_masks:
        hard = np.zeros((H, W), dtype=bool)
        for m in hard_ignore_masks:
            hard |= m.astype(bool)
        score_map[hard] = 1e9
        lbl[hard] = ignore_label

    for inst in instances:
        m = inst["mask"].astype(bool)
        cid = int(inst["class_id"])
        score = float(inst["score"])
        if cid < 0 or cid >= NUM_CLASSES:
            continue
        take = m & (score > score_map)
        lbl[take] = cid
        score_map[take] = score

    return sanitize_label_map(lbl)


def resize_np_max_side(np_img: np.ndarray, max_side: int) -> Tuple[np.ndarray, Tuple[int, int], float]:
    H, W = np_img.shape[:2]
    s = max(H, W)
    if s <= max_side:
        return np_img, (H, W), 1.0
    scale = max_side / float(s)
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    pil = Image.fromarray(np_img)
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.array(pil), (H, W), scale


def upsample_bool_mask(mask_bool: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    m = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    m = m.resize((Wt, Ht), resample=Image.NEAREST)
    return np.array(m) > 127


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def resolve_sam_ckpt(user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"SAM checkpoint was provided but does not exist: {p}")
    for p in SAM_CKPT_CANDIDATES:
        if p.exists():
            return p
    msg = "Could not find SAM checkpoint. Tried:\n" + "\n".join(f"  - {p}" for p in SAM_CKPT_CANDIDATES)
    msg += "\nPass it explicitly with --sam-ckpt /path/to/sam_vit_h_4b8939.pth"
    raise FileNotFoundError(msg)


def load_sam_generator(ckpt_path: Path, model_type: str, device: str, points_per_side: int, min_region_area: int):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    print(f"[SAM] Loading {model_type}: {ckpt_path}", flush=True)
    sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
    sam.to(device=device)
    sam.eval()

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=min_region_area,
    )
    return generator


def import_clip_components(project_root: Path):
    add_import_paths(project_root)
    try:
        from clip_module import load_eva_clip, EVACLIPWrapper
        return load_eva_clip, EVACLIPWrapper
    except Exception:
        from sam_clip_full.clip_module import load_eva_clip, EVACLIPWrapper
        return load_eva_clip, EVACLIPWrapper


def load_eva_finetuned(project_root: Path, ckpt_path: Path, device: str):
    load_eva_clip, EVACLIPWrapper = import_clip_components(project_root)
    print("[EVA] Loading EVA02-L-14 base model...", flush=True)
    clip_model, clip_preprocess, clip_tokenizer = load_eva_clip(
        device=device,
        model_name="EVA02-L-14",
        pretrained="merged2b_s4b_b131k",
        to_float32=False,
    )

    print(f"[EVA] Loading fine-tuned checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(torch.is_tensor(v) for v in ckpt.values()):
        sd = ckpt
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    missing = clip_model.load_state_dict(sd, strict=False)
    print("[EVA] load_state_dict missing/unexpected:", missing, flush=True)

    if isinstance(ckpt, dict) and "logit_scale_exp" in ckpt:
        with torch.no_grad():
            clip_model.logit_scale.copy_(torch.log(torch.tensor(float(ckpt["logit_scale_exp"]))))
        print("[EVA] Restored logit_scale_exp.", flush=True)

    eva = EVACLIPWrapper(
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        device=device,
        combine="add",
        embed_dim=1024,
    ).to(device).eval()
    return eva, clip_model, clip_preprocess, clip_tokenizer


def load_blip_model(device: str, model_name: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    print(f"[BLIP] Loading {model_name} on {device}...", flush=True)
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    return model, processor


# -----------------------------------------------------------------------------
# Embedding helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def encode_text_list(clip_model, clip_tokenizer, texts: Sequence[str], device: str) -> torch.Tensor:
    toks = clip_tokenizer(list(texts))
    if isinstance(toks, dict):
        toks = {k: v.to(device) for k, v in toks.items()}
    else:
        toks = toks.to(device)

    use_amp = device.startswith("cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        z = clip_model.encode_text(toks)
    z = F.normalize(z.float(), dim=-1)
    return z


@torch.no_grad()
def build_class_text_embeds(clip_model, clip_tokenizer, device: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Build one normalized text vector per class by mean-pooling synonym/template prompts."""
    class_vecs = []
    counts: Dict[str, int] = {}
    for cname in CAT_NAMES:
        phrases = IOSB_SYNONYMS.get(cname, [cname.replace("_", " ")])
        prompts = []
        templates = IOSB_TEMPLATES if cname != "background" else IOSB_TEMPLATES[:3]
        for phrase in phrases:
            prompts.extend([t.format(phrase) for t in templates])
        z = encode_text_list(clip_model, clip_tokenizer, prompts, device)
        zc = F.normalize(z.mean(dim=0, keepdim=True), dim=-1)
        class_vecs.append(zc)
        counts[cname] = len(prompts)
    text_embeds = torch.cat(class_vecs, dim=0)
    text_mean = F.normalize(text_embeds.mean(dim=0, keepdim=True), dim=-1)
    return text_embeds, text_mean, counts


@torch.no_grad()
def caption_to_class(
    caption: str,
    clip_model,
    clip_tokenizer,
    class_text_embeds: torch.Tensor,
    device: str,
    temp_calib: float,
) -> Tuple[int, str, float]:
    """Map BLIP caption to one of the fixed 9 waste classes."""
    if caption is None:
        return 0, "empty", 0.0
    s = caption.lower().strip()
    if not s:
        return 0, "empty", 0.0

    if any(re.search(p, s) for p in BLIP_BG_PATTERNS):
        return 0, "rule", 1.0

    hits = []
    for cname, pats in BLIP_KEYWORDS.items():
        if any(re.search(p, s) for p in pats):
            hits.append(cname)
    if len(hits) == 1:
        return int(CAT_NAMES.index(hits[0])), "rule", 1.0

    z = encode_text_list(clip_model, clip_tokenizer, [caption], device)
    sims = (z @ class_text_embeds.T).squeeze(0)
    probs = torch.softmax(sims / max(temp_calib, 1e-8), dim=0)
    cid = int(torch.argmax(probs).item())
    conf = float(probs[cid].item())
    return cid, "clip_text", conf


@torch.no_grad()
def blip_caption_batch(
    blip_model,
    blip_processor,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    prompt: Optional[str] = None,
) -> List[str]:
    if len(images) == 0:
        return []
    captions: List[str] = []
    for start in range(0, len(images), batch_size):
        batch = list(images[start : start + batch_size])
        if prompt is None:
            inputs = blip_processor(images=batch, return_tensors="pt")
        else:
            inputs = blip_processor(images=batch, text=[prompt] * len(batch), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = blip_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        caps = blip_processor.batch_decode(out, skip_special_tokens=True)
        captions.extend([c.strip() for c in caps])
    return captions


# -----------------------------------------------------------------------------
# SAM and classification
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_sam_masks(
    sam_generator,
    pil_img: Image.Image,
    device: str,
    max_side: int,
    max_masks: int,
    min_mask_area: int,
) -> List[np.ndarray]:
    np_img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    np_small, orig_hw, scale = resize_np_max_side(np_img, max_side=max_side)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    masks_raw = sam_generator.generate(np_small)

    candidates: List[Tuple[np.ndarray, float, int]] = []
    for item in masks_raw:
        seg = item.get("segmentation", None)
        if seg is None:
            continue
        m_small = np.asarray(seg, dtype=bool)
        m = upsample_bool_mask(m_small, orig_hw) if scale != 1.0 else m_small
        area = int(m.sum())
        if area < min_mask_area:
            continue
        score = float(item.get("predicted_iou", 0.0)) + float(item.get("stability_score", 0.0))
        candidates.append((m, score, area))

    # Prefer high SAM confidence, then larger masks. Cap to avoid excessive CLIP/BLIP work.
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    masks = [m for m, _, _ in candidates[:max_masks]]
    return masks


@torch.no_grad()
def classify_masks_with_clip(
    eva,
    clip_model,
    pil_img: Image.Image,
    masks_bool: Sequence[np.ndarray],
    class_text_embeds: torch.Tensor,
    text_mean: torch.Tensor,
    device: str,
) -> List[Dict[str, Any]]:
    W, H = pil_img.size
    results: List[Dict[str, Any]] = []

    for midx, mask in enumerate(masks_bool):
        mask = mask.astype(bool)
        bbox = mask_to_bbox(mask, pad=4)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < 4 or (y1 - y0) < 4:
            continue

        crop = crop_image(pil_img, bbox)
        mask_crop = mask[y0:y1, x0:x1]
        crop_soft = apply_soft_mask_gray(crop, mask_crop, bg_weight=0.0)
        alpha, beta, _ = dynamic_alpha_beta_from_box_local(bbox, W, H)

        logits_by_class, probs, *_ = eva.classify_crop_with_context_residual_topk(
            crop_pil=crop_soft,
            context_pil=pil_img,
            text_embeds=class_text_embeds,
            text_feat_mean=text_mean,
            alpha=alpha,
            beta=beta,
            cat_names=CAT_NAMES,
            templates=DEFAULT_TEMPLATES_LOCAL,
            topk_list=(1, 3, 5),
        )

        probs_np = probs.detach().cpu().numpy()
        logits_np = logits_by_class.detach().cpu().numpy()
        pred_idx = int(probs_np.argmax())
        results.append(
            {
                "mask_index": int(midx),
                "bbox": tuple(map(int, bbox)),
                "mask": mask,
                "logits": logits_np,
                "probs": probs_np,
                "max_prob": float(probs_np.max()),
                "pred_idx": pred_idx,
                "pred_label": CAT_NAMES[pred_idx],
            }
        )
    return results


# -----------------------------------------------------------------------------
# Pseudo-label generation per image
# -----------------------------------------------------------------------------

@dataclass
class PseudoConfig:
    max_masks_per_image: int = 15
    clip_keep_pcal: float = 0.85
    clip_keep_margin: float = 0.18
    max_blip_per_image: int = 6
    blip_min_conf_accept: float = 0.35
    blip_score_boost: float = 0.10
    blip_min_bbox_frac: float = 0.001
    blip_max_bbox_frac: float = 0.20
    temp_calib: float = 0.07


@torch.no_grad()
def build_clip_only_label(
    pil_img: Image.Image,
    masks_bool: Sequence[np.ndarray],
    clip_results: Sequence[Dict[str, Any]],
    logit_scale_exp: float,
    cfg: PseudoConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    W, H = pil_img.size
    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    for r in clip_results:
        top1, top2, margin, pcal, entropy = compute_uncertainty_from_logits(
            r["logits"], logit_scale_exp=logit_scale_exp, temp_calib=cfg.temp_calib
        )
        confident = (pcal >= cfg.clip_keep_pcal) and (margin >= cfg.clip_keep_margin)
        decisions.append(
            {
                "mask_index": r["mask_index"],
                "bbox": r["bbox"],
                "top1": CAT_NAMES[top1],
                "top2": CAT_NAMES[top2],
                "margin_cos": margin,
                "pcal_top1": pcal,
                "entropy": entropy,
                "kept": confident,
            }
        )
        if confident:
            kept.append(
                {
                    "mask": r["mask"],
                    "class_id": int(top1),
                    "score": float(pcal),
                    "bbox": r["bbox"],
                    "src": "eva_clip",
                }
            )

    kept = sorted(kept, key=lambda x: x["score"], reverse=True)[: cfg.max_masks_per_image]
    lbl = build_semantic_label_map(H, W, kept)
    meta = {
        "mode": "eva_only",
        "n_sam_raw": len(masks_bool),
        "n_clip_results": len(clip_results),
        "n_kept": len(kept),
        "cat_names": CAT_NAMES,
        "num_classes": NUM_CLASSES,
        "ignore_label": IGNORE_LABEL,
        "config": asdict(cfg),
        "decisions_preview": decisions[:20],
    }
    return lbl, meta


@torch.no_grad()
def build_clip_blip_label(
    pil_img: Image.Image,
    masks_bool: Sequence[np.ndarray],
    clip_results: Sequence[Dict[str, Any]],
    logit_scale_exp: float,
    cfg: PseudoConfig,
    blip_model,
    blip_processor,
    blip_device: str,
    blip_batch_size: int,
    blip_max_new_tokens: int,
    clip_model,
    clip_tokenizer,
    class_text_embeds: torch.Tensor,
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    W, H = pil_img.size

    clip_kept_by_mask: Dict[int, Dict[str, Any]] = {}
    blip_candidates: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    for r in clip_results:
        top1, top2, margin, pcal, entropy = compute_uncertainty_from_logits(
            r["logits"], logit_scale_exp=logit_scale_exp, temp_calib=cfg.temp_calib
        )
        bbox = r["bbox"]
        frac = bbox_area_frac(bbox, H, W)
        confident = (pcal >= cfg.clip_keep_pcal) and (margin >= cfg.clip_keep_margin)

        if confident:
            clip_kept_by_mask[int(r["mask_index"])] = {
                "mask": r["mask"],
                "class_id": int(top1),
                "score": float(pcal),
                "bbox": bbox,
                "src": "eva_clip",
            }
            route_blip = False
        else:
            route_blip = should_trigger_blip(
                margin_cos=margin,
                pcal_top1=pcal,
                entropy=entropy,
                bbox_frac=frac,
                min_bbox_frac=cfg.blip_min_bbox_frac,
                max_bbox_frac=cfg.blip_max_bbox_frac,
            )
            if route_blip:
                crop = make_blip_focus_crop(pil_img, bbox, r["mask"].astype(bool))
                blip_candidates.append(
                    {
                        "mask_index": int(r["mask_index"]),
                        "mask": r["mask"].astype(bool),
                        "bbox": bbox,
                        "crop": crop,
                        "clip_top1": int(top1),
                        "clip_top2": int(top2),
                        "clip_top1_name": CAT_NAMES[top1],
                        "clip_top2_name": CAT_NAMES[top2],
                        "margin_cos": float(margin),
                        "pcal_top1": float(pcal),
                        "entropy": float(entropy),
                        "bbox_frac": float(frac),
                    }
                )

        decisions.append(
            {
                "mask_index": r["mask_index"],
                "bbox": bbox,
                "clip_top1": CAT_NAMES[top1],
                "clip_top2": CAT_NAMES[top2],
                "margin_cos": margin,
                "pcal_top1": pcal,
                "entropy": entropy,
                "bbox_frac": frac,
                "clip_confident": confident,
                "routed_to_blip": route_blip,
            }
        )

    # Most uncertain first. The cap is per image so BLIP does not dominate runtime.
    blip_candidates = sorted(blip_candidates, key=lambda e: (e["margin_cos"], -e["entropy"]))[: cfg.max_blip_per_image]

    captions = blip_caption_batch(
        blip_model,
        blip_processor,
        [e["crop"] for e in blip_candidates],
        device=blip_device,
        batch_size=blip_batch_size,
        max_new_tokens=blip_max_new_tokens,
    )

    blip_instances: List[Dict[str, Any]] = []
    blip_meta: List[Dict[str, Any]] = []
    for e, caption in zip(blip_candidates, captions):
        mapped_idx, source, conf = caption_to_class(
            caption,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            class_text_embeds=class_text_embeds,
            device=device,
            temp_calib=cfg.temp_calib,
        )
        accepted = float(conf) >= cfg.blip_min_conf_accept
        blip_meta.append(
            {
                "mask_index": e["mask_index"],
                "bbox": e["bbox"],
                "caption": caption,
                "mapped_idx": int(mapped_idx),
                "mapped_label": CAT_NAMES[int(mapped_idx)],
                "mapping_source": source,
                "mapping_conf": float(conf),
                "accepted": bool(accepted),
                "clip_top1": e["clip_top1_name"],
                "clip_top2": e["clip_top2_name"],
                "clip_pcal_top1": e["pcal_top1"],
                "clip_margin_cos": e["margin_cos"],
            }
        )
        if accepted:
            blip_instances.append(
                {
                    "mask": e["mask"],
                    "class_id": int(mapped_idx),
                    "score": min(1.0, float(conf) + cfg.blip_score_boost),
                    "bbox": e["bbox"],
                    "src": "blip_rescue",
                    "caption": caption,
                }
            )

    merged = list(clip_kept_by_mask.values()) + blip_instances
    merged = sorted(merged, key=lambda x: x["score"], reverse=True)[: cfg.max_masks_per_image]
    lbl = build_semantic_label_map(H, W, merged)
    meta = {
        "mode": "eva_blip",
        "n_sam_raw": len(masks_bool),
        "n_clip_results": len(clip_results),
        "n_clip_kept": len(clip_kept_by_mask),
        "n_blip_candidates": len(blip_candidates),
        "n_blip_accepted": len(blip_instances),
        "n_final_merged": len(merged),
        "cat_names": CAT_NAMES,
        "num_classes": NUM_CLASSES,
        "ignore_label": IGNORE_LABEL,
        "config": asdict(cfg),
        "decisions_preview": decisions[:20],
        "blip_preview": blip_meta[:20],
    }
    return lbl, meta


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Generate Lobbe EVA-only and EVA+BLIP pseudo-labels")
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    ap.add_argument("--lobbe-root", type=Path, default=LOBBE_ROOT_DEFAULT)
    ap.add_argument("--img-dir", type=Path, default=None, help="Defaults to <lobbe-root>/val_imgs")
    ap.add_argument("--eva-ckpt", type=Path, default=CKPT_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--pattern", type=str, default="*.png")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-copy-images", action="store_true")

    ap.add_argument("--sam-ckpt", type=str, default=None)
    ap.add_argument("--sam-model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--sam-points-per-side", type=int, default=16)
    ap.add_argument("--sam-max-side", type=int, default=640)
    ap.add_argument("--sam-max-masks", type=int, default=60)
    ap.add_argument("--min-mask-area", type=int, default=800)

    ap.add_argument("--max-masks-per-image", type=int, default=15)
    ap.add_argument("--clip-keep-pcal", type=float, default=0.85)
    ap.add_argument("--clip-keep-margin", type=float, default=0.18)
    ap.add_argument("--max-blip-per-image", type=int, default=6)
    ap.add_argument("--blip-min-conf-accept", type=float, default=0.35)
    ap.add_argument("--temp-calib", type=float, default=0.07)

    ap.add_argument("--blip-device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--blip-model", type=str, default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--blip-batch-size", type=int, default=4)
    ap.add_argument("--blip-max-new-tokens", type=int, default=15)

    ap.add_argument("--allow-cpu", action="store_true", help="Only for debugging. Full generation should use CUDA.")
    ap.add_argument("--fast-dev-run", action="store_true", help="Process only 3 images and loosen runtime checks.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    img_dir = args.img_dir if args.img_dir is not None else args.lobbe_root / "val_imgs"
    img_dir = img_dir.resolve()
    out_eva = args.out_root / "lobbe_eva_only"
    out_blip = args.out_root / "lobbe_eva_blip"

    add_import_paths(project_root)

    print("[PATH] project_root:", project_root, "exists=", project_root.exists(), flush=True)
    print("[PATH] img_dir:", img_dir, "exists=", img_dir.exists(), flush=True)
    print("[PATH] eva_ckpt:", args.eva_ckpt, "exists=", args.eva_ckpt.exists(), flush=True)
    print("[PATH] out_eva:", out_eva, flush=True)
    print("[PATH] out_blip:", out_blip, flush=True)

    if not project_root.exists():
        raise FileNotFoundError(project_root)
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    if not args.eva_ckpt.exists():
        raise FileNotFoundError(f"EVA checkpoint missing: {args.eva_ckpt}")

    print("[TORCH]", torch.__version__, "cuda_build=", torch.version.cuda, flush=True)
    print("[CUDA] available=", torch.cuda.is_available(), "count=", torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available() and not (args.allow_cpu or args.fast_dev_run):
        raise RuntimeError("CUDA is not available. Run through sbatch with a GPU, or pass --allow-cpu only for debugging.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.fast_dev_run:
        args.max_images = min(args.max_images, 3)
        args.sam_points_per_side = min(args.sam_points_per_side, 8)
        args.sam_max_masks = min(args.sam_max_masks, 10)
        args.max_blip_per_image = min(args.max_blip_per_image, 2)
        print("[FAST_DEV_RUN] max_images=", args.max_images, flush=True)

    for d in [out_eva, out_blip]:
        (d / "imgs").mkdir(parents=True, exist_ok=True)
        (d / "lbls").mkdir(parents=True, exist_ok=True)
        (d / "meta").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(img_dir.glob(args.pattern))
    if args.start_index > 0:
        image_paths = image_paths[args.start_index :]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    print(f"[DATA] Selected {len(image_paths)} images from {img_dir} pattern={args.pattern}", flush=True)
    if len(image_paths) == 0:
        raise RuntimeError("No images selected.")
    print("[DATA] First image:", image_paths[0], flush=True)

    sam_ckpt = resolve_sam_ckpt(args.sam_ckpt)
    sam_generator = load_sam_generator(
        sam_ckpt,
        model_type=args.sam_model_type,
        device=device,
        points_per_side=args.sam_points_per_side,
        min_region_area=args.min_mask_area,
    )

    eva, clip_model, _clip_preprocess, clip_tokenizer = load_eva_finetuned(project_root, args.eva_ckpt.resolve(), device)
    class_text_embeds, text_mean, prompt_counts = build_class_text_embeds(clip_model, clip_tokenizer, device)
    print("[TEXT] class_text_embeds:", tuple(class_text_embeds.shape), "prompt_counts=", prompt_counts, flush=True)

    blip_device = args.blip_device
    if blip_device == "cuda" and not torch.cuda.is_available():
        blip_device = "cpu"
    blip_model, blip_processor = load_blip_model(blip_device, args.blip_model)

    cfg = PseudoConfig(
        max_masks_per_image=args.max_masks_per_image,
        clip_keep_pcal=args.clip_keep_pcal,
        clip_keep_margin=args.clip_keep_margin,
        max_blip_per_image=args.max_blip_per_image,
        blip_min_conf_accept=args.blip_min_conf_accept,
        temp_calib=args.temp_calib,
    )
    print("[CONFIG]", asdict(cfg), flush=True)

    summary = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_dir": str(img_dir),
        "num_images_selected": len(image_paths),
        "eva_ckpt": str(args.eva_ckpt),
        "sam_ckpt": str(sam_ckpt),
        "cat_names": CAT_NAMES,
        "num_classes": NUM_CLASSES,
        "ignore_label": IGNORE_LABEL,
        "config": asdict(cfg),
        "processed": 0,
        "failed": [],
    }

    copy_images = not args.no_copy_images
    for img_path in tqdm(image_paths, desc="Lobbe pseudo-labels"):
        stem = safe_stem(img_path)
        eva_lbl_path = out_eva / "lbls" / f"{stem}.png"
        blip_lbl_path = out_blip / "lbls" / f"{stem}.png"
        if (not args.overwrite) and eva_lbl_path.exists() and blip_lbl_path.exists():
            summary["processed"] += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            W, H = pil_img.size
            masks_bool = generate_sam_masks(
                sam_generator,
                pil_img,
                device=device,
                max_side=args.sam_max_side,
                max_masks=args.sam_max_masks,
                min_mask_area=args.min_mask_area,
            )
            clip_results = classify_masks_with_clip(
                eva,
                clip_model,
                pil_img,
                masks_bool,
                class_text_embeds,
                text_mean,
                device,
            )
            logit_scale_exp = float(eva.model.logit_scale.exp().detach().cpu().item())

            eva_lbl, eva_meta = build_clip_only_label(
                pil_img,
                masks_bool,
                clip_results,
                logit_scale_exp,
                cfg,
            )
            eva_meta.update({"image_path": str(img_path), "width": W, "height": H})
            save_pseudo(out_eva, img_path, eva_lbl, eva_meta, copy_image=copy_images)

            blip_lbl, blip_meta = build_clip_blip_label(
                pil_img,
                masks_bool,
                clip_results,
                logit_scale_exp,
                cfg,
                blip_model,
                blip_processor,
                blip_device,
                args.blip_batch_size,
                args.blip_max_new_tokens,
                clip_model,
                clip_tokenizer,
                class_text_embeds,
                device,
            )
            blip_meta.update({"image_path": str(img_path), "width": W, "height": H})
            save_pseudo(out_blip, img_path, blip_lbl, blip_meta, copy_image=copy_images)

            summary["processed"] += 1
        except Exception as e:
            err = {"image_path": str(img_path), "error": repr(e)}
            print("[ERROR]", err, flush=True)
            summary["failed"].append(err)
            # Continue overnight rather than killing the whole job on one bad file.
            continue

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_path = args.out_root / "lobbe_pseudo_generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(json_sanitize(summary), f, indent=2)

    print("[DONE] Processed:", summary["processed"], "failed:", len(summary["failed"]), flush=True)
    print("[DONE] EVA-only labels:", out_eva / "lbls", flush=True)
    print("[DONE] EVA+BLIP labels:", out_blip / "lbls", flush=True)
    print("[DONE] Summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
