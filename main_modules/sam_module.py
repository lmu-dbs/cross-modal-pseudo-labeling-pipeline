"""
sam_module.py
SAM1 (segment-anything) + SAM2 (facebookresearch/sam2) wrapper for automatic mask generation.

- load_sam_predictor(..., backend="sam1"|"sam2"|"auto")
- sam_predict_masks(predictor, np_image) -> List[np.ndarray] (uint8 masks in {0,1})
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import os
import glob
import numpy as np
import torch

# -------------------------
# SAM1 imports (optional)
# -------------------------
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except Exception:
    sam_model_registry, SamPredictor, SamAutomaticMaskGenerator = None, None, None

# -------------------------
# SAM2 imports (optional)
# -------------------------
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import sam2 as _sam2_pkg
except Exception:
    build_sam2, SAM2AutomaticMaskGenerator, _sam2_pkg = None, None, None


# ====== SAM1 prompt predictor (optional utility) ==============================

def load_sam(
    model_type: str = "vit_h",
    checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> "SamPredictor":
    if sam_model_registry is None:
        raise ImportError("segment-anything is not installed. `pip install segment-anything`")
    if checkpoint is None:
        raise ValueError("Please provide a SAM1 checkpoint path (.pth).")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)


class SamWrapper:
    def __init__(self, predictor: "SamPredictor"):
        self.predictor = predictor

    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray) -> None:
        self.predictor.set_image(image_rgb)

    @torch.no_grad()
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        boxes_xyxy: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = True,
    ) -> Dict[str, Any]:
        masks_list, ious_list, logits_list = [], [], []
        if boxes_xyxy is None:
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
            masks_list.append(masks)
            ious_list.append(scores)
            if return_logits:
                logits_list.append(logits)
        else:
            for box in boxes_xyxy:
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box.astype(np.float32),
                    multimask_output=multimask_output,
                )
                masks_list.append(masks)
                ious_list.append(scores)
                if return_logits:
                    logits_list.append(logits)

        masks_out = np.concatenate(masks_list, axis=0) if masks_list else np.zeros((0, 1, 1), dtype=bool)
        ious_out  = np.concatenate(ious_list, axis=0) if ious_list  else np.zeros((0,), dtype=np.float32)
        logits_out = np.concatenate(logits_list, axis=0) if (logits_list and return_logits) else None

        return {
            "masks": masks_out.astype(bool),
            "iou_preds": ious_out.astype(np.float32),
            "low_res_logits": logits_out.astype(np.float32) if logits_out is not None else None,
        }


# ====== Automatic mask generation (SAM1 + SAM2) ==============================

DEFAULT_SAM1_CHECKPOINT = "/home/scs_deal_projects_notapebackup/user/shubhang/thesis/third_party/checkpoints/sam_vit_h_4b8939.pth"

# IMPORTANT: SAM2 "large" uses the "..._hiera_l.yaml" config name (NOT "..._large.yaml")
DEFAULT_SAM2_CFG_NAME   = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _resolve_ckpt(checkpoint: Optional[str], default_ckpt: str) -> str:
    if checkpoint is not None:
        return checkpoint
    env_ckpt = os.environ.get("SAM_CHECKPOINT", "")
    if env_ckpt:
        return env_ckpt
    if default_ckpt:
        return default_ckpt
    raise ValueError("No checkpoint provided. Pass `checkpoint=...` or set SAM_CHECKPOINT env var.")


def _sam2_pkg_root() -> Optional[str]:
    if _sam2_pkg is None:
        return None
    # sam2.__path__[0] points to ".../site-packages/sam2" (or editable src)
    return list(_sam2_pkg.__path__)[0]


def _rewrite_large_cfg_name(cfg: str) -> str:
    # Users often pass "..._large.yaml" but SAM2 uses "..._l.yaml"
    # e.g. sam2.1_hiera_large.yaml -> sam2.1_hiera_l.yaml
    if "hiera_large" in cfg:
        cfg = cfg.replace("hiera_large", "hiera_l")
    if cfg.endswith("_large.yaml"):
        cfg = cfg.replace("_large.yaml", "_l.yaml")
    return cfg


def _resolve_sam2_cfg(cfg: Optional[str]) -> str:
    """
    Returns a Hydra config name that SAM2 expects, typically:
      'configs/sam2.1/sam2.1_hiera_l.yaml'

    Accepts:
      - None -> DEFAULT_SAM2_CFG_NAME
      - 'sam2.1_hiera_l.yaml' (filename) -> 'configs/sam2.1/<filename>'
      - absolute path containing '/configs/...' -> normalized to 'configs/...'
      - mistaken '..._large.yaml' -> rewritten to '..._l.yaml'
    """
    if cfg is None or str(cfg).strip() == "":
        return DEFAULT_SAM2_CFG_NAME

    cfg = str(cfg).strip()
    cfg = _rewrite_large_cfg_name(cfg)

    # absolute path? normalize to the 'configs/...' suffix if possible
    if os.path.isabs(cfg):
        if "configs" + os.sep in cfg:
            cfg = cfg[cfg.find("configs" + os.sep):].replace(os.sep, "/")
        else:
            raise ValueError(
                f"SAM2 cfg absolute path must contain a 'configs/' segment so we can normalize it. Got: {cfg}"
            )

    # already a hydra-ish name containing configs/
    if "configs/" in cfg:
        cfg = cfg[cfg.find("configs/"):]
    else:
        # just a filename -> assume sam2.1 family by default
        if cfg.endswith(".yaml") and ("/" not in cfg):
            cfg = "configs/sam2.1/" + cfg
        else:
            raise ValueError(
                f"SAM2 config must be a Hydra config name like 'configs/...yaml' (got: {cfg})."
            )

    # sanity check existence on disk (best-effort; works for editable installs too)
    root = _sam2_pkg_root()
    if root is not None:
        candidate = os.path.join(root, cfg)
        if not os.path.isfile(candidate):
            # show what exists to stop guessing
            hint_dir = os.path.join(root, "configs", "sam2.1")
            avail = sorted(glob.glob(os.path.join(hint_dir, "*.yaml"))) if os.path.isdir(hint_dir) else []
            avail = [os.path.basename(a) for a in avail][:20]
            raise FileNotFoundError(
                f"SAM2 config not found on disk: {candidate}\n"
                f"Tip: for sam2.1_hiera_large.pt use: configs/sam2.1/sam2.1_hiera_l.yaml\n"
                f"Available in {hint_dir}: {avail}"
            )

    return cfg


def _build_sam2_notebook_safe(cfg_name: str, ckpt: str, device: str):
    """
    build_sam2() uses Hydra compose/instantiate internally.
    In notebooks, Hydra is often uninitialized, so we initialize it from the sam2 package directory.
    """
    if build_sam2 is None:
        raise ImportError("sam2 is not installed (facebookresearch/sam2). `pip install -e` it and restart kernel.")

    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir

    def _build():
        return build_sam2(cfg_name, ckpt, device=device)

    gh = GlobalHydra.instance()
    if not gh.is_initialized():
        root = _sam2_pkg_root()
        if root is None or not os.path.isdir(root):
            raise RuntimeError("Could not locate sam2 package root to initialize Hydra.")
        # IMPORTANT: point Hydra to the package root (it contains ./configs/...)
        with initialize_config_dir(config_dir=root, version_base=None):
            return _build()

    # already initialized -> try directly
    try:
        return _build()
    except Exception:
        # If Hydra was initialized with some other search path, re-init to sam2 root.
        try:
            gh.clear()
        except Exception:
            pass
        root = _sam2_pkg_root()
        from hydra import initialize_config_dir
        with initialize_config_dir(config_dir=root, version_base=None):
            return _build()


def load_sam_predictor(
    device: str = "cuda",
    model_type: str = "vit_h",                 # SAM1 only
    checkpoint: Optional[str] = None,          # .pth for sam1, .pt for sam2
    backend: str = "auto",                     # "auto"|"sam1"|"sam2"
    sam2_model_cfg: Optional[str] = None,      # "configs/sam2.1/....yaml"
    # generator knobs (shared-ish)
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.90,
    stability_score_thresh: float = 0.96,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 2,
    min_mask_region_area: int = 800,
    box_nms_thresh: float = 0.7,
):
    backend = backend.lower().strip()

    if backend == "auto":
        backend = "sam2" if (build_sam2 is not None and SAM2AutomaticMaskGenerator is not None) else "sam1"

    if backend == "sam1":
        if sam_model_registry is None or SamAutomaticMaskGenerator is None:
            raise ImportError("segment-anything is not installed. `pip install segment-anything`")
        ckpt = _resolve_ckpt(checkpoint, DEFAULT_SAM1_CHECKPOINT)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"SAM1 checkpoint not found: {ckpt}")
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device)
        return SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            box_nms_thresh=box_nms_thresh,
        )

    if backend == "sam2":
        if build_sam2 is None or SAM2AutomaticMaskGenerator is None:
            raise ImportError("sam2 is not installed (facebookresearch/sam2). `pip install -e` it and restart kernel.")
        if checkpoint is None:
            raise ValueError("SAM2 requires an explicit checkpoint (.pt). Pass checkpoint=...")

        ckpt = checkpoint
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")

        cfg_name = _resolve_sam2_cfg(sam2_model_cfg)

        # notebook-safe Hydra init + build
        model = _build_sam2_notebook_safe(cfg_name, ckpt, device=device)

        return SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            box_nms_thresh=box_nms_thresh,
        )

    raise ValueError(f"Unknown backend='{backend}'. Use auto|sam1|sam2")


def _bool_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _filter_raw_masks(
    masks_raw: list,
    max_keep: int = 60,
    min_area: int = 800,
    iou_thresh: float = 0.8,
):
    masks_sorted = sorted(masks_raw, key=lambda m: m.get("area", 0), reverse=True)
    kept = []

    for m in masks_sorted:
        seg = m.get("segmentation", None)
        area = m.get("area", 0)
        if seg is None or area < min_area:
            continue

        seg_arr = np.array(seg, dtype=bool)
        too_close = False
        for km in kept:
            if _bool_iou(seg_arr, km["segmentation_bool"]) > iou_thresh:
                too_close = True
                break
        if too_close:
            continue

        kept.append({**m, "segmentation_bool": seg_arr})
        if len(kept) >= max_keep:
            break

    return kept


def sam_predict_masks(predictor, np_image: np.ndarray) -> List[np.ndarray]:
    if np_image.dtype != np.uint8:
        np_image = np_image.astype(np.uint8)
    masks_raw = predictor.generate(np_image)
    filtered = _filter_raw_masks(masks_raw, max_keep=60, min_area=800, iou_thresh=0.8)
    return [np.array(m["segmentation_bool"], dtype=np.uint8) for m in filtered]
