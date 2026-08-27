"""
sam_module.py
Lightweight wrapper around Meta's Segment Anything (SAM) predictor for mask generation.
Supports point and/or box prompts and returns masks + logits for downstream scoring.

Extended with:
    - load_sam_predictor(device, model_type, checkpoint)
    - sam_predict_masks(predictor, np_image)
for annotation-free pseudo-label generation.
"""

from typing import List, Tuple, Optional, Dict, Any
import os
import numpy as np
import torch

try:
    # Official SAM package (pip install segment-anything)
    from segment_anything import (
        sam_model_registry,
        SamPredictor,
        SamAutomaticMaskGenerator,
    )
except Exception:
    sam_model_registry, SamPredictor, SamAutomaticMaskGenerator = None, None, None


# ===== Original API (kept) ===================================================

def load_sam(
    model_type: str = "vit_h",
    checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> "SamPredictor":
    """
    Load a SAM predictor (prompt-based).
    Args:
        model_type: "vit_h" | "vit_l" | "vit_b"
        checkpoint: path to SAM checkpoint .pth
        device: "cuda" or "cpu"
    """
    if sam_model_registry is None:
        raise ImportError("segment-anything is not installed. Please `pip install segment-anything`")
    if checkpoint is None:
        raise ValueError("Please provide a checkpoint path for SAM.")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


class SamWrapper:
    """
    Thin convenience layer around SamPredictor for batched prompt -> mask generation.
    """
    def __init__(self, predictor: "SamPredictor"):
        self.predictor = predictor

    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray) -> None:
        """
        Set the current image.
        image_rgb: HxWx3 uint8 RGB
        """
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
        """
        Run SAM with optional points and/or boxes.
        Returns:
            dict with keys:
                - masks: (N, H, W) boolean ndarray
                - iou_preds: (N,) IoU predictions from SAM
                - low_res_logits: (N, 256, 256) float32 (if return_logits)
        """
        masks_list, ious_list, logits_list = [], [], []
        # Case 1: only points
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
            # Case 2: iterate over boxes (SAM expects per-box call)
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

        masks_out = (
            np.concatenate(masks_list, axis=0)
            if len(masks_list) > 0
            else np.zeros((0, 1, 1), dtype=bool)
        )
        ious_out = (
            np.concatenate(ious_list, axis=0)
            if len(ious_list) > 0
            else np.zeros((0,), dtype=np.float32)
        )
        logits_out = (
            np.concatenate(logits_list, axis=0)
            if (len(logits_list) > 0 and return_logits)
            else None
        )

        return {
            "masks": masks_out.astype(bool),
            "iou_preds": ious_out.astype(np.float32),
            "low_res_logits": logits_out.astype(np.float32)
            if logits_out is not None
            else None,
        }


# ===== New: automatic mask generator for unlabeled images ====================

# Set this to your actual SAM checkpoint OR use env var SAM_CHECKPOINT.
DEFAULT_SAM_CHECKPOINT = "/home/scs_deal_projects_notapebackup/user/shubhang/thesis/third_party/checkpoints/sam_vit_h_4b8939.pth"


def _resolve_sam_checkpoint(checkpoint: Optional[str]) -> str:
    """
    Resolve SAM checkpoint:
      1) explicit argument
      2) SAM_CHECKPOINT env var
      3) DEFAULT_SAM_CHECKPOINT
    """
    if checkpoint is not None:
        return checkpoint
    env_ckpt = os.environ.get("SAM_CHECKPOINT", "")
    if env_ckpt:
        return env_ckpt
    if DEFAULT_SAM_CHECKPOINT:
        return DEFAULT_SAM_CHECKPOINT
    raise ValueError(
        "No SAM checkpoint provided. Set `checkpoint` arg, "
        "SAM_CHECKPOINT env var, or DEFAULT_SAM_CHECKPOINT."
    )


def load_sam_predictor(
    device: str = "cuda",
    model_type: str = "vit_h",
    checkpoint: Optional[str] = None,
) -> "SamAutomaticMaskGenerator":
    """
    Load a SAM AutomaticMaskGenerator for annotation-free mask proposals.
    This is what's used for pseudo labels on target domains.
    """
    if sam_model_registry is None or SamAutomaticMaskGenerator is None:
        raise ImportError(
            "segment-anything is not installed. Please `pip install segment-anything`"
        )

    ckpt = _resolve_sam_checkpoint(checkpoint)
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )
    return mask_generator


def sam_predict_masks(
    predictor: "SamAutomaticMaskGenerator",
    np_image: np.ndarray,
) -> List[np.ndarray]:
    """
    Run SAM automatic mask generator on an image.
    Returns a list of binary masks (H, W) in {0,1}.
    """
    if np_image.dtype != np.uint8:
        np_image = np_image.astype(np.uint8)

    masks_raw = predictor.generate(np_image)
    masks: List[np.ndarray] = []
    for m in masks_raw:
        seg = m.get("segmentation", None)
        if seg is None:
            continue
        seg_arr = np.array(seg, dtype=np.uint8)  # {0,1}
        masks.append(seg_arr)
    return masks
