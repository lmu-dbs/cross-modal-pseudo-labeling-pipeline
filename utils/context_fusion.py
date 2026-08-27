# sam_clip_full/utils/context_fusion.py

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# ========== Defaults / imports from clip_module ==========
try:
    from sam_clip_full.clip_module import DEFAULT_TEMPLATES
except Exception:
    DEFAULT_TEMPLATES = [
        "a photo of a {}.",
        "a cropped photo of a {}.",
        "a close-up of a {}.",
        "a bright photo of a {}.",
        "a low-resolution photo of a {}.",
    ]


# ========== Geometry / box helpers ==========
def mask_to_tight_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def pad_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int,
    ratio: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    """Pad an xyxy box by a multiplicative ratio, clamp to image bounds."""
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w2 = w * (1.0 + ratio)
    h2 = h * (1.0 + ratio)
    nxmin = int(max(0, np.floor(cx - w2 / 2)))
    nymin = int(max(0, np.floor(cy - h2 / 2)))
    nxmax = int(min(W - 1, np.ceil(cx + w2 / 2)))
    nymax = int(min(H - 1, np.ceil(cy + h2 / 2)))
    return nxmin, nymin, nxmax, nymax


def crop_image(img_pil: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    """Inclusive xyxy → PIL crop (ensure at least 1×1)."""
    xmin, ymin, xmax, ymax = box
    if (xmax - xmin) < 1:
        xmax = xmin + 1
    if (ymax - ymin) < 1:
        ymax = ymin + 1
    # Note: PIL crop uses half-open [x1, y1, x2, y2)
    return img_pil.crop((xmin, ymin, xmax, ymax))


# ========== Soft mask weighting ==========
def apply_soft_mask_weighting(
    crop_pil: Image.Image,
    mask_crop: np.ndarray,
    eps: float = 1e-6
) -> Image.Image:
    """
    Emphasize FG by multiplying with a soft mask in [0,1].
    Robust to off-by-one shape mismatches by resizing the mask with NEAREST.
    FG: 1.0 ×, BG: 0.5 ×
    """
    crop = np.asarray(crop_pil).astype(np.float32)  # H, W, 3
    H, W = crop.shape[:2]

    # mask to float in [0,1], ensure (H, W, 1)
    m = (mask_crop > 0).astype(np.float32)
    if m.ndim == 2:
        m = m[..., None]
    mh, mw = m.shape[:2]
    if (mh, mw) != (H, W):
        m_img = Image.fromarray((m.squeeze(-1) > 0.5).astype(np.uint8) * 255)
        m_img = m_img.resize((W, H), resample=Image.NEAREST)
        m = (np.array(m_img, dtype=np.uint8) / 255.0)[..., None].astype(np.float32)

    out = crop * (0.5 + 0.5 * m)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ========== Dynamic α / β from box size ==========
def _area_frac(xmin: int, ymin: int, xmax: int, ymax: int, W: int, H: int) -> float:
    boxA = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    return float(boxA) / float(max(1, W * H))


def _map_linear(x: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    x = np.clip(x, lo, hi)
    t = 0.0 if hi - lo < 1e-8 else (x - lo) / (hi - lo)
    return float(out_lo + t * (out_hi - out_lo))


def dynamic_alpha_from_box(
    det_xyxy, img_w: int, img_h: int,
    a_min: float = 0.05, a_max: float = 0.20,
    lo: float = 0.01, hi: float = 0.20
) -> float:
    """Smaller objects → bigger α (more context)."""
    xmin, ymin, xmax, ymax = map(int, det_xyxy)
    af = _area_frac(xmin, ymin, xmax, ymax, img_w, img_h)
    return float(_map_linear(af, lo, hi, a_min, a_max))


def dynamic_beta_from_box(
    det_xyxy, img_w: int, img_h: int,
    b_min: float = 0.05, b_max: float = 0.25,
    lo: float = 0.01, hi: float = 0.20
) -> float:
    """Smaller objects → bigger β (stronger residual against text-mean)."""
    xmin, ymin, xmax, ymax = map(int, det_xyxy)
    af = _area_frac(xmin, ymin, xmax, ymax, img_w, img_h)
    return float(_map_linear(af, lo, hi, b_min, b_max))


def dynamic_alpha_beta_from_box(
    det_xyxy, img_w: int, img_h: int,
    a_min: float = 0.05, a_max: float = 0.20,
    b_min: float = 0.05, b_max: float = 0.25,
    lo: float = 0.01, hi: float = 0.20,
):
    """
    Convenience wrapper: returns (alpha, beta, area_frac)
    Smaller objects => larger alpha/beta.
    """
    xmin, ymin, xmax, ymax = map(int, det_xyxy)
    af = _area_frac(xmin, ymin, xmax, ymax, img_w, img_h)
    alpha = float(_map_linear(af, lo, hi, a_min, a_max))
    beta  = float(_map_linear(af, lo, hi, b_min, b_max))
    return alpha, beta, af


# ========== Thin EVA fusion wrapper ==========
class ContextResidualClassifier:
    """
    Thin helper around EVACLIPWrapper to do context-residual fusion:
        z_fused = norm(z_crop + α·z_ctx + β·(z_ctx − text_mean))

    Works with text embeddings shaped [C, D] ("mean" layout) OR [(C*T), D] ("stack" layout).
    If you pass stacked embeddings, we infer C from len(templates).
    """

    def __init__(
        self,
        eva_wrapper,                           # EVACLIPWrapper instance
        text_embeds: torch.Tensor,             # [C, D] or [(C*T), D] (already normalized)
        text_feat_mean: Optional[torch.Tensor] = None,  # [1, D] (normalized); optional
        templates: Optional[List[str]] = None,
        layout: str = "mean"                   # "mean" or "stack" (used only for inference of C)
    ):
        self.eva = eva_wrapper
        self.text_embeds = text_embeds
        self.templates = templates or DEFAULT_TEMPLATES
        self.layout = layout

        # Precompute / normalize text mean if not provided
        if text_feat_mean is None:
            self.text_feat_mean = self.eva.compute_text_mean(self.text_embeds)
        else:
            self.text_feat_mean = F.normalize(text_feat_mean.float(), dim=-1)

        # Infer class count C for template pooling downstream
        self._C = self._infer_C(self.text_embeds, self.templates, self.layout)
        # Create a stub cat_names of length C so the wrapper can pool templates
        self._cat_names_stub = [f"class_{i}" for i in range(self._C)]

    @staticmethod
    def _infer_C(text_embeds: torch.Tensor, templates: List[str], layout: str) -> int:
        rows = int(text_embeds.shape[0])
        T = max(1, len(templates))
        if layout == "stack":
            # Expect rows == C*T
            if rows % T != 0:
                # Fallback: best guess
                return rows
            return rows // T
        # layout == "mean" (or unknown): rows == C
        return rows

    @torch.no_grad()
    def classify(
        self,
        crop_pil: Image.Image,
        context_pil: Image.Image,
        alpha: float = 0.15,
        beta: float = 0.15
    ) -> Tuple[int, float, float, float]:
        """Top-1 classification with residual fusion."""
        logits_c, probs_c, topk_dict, a_used, b_used = self.eva.classify_crop_with_context_residual_topk(
            crop_pil, context_pil,
            self.text_embeds, self.text_feat_mean,
            alpha=alpha, beta=beta,
            cat_names=self._cat_names_stub,      # ensures correct C for template pooling
            templates=self.templates,
            topk_list=(1,)
        )
        pred_idx = topk_dict[1][0]
        # probs_c already softmaxed over classes
        prob = float(probs_c[pred_idx].item())
        return int(pred_idx), prob, float(a_used), float(b_used)

    @torch.no_grad()
    def classify_topk(
        self,
        crop_pil: Image.Image,
        context_pil: Image.Image,
        topk_list: Tuple[int, ...] = (1, 3, 5),
        alpha: float = 0.15,
        beta: float = 0.15
    ):
        """Top-k classification with residual fusion."""
        return self.eva.classify_crop_with_context_residual_topk(
            crop_pil, context_pil,
            self.text_embeds, self.text_feat_mean,
            alpha=alpha, beta=beta,
            cat_names=self._cat_names_stub,      # ensures correct C for template pooling
            templates=self.templates,
            topk_list=topk_list
        )


__all__ = [
    "mask_to_tight_bbox",
    "pad_bbox",
    "crop_image",
    "apply_soft_mask_weighting",
    "dynamic_alpha_from_box",
    "dynamic_beta_from_box",
    "ContextResidualClassifier",
    "dynamic_alpha_beta_from_box"
]
