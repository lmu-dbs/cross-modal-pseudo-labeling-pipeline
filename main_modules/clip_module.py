"""
clip_module.py
EVA-CLIP wrapper (open_clip) for classifying crops with:
 - padding utilities
 - dynamic α/β from box size
 - context + context-residual fusion
 - robust template pooling (supports [C,D] and [C*T,D])
 - load/save finetuned checkpoints (incl. logit_scale)
"""
#sam_clip_full/clip_module.py
from typing import List, Tuple, Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    import open_clip
except Exception:
    open_clip = None


# -------------------------
# Defaults
# -------------------------
DEFAULT_TEMPLATES = [
    "a photo of a {}.",
    "a cropped photo of a {}.",
    "a close-up of a {}.",
    "a bright photo of a {}.",
    "a low-resolution photo of a {}.",
]


# -------------------------
# Model I/O
# -------------------------
def load_eva_clip(
    device: str = "cuda",
    model_name: str = "EVA02-L-14",         # <- use the same you finetuned
    pretrained: str = "merged2b_s4b_b131k", # or tag you used
    to_float32: bool = False,
):
    if open_clip is None:
        raise ImportError("open_clip is not installed. Please `pip install open-clip-torch`.")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_token(model_name) if hasattr(open_clip, "get_token") else open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    if to_float32:
        model = model.float()
    return model, preprocess, tokenizer


def load_finetuned_checkpoint(model: nn.Module, ckpt_path: str):
    """
    Loads a finetuned state dict (and restores logit_scale if you saved exp(value)).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing = model.load_state_dict(ckpt["model"], strict=False)
    # Restore temperature if present (you saved exp(value) earlier)
    if "logit_scale_exp" in ckpt:
        with torch.no_grad():
            v = torch.tensor(ckpt["logit_scale_exp"], dtype=torch.float32)
            model.logit_scale.copy_(v.log())
    return missing


def save_finetuned_checkpoint(model: nn.Module, ckpt_path: str):
    """
    Saves model + current logit_scale (as exp(value) to be robust).
    """
    torch.save({
        "model": model.state_dict(),
        "logit_scale_exp": float(model.logit_scale.exp().detach().cpu().item())
    }, ckpt_path)


# -------------------------
# Geometry helpers (padding / clamps)
# -------------------------
def clamp_box(x0, y0, x1, y1, W, H):
    x0 = max(0, int(np.floor(x0))); y0 = max(0, int(np.floor(y0)))
    x1 = min(W, int(np.ceil(x1)));  y1 = min(H, int(np.ceil(y1)))
    return x0, y0, x1, y1


def pad_bbox(xmin, ymin, xmax, ymax, pad_ratio, img_w, img_h):
    w = xmax - xmin
    h = ymax - ymin
    pw = w * float(pad_ratio)
    ph = h * float(pad_ratio)
    x0 = xmin - 0.5 * pw
    y0 = ymin - 0.5 * ph
    x1 = xmax + 0.5 * pw
    y1 = ymax + 0.5 * ph
    return clamp_box(x0, y0, x1, y1, img_w, img_h)


def crop_image(pil_img: Image.Image, xyxy):
    x0, y0, x1, y1 = xyxy
    if (x1 - x0) < 1 or (y1 - y0) < 1:
        # return a tiny valid crop to avoid PIL errors
        x1 = max(x0 + 1, x1); y1 = max(y0 + 1, y1)
    return pil_img.crop((x0, y0, x1, y1))


# -------------------------
# Dynamic α/β (from relative box size)
# -------------------------
def _lerp(a, b, t):  # clamp t to [0,1]
    t = float(np.clip(t, 0.0, 1.0))
    return a + (b - a) * t


def dynamic_alpha_beta_from_box(
    det_xyxy: Tuple[float, float, float, float],
    img_w: int, img_h: int,
    a_min=0.05, a_max=0.20,
    b_min=0.05, b_max=0.25
) -> Tuple[float, float, float]:
    """
    Map box area ratio -> α, β. Also returns fg_ratio.
    """
    x0, y0, x1, y1 = det_xyxy
    box_area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    fg_ratio = float(box_area / max(1.0, img_w * img_h))
    # heuristic: smaller objects → larger context weight
    t = np.sqrt(np.clip(fg_ratio / 0.10, 0.0, 1.0))  # 0..~0.1 area → 0..1
    alpha = _lerp(a_max, a_min, t)  # small → ~a_max; large → ~a_min
    beta  = _lerp(b_max, b_min, t)
    return float(alpha), float(beta), fg_ratio


# -------------------------
# Soft mask weighting (optional but useful)
# -------------------------
def apply_soft_mask_weighting(crop_pil: Image.Image, mask_crop: Optional[np.ndarray]):
    """
    Simple soft focus on the object: fg*1.0 + bg*0.5 (tweak as needed).
    """
    if mask_crop is None:
        return crop_pil
    if mask_crop.size == 0:
        return crop_pil
    m = Image.fromarray((mask_crop.astype(np.uint8) * 255))
    m = m.resize((crop_pil.size[0], crop_pil.size[1]), resample=Image.NEAREST)
    m = np.array(m) > 127
    img = np.array(crop_pil).astype(np.float32)
    w = np.where(m[..., None], 1.0, 0.5)  # foreground 1.0, background 0.5
    out = np.clip(img * w, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# -------------------------
# Mask → tight bbox utility (legacy)
# -------------------------
def _mask_to_bbox(mask: np.ndarray, pad: int = 4) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1]-1, x1 + pad); y1 = min(mask.shape[0]-1, y1 + pad)
    return (x0, y0, x1, y1)


# -------------------------
# Wrapper
# -------------------------
class EVACLIPWrapper(nn.Module):
    """
    EVA-CLIP scorer with crop/context fusion and residual fusion:
      z_fused = norm(z_crop + α·z_ctx + β·(z_ctx − text_mean))
    Supports robust template pooling for [C,D] or [C*T,D] text embeddings.
    """
    def __init__(
        self,
        clip_model,
        preprocess,
        tokenizer,
        device: str = "cuda",
        combine: str = "add",
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.model = clip_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.combine = combine

        if combine == "concat+mlp":
            self.proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.proj = None

    # -------------------------
    # Text side
    # -------------------------
    @torch.no_grad()
    def build_text_cache(
        self,
        labels: Iterable[str],
        templates: Optional[List[str]] = None,
        layout: str = "mean",  # "mean" -> [C,D], "stack" -> [C*T,D]
    ) -> torch.Tensor:
        templates = templates or DEFAULT_TEMPLATES
        zs = []
        for cname in labels:
            texts = [t.format(cname) for t in templates]
            toks = self.tokenizer(texts).to(self.device)
            z = self.model.encode_text(toks)         # [T,D]
            z = F.normalize(z, dim=-1)
            if layout == "mean":
                z = F.normalize(z.mean(0, keepdim=True), dim=-1)  # [1,D]
            zs.append(z)
        if layout == "mean":
            return torch.cat(zs, dim=0)  # [C,D]
        else:  # "stack": cat per class (cat-major)
            return torch.cat(zs, dim=0)  # [(C*T),D]

    @torch.no_grad()
    def compute_text_mean(self, text_embeds: torch.Tensor) -> torch.Tensor:
        tmean = text_embeds.float().mean(dim=0, keepdim=True)
        return F.normalize(tmean, dim=-1)

    # -------------------------
    # Image encoders
    # -------------------------
    @torch.no_grad()
    def _encode_image(self, pil_img: Image.Image) -> torch.Tensor:
        model_dtype = next(self.model.parameters()).dtype
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        if model_dtype in (torch.float16, torch.bfloat16):
            x = x.to(model_dtype)
        feat = self.model.encode_image(x)
        return F.normalize(feat.float(), dim=-1)  # float32 for downstream math

    @torch.no_grad()
    def encode_image_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        if len(pil_images) == 0:
            return torch.empty(0, device=self.device)
        model_dtype = next(self.model.parameters()).dtype
        batch = torch.stack([self.preprocess(pil) for pil in pil_images], dim=0).to(self.device)
        if model_dtype in (torch.float16, torch.bfloat16):
            batch = batch.to(model_dtype)
        feats = self.model.encode_image(batch)
        return F.normalize(feats.float(), dim=-1)

    # -------------------------
    # Legacy SAM-scoring helper
    # -------------------------
    @torch.no_grad()
    def combine_feats(self, full_feat: Optional[torch.Tensor], mask_feat: Optional[torch.Tensor]) -> torch.Tensor:
        if self.combine == "image_only" or mask_feat is None:
            return full_feat
        if self.combine == "mask_only" or full_feat is None:
            return mask_feat
        if self.combine == "add":
            return F.normalize(full_feat + mask_feat, dim=-1)
        if self.combine == "concat+mlp":
            z = torch.cat([full_feat, mask_feat], dim=-1)
            z = self.proj(z)
            return F.normalize(z, dim=-1)
        return F.normalize(full_feat + mask_feat, dim=-1)

    # -------------------------
    # Crop-only / Context / Context-Residual classification (Top-k)
    # -------------------------
    @torch.no_grad()
    def _pool_templates_to_classes(self, logits_all: torch.Tensor, C: int, T: int, text_rows: int) -> torch.Tensor:
        # logits_all: [1, text_rows]
        if text_rows == C * T:
            return logits_all.view(1, C, T).max(dim=2).values[0]  # [C]
        elif text_rows == C:
            return logits_all.view(1, C).squeeze(0)               # [C]
        else:
            raise ValueError(f"text_embeds rows={text_rows} incompatible with C={C}, T={T}")

    @torch.no_grad()
    def classify_crop_with_context_residual_topk(
        self,
        crop_pil: Image.Image,
        context_pil: Image.Image,
        text_embeds: torch.Tensor,                       # [C,D] or [C*T,D], normalized
        text_feat_mean: Optional[torch.Tensor] = None,   # [1,D], normalized
        alpha: float = 0.15,
        beta: float  = 0.15,
        cat_names: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
        topk_list: Tuple[int, ...] = (1,3,5)
    ):
        z_crop = self._encode_image(crop_pil)   # [1,D], float32
        z_ctx  = self._encode_image(context_pil)

        t_mean = self.compute_text_mean(text_embeds) if text_feat_mean is None else F.normalize(text_feat_mean, dim=-1)
        z_fused = F.normalize(z_crop + alpha * z_ctx + beta * (z_ctx - t_mean), dim=-1)

        text32 = text_embeds.to(self.device, dtype=torch.float32)
        logit_scale = self.model.logit_scale.exp().float()
        logits_all  = (z_fused @ text32.T) * logit_scale          # [1, N_text]

        C = len(cat_names) if cat_names is not None else text32.shape[0]
        T = len(templates) if templates is not None else len(DEFAULT_TEMPLATES)
        logits_by_class = self._pool_templates_to_classes(logits_all, C, T, text32.shape[0])  # [C]

        probs = F.softmax(logits_by_class, dim=-1)
        out = {}
        for k in topk_list:
            k = min(k, logits_by_class.numel())
            out[k] = torch.topk(logits_by_class, k=k).indices.tolist()
        pred_idx = out[min(1, max(out.keys()))][0] if 1 in out else int(torch.argmax(probs).item())
        return logits_by_class, probs, out, float(alpha), float(beta)

# -------------------------
# High-level crop classification helper
# -------------------------

@torch.no_grad()
def classify_crop_with_threshold(
    eva: EVACLIPWrapper,
    crop_pil: Image.Image,
    context_pil: Image.Image,
    text_embeds: torch.Tensor,
    cat_names: List[str],
    text_feat_mean: Optional[torch.Tensor] = None,
    templates: Optional[List[str]] = None,
    tau_clip: float = 0.30,
) -> dict:
    """
    Run EVA-CLIP classification on a crop+context and apply a probability threshold.

    Returns:
      {
        "logits":     logits_by_class,   # [C] tensor
        "probs":      probs,             # [C] tensor
        "max_prob":   float,
        "pred_idx":   int,
        "pred_label": str,
        "is_confident": bool
      }
    """
    logits, probs, _, alpha, beta = eva.classify_crop_with_context_residual_topk(
        crop_pil=crop_pil,
        context_pil=context_pil,
        text_embeds=text_embeds,
        text_feat_mean=text_feat_mean,
        alpha=0.15,
        beta=0.15,
        cat_names=cat_names,
        templates=templates or DEFAULT_TEMPLATES,
        topk_list=(1, 3, 5),
    )

    max_prob, pred_idx_t = torch.max(probs, dim=-1)
    pred_idx = int(pred_idx_t.item())
    max_prob = float(max_prob.item())
    pred_label = cat_names[pred_idx]

    return {
        "logits": logits,
        "probs": probs,
        "max_prob": max_prob,
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "is_confident": max_prob >= tau_clip,
        "alpha": alpha,
        "beta": beta,
    }

# -------------------------
# Text-side utilities (for BLIP caption mapping)
# -------------------------

import re

# You can refine this per dataset (COCO, Cityscapes, etc.)
ONTOLOGY_SYNONYMS = {
    "person": ["person", "man", "woman", "boy", "girl"],
    "car": ["car", "vehicle", "automobile"],
    "teddy bear": ["teddy", "bear", "teddy bear"],
    "book": ["book", "paper", "page"],
    "dining table": ["table"],
    "mouse": ["mouse"],
    "knife": ["knife"],
    # add more as needed...
}


@torch.no_grad()
def map_caption_to_class_cpu(
    eva: EVACLIPWrapper,
    caption: str,
    text_embeds_cpu: torch.Tensor,
    cat_names: List[str],
    device: str = "cpu",
) -> dict:
    """
    Encode caption with EVA-CLIP text tower (on CPU) and map to class
    using precomputed text_embeds_cpu [C,D].
    """
    # ensure model on CPU text device
    eva.model = eva.model.to(device)
    toks = eva.tokenizer([caption]).to(device)
    z_text = eva.model.encode_text(toks)        # [1,D]
    z_text = F.normalize(z_text.float(), dim=-1)

    sims = (z_text @ text_embeds_cpu.T)[0]      # [C]
    best_idx_t = torch.argmax(sims)
    best_idx = int(best_idx_t.item())
    best_sim = float(sims[best_idx].item())
    best_label = cat_names[best_idx]

    return {
        "best_idx": best_idx,
        "best_label": best_label,
        "best_sim": best_sim,
    }


def caption_contains_label_or_synonym(caption: str, label: str) -> bool:
    caption_l = caption.lower()
    words = set(re.findall(r"\w+", caption_l))
    syns = ONTOLOGY_SYNONYMS.get(label, [label])
    for s in syns:
        s_l = s.lower()
        if s_l in words or s_l in caption_l:
            return True
    return False


def accept_caption_mapping(
    caption: str,
    mapping: dict,
    sim_high: float = 0.70,
    sim_mid: float = 0.58,
) -> bool:
    """
    Decide whether to accept BLIP→class mapping.

    - Accept if similarity >= sim_high.
    - Accept if similarity >= sim_mid and caption mentions label/synonym.
    """
    sim = mapping["best_sim"]
    label = mapping["best_label"]
    if sim >= sim_high:
        return True
    if sim >= sim_mid and caption_contains_label_or_synonym(caption, label):
        return True
    return False

def prepare_text_embeds_cpu(
    eva: EVACLIPWrapper,
    cat_names: List[str],
    templates: Optional[List[str]] = None,
):
    with torch.no_grad():
        text_embeds = eva.build_text_cache(
            labels=cat_names,
            templates=templates or DEFAULT_TEMPLATES,
            layout="mean",
        )
        text_mean = eva.compute_text_mean(text_embeds)
    return text_embeds, text_mean, text_embeds.detach().cpu(), text_mean.detach().cpu()
