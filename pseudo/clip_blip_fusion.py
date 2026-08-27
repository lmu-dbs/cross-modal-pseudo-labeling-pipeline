# sam_clip_full/pseudo/clip_blip_fusion.py

from typing import List, Optional, Dict, Any

import torch
from PIL import Image

from sam_clip_full.blip_module import load_blip, blip_caption
from sam_clip_full.clip_module import (
    EVACLIPWrapper,
    classify_crop_with_threshold,
    map_caption_to_class_cpu,
    accept_caption_mapping,
)


def classify_mask_with_clip_and_optional_blip(
    eva: EVACLIPWrapper,
    crop_pil: Image.Image,
    context_pil: Image.Image,
    text_embeds: torch.Tensor,        # [C,D] on eva.device for image scoring
    text_embeds_cpu: torch.Tensor,    # [C,D] on CPU for caption mapping
    cat_names: List[str],
    use_blip: bool,
    blip_model=None,
    blip_processor=None,
    tau_clip: float = 0.30,
    tau_blip_high: float = 0.70,
    tau_blip_mid: float = 0.58,
    open_vocab_mode: str = "off",     # "off" | "log_only" | "map_to_other"
) -> Dict[str, Any]:
    """
    Core decision unit per SAM mask.

    Returns a dict with e.g.:
      - status: "clip_confident" | "blip_label" | "open_vocab" | "unlabeled"
      - label_idx / label_name (if any)
      - caption / mapping info (if BLIP used)
    """
    # 1) CLIP classification
    clip_res = classify_crop_with_threshold(
        eva=eva,
        crop_pil=crop_pil,
        context_pil=context_pil,
        text_embeds=text_embeds,
        cat_names=cat_names,
        tau_clip=tau_clip,
    )

    # --- CLIP-only paths ---
    if clip_res["is_confident"] and not use_blip:
        return {
            "status": "clip_confident",
            "label_idx": clip_res["pred_idx"],
            "label_name": clip_res["pred_label"],
            "source": "clip",
            "clip_max_prob": clip_res["max_prob"],
        }

    if not use_blip:
        # no BLIP, CLIP not confident → unlabeled
        return {
            "status": "unlabeled",
            "label_idx": None,
            "label_name": None,
            "source": "none",
            "clip_max_prob": clip_res["max_prob"],
        }

    # --- use_blip=True from here on ---

    if blip_model is None or blip_processor is None:
        raise ValueError("BLIP model/processor must be provided when use_blip=True.")

    # 2) BLIP caption on CPU
    captions = blip_caption(
        model=blip_model,
        processor=blip_processor,
        pil_images=[crop_pil],
        device="cpu",
        batch_size=1,
        max_new_tokens=15,
    )
    caption = captions[0]

    # 3) Map caption -> class via EVA text tower (CPU)
    mapping = map_caption_to_class_cpu(
        eva=eva,
        caption=caption,
        text_embeds_cpu=text_embeds_cpu,
        cat_names=cat_names,
        device="cpu",
    )

    # 3a) If BLIP mapping is accepted, use it as label
    if accept_caption_mapping(
        caption,
        mapping,
        sim_high=tau_blip_high,
        sim_mid=tau_blip_mid,
    ):
        return {
            "status": "blip_label",
            "label_idx": mapping["best_idx"],
            "label_name": mapping["best_label"],
            "source": "blip",
            "caption": caption,
            "blip_sim": mapping["best_sim"],
            "clip_max_prob": clip_res["max_prob"],
        }

    # 3b) Optional: open-vocab label from caption
    if open_vocab_mode != "off":
        import re
        tokens = re.findall(r"\w+", caption.lower())
        open_word = tokens[-1] if tokens else None

        return {
            "status": "open_vocab",
            "open_word": open_word,
            "caption": caption,
            "source": "blip_open",
            "clip_max_prob": clip_res["max_prob"],
            "blip_sim": mapping["best_sim"],
        }

    # 3c) Still unlabeled
    return {
        "status": "unlabeled",
        "label_idx": None,
        "label_name": None,
        "source": "none",
        "caption": caption,
        "clip_max_prob": clip_res["max_prob"],
        "blip_sim": mapping["best_sim"],
    }
