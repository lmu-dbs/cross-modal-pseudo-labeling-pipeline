# sam_clip_full/pseudo/__init__.py
"""
Utilities for generating and evaluating pseudo-labels with
SAM + EVA-CLIP (+ optional BLIP).

This package groups together:

- Converters between JSONL + semantic PNGs
- Pseudo-label quality evaluation (mIoU vs GT)
- Dataset split helpers for Foggy Cityscapes
- CLIP+BLIP fusion logic for mask classification
"""

from .convert_jsonl_to_semantic import convert_jsonl_to_semantic
from .eval_pseudo_vs_gt import main as eval_pseudo_vs_gt
from .clip_blip_fusion import classify_mask_with_clip_and_optional_blip

__all__ = [
    "convert_jsonl_to_semantic",
    "eval_pseudo_vs_gt",
    "classify_mask_with_clip_and_optional_blip",
]
