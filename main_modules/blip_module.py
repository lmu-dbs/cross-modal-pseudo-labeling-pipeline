
# (optional) Keep your existing BLIP code here if needed for captions.
# sam_clip_full/blip_module.py
"""
blip_module.py
Lightweight BLIP wrapper for captioning crops.

Provides:
  - load_blip(...)             → load BLIP model + processor
  - blip_caption(...)          → caption a list of PIL images in small batches
"""

from typing import List, Optional
import torch
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration


def load_blip(
    device: str = "cpu",
    model_name: str = "Salesforce/blip-image-captioning-base",
):
    """
    Load BLIP (base) for image captioning.
    Default: run on CPU so GPU is free for SAM + EVA-CLIP.
    """
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device).eval()
    return model, processor, device


@torch.no_grad()
def blip_caption(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    pil_images: List[Image.Image],
    device: str = "cpu",
    prompt: Optional[str] = None,
    batch_size: int = 4,
    max_new_tokens: int = 15,
) -> List[str]:
    """
    Run BLIP captioning in small batches.

    Args:
      model, processor: output of load_blip(...)
      pil_images: list of PIL RGB images
      device: device where BLIP model lives ("cpu" recommended)
      prompt: if not None, used as text input for all images
      batch_size: how many crops per BLIP forward
      max_new_tokens: generation length cap

    Returns:
      List[str]: one caption per input image.
    """
    captions_all: List[str] = []
    n = len(pil_images)
    if n == 0:
        return captions_all

    for start in range(0, n, batch_size):
        batch = pil_images[start:start + batch_size]

        if prompt is None:
            inputs = processor(images=batch, return_tensors="pt")
        else:
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        caps = processor.batch_decode(out, skip_special_tokens=True)
        captions_all.extend([c.strip() for c in caps])

    return captions_all
