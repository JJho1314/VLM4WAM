import os

from . import models
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
)

# Pad value for the letterboxed SAM image (channel mean ~= 0.5*255), so the
# padded band normalizes to ~0 and is benign to the SAM3 encoder.
_SAM_LETTERBOX_FILL = (127, 127, 127)


def _letterbox_to_square(image: Image.Image, fill=_SAM_LETTERBOX_FILL):
    """Pad a PIL image to a square (aspect-preserving), original at top-left.

    Returns (square_image, (orig_w, orig_h, side)). This avoids the anisotropic
    stretch that Sam3ImageProcessorFast (size=square, do_pad off) otherwise
    applies to non-square inputs, which deforms small objects.
    """
    w, h = image.size
    side = max(w, h)
    if w == h:
        return image, (w, h, side)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, (0, 0))
    return canvas, (w, h, side)


def _unletterbox_masks(masks: torch.Tensor, letterbox_meta) -> torch.Tensor:
    """Map masks decoded in the padded-square frame back to the original image.

    This is the letterbox-aware counterpart of SAM's ``post_process_masks``
    (the native Sam3 one only takes ``original_sizes`` and assumes a full-frame
    stretch, so it cannot remove letterbox padding). Steps: upscale the mask to
    the square side, then crop the top-left content region of size (orig_h, orig_w).
    Returns masks at original (H, W); leading dims are preserved.
    """
    orig_w, orig_h, side = letterbox_meta
    lead = masks.shape[:-2]
    flat = masks.reshape(-1, 1, masks.shape[-2], masks.shape[-1]).float()
    flat = F.interpolate(flat, size=(side, side), mode="bilinear", align_corners=False)
    flat = flat[..., :orig_h, :orig_w]
    return flat.reshape(*lead, orig_h, orig_w)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def mm_infer_segmentation(image_path, processor, conversation, model, tokenizer, sam_letterbox=None, **kwargs):
    seg_processor = AutoProcessor.from_pretrained(model.config.mask_decoder_model)

    # Default OFF to match official training/inference (anisotropic stretch via the
    # Sam3 processor; GT masks are likewise stretched to 288 in the loss). Letterbox
    # is an opt-in experiment only — turning it on diverges from the pretrained prior.
    if sam_letterbox is None:
        sam_letterbox = os.environ.get("INSTRUCTSAM_SAM_LETTERBOX", "0") == "1"

    # sam image
    sam_images = []
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Letterbox the SAM image to a square so the processor's square resize is
    # aspect-preserving (uniform) instead of an anisotropic stretch. Masks are
    # un-letterboxed back to the original frame after inference.
    letterbox_meta = None
    if sam_letterbox:
        sam_image, letterbox_meta = _letterbox_to_square(image)
    else:
        sam_image = image
    sam_inputs = seg_processor(sam_image)
    sam_images.append(sam_inputs['pixel_values'][0])
    sam_size = sam_inputs.original_sizes[0]
    sam_images = torch.cat(sam_images, dim=0)

    # model inputs
    inputs = processor.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    # Newer Transformers validates generation kwargs strictly; Qwen3-VL's
    # processor may return this field even when the model generate path does
    # not consume it.
    inputs.pop("mm_token_type_ids", None)

    with torch.inference_mode():
        output_ids, pred_masks, cls_score = model.inference(
            **inputs,
            sam_images=[sam_images.to(model.device)],
            max_new_tokens=1024,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False
        )
    # post-process: un-letterbox masks (288x288 square frame -> original H,W).
    # Native Sam3 post_process_masks only handles the stretch convention, so we
    # do the letterbox-aware remap here.
    if pred_masks is not None and letterbox_meta is not None:
        pred_masks = _unletterbox_masks(pred_masks, letterbox_meta)

    outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    outputs = outputs.replace("<|object_ref_end|>", "<|object_ref_end|><|mask_start|>[SEG]<|mask_end|>")
    return outputs, pred_masks, cls_score
