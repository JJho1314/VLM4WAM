"""Multi-source InstructSAM feature extraction for text-free Cosmos guidance.

Extends the single-feature InstructSAM bridge (``instructsam_mask.py``) to pull
THREE representations from one inference call:

- ``mask``   : ``mask_hidden_fcs[0](seg_output_embeddings)``           -> ``[Lm, 256]``
- ``detect`` : SAM3 ``detr_decoder.intermediate_hidden_states[-1]``    -> ``[Ld, 256]``
              (best query per object by ``pred_logits``)
- ``vtext``  : Qwen3VL ``language_model.last_hidden_state`` at prefill,
              adaptive mean-pooled over the sequence                   -> ``[Lv, 4096]``

The detect / vtext tensors are captured with forward hooks so InstructSAM itself
is not edited. Hooks are attached by module *type* / attribute, which is robust
to the exact nesting of the released checkpoint
(``model.model.grounding_model.model`` is the ``Sam3Model``;
``model.model.language_model`` is the Qwen3VL LM).

Fusion into a single ``[L, 256]`` Cosmos ``target_feature`` happens in the
precompute script, not here, so this module stays "pure InstructSAM".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from cosmos_predict2._src.predict2.target_aware.instructsam_mask import (
    InstructSAMTargetMaskGenerator,
    _as_image_path,
    _ensure_instructsam_importable,
    masks_to_cosmos_target_mask,
    read_first_frame_image,
)


@dataclass(slots=True)
class MultiSourceFeatureResult:
    """Native-dim InstructSAM representations for one sample.

    Any field may be ``None`` if InstructSAM did not expose it (e.g. no target
    found). Shapes are ``[L, D]`` (2-D, CPU, float32).
    """

    mask_L_Dm: Optional[torch.Tensor]
    detect_L_Dd: Optional[torch.Tensor]
    vtext_L_Dv: Optional[torch.Tensor]
    text: str = ""
    score: Optional[float] = None
    # Best combined binary segmentation mask at the input image size [H, W]
    # (for visualization / record-keeping; not part of the fused feature).
    mask_HW: Optional[torch.Tensor] = None

    def any_present(self) -> bool:
        return any(t is not None for t in (self.mask_L_Dm, self.detect_L_Dd, self.vtext_L_Dv))


def _to_2d_float(t: torch.Tensor) -> torch.Tensor:
    """Collapse leading dims so the result is ``[L, D]`` float on CPU."""
    t = torch.nan_to_num(t.detach().float().cpu())
    if t.ndim == 1:
        return t.view(1, -1)
    if t.ndim > 2:
        t = t.reshape(-1, t.shape[-1])
    return t.contiguous()


def _adaptive_pool_tokens(t_L_D: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Order-preserving reduction of a token sequence to <= ``max_tokens`` rows."""
    if max_tokens <= 0 or t_L_D.shape[0] <= max_tokens:
        return t_L_D.contiguous()
    # [L, D] -> [1, D, L] -> adaptive pool over L -> [max_tokens, D]
    pooled = F.adaptive_avg_pool1d(t_L_D.transpose(0, 1).unsqueeze(0), max_tokens)
    return pooled.squeeze(0).transpose(0, 1).contiguous()


class InstructSAMMultiSourceGenerator(InstructSAMTargetMaskGenerator):
    """``InstructSAMTargetMaskGenerator`` that also returns detect + vtext reps."""

    def __init__(
        self,
        *args,
        detect_max_tokens: int = 64,
        vtext_max_tokens: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.detect_max_tokens = int(detect_max_tokens)
        self.vtext_max_tokens = int(vtext_max_tokens)
        # Per-call capture buffers, reset in predict_multi_source().
        self._detect_capture: list[torch.Tensor] = []
        self._vtext_capture: list[torch.Tensor] = []
        self._hook_handles: list = []
        self._register_capture_hooks()

    # -- hook plumbing ----------------------------------------------------- #
    def _find_module(self, *, class_name: Optional[str] = None, attr_path: Optional[str] = None):
        """Locate a submodule by attribute path first, then by class name."""
        if attr_path:
            obj = self.model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, torch.nn.Module):
                    return obj
            except AttributeError:
                pass
        if class_name:
            for module in self.model.modules():
                if type(module).__name__ == class_name:
                    return module
        return None

    def _register_capture_hooks(self) -> None:
        detr = self._find_module(
            class_name="Sam3DetrDecoder",
            attr_path="model.grounding_model.model.detr_decoder",
        )
        if detr is not None:
            def _detr_hook(_module, _inp, out):
                hidden = getattr(out, "intermediate_hidden_states", None)
                if hidden is None and isinstance(out, (tuple, list)) and out:
                    hidden = out[0]
                if hidden is not None:
                    # [num_layers, B, num_queries, D] -> last layer.
                    last = hidden[-1] if hidden.ndim == 4 else hidden
                    self._detect_capture.append(last.detach())
            self._hook_handles.append(detr.register_forward_hook(_detr_hook))

        lm = self._find_module(attr_path="model.language_model", class_name=None)
        if lm is not None:
            def _lm_hook(_module, _inp, out):
                hs = getattr(out, "last_hidden_state", None)
                if hs is None and isinstance(out, (tuple, list)) and out:
                    hs = out[0]
                if hs is not None and hs.ndim >= 2:
                    # Keep only the longest sequence (= prefill), drop per-token
                    # decode steps to bound memory.
                    seq = hs.shape[-2]
                    if not self._vtext_capture or seq > self._vtext_capture[0].shape[-2]:
                        self._vtext_capture = [hs.detach()]
            self._hook_handles.append(lm.register_forward_hook(_lm_hook))

    # -- representation assembly ------------------------------------------ #
    def _mask_rep(self) -> Optional[torch.Tensor]:
        feature = self._extract_target_feature(feature_mode="mask_query")  # [1, L, 256]
        if feature is None:
            return None
        return _to_2d_float(feature)

    def _detect_rep(self, cls_score: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self._detect_capture:
            return None
        queries = self._detect_capture[-1]  # [B(obj), num_queries, D] or [num_queries, D]
        q = queries.detach().float().cpu()
        if q.ndim == 2:
            return _adaptive_pool_tokens(_to_2d_float(q), self.detect_max_tokens)
        if q.ndim != 3:
            return _adaptive_pool_tokens(_to_2d_float(q), self.detect_max_tokens)
        # [B, Q, D]: pick the best query per object by detection score when we can
        # line the scores up, else mean-pool over queries.
        B, Q, _ = q.shape
        score = None
        if cls_score is not None:
            s = cls_score.detach().float().cpu().reshape(-1)
            if s.numel() == B * Q:
                score = s.view(B, Q)
            elif s.numel() == Q:
                score = s.view(1, Q).expand(B, Q)
        if score is not None:
            best = score.argmax(dim=1)  # [B]
            rep = q[torch.arange(B), best]  # [B, D]
        else:
            rep = q.mean(dim=1)  # [B, D]
        return _adaptive_pool_tokens(_to_2d_float(rep), self.detect_max_tokens)

    def _vtext_rep(self) -> Optional[torch.Tensor]:
        if not self._vtext_capture:
            return None
        hs = self._vtext_capture[0]  # [B, seq, D]
        rep = _to_2d_float(hs)  # [seq*B, D] (B==1 at inference)
        return _adaptive_pool_tokens(rep, self.vtext_max_tokens)

    @torch.inference_mode()
    def predict_multi_source(
        self,
        image: str | os.PathLike[str] | Image.Image,
        query: str,
    ) -> MultiSourceFeatureResult:
        _ensure_instructsam_importable()
        from instructsam import mm_infer_segmentation

        self._detect_capture = []
        self._vtext_capture = []
        with _as_image_path(image) as image_path:
            with Image.open(image_path) as pil_image:
                output_size = (pil_image.height, pil_image.width)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            output, pred_masks, cls_score = mm_infer_segmentation(
                image_path,
                self.processor,
                conversation,
                self.model,
                self.tokenizer,
            )
        mask_rep = self._mask_rep()
        detect_rep = self._detect_rep(cls_score)
        vtext_rep = self._vtext_rep()
        score = None
        if cls_score is not None and cls_score.numel() > 0:
            score = float(cls_score.detach().float().max().item())
        mask_HW = None
        if pred_masks is not None:
            try:
                mask_B, _ = masks_to_cosmos_target_mask(
                    pred_masks, cls_score, combine_mode="best", mask_threshold=0.0, output_size=output_size
                )
                mask_HW = mask_B[0, 0, 0].cpu()
            except Exception:
                mask_HW = None
        return MultiSourceFeatureResult(
            mask_L_Dm=mask_rep,
            detect_L_Dd=detect_rep,
            vtext_L_Dv=vtext_rep,
            text=output,
            score=score,
            mask_HW=mask_HW,
        )

    def predict_multi_source_from_input(
        self,
        input_path: str | os.PathLike[str],
        query: str,
    ) -> MultiSourceFeatureResult:
        image = read_first_frame_image(input_path)
        return self.predict_multi_source(image, query)

    def close(self) -> None:
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles = []
