"""InstructSAM target-mask bridge for Cosmos Predict2 inference."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

MaskCombineMode = Literal["best", "union"]
TargetFeatureMode = Literal["mask_query", "raw_seg"]


@dataclass(slots=True)
class TargetMaskResult:
    mask_B_C_T_H_W: torch.Tensor
    text: str
    score: float | None = None
    feature_B_L_D: torch.Tensor | None = None


def _repo_third_party_instructsam() -> Path:
    # .../third_party/cosmos-predict2.5/cosmos_predict2/_src/predict2/target_aware/this.py
    return Path(__file__).resolve().parents[5] / "InstructSAM"


def _ensure_instructsam_importable(source_root: str | os.PathLike[str] | None = None) -> None:
    candidates = []
    if source_root is not None:
        candidates.append(Path(source_root))
    candidates.append(_repo_third_party_instructsam())
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def read_first_frame_image(input_path: str | os.PathLike[str]) -> Image.Image:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if suffix == ".mp4":
        from decord import VideoReader, cpu

        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        frame = vr.get_batch([0]).asnumpy()[0]
        return Image.fromarray(frame).convert("RGB")
    raise ValueError(f"Unsupported input extension for InstructSAM target query: {suffix}")


@contextlib.contextmanager
def _as_image_path(image: str | os.PathLike[str] | Image.Image):
    if isinstance(image, (str, os.PathLike)):
        yield str(image)
        return

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        yield tmp_path
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _flatten_masks_and_scores(
    pred_masks: torch.Tensor,
    cls_score: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    masks = pred_masks.detach().float().cpu()
    if masks.ndim < 2:
        raise ValueError(f"Unsupported InstructSAM mask shape: {tuple(masks.shape)}")
    masks = masks.reshape(-1, *masks.shape[-2:])

    scores = None
    if cls_score is not None:
        score_tensor = cls_score.detach().float().cpu().reshape(-1)
        if score_tensor.numel() == masks.shape[0]:
            scores = score_tensor
    return masks, scores


def masks_to_cosmos_target_mask(
    pred_masks: torch.Tensor,
    cls_score: torch.Tensor | None = None,
    *,
    combine_mode: MaskCombineMode = "best",
    mask_threshold: float = 0.0,
    output_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, float | None]:
    """Select InstructSAM masks and return Cosmos shape [1, 1, 1, H, W]."""
    masks, scores = _flatten_masks_and_scores(pred_masks, cls_score)
    binary_masks = masks > mask_threshold
    if binary_masks.numel() == 0:
        raise ValueError("InstructSAM returned an empty mask tensor")

    selected_score = None
    if combine_mode == "union":
        mask_2d = binary_masks.any(dim=0)
        if scores is not None and scores.numel() > 0:
            selected_score = float(scores.max().item())
    elif combine_mode == "best":
        if scores is not None and scores.numel() > 0:
            best_idx = int(scores.argmax().item())
            selected_score = float(scores[best_idx].item())
        else:
            areas = binary_masks.flatten(1).float().sum(dim=1)
            best_idx = int(areas.argmax().item())
        mask_2d = binary_masks[best_idx]
    else:
        raise ValueError(f"Unsupported mask combine mode: {combine_mode}")

    mask_B_C_H_W = mask_2d.float().unsqueeze(0).unsqueeze(0)
    if output_size is not None and tuple(mask_B_C_H_W.shape[-2:]) != tuple(output_size):
        mask_B_C_H_W = F.interpolate(mask_B_C_H_W, size=output_size, mode="nearest")
    return mask_B_C_H_W.unsqueeze(2).contiguous(), selected_score


def load_target_mask_file(
    mask_path: str | os.PathLike[str],
    *,
    mask_threshold: float = 0.0,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Load a precomputed target mask and return Cosmos shape [1, 1, T, H, W]."""
    path = Path(mask_path)
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        mask = Image.open(path).convert("L")
        arr = np.asarray(mask)
    elif suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        if "masks" in npz.files:
            arr = npz["masks"]
        elif "masks_packed" in npz.files and "shape" in npz.files:
            shape = tuple(int(dim) for dim in npz["shape"].tolist())
            flat_pixels = int(np.prod(shape[1:]))
            arr = np.unpackbits(npz["masks_packed"], axis=1)[:, :flat_pixels].reshape(shape)
        else:
            arr = npz[npz.files[0]]
    else:
        raise ValueError(f"Unsupported target mask file extension: {suffix}")

    tensor = torch.from_numpy(np.asarray(arr)).float()
    if tensor.ndim == 5:
        # [N,T,1,H,W] or similar instance stacks.
        tensor = tensor.max(dim=0).values
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:  # [T,1,H,W]
            tensor = tensor[:, 0]
        elif tensor.shape[0] == 1:  # [1,T,H,W]
            tensor = tensor[0]
        else:
            tensor = tensor.max(dim=0).values
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Unsupported target mask file shape: {tuple(tensor.shape)}")
    tensor = (tensor > mask_threshold).float().unsqueeze(0).unsqueeze(0)
    if output_size is not None and tuple(tensor.shape[-2:]) != tuple(output_size):
        tensor = F.interpolate(tensor, size=(tensor.shape[2], *output_size), mode="nearest")
    return tensor.contiguous()


class InstructSAMTargetMaskGenerator:
    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        source_root: str | os.PathLike[str] | None = None,
        device_map: str | dict = "auto",
        attn_implementation: str = "sdpa",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        _ensure_instructsam_importable(source_root)
        from instructsam.models import load_pretrained_model

        self.model_path = str(model_path)
        self.tokenizer, self.model, self.processor = load_pretrained_model(
            self.model_path,
            None,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.model.to(dtype=torch_dtype)
        self.model.eval()

    def _extract_target_feature(self, feature_mode: TargetFeatureMode = "mask_query") -> torch.Tensor | None:
        """Return InstructSAM target embeddings as [1, L, D] CPU float tokens.

        InstructSAM stores the generated segmentation query hidden states on
        `model.seg_output_embeddings` during `model.inference`.  The default
        `mask_query` mode reuses InstructSAM's own `mask_hidden_fcs` projection,
        which makes the exported target token dimension 256 in the released
        model and keeps the Cosmos bridge small.
        """
        seg_embeddings = getattr(self.model, "seg_output_embeddings", None)
        if not seg_embeddings:
            return None

        raw_features = [emb.detach() for emb in seg_embeddings]
        if feature_mode == "mask_query":
            core_model = getattr(self.model, "model", None)
            mask_hidden_fcs = getattr(core_model, "mask_hidden_fcs", None)
            if not mask_hidden_fcs:
                return None
            projector = mask_hidden_fcs[0]
            first_param = next(projector.parameters(), None)
            pieces = []
            for raw_feature in raw_features:
                if first_param is not None:
                    raw_feature = raw_feature.to(device=first_param.device, dtype=first_param.dtype)
                pieces.append(projector(raw_feature))
        elif feature_mode == "raw_seg":
            pieces = raw_features
        else:
            raise ValueError(f"Unsupported InstructSAM target feature mode: {feature_mode}")

        flattened_pieces = []
        for piece in pieces:
            piece = torch.nan_to_num(piece.detach().float())
            if piece.ndim == 1:
                piece = piece.view(1, -1)
            elif piece.ndim > 2:
                piece = piece.reshape(-1, piece.shape[-1])
            flattened_pieces.append(piece)
        feature = torch.cat(flattened_pieces, dim=0).unsqueeze(0)
        return feature.cpu().contiguous()

    @torch.inference_mode()
    def predict(
        self,
        image: str | os.PathLike[str] | Image.Image,
        query: str,
        *,
        combine_mode: MaskCombineMode = "best",
        mask_threshold: float = 0.0,
        output_size: tuple[int, int] | None = None,
        feature_mode: TargetFeatureMode = "mask_query",
    ) -> TargetMaskResult:
        _ensure_instructsam_importable()
        from instructsam import mm_infer_segmentation

        with _as_image_path(image) as image_path:
            contents = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query},
            ]
            conversation = [{"role": "user", "content": contents}]
            output, pred_masks, cls_score = mm_infer_segmentation(
                image_path,
                self.processor,
                conversation,
                self.model,
                self.tokenizer,
            )

            if output_size is None:
                with Image.open(image_path) as pil_image:
                    width, height = pil_image.size
                output_size = (height, width)
            feature = self._extract_target_feature(feature_mode=feature_mode)
            if pred_masks is None:
                if feature is None:
                    raise RuntimeError(
                        f"InstructSAM did not return a mask or target feature for query: {query}; output={output!r}"
                    )
                mask = torch.zeros(1, 1, 1, *output_size, dtype=torch.float32)
                return TargetMaskResult(mask_B_C_T_H_W=mask, text=output, score=None, feature_B_L_D=feature)
            mask, score = masks_to_cosmos_target_mask(
                pred_masks,
                cls_score,
                combine_mode=combine_mode,
                mask_threshold=mask_threshold,
                output_size=output_size,
            )
            return TargetMaskResult(mask_B_C_T_H_W=mask, text=output, score=score, feature_B_L_D=feature)

    def predict_from_input(
        self,
        input_path: str | os.PathLike[str],
        query: str,
        *,
        combine_mode: MaskCombineMode = "best",
        mask_threshold: float = 0.0,
        feature_mode: TargetFeatureMode = "mask_query",
    ) -> TargetMaskResult:
        image = read_first_frame_image(input_path)
        return self.predict(
            image,
            query,
            combine_mode=combine_mode,
            mask_threshold=mask_threshold,
            output_size=(image.height, image.width),
            feature_mode=feature_mode,
        )
