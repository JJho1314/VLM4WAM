"""Frozen SigLIP2 phrase embeddings and gradient-based dense relevance maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from qwen35_planx.instruction import InstructionFields


_GRID_SIZE = 27
_TOKENS = _GRID_SIZE * _GRID_SIZE
_TEXT_WIDTH = 1152


@dataclass(frozen=True)
class PhraseRelevance:
    """Phrase embeddings and frame-local probability maps."""

    phrase_embeddings: Tensor
    maps: Tensor
    confidence: Tensor

    @property
    def confidences(self) -> Tensor:
        """Plural compatibility alias for callers handling several phrases."""

        return self.confidence


def _module_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _as_tensor_mapping(value: Any) -> dict[str, Tensor]:
    if not hasattr(value, "items"):
        raise TypeError("SigLIP2 processor must return a tensor mapping")
    return {
        key: item
        for key, item in value.items()
        if isinstance(item, Tensor)
    }


def _pooling_attention_from_output(output: Any, captured: Tensor | None) -> Tensor | None:
    attention = getattr(output, "pooling_attention", None)
    if attention is None:
        attention = captured
    if attention is None:
        return None
    if attention.ndim == 4:
        attention = attention.mean(dim=1).squeeze(-2)
    elif attention.ndim == 3:
        attention = attention.mean(dim=1)
    if attention.ndim != 2 or attention.shape[-1] != _TOKENS:
        return None
    return attention


class SiglipRelevanceTeacher:
    """Offline-only SigLIP2 teacher with frozen parameters.

    A complete model and local processor can be injected with
    :meth:`from_components`, which is also the deterministic unit-test path.
    """

    def __init__(self, *, model: nn.Module, processor: Any) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        self.model = model.eval()
        self.processor = processor
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._captured_spatial_activations: Tensor | None = None
        self.last_spatial_gradients: tuple[Tensor, ...] = ()

    @property
    def captured_spatial_activations(self) -> Tensor | None:
        """Currently captured activation, or ``None`` outside relevance calls."""

        return self._captured_spatial_activations

    @classmethod
    def from_components(
        cls,
        *,
        model: nn.Module,
        processor: Any,
    ) -> SiglipRelevanceTeacher:
        return cls(model=model, processor=processor)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        local_files_only: bool = True,
    ) -> SiglipRelevanceTeacher:
        """Load a local SigLIP2 teacher without permitting an implicit download."""

        from transformers import AutoProcessor, Siglip2Model

        model = Siglip2Model.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
        )
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
        )
        return cls(model=model, processor=processor)

    def encode_fields(
        self,
        rgb: Tensor,
        fields: InstructionFields,
        *,
        counterfactual_phrases: Sequence[str] | None = None,
    ) -> PhraseRelevance:
        """Encode fields in the cache's canonical source/target/action order."""

        if not isinstance(fields, InstructionFields):
            raise TypeError("fields must be InstructionFields")
        phrases = (fields.source, fields.target, fields.action)
        result = self.encode(
            rgb,
            phrases=phrases,
            counterfactual_phrases=counterfactual_phrases,
        )
        field_confidence = torch.tensor(
            fields.confidences,
            device=result.confidence.device,
            dtype=result.confidence.dtype,
        )
        return PhraseRelevance(
            phrase_embeddings=result.phrase_embeddings,
            maps=result.maps,
            confidence=result.confidence * field_confidence.unsqueeze(0),
        )

    def _pooling_attention_hook(self) -> tuple[Any | None, list[Tensor]]:
        captured: list[Tensor] = []
        module: Any = self.model
        for path in (
            ("vision_model", "vision_model", "head", "attention"),
            ("vision_model", "head", "attention"),
        ):
            candidate: Any = module
            for name in path:
                candidate = getattr(candidate, name, None)
                if candidate is None:
                    break
            if candidate is not None and hasattr(candidate, "register_forward_hook"):
                def save_attention(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                ) -> None:
                    if isinstance(output, tuple) and len(output) > 1:
                        captured.append(output[1])

                return candidate.register_forward_hook(save_attention), captured
        return None, captured

    def _encode_one(
        self,
        images: Tensor,
        phrase: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        processed = self.processor(
            images=images,
            text=[phrase],
            padding="max_length",
            return_tensors="pt",
        )
        device = _module_device(self.model)
        inputs = {
            key: value.to(device=device)
            for key, value in _as_tensor_mapping(processed).items()
        }
        if "pixel_values" not in inputs:
            raise ValueError("SigLIP2 processor did not return pixel_values")
        pixels = inputs["pixel_values"].detach().requires_grad_(True)
        inputs["pixel_values"] = pixels

        hook, captured_attention = self._pooling_attention_hook()
        try:
            output = self.model(**inputs)
        finally:
            if hook is not None:
                hook.remove()

        vision_output = getattr(output, "vision_model_output", None)
        spatial = getattr(vision_output, "last_hidden_state", None)
        if spatial is None or tuple(spatial.shape[1:2]) != (_TOKENS,):
            actual = None if spatial is None else tuple(spatial.shape)
            raise ValueError(
                "SigLIP2 final spatial activations must have shape "
                f"[frames, {_TOKENS}, width], got {actual}"
            )
        self._captured_spatial_activations = spatial
        spatial.retain_grad()

        logits = getattr(output, "logits_per_image", None)
        if logits is None or logits.shape[0] != images.shape[0]:
            raise ValueError("SigLIP2 output must contain per-image similarity logits")
        score = logits[:, 0]
        score.sum().backward()
        gradient = spatial.grad
        if gradient is None:
            raise RuntimeError("SigLIP2 similarity did not reach spatial activations")

        relevance = (gradient * spatial).sum(dim=-1).clamp_min(0)
        captured = captured_attention[-1] if captured_attention else None
        attention = _pooling_attention_from_output(output, captured)
        if attention is not None:
            attention = attention.to(relevance).clamp_min(0)
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
            relevance = 0.5 * relevance + 0.5 * attention
        finite = torch.isfinite(relevance).all(dim=-1) & torch.isfinite(score)
        relevance = torch.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
        mass = relevance.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(relevance, 1.0 / _TOKENS)
        relevance = torch.where(mass > 0, relevance / mass.clamp_min(1e-12), uniform)

        embedding = getattr(output, "text_embeds", None)
        if embedding is None or embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError("SigLIP2 output must contain one text embedding")
        if embedding.shape[-1] != _TEXT_WIDTH:
            raise ValueError(
                f"SigLIP2 text embeddings must have width {_TEXT_WIDTH}, "
                f"got {embedding.shape[-1]}"
            )
        embedding_finite = torch.isfinite(embedding).all()
        finite = finite & embedding_finite
        embedding = torch.nan_to_num(
            embedding[0].float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        embedding = F.normalize(embedding, dim=-1)
        return embedding.detach(), relevance.detach(), score.detach(), finite.detach(), gradient.detach()

    @staticmethod
    def _confidence(
        maps: Tensor,
        positive_score: Tensor,
        negative_score: Tensor | None,
        finite: Tensor,
    ) -> Tensor:
        peak = maps.amax(dim=-1)
        peak_signal = ((peak - (1.0 / _TOKENS)) / (1.0 - (1.0 / _TOKENS))).clamp(
            0, 1
        )
        baseline = (
            torch.zeros_like(positive_score)
            if negative_score is None
            else negative_score
        )
        margin_signal = torch.sigmoid(positive_score - baseline)
        confidence = torch.sqrt(peak_signal * margin_signal)
        return torch.where(finite, confidence, torch.zeros_like(confidence)).clamp(0, 1)

    def _counterfactual_scores(self, images: Tensor, phrase: str) -> Tensor:
        processed = self.processor(
            images=images,
            text=[phrase],
            padding="max_length",
            return_tensors="pt",
        )
        device = _module_device(self.model)
        inputs = {
            key: value.to(device=device)
            for key, value in _as_tensor_mapping(processed).items()
        }
        if "pixel_values" not in inputs:
            raise ValueError("SigLIP2 processor did not return pixel_values")
        inputs["pixel_values"] = inputs["pixel_values"].detach().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(**inputs)
        logits = getattr(output, "logits_per_image", None)
        if logits is None:
            raise ValueError("SigLIP2 output must contain per-image similarity logits")
        return logits[:, 0].detach()

    def encode(
        self,
        rgb: Tensor,
        *,
        phrases: Sequence[str],
        counterfactual_phrases: Sequence[str] | None = None,
    ) -> PhraseRelevance:
        """Encode phrases and produce normalized 27x27 relevance per frame."""

        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [frames, 3, height, width]")
        if len(phrases) == 0:
            raise ValueError("phrases must not be empty")
        if counterfactual_phrases is not None and len(counterfactual_phrases) != len(
            phrases
        ):
            raise ValueError("counterfactual_phrases must align with phrases")
        if not rgb.is_floating_point():
            rgb = rgb.float().div(255)
        if not bool(torch.isfinite(rgb).all()):
            raise ValueError("rgb must contain finite values")
        images = F.interpolate(
            rgb,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        embeddings: list[Tensor] = []
        maps: list[Tensor] = []
        confidences: list[Tensor] = []
        gradients: list[Tensor] = []
        device = _module_device(self.model)
        try:
            for index, raw_phrase in enumerate(phrases):
                phrase = raw_phrase.strip()
                if not phrase:
                    embeddings.append(torch.zeros(_TEXT_WIDTH, device=device))
                    maps.append(
                        torch.zeros(
                            images.shape[0],
                            _TOKENS,
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    confidences.append(
                        torch.zeros(images.shape[0], device=device, dtype=torch.float32)
                    )
                    continue
                with torch.enable_grad():
                    embedding, relevance, score, finite, gradient = self._encode_one(
                        images, phrase
                    )
                negative_score = None
                if counterfactual_phrases is not None:
                    negative = counterfactual_phrases[index].strip()
                    if negative:
                        negative_score = self._counterfactual_scores(images, negative)
                embeddings.append(embedding)
                maps.append(relevance)
                confidences.append(
                    self._confidence(relevance, score, negative_score, finite)
                )
                gradients.append(gradient)
                self._captured_spatial_activations = None
        finally:
            self._captured_spatial_activations = None
            for parameter in self.model.parameters():
                parameter.grad = None
        self.last_spatial_gradients = tuple(gradients)
        return PhraseRelevance(
            phrase_embeddings=torch.stack(embeddings),
            maps=torch.stack(maps, dim=1).reshape(
                images.shape[0], len(phrases), _GRID_SIZE, _GRID_SIZE
            ),
            confidence=torch.stack(confidences, dim=1),
        )
