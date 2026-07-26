"""Project-specific text-aligned tokenizer over SigLIP2 spatial features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from qwen35_planx.config import PlanGeometry


def nearest_code_indices(
    queries: Tensor,
    codebook: Tensor,
    *,
    codebook_chunk_size: int = 2048,
) -> Tensor:
    """Return cosine-nearest code IDs without materializing the full matrix."""

    if queries.ndim < 2 or codebook.ndim != 2:
        raise ValueError("queries must be [..., D] and codebook must be [V, D]")
    if queries.shape[-1] != codebook.shape[-1]:
        raise ValueError("queries and codebook must have the same feature width")
    if codebook.shape[0] == 0 or codebook_chunk_size <= 0:
        raise ValueError("codebook and codebook_chunk_size must be nonempty")

    original_shape = queries.shape[:-1]
    flat_queries = F.normalize(queries.reshape(-1, queries.shape[-1]), dim=-1)
    normalized_codebook = F.normalize(codebook, dim=-1)
    best_similarity = torch.full(
        (flat_queries.shape[0],),
        -torch.inf,
        dtype=flat_queries.dtype,
        device=flat_queries.device,
    )
    best_indices = torch.zeros(
        flat_queries.shape[0], dtype=torch.long, device=flat_queries.device
    )
    for start in range(0, codebook.shape[0], codebook_chunk_size):
        stop = min(start + codebook_chunk_size, codebook.shape[0])
        similarities = flat_queries @ normalized_codebook[start:stop].T
        chunk_similarity, chunk_index = similarities.max(dim=-1)
        improved = chunk_similarity > best_similarity
        best_similarity = torch.where(
            improved, chunk_similarity, best_similarity
        )
        best_indices = torch.where(
            improved, chunk_index + start, best_indices
        )
    return best_indices.reshape(original_shape)


def codebook_usage_metrics(
    codes: Tensor,
    *,
    vocabulary_size: int,
) -> dict[str, Tensor]:
    """Compute usage, perplexity, and dead-code ratio for integer codes."""

    if vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be positive")
    flat_codes = codes.detach().reshape(-1).to(dtype=torch.long)
    if flat_codes.numel() == 0:
        raise ValueError("codes must not be empty")
    if flat_codes.min() < 0 or flat_codes.max() >= vocabulary_size:
        raise ValueError("codes are outside vocabulary range")
    counts = torch.bincount(flat_codes, minlength=vocabulary_size).float()
    probabilities = counts / counts.sum()
    nonzero = probabilities > 0
    perplexity = torch.exp(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    )
    used = nonzero.float().mean()
    return {
        "code_usage": used,
        "perplexity": perplexity,
        "dead_code_ratio": 1.0 - used,
    }


@dataclass
class TATokOutput:
    codes: Tensor
    quantized: Tensor
    reconstruction: Tensor
    losses: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]

    @property
    def loss(self) -> Tensor:
        return sum(self.losses.values())


class TextAlignedTokenizer(nn.Module):
    """SigLIP2 student quantized by projected frozen Qwen embeddings."""

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        frozen_anchors: Tensor,
        feature_dim: int,
        qwen_dim: int,
        decoder_depth: int = 3,
        decoder_num_heads: int = 8,
        codebook_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        geometry = PlanGeometry()
        if frozen_anchors.ndim != 2:
            raise ValueError("frozen_anchors must have shape [V, Dq]")
        if frozen_anchors.shape[1] != qwen_dim:
            raise ValueError("frozen_anchors width must equal qwen_dim")
        if feature_dim <= 0 or qwen_dim <= 0:
            raise ValueError("feature_dim and qwen_dim must be positive")
        if decoder_depth != 3:
            raise ValueError("production TA-Tok decoder_depth must be 3")
        if decoder_num_heads <= 0 or feature_dim % decoder_num_heads:
            raise ValueError("decoder_num_heads must divide feature_dim")
        if codebook_chunk_size <= 0:
            raise ValueError("codebook_chunk_size must be positive")

        self.student = student
        self.teacher = teacher
        self.feature_dim = feature_dim
        self.qwen_dim = qwen_dim
        self.tokens_per_frame = geometry.tokens_per_frame
        self.codebook_chunk_size = codebook_chunk_size
        self.register_buffer(
            "frozen_anchors",
            frozen_anchors.detach().float().clone(),
            persistent=True,
        )

        self.student_projection = nn.Linear(feature_dim, qwen_dim, bias=False)
        self.codebook_projection = nn.Linear(qwen_dim, qwen_dim, bias=False)
        nn.init.eye_(self.codebook_projection.weight)
        self.decoder_input = nn.Linear(qwen_dim, feature_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=decoder_num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth
        )
        self.decoder_output = nn.Linear(feature_dim, feature_dim)
        self._freeze_teacher()

    @classmethod
    def from_modules(
        cls,
        *,
        student: nn.Module,
        teacher: nn.Module,
        frozen_anchors: Tensor,
        feature_dim: int,
        qwen_dim: int,
        decoder_depth: int = 3,
        decoder_num_heads: int = 8,
        codebook_chunk_size: int = 2048,
    ) -> TextAlignedTokenizer:
        student.load_state_dict(teacher.state_dict(), strict=True)
        return cls(
            student=student,
            teacher=teacher,
            frozen_anchors=frozen_anchors,
            feature_dim=feature_dim,
            qwen_dim=qwen_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            codebook_chunk_size=codebook_chunk_size,
        )

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> TextAlignedTokenizer:
        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def preprocess_images(images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if not images.is_floating_point():
            raise ValueError("images must be floating point RGB in [0, 1]")
        if images.shape[-2:] != (256, 256):
            images = F.interpolate(
                images,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (images - 0.5) / 0.5

    def _extract_features(self, module: nn.Module, images: Tensor) -> Tensor:
        output = module(
            pixel_values=self.preprocess_images(images),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None or len(hidden_states) < 2:
            raise ValueError("SigLIP2 must return at least two hidden states")
        features = hidden_states[-2]
        if (
            features.ndim != 3
            or features.shape[1] != self.tokens_per_frame
            or features.shape[2] != self.feature_dim
        ):
            raise ValueError(
                "SigLIP2 penultimate hidden state must have shape "
                f"[B, 256, {self.feature_dim}], got {tuple(features.shape)}"
            )
        return features

    def extract_student_features(self, images: Tensor) -> Tensor:
        return self._extract_features(self.student, images)

    def extract_teacher_features(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            return self._extract_features(self.teacher, images).detach()

    def projected_codebook(self) -> Tensor:
        return F.normalize(
            self.codebook_projection(self.frozen_anchors.float()), dim=-1
        )

    def _decode_quantized(self, quantized: Tensor) -> Tensor:
        decoded = self.decoder_input(quantized)
        decoded = self.decoder_blocks(decoded)
        return self.decoder_output(decoded)

    def encode_codes(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.extract_student_features(images)
            student_z = F.normalize(
                self.student_projection(features.float()), dim=-1
            )
            return nearest_code_indices(
                student_z,
                self.projected_codebook(),
                codebook_chunk_size=self.codebook_chunk_size,
            )

    def decode_codes(self, codes: Tensor) -> Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.tokens_per_frame:
            raise ValueError(
                f"codes must have shape [B, {self.tokens_per_frame}]"
            )
        if codes.min() < 0 or codes.max() >= self.frozen_anchors.shape[0]:
            raise ValueError("codes are outside TA-Tok vocabulary range")
        selected = F.embedding(codes.long(), self.projected_codebook())
        return self._decode_quantized(selected)

    def forward(self, images: Tensor) -> TATokOutput:
        student_features = self.extract_student_features(images)
        teacher_features = self.extract_teacher_features(images)
        student_z = F.normalize(
            self.student_projection(student_features.float()), dim=-1
        )
        codebook = self.projected_codebook()
        with torch.no_grad():
            codes = nearest_code_indices(
                student_z.detach(),
                codebook.detach(),
                codebook_chunk_size=self.codebook_chunk_size,
            )
        selected = F.embedding(codes, codebook)
        quantized = student_z + (selected - student_z).detach()
        reconstruction = self._decode_quantized(quantized)

        reconstruction_cosine = F.cosine_similarity(
            reconstruction, teacher_features.detach().float(), dim=-1
        ).mean()
        losses = {
            "reconstruction": 1.0 - reconstruction_cosine,
            "commitment": 0.25
            * F.mse_loss(student_z, selected.detach()),
            "codebook": F.mse_loss(selected, student_z.detach()),
        }
        metrics = codebook_usage_metrics(
            codes, vocabulary_size=codebook.shape[0]
        )
        metrics["reconstruction_cosine"] = reconstruction_cosine.detach()
        return TATokOutput(
            codes=codes,
            quantized=quantized,
            reconstruction=reconstruction,
            losses=losses,
            metrics=metrics,
        )
