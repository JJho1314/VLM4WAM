"""Safe, frozen inference adapter for the released 384px TA-Tok checkpoint.

This module intentionally reproduces only the released inference graph.  It
does not depend on the historical TA-Tok training implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from easydict import EasyDict
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoModel, Siglip2VisionConfig, Siglip2VisionModel

from qwen35_planx.config import PlanGeometry, ReleasedTATokMetadata
from qwen35_planx.hashing import sha256_file


_TEACHER = "google/siglip2-so400m-patch14-384"
_CODEBOOK_KEYS = (
    "bottleneck.regularizer.embedding.weight",
    "bottleneck.regularizer.embedding_proj.weight",
    "bottleneck.regularizer.embedding_proj.bias",
)
_EXPECTED_CODEBOOK_SHAPES = {
    _CODEBOOK_KEYS[0]: (65_536, 1536),
    _CODEBOOK_KEYS[1]: (1536, 1536),
    _CODEBOOK_KEYS[2]: (1536,),
}
_EXPECTED_ARGUMENTS: Mapping[tuple[str, ...], Any] = {
    ("bottleneck", "name"): "bottleneck",
    ("bottleneck", "args", "bottleneck_dim"): 1536,
    ("bottleneck", "args", "norm"): "none",
    ("bottleneck", "args", "regularizer", "name"): "simvq",
    (
        "bottleneck",
        "args",
        "regularizer",
        "args",
        "codebook_size",
    ): 65_536,
    (
        "bottleneck",
        "args",
        "regularizer",
        "args",
        "commitment_loss_weight",
    ): 0.25,
    (
        "bottleneck",
        "args",
        "regularizer",
        "args",
        "codebook_loss_weight",
    ): 1.0,
    (
        "bottleneck",
        "args",
        "regularizer",
        "args",
        "entropy_loss_weight",
    ): 0.0,
    (
        "bottleneck",
        "args",
        "regularizer",
        "args",
        "entropy_loss_temperature",
    ): 0.01,
    ("bottleneck", "args", "regularizer", "args", "l2_normalized"): True,
    ("bottleneck", "args", "regularizer", "args", "stochastic"): True,
    (
        "bottleneck",
        "args",
        "regularizer",
        "args",
        "stochastic_temperature",
    ): 0.03,
    ("bottleneck", "args", "regularizer", "args", "top_k"): 4,
    ("bottleneck", "args", "regularizer", "args", "top_k_prob"): 0.5,
    ("bottleneck", "args", "regularizer", "args", "residual_weight"): 0.1,
    ("bottleneck_token_num",): 729,
    ("input_size",): 384,
    ("teacher",): _TEACHER,
    ("ckpt_path",): _TEACHER,
    ("pool_scale",): 1,
    ("rand_scale",): True,
}


@dataclass(frozen=True)
class EncodedCodes:
    """Discrete released-TA-Tok output for a batch of images."""

    codes: torch.Tensor


@dataclass(frozen=True)
class ReleasedCheckpointInspection:
    """Validated, CPU-only contents needed to construct the inference model."""

    arguments: Mapping[str, Any]
    state_dict: Mapping[str, torch.Tensor]
    checkpoint_hash: str
    state_hash: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return (729, 65_536, 1536)


def _nested_value(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"missing released checkpoint argument: {'.'.join(path)}")
        value = value[part]
    return value


def _validate_arguments(arguments: Mapping[str, Any]) -> None:
    if not isinstance(arguments, Mapping):
        raise ValueError("released checkpoint model.args must be a mapping")
    for path, expected in _EXPECTED_ARGUMENTS.items():
        actual = _nested_value(arguments, path)
        if actual != expected:
            name = ".".join(path)
            raise ValueError(
                f"released checkpoint {name} must be {expected!r}, got {actual!r}"
            )


def _validate_state_dict(state_dict: Mapping[str, Any]) -> None:
    if not isinstance(state_dict, Mapping):
        raise ValueError("released checkpoint model.sd must be a mapping")
    for key in _CODEBOOK_KEYS:
        if key not in state_dict:
            raise ValueError(f"released checkpoint is missing required key: {key}")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"released checkpoint {key} must be a tensor")
        expected = _EXPECTED_CODEBOOK_SHAPES[key]
        if tuple(tensor.shape) != expected:
            label = "codebook shape" if key.endswith("embedding.weight") else key
            raise ValueError(
                f"released checkpoint {label} must be {expected}, "
                f"got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"released checkpoint {key} contains non-finite values")


def _hash_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor state deterministically without serializing Python objects."""

    digest = hashlib.sha256()
    chunk_elements = 4 * 1024 * 1024
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        flattened = tensor.reshape(-1)
        for start in range(0, flattened.numel(), chunk_elements):
            chunk = flattened[start : start + chunk_elements].contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inspect_released_checkpoint(
    checkpoint_path: Path | str,
    *,
    compute_state_hash: bool = True,
) -> ReleasedCheckpointInspection:
    """Safely load and validate released checkpoint metadata on CPU."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"released TA-Tok checkpoint does not exist: {path}")
    with torch.serialization.safe_globals([EasyDict]):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("released checkpoint is missing model")
    model_payload = checkpoint["model"]
    if not isinstance(model_payload, Mapping):
        raise ValueError("released checkpoint model must be a mapping")
    if "args" not in model_payload:
        raise ValueError("released checkpoint is missing model.args")
    if "sd" not in model_payload:
        raise ValueError("released checkpoint is missing model.sd")
    arguments = model_payload["args"]
    state_dict = model_payload["sd"]
    _validate_arguments(arguments)
    _validate_state_dict(state_dict)
    checkpoint_hash = sha256_file(path)
    state_hash = _hash_state_dict(state_dict) if compute_state_hash else ""
    return ReleasedCheckpointInspection(
        arguments=arguments,
        state_dict=state_dict,
        checkpoint_hash=checkpoint_hash,
        state_hash=state_hash,
    )


class _ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.full((1, 3, 1, 1), 0.5))
        self.register_buffer("scale", torch.full((1, 3, 1, 1), 0.5))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.shift) / self.scale


class SimVectorQuantizer(nn.Module):
    """Inference-only form of the released similarity vector quantizer."""

    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        l2_normalized: bool,
        stochastic: bool,
        stochastic_temperature: float,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.default_stochastic_temperature = stochastic_temperature
        self.eval_deterministic = False
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding_proj = nn.Linear(dim, dim)

    def set_eval_deterministic(self, deterministic: bool = True) -> None:
        self.eval_deterministic = deterministic

    def get_emb(self) -> torch.Tensor:
        weight = self.embedding_proj.weight.float()
        embedding = self.embedding.weight.float()
        bias = self.embedding_proj.bias.float()
        if not bool(torch.count_nonzero(weight)):
            projected = bias.expand(embedding.shape[0], -1)
        else:
            projected = F.linear(embedding, weight, bias)
        return (
            F.normalize(projected, dim=-1)
            if self.l2_normalized
            else projected
        )

    def encode_indices(self, values: torch.Tensor) -> torch.Tensor:
        embeddings = self.get_emb()
        values = values.float()
        if self.l2_normalized:
            values = F.normalize(values, dim=-1)
        flattened = values.reshape(-1, values.shape[-1])
        if not bool(torch.count_nonzero(flattened)):
            return torch.zeros(
                values.shape[:-1], dtype=torch.long, device=values.device
            )
        best_indices = torch.zeros(
            flattened.shape[0], dtype=torch.long, device=flattened.device
        )
        best_scores = torch.full(
            (flattened.shape[0],),
            -torch.inf,
            dtype=torch.float32,
            device=flattened.device,
        )
        for start in range(0, self.codebook_size, 4096):
            scores = flattened @ embeddings[start : start + 4096].T
            chunk_scores, chunk_indices = scores.max(dim=-1)
            replace = chunk_scores > best_scores
            best_scores = torch.where(replace, chunk_scores, best_scores)
            best_indices = torch.where(replace, chunk_indices + start, best_indices)
        return best_indices.reshape(values.shape[:-1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.get_emb())


class _Bottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_linear = nn.Linear(1152, 1536)
        self.out_linear = nn.Linear(1536, 1536)
        self.regularizer = SimVectorQuantizer(
            dim=1536,
            codebook_size=65_536,
            l2_normalized=True,
            stochastic=True,
            stochastic_temperature=0.03,
        )


class ReleasedTATok(nn.Module):
    """Frozen adapter matching the publicly released 384px TA-Tok state."""

    IMAGE_SIZE = 384
    TOKENS = 729
    CODEBOOK_SIZE = 65_536
    CODE_DIM = 1536
    FEATURE_DIM = 1152
    SELECTED_LAYER = -2

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        decoder_config = Siglip2VisionConfig()
        decoder_config.update(
            {
                "patch_size": 1,
                "num_hidden_layers": 3,
                "num_channels": self.CODE_DIM,
                "hidden_size": self.FEATURE_DIM,
            }
        )
        self.decoder = Siglip2VisionModel(decoder_config)
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.FEATURE_DIM),
            nn.Tanh(),
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.FEATURE_DIM),
            nn.Tanh(),
            nn.Linear(self.FEATURE_DIM, self.FEATURE_DIM),
        )
        self.bottleneck = _Bottleneck()
        self.scale_layer = _ScalingLayer()
        self.checkpoint_hash = ""
        self.state_hash = ""
        self.metadata = ReleasedTATokMetadata.example()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        siglip_model_path: Path | str | None = None,
        encoder_factory: Callable[[], nn.Module] | None = None,
        weights_only: bool = True,
    ) -> "ReleasedTATok":
        if not weights_only:
            raise ValueError("released TA-Tok loading requires weights_only=True")
        inspection = inspect_released_checkpoint(checkpoint_path)
        if encoder_factory is not None:
            encoder = encoder_factory()
        else:
            if siglip_model_path is None:
                raise ValueError(
                    "siglip_model_path is required when encoder_factory is not provided"
                )
            config = AutoConfig.from_pretrained(
                siglip_model_path,
                local_files_only=True,
            )
            encoder = AutoModel.from_config(config).vision_model
        model = cls(encoder)
        model.load_state_dict(inspection.state_dict, strict=True)
        model.bottleneck.regularizer.set_eval_deterministic(True)
        model.eval()
        model.requires_grad_(False)
        model.checkpoint_hash = inspection.checkpoint_hash
        model.state_hash = inspection.state_hash
        model.metadata = ReleasedTATokMetadata(
            format_version=ReleasedTATokMetadata.FORMAT_VERSION,
            tokenizer_type="released_ta_tok",
            teacher=_TEACHER,
            image_size=cls.IMAGE_SIZE,
            grid_size=27,
            bottleneck_token_num=cls.TOKENS,
            codebook_size=cls.CODEBOOK_SIZE,
            codebook_dim=cls.CODE_DIM,
            selected_layer=cls.SELECTED_LAYER,
            pool_scale=1,
            checkpoint_hash=inspection.checkpoint_hash,
        )
        return model

    @property
    def codebook(self) -> torch.Tensor:
        regularizer = self.bottleneck.regularizer
        embedding = regularizer.embedding.weight
        projection = regularizer.embedding_proj
        if not bool(torch.count_nonzero(projection.weight)):
            projected = projection.bias.float().expand(embedding.shape[0], -1)
        else:
            projected = projection(embedding).float()
        return (
            F.normalize(projected, dim=-1)
            if regularizer.l2_normalized
            else projected
        )

    @torch.inference_mode()
    def encode_codes(self, images: torch.Tensor) -> EncodedCodes:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if not images.is_floating_point():
            raise ValueError("images must be floating RGB values in [0, 1]")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("images contain non-finite values")
        if bool((images < 0).any()) or bool((images > 1).any()):
            raise ValueError("images must contain RGB values in [0, 1]")
        images = F.interpolate(
            images,
            size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = self.scale_layer(images)
        hidden_states = self.encoder(
            images,
            output_hidden_states=True,
        ).hidden_states
        features = hidden_states[self.SELECTED_LAYER]
        if tuple(features.shape[1:]) != (self.TOKENS, self.FEATURE_DIM):
            raise ValueError(
                "SigLIP2 hidden layer -2 must have shape "
                f"[batch, {self.TOKENS}, {self.FEATURE_DIM}]"
            )
        features = self.encode_task_layer(
            features.to(dtype=self.bottleneck.in_linear.weight.dtype)
        )
        projected = self.bottleneck.in_linear(features)
        codes = self.bottleneck.regularizer.encode_indices(projected)
        return EncodedCodes(codes=codes)

    @torch.inference_mode()
    def lookup_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dtype != torch.long:
            raise ValueError("TA-Tok codes must have dtype torch.long")
        if codes.ndim < 1:
            raise ValueError("TA-Tok codes must have at least one dimension")
        if bool((codes < 0).any()) or bool((codes >= self.CODEBOOK_SIZE).any()):
            raise ValueError("TA-Tok codes are outside the released codebook")
        return F.embedding(codes, self.codebook)

    @torch.inference_mode()
    def decode_features(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.lookup_codes(codes)
        if quantized.ndim != 3 or quantized.shape[1] != self.TOKENS:
            raise ValueError(f"codes must have shape [batch, {self.TOKENS}]")
        values = self.bottleneck.out_linear(
            quantized.to(dtype=self.bottleneck.out_linear.weight.dtype)
        )
        attention_mask = torch.ones(
            values.shape[:2],
            dtype=torch.int,
            device=values.device,
        )
        spatial_shapes = torch.tensor(
            [[27, 27]] * values.shape[0],
            dtype=torch.long,
            device=values.device,
        )
        decoded = self.decoder(
            values,
            attention_mask,
            spatial_shapes,
            output_hidden_states=True,
        ).last_hidden_state
        return self.decode_task_layer(decoded)


def _temporary_sibling(path: Path) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temporary)


def export_codebook_safetensors(
    tokenizer: ReleasedTATok,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Atomically export only the immutable float32 codebook and its hashes."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path = directory / "ta_codebook.safetensors"
    metadata_path = directory / "ta_codebook.json"
    tensor_temporary = _temporary_sibling(tensor_path)
    metadata_temporary = _temporary_sibling(metadata_path)
    geometry = PlanGeometry()
    metadata = {
        "format_version": 1,
        "checkpoint_sha256": tokenizer.checkpoint_hash,
        "state_sha256": tokenizer.state_hash,
        "geometry": {
            **asdict(geometry),
            "tokens_per_frame": geometry.tokens_per_frame,
        },
        "teacher": tokenizer.metadata.to_dict(),
    }
    try:
        codebook = tokenizer.codebook.detach().float().cpu().contiguous()
        if tuple(codebook.shape) != (65_536, 1536):
            raise ValueError("released codebook has an incompatible shape")
        if not bool(torch.isfinite(codebook).all()):
            raise ValueError("released codebook contains non-finite values")
        save_file({"codebook": codebook}, tensor_temporary)
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tensor_temporary, tensor_path)
        os.replace(metadata_temporary, metadata_path)
    finally:
        tensor_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
    return tensor_path, metadata_path
