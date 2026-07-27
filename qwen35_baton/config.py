"""Immutable compatibility contracts for continuous Baton checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, ClassVar, Mapping


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _require_nonempty_strings(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(instance, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(instance, name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} must be a canonical lowercase SHA-256 hex digest")


def _require_keys(payload: Mapping[str, Any], required: tuple[str, ...]) -> None:
    missing = sorted(set(required).difference(payload))
    unexpected = sorted(set(payload).difference(required))
    if missing:
        raise ValueError(f"missing required metadata fields: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"unexpected metadata fields: {', '.join(unexpected)}")


@dataclass(frozen=True)
class BatonGeometry:
    """Fixed dual-camera, four-keyframe continuous feature geometry."""

    camera_names: tuple[str, ...] = ("main", "wrist")
    future_indices: tuple[int, ...] = (0, 3, 5, 8)
    image_size: int = 256
    patch_size: int = 16
    grid_size: int = 16
    feature_dim: int = 1024
    query_dim: int = 1024
    query_layers: int = 4
    query_heads: int = 16
    query_ffn_dim: int = 4096
    query_dropout: float = 0.1

    def __post_init__(self) -> None:
        _require_equal("camera_names", self.camera_names, ("main", "wrist"))
        _require_equal("future_indices", self.future_indices, (0, 3, 5, 8))
        _require_equal("image_size", self.image_size, 256)
        _require_equal("patch_size", self.patch_size, 16)
        _require_equal("grid_size", self.grid_size, 16)
        _require_equal("feature_dim", self.feature_dim, 1024)
        _require_equal("query_dim", self.query_dim, 1024)
        _require_equal("query_layers", self.query_layers, 4)
        _require_equal("query_heads", self.query_heads, 16)
        _require_equal("query_ffn_dim", self.query_ffn_dim, 4096)
        _require_equal("query_dropout", self.query_dropout, 0.1)

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def tokens_per_camera(self) -> int:
        return len(self.future_indices) * self.tokens_per_frame

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        return (
            batch_size,
            len(self.camera_names),
            len(self.future_indices),
            self.tokens_per_frame,
            self.feature_dim,
        )


@dataclass(frozen=True)
class BatonLossWeights:
    """Fixed Stage-1 continuous-feature objective weights."""

    mse: float = 1.0
    cosine: float = 0.5
    delta: float = 0.5
    instruction_counterfactual: float = 0.2
    counterfactual_margin: float = 0.1

    def __post_init__(self) -> None:
        _require_equal("mse", self.mse, 1.0)
        _require_equal("cosine", self.cosine, 0.5)
        _require_equal("delta", self.delta, 0.5)
        _require_equal("instruction_counterfactual", self.instruction_counterfactual, 0.2)
        _require_equal("counterfactual_margin", self.counterfactual_margin, 0.1)


@dataclass(frozen=True)
class BatonCheckpointMetadata:
    """Serialized compatibility contract for a continuous Baton checkpoint."""

    FORMAT_VERSION: ClassVar[int] = 1
    ARCHITECTURE_KIND: ClassVar[str] = "qwen35_baton_continuous"
    QWEN_BACKBONE: ClassVar[str] = "dense Qwen3.5-2B"
    SIGLIP2_MODEL: ClassVar[str] = "SigLIP2-large-patch16-256"
    QUERY_NORM_STYLE: ClassVar[str] = "pre_norm"
    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "format_version",
        "architecture_kind",
        "qwen_backbone",
        "qwen_config_hash",
        "tokenizer_hash",
        "processor_hash",
        "input_template_hash",
        "added_tokens",
        "added_token_ids",
        "camera_names",
        "camera_flattening",
        "siglip2_model",
        "siglip2_config_hash",
        "siglip2_artifact_hash",
        "teacher_image_size",
        "teacher_patch_size",
        "teacher_feature_layer",
        "teacher_preprocessing_hash",
        "teacher_dtype",
        "target_shape",
        "future_indices",
        "query_dim",
        "query_layers",
        "query_heads",
        "query_ffn_dim",
        "query_dropout",
        "query_norm_style",
        "query_mask_version",
        "trainable_qwen_layer_indices",
        "loss_weights",
        "hdf5_manifest_hash",
        "optimizer_topology_hash",
        "scheduler_topology_hash",
        "global_step",
        "distributed_cursor",
        "rng_state_hash",
    )
    _HASH_FIELDS: ClassVar[tuple[str, ...]] = (
        "qwen_config_hash",
        "tokenizer_hash",
        "processor_hash",
        "input_template_hash",
        "siglip2_config_hash",
        "siglip2_artifact_hash",
        "teacher_preprocessing_hash",
        "hdf5_manifest_hash",
        "optimizer_topology_hash",
        "scheduler_topology_hash",
        "rng_state_hash",
    )

    format_version: int
    architecture_kind: str
    qwen_backbone: str
    qwen_config_hash: str
    tokenizer_hash: str
    processor_hash: str
    input_template_hash: str
    added_tokens: tuple[str, ...]
    added_token_ids: tuple[int, ...]
    camera_names: tuple[str, ...]
    camera_flattening: str
    siglip2_model: str
    siglip2_config_hash: str
    siglip2_artifact_hash: str
    teacher_image_size: int
    teacher_patch_size: int
    teacher_feature_layer: int
    teacher_preprocessing_hash: str
    teacher_dtype: str
    target_shape: tuple[int, ...]
    future_indices: tuple[int, ...]
    query_dim: int
    query_layers: int
    query_heads: int
    query_ffn_dim: int
    query_dropout: float
    query_norm_style: str
    query_mask_version: str
    trainable_qwen_layer_indices: tuple[int, ...]
    loss_weights: BatonLossWeights
    hdf5_manifest_hash: str
    optimizer_topology_hash: str
    scheduler_topology_hash: str
    global_step: int
    distributed_cursor: tuple[tuple[str, int], ...]
    rng_state_hash: str

    def __post_init__(self) -> None:
        geometry = BatonGeometry()
        _require_equal("format_version", self.format_version, self.FORMAT_VERSION)
        _require_equal(
            "architecture_kind", self.architecture_kind, self.ARCHITECTURE_KIND
        )
        _require_equal("qwen_backbone", self.qwen_backbone, self.QWEN_BACKBONE)
        _require_sha256(self, self._HASH_FIELDS)
        _require_nonempty_strings(self, ("teacher_dtype", "query_mask_version"))
        _require_equal("added_tokens", self.added_tokens, _PLAN_TOKENS)
        if (
            len(self.added_token_ids) != len(self.added_tokens)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in self.added_token_ids
            )
            or len(set(self.added_token_ids)) != len(self.added_token_ids)
        ):
            raise ValueError("added_token_ids must contain one unique ID per added token")
        _require_equal("camera_names", self.camera_names, geometry.camera_names)
        _require_equal("camera_flattening", self.camera_flattening, "sample_major")
        _require_equal("siglip2_model", self.siglip2_model, self.SIGLIP2_MODEL)
        _require_equal("teacher_image_size", self.teacher_image_size, geometry.image_size)
        _require_equal("teacher_patch_size", self.teacher_patch_size, geometry.patch_size)
        _require_equal("teacher_feature_layer", self.teacher_feature_layer, -2)
        _require_equal("target_shape", self.target_shape, (2, 4, 256, 1024))
        _require_equal("future_indices", self.future_indices, geometry.future_indices)
        _require_equal("query_dim", self.query_dim, geometry.query_dim)
        _require_equal("query_layers", self.query_layers, geometry.query_layers)
        _require_equal("query_heads", self.query_heads, geometry.query_heads)
        _require_equal("query_ffn_dim", self.query_ffn_dim, geometry.query_ffn_dim)
        _require_equal("query_dropout", self.query_dropout, geometry.query_dropout)
        _require_equal("query_norm_style", self.query_norm_style, self.QUERY_NORM_STYLE)
        _require_equal("query_mask_version", self.query_mask_version, "block_causal_v1")
        _require_equal(
            "trainable_qwen_layer_indices",
            self.trainable_qwen_layer_indices,
            tuple(range(16, 24)),
        )
        if not isinstance(self.loss_weights, BatonLossWeights):
            raise ValueError("loss_weights must be BatonLossWeights")
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")
        if not self.distributed_cursor:
            raise ValueError("distributed_cursor must not be empty")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in self.distributed_cursor
        ):
            raise ValueError("distributed_cursor must contain non-negative named integer values")
        if len({name for name, _ in self.distributed_cursor}) != len(
            self.distributed_cursor
        ):
            raise ValueError("distributed_cursor names must be unique")

    @classmethod
    def example(cls) -> BatonCheckpointMetadata:
        geometry = BatonGeometry()
        return cls(
            format_version=cls.FORMAT_VERSION,
            architecture_kind=cls.ARCHITECTURE_KIND,
            qwen_backbone=cls.QWEN_BACKBONE,
            qwen_config_hash=_example_sha256("qwen-config"),
            tokenizer_hash=_example_sha256("tokenizer"),
            processor_hash=_example_sha256("processor"),
            input_template_hash=_example_sha256("input-template"),
            added_tokens=_PLAN_TOKENS,
            added_token_ids=tuple(range(151_665, 151_672)),
            camera_names=geometry.camera_names,
            camera_flattening="sample_major",
            siglip2_model=cls.SIGLIP2_MODEL,
            siglip2_config_hash=_example_sha256("siglip2-config"),
            siglip2_artifact_hash=_example_sha256("siglip2-artifact"),
            teacher_image_size=geometry.image_size,
            teacher_patch_size=geometry.patch_size,
            teacher_feature_layer=-2,
            teacher_preprocessing_hash=_example_sha256("siglip2-preprocessing"),
            teacher_dtype="bfloat16",
            target_shape=(2, 4, geometry.tokens_per_frame, geometry.feature_dim),
            future_indices=geometry.future_indices,
            query_dim=geometry.query_dim,
            query_layers=geometry.query_layers,
            query_heads=geometry.query_heads,
            query_ffn_dim=geometry.query_ffn_dim,
            query_dropout=geometry.query_dropout,
            query_norm_style=cls.QUERY_NORM_STYLE,
            query_mask_version="block_causal_v1",
            trainable_qwen_layer_indices=tuple(range(16, 24)),
            loss_weights=BatonLossWeights(),
            hdf5_manifest_hash=_example_sha256("hdf5-manifest"),
            optimizer_topology_hash=_example_sha256("optimizer-topology"),
            scheduler_topology_hash=_example_sha256("scheduler-topology"),
            global_step=0,
            distributed_cursor=(("epoch", 0), ("consumed_microbatches", 0), ("sampler_seed", 0)),
            rng_state_hash=_example_sha256("rng-state"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["added_tokens"] = list(self.added_tokens)
        payload["added_token_ids"] = list(self.added_token_ids)
        payload["camera_names"] = list(self.camera_names)
        payload["target_shape"] = list(self.target_shape)
        payload["future_indices"] = list(self.future_indices)
        payload["trainable_qwen_layer_indices"] = list(self.trainable_qwen_layer_indices)
        payload["distributed_cursor"] = dict(self.distributed_cursor)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BatonCheckpointMetadata:
        _require_keys(payload, cls._REQUIRED_FIELDS)
        values = {name: payload[name] for name in cls._REQUIRED_FIELDS}
        values["added_tokens"] = tuple(values["added_tokens"])
        values["added_token_ids"] = tuple(values["added_token_ids"])
        values["camera_names"] = tuple(values["camera_names"])
        values["target_shape"] = tuple(values["target_shape"])
        values["future_indices"] = tuple(values["future_indices"])
        values["trainable_qwen_layer_indices"] = tuple(values["trainable_qwen_layer_indices"])
        values["loss_weights"] = BatonLossWeights(**values["loss_weights"])
        values["distributed_cursor"] = tuple(values["distributed_cursor"].items())
        return cls(**values)


_PLAN_TOKENS = (
    "<PLAN_START>",
    "<FRAME_0>",
    "<FRAME_1>",
    "<FRAME_2>",
    "<FRAME_3>",
    "<PLAN_PAD>",
    "<PLAN_END>",
)


def _example_sha256(label: str) -> str:
    """Return deterministic, syntactically valid fixture provenance."""

    from hashlib import sha256

    return sha256(f"qwen35-baton-example:{label}".encode("utf-8")).hexdigest()
