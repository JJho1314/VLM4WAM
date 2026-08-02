"""Immutable compatibility contracts for continuous Baton checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, ClassVar, Mapping


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _require_exact_int(name: str, value: Any) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")


def _require_exact_int_tuple(name: str, value: Any) -> None:
    if not isinstance(value, tuple) or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must contain only integers")


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
    query_dim: int = 2048
    query_layers: int = 1
    query_heads: int = 16
    query_ffn_dim: int = 0
    query_dropout: float = 0.0

    def __post_init__(self) -> None:
        _require_equal("camera_names", self.camera_names, ("main", "wrist"))
        _require_equal("future_indices", self.future_indices, (0, 3, 5, 8))
        _require_equal("image_size", self.image_size, 256)
        _require_equal("patch_size", self.patch_size, 16)
        _require_equal("grid_size", self.grid_size, 16)
        _require_equal("feature_dim", self.feature_dim, 1024)
        _require_equal("query_dim", self.query_dim, 2048)
        _require_equal("query_layers", self.query_layers, 1)
        _require_equal("query_heads", self.query_heads, 16)
        _require_equal("query_ffn_dim", self.query_ffn_dim, 0)
        _require_equal("query_dropout", self.query_dropout, 0.0)

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


def exact_half_even_round(numerator: int, denominator: int) -> int:
    """Round a non-negative rational exactly, with ties going to an even integer."""

    if (
        type(numerator) is not int
        or numerator < 0
        or type(denominator) is not int
        or denominator <= 0
    ):
        raise ValueError(
            "exact half-even rounding requires a non-negative integer numerator "
            "and a positive integer denominator"
        )
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = 2 * remainder
    return quotient + int(
        twice_remainder > denominator
        or (twice_remainder == denominator and quotient % 2 == 1)
    )


@dataclass(frozen=True)
class BatonTemporalPolicy:
    """Dataset-specific meaning of the four predicted semantic frames."""

    _WORLD_ARENA_FORMULA: ClassVar[str] = (
        "f_k = c + round_half_even((k + 1) * (120 - c) / 4), k=0..3"
    )
    _ROUNDING: ClassVar[str] = "round_half_even_exact_integer_v1"

    kind: str
    offsets: tuple[int, ...] | None = None
    canonical_frame_count: int | None = None
    current_index_range: tuple[int, int] | None = None
    target_count_value: int | None = None
    rounding: str | None = None
    formula: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "fixed_offsets":
            _require_exact_int_tuple("temporal_policy.offsets", self.offsets)
            expected = {
                "offsets": (0, 3, 5, 8),
                "canonical_frame_count": None,
                "current_index_range": None,
                "target_count_value": None,
                "rounding": None,
                "formula": None,
            }
        elif self.kind == "normalized_remaining_horizon":
            _require_exact_int(
                "temporal_policy.canonical_frame_count",
                self.canonical_frame_count,
            )
            _require_exact_int_tuple(
                "temporal_policy.current_index_range",
                self.current_index_range,
            )
            _require_exact_int(
                "temporal_policy.target_count",
                self.target_count_value,
            )
            expected = {
                "offsets": None,
                "canonical_frame_count": 121,
                "current_index_range": (0, 116),
                "target_count_value": 4,
                "rounding": self._ROUNDING,
                "formula": self._WORLD_ARENA_FORMULA,
            }
        else:
            raise ValueError(f"unsupported Baton temporal policy kind: {self.kind!r}")
        for name, expected_value in expected.items():
            _require_equal(
                f"temporal_policy.{name}",
                getattr(self, name),
                expected_value,
            )

    @classmethod
    def libero_fixed(cls) -> BatonTemporalPolicy:
        return cls(
            kind="fixed_offsets",
            offsets=(0, 3, 5, 8),
        )

    @classmethod
    def worldarena_normalized(cls) -> BatonTemporalPolicy:
        return cls(
            kind="normalized_remaining_horizon",
            canonical_frame_count=121,
            current_index_range=(0, 116),
            target_count_value=4,
            rounding=cls._ROUNDING,
            formula=cls._WORLD_ARENA_FORMULA,
        )

    @property
    def target_count(self) -> int:
        if self.offsets is not None:
            return len(self.offsets)
        if self.target_count_value is None:
            raise AssertionError("validated temporal policy has no target count")
        return self.target_count_value

    def resolve_future_indices(
        self,
        *,
        current_index: int | None = None,
    ) -> tuple[int, int, int, int]:
        if self.kind == "fixed_offsets":
            if current_index is not None:
                raise ValueError(
                    "fixed-offset temporal policy does not accept current_index"
                )
            if self.offsets is None:
                raise AssertionError("validated fixed policy has no offsets")
            return self.offsets  # type: ignore[return-value]
        if type(current_index) is not int:
            raise ValueError(
                "normalized temporal policy requires an integer current_index"
            )
        if self.current_index_range is None or self.canonical_frame_count is None:
            raise AssertionError("validated normalized policy is incomplete")
        minimum, maximum = self.current_index_range
        if not minimum <= current_index <= maximum:
            raise ValueError(
                f"current_index must be in [{minimum}, {maximum}], got {current_index}"
            )
        last = self.canonical_frame_count - 1
        resolved = tuple(
            current_index
            + exact_half_even_round((step + 1) * (last - current_index), 4)
            for step in range(self.target_count)
        )
        if (
            len(resolved) != 4
            or len(set(resolved)) != 4
            or tuple(sorted(resolved)) != resolved
        ):
            raise ValueError(
                "normalized future indices must be four unique ordered positions"
            )
        return resolved  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "fixed_offsets":
            return {"kind": self.kind, "offsets": list(self.offsets or ())}
        return {
            "kind": self.kind,
            "canonical_frame_count": self.canonical_frame_count,
            "current_index_range": list(self.current_index_range or ()),
            "target_count": self.target_count,
            "rounding": self.rounding,
            "formula": self.formula,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> BatonTemporalPolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("temporal_policy must contain a JSON object")
        kind = payload.get("kind")
        if kind == "fixed_offsets":
            _require_keys(payload, ("kind", "offsets"))
            offsets = payload["offsets"]
            if not isinstance(offsets, (list, tuple)):
                raise ValueError("temporal_policy.offsets must be a sequence")
            return cls(kind=kind, offsets=tuple(offsets))
        if kind == "normalized_remaining_horizon":
            _require_keys(
                payload,
                (
                    "kind",
                    "canonical_frame_count",
                    "current_index_range",
                    "target_count",
                    "rounding",
                    "formula",
                ),
            )
            current_range = payload["current_index_range"]
            if not isinstance(current_range, (list, tuple)):
                raise ValueError(
                    "temporal_policy.current_index_range must be a sequence"
                )
            return cls(
                kind=kind,
                canonical_frame_count=payload["canonical_frame_count"],
                current_index_range=tuple(current_range),  # type: ignore[arg-type]
                target_count_value=payload["target_count"],
                rounding=payload["rounding"],
                formula=payload["formula"],
            )
        raise ValueError(f"unsupported Baton temporal policy kind: {kind!r}")


@dataclass(frozen=True)
class BatonCheckpointMetadata:
    """Serialized compatibility contract for a continuous Baton checkpoint."""

    FORMAT_VERSION: ClassVar[int] = 4
    ARCHITECTURE_KIND: ClassVar[str] = "qwen35_baton_continuous"
    QWEN_BACKBONE: ClassVar[str] = "dense Qwen3.5-2B"
    SIGLIP2_MODEL: ClassVar[str] = "SigLIP2-large-patch16-256"
    QUERY_NORM_STYLE: ClassVar[str] = "none"
    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "format_version",
        "architecture_kind",
        "distributed_strategy",
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
        "temporal_policy",
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
        "planner_topology_hash",
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
        "planner_topology_hash",
        "optimizer_topology_hash",
        "scheduler_topology_hash",
        "rng_state_hash",
    )

    format_version: int
    architecture_kind: str
    distributed_strategy: str
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
    temporal_policy: BatonTemporalPolicy
    query_dim: int
    query_layers: int
    query_heads: int
    query_ffn_dim: int
    query_dropout: float
    query_norm_style: str
    query_mask_version: str
    trainable_qwen_layer_indices: tuple[int, ...]
    loss_weights: Mapping[str, float]
    hdf5_manifest_hash: str
    planner_topology_hash: str
    optimizer_topology_hash: str
    scheduler_topology_hash: str
    global_step: int
    distributed_cursor: tuple[tuple[str, int], ...]
    rng_state_hash: str

    def __post_init__(self) -> None:
        geometry = BatonGeometry()
        for name in (
            "format_version",
            "teacher_image_size",
            "teacher_patch_size",
            "teacher_feature_layer",
            "query_dim",
            "query_layers",
            "query_heads",
            "query_ffn_dim",
            "global_step",
        ):
            _require_exact_int(name, getattr(self, name))
        _require_exact_int_tuple("target_shape", self.target_shape)
        _require_exact_int_tuple(
            "trainable_qwen_layer_indices",
            self.trainable_qwen_layer_indices,
        )
        if type(self.query_dropout) is not float:
            raise ValueError("query_dropout must be a floating-point number")
        _require_equal("format_version", self.format_version, self.FORMAT_VERSION)
        _require_equal(
            "architecture_kind", self.architecture_kind, self.ARCHITECTURE_KIND
        )
        if self.distributed_strategy not in {"ddp", "zero2"}:
            raise ValueError("distributed_strategy must be 'ddp' or 'zero2'")
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
            raise ValueError(
                "added_token_ids must contain one unique ID per added token"
            )
        if self.camera_names not in (("main", "wrist"), ("head",)):
            raise ValueError(
                "camera_names must be either ('main', 'wrist') or ('head',)"
            )
        _require_equal("camera_flattening", self.camera_flattening, "sample_major")
        _require_equal("siglip2_model", self.siglip2_model, self.SIGLIP2_MODEL)
        _require_equal(
            "teacher_image_size", self.teacher_image_size, geometry.image_size
        )
        _require_equal(
            "teacher_patch_size", self.teacher_patch_size, geometry.patch_size
        )
        _require_equal("teacher_feature_layer", self.teacher_feature_layer, -2)
        _require_equal("teacher_dtype", self.teacher_dtype, "bfloat16")
        _require_equal(
            "target_shape",
            self.target_shape,
            (len(self.camera_names), 4, 256, 1024),
        )
        if not isinstance(self.temporal_policy, BatonTemporalPolicy):
            raise TypeError("temporal_policy must be a BatonTemporalPolicy")
        expected_policy = (
            BatonTemporalPolicy.libero_fixed()
            if self.camera_names == ("main", "wrist")
            else BatonTemporalPolicy.worldarena_normalized()
        )
        if self.temporal_policy != expected_policy:
            raise ValueError(
                "camera_names and temporal_policy must identify the same dataset contract"
            )
        _require_equal("query_dim", self.query_dim, geometry.query_dim)
        _require_equal("query_layers", self.query_layers, geometry.query_layers)
        _require_equal("query_heads", self.query_heads, geometry.query_heads)
        _require_equal("query_ffn_dim", self.query_ffn_dim, geometry.query_ffn_dim)
        _require_equal("query_dropout", self.query_dropout, geometry.query_dropout)
        _require_equal("query_norm_style", self.query_norm_style, self.QUERY_NORM_STYLE)
        _require_equal(
            "query_mask_version",
            self.query_mask_version,
            "full_cross_attention_v1",
        )
        _require_equal(
            "trainable_qwen_layer_indices",
            self.trainable_qwen_layer_indices,
            tuple(range(24)),
        )
        if (
            dict(self.loss_weights) != {"mse": 1.0}
            or type(self.loss_weights.get("mse")) is not float
        ):
            raise ValueError("loss_weights must contain only Equation-8 MSE weight 1.0")
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
            raise ValueError(
                "distributed_cursor must contain non-negative named integer values"
            )
        if len({name for name, _ in self.distributed_cursor}) != len(
            self.distributed_cursor
        ):
            raise ValueError("distributed_cursor names must be unique")

    @classmethod
    def example(
        cls,
        *,
        camera_names: tuple[str, ...] = ("main", "wrist"),
        distributed_strategy: str = "ddp",
    ) -> BatonCheckpointMetadata:
        geometry = BatonGeometry()
        return cls(
            format_version=cls.FORMAT_VERSION,
            architecture_kind=cls.ARCHITECTURE_KIND,
            distributed_strategy=distributed_strategy,
            qwen_backbone=cls.QWEN_BACKBONE,
            qwen_config_hash=_example_sha256("qwen-config"),
            tokenizer_hash=_example_sha256("tokenizer"),
            processor_hash=_example_sha256("processor"),
            input_template_hash=_example_sha256("input-template"),
            added_tokens=_PLAN_TOKENS,
            added_token_ids=tuple(range(151_665, 151_672)),
            camera_names=camera_names,
            camera_flattening="sample_major",
            siglip2_model=cls.SIGLIP2_MODEL,
            siglip2_config_hash=_example_sha256("siglip2-config"),
            siglip2_artifact_hash=_example_sha256("siglip2-artifact"),
            teacher_image_size=geometry.image_size,
            teacher_patch_size=geometry.patch_size,
            teacher_feature_layer=-2,
            teacher_preprocessing_hash=_example_sha256("siglip2-preprocessing"),
            teacher_dtype="bfloat16",
            target_shape=(
                len(camera_names),
                4,
                geometry.tokens_per_frame,
                geometry.feature_dim,
            ),
            temporal_policy=(
                BatonTemporalPolicy.libero_fixed()
                if camera_names == ("main", "wrist")
                else BatonTemporalPolicy.worldarena_normalized()
            ),
            query_dim=geometry.query_dim,
            query_layers=geometry.query_layers,
            query_heads=geometry.query_heads,
            query_ffn_dim=geometry.query_ffn_dim,
            query_dropout=geometry.query_dropout,
            query_norm_style=cls.QUERY_NORM_STYLE,
            query_mask_version="full_cross_attention_v1",
            trainable_qwen_layer_indices=tuple(range(24)),
            loss_weights={"mse": 1.0},
            hdf5_manifest_hash=_example_sha256("hdf5-manifest"),
            planner_topology_hash=_example_sha256("planner-topology"),
            optimizer_topology_hash=_example_sha256("optimizer-topology"),
            scheduler_topology_hash=_example_sha256("scheduler-topology"),
            global_step=0,
            distributed_cursor=(
                ("epoch", 0),
                ("consumed_microbatches", 0),
                ("sampler_seed", 0),
            ),
            rng_state_hash=_example_sha256("rng-state"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["added_tokens"] = list(self.added_tokens)
        payload["added_token_ids"] = list(self.added_token_ids)
        payload["camera_names"] = list(self.camera_names)
        payload["target_shape"] = list(self.target_shape)
        payload["temporal_policy"] = self.temporal_policy.to_dict()
        payload["trainable_qwen_layer_indices"] = list(
            self.trainable_qwen_layer_indices
        )
        payload["distributed_cursor"] = dict(self.distributed_cursor)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BatonCheckpointMetadata:
        if isinstance(payload, Mapping) and payload.get("format_version") in (1, 2):
            raise ValueError(
                "Baton checkpoint format versions 1 and 2 are incompatible "
                "with strict Baton version 4"
            )
        if isinstance(payload, Mapping) and payload.get("format_version") == 3:
            return cls._from_legacy_v3(payload, allow_head=False)
        if (
            isinstance(payload, Mapping)
            and payload.get("format_version") == cls.FORMAT_VERSION
            and "distributed_strategy" not in payload
        ):
            payload = {**payload, "distributed_strategy": "ddp"}
        _require_keys(payload, cls._REQUIRED_FIELDS)
        values = {name: payload[name] for name in cls._REQUIRED_FIELDS}
        values["added_tokens"] = tuple(values["added_tokens"])
        values["added_token_ids"] = tuple(values["added_token_ids"])
        values["camera_names"] = tuple(values["camera_names"])
        values["target_shape"] = tuple(values["target_shape"])
        values["temporal_policy"] = BatonTemporalPolicy.from_dict(
            values["temporal_policy"]
        )
        values["trainable_qwen_layer_indices"] = tuple(
            values["trainable_qwen_layer_indices"]
        )
        values["loss_weights"] = dict(values["loss_weights"])
        values["distributed_cursor"] = tuple(values["distributed_cursor"].items())
        return cls(**values)

    @classmethod
    def _from_legacy_v3(
        cls,
        payload: Mapping[str, Any],
        *,
        allow_head: bool,
    ) -> BatonCheckpointMetadata:
        legacy_payload = dict(payload)
        if "distributed_strategy" not in legacy_payload:
            legacy_payload["distributed_strategy"] = "ddp"
        elif legacy_payload["distributed_strategy"] != "ddp":
            raise ValueError("legacy Baton v3 checkpoints must use DDP")
        required = tuple(
            "future_indices" if name == "temporal_policy" else name
            for name in cls._REQUIRED_FIELDS
        )
        _require_keys(legacy_payload, required)
        camera_names = legacy_payload.get("camera_names")
        if camera_names == ["head"] or camera_names == ("head",):
            if not allow_head:
                raise ValueError(
                    "legacy head Baton v3 checkpoint has ambiguous temporal "
                    "metadata; migration required before loading"
                )
            temporal_policy = BatonTemporalPolicy.worldarena_normalized()
        else:
            temporal_policy = BatonTemporalPolicy.libero_fixed()
        if legacy_payload.get("future_indices") not in (
            [0, 3, 5, 8],
            (0, 3, 5, 8),
        ):
            raise ValueError("legacy Baton v3 future_indices must be (0,3,5,8)")
        migrated = legacy_payload
        migrated["format_version"] = cls.FORMAT_VERSION
        migrated["temporal_policy"] = temporal_policy.to_dict()
        del migrated["future_indices"]
        return cls.from_dict(migrated)


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
