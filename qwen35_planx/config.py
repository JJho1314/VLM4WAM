"""Immutable production contracts for the grounded Qwen3.5 Plan-X pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Mapping

from qwen35_planx.hashing import sha256_json


CAMERA_NAMES = ("main", "wrist")
CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)

_IMAGE_SIZE = 384
_GRID_SIZE = 27
_NUM_KEYFRAMES = 4
_VISUAL_VOCAB_SIZE = 65_536
_TA_CODE_DIM = 1536
_QWEN_HIDDEN_DIM = 2048
_TEXT_ALIGN_DIM = 1152
_FUTURE_FRAME_OFFSETS = (1, 4, 6, 9)
_GE_ACT_FUTURE_INDICES = (0, 3, 5, 8)
_GROUNDING_ROLES = ("source", "target", "action")
_SUPERSEDED_QWEN_ANCHOR_FIELDS = (
    "anchor_token_ids",
    "anchor_embedding_hash",
)


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _require_keys(payload: Mapping[str, Any], required: tuple[str, ...]) -> None:
    missing = sorted(set(required).difference(payload))
    if missing:
        raise ValueError(f"missing required metadata fields: {', '.join(missing)}")


def _reject_superseded_fields(payload: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    present = sorted(set(fields).intersection(payload))
    if present:
        raise ValueError(
            "superseded Qwen anchor fields are not permitted: " + ", ".join(present)
        )


def _require_hashes(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        if not getattr(instance, name):
            raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class PlanGeometry:
    """Fixed 384px, four-frame, dual-camera released-checkpoint geometry."""

    image_size: int = _IMAGE_SIZE
    num_keyframes: int = _NUM_KEYFRAMES
    grid_size: int = _GRID_SIZE
    future_frame_offsets: tuple[int, ...] = _FUTURE_FRAME_OFFSETS
    ge_act_future_indices: tuple[int, ...] = _GE_ACT_FUTURE_INDICES
    visual_vocab_size: int = _VISUAL_VOCAB_SIZE
    ta_code_dim: int = _TA_CODE_DIM
    qwen_hidden_dim: int = _QWEN_HIDDEN_DIM
    text_align_dim: int = _TEXT_ALIGN_DIM

    def __post_init__(self) -> None:
        _require_equal("image_size", self.image_size, _IMAGE_SIZE)
        _require_equal("num_keyframes", self.num_keyframes, _NUM_KEYFRAMES)
        _require_equal("grid_size", self.grid_size, _GRID_SIZE)
        _require_equal(
            "future_frame_offsets",
            tuple(self.future_frame_offsets),
            _FUTURE_FRAME_OFFSETS,
        )
        _require_equal(
            "ge_act_future_indices",
            tuple(self.ge_act_future_indices),
            _GE_ACT_FUTURE_INDICES,
        )
        _require_equal("visual_vocab_size", self.visual_vocab_size, _VISUAL_VOCAB_SIZE)
        _require_equal("ta_code_dim", self.ta_code_dim, _TA_CODE_DIM)
        _require_equal("qwen_hidden_dim", self.qwen_hidden_dim, _QWEN_HIDDEN_DIM)
        _require_equal("text_align_dim", self.text_align_dim, _TEXT_ALIGN_DIM)

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def tokens_per_camera(self) -> int:
        return self.num_keyframes * self.tokens_per_frame

    @property
    def tokens_per_sample(self) -> int:
        return len(CAMERA_NAMES) * self.tokens_per_camera

    @property
    def response_tokens_per_camera(self) -> int:
        return 2 + (2 * self.num_keyframes) + self.tokens_per_camera


@dataclass(frozen=True)
class ReleasedTATokMetadata:
    """Compatibility metadata for the released SigLIP2 TA-Tok checkpoint."""

    FORMAT_VERSION: ClassVar[int] = 1

    format_version: int
    tokenizer_type: str
    teacher: str
    image_size: int
    grid_size: int
    bottleneck_token_num: int
    codebook_size: int
    codebook_dim: int
    selected_layer: int
    pool_scale: int
    checkpoint_hash: str

    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "format_version",
        "tokenizer_type",
        "teacher",
        "image_size",
        "grid_size",
        "bottleneck_token_num",
        "codebook_size",
        "codebook_dim",
        "selected_layer",
        "pool_scale",
        "checkpoint_hash",
    )

    def __post_init__(self) -> None:
        _require_equal("format_version", self.format_version, self.FORMAT_VERSION)
        _require_equal("tokenizer_type", self.tokenizer_type, "released_ta_tok")
        _require_equal(
            "teacher", self.teacher, "google/siglip2-so400m-patch14-384"
        )
        _require_equal("image_size", self.image_size, _IMAGE_SIZE)
        _require_equal("grid_size", self.grid_size, _GRID_SIZE)
        _require_equal("bottleneck_token_num", self.bottleneck_token_num, 729)
        _require_equal("codebook_size", self.codebook_size, _VISUAL_VOCAB_SIZE)
        _require_equal("codebook_dim", self.codebook_dim, _TA_CODE_DIM)
        _require_equal("selected_layer", self.selected_layer, -2)
        _require_equal("pool_scale", self.pool_scale, 1)
        _require_hashes(self, ("checkpoint_hash",))

    @classmethod
    def example(cls) -> ReleasedTATokMetadata:
        geometry = PlanGeometry()
        return cls(
            format_version=cls.FORMAT_VERSION,
            tokenizer_type="released_ta_tok",
            teacher="google/siglip2-so400m-patch14-384",
            image_size=geometry.image_size,
            grid_size=geometry.grid_size,
            bottleneck_token_num=geometry.tokens_per_frame,
            codebook_size=geometry.visual_vocab_size,
            codebook_dim=geometry.ta_code_dim,
            selected_layer=-2,
            pool_scale=1,
            checkpoint_hash="example-released-ta-tok-sha256",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReleasedTATokMetadata:
        _require_keys(payload, cls._REQUIRED_FIELDS)
        _reject_superseded_fields(payload, _SUPERSEDED_QWEN_ANCHOR_FIELDS)
        return cls(**{name: payload[name] for name in cls._REQUIRED_FIELDS})


@dataclass(frozen=True)
class HindsightCacheMetadata:
    """Hashes needed to reproduce a video-hindsight cache exactly."""

    FORMAT_VERSION: ClassVar[int] = 1

    format_version: int
    hdf5_manifest_hash: str
    window_manifest_hash: str
    instruction_parser_hash: str
    ta_tok_hash: str
    siglip2_hash: str
    dinov3_hash: str
    preprocessing_hash: str

    _HASH_FIELDS: ClassVar[tuple[str, ...]] = (
        "hdf5_manifest_hash",
        "window_manifest_hash",
        "instruction_parser_hash",
        "ta_tok_hash",
        "siglip2_hash",
        "dinov3_hash",
        "preprocessing_hash",
    )
    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = ("format_version",) + _HASH_FIELDS

    def __post_init__(self) -> None:
        _require_equal("format_version", self.format_version, self.FORMAT_VERSION)
        _require_hashes(self, self._HASH_FIELDS)

    @property
    def cache_hash(self) -> str:
        """Content hash used by planner checkpoints to identify this cache."""

        return sha256_json(self.to_dict())

    @classmethod
    def example(cls) -> HindsightCacheMetadata:
        return cls(
            format_version=cls.FORMAT_VERSION,
            hdf5_manifest_hash="example-hdf5-manifest-sha256",
            window_manifest_hash="example-window-manifest-sha256",
            instruction_parser_hash="example-instruction-parser-sha256",
            ta_tok_hash="example-ta-tok-sha256",
            siglip2_hash="example-siglip2-sha256",
            dinov3_hash="example-dinov3-sha256",
            preprocessing_hash="example-preprocessing-sha256",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HindsightCacheMetadata:
        _require_keys(payload, cls._REQUIRED_FIELDS)
        return cls(**{name: payload[name] for name in cls._REQUIRED_FIELDS})


@dataclass(frozen=True)
class GroundedPlannerMetadata:
    """Immutable checkpoint contract for the grounded Qwen3.5 planner."""

    FORMAT_VERSION: ClassVar[int] = 1
    BACKEND: ClassVar[str] = "qwen35_planx_grounded"
    HIDDEN_ALIGNMENT: ClassVar[str] = "pre_predicts_code_post_conditions_semantics"

    format_version: int
    planner_backend: str
    base_model: str
    model_type: str
    camera_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
    image_size: int
    num_keyframes: int
    grid_size: int
    visual_vocab_size: int
    future_frame_offsets: tuple[int, ...]
    ge_act_future_indices: tuple[int, ...]
    tokens_per_frame: int
    response_tokens_per_camera: int
    visual_token_start_id: int
    visual_token_end_id: int
    structure_token_ids: tuple[tuple[str, int], ...]
    loss_weights: tuple[tuple[str, float], ...]
    phrase_roles: tuple[str, ...]
    hidden_alignment: str
    qwen_hidden_dim: int
    text_align_dim: int
    tokenizer_hash: str
    ta_tok_hash: str
    base_model_hash: str
    hindsight_cache_hash: str

    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "format_version",
        "planner_backend",
        "base_model",
        "model_type",
        "camera_names",
        "camera_keys",
        "image_size",
        "num_keyframes",
        "grid_size",
        "visual_vocab_size",
        "future_frame_offsets",
        "ge_act_future_indices",
        "tokens_per_frame",
        "response_tokens_per_camera",
        "visual_token_start_id",
        "visual_token_end_id",
        "structure_token_ids",
        "loss_weights",
        "phrase_roles",
        "hidden_alignment",
        "qwen_hidden_dim",
        "text_align_dim",
        "tokenizer_hash",
        "ta_tok_hash",
        "base_model_hash",
        "hindsight_cache_hash",
    )

    def __post_init__(self) -> None:
        _require_equal("format_version", self.format_version, self.FORMAT_VERSION)
        _require_equal("planner_backend", self.planner_backend, self.BACKEND)
        _require_equal("model_type", self.model_type, "qwen3_5")
        _require_equal("camera_names", tuple(self.camera_names), CAMERA_NAMES)
        _require_equal("camera_keys", tuple(self.camera_keys), CAMERA_KEYS)
        geometry = PlanGeometry(
            image_size=self.image_size,
            num_keyframes=self.num_keyframes,
            grid_size=self.grid_size,
            future_frame_offsets=tuple(self.future_frame_offsets),
            ge_act_future_indices=tuple(self.ge_act_future_indices),
            visual_vocab_size=self.visual_vocab_size,
            qwen_hidden_dim=self.qwen_hidden_dim,
            text_align_dim=self.text_align_dim,
        )
        _require_equal("tokens_per_frame", self.tokens_per_frame, geometry.tokens_per_frame)
        _require_equal(
            "response_tokens_per_camera",
            self.response_tokens_per_camera,
            geometry.response_tokens_per_camera,
        )
        _require_equal("hidden_alignment", self.hidden_alignment, self.HIDDEN_ALIGNMENT)
        _require_equal("phrase_roles", tuple(self.phrase_roles), _GROUNDING_ROLES)
        if self.visual_token_start_id < 0:
            raise ValueError("visual_token_start_id must be non-negative")
        if self.visual_token_end_id - self.visual_token_start_id != self.visual_vocab_size:
            raise ValueError(
                "visual token ID range must contain exactly visual_vocab_size IDs"
            )
        if not self.structure_token_ids:
            raise ValueError("structure_token_ids must not be empty")
        if len({name for name, _ in self.structure_token_ids}) != len(
            self.structure_token_ids
        ):
            raise ValueError("structure_token_ids names must be unique")
        if not self.loss_weights:
            raise ValueError("loss_weights must not be empty")
        if len({name for name, _ in self.loss_weights}) != len(self.loss_weights):
            raise ValueError("loss_weights names must be unique")
        if any(weight < 0 for _, weight in self.loss_weights):
            raise ValueError("loss_weights must be non-negative")
        _require_hashes(
            self,
            ("tokenizer_hash", "ta_tok_hash", "base_model_hash", "hindsight_cache_hash"),
        )

    @classmethod
    def example(cls) -> GroundedPlannerMetadata:
        geometry = PlanGeometry()
        visual_token_start_id = 250_000
        return cls(
            format_version=cls.FORMAT_VERSION,
            planner_backend=cls.BACKEND,
            base_model="Qwen/Qwen3.5-VL-4B-Instruct",
            model_type="qwen3_5",
            camera_names=CAMERA_NAMES,
            camera_keys=CAMERA_KEYS,
            image_size=geometry.image_size,
            num_keyframes=geometry.num_keyframes,
            grid_size=geometry.grid_size,
            visual_vocab_size=geometry.visual_vocab_size,
            future_frame_offsets=geometry.future_frame_offsets,
            ge_act_future_indices=geometry.ge_act_future_indices,
            tokens_per_frame=geometry.tokens_per_frame,
            response_tokens_per_camera=geometry.response_tokens_per_camera,
            visual_token_start_id=visual_token_start_id,
            visual_token_end_id=visual_token_start_id + geometry.visual_vocab_size,
            structure_token_ids=(
                ("plan_start", 249_994),
                ("plan_end", 249_995),
                ("frame_start", 249_996),
                ("frame_end", 249_997),
                ("camera_main", 249_998),
                ("camera_wrist", 249_999),
            ),
            loss_weights=(
                ("code", 1.0),
                ("grounding", 1.0),
                ("hindsight", 1.0),
            ),
            phrase_roles=_GROUNDING_ROLES,
            hidden_alignment=cls.HIDDEN_ALIGNMENT,
            qwen_hidden_dim=geometry.qwen_hidden_dim,
            text_align_dim=geometry.text_align_dim,
            tokenizer_hash="example-tokenizer-sha256",
            ta_tok_hash="example-ta-tok-sha256",
            base_model_hash="example-base-model-sha256",
            hindsight_cache_hash=HindsightCacheMetadata.example().cache_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GroundedPlannerMetadata:
        _require_keys(payload, cls._REQUIRED_FIELDS)
        values = {name: payload[name] for name in cls._REQUIRED_FIELDS}
        for name in (
            "camera_names",
            "camera_keys",
            "future_frame_offsets",
            "ge_act_future_indices",
            "phrase_roles",
        ):
            values[name] = tuple(values[name])
        values["structure_token_ids"] = tuple(
            (str(name), int(token_id)) for name, token_id in values["structure_token_ids"]
        )
        values["loss_weights"] = tuple(
            (str(name), float(weight)) for name, weight in values["loss_weights"]
        )
        return cls(**values)

    def validate_runtime(
        self,
        *,
        tokenizer_hash: str | None = None,
        ta_tok_hash: str | None = None,
        base_model_hash: str | None = None,
        hindsight_cache_hash: str | None = None,
    ) -> None:
        """Reject runtime artifacts that differ from checkpoint provenance."""

        for name, actual in {
            "tokenizer_hash": tokenizer_hash,
            "ta_tok_hash": ta_tok_hash,
            "base_model_hash": base_model_hash,
            "hindsight_cache_hash": hindsight_cache_hash,
        }.items():
            if actual is not None:
                _require_equal(name, actual, getattr(self, name))


# Existing callers imported this generic name directly.  It now denotes the
# released grounded contract rather than retaining the superseded 256px schema.
PlannerMetadata = GroundedPlannerMetadata
