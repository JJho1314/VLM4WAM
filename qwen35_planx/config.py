"""Immutable production contracts for the Qwen3.5 Plan-X pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Mapping


CAMERA_NAMES = ("main", "wrist")
CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)

_FUTURE_FRAME_OFFSETS = (1, 4, 6, 9)
_GE_ACT_FUTURE_INDICES = (0, 3, 5, 8)
_NUM_KEYFRAMES = 4
_GRID_SIZE = 16
_VISUAL_VOCAB_SIZE = 65_536


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _require_keys(payload: Mapping[str, Any], required: tuple[str, ...]) -> None:
    missing = sorted(set(required).difference(payload))
    if missing:
        raise ValueError(f"missing required metadata fields: {', '.join(missing)}")


@dataclass(frozen=True)
class PlanGeometry:
    """Fixed token and frame geometry shared by every production artifact."""

    num_keyframes: int = _NUM_KEYFRAMES
    grid_size: int = _GRID_SIZE
    future_frame_offsets: tuple[int, ...] = _FUTURE_FRAME_OFFSETS
    ge_act_future_indices: tuple[int, ...] = _GE_ACT_FUTURE_INDICES
    visual_vocab_size: int = _VISUAL_VOCAB_SIZE

    def __post_init__(self) -> None:
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
        _require_equal(
            "visual_vocab_size",
            self.visual_vocab_size,
            _VISUAL_VOCAB_SIZE,
        )

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
        plan_delimiters = 2
        frame_delimiters = 2 * self.num_keyframes
        return plan_delimiters + frame_delimiters + self.tokens_per_camera


@dataclass(frozen=True)
class TATokMetadata:
    """Compatibility metadata saved with a trained domain TA-Tok."""

    FORMAT_VERSION: ClassVar[int] = 1

    format_version: int
    tokenizer_type: str
    siglip_model: str
    qwen_model: str
    selected_layer: int
    image_size: int
    grid_size: int
    feature_dim: int
    qwen_hidden_dim: int
    visual_vocab_size: int
    camera_names: tuple[str, ...]
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    anchor_token_ids: tuple[int, ...]
    anchor_embedding_hash: str
    state_hash: str

    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "format_version",
        "tokenizer_type",
        "siglip_model",
        "qwen_model",
        "selected_layer",
        "image_size",
        "grid_size",
        "feature_dim",
        "qwen_hidden_dim",
        "visual_vocab_size",
        "camera_names",
        "image_mean",
        "image_std",
        "anchor_token_ids",
        "anchor_embedding_hash",
        "state_hash",
    )

    def __post_init__(self) -> None:
        _require_equal("format_version", self.format_version, self.FORMAT_VERSION)
        _require_equal("tokenizer_type", self.tokenizer_type, "ta_tok")
        _require_equal("selected_layer", self.selected_layer, -2)
        _require_equal("image_size", self.image_size, 256)
        _require_equal("grid_size", self.grid_size, _GRID_SIZE)
        _require_equal(
            "visual_vocab_size",
            self.visual_vocab_size,
            _VISUAL_VOCAB_SIZE,
        )
        _require_equal("camera_names", tuple(self.camera_names), CAMERA_NAMES)
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must contain three channels")
        if self.feature_dim <= 0 or self.qwen_hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if len(self.anchor_token_ids) != self.visual_vocab_size:
            raise ValueError(
                "anchor_token_ids must contain exactly visual_vocab_size entries"
            )
        if len(set(self.anchor_token_ids)) != len(self.anchor_token_ids):
            raise ValueError("anchor_token_ids must be unique")
        if any(token_id < 0 for token_id in self.anchor_token_ids):
            raise ValueError("anchor_token_ids must be non-negative")
        if not self.anchor_embedding_hash:
            raise ValueError("anchor_embedding_hash must not be empty")
        if not self.state_hash:
            raise ValueError("state_hash must not be empty")

    @classmethod
    def example(cls) -> TATokMetadata:
        return cls(
            format_version=cls.FORMAT_VERSION,
            tokenizer_type="ta_tok",
            siglip_model="google/siglip2-large-patch16-256",
            qwen_model="Qwen/Qwen3.5-VL-4B-Instruct",
            selected_layer=-2,
            image_size=256,
            grid_size=_GRID_SIZE,
            feature_dim=1024,
            qwen_hidden_dim=2560,
            visual_vocab_size=_VISUAL_VOCAB_SIZE,
            camera_names=CAMERA_NAMES,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            anchor_token_ids=tuple(range(_VISUAL_VOCAB_SIZE)),
            anchor_embedding_hash="example-anchor-sha256",
            state_hash="example-state-sha256",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TATokMetadata:
        _require_keys(payload, cls._REQUIRED_FIELDS)
        values = {name: payload[name] for name in cls._REQUIRED_FIELDS}
        for name in ("camera_names", "image_mean", "image_std", "anchor_token_ids"):
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class PlannerMetadata:
    """Compatibility metadata saved with an expanded Qwen3.5 planner."""

    FORMAT_VERSION: ClassVar[int] = 1
    BACKEND: ClassVar[str] = "qwen35_planx"
    HIDDEN_ALIGNMENT: ClassVar[str] = "causal_pre_output_head_predicts_code"

    format_version: int
    planner_backend: str
    base_model: str
    model_type: str
    camera_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
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
    hidden_alignment: str
    qwen_hidden_dim: int
    tokenizer_hash: str
    ta_tok_hash: str
    base_model_hash: str

    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "format_version",
        "planner_backend",
        "base_model",
        "model_type",
        "camera_names",
        "camera_keys",
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
        "hidden_alignment",
        "qwen_hidden_dim",
        "tokenizer_hash",
        "ta_tok_hash",
        "base_model_hash",
    )

    def __post_init__(self) -> None:
        _require_equal("format_version", self.format_version, self.FORMAT_VERSION)
        _require_equal("planner_backend", self.planner_backend, self.BACKEND)
        _require_equal("model_type", self.model_type, "qwen3_5")
        _require_equal("camera_names", tuple(self.camera_names), CAMERA_NAMES)
        _require_equal("camera_keys", tuple(self.camera_keys), CAMERA_KEYS)

        geometry = PlanGeometry(
            num_keyframes=self.num_keyframes,
            grid_size=self.grid_size,
            future_frame_offsets=tuple(self.future_frame_offsets),
            ge_act_future_indices=tuple(self.ge_act_future_indices),
            visual_vocab_size=self.visual_vocab_size,
        )
        _require_equal(
            "tokens_per_frame", self.tokens_per_frame, geometry.tokens_per_frame
        )
        _require_equal(
            "response_tokens_per_camera",
            self.response_tokens_per_camera,
            geometry.response_tokens_per_camera,
        )
        _require_equal("hidden_alignment", self.hidden_alignment, self.HIDDEN_ALIGNMENT)
        if self.visual_token_end_id - self.visual_token_start_id != self.visual_vocab_size:
            raise ValueError(
                "visual token ID range must contain exactly visual_vocab_size IDs"
            )
        if self.qwen_hidden_dim <= 0:
            raise ValueError("qwen_hidden_dim must be positive")
        if not self.structure_token_ids:
            raise ValueError("structure_token_ids must not be empty")
        if len({name for name, _ in self.structure_token_ids}) != len(
            self.structure_token_ids
        ):
            raise ValueError("structure_token_ids names must be unique")
        for hash_name in ("tokenizer_hash", "ta_tok_hash", "base_model_hash"):
            if not getattr(self, hash_name):
                raise ValueError(f"{hash_name} must not be empty")

    @classmethod
    def example(cls) -> PlannerMetadata:
        geometry = PlanGeometry()
        visual_token_start_id = 250_000
        return cls(
            format_version=cls.FORMAT_VERSION,
            planner_backend=cls.BACKEND,
            base_model="Qwen/Qwen3.5-VL-4B-Instruct",
            model_type="qwen3_5",
            camera_names=CAMERA_NAMES,
            camera_keys=CAMERA_KEYS,
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
            hidden_alignment=cls.HIDDEN_ALIGNMENT,
            qwen_hidden_dim=2560,
            tokenizer_hash="example-tokenizer-sha256",
            ta_tok_hash="example-ta-tok-sha256",
            base_model_hash="example-base-model-sha256",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerMetadata:
        _require_keys(payload, cls._REQUIRED_FIELDS)
        values = {name: payload[name] for name in cls._REQUIRED_FIELDS}
        for name in (
            "camera_names",
            "camera_keys",
            "future_frame_offsets",
            "ge_act_future_indices",
        ):
            values[name] = tuple(values[name])
        values["structure_token_ids"] = tuple(
            (str(name), int(token_id))
            for name, token_id in values["structure_token_ids"]
        )
        return cls(**values)

    def validate_runtime(
        self,
        *,
        tokenizer_hash: str | None = None,
        ta_tok_hash: str | None = None,
        base_model_hash: str | None = None,
    ) -> None:
        """Reject runtime artifacts that were not used to build this planner."""

        provided = {
            "tokenizer_hash": tokenizer_hash,
            "ta_tok_hash": ta_tok_hash,
            "base_model_hash": base_model_hash,
        }
        for name, actual in provided.items():
            if actual is not None:
                _require_equal(name, actual, getattr(self, name))
