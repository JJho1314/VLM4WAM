"""Frozen, local-only inference for the continuous Baton semantic planner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_model
import torch
import torch.nn as nn

from qwen35_baton.config import (
    BatonCheckpointMetadata,
    BatonGeometry,
    BatonTemporalPolicy,
)
from qwen35_baton.data import BatonPlannerCollator
from qwen35_baton.hashing import sha256_artifact, sha256_file, sha256_json
from qwen35_baton.model import BatonPlannerOutput, BatonQwen35Planner
from qwen35_baton.sequence import (
    ADDED_TOKENS,
    BATON_TEMPLATE_KIND,
    LEGACY_TEMPLATE_KIND,
    VERBATIM_INSTRUCTION_KIND,
    find_plan_positions,
    input_template_contract,
    render_instruction,
)


_CHECKPOINT_FILES = (
    "planner.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rank_rng.pt",
    "cursor.json",
    "metadata.json",
)


def _validated_provider_device(
    device: torch.device | str,
    *,
    torch_dtype: torch.dtype,
) -> torch.device:
    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"unsupported provider device: {device!r}") from error
    if target.type == "cpu":
        if target.index is not None:
            raise ValueError(
                "provider requires the canonical CPU device without an index"
            )
        return torch.device("cpu")
    if target.type != "cuda":
        raise ValueError("provider device must support CPU/CUDA autocast semantics")
    if not torch.cuda.is_available():
        raise ValueError("CUDA provider device requested but CUDA is unavailable")
    try:
        device_count = torch.cuda.device_count()
        ordinal = torch.cuda.current_device() if target.index is None else target.index
    except (RuntimeError, AssertionError) as error:
        raise ValueError("CUDA device ordinal could not be resolved") from error
    if (
        type(device_count) is not int
        or type(ordinal) is not int
        or not 0 <= ordinal < device_count
    ):
        raise ValueError(
            f"CUDA device ordinal {ordinal!r} is outside [0, {device_count})"
        )
    if torch_dtype is torch.bfloat16:
        try:
            with torch.cuda.device(ordinal):
                bf16_supported = torch.cuda.is_bf16_supported()
        except (RuntimeError, AssertionError) as error:
            raise ValueError(
                f"CUDA device ordinal {ordinal} could not validate bfloat16 support"
            ) from error
        if not bf16_supported:
            raise ValueError(f"CUDA device ordinal {ordinal} does not support bfloat16")
    return torch.device("cuda", ordinal)


@dataclass(frozen=True)
class BatonSemanticPlan:
    """Detached full-grid planner output consumed by downstream GE-Act."""

    tokens: torch.Tensor
    future_indices: tuple[int, ...] | None
    positions_xy: torch.Tensor
    cross_attention_maps: tuple[torch.Tensor, ...] | None
    camera_names: tuple[str, ...] = ("main", "wrist")
    temporal_policy: BatonTemporalPolicy = field(
        default_factory=BatonTemporalPolicy.libero_fixed
    )
    instruction_sensitivity: None = None
    relevance: None = None

    def __post_init__(self) -> None:
        geometry = BatonGeometry()
        _validate_camera_temporal_contract(
            self.camera_names,
            self.temporal_policy,
        )
        expected_shape = (
            len(self.camera_names),
            self.temporal_policy.target_count,
            geometry.tokens_per_frame,
            geometry.feature_dim,
        )
        if (
            not isinstance(self.tokens, torch.Tensor)
            or self.tokens.ndim != 5
            or tuple(self.tokens.shape[1:]) != expected_shape
            or not self.tokens.dtype.is_floating_point
            or not bool(torch.isfinite(self.tokens).all())
            or self.tokens.requires_grad
        ):
            raise ValueError(
                "tokens must be detached finite floating-point "
                f"[B,{len(self.camera_names)},4,256,1024]"
            )
        batch_size = int(self.tokens.shape[0])
        if batch_size <= 0:
            raise ValueError("tokens must contain at least one sample")
        expected_indices = (
            self.temporal_policy.resolve_future_indices()
            if self.temporal_policy.kind == "fixed_offsets"
            else None
        )
        if self.future_indices != expected_indices:
            raise ValueError(
                "future_indices must match the static temporal policy; normalized "
                "plans must leave them unset"
            )
        expected_positions = build_patch_center_positions(
            batch_size,
            device=self.tokens.device,
            camera_names=self.camera_names,
            temporal_policy=self.temporal_policy,
        )
        if (
            not isinstance(self.positions_xy, torch.Tensor)
            or self.positions_xy.shape != expected_positions.shape
            or self.positions_xy.dtype != torch.float32
            or self.positions_xy.device != self.tokens.device
            or self.positions_xy.requires_grad
            or not torch.equal(self.positions_xy, expected_positions)
        ):
            raise ValueError(
                "positions_xy must be detached normalized 16x16 patch centers "
                f"with shape [B,{len(self.camera_names)},4,256,2]"
            )
        if self.relevance is not None:
            raise ValueError("continuous Baton plans never contain relevance")
        if self.instruction_sensitivity is not None:
            raise ValueError(
                "strict Baton plans do not contain counterfactual sensitivity"
            )
        if self.cross_attention_maps is not None:
            if (
                not isinstance(self.cross_attention_maps, tuple)
                or len(self.cross_attention_maps) != geometry.query_layers
            ):
                raise ValueError(
                    "cross_attention_maps must contain the Baton alignment layer"
                )
            expected = (
                batch_size,
                len(self.camera_names),
                geometry.tokens_per_camera,
                geometry.tokens_per_camera,
            )
            for attention in self.cross_attention_maps:
                if (
                    not isinstance(attention, torch.Tensor)
                    or tuple(attention.shape) != expected
                    or not attention.dtype.is_floating_point
                    or attention.device != self.tokens.device
                    or attention.requires_grad
                    or not bool(torch.isfinite(attention).all())
                ):
                    raise ValueError(
                        "each cross-attention map must be detached finite "
                        f"floating-point [B,{len(self.camera_names)},1024,1024]"
                    )

    def resolve_future_indices(
        self,
        *,
        current_canonical_index: int | None = None,
    ) -> tuple[int, int, int, int]:
        """Resolve policy positions; normalized plans require the sample's current index."""

        return self.temporal_policy.resolve_future_indices(
            current_index=current_canonical_index
        )


def _validate_camera_temporal_contract(
    camera_names: tuple[str, ...],
    temporal_policy: BatonTemporalPolicy,
) -> None:
    if camera_names == ("main", "wrist"):
        expected = BatonTemporalPolicy.libero_fixed()
    elif camera_names == ("head",):
        expected = BatonTemporalPolicy.worldarena_normalized()
    else:
        raise ValueError("camera_names must be either ('main', 'wrist') or ('head',)")
    if (
        not isinstance(temporal_policy, BatonTemporalPolicy)
        or temporal_policy != expected
    ):
        raise ValueError(
            "camera_names and temporal_policy must identify the same dataset contract"
        )


def build_patch_center_positions(
    batch_size: int,
    *,
    device: torch.device | str | None = None,
    camera_names: tuple[str, ...] = ("main", "wrist"),
    temporal_policy: BatonTemporalPolicy | None = None,
) -> torch.Tensor:
    """Return exact row-major normalized centers of the fixed ``16 x 16`` grid."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    geometry = BatonGeometry()
    policy = (
        BatonTemporalPolicy.libero_fixed()
        if temporal_policy is None
        else temporal_policy
    )
    _validate_camera_temporal_contract(camera_names, policy)
    centers = (
        torch.arange(geometry.grid_size, device=device, dtype=torch.float32) + 0.5
    ) / geometry.grid_size
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    xy = torch.stack((x, y), dim=-1).reshape(
        1,
        1,
        1,
        geometry.tokens_per_frame,
        2,
    )
    return xy.expand(
        batch_size,
        len(camera_names),
        policy.target_count,
        -1,
        -1,
    ).contiguous()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _validate_checkpoint_envelope(checkpoint: Path) -> BatonCheckpointMetadata:
    missing = [
        name
        for name in (*_CHECKPOINT_FILES, "manifest.json")
        if not (checkpoint / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"incomplete Baton checkpoint {checkpoint}: missing {missing}"
        )
    metadata = BatonCheckpointMetadata.from_dict(
        _read_json(checkpoint / "metadata.json", label="checkpoint metadata")
    )
    manifest = _read_json(
        checkpoint / "manifest.json",
        label="checkpoint hash manifest",
    )
    hashes = manifest.get("files")
    if (
        manifest.get("format_version") != 2
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(_CHECKPOINT_FILES)
        or any(
            not isinstance(value, str) or len(value) != 64 for value in hashes.values()
        )
    ):
        raise ValueError("checkpoint file hash manifest is invalid")
    for name in _CHECKPOINT_FILES:
        actual = sha256_file(checkpoint / name)
        if hashes[name] != actual:
            raise ValueError(f"checkpoint hash mismatch for {name}")

    # Bind the manifest-valid state payloads back to the complete metadata
    # contract. This is read-only and intentionally precedes component loading.
    from qwen35_baton.checkpoint import (
        BatonTrainingCursor,
        _optimizer_topology,
        _scheduler_topology,
        _validate_cursor_metadata,
        _validate_optimizer_scheduler_lrs,
        _validate_persisted_steps,
        _validate_rank_rng_state,
        _validate_scheduler_state_values,
        _validate_zero2_marker,
    )

    optimizer_state = torch.load(
        checkpoint / "optimizer.pt",
        weights_only=True,
        map_location="cpu",
    )
    scheduler_state = torch.load(
        checkpoint / "scheduler.pt",
        weights_only=True,
        map_location="cpu",
    )
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer state topology is invalid")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("checkpoint scheduler state topology is invalid")
    if metadata.distributed_strategy == "zero2":
        _validate_zero2_marker(
            checkpoint,
            optimizer_state,
            expected_hash=metadata.optimizer_topology_hash,
        )
    elif (
        sha256_json(_optimizer_topology(optimizer_state))
        != metadata.optimizer_topology_hash
    ):
        raise ValueError("checkpoint optimizer topology hash differs from metadata")
    if (
        sha256_json(_scheduler_topology(scheduler_state))
        != metadata.scheduler_topology_hash
    ):
        raise ValueError("checkpoint scheduler topology hash differs from metadata")
    cursor = BatonTrainingCursor.from_dict(
        _read_json(checkpoint / "cursor.json", label="checkpoint cursor")
    )
    rng_payload = torch.load(
        checkpoint / "rank_rng.pt",
        weights_only=True,
        map_location="cpu",
    )
    if (
        not isinstance(rng_payload, Mapping)
        or rng_payload.get("format_version") != 1
        or type(rng_payload.get("world_size")) is not int
        or rng_payload["world_size"] <= 0
        or not isinstance(rng_payload.get("states"), list)
        or len(rng_payload["states"]) != rng_payload["world_size"]
    ):
        raise ValueError("checkpoint RNG world-size/rank coverage is invalid")
    for rank, state in enumerate(rng_payload["states"]):
        _validate_rank_rng_state(state, expected_rank=rank)
    if sha256_file(checkpoint / "rank_rng.pt") != metadata.rng_state_hash:
        raise ValueError("checkpoint RNG hash differs from metadata")
    _validate_cursor_metadata(
        cursor,
        metadata,
        world_size=rng_payload["world_size"],
    )
    _validate_scheduler_state_values(scheduler_state)
    if metadata.distributed_strategy == "zero2":
        if scheduler_state.get("last_epoch") != cursor.global_step:
            raise ValueError("scheduler step differs from checkpoint cursor")
    else:
        _validate_persisted_steps(optimizer_state, scheduler_state, cursor)
        _validate_optimizer_scheduler_lrs(optimizer_state, scheduler_state)
    return metadata


def _validate_local_artifact_contract(
    metadata: BatonCheckpointMetadata,
    *,
    qwen_model_path: Path,
    qwen_tokenizer_path: Path,
    qwen_processor_path: Path,
) -> None:
    for label, path in (
        ("Qwen model", qwen_model_path),
        ("Qwen tokenizer", qwen_tokenizer_path),
        ("Qwen processor", qwen_processor_path),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")

    # Reuse Stage-1's exact CPU-only geometry/token checks so inference cannot
    # silently drift from the artifacts used to construct training rows.
    from qwen35_baton.cli.preflight import (
        _added_token_ids,
        _validate_processor,
        _validate_qwen_config,
    )

    _validate_qwen_config(qwen_model_path)
    _validate_processor(qwen_processor_path)
    token_ids = _added_token_ids(qwen_tokenizer_path)
    checks = (
        (
            "Qwen config",
            sha256_file(qwen_model_path / "config.json"),
            metadata.qwen_config_hash,
        ),
        (
            "tokenizer",
            sha256_artifact(qwen_tokenizer_path),
            metadata.tokenizer_hash,
        ),
        (
            "processor",
            sha256_artifact(qwen_processor_path),
            metadata.processor_hash,
        ),
        (
            "input template",
            sha256_json(input_template_contract(metadata.input_template_kind)),
            metadata.input_template_hash,
        ),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )
    if token_ids != metadata.added_token_ids:
        raise ValueError("tokenizer added-token IDs differ from checkpoint metadata")


def _validate_siglip2_artifact_contract(
    metadata: BatonCheckpointMetadata,
    *,
    siglip2_model_path: Path,
) -> None:
    if not siglip2_model_path.is_dir():
        raise FileNotFoundError(
            f"SigLIP2 model directory does not exist: {siglip2_model_path}"
        )
    config_hash = sha256_file(siglip2_model_path / "config.json")
    if config_hash != metadata.siglip2_config_hash:
        raise ValueError(
            "SigLIP2 config hash mismatch: "
            f"expected {metadata.siglip2_config_hash}, got {config_hash}"
        )
    artifact_hash = sha256_artifact(siglip2_model_path)
    if artifact_hash != metadata.siglip2_artifact_hash:
        raise ValueError(
            "SigLIP2 artifact hash mismatch: "
            f"expected {metadata.siglip2_artifact_hash}, got {artifact_hash}"
        )
    if artifact_hash != metadata.teacher_preprocessing_hash:
        raise ValueError("SigLIP2 preprocessing hash differs from checkpoint metadata")
    from qwen35_baton.cli.preflight import _siglip_geometry

    geometry = _siglip_geometry(siglip2_model_path)
    if geometry != {"image_size": 256, "patch_size": 16, "hidden_size": 1024}:
        raise ValueError("SigLIP2 geometry differs from the 256/16/1024 contract")
    if metadata.teacher_feature_layer != -2 or metadata.teacher_dtype != "bfloat16":
        raise ValueError("SigLIP2 teacher layer/dtype contract must be -2/bfloat16")


def _validate_trusted_planner_topology(
    metadata: BatonCheckpointMetadata,
    *,
    checkpoint: Path,
    expected: str | Path | Mapping[str, Any],
) -> None:
    from qwen35_baton.checkpoint import (
        load_trusted_planner_topology,
        planner_safetensors_topology,
    )

    trusted, trusted_hash = load_trusted_planner_topology(expected)
    if trusted_hash != metadata.planner_topology_hash:
        raise ValueError(
            "trusted planner topology hash differs from checkpoint metadata"
        )
    actual = planner_safetensors_topology(checkpoint / "planner.safetensors")
    if sha256_json(actual) != metadata.planner_topology_hash:
        raise ValueError(
            "planner safetensors topology hash differs from checkpoint metadata"
        )
    if actual != trusted:
        raise ValueError("planner safetensors topology differs from trusted contract")


def _load_local_components(
    *,
    qwen_model_path: Path,
    qwen_tokenizer_path: Path,
    qwen_processor_path: Path,
    added_token_ids: tuple[int, ...],
    torch_dtype: torch.dtype,
) -> tuple[Any, nn.Module]:
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        qwen_tokenizer_path,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        qwen_processor_path,
        local_files_only=True,
    )
    qwen = AutoModelForImageTextToText.from_pretrained(
        qwen_model_path,
        local_files_only=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    try:
        processor.tokenizer = tokenizer
    except (AttributeError, TypeError):
        if getattr(processor, "tokenizer", None) is not tokenizer:
            raise ValueError("persisted Qwen processor did not accept its tokenizer")
    actual_ids = tuple(
        int(tokenizer.convert_tokens_to_ids(token)) for token in ADDED_TOKENS
    )
    if actual_ids != added_token_ids:
        raise ValueError(
            "loaded tokenizer added-token IDs differ from checkpoint metadata"
        )
    return processor, BatonQwen35Planner(qwen, added_token_ids=added_token_ids)


def _validate_model_topology(checkpoint: Path, planner: nn.Module) -> None:
    runtime = planner.state_dict()
    with safe_open(
        checkpoint / "planner.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        saved = {
            name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()
        }
        aliases = dict(handle.metadata() or {})
    for name, shape in saved.items():
        if name not in runtime or tuple(runtime[name].shape) != shape:
            raise ValueError(f"planner state topology mismatch at {name}")
    missing = set(runtime).difference(saved)
    alias_names = {name for name, target in aliases.items() if target in saved}
    if missing.difference(alias_names):
        raise ValueError("planner state topology keys differ from checkpoint")


class FrozenBatonPlanner(nn.Module):
    """Frozen full-grid provider using the Stage-1 processor and text template."""

    def __init__(
        self,
        *,
        planner: nn.Module,
        processor: Any,
        added_token_ids: tuple[int, ...],
        camera_names: tuple[str, ...] = ("main", "wrist"),
        temporal_policy: BatonTemporalPolicy | None = None,
        input_template_kind: str = LEGACY_TEMPLATE_KIND,
        worldarena_sampling_kind: str = "episode_random_v1",
        instruction_rendering_kind: str = VERBATIM_INSTRUCTION_KIND,
    ) -> None:
        super().__init__()
        if not isinstance(planner, nn.Module):
            raise TypeError("planner must be a torch module")
        if not callable(getattr(planner, "forward_rows", None)):
            raise TypeError("planner must expose forward_rows")
        if (
            not isinstance(added_token_ids, tuple)
            or len(added_token_ids) != len(ADDED_TOKENS)
            or any(type(value) is not int or value < 0 for value in added_token_ids)
            or len(set(added_token_ids)) != len(added_token_ids)
        ):
            raise ValueError(
                "added_token_ids must contain seven unique non-negative integers"
            )
        tokenizer = getattr(processor, "tokenizer", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            raise TypeError("processor tokenizer must expose convert_tokens_to_ids")
        actual_ids = tuple(int(convert(token)) for token in ADDED_TOKENS)
        if actual_ids != added_token_ids:
            raise ValueError(
                "processor tokenizer added-token IDs differ from provider contract"
            )
        policy = (
            BatonTemporalPolicy.libero_fixed()
            if temporal_policy is None
            else temporal_policy
        )
        _validate_camera_temporal_contract(camera_names, policy)
        if input_template_kind == LEGACY_TEMPLATE_KIND:
            expected_behavior = ("episode_random_v1", "verbatim_v1")
        elif input_template_kind == BATON_TEMPLATE_KIND and camera_names == ("head",):
            expected_behavior = (
                "all_windows_v1",
                "strip_worldarena_boilerplate_v1",
            )
        elif input_template_kind == BATON_TEMPLATE_KIND:
            expected_behavior = ("episode_random_v1", "verbatim_v1")
        else:
            raise ValueError(f"unsupported input template kind: {input_template_kind!r}")
        if (
            worldarena_sampling_kind,
            instruction_rendering_kind,
        ) != expected_behavior:
            raise ValueError(
                "provider template, sampling, and instruction rendering contracts "
                "are inconsistent"
            )
        self.planner = planner
        self.processor = processor
        self.added_token_ids = added_token_ids
        self.camera_names = camera_names
        self.temporal_policy = policy
        self.input_template_kind = input_template_kind
        self.worldarena_sampling_kind = worldarena_sampling_kind
        self.instruction_rendering_kind = instruction_rendering_kind
        self.geometry = BatonGeometry()
        self._freeze_for_inference()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        qwen_model_path: str | Path,
        qwen_tokenizer_path: str | Path,
        qwen_processor_path: str | Path,
        siglip2_model_path: str | Path,
        expected_planner_topology: str | Path | Mapping[str, Any] | None = None,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        _component_loader: Callable[..., tuple[Any, nn.Module]] | None = None,
    ) -> "FrozenBatonPlanner":
        """Validate all metadata/hashes before loading or mutating model state."""

        if torch_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("torch_dtype must be torch.bfloat16 or torch.float32")
        target_device = _validated_provider_device(
            device,
            torch_dtype=torch_dtype,
        )

        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        model_path = Path(qwen_model_path).expanduser().resolve()
        tokenizer_path = Path(qwen_tokenizer_path).expanduser().resolve()
        processor_path = Path(qwen_processor_path).expanduser().resolve()
        siglip_path = Path(siglip2_model_path).expanduser().resolve()

        # Everything through the artifact checks is read-only and precedes both
        # Transformers construction and safetensors state mutation.
        metadata = _validate_checkpoint_envelope(checkpoint)
        trusted_topology = (
            checkpoint.parent / "planner_topology.json"
            if expected_planner_topology is None
            else expected_planner_topology
        )
        _validate_trusted_planner_topology(
            metadata,
            checkpoint=checkpoint,
            expected=trusted_topology,
        )
        _validate_local_artifact_contract(
            metadata,
            qwen_model_path=model_path,
            qwen_tokenizer_path=tokenizer_path,
            qwen_processor_path=processor_path,
        )
        _validate_siglip2_artifact_contract(
            metadata,
            siglip2_model_path=siglip_path,
        )
        loader = (
            _load_local_components if _component_loader is None else _component_loader
        )
        processor, planner = loader(
            qwen_model_path=model_path,
            qwen_tokenizer_path=tokenizer_path,
            qwen_processor_path=processor_path,
            added_token_ids=metadata.added_token_ids,
            torch_dtype=torch_dtype,
        )
        if not isinstance(planner, nn.Module):
            raise TypeError("component loader must return a torch planner module")
        _validate_model_topology(checkpoint, planner)
        load_model(
            planner,
            str(checkpoint / "planner.safetensors"),
            strict=True,
        )
        planner.to(device=target_device)
        return cls(
            planner=planner,
            processor=processor,
            added_token_ids=metadata.added_token_ids,
            camera_names=metadata.camera_names,
            temporal_policy=metadata.temporal_policy,
            input_template_kind=metadata.input_template_kind,
            worldarena_sampling_kind=metadata.worldarena_sampling_kind,
            instruction_rendering_kind=metadata.instruction_rendering_kind,
        )

    def _freeze_for_inference(self) -> None:
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenBatonPlanner":
        """Remain frozen/eval even when a parent module enters training mode."""

        del mode
        self._freeze_for_inference()
        return self

    def _device(self) -> torch.device:
        for value in (*self.planner.parameters(), *self.planner.buffers()):
            return value.device
        return torch.device("cpu")

    def _validate_inputs(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
    ) -> tuple[int, tuple[str, ...]]:
        if (
            not isinstance(current_images, torch.Tensor)
            or current_images.ndim != 5
            or tuple(current_images.shape[1:3]) != (len(self.camera_names), 3)
            or current_images.shape[-2] <= 0
            or current_images.shape[-1] <= 0
        ):
            raise ValueError(
                f"current_images must have shape [B,{len(self.camera_names)},3,H,W]"
            )
        if current_images.dtype != torch.uint8:
            raise TypeError("current_images must contain uint8 RGB")
        batch_size = int(current_images.shape[0])
        if isinstance(instructions, (str, bytes)) or not isinstance(
            instructions, Sequence
        ):
            raise TypeError("instructions must be an outer sequence of strings")
        if batch_size <= 0 or len(instructions) != batch_size:
            raise ValueError("images and instructions batch sizes must match")
        positive = tuple(instructions)
        if any(type(value) is not str or not value.strip() for value in positive):
            raise ValueError("instructions must contain nonblank strings")
        return batch_size, positive

    def _autocast_context(self) -> Any:
        backbone = getattr(self.planner, "backbone", None)
        if not isinstance(backbone, nn.Module):
            return nullcontext()
        dtype = next(
            (
                parameter.dtype
                for parameter in backbone.parameters()
                if parameter.dtype.is_floating_point
            ),
            None,
        )
        device_type = self._device().type
        if dtype == torch.bfloat16 and device_type in {"cpu", "cuda"}:
            return torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=True,
            )
        return nullcontext()

    def _build_rows(
        self,
        current_images: torch.Tensor,
        instructions: tuple[str, ...],
        source_indices: tuple[
            tuple[int, int, int, int, int] | None,
            ...,
        ],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        collator = BatonPlannerCollator(
            self.processor,
            camera_names=self.camera_names,
            plan_pad_token_id=self.added_token_ids[5],
            input_template_kind=self.input_template_kind,
            instruction_rendering_kind=self.instruction_rendering_kind,
        )
        sequences: list[torch.Tensor] = []
        processed_rows: list[Mapping[str, torch.Tensor]] = []
        cpu_images = current_images.detach().to(device="cpu")
        for sample_index, instruction in enumerate(instructions):
            rendered_instruction = render_instruction(
                instruction,
                self.instruction_rendering_kind,
            )
            for camera_index, _ in enumerate(self.camera_names):
                sequence, processed = collator._process_row(
                    cpu_images[sample_index, camera_index],
                    rendered_instruction,
                    source_indices[sample_index],
                )
                sequences.append(sequence)
                processed_rows.append(processed)
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if type(pad_token_id) is not int:
            raise ValueError("processor tokenizer pad_token_id must be an integer")
        maximum = max(sequence.numel() for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), maximum),
            pad_token_id,
            dtype=sequences[0].dtype,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, sequence in enumerate(sequences):
            input_ids[row_index, : sequence.numel()] = sequence
            attention_mask[row_index, : sequence.numel()] = 1
        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **collator._merge_processor_values(processed_rows, sequences),
        }
        positions = find_plan_positions(input_ids, self.added_token_ids[5])
        device = self._device()
        return (
            {name: value.to(device=device) for name, value in qwen_inputs.items()},
            positions.to(device=device),
        )

    def _validate_output(
        self,
        output: Any,
        *,
        rows: int,
        return_attention: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        flat = getattr(output, "flat", None)
        expected = (
            rows,
            self.temporal_policy.target_count,
            self.geometry.tokens_per_frame,
            self.geometry.feature_dim,
        )
        if (
            not isinstance(output, BatonPlannerOutput)
            or not isinstance(flat, torch.Tensor)
            or tuple(flat.shape) != expected
            or not flat.dtype.is_floating_point
            or flat.device != self._device()
            or not bool(torch.isfinite(flat).all())
        ):
            raise RuntimeError(
                "planner output must be finite floating-point "
                "[rows,4,256,1024] on the planner device"
            )
        maps = output.cross_attention_maps
        if not return_attention:
            if maps is not None:
                raise RuntimeError(
                    "planner allocated cross-attention maps when not requested"
                )
            return flat, None
        if not isinstance(maps, tuple) or len(maps) != self.geometry.query_layers:
            raise RuntimeError(
                "planner must return the Baton alignment cross-attention map"
            )
        for attention in maps:
            if (
                not isinstance(attention, torch.Tensor)
                or attention.shape
                != (
                    rows,
                    self.geometry.tokens_per_camera,
                    self.geometry.tokens_per_camera,
                )
                or not attention.dtype.is_floating_point
                or attention.device != flat.device
                or not bool(torch.isfinite(attention).all())
            ):
                raise RuntimeError(
                    "Query Tower cross-attention maps must be finite "
                    "[rows,1024,1024] on the planner device"
                )
        return flat, maps

    @torch.no_grad()
    def predict(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
        *,
        return_attention: bool = False,
        current_canonical_indices: Sequence[int] | None = None,
    ) -> BatonSemanticPlan:
        """Predict full grids, optionally tracing Baton cross-attention."""

        if type(return_attention) is not bool:
            raise TypeError("return_attention must be a boolean")
        batch_size, positive = self._validate_inputs(
            current_images,
            instructions,
        )
        if self.input_template_kind == LEGACY_TEMPLATE_KIND:
            if current_canonical_indices is not None:
                raise ValueError(
                    "legacy_user_plan_v1 does not accept current_canonical_indices"
                )
            source_indices: tuple[
                tuple[int, int, int, int, int] | None,
                ...,
            ] = (None,) * batch_size
        elif self.camera_names == ("main", "wrist"):
            if current_canonical_indices is not None:
                raise ValueError(
                    "fixed LIBERO Baton inputs do not accept current_canonical_indices"
                )
            source_indices = ((3, 4, 7, 9, 12),) * batch_size
        else:
            if (
                isinstance(current_canonical_indices, (str, bytes))
                or not isinstance(current_canonical_indices, Sequence)
                or len(current_canonical_indices) != batch_size
                or any(type(value) is not int for value in current_canonical_indices)
            ):
                raise ValueError(
                    "current_canonical_indices must contain one integer per image"
                )
            source_indices = tuple(
                (
                    current,
                    *self.temporal_policy.resolve_future_indices(
                        current_index=current
                    ),
                )
                for current in current_canonical_indices
            )
        self._freeze_for_inference()
        qwen_inputs, plan_positions = self._build_rows(
            current_images,
            positive,
            source_indices,
        )
        camera_count = len(self.camera_names)
        rows = batch_size * camera_count
        with self._autocast_context():
            output = self.planner.forward_rows(
                qwen_inputs,
                plan_positions,
                return_attention_maps=return_attention,
            )
        flat, raw_maps = self._validate_output(
            output,
            rows=rows,
            return_attention=return_attention,
        )
        tokens = flat.reshape(
            batch_size,
            camera_count,
            self.temporal_policy.target_count,
            self.geometry.tokens_per_frame,
            self.geometry.feature_dim,
        )
        attention_maps = None
        if raw_maps is not None:
            attention_maps = tuple(
                attention.reshape(
                    batch_size,
                    camera_count,
                    self.geometry.tokens_per_camera,
                    self.geometry.tokens_per_camera,
                )
                for attention in raw_maps
            )
        return BatonSemanticPlan(
            tokens=tokens.detach(),
            future_indices=(
                self.temporal_policy.resolve_future_indices()
                if self.temporal_policy.kind == "fixed_offsets"
                else None
            ),
            positions_xy=build_patch_center_positions(
                batch_size,
                device=tokens.device,
                camera_names=self.camera_names,
                temporal_policy=self.temporal_policy,
            ),
            cross_attention_maps=(
                None
                if attention_maps is None
                else tuple(value.detach() for value in attention_maps)
            ),
            camera_names=self.camera_names,
            temporal_policy=self.temporal_policy,
            instruction_sensitivity=None,
        )
