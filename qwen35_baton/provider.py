"""Frozen, local-only inference for the continuous Baton semantic planner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen35_baton.config import BatonCheckpointMetadata, BatonGeometry
from qwen35_baton.data import BatonPlannerCollator
from qwen35_baton.hashing import sha256_artifact, sha256_file, sha256_json
from qwen35_baton.model import BatonPlannerOutput, BatonQwen35Planner
from qwen35_baton.sequence import ADDED_TOKENS, build_plan_text, find_plan_positions


_CHECKPOINT_FILES = (
    "planner.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rank_rng.pt",
    "cursor.json",
    "metadata.json",
)


@dataclass(frozen=True)
class BatonSemanticPlan:
    """Detached full-grid planner output consumed by downstream GE-Act."""

    tokens: torch.Tensor
    future_indices: tuple[int, ...]
    positions_xy: torch.Tensor
    cross_attention_maps: tuple[torch.Tensor, ...] | None
    instruction_sensitivity: torch.Tensor | None
    relevance: None = None

    def __post_init__(self) -> None:
        geometry = BatonGeometry()
        if (
            not isinstance(self.tokens, torch.Tensor)
            or self.tokens.ndim != 5
            or tuple(self.tokens.shape[1:]) != geometry.output_shape(1)[1:]
            or not self.tokens.dtype.is_floating_point
            or not bool(torch.isfinite(self.tokens).all())
            or self.tokens.requires_grad
        ):
            raise ValueError(
                "tokens must be detached finite floating-point "
                "[B,2,4,256,1024]"
            )
        batch_size = int(self.tokens.shape[0])
        if batch_size <= 0:
            raise ValueError("tokens must contain at least one sample")
        if self.future_indices != geometry.future_indices:
            raise ValueError("future_indices must be exactly (0,3,5,8)")
        expected_positions = build_patch_center_positions(
            batch_size,
            device=self.tokens.device,
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
                "with shape [B,2,4,256,2]"
            )
        if self.relevance is not None:
            raise ValueError("continuous Baton plans never contain relevance")
        if self.instruction_sensitivity is not None:
            sensitivity = self.instruction_sensitivity
            if (
                not isinstance(sensitivity, torch.Tensor)
                or sensitivity.shape != (batch_size, 2, 4, 256)
                or not sensitivity.dtype.is_floating_point
                or sensitivity.device != self.tokens.device
                or sensitivity.requires_grad
                or not bool(torch.isfinite(sensitivity).all())
            ):
                raise ValueError(
                    "instruction_sensitivity must be detached finite "
                    "floating-point [B,2,4,256]"
                )
        if self.cross_attention_maps is not None:
            if (
                not isinstance(self.cross_attention_maps, tuple)
                or len(self.cross_attention_maps) != geometry.query_layers
            ):
                raise ValueError(
                    "cross_attention_maps must contain all four Query Tower layers"
                )
            expected = (
                batch_size,
                len(geometry.camera_names),
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
                        "floating-point [B,2,1024,1024]"
                    )


def build_patch_center_positions(
    batch_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return exact row-major normalized centers of the fixed ``16 x 16`` grid."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    geometry = BatonGeometry()
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
        len(geometry.camera_names),
        len(geometry.future_indices),
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
        manifest.get("format_version") != 1
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(_CHECKPOINT_FILES)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in hashes.values()
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
    if sha256_json(_optimizer_topology(optimizer_state)) != metadata.optimizer_topology_hash:
        raise ValueError("checkpoint optimizer topology hash differs from metadata")
    if sha256_json(_scheduler_topology(scheduler_state)) != metadata.scheduler_topology_hash:
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
    _validate_persisted_steps(optimizer_state, scheduler_state, cursor)
    _validate_scheduler_state_values(scheduler_state)
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
            sha256_json(build_plan_text("{instruction}")),
            metadata.input_template_hash,
        ),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )
    if token_ids != metadata.added_token_ids:
        raise ValueError(
            "tokenizer added-token IDs differ from checkpoint metadata"
        )


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
        raise ValueError(
            "SigLIP2 preprocessing hash differs from checkpoint metadata"
        )
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
        planner_safetensors_topology,
        validate_planner_topology_contract,
    )

    if isinstance(expected, Mapping):
        trusted = dict(expected)
    else:
        trusted = dict(
            _read_json(
                Path(expected).expanduser().resolve(),
                label="trusted planner topology",
            )
        )
    validate_planner_topology_contract(trusted)
    trusted_hash = sha256_json(trusted)
    if trusted_hash != metadata.planner_topology_hash:
        raise ValueError(
            "trusted planner topology hash differs from checkpoint metadata"
        )
    actual = planner_safetensors_topology(checkpoint / "planner.safetensors")
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
            name: tuple(handle.get_slice(name).get_shape())
            for name in handle.keys()
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
            raise TypeError(
                "processor tokenizer must expose convert_tokens_to_ids"
            )
        actual_ids = tuple(int(convert(token)) for token in ADDED_TOKENS)
        if actual_ids != added_token_ids:
            raise ValueError(
                "processor tokenizer added-token IDs differ from provider contract"
            )
        self.planner = planner
        self.processor = processor
        self.added_token_ids = added_token_ids
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
        expected_planner_topology: str | Path | Mapping[str, Any],
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        _component_loader: Callable[..., tuple[Any, nn.Module]] | None = None,
    ) -> "FrozenBatonPlanner":
        """Validate all metadata/hashes before loading or mutating model state."""

        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        model_path = Path(qwen_model_path).expanduser().resolve()
        tokenizer_path = Path(qwen_tokenizer_path).expanduser().resolve()
        processor_path = Path(qwen_processor_path).expanduser().resolve()
        siglip_path = Path(siglip2_model_path).expanduser().resolve()

        # Everything through the artifact checks is read-only and precedes both
        # Transformers construction and safetensors state mutation.
        metadata = _validate_checkpoint_envelope(checkpoint)
        _validate_trusted_planner_topology(
            metadata,
            checkpoint=checkpoint,
            expected=expected_planner_topology,
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
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError("torch_dtype must be a torch dtype")
        target_device = torch.device(device)
        loader = _load_local_components if _component_loader is None else _component_loader
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
        counterfactual_instructions: Sequence[str] | None,
    ) -> tuple[int, tuple[str, ...], tuple[str, ...] | None]:
        if (
            not isinstance(current_images, torch.Tensor)
            or current_images.ndim != 5
            or tuple(current_images.shape[1:3]) != (2, 3)
            or current_images.shape[-2] <= 0
            or current_images.shape[-1] <= 0
        ):
            raise ValueError("current_images must have shape [B,2,3,H,W]")
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
        negative = None
        if counterfactual_instructions is not None:
            if isinstance(counterfactual_instructions, (str, bytes)) or not isinstance(
                counterfactual_instructions, Sequence
            ):
                raise TypeError(
                    "counterfactual instructions must be an outer sequence of strings"
                )
            negative = tuple(counterfactual_instructions)
            if len(negative) != batch_size:
                raise ValueError(
                    "images and counterfactual instructions batch sizes must match"
                )
            if any(type(value) is not str or not value.strip() for value in negative):
                raise ValueError(
                    "counterfactual instructions must contain nonempty strings"
                )
            if any(a == b for a, b in zip(positive, negative, strict=True)):
                raise ValueError(
                    "counterfactual instructions must differ from positives"
                )
        return batch_size, positive, negative

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
        instruction_sets: tuple[tuple[str, ...], ...],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        collator = BatonPlannerCollator(
            self.processor,
            plan_pad_token_id=self.added_token_ids[5],
        )
        sequences: list[torch.Tensor] = []
        processed_rows: list[Mapping[str, torch.Tensor]] = []
        camera_ids: list[int] = []
        cpu_images = current_images.detach().to(device="cpu")
        for instructions in instruction_sets:
            for sample_index, instruction in enumerate(instructions):
                for camera_index in range(2):
                    sequence, processed = collator._process_row(
                        cpu_images[sample_index, camera_index],
                        instruction,
                    )
                    sequences.append(sequence)
                    processed_rows.append(processed)
                    camera_ids.append(camera_index)
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
            torch.tensor(camera_ids, dtype=torch.long, device=device),
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
            len(self.geometry.future_indices),
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
        if (
            not isinstance(maps, tuple)
            or len(maps) != self.geometry.query_layers
        ):
            raise RuntimeError(
                "planner must return four Query Tower cross-attention maps"
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
        counterfactual_instructions: Sequence[str] | None = None,
        return_attention: bool = False,
    ) -> BatonSemanticPlan:
        """Predict full grids, optionally tracing attention and instruction sensitivity."""

        if type(return_attention) is not bool:
            raise TypeError("return_attention must be a boolean")
        batch_size, positive, negative = self._validate_inputs(
            current_images,
            instructions,
            counterfactual_instructions,
        )
        self._freeze_for_inference()
        instruction_sets = (positive,) if negative is None else (positive, negative)
        qwen_inputs, plan_positions, camera_ids = self._build_rows(
            current_images,
            instruction_sets,
        )
        rows = batch_size * 2 * len(instruction_sets)
        with self._autocast_context():
            output = self.planner.forward_rows(
                qwen_inputs,
                plan_positions,
                camera_ids,
                return_attention_maps=return_attention,
            )
        flat, raw_maps = self._validate_output(
            output,
            rows=rows,
            return_attention=return_attention,
        )
        positive_rows = batch_size * 2
        tokens = flat[:positive_rows].reshape(
            self.geometry.output_shape(batch_size)
        )
        sensitivity = None
        if negative is not None:
            negative_tokens = flat[positive_rows:].reshape(
                self.geometry.output_shape(batch_size)
            )
            sensitivity = 1.0 - F.cosine_similarity(
                tokens.float(),
                negative_tokens.float(),
                dim=-1,
            )
        attention_maps = None
        if raw_maps is not None:
            attention_maps = tuple(
                attention[:positive_rows].reshape(
                    batch_size,
                    2,
                    self.geometry.tokens_per_camera,
                    self.geometry.tokens_per_camera,
                )
                for attention in raw_maps
            )
        return BatonSemanticPlan(
            tokens=tokens.detach(),
            future_indices=self.geometry.future_indices,
            positions_xy=build_patch_center_positions(
                batch_size,
                device=tokens.device,
            ),
            cross_attention_maps=(
                None
                if attention_maps is None
                else tuple(value.detach() for value in attention_maps)
            ),
            instruction_sensitivity=(
                None if sensitivity is None else sensitivity.detach()
            ),
        )
