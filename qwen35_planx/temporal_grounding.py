"""Frozen DINOv3 correspondence, action phases, and hindsight-map fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


_GRID_SIZE = 27
_TOKENS = _GRID_SIZE * _GRID_SIZE
_MIN_EVIDENCE = 1e-6

EVIDENCE_EXPONENTS = {
    "source": (0.45, 0.30, 0.15, 0.10),
    "target": (0.45, 0.15, 0.25, 0.15),
    "action": (0.35, 0.25, 0.20, 0.20),
}


@dataclass(frozen=True)
class DinoTracks:
    """Adjacent-frame flow ``(dx, dy, cycle confidence)`` on a patch grid."""

    flow: Tensor


@dataclass(frozen=True)
class ActionPhases:
    """Normalized temporal source, transport, and target priors."""

    source: Tensor
    transport: Tensor
    target: Tensor
    confidence: float


@dataclass(frozen=True)
class FusedHindsightMap:
    """One normalized role map and its usable grounding confidence."""

    map: Tensor
    confidence: float

    def __iter__(self):
        yield self.map
        yield self.confidence


def _module_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _grid_size(tokens: int) -> int:
    grid = math.isqrt(tokens)
    if grid * grid != tokens:
        raise ValueError("feature token count must form a square grid")
    return grid


def _local_nearest(
    query: Tensor,
    candidates: Tensor,
    *,
    grid: int,
    radius: int,
) -> tuple[Tensor, Tensor]:
    """Return candidate indices and cosine scores for each query position."""

    pairs, _, width = query.shape
    query_grid = query.transpose(1, 2).reshape(pairs, width, grid, grid)
    candidate_grid = candidates.transpose(1, 2).reshape(pairs, width, grid, grid)
    scores = torch.full(
        (pairs, grid, grid),
        -torch.inf,
        device=query.device,
        dtype=query.dtype,
    )
    indices = torch.zeros(
        (pairs, grid, grid),
        device=query.device,
        dtype=torch.long,
    )
    target_positions = torch.arange(
        grid * grid,
        device=query.device,
    ).reshape(grid, grid)
    for dy in range(-radius, radius + 1):
        source_y = slice(max(0, -dy), min(grid, grid - dy))
        target_y = slice(max(0, dy), min(grid, grid + dy))
        for dx in range(-radius, radius + 1):
            source_x = slice(max(0, -dx), min(grid, grid - dx))
            target_x = slice(max(0, dx), min(grid, grid + dx))
            similarity = torch.einsum(
                "bchw,bchw->bhw",
                query_grid[:, :, source_y, source_x],
                candidate_grid[:, :, target_y, target_x],
            )
            current = scores[:, source_y, source_x]
            better = similarity > current
            scores[:, source_y, source_x] = torch.where(
                better,
                similarity,
                current,
            )
            proposed = target_positions[target_y, target_x].expand(pairs, -1, -1)
            current_indices = indices[:, source_y, source_x]
            indices[:, source_y, source_x] = torch.where(
                better,
                proposed,
                current_indices,
            )
    return indices.flatten(1), scores.flatten(1)


def _track_single(features: Tensor, *, search_radius: int) -> Tensor:
    if features.ndim != 3 or features.shape[0] < 2:
        raise ValueError("features must have shape [frames>=2, tokens, width]")
    grid = _grid_size(features.shape[1])
    normalized = F.normalize(features.float(), dim=-1)
    before, after = normalized[:-1], normalized[1:]
    forward_chunks: list[Tensor] = []
    score_chunks: list[Tensor] = []
    backward_chunks: list[Tensor] = []
    for start in range(0, before.shape[0], 16):
        stop = start + 16
        forward_indices, forward_scores = _local_nearest(
            before[start:stop],
            after[start:stop],
            grid=grid,
            radius=search_radius,
        )
        backward_indices, _ = _local_nearest(
            after[start:stop],
            before[start:stop],
            grid=grid,
            radius=search_radius,
        )
        forward_chunks.append(forward_indices)
        score_chunks.append(forward_scores)
        backward_chunks.append(backward_indices)
    forward_indices = torch.cat(forward_chunks)
    forward_scores = torch.cat(score_chunks)
    backward_indices = torch.cat(backward_chunks)
    cycle_indices = backward_indices.gather(1, forward_indices)

    source = torch.arange(features.shape[1], device=features.device).view(1, -1)
    source_y = torch.div(source, grid, rounding_mode="floor")
    source_x = source.remainder(grid)
    cycle_y = torch.div(cycle_indices, grid, rounding_mode="floor")
    cycle_x = cycle_indices.remainder(grid)
    cycle_consistent = torch.maximum(
        (cycle_x - source_x).abs(),
        (cycle_y - source_y).abs(),
    ) <= 1

    target_y = torch.div(forward_indices, grid, rounding_mode="floor")
    target_x = forward_indices.remainder(grid)
    dx = (target_x - source_x).to(features.dtype)
    dy = (target_y - source_y).to(features.dtype)
    confidence = forward_scores.clamp(0, 1).to(features.dtype)
    valid = (
        cycle_consistent
        & torch.isfinite(forward_scores)
        & (forward_scores >= 0.5)
    )
    dx = torch.where(valid, dx, torch.zeros_like(dx))
    dy = torch.where(valid, dy, torch.zeros_like(dy))
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    return torch.stack((dx, dy, confidence), dim=-1)


def track_keyframes(features: Tensor, *, search_radius: int = 4) -> DinoTracks:
    """Track normalized DINO patches with local cycle-consistent cosine matches.

    Inputs may be ``[frames,tokens,width]`` or
    ``[camera,frames,tokens,width]``. Camera streams never share candidates.
    """

    if search_radius < 0:
        raise ValueError("search_radius must be non-negative")
    if features.ndim == 3:
        return DinoTracks(_track_single(features, search_radius=search_radius))
    if features.ndim == 4:
        return DinoTracks(
            torch.stack(
                [
                    _track_single(camera, search_radius=search_radius)
                    for camera in features
                ]
            )
        )
    raise ValueError(
        "features must have shape [frames,tokens,width] or "
        "[camera,frames,tokens,width]"
    )


class DinoTemporalTeacher:
    """Frozen offline DINOv3 encoder producing normalized 27x27 features."""

    def __init__(self, *, model: nn.Module, processor: Any) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        self.model = model.eval()
        self.processor = processor
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    @classmethod
    def from_components(
        cls,
        *,
        model: nn.Module,
        processor: Any,
    ) -> DinoTemporalTeacher:
        return cls(model=model, processor=processor)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        local_files_only: bool = True,
    ) -> DinoTemporalTeacher:
        """Load a local DINOv3 model without permitting an implicit download."""

        from transformers import AutoImageProcessor, DINOv3ViTModel

        model = DINOv3ViTModel.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
        )
        processor = AutoImageProcessor.from_pretrained(
            str(model_path),
            local_files_only=local_files_only,
        )
        return cls(model=model, processor=processor)

    def _patch_features(self, pixels: Tensor) -> Tensor:
        output = self.model(pixel_values=pixels)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None or hidden.ndim != 3:
            raise ValueError("DINOv3 must return last_hidden_state [batch,tokens,width]")
        register_tokens = int(getattr(self.model.config, "num_register_tokens", 0))
        prefix = 1 + register_tokens
        patches = hidden[:, prefix:]
        source_grid = _grid_size(patches.shape[1])
        patches = patches.transpose(1, 2).reshape(
            patches.shape[0],
            patches.shape[2],
            source_grid,
            source_grid,
        )
        patches = F.interpolate(
            patches,
            size=(_GRID_SIZE, _GRID_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return F.normalize(patches.flatten(2).transpose(1, 2).float(), dim=-1)

    @torch.inference_mode()
    def encode(self, rgb: Tensor, *, microbatch_size: int = 16) -> Tensor:
        """Encode complete RGB video, preserving optional camera/frame axes."""

        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if rgb.ndim == 4:
            leading = (rgb.shape[0],)
            flat = rgb
        elif rgb.ndim == 5:
            leading = (rgb.shape[0], rgb.shape[1])
            flat = rgb.flatten(0, 1)
        else:
            raise ValueError(
                "rgb must have shape [frames,3,height,width] or "
                "[camera,frames,3,height,width]"
            )
        if flat.shape[1] != 3:
            raise ValueError("rgb channel axis must have width three")
        if not flat.is_floating_point():
            flat = flat.float().div(255)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("rgb must contain finite values")

        chunks: list[Tensor] = []
        device = _module_device(self.model)
        for start in range(0, flat.shape[0], microbatch_size):
            processed = self.processor(
                images=flat[start : start + microbatch_size],
                return_tensors="pt",
            )
            if not hasattr(processed, "items"):
                raise TypeError("DINOv3 processor must return a tensor mapping")
            inputs = {
                key: value.to(device)
                for key, value in processed.items()
                if isinstance(value, Tensor)
            }
            pixels = inputs.get("pixel_values")
            if pixels is None:
                raise ValueError("DINOv3 processor did not return pixel_values")
            chunks.append(self._patch_features(pixels))
        encoded = torch.cat(chunks)
        return encoded.reshape(*leading, _TOKENS, encoded.shape[-1])


def _uniform_phases(length: int, *, device: torch.device) -> ActionPhases:
    prior = torch.full((length,), 1.0 / length, device=device)
    return ActionPhases(prior, prior.clone(), prior.clone(), 0.0)


def _transition(
    closed: Tensor,
    *,
    from_closed: bool,
    to_closed: bool,
    start: int,
    persistence: int,
) -> int | None:
    for index in range(max(start, persistence), len(closed) - persistence + 1):
        before = closed[index - persistence : index]
        after = closed[index : index + persistence]
        if bool(torch.all(before == from_closed)) and bool(
            torch.all(after == to_closed)
        ):
            return index
    return None


def _gaussian_prior(length: int, center: int, *, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)
    sigma = max(1.0, 0.08 * length)
    prior = torch.exp(-0.5 * ((positions - center) / sigma).square())
    return prior / prior.sum().clamp_min(_MIN_EVIDENCE)


def detect_action_phases(
    actions: Tensor | None,
    states: Tensor | None,
    *,
    persistence: int = 3,
) -> ActionPhases:
    """Derive close/transport/release priors from the seventh action channel."""

    if actions is None:
        if states is None:
            raise ValueError("actions and states cannot both be absent")
        states = torch.as_tensor(states)
        if states.ndim != 2 or states.shape[0] == 0:
            raise ValueError("states must have shape [steps,channels]")
        return _uniform_phases(states.shape[0], device=states.device)
    actions = torch.as_tensor(actions)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("actions must have shape [steps,channels]")
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    length = actions.shape[0]
    if states is not None:
        states = torch.as_tensor(states)
        if states.ndim != 2 or states.shape[0] != length:
            raise ValueError("states must align with action steps")
    if (
        actions.shape[1] < 7
        or not bool(torch.isfinite(actions[:, 6]).all())
        or length < 3 * persistence
    ):
        return _uniform_phases(length, device=actions.device)

    command = actions[:, 6]
    threshold = 0.5 * (command.amin() + command.amax())
    closed = command <= threshold
    closure = _transition(
        closed,
        from_closed=False,
        to_closed=True,
        start=persistence,
        persistence=persistence,
    )
    if closure is None:
        return _uniform_phases(length, device=actions.device)
    release = _transition(
        closed,
        from_closed=True,
        to_closed=False,
        start=closure + persistence,
        persistence=persistence,
    )
    if release is None or release <= closure + 1:
        return _uniform_phases(length, device=actions.device)

    source = _gaussian_prior(length, closure, device=actions.device)
    target = _gaussian_prior(length, release, device=actions.device)
    transport = torch.zeros(length, device=actions.device, dtype=torch.float32)
    transport[closure + 1 : release] = 1
    transport /= transport.sum().clamp_min(_MIN_EVIDENCE)
    return ActionPhases(source, transport, target, 1.0)


def _prepare_evidence(value: Tensor | Sequence[float]) -> tuple[Tensor, bool]:
    tensor = torch.as_tensor(value).float()
    if tensor.shape[-2:] == (_GRID_SIZE, _GRID_SIZE):
        tensor = tensor.flatten(-2)
    if tensor.shape != (_TOKENS,):
        raise ValueError(f"each evidence map must contain exactly {_TOKENS} values")
    valid = bool(torch.isfinite(tensor).all()) and bool((tensor >= 0).all())
    valid = valid and bool(tensor.sum() > 0)
    return tensor, valid


def fuse_hindsight_maps(
    role: str,
    *,
    text: Tensor | Sequence[float],
    track: Tensor | Sequence[float],
    change: Tensor | Sequence[float],
    phase: Tensor | Sequence[float],
    confidences: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> FusedHindsightMap:
    """Fuse four spatial signals with a confidence-weighted geometric mean."""

    if role not in EVIDENCE_EXPONENTS:
        raise ValueError(f"role must be one of {tuple(EVIDENCE_EXPONENTS)}")
    if len(confidences) != 4:
        raise ValueError("confidences must contain text, track, change, and phase")
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in confidences):
        raise ValueError("evidence confidences must be finite values in [0, 1]")

    prepared = [
        _prepare_evidence(value)
        for value in (text, track, change, phase)
    ]
    reference = prepared[0][0]
    base_weights = EVIDENCE_EXPONENTS[role]
    active: list[tuple[Tensor, float]] = []
    for (evidence, finite_valid), base, confidence in zip(
        prepared, base_weights, confidences
    ):
        valid = finite_valid and float(confidence) > 0
        if valid:
            active.append((evidence.to(reference), base * float(confidence)))

    if active:
        total_weight = sum(weight for _, weight in active)
        log_map = sum(
            weight * torch.log(evidence.clamp_min(_MIN_EVIDENCE))
            for evidence, weight in active
        ) / total_weight
        fused = torch.exp(log_map)
        fused = fused / fused.sum().clamp_min(_MIN_EVIDENCE)
    else:
        fused = torch.full_like(reference, 1.0 / _TOKENS)

    output_confidence = 0.0
    if len(active) >= 2:
        output_confidence = min(
            1.0,
            sum(
                base * float(confidence)
                for base, confidence, (_, valid) in zip(
                    base_weights, confidences, prepared
                )
                if valid and float(confidence) > 0
            )
            / sum(base_weights),
        )
    return FusedHindsightMap(fused, output_confidence)
