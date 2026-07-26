"""Offline orchestration for content-bounded video-hindsight targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from qwen35_planx.config import PlanGeometry, _GROUNDING_ROLES
from qwen35_planx.hindsight_data import HDF5Trajectory, HindsightWindowRecord
from qwen35_planx.instruction import (
    InstructionFields,
    InstructionVocabulary,
    parse_libero_instruction,
)
from qwen35_planx.temporal_grounding import (
    ActionPhases,
    detect_action_phases,
    fuse_hindsight_maps,
    track_keyframes,
)


_TOKENS = 729
_TEXT_WIDTH = 1152
_ROLE_PHASES = ("source", "target", "transport")


@dataclass(frozen=True)
class HindsightTarget:
    """The complete and exclusive set of per-window cache products."""

    codes: Tensor
    relevance: Tensor
    confidence: Tensor
    flow: Tensor
    phrase_embeddings: Tensor

    def __post_init__(self) -> None:
        expected_shapes = {
            "codes": (2, 4, _TOKENS),
            "relevance": (2, 4, 3, _TOKENS),
            "confidence": (2, 4, 3),
            "flow": (2, 3, _TOKENS, 3),
            "phrase_embeddings": (3, _TEXT_WIDTH),
        }
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if name != "codes" and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite values")
        if self.codes.dtype != torch.long:
            raise ValueError("codes must have dtype torch.long")
        if bool((self.codes < 0).any()) or bool(
            (self.codes >= PlanGeometry().visual_vocab_size).any()
        ):
            raise ValueError("codes are outside the released TA-Tok vocabulary")
        if bool((self.relevance < 0).any()):
            raise ValueError("relevance must be non-negative")
        if not torch.allclose(
            self.relevance.sum(dim=-1),
            torch.ones_like(self.relevance[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("relevance maps must be normalized")
        if bool((self.confidence < 0).any()) or bool((self.confidence > 1).any()):
            raise ValueError("confidence must be in [0, 1]")

    @property
    def teacher_only_fields(self) -> tuple[()]:
        """Teacher inputs and dense intermediates are intentionally absent."""

        return ()


def build_counterfactual_vocabulary(
    records: Sequence[HindsightWindowRecord],
    *,
    parser: Callable[[str], InstructionFields] = parse_libero_instruction,
) -> InstructionVocabulary:
    """Derive deterministic substitutions from train records and never val."""

    train = sorted(
        (
            record
            for record in records
            if isinstance(record, HindsightWindowRecord) and record.split == "train"
        ),
        key=lambda record: record.sample_id,
    )
    if not train:
        raise ValueError("counterfactual vocabulary requires train-split windows")
    fields = [parser(record.caption) for record in train]
    return InstructionVocabulary(
        actions=tuple(field.action for field in fields if field.action),
        sources=tuple(field.source for field in fields if field.source),
        targets=tuple(field.target for field in fields if field.target),
    )


def _negative_phrases(
    fields: InstructionFields,
    vocabulary: InstructionVocabulary | None,
) -> tuple[str, str, str]:
    if vocabulary is None:
        return ("", "", "")
    candidates = {
        "source": vocabulary.sources,
        "target": vocabulary.targets,
        "action": vocabulary.actions,
    }
    result = []
    for role in _GROUNDING_ROLES:
        positive = getattr(fields, role)
        result.append(next((value for value in candidates[role] if value != positive), ""))
    return tuple(result)  # type: ignore[return-value]


def _normalize_map(value: Tensor) -> Tensor:
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    value = value.clamp_min(0).flatten()
    mass = value.sum()
    if not bool(torch.isfinite(mass)) or float(mass) <= 0:
        return torch.full_like(value, 1.0 / value.numel())
    return value / mass


def _warp_once(distribution: Tensor, flow: Tensor, *, forward: bool) -> Tensor:
    distribution = _normalize_map(distribution)
    flow = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
    confidence = flow[:, 2].clamp(0, 1)
    positions = torch.arange(_TOKENS, device=flow.device)
    source_y = torch.div(positions, 27, rounding_mode="floor")
    source_x = positions.remainder(27)
    target_x = (source_x + flow[:, 0].round().long()).clamp(0, 26)
    target_y = (source_y + flow[:, 1].round().long()).clamp(0, 26)
    target = target_y * 27 + target_x
    if forward:
        output = torch.zeros_like(distribution)
        output.scatter_add_(0, target, distribution * confidence)
        output.scatter_add_(0, positions, distribution * (1 - confidence))
    else:
        output = distribution[target] * confidence + distribution * (1 - confidence)
    return _normalize_map(output)


def _propagate(
    seed: Tensor,
    adjacent_flow: Tensor,
    *,
    start: int,
    stop: int,
) -> Tensor:
    result = _normalize_map(seed).to(adjacent_flow)
    if start < stop:
        for index in range(start, stop):
            result = _warp_once(result, adjacent_flow[index], forward=True)
    elif start > stop:
        for index in range(start - 1, stop - 1, -1):
            result = _warp_once(result, adjacent_flow[index], forward=False)
    return result


def _change_map(
    initial: Tensor,
    final: Tensor,
    flow: Tensor,
) -> tuple[Tensor, float]:
    """Measure feature change only after following cycle-valid correspondence."""

    flow = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
    positions = torch.arange(_TOKENS, device=flow.device)
    source_y = torch.div(positions, 27, rounding_mode="floor")
    source_x = positions.remainder(27)
    target_x = (source_x + flow[:, 0].round().long()).clamp(0, 26)
    target_y = (source_y + flow[:, 1].round().long()).clamp(0, 26)
    target = target_y * 27 + target_x
    matched_final = final[target]
    similarity = (
        F.normalize(initial.float(), dim=-1)
        * F.normalize(matched_final.float(), dim=-1)
    ).sum(dim=-1)
    change = torch.nan_to_num(
        1 - similarity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0)
    change = torch.where(change > 1e-5, change, torch.zeros_like(change))
    match_confidence = flow[:, 2].clamp(0, 1)
    change = change * match_confidence
    confidence = float(change.mean().clamp(0, 1))
    return _normalize_map(change), confidence


def _phase_for_role(phases: ActionPhases, role: str) -> Tensor:
    if role == "source":
        return phases.source
    if role == "target":
        return phases.target
    return phases.transport


def _phase_anchors(phases: ActionPhases) -> tuple[int, int, int]:
    return (
        int(torch.argmax(phases.source)),
        int(torch.argmax(phases.target)),
        int(torch.argmax(phases.transport)),
    )


def _validate_relevance_output(
    output: object,
    *,
    frames: int,
) -> None:
    for name, expected in {
        "phrase_embeddings": (3, _TEXT_WIDTH),
        "maps": (frames, 3, 27, 27),
        "confidence": (frames, 3),
    }.items():
        value = getattr(output, name, None)
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"SigLIP relevance {name} must have shape {expected}")


def _component_device(component: object) -> torch.device:
    parameters = getattr(component, "parameters", None)
    if callable(parameters):
        parameter = next(parameters(), None)
        if parameter is not None:
            return parameter.device
    return torch.device("cpu")


class HindsightTargetBuilder:
    """Run frozen teachers on a complete trajectory and retain only K=4."""

    def __init__(
        self,
        *,
        ta_tokenizer: object,
        siglip_teacher: object,
        dino_teacher: object,
        vocabulary: InstructionVocabulary | None,
        instruction_parser: Callable[[str], InstructionFields],
        microbatch_size: int,
    ) -> None:
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        for component, method in (
            (ta_tokenizer, "encode_codes"),
            (siglip_teacher, "encode_fields"),
            (dino_teacher, "encode"),
        ):
            if not callable(getattr(component, method, None)):
                raise TypeError(f"teacher component must provide {method}()")
        self.ta_tokenizer = ta_tokenizer
        self.siglip_teacher = siglip_teacher
        self.dino_teacher = dino_teacher
        self.vocabulary = vocabulary
        self.instruction_parser = instruction_parser
        self.microbatch_size = int(microbatch_size)

    @classmethod
    def from_components(
        cls,
        *,
        ta_tokenizer: object,
        siglip_teacher: object,
        dino_teacher: object,
        vocabulary: InstructionVocabulary | None = None,
        instruction_parser: Callable[
            [str], InstructionFields
        ] = parse_libero_instruction,
        microbatch_size: int = 16,
    ) -> HindsightTargetBuilder:
        return cls(
            ta_tokenizer=ta_tokenizer,
            siglip_teacher=siglip_teacher,
            dino_teacher=dino_teacher,
            vocabulary=vocabulary,
            instruction_parser=instruction_parser,
            microbatch_size=microbatch_size,
        )

    @staticmethod
    def _validate_bounds(
        trajectory: HDF5Trajectory,
        window: HindsightWindowRecord,
    ) -> None:
        length = trajectory.rgb.shape[1]
        indices = (
            window.current_index,
            *window.future_indices,
            *window.frame_indices,
            *window.action_indices,
        )
        if any(index < 0 or index >= length for index in indices):
            raise ValueError(
                f"window indices exceed complete trajectory bounds [0, {length})"
            )

    def build_window(
        self,
        trajectory: HDF5Trajectory,
        window: HindsightWindowRecord,
    ) -> HindsightTarget:
        if not isinstance(trajectory, HDF5Trajectory):
            raise TypeError("trajectory must be HDF5Trajectory")
        if not isinstance(window, HindsightWindowRecord):
            raise TypeError("window must be HindsightWindowRecord")
        self._validate_bounds(trajectory, window)

        fields = self.instruction_parser(window.caption)
        negatives = _negative_phrases(fields, self.vocabulary)
        rgb = (
            torch.from_numpy(np.array(trajectory.rgb, copy=True))
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        actions = torch.from_numpy(np.array(trajectory.actions, copy=True))
        states = torch.from_numpy(np.array(trajectory.states, copy=True))

        phases = detect_action_phases(actions, states)
        role_anchors = _phase_anchors(phases)
        selected_indices = tuple(
            sorted(set((*role_anchors, *window.future_indices)))
        )
        selected_lookup = {
            frame_index: selected_index
            for selected_index, frame_index in enumerate(selected_indices)
        }

        full_features = self.dino_teacher.encode(
            rgb,
            microbatch_size=self.microbatch_size,
        )
        if (
            not isinstance(full_features, Tensor)
            or full_features.ndim != 4
            or tuple(full_features.shape[:3])
            != (2, trajectory.rgb.shape[1], _TOKENS)
            or not bool(torch.isfinite(full_features).all())
        ):
            raise ValueError(
                "DINO features must be finite [2, complete_frames, 729, width]"
            )
        adjacent_flow = track_keyframes(full_features).flow
        future_indices = list(window.future_indices)
        keyframe_flow = track_keyframes(full_features[:, future_indices]).flow

        relevance_outputs = []
        for camera in range(2):
            output = self.siglip_teacher.encode_fields(
                rgb[camera, list(selected_indices)],
                fields,
                counterfactual_phrases=negatives,
            )
            _validate_relevance_output(output, frames=len(selected_indices))
            relevance_outputs.append(output)

        embeddings = torch.stack(
            [output.phrase_embeddings.float() for output in relevance_outputs]
        ).mean(dim=0)
        embeddings = F.normalize(embeddings, dim=-1)
        field_mask = torch.tensor(
            fields.confidences,
            device=embeddings.device,
            dtype=torch.float32,
        )
        embeddings = embeddings * field_mask.unsqueeze(-1)

        relevance = torch.empty(
            2,
            4,
            3,
            _TOKENS,
            device=full_features.device,
            dtype=torch.float32,
        )
        confidence = torch.empty(
            2,
            4,
            3,
            device=full_features.device,
            dtype=torch.float32,
        )
        for camera, output in enumerate(relevance_outputs):
            camera_flow = adjacent_flow[camera]
            change_cache: dict[int, tuple[Tensor, float]] = {}
            for keyframe, frame_index in enumerate(window.future_indices):
                selected = selected_lookup[frame_index]
                if frame_index not in change_cache:
                    correspondence = track_keyframes(
                        full_features[
                            camera,
                            [window.current_index, frame_index],
                        ]
                    ).flow[0]
                    change_cache[frame_index] = _change_map(
                        full_features[camera, window.current_index],
                        full_features[camera, frame_index],
                        correspondence,
                    )
                for role_index, role in enumerate(_GROUNDING_ROLES):
                    anchor = role_anchors[_ROLE_PHASES.index(
                        role if role != "action" else "transport"
                    )]
                    seed = output.maps[
                        selected_lookup[anchor],
                        role_index,
                    ].flatten()
                    track = _propagate(
                        seed,
                        camera_flow,
                        start=anchor,
                        stop=frame_index,
                    )
                    track_confidence = float(
                        camera_flow[
                            min(anchor, frame_index) : max(anchor, frame_index),
                            :,
                            2,
                        ].mean()
                    ) if anchor != frame_index else 1.0
                    change, change_confidence = change_cache[frame_index]
                    phase_prior = _phase_for_role(phases, role)
                    phase_strength = float(
                        phase_prior[frame_index]
                        / phase_prior.max().clamp_min(torch.finfo(torch.float32).tiny)
                    )
                    fused = fuse_hindsight_maps(
                        role,
                        text=output.maps[selected, role_index],
                        track=track,
                        change=change,
                        phase=track,
                        confidences=(
                            float(output.confidence[selected, role_index]),
                            max(0.0, min(1.0, track_confidence)),
                            change_confidence,
                            float(phases.confidence) * phase_strength,
                        ),
                    )
                    relevance[camera, keyframe, role_index] = _normalize_map(
                        fused.map
                    )
                    counterfactual_valid = float(
                        output.confidence[selected, role_index]
                    ) > 0
                    confidence[camera, keyframe, role_index] = (
                        float(fused.confidence)
                        * field_mask[role_index]
                        * counterfactual_valid
                    )

        future_rgb = (
            rgb[:, future_indices]
            .flatten(0, 1)
            .float()
            .div(255)
            .to(_component_device(self.ta_tokenizer))
        )
        encoded = self.ta_tokenizer.encode_codes(future_rgb)
        codes = getattr(encoded, "codes", encoded)
        if not isinstance(codes, Tensor) or tuple(codes.shape) != (8, _TOKENS):
            raise ValueError("TA-Tok must return codes with shape [8, 729]")

        return HindsightTarget(
            codes=codes.detach().to(device="cpu", dtype=torch.long).reshape(
                2, 4, _TOKENS
            ),
            relevance=relevance.detach().cpu(),
            confidence=confidence.detach().cpu(),
            flow=keyframe_flow.detach().float().cpu(),
            phrase_embeddings=embeddings.detach().float().cpu(),
        )
