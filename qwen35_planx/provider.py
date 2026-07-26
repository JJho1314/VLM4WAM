"""Frozen generation and differentiable teacher-forcing provider for GE-Act."""

from __future__ import annotations

import math
from numbers import Real
from dataclasses import fields, replace
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from qwen35_planx.config import CAMERA_NAMES, PlanGeometry
from qwen35_planx.decoding import (
    _current_visual_embedding_rows,
    GeneratedGroundedPlan,
    generate_grounded_plan,
    unflatten_generated_plan,
)
from qwen35_planx.planner import GroundedPlannerOutput
from qwen35_planx.planner_dataset import (
    CachedPlannerTargets,
    GroundedPlannerBatch,
    GroundedPlannerCollator,
)
from qwen35_planx.vocabulary import VisualVocabularyLayout


_GEOMETRY = PlanGeometry()
_NORMALIZED_TIMES = (0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0)


def scale_gradient(x: Tensor, scale: float) -> Tensor:
    """Keep ``x`` identical in forward while scaling only its incoming gradient."""

    if not isinstance(x, Tensor):
        raise TypeError("x must be a tensor")
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("scale must be a real number in [0,1]")
    scale = float(scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be finite and in [0,1]")
    return x.detach() + scale * (x - x.detach())


class LearnedPlanPositionEncoder(nn.Module):
    """Learned 27x27 spatial, four-time, and dual-camera position features."""

    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        if type(condition_dim) is not int or condition_dim <= 0:
            raise ValueError("condition_dim must be a positive integer")
        self.condition_dim = condition_dim
        self.spatial_embedding = nn.Parameter(
            torch.empty(_GEOMETRY.tokens_per_frame, condition_dim)
        )
        self.time_embedding = nn.Parameter(
            torch.empty(_GEOMETRY.num_keyframes, condition_dim)
        )
        self.camera_embedding = nn.Parameter(
            torch.empty(len(CAMERA_NAMES), condition_dim)
        )
        nn.init.normal_(self.spatial_embedding, std=0.02)
        nn.init.normal_(self.time_embedding, std=0.02)
        nn.init.normal_(self.camera_embedding, std=0.02)

    def forward(
        self,
        *,
        grid_size: tuple[int, int],
        times: Tensor,
        batch_shape: Sequence[int],
    ) -> Tensor:
        if tuple(grid_size) != (_GEOMETRY.grid_size, _GEOMETRY.grid_size):
            raise ValueError("position grid must be exactly 27x27")
        batch_shape = tuple(int(value) for value in batch_shape)
        if len(batch_shape) != 3 or batch_shape[1:] != (
            len(CAMERA_NAMES),
            _GEOMETRY.num_keyframes,
        ):
            raise ValueError("plan batch shape must be [B,2,4]")
        if batch_shape[0] <= 0:
            raise ValueError("position batch must be nonempty")
        if (
            not isinstance(times, Tensor)
            or times.shape != (_GEOMETRY.num_keyframes,)
            or not times.dtype.is_floating_point
            or not bool(torch.isfinite(times).all())
        ):
            raise ValueError("times must contain four finite normalized positions")
        expected_times = torch.tensor(
            _NORMALIZED_TIMES,
            dtype=times.dtype,
            device=times.device,
        )
        if not torch.allclose(times, expected_times, atol=1e-6, rtol=0):
            raise ValueError("times differ from the four grounded keyframe positions")
        positions = (
            self.spatial_embedding.view(1, 1, 1, 729, self.condition_dim)
            + self.time_embedding.view(1, 1, 4, 1, self.condition_dim)
            + self.camera_embedding.view(1, 2, 1, 1, self.condition_dim)
        )
        return positions.expand(batch_shape[0], -1, -1, -1, -1)


class Qwen35GroundedPlanProvider(nn.Module):
    """Expose one shared grounded plan representation to downstream GE-Act."""

    def __init__(
        self,
        *,
        planner: nn.Module,
        collator: GroundedPlannerCollator | Any,
        layout: VisualVocabularyLayout | Any,
        condition_dim: int = 1024,
        _enforce_released_geometry: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(planner, nn.Module):
            raise TypeError("planner must be a torch module")
        if not callable(getattr(collator, "build_teacher_forced", None)):
            raise TypeError("collator must expose build_teacher_forced")
        if type(condition_dim) is not int or condition_dim <= 0:
            raise ValueError("condition_dim must be a positive integer")
        codebook = getattr(planner, "codebook", None)
        hidden_dim = getattr(planner, "hidden_dim", None)
        if (
            not isinstance(codebook, Tensor)
            or codebook.ndim != 2
            or type(hidden_dim) is not int
            or hidden_dim <= 0
        ):
            raise TypeError("planner must expose Task 7 codebook and hidden_dim")
        if _enforce_released_geometry:
            released_widths = (
                int(codebook.shape[0]),
                int(codebook.shape[1]),
                hidden_dim,
                int(getattr(planner, "text_dim", -1)),
            )
            if released_widths != (
                _GEOMETRY.visual_vocab_size,
                _GEOMETRY.ta_code_dim,
                _GEOMETRY.qwen_hidden_dim,
                _GEOMETRY.text_align_dim,
            ):
                raise ValueError(
                    "released provider requires visual/code/hidden/text geometry "
                    "65536/1536/2048/1152"
                )
            if not isinstance(layout, VisualVocabularyLayout):
                raise TypeError("released provider layout must be a visual layout")
            collator_layout = getattr(collator, "layout", None)
            if (
                not isinstance(collator_layout, VisualVocabularyLayout)
                or collator_layout != layout
            ):
                raise ValueError(
                    "provider and teacher-forcing collator layout must match"
                )
        self.planner = planner
        self.collator = collator
        self.layout = layout
        self.condition_dim = condition_dim
        self.visual_adapter = nn.Linear(int(codebook.shape[1]), condition_dim)
        self.hidden_adapter = nn.Linear(hidden_dim, condition_dim)
        self.position_encoder = LearnedPlanPositionEncoder(condition_dim)
        self.output_norm = nn.LayerNorm(condition_dim)

    @classmethod
    def _from_test_components(
        cls,
        *,
        planner: nn.Module,
        collator: GroundedPlannerCollator | Any,
        layout: VisualVocabularyLayout | Any,
        condition_dim: int,
    ) -> Qwen35GroundedPlanProvider:
        """Construct a reduced-geometry provider for unit tests only."""

        return cls(
            planner=planner,
            collator=collator,
            layout=layout,
            condition_dim=condition_dim,
            _enforce_released_geometry=False,
        )

    def _planner_device_and_refresh_visual_rows(self) -> torch.device:
        if isinstance(self.layout, VisualVocabularyLayout):
            rows = _current_visual_embedding_rows(self.planner, self.layout)
            registered = self.planner._parameters.get("visual_embedding_weight")
            if registered is None:
                self.planner.visual_embedding_weight = rows
            elif registered.data_ptr() != rows.data_ptr():
                raise RuntimeError(
                    "registered visual rows do not alias current Qwen embeddings"
                )
            return rows.device
        return self.planner.codebook.device

    @staticmethod
    def _move_batch_to_device(
        batch: Any,
        *,
        device: torch.device,
    ) -> Any:
        if not isinstance(batch, GroundedPlannerBatch):
            return batch
        updates = {
            item.name: getattr(batch, item.name).to(device=device)
            for item in fields(batch)
            if item.name != "qwen_inputs"
        }
        updates["qwen_inputs"] = {
            name: value.to(device=device)
            for name, value in batch.qwen_inputs.items()
        }
        return replace(batch, **updates)

    @staticmethod
    def _validate_images_and_instructions(
        current_images: Tensor,
        instructions: Sequence[str],
    ) -> int:
        if (
            not isinstance(current_images, Tensor)
            or current_images.ndim != 5
            or current_images.shape[1] != len(CAMERA_NAMES)
            or current_images.shape[2] != 3
        ):
            raise ValueError("current_images must have shape [B,2,3,H,W]")
        batch_size = int(current_images.shape[0])
        if batch_size <= 0:
            raise ValueError("current_images must contain at least one sample")
        if len(instructions) != batch_size:
            raise ValueError("images and instructions batch sizes must match")
        if any(type(value) is not str or not value for value in instructions):
            raise ValueError("instructions must contain nonempty strings")
        return batch_size

    def generate(
        self,
        current_images: Tensor,
        instructions: Sequence[str],
    ) -> GeneratedGroundedPlan:
        """Generate with a temporarily frozen/eval planner and restore all state."""

        batch_size = self._validate_images_and_instructions(
            current_images,
            instructions,
        )
        flat_images = current_images.reshape(
            batch_size * len(CAMERA_NAMES),
            3,
            *current_images.shape[-2:],
        )
        camera_names = tuple(
            camera for _ in range(batch_size) for camera in CAMERA_NAMES
        )
        flat_instructions = tuple(
            instruction for instruction in instructions for _ in CAMERA_NAMES
        )
        training_states = tuple(
            (module, module.training) for module in self.planner.modules()
        )
        rope_states = tuple(
            (module, module.rope_deltas)
            for module in self.planner.modules()
            if hasattr(module, "rope_deltas")
        )
        try:
            self.planner.eval()
            with torch.no_grad():
                flat_plan = generate_grounded_plan(
                    self.planner,
                    current_images=flat_images,
                    instructions=flat_instructions,
                    camera_names=camera_names,
                    layout=self.layout,
                    processor=self.collator.processor,
                )
                return unflatten_generated_plan(flat_plan, batch_size)
        finally:
            for module, rope_deltas in rope_states:
                module.rope_deltas = rope_deltas
            for module, training in training_states:
                module.training = training

    def teacher_force(
        self,
        current_images: Tensor,
        instructions: Sequence[str],
        targets: CachedPlannerTargets,
    ) -> GroundedPlannerOutput:
        """Run the differentiable Task 6/7 path with complete cached targets."""

        batch_size = self._validate_images_and_instructions(
            current_images,
            instructions,
        )
        if not isinstance(targets, CachedPlannerTargets):
            raise TypeError("targets must be complete CachedPlannerTargets")
        if targets.batch_size != batch_size:
            raise ValueError("images, instructions, and targets batch sizes must match")
        batch = self.collator.build_teacher_forced(
            current_images=current_images,
            instructions=instructions,
            targets=targets,
        )
        device = self._planner_device_and_refresh_visual_rows()
        batch = self._move_batch_to_device(batch, device=device)
        return self.planner(batch).unflatten_cameras(batch_size)

    def fuse(
        self,
        plan: GeneratedGroundedPlan | GroundedPlannerOutput,
        *,
        qwen_gradient_scale: float = 1.0,
    ) -> Tensor:
        """Perform the sole Qwen/TA-to-GE-Act condition-width conversion."""

        required = (
            "codes",
            "code_embeddings",
            "post_hidden",
            "fusion_gate",
            "times",
        )
        if any(not hasattr(plan, name) for name in required):
            raise TypeError("plan is missing grounded fusion fields")
        if not isinstance(plan.codes, Tensor):
            raise TypeError("plan codes must be a tensor")
        if plan.codes.ndim != 4:
            raise ValueError("plan codes must have shape [B,2,4,729]")
        if tuple(plan.codes.shape) != (
            plan.codes.shape[0],
            len(CAMERA_NAMES),
            _GEOMETRY.num_keyframes,
            _GEOMETRY.tokens_per_frame,
        ):
            raise ValueError("plan codes must have shape [B,2,4,729]")
        batch_size = int(plan.codes.shape[0])
        code_dim = self.visual_adapter.in_features
        hidden_dim = self.hidden_adapter.in_features
        shapes = {
            "code_embeddings": (batch_size, 2, 4, 729, code_dim),
            "post_hidden": (batch_size, 2, 4, 729, hidden_dim),
            "fusion_gate": (batch_size, 2, 4, 729, 1),
        }
        for name, shape in shapes.items():
            value = getattr(plan, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not value.dtype.is_floating_point or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must contain finite floating-point values")

        code_embeddings = scale_gradient(
            plan.code_embeddings,
            qwen_gradient_scale,
        )
        post_hidden = scale_gradient(plan.post_hidden, qwen_gradient_scale)
        fusion_gate = scale_gradient(plan.fusion_gate, qwen_gradient_scale)
        position_features = self.position_encoder(
            grid_size=(_GEOMETRY.grid_size, _GEOMETRY.grid_size),
            times=plan.times,
            batch_shape=plan.codes.shape[:3],
        )
        return self.output_norm(
            self.visual_adapter(code_embeddings)
            + fusion_gate * self.hidden_adapter(post_hidden)
            + position_features
        )
