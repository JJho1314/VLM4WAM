from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPOSITORY_ROOT / "ge_act"
for path in (REPOSITORY_ROOT, GE_ACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ge_act.models.ltx_models import vlm_semantic_planner as planner_module  # noqa: E402
from ge_act.models.ltx_models.vlm_semantic_planner import (  # noqa: E402
    FrozenDualCameraVLMPlanner,
    validate_dual_camera_planner_metadata,
)


class FakeDualWrapper(nn.Module):
    def __init__(self, *, output_shape: tuple[int, ...] = (2, 2, 256, 1024)):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.output_shape = output_shape
        self.received: dict[str, Any] | None = None

    def predict_current_future_plans(self, **inputs: Any) -> dict[str, torch.Tensor]:
        self.received = inputs
        values = torch.arange(self.output_shape[1], dtype=torch.float32).reshape(
            1, self.output_shape[1], 1, 1
        )
        return {
            "future_dino": values.expand(self.output_shape).clone(),
        }


class FakeProcessor:
    pass


def valid_metadata() -> dict[str, Any]:
    return {
        "planner_input_layout": "separate_camera_images",
        "camera_names": ["main", "wrist"],
        "num_camera_views": 2,
        "camera_head_sharing": "shared_head_per_view_image_context",
        "semantic_output_layout": "batch_view_token_feature",
        "semantic_teacher": "siglip2-large-patch16-256",
        "future_keyframe_offsets": [8],
        "num_keyframes": 1,
        "grid_size": 16,
        "semantic_dim": 1024,
        "target_tokens": 256,
        "video_target_type": "siglip2",
        "plan_token_strings": [f"<|sem_plan_{index}|>" for index in range(256)],
    }


def fake_input_builder(
    _processor: Any,
    image_pairs: list[tuple[Image.Image, Image.Image]],
    instructions: list[str],
    plan_tokens: list[str],
) -> dict[str, Any]:
    assert len(image_pairs) == len(instructions)
    assert all(len(pair) == 2 for pair in image_pairs)
    assert len(plan_tokens) == 256
    return {
        "input_ids": torch.arange(len(instructions)).reshape(-1, 1),
        "image_pairs": image_pairs,
    }


def test_frozen_provider_returns_one_ordered_future_grid_per_camera() -> None:
    wrapper = FakeDualWrapper()
    provider = FrozenDualCameraVLMPlanner.from_components(
        wrapper=wrapper,
        processor=FakeProcessor(),
        input_builder=fake_input_builder,
        input_mover=lambda value: value,
        plan_tokens=[f"<|sem_plan_{index}|>" for index in range(256)],
        device="cpu",
    )

    plan = provider.predict(
        torch.zeros(2, 2, 3, 8, 8),
        ["pick", "place"],
    )

    assert plan.semantic_tokens.shape == (2, 2, 1, 256, 1024)
    assert plan.times.shape == (4, 1)
    torch.testing.assert_close(plan.times, torch.ones(4, 1))
    assert plan.semantic_tokens[:, 0].eq(0).all()
    assert plan.semantic_tokens[:, 1].eq(1).all()
    assert wrapper.received is not None
    assert all(not parameter.requires_grad for parameter in provider.wrapper.parameters())
    assert not provider.wrapper.training


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("planner_input_layout", "fastwam_current_multicamera_composite"),
        ("camera_names", ["wrist", "main"]),
        ("num_camera_views", 1),
        ("future_keyframe_offsets", [0, 8]),
        ("num_keyframes", 2),
        ("grid_size", 9),
        ("semantic_dim", 1152),
        ("target_tokens", 81),
        ("video_target_type", "dinov3"),
    ],
)
def test_provider_rejects_incompatible_checkpoint_metadata(
    field: str,
    value: Any,
) -> None:
    metadata = valid_metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match=field):
        validate_dual_camera_planner_metadata(metadata)


def test_provider_validates_metadata_before_allocating_qwen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = valid_metadata()
    metadata["num_camera_views"] = 1
    (tmp_path / "planner_meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    called = False

    def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        FrozenDualCameraVLMPlanner,
        "_load_checkpoint_components",
        staticmethod(fail_if_called),
        raising=False,
    )

    with pytest.raises(ValueError, match="num_camera_views"):
        FrozenDualCameraVLMPlanner.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )

    assert not called


def test_provider_rejects_invalid_input_and_output_shapes() -> None:
    provider = FrozenDualCameraVLMPlanner.from_components(
        wrapper=FakeDualWrapper(output_shape=(1, 2, 81, 1024)),
        processor=FakeProcessor(),
        input_builder=fake_input_builder,
        input_mover=lambda value: value,
        plan_tokens=[f"<|sem_plan_{index}|>" for index in range(256)],
        device="cpu",
    )

    with pytest.raises(ValueError, match=r"\[B,2,3,H,W\]"):
        provider.predict(torch.zeros(1, 1, 3, 8, 8), ["pick"])
    with pytest.raises(RuntimeError, match="future_siglip"):
        provider.predict(torch.zeros(1, 2, 3, 8, 8), ["pick"])


def test_grounded_provider_validates_checkpoint_and_cache_before_qwen_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocation_called = False

    def reject_cache(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("planner/cache hash mismatch")

    def record_allocation(*_args: Any, **_kwargs: Any) -> None:
        nonlocal allocation_called
        allocation_called = True

    monkeypatch.setattr(
        planner_module,
        "_validate_grounded_checkpoint_contract",
        reject_cache,
        raising=False,
    )
    monkeypatch.setattr(
        planner_module,
        "_load_grounded_checkpoint_components",
        record_allocation,
        raising=False,
    )

    with pytest.raises(ValueError, match="planner/cache hash mismatch"):
        planner_module.load_qwen35_grounded_provider(
            tmp_path,
            hindsight_cache_hash="cache-hash",
            cache_dir=tmp_path / "cache",
            dataset=object(),
            condition_dim=8,
            device="cpu",
            dtype=torch.float32,
        )

    assert not allocation_called
