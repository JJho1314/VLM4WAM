from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


GE_ACT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from models.pipeline.custom_pipeline import prepare_pipeline_semantic_conditioning


def test_pipeline_semantics_follow_cfg_batch_order() -> None:
    plan = torch.arange(2 * 2 * 8 * 3, dtype=torch.float32).reshape(2, 2, 8, 3)
    times = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(4, 4)
    mask = torch.tensor([1.0, 0.0])

    cfg_plan, cfg_times, cfg_mask = prepare_pipeline_semantic_conditioning(
        semantic_plan=plan,
        semantic_plan_times=times,
        semantic_condition_mask=mask,
        batch_size=2,
        n_view=2,
        num_keyframes=4,
        do_classifier_free_guidance=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cfg_plan.shape == (4, 2, 8, 3)
    torch.testing.assert_close(cfg_plan[:2], plan)
    torch.testing.assert_close(cfg_plan[2:], plan)
    assert cfg_times.shape == (8, 4)
    torch.testing.assert_close(cfg_times[:4], times)
    torch.testing.assert_close(cfg_times[4:], times)
    assert cfg_mask.tolist() == [1, 1, 0, 0, 1, 1, 0, 0]


def test_pipeline_rejects_misaligned_camera_or_token_layout() -> None:
    with pytest.raises(ValueError, match="batch/view"):
        prepare_pipeline_semantic_conditioning(
            semantic_plan=torch.randn(1, 3, 1024, 8),
            semantic_plan_times=torch.randn(2, 4),
            semantic_condition_mask=None,
            batch_size=1,
            n_view=2,
            num_keyframes=4,
            do_classifier_free_guidance=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="divisible"):
        prepare_pipeline_semantic_conditioning(
            semantic_plan=torch.randn(1, 2, 1023, 8),
            semantic_plan_times=torch.randn(2, 4),
            semantic_condition_mask=None,
            batch_size=1,
            n_view=2,
            num_keyframes=4,
            do_classifier_free_guidance=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_pipeline_allows_explicit_no_semantic_ablation() -> None:
    assert prepare_pipeline_semantic_conditioning(
        semantic_plan=None,
        semantic_plan_times=None,
        semantic_condition_mask=None,
        batch_size=1,
        n_view=2,
        num_keyframes=4,
        do_classifier_free_guidance=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) == (None, None, None)
