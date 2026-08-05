from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _translated_features(
    *,
    frames: int = 4,
    grid: int = 27,
    dx: int = 1,
    dy: int = 0,
) -> torch.Tensor:
    width = grid + abs(dx) * (frames - 1)
    height = grid + abs(dy) * (frames - 1)
    generator = torch.Generator().manual_seed(7)
    canvas = torch.randn(height, width, 64, generator=generator)
    clips = []
    for frame in range(frames):
        x0 = (frames - 1 - frame) * max(dx, 0)
        y0 = (frames - 1 - frame) * max(dy, 0)
        clips.append(canvas[y0 : y0 + grid, x0 : x0 + grid])
    return torch.stack(clips).flatten(1, 2)


def test_cycle_consistent_tracking_recovers_translation() -> None:
    from qwen35_planx.temporal_grounding import track_keyframes

    features = _translated_features(frames=4, grid=27, dx=1, dy=0)
    tracks = track_keyframes(features, search_radius=4)

    assert tracks.flow.shape == (3, 729, 3)
    assert tracks.flow[:, :, 0].median().item() == 1
    assert tracks.flow[:, :, 1].abs().max().item() == 0
    assert tracks.flow[:, :, 2].median().item() > 0.9


def test_main_and_wrist_tracks_are_computed_independently() -> None:
    from qwen35_planx.temporal_grounding import track_keyframes

    main = _translated_features(dx=1)
    wrist = _translated_features(dx=0, dy=1)
    tracks = track_keyframes(torch.stack((main, wrist)), search_radius=4)

    assert tracks.flow.shape == (2, 3, 729, 3)
    assert tracks.flow[0, :, :, 0].median() == 1
    assert tracks.flow[0, :, :, 1].median() == 0
    assert tracks.flow[1, :, :, 0].median() == 0
    assert tracks.flow[1, :, :, 1].median() == 1


def test_forward_match_with_ambiguous_reverse_is_low_confidence() -> None:
    from qwen35_planx.temporal_grounding import track_keyframes

    before = torch.eye(32)[:25]
    before[0] = torch.tensor([0.31, 0.95] + [0.0] * 30)
    before[0] /= before[0].norm()
    after = torch.eye(32)[:25]
    after[6] = before[1]
    features = torch.stack((before, after))

    tracks = track_keyframes(features, search_radius=4)

    assert tracks.flow[0, 0, 2] < 0.6
    assert tracks.flow[0, 0, 2] > 0


def _pick_place_actions() -> torch.Tensor:
    actions = torch.zeros(15, 7)
    actions[:3, 6] = 1.0
    actions[3:10, 6] = 0.0
    actions[10:, 6] = 1.0
    return actions


def _pick_place_states() -> torch.Tensor:
    states = torch.zeros(15, 8)
    aperture = torch.full((15,), 0.08)
    aperture[3:10] = 0.01
    states[:, 6] = 0.5 * aperture
    states[:, 7] = -0.5 * aperture
    return states


def test_action_phases_detect_close_transport_and_release() -> None:
    from qwen35_planx.temporal_grounding import detect_action_phases

    phases = detect_action_phases(_pick_place_actions(), _pick_place_states())

    assert phases.source.shape == (15,)
    assert phases.transport.shape == (15,)
    assert phases.target.shape == (15,)
    torch.testing.assert_close(phases.source.sum(), torch.tensor(1.0))
    torch.testing.assert_close(phases.transport.sum(), torch.tensor(1.0))
    torch.testing.assert_close(phases.target.sum(), torch.tensor(1.0))
    assert phases.source.argmax().item() == 3
    assert phases.transport.argmax().item() in range(4, 10)
    assert phases.target.argmax().item() == 10
    assert phases.confidence == 1.0


def test_action_phase_confidence_requires_state_gripper_agreement() -> None:
    from qwen35_planx.temporal_grounding import detect_action_phases

    actions = _pick_place_actions()
    valid = detect_action_phases(actions, _pick_place_states())
    constant = detect_action_phases(actions, torch.zeros(15, 8))
    contradictory_states = torch.zeros(15, 8)
    contradictory_aperture = torch.full((15,), 0.01)
    contradictory_aperture[3:10] = 0.08
    contradictory_states[:, 6] = 0.5 * contradictory_aperture
    contradictory_states[:, 7] = -0.5 * contradictory_aperture
    contradictory = detect_action_phases(actions, contradictory_states)

    assert valid.confidence == 1.0
    assert constant.confidence == 0.0
    assert contradictory.confidence == 0.0
    uniform = torch.full((15,), 1.0 / 15)
    torch.testing.assert_close(constant.source, uniform)
    torch.testing.assert_close(contradictory.target, uniform)


def test_one_persistent_state_transition_lowers_phase_confidence() -> None:
    from qwen35_planx.temporal_grounding import detect_action_phases

    states = _pick_place_states()
    states[10:, 6:8] = states[9, 6:8]
    phases = detect_action_phases(_pick_place_actions(), states)

    assert phases.confidence == 0.5
    assert phases.source.argmax().item() == 3


@pytest.mark.parametrize(
    "actions",
    (
        torch.zeros(12, 6),
        torch.cat((torch.ones(12, 1), torch.zeros(12, 6)), dim=-1),
    ),
)
def test_missing_gripper_or_arm_only_motion_uses_visual_fallback(
    actions: torch.Tensor,
) -> None:
    from qwen35_planx.temporal_grounding import detect_action_phases

    phases = detect_action_phases(actions, torch.zeros(12, 8))

    uniform = torch.full((12,), 1.0 / 12)
    torch.testing.assert_close(phases.source, uniform)
    torch.testing.assert_close(phases.transport, uniform)
    torch.testing.assert_close(phases.target, uniform)
    assert phases.confidence == 0.0


def test_gripper_transition_must_persist_for_three_steps() -> None:
    from qwen35_planx.temporal_grounding import detect_action_phases

    actions = torch.zeros(12, 7)
    actions[:, 6] = 1.0
    actions[4:6, 6] = -1.0
    phases = detect_action_phases(actions, None)

    assert phases.confidence == 0.0


def test_absent_actions_use_state_length_for_visual_fallback() -> None:
    from qwen35_planx.temporal_grounding import detect_action_phases

    phases = detect_action_phases(None, torch.zeros(12, 8))

    assert phases.source.shape == (12,)
    assert phases.confidence == 0.0


def test_fusion_uses_fixed_geometric_weights_and_invalid_evidence_confidence() -> None:
    from qwen35_planx.temporal_grounding import fuse_hindsight_maps

    positions = torch.arange(1, 730, dtype=torch.float32)
    output = fuse_hindsight_maps(
        "source",
        text=positions,
        track=positions.flip(0),
        change=torch.ones(729),
        phase=torch.ones(729),
        confidences=(1.0, 1.0, 0.0, 0.0),
    )
    expected = torch.exp(
        (0.45 * torch.log(positions) + 0.30 * torch.log(positions.flip(0)))
        / 0.75
    )
    expected = expected / expected.sum()

    torch.testing.assert_close(output.map, expected)
    torch.testing.assert_close(output.map.sum(), torch.tensor(1.0))
    assert output.confidence > 0

    one_source = fuse_hindsight_maps(
        "source",
        text=positions,
        track=torch.zeros(729),
        change=torch.full((729,), torch.nan),
        phase=torch.zeros(729),
        confidences=(1.0, 1.0, 1.0, 0.0),
    )
    torch.testing.assert_close(one_source.map.sum(), torch.tensor(1.0))
    assert one_source.confidence == 0.0


def test_fusion_confidence_fails_closed_for_disjoint_valid_evidence() -> None:
    from qwen35_planx.temporal_grounding import fuse_hindsight_maps

    left = torch.zeros(729)
    left[:243] = 1
    right = torch.zeros(729)
    right[-243:] = 1

    aligned = fuse_hindsight_maps(
        "target",
        text=left,
        track=left,
        change=left,
        phase=left,
    )
    disjoint = fuse_hindsight_maps(
        "target",
        text=left,
        track=left,
        change=right,
        phase=right,
    )

    assert aligned.confidence > 0
    assert disjoint.confidence == 0.0
    assert torch.isfinite(disjoint.map).all()
    torch.testing.assert_close(disjoint.map.sum(), torch.tensor(1.0))


class _FakeDinoProcessor:
    def __call__(self, *, images: torch.Tensor, return_tensors: str):
        assert return_tensors == "pt"
        return {"pixel_values": images}


class _FakeDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.config = SimpleNamespace(num_register_tokens=0)

    def forward(self, *, pixel_values: torch.Tensor):
        batch = pixel_values.shape[0]
        patches = torch.linspace(
            0.0, 1.0, 24 * 24 * 8, device=pixel_values.device
        ).view(1, 24 * 24, 8)
        cls = torch.zeros(1, 1, 8, device=pixel_values.device)
        return SimpleNamespace(
            last_hidden_state=torch.cat((cls, patches), dim=1).expand(batch, -1, -1)
        )


def test_dino_teacher_resizes_features_to_cache_grid_and_freezes_model() -> None:
    from qwen35_planx.temporal_grounding import DinoTemporalTeacher

    teacher = DinoTemporalTeacher.from_components(
        model=_FakeDino(),
        processor=_FakeDinoProcessor(),
    )
    features = teacher.encode(torch.zeros(2, 4, 3, 256, 256), microbatch_size=3)

    assert features.shape == (2, 4, 729, 8)
    torch.testing.assert_close(features.norm(dim=-1), torch.ones(2, 4, 729))
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())
