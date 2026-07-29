from __future__ import annotations

import numpy as np
import pytest

from semantic_localization.wan_action_attention.single_frame_action_attention import (
    KEYFRAME_INDICES,
    NUM_FUTURE,
    NUM_PREVIOUS,
    WINDOW_END,
    WINDOW_START,
    aggregate_layer_maps,
    extract_frame_map,
    normalize_map,
    normalize_mean_std,
    normalize_q01_q99,
    pack_real_action_state,
    positive_gain,
    resolve_action_model_config,
)


def test_temporal_window_contains_four_memory_and_nine_future_frames() -> None:
    assert WINDOW_START == 77
    assert WINDOW_END == 90
    assert WINDOW_END - WINDOW_START == NUM_PREVIOUS + NUM_FUTURE == 13


def test_semantic_keyframes_match_joint_training_offsets() -> None:
    assert KEYFRAME_INDICES == (2, 4, 6, 8)


def test_normalize_q01_q99_maps_range_and_clips() -> None:
    values = np.array([[-1.0, 5.0], [2.0, 8.0]], dtype=np.float32)
    stats = {"q01": [0.0, 0.0], "q99": [4.0, 10.0]}

    actual = normalize_q01_q99(values, stats)

    np.testing.assert_allclose(
        actual,
        np.array([[-1.0, 0.0], [0.0, 0.6]], dtype=np.float32),
        atol=1e-5,
    )


def test_normalize_mean_std_matches_training_convention() -> None:
    values = np.array([[1.0, 5.0], [3.0, 9.0]], dtype=np.float32)
    stats = {"mean": [1.0, 1.0], "std": [2.0, 4.0]}

    actual = normalize_mean_std(values, stats)

    np.testing.assert_allclose(
        actual,
        np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_pack_real_action_state_builds_15d_training_tokens() -> None:
    actions = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    states = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    action_stats = {"mean": [0.0] * 7, "std": [1.0] * 7}
    state_stats = {"mean": [0.0] * 8, "std": [1.0] * 8}

    packed, history = pack_real_action_state(
        actions,
        states,
        frame_index=1,
        horizon=3,
        action_stats=action_stats,
        state_stats=state_stats,
    )

    assert packed.shape == (3, 15)
    assert history.shape == (15,)
    np.testing.assert_allclose(packed[:, :7], actions[1:4], atol=3e-5)
    np.testing.assert_allclose(packed[:, 7:], states[1:4], atol=3e-5)
    np.testing.assert_array_equal(history[:7], np.zeros(7))
    np.testing.assert_allclose(history[7:], states[1], atol=2e-5)


def test_resolve_action_model_config_restores_kwargs_lost_from_checkpoint() -> None:
    checkpoint_config = {"action_expert": True, "num_layers": 28}
    training_config = {
        "diffusion_model": {
            "config": {
                "action_in_channels": 15,
                "action_out_channels": 15,
                "action_num_attention_heads": 16,
                "action_attention_head_dim": 32,
            }
        }
    }

    resolved = resolve_action_model_config(checkpoint_config, training_config)

    assert resolved["action_in_channels"] == 15
    assert resolved["action_out_channels"] == 15
    assert resolved["action_num_attention_heads"] == 16
    assert resolved["action_attention_head_dim"] == 32
    assert resolved["num_layers"] == 28


def test_normalize_map_rejects_nonfinite_and_constant_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        normalize_map(np.array([[np.nan, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="constant"):
        normalize_map(np.ones((2, 2), dtype=np.float32))


def test_positive_gain_keeps_only_added_focus() -> None:
    plan_off = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    plan_on = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)

    gain = positive_gain(plan_off, plan_on)

    assert gain[0, 1] == 0.0
    assert gain[1, 1] > 0.0


def test_extract_frame_map_selects_requested_view_and_time() -> None:
    values = np.arange(2 * 3 * 2 * 2, dtype=np.float32)

    actual = extract_frame_map(
        values,
        view=1,
        time_index=2,
        temporal=3,
        height=2,
        width=2,
    )

    np.testing.assert_array_equal(actual, values.reshape(2, 3, 2, 2)[1, 2])


def test_extract_frame_map_rejects_shape_or_index_mismatch() -> None:
    values = np.arange(24, dtype=np.float32)
    with pytest.raises(ValueError, match="expected"):
        extract_frame_map(
            values[:-1],
            view=1,
            time_index=2,
            temporal=3,
            height=2,
            width=2,
        )
    with pytest.raises(IndexError, match="view"):
        extract_frame_map(
            values,
            view=2,
            time_index=2,
            temporal=3,
            height=2,
            width=2,
        )
    with pytest.raises(IndexError, match="time_index"):
        extract_frame_map(
            values,
            view=1,
            time_index=3,
            temporal=3,
            height=2,
            width=2,
        )


def test_aggregate_layer_maps_validates_and_averages() -> None:
    first = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    second = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    actual = aggregate_layer_maps({0: first, 1: second})

    np.testing.assert_allclose(actual, (first + second) / 2.0)
    with pytest.raises(ValueError, match="at least one"):
        aggregate_layer_maps({})
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_layer_maps({0: np.array([[np.nan, 1.0]])})
    with pytest.raises(ValueError, match="constant"):
        aggregate_layer_maps({0: np.ones((2, 2), dtype=np.float32)})
    with pytest.raises(ValueError, match="same shape"):
        aggregate_layer_maps({0: first, 1: np.ones((3, 3), dtype=np.float32)})
