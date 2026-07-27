from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from qwen35_baton.cli.visualize_attention import main as visualize_main
from qwen35_baton.provider import BatonSemanticPlan
from qwen35_baton.visualization import (
    render_attention_panels,
    summarize_query_cross_attention,
)


def _positions(batch_size: int = 1) -> torch.Tensor:
    centers = (torch.arange(16, dtype=torch.float32) + 0.5) / 16
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    grid = torch.stack((x, y), dim=-1).reshape(1, 1, 1, 256, 2)
    return grid.expand(batch_size, 2, 4, 256, 2).contiguous()


def _plan(*, sensitivity: torch.Tensor | None = None) -> BatonSemanticPlan:
    if sensitivity is None:
        sensitivity = torch.arange(2048, dtype=torch.float32).reshape(1, 2, 4, 256)
    maps = tuple(
        torch.full((1, 2, 1024, 1024), float(layer + 1))
        for layer in range(4)
    )
    return BatonSemanticPlan(
        tokens=torch.zeros(1, 2, 4, 256, 1024),
        future_indices=(0, 3, 5, 8),
        positions_xy=_positions(),
        cross_attention_maps=maps,
        instruction_sensitivity=sensitivity,
    )


def test_cross_attention_summary_averages_layer_and_head_then_sums_key_blocks() -> None:
    # Two layers x two heads. Their elementwise mean is the literal matrix below.
    mean = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    maps = tuple(
        torch.stack((mean - 2.0 + layer, mean + layer), dim=0)[
            None, None
        ]
        for layer in (0.0, 2.0)
    )

    summary = summarize_query_cross_attention(
        maps,
        num_frames=2,
        tokens_per_frame=2,
    )

    expected = torch.tensor([[[[7.0, 11.0], [23.0, 27.0]]]])
    torch.testing.assert_close(summary, expected)


def test_attention_visualization_writes_one_panel_per_camera_and_raw_npz(
    tmp_path: Path,
) -> None:
    sample = {
        "current_images": torch.stack(
            (
                torch.full((3, 24, 32), 32, dtype=torch.uint8),
                torch.full((3, 24, 32), 192, dtype=torch.uint8),
            )
        )[None],
        "instructions": ("pick up the blue cup",),
    }
    plan = _plan()

    paths = render_attention_panels(sample, plan, output_dir=tmp_path)

    assert [path.name for path in paths] == [
        "sample_000_main.png",
        "sample_000_wrist.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)
    for camera, path in zip(("main", "wrist"), paths, strict=True):
        with Image.open(path) as image:
            assert image.width > sample["current_images"].shape[-1] * 4
            assert image.info["camera"] == camera
            assert (
                image.info["heatmap_label"]
                == "instruction-conditioned query sensitivity"
            )
            assert image.info["aggregation"] == (
                "display: per-keyframe min-max; Query Tower: mean layer/head, "
                "sum keys per Qwen plan block"
            )
            assert "bounding" not in " ".join(image.info.values()).lower()

    archive_path = tmp_path / "sample_000_attention.npz"
    assert archive_path.is_file()
    with np.load(archive_path) as archive:
        assert set(archive.files) == {
            "instruction_sensitivity",
            "query_tower_frame_attention",
            "future_indices",
        }
        np.testing.assert_array_equal(
            archive["instruction_sensitivity"],
            plan.instruction_sensitivity[0].numpy(),
        )
        assert archive["query_tower_frame_attention"].shape == (2, 4, 4)
        np.testing.assert_array_equal(
            archive["future_indices"],
            np.array([0, 3, 5, 8], dtype=np.int64),
        )


def test_display_normalization_does_not_modify_raw_saved_sensitivity(
    tmp_path: Path,
) -> None:
    raw = torch.full((1, 2, 4, 256), 7.5)
    sample = {
        "current_images": torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
        "instructions": ("pick",),
    }

    render_attention_panels(sample, _plan(sensitivity=raw), output_dir=tmp_path)

    with np.load(tmp_path / "sample_000_attention.npz") as archive:
        np.testing.assert_array_equal(
            archive["instruction_sensitivity"],
            np.full((2, 4, 256), 7.5, dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("sample", "plan", "message"),
    [
        (
            {
                "current_images": torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
                "instructions": ("pick",),
            },
            _plan(sensitivity=None),
            "",
        ),
        (
            {
                "current_images": torch.zeros(1, 2, 3, 8, 8),
                "instructions": ("pick",),
            },
            _plan(),
            "uint8",
        ),
    ],
)
def test_attention_visualization_rejects_malformed_inputs(
    sample: dict[str, object],
    plan: BatonSemanticPlan,
    message: str,
    tmp_path: Path,
) -> None:
    if not message:
        plan = BatonSemanticPlan(
            tokens=plan.tokens,
            future_indices=plan.future_indices,
            positions_xy=plan.positions_xy,
            cross_attention_maps=plan.cross_attention_maps,
            instruction_sensitivity=None,
        )
        message = "instruction sensitivity"

    with pytest.raises((TypeError, ValueError), match=message):
        render_attention_panels(sample, plan, output_dir=tmp_path)


def test_visualization_cli_fails_closed_before_checkpoint_load_on_bad_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.npz"
    np.savez_compressed(
        input_path,
        current_images=np.zeros((2, 2, 3, 8, 8), dtype=np.uint8),
    )
    instructions = tmp_path / "instructions.json"
    counterfactuals = tmp_path / "counterfactuals.json"
    instructions.write_text('["pick", "place"]')
    counterfactuals.write_text('["wrong"]')

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("checkpoint loading must follow input validation")

    monkeypatch.setattr(
        "qwen35_baton.cli.visualize_attention.FrozenBatonPlanner.from_checkpoint",
        forbidden_load,
    )

    with pytest.raises(ValueError, match="batch sizes"):
        visualize_main(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--qwen-model-path",
                str(tmp_path / "qwen"),
                "--qwen-tokenizer-path",
                str(tmp_path / "tokenizer"),
                "--qwen-processor-path",
                str(tmp_path / "processor"),
                "--input-npz",
                str(input_path),
                "--instructions-json",
                str(instructions),
                "--counterfactual-instructions-json",
                str(counterfactuals),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
