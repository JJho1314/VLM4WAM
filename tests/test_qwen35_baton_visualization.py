from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import qwen35_baton.provider as provider_module
from qwen35_baton.cli.visualize_attention import (
    _parser as visualization_parser,
    main as visualize_main,
)
from qwen35_baton.provider import BatonSemanticPlan
from qwen35_baton.visualization import (
    query_cross_attention_focus,
    render_attention_panels,
    summarize_query_cross_attention,
)


def _positions(batch_size: int = 1) -> torch.Tensor:
    centers = (torch.arange(16, dtype=torch.float32) + 0.5) / 16
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    grid = torch.stack((x, y), dim=-1).reshape(1, 1, 1, 256, 2)
    return grid.expand(batch_size, 2, 4, 256, 2).contiguous()


def _attention_maps() -> tuple[torch.Tensor, ...]:
    keys = torch.arange(1024, dtype=torch.float32)
    queries = torch.arange(1024, dtype=torch.float32)
    logits = -((queries[:, None] - keys[None]) / 32.0).square()
    attention = torch.softmax(logits, dim=-1)
    return (attention[None, None].expand(1, 2, -1, -1).contiguous(),)


def _plan(*, with_attention: bool = True) -> BatonSemanticPlan:
    return BatonSemanticPlan(
        tokens=torch.zeros(1, 2, 4, 256, 1024),
        future_indices=(0, 3, 5, 8),
        positions_xy=_positions(),
        cross_attention_maps=_attention_maps() if with_attention else None,
    )


def test_cross_attention_summary_averages_head_then_sums_key_blocks() -> None:
    mean = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    maps = (
        torch.stack((mean - 1.0, mean + 1.0), dim=0)[None, None],
    )

    summary = summarize_query_cross_attention(
        maps,
        num_frames=2,
        tokens_per_frame=2,
    )

    expected = torch.tensor([[[[7.0, 11.0], [23.0, 27.0]]]])
    torch.testing.assert_close(summary, expected)


def test_query_focus_is_normalized_per_query_attention_entropy() -> None:
    uniform = torch.full((1, 2, 8, 8), 1.0 / 8)
    peaked = uniform.clone()
    peaked[:, :, 0] = 0
    peaked[:, :, 0, 0] = 1

    focus = query_cross_attention_focus(
        (peaked,),
        num_frames=2,
        tokens_per_frame=4,
    )

    assert focus.shape == (1, 2, 2, 4)
    torch.testing.assert_close(focus[:, :, 0, 0], torch.ones(1, 2))
    torch.testing.assert_close(
        focus[:, :, 0, 1:],
        torch.zeros(1, 2, 3),
        atol=1e-6,
        rtol=0,
    )


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

    paths = render_attention_panels(sample, _plan(), output_dir=tmp_path)

    assert [path.name for path in paths] == [
        "sample_000_main.png",
        "sample_000_wrist.png",
    ]
    for camera, path in zip(("main", "wrist"), paths, strict=True):
        with Image.open(path) as image:
            assert image.width > sample["current_images"].shape[-1] * 4
            assert image.info["camera"] == camera
            assert image.info["heatmap_label"] == "VLM planner cross-attention focus"
            assert "normalized query entropy" in image.info["aggregation"]

    with np.load(tmp_path / "sample_000_attention.npz") as archive:
        assert set(archive.files) == {
            "query_attention_focus",
            "query_tower_frame_attention",
            "future_indices",
        }
        assert archive["query_attention_focus"].shape == (2, 4, 256)
        assert archive["query_tower_frame_attention"].shape == (2, 4, 4)
        np.testing.assert_array_equal(
            archive["future_indices"],
            np.array([0, 3, 5, 8], dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("sample", "plan", "message"),
    [
        (
            {
                "current_images": torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
                "instructions": ("pick",),
            },
            _plan(with_attention=False),
            "cross-attention",
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
    with pytest.raises((TypeError, ValueError), match=message):
        render_attention_panels(sample, plan, output_dir=tmp_path)


def _cli_args(tmp_path: Path, instructions: Path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--qwen-model-path",
        str(tmp_path / "qwen"),
        "--qwen-tokenizer-path",
        str(tmp_path / "tokenizer"),
        "--qwen-processor-path",
        str(tmp_path / "processor"),
        "--siglip2-model-path",
        str(tmp_path / "siglip2"),
        "--expected-planner-topology",
        str(tmp_path / "topology.json"),
        "--input-npz",
        str(tmp_path / "input.npz"),
        "--instructions-json",
        str(instructions),
        "--output-dir",
        str(tmp_path / "output"),
    ]


def test_visualization_cli_validates_batch_before_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np.savez_compressed(
        tmp_path / "input.npz",
        current_images=np.zeros((2, 2, 3, 8, 8), dtype=np.uint8),
    )
    instructions = tmp_path / "instructions.json"
    instructions.write_text('["pick"]')
    monkeypatch.setattr(
        "qwen35_baton.cli.visualize_attention.FrozenBatonPlanner.from_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint loading must follow input validation")
        ),
    )

    with pytest.raises(ValueError, match="batch sizes"):
        visualize_main(_cli_args(tmp_path, instructions))


def test_visualization_cli_rejects_float16_dtype_choice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        visualization_parser().parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--qwen-model-path",
                "qwen",
                "--qwen-tokenizer-path",
                "tokenizer",
                "--qwen-processor-path",
                "processor",
                "--siglip2-model-path",
                "siglip2",
                "--input-npz",
                "input.npz",
                "--instructions-json",
                "instructions.json",
                "--output-dir",
                "output",
                "--dtype",
                "float16",
            ]
        )

    assert "invalid choice" in capsys.readouterr().err


def test_visualization_cli_rejects_indexed_cpu_before_checkpoint_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np.savez_compressed(
        tmp_path / "input.npz",
        current_images=np.zeros((1, 2, 3, 8, 8), dtype=np.uint8),
    )
    instructions = tmp_path / "instructions.json"
    instructions.write_text('["pick"]')
    monkeypatch.setattr(
        provider_module,
        "_validate_checkpoint_envelope",
        lambda _: (_ for _ in ()).throw(
            AssertionError("device validation must precede checkpoint access")
        ),
    )

    with pytest.raises(ValueError, match="canonical CPU device"):
        visualize_main(
            [
                *_cli_args(tmp_path, instructions),
                "--device",
                "cpu:0",
                "--dtype",
                "fp32",
            ]
        )
