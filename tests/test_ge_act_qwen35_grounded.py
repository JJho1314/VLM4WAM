from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPOSITORY_ROOT / "ge_act"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from models.ltx_models.ltx_attention_processor import Attention  # noqa: E402
from models.ltx_models.semantic_conditioning import SemanticContextAdapter  # noqa: E402
from models.ltx_models import transformer_ltx_multiview as transformer_module  # noqa: E402
from models.ltx_models.transformer_ltx_multiview import (  # noqa: E402
    LTXVideoSemanticAttentionProcessor2_0,
    LTXVideoTransformer3DModel,
    LTXVideoTransformerBlock,
)


def _write_joint_hdf5_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    shard_path = root / "shard_00000.h5"
    episode_key = "libero_goal:000000"
    length = 80
    with h5py.File(shard_path, "w") as handle:
        group = handle.create_group(f"episodes/{episode_key}")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(
            "caption",
            data="pick up the red mug",
            dtype=string_dtype,
        )
        group.create_dataset("domain", data="libero_goal", dtype=string_dtype)
        group.create_dataset("episode_index", data=0, dtype=np.int64)
        group.create_dataset("length", data=length, dtype=np.int64)
        pixels = np.arange(length, dtype=np.uint8)[:, None, None, None]
        group.create_dataset(
            "rgb_main",
            data=np.broadcast_to(pixels, (length, 256, 256, 3)),
        )
        group.create_dataset(
            "rgb_wrist",
            data=np.broadcast_to(pixels + 80, (length, 256, 256, 3)),
        )
        group.create_dataset(
            "action",
            data=np.arange(length * 7, dtype=np.float32).reshape(length, 7),
        )
        group.create_dataset(
            "state",
            data=np.arange(length * 8, dtype=np.float32).reshape(length, 8),
        )
    manifest = {
        "schema_version": 1,
        "camera_names": ["main", "wrist"],
        "image_size": [256, 256],
        "source_fps": 20,
        "n_previous": 4,
        "chunk": 9,
        "action_chunk": 36,
        "action_type": "absolute",
        "action_space": "eef",
        "compression": "none",
        "source_roots": [str(root / "source")],
        "datasets": {
            "rgb_main": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
            "rgb_wrist": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
            "action": {"width": 7, "dtype": "float32"},
            "state": {"width": 8, "dtype": "float32"},
        },
        "converter_fingerprint": "a" * 64,
        "episodes": [
            {
                "key": episode_key,
                "shard": shard_path.name,
                "group": f"episodes/{episode_key}",
                "caption": "pick up the red mug",
                "domain": "libero_goal",
                "episode_index": 0,
                "length": length,
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats = {
        "libero_goal_eef": {
            "mean": [10.0] * 7,
            "std": [2.0] * 7,
        },
        "libero_goal_state_eef": {
            "mean": [20.0] * 8,
            "std": [4.0] * 8,
        },
    }
    stats_path = root / "stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return manifest_path, stats_path


def _write_joint_cache(
    root: Path,
    manifest_path: Path,
) -> tuple[Path, object]:
    from qwen35_planx.config import HindsightCacheMetadata
    from qwen35_planx.hashing import sha256_file, sha256_json
    from qwen35_planx.hindsight_data import build_fixed_windows
    from qwen35_planx.hindsight_schema import (
        HindsightShardWriter,
        finalize_hindsight_cache,
    )

    window = build_fixed_windows(
        manifest_path,
        split_seed=42,
        window_stride=80,
        sample_n_frames=500,
    )[0]
    metadata = HindsightCacheMetadata(
        format_version=1,
        hdf5_manifest_hash=sha256_file(manifest_path),
        window_manifest_hash=sha256_json([window.to_dict()]),
        instruction_parser_hash="parser-hash",
        ta_tok_hash="ta-hash",
        siglip2_hash="siglip-hash",
        dinov3_hash="dino-hash",
        preprocessing_hash="preprocessing-hash",
    )
    arrays = {
        "codes": np.arange(2 * 4 * 729, dtype=np.int64).reshape(1, 2, 4, 729),
        "relevance_q": np.ones((1, 2, 4, 3, 729), dtype=np.uint8),
        "relevance_scale": np.ones((1, 2, 4, 3), dtype=np.float16),
        "confidence": np.full((1, 2, 4, 3), 0.75, dtype=np.float16),
        "flow": np.zeros((1, 2, 3, 729, 3), dtype=np.float16),
        "phrase_embeddings": np.ones((1, 3, 1152), dtype=np.float16),
    }
    shard = root / "cache_shards" / "episode.npz"
    HindsightShardWriter(shard, metadata=metadata).write([window], **arrays)
    cache_dir = root / "cache"
    finalize_hindsight_cache(
        cache_dir,
        shard_paths=[shard],
        metadata=metadata,
        expected_records=[window],
    )
    return cache_dir, window


def _deterministic_semantic_attention() -> Attention:
    attention = Attention(
        query_dim=2,
        cross_attention_dim=2,
        heads=1,
        kv_heads=1,
        dim_head=2,
        bias=False,
        out_bias=False,
        qk_norm=None,
        processor=LTXVideoSemanticAttentionProcessor2_0(),
    )
    with torch.no_grad():
        attention.to_q.weight.zero_()
        attention.to_k.weight.zero_()
        attention.to_v.weight.copy_(torch.eye(2))
        attention.to_out[0].weight.copy_(torch.eye(2))
    return attention


def test_adapter_maps_compressed_xy_positions_and_flattens_camera_samples() -> None:
    adapter = SemanticContextAdapter(
        input_dim=4,
        hidden_dim=6,
        coordinate_dim=3,
        num_views=2,
    )
    tokens = torch.randn(1, 2, 2, 96, 4, requires_grad=True)
    positions = torch.zeros(1, 2, 2, 96, 2, requires_grad=True)
    with torch.no_grad():
        positions[0, 0, 0, 0] = torch.tensor([0.25, 0.75])
        positions[0, 1, 1, 95] = torch.tensor([1.0, 0.0])
    mask = torch.ones(1, 2, 2, 96, dtype=torch.bool)
    mask[0, 0, 1, 3] = False
    relevance = torch.linspace(0.01, 1.0, 1 * 2 * 2 * 96).reshape(1, 2, 2, 96)
    times = torch.tensor([[[0.25, 0.75], [0.5, 1.0]]])

    context = adapter(
        tokens,
        semantic_plan_times=times,
        latent_height=9,
        latent_width=5,
        latent_num_frames=5,
        semantic_positions_xy=positions,
        semantic_token_mask=mask,
        semantic_relevance=relevance,
    )

    assert context.hidden_states.shape == (2, 2 * 96, 6)
    assert context.positions.shape == (2, 2 * 96, 3)
    assert context.key_mask.shape == (2, 2 * 96)
    assert context.relevance.shape == (2, 2 * 96)
    torch.testing.assert_close(context.positions[0, 0], torch.tensor([1.0, 6.0, 1.0]))
    torch.testing.assert_close(context.positions[1, -1], torch.tensor([4.0, 0.0, 4.0]))
    assert not context.key_mask[0, 96 + 3]
    torch.testing.assert_close(context.relevance, relevance.reshape(2, 2 * 96))

    context.hidden_states.sum().backward()
    assert tokens.grad is not None
    assert positions.grad is not None
    assert torch.count_nonzero(positions.grad) > 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "semantic_positions_xy",
            torch.zeros(1, 2, 2, 95, 2),
            "semantic_positions_xy must have shape",
        ),
        (
            "semantic_token_mask",
            torch.ones(1, 2, 2, 96),
            "boolean",
        ),
        (
            "semantic_relevance",
            torch.full((1, 2, 2, 96), -0.1),
            "non-negative",
        ),
    ],
)
def test_adapter_rejects_invalid_grounding_fields(
    field: str,
    value: torch.Tensor,
    error: str,
) -> None:
    kwargs = {
        "semantic_positions_xy": torch.zeros(1, 2, 2, 96, 2),
        "semantic_token_mask": torch.ones(1, 2, 2, 96, dtype=torch.bool),
        "semantic_relevance": torch.ones(1, 2, 2, 96),
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        SemanticContextAdapter(
            input_dim=4,
            hidden_dim=6,
            coordinate_dim=3,
            num_views=2,
        )(
            torch.randn(1, 2, 2, 96, 4),
            semantic_plan_times=torch.ones(2, 2),
            latent_height=4,
            latent_width=4,
            **kwargs,
        )


def test_zero_bias_gate_uses_baseline_sdpa_and_matches_bitwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _deterministic_semantic_attention()
    queries = torch.randn(2, 3, 2)
    context = torch.randn(2, 4, 2)
    attention_masks: list[torch.Tensor | None] = []
    scaled_dot_product_attention = transformer_module.F.scaled_dot_product_attention

    def record_attention_mask(*args, **kwargs):
        attention_masks.append(kwargs.get("attn_mask"))
        return scaled_dot_product_attention(*args, **kwargs)

    monkeypatch.setattr(
        transformer_module.F,
        "scaled_dot_product_attention",
        record_attention_mask,
    )

    expected = attention(queries, encoder_hidden_states=context)
    attention_masks.clear()
    actual = attention(
        queries,
        encoder_hidden_states=context,
        relevance=torch.tensor([[1.0, 0.5, 0.01, 0.0]]).repeat(2, 1),
    )

    assert attention.processor.raw_semantic_bias_gate.item() == 0.0
    assert len(attention_masks) == 2
    assert attention_masks[0] is None
    assert attention_masks[1] is not None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_bias_gate_matches_cuda_mixed_precision_bitwise(
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    attention = _deterministic_semantic_attention().to(device="cuda", dtype=dtype)
    queries = torch.randn(2, 7, 2, device="cuda", dtype=dtype)
    context = torch.randn(2, 11, 2, device="cuda", dtype=dtype)
    relevance = torch.linspace(
        0.01,
        1.0,
        2 * 11,
        device="cuda",
        dtype=dtype,
    ).reshape(2, 11)

    expected = attention(queries, encoder_hidden_states=context)
    actual = attention(
        queries,
        encoder_hidden_states=context,
        relevance=relevance,
    )

    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    gate_grad = attention.processor.raw_semantic_bias_gate.grad
    assert gate_grad is not None
    assert torch.count_nonzero(gate_grad) > 0


def test_bias_is_bounded_and_prefers_the_more_relevant_key() -> None:
    attention = _deterministic_semantic_attention()
    attention.processor.raw_semantic_bias_gate.data.fill_(10.0)

    output = attention(
        torch.zeros(1, 1, 2),
        encoder_hidden_states=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        relevance=torch.tensor([[1.0, 0.01]]),
    )

    assert 1.99 < attention.processor.semantic_bias_gate.item() <= 2.0
    assert output[0, 0, 0] > output[0, 0, 1]


@pytest.mark.parametrize("raw_gate", [0.0, 1.0])
def test_padding_mask_blocks_keys_and_all_masked_context_is_finite(
    raw_gate: float,
) -> None:
    attention = _deterministic_semantic_attention()
    attention.processor.raw_semantic_bias_gate.data.fill_(raw_gate)
    context = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    output = attention(
        torch.zeros(1, 1, 2),
        encoder_hidden_states=context,
        attention_mask=torch.tensor([[False, True]]),
        relevance=torch.ones(1, 2),
    )
    all_masked = attention(
        torch.zeros(1, 1, 2),
        encoder_hidden_states=context,
        attention_mask=torch.zeros(1, 2, dtype=torch.bool),
        relevance=torch.ones(1, 2),
    )

    torch.testing.assert_close(output, torch.tensor([[[0.0, 1.0]]]))
    assert torch.isfinite(all_masked).all()
    torch.testing.assert_close(all_masked, torch.zeros_like(all_masked))


def test_relevance_bias_keeps_camera_batches_isolated_and_differentiable() -> None:
    attention = _deterministic_semantic_attention()
    attention.processor.raw_semantic_bias_gate.data.fill_(0.5)
    queries = torch.zeros(2, 1, 2)
    context = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    relevance = torch.tensor(
        [[1.0, 0.01], [1.0, 0.01]],
        requires_grad=True,
    )

    baseline = attention(
        queries,
        encoder_hidden_states=context,
        relevance=relevance,
    )
    changed_relevance = relevance.detach().clone()
    changed_relevance[1] = torch.tensor([0.01, 1.0])
    changed = attention(
        queries,
        encoder_hidden_states=context,
        relevance=changed_relevance,
    )

    torch.testing.assert_close(baseline[0], changed[0], rtol=0, atol=0)
    assert not torch.allclose(baseline[1], changed[1])
    baseline.sum().backward()
    assert relevance.grad is not None
    assert torch.count_nonzero(relevance.grad) > 0
    assert attention.processor.raw_semantic_bias_gate.grad is not None
    assert torch.count_nonzero(attention.processor.raw_semantic_bias_gate.grad) > 0


def test_old_semantic_attention_state_dict_loads_with_zero_bias_gate() -> None:
    block = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        semantic_cross_attention=True,
        semantic_adaln_rank=4,
    )
    old_state_dict = {
        key: value
        for key, value in block.state_dict().items()
        if not key.endswith("raw_semantic_bias_gate")
    }
    restored = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        semantic_cross_attention=True,
        semantic_adaln_rank=4,
    )

    restored.load_state_dict(old_state_dict, strict=True)

    assert restored.semantic_attn.processor.raw_semantic_bias_gate.item() == 0.0


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_untouched_zero_residual_learns_bias_gate_on_first_backward(
    gradient_checkpointing: bool,
) -> None:
    torch.manual_seed(29)
    model = LTXVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        caption_channels=8,
        num_layers=1,
        semantic_plan_context=True,
        semantic_plan_in_dim=8,
        semantic_plan_coordinate_dim=4,
        semantic_plan_num_keyframes=2,
        semantic_plan_num_views=2,
        semantic_plan_cross_attention_blocks=(0,),
        semantic_plan_adaln_rank=4,
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    block = model.transformer_blocks[0]
    assert torch.count_nonzero(block.semantic_modulation[-1].weight) == 0
    assert block.semantic_attn.processor.raw_semantic_bias_gate.item() == 0.0

    common_inputs = {
        "hidden_states": torch.randn(2, 2, 4),
        "encoder_hidden_states": torch.randn(1, 3, 8),
        "timestep": torch.ones(2, 2),
        "encoder_attention_mask": torch.ones(1, 3),
        "n_view": 2,
        "rope_interpolation_scale": (1.6, 32.0, 32.0),
        "num_frames": 2,
        "height": 1,
        "width": 1,
        "semantic_plan": torch.randn(1, 2, 2, 3, 8),
        "semantic_plan_times": torch.tensor([[0.5, 1.0]]).repeat(2, 1),
        "semantic_plan_positions": torch.rand(1, 2, 2, 3, 2),
        "return_dict": False,
    }
    with torch.no_grad():
        expected = model(**common_inputs)[0]["video"]

    semantic_relevance = torch.tensor(
        [[[[1.0, 0.4, 0.2], [0.8, 0.3, 0.1]]] * 2],
        requires_grad=True,
    )
    actual = model(
        **common_inputs,
        semantic_plan_relevance=semantic_relevance,
    )[0]["video"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()

    gate_grad = block.semantic_attn.processor.raw_semantic_bias_gate.grad
    assert gate_grad is not None
    assert torch.count_nonzero(gate_grad) > 0
    for projection in (
        block.semantic_attn.to_q,
        block.semantic_attn.to_k,
        block.semantic_attn.to_v,
        block.semantic_attn.to_out[0],
    ):
        projection_grad = projection.weight.grad
        assert projection_grad is None or torch.count_nonzero(projection_grad) == 0


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_transformer_threads_grounding_to_every_semantic_block(
    gradient_checkpointing: bool,
) -> None:
    torch.manual_seed(23)
    model = LTXVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        caption_channels=8,
        num_layers=2,
        semantic_plan_context=True,
        semantic_plan_in_dim=8,
        semantic_plan_coordinate_dim=4,
        semantic_plan_num_keyframes=2,
        semantic_plan_num_views=2,
        semantic_plan_cross_attention_blocks=(0, 1),
        semantic_plan_adaln_rank=4,
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    with torch.no_grad():
        for block in model.transformer_blocks:
            block.semantic_modulation[1].weight.fill_(0.1)
            block.semantic_modulation[-1].weight.fill_(0.1)
            block.semantic_attn.processor.raw_semantic_bias_gate.fill_(0.5)

    semantic_plan = torch.randn(1, 2, 2, 3, 8, requires_grad=True)
    semantic_positions = torch.rand(1, 2, 2, 3, 2, requires_grad=True)
    semantic_relevance = torch.tensor(
        [[[[1.0, 0.4, 0.2], [0.8, 0.3, 0.1]]] * 2],
        requires_grad=True,
    )
    output = model(
        hidden_states=torch.randn(2, 2, 4),
        encoder_hidden_states=torch.randn(1, 3, 8),
        timestep=torch.ones(2, 2),
        encoder_attention_mask=torch.ones(1, 3),
        n_view=2,
        rope_interpolation_scale=(1.6, 32.0, 32.0),
        num_frames=2,
        height=1,
        width=1,
        semantic_plan=semantic_plan,
        semantic_plan_times=torch.tensor([[0.5, 1.0]]).repeat(2, 1),
        semantic_plan_positions=semantic_positions,
        semantic_plan_mask=torch.ones(1, 2, 2, 3, dtype=torch.bool),
        semantic_plan_relevance=semantic_relevance,
        return_dict=False,
    )[0]["video"]
    output.square().mean().backward()

    assert semantic_plan.grad is not None
    assert semantic_positions.grad is not None
    assert semantic_relevance.grad is not None
    assert torch.count_nonzero(semantic_relevance.grad) > 0
    for block in model.transformer_blocks:
        gate_grad = block.semantic_attn.processor.raw_semantic_bias_gate.grad
        assert gate_grad is not None
        assert torch.count_nonzero(gate_grad) > 0


def test_joint_dataset_delegates_exact_cache_window_without_changing_base_values(
    tmp_path: Path,
) -> None:
    from ge_act.data.libero_fastwam_hdf5_dataset import (
        LiberoFastWAMHDF5Dataset,
    )
    from ge_act.data.libero_hindsight_hdf5_dataset import (
        LiberoHindsightHDF5Dataset,
    )

    manifest_path, stats_path = _write_joint_hdf5_fixture(tmp_path / "hdf5")
    cache_dir, window = _write_joint_cache(tmp_path, manifest_path)
    train_dataset = window.split == "train"
    joint = LiberoHindsightHDF5Dataset(
        manifest_path=manifest_path,
        hindsight_cache=cache_dir,
        stat_file=stats_path,
        train_dataset=train_dataset,
    )
    base = LiberoFastWAMHDF5Dataset(
        manifest_path=manifest_path,
        stat_file=stats_path,
        previous_pick_mode="uniform",
        train_dataset=train_dataset,
    )

    sample = joint[0]
    expected = base.read_by_indexes(
        0,
        window.frame_indices,
        window.action_indices,
    )

    assert sample["episode_key"] == window.episode_key
    assert sample["current_index"] == window.current_index
    assert sample["target_codes"].shape == (2, 4, 729)
    assert sample["target_relevance"].shape == (2, 4, 3, 729)
    assert sample["video"].shape[1:3] == (2, 13)
    for key in ("video", "actions", "state"):
        torch.testing.assert_close(sample[key], expected[key], rtol=0, atol=0)
    assert sample["caption"] == expected["caption"]


def test_joint_dataset_rejects_cache_for_a_different_hdf5_before_reads(
    tmp_path: Path,
) -> None:
    from ge_act.data.libero_hindsight_hdf5_dataset import (
        LiberoHindsightHDF5Dataset,
    )

    manifest_path, stats_path = _write_joint_hdf5_fixture(tmp_path / "hdf5")
    cache_dir, _ = _write_joint_cache(tmp_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["converter_fingerprint"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.*hindsight cache"):
        LiberoHindsightHDF5Dataset(
            manifest_path=manifest_path,
            hindsight_cache=cache_dir,
            stat_file=stats_path,
            train_dataset=True,
        )


class _TinyQwenLanguage(nn.Module):
    def __init__(self, num_layers: int = 12) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(4)


class _TinyQwenRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyQwenLanguage()
        self.visual = nn.Sequential(nn.Linear(4, 4), nn.GELU())


class _TinyGroundedPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = _TinyQwenRoot()
        self.visual_regression = nn.Linear(4, 3)
        self.semantic_projection = nn.Linear(4, 3)
        self.phrase_projection = nn.Linear(4, 3)
        self.grounding_query = nn.Linear(4, 3)
        self.fusion_gate = nn.Linear(4, 1)
        self.register_buffer("codebook", torch.ones(8, 3))


class _TinyGroundedProvider(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.planner = _TinyGroundedPlanner()
        self.visual_adapter = nn.Linear(3, 4)
        self.hidden_adapter = nn.Linear(4, 4)
        self.position_encoder = nn.Linear(4, 4)
        self.output_norm = nn.LayerNorm(4)


class _TinyJointLTX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_proj = nn.Linear(4, 4)
        self.semantic_adapter = nn.Linear(4, 4)
        self.semantic_attn = nn.Linear(4, 4)
        self.action_proj = nn.Linear(4, 4)

    def forward(self, semantic_plan: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = semantic_plan.mean(dim=(1, 2, 3))
        semantic = self.semantic_attn(torch.tanh(self.semantic_adapter(pooled)))
        shared = pooled + semantic
        return {
            "video": self.video_proj(shared),
            "action": self.action_proj(shared),
        }


class _JointForwardDiffusion(_TinyJointLTX):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(
        self,
        *,
        semantic_plan: torch.Tensor,
        semantic_plan_positions: torch.Tensor,
        semantic_plan_mask: torch.Tensor,
        semantic_plan_relevance: torch.Tensor,
        semantic_plan_times: torch.Tensor,
        semantic_condition_mask: torch.Tensor,
        **_diffusion_kwargs,
    ) -> tuple[dict[str, torch.Tensor]]:
        self.forward_calls += 1
        assert semantic_plan_positions.shape == (1, 2, 4, 96, 2)
        assert semantic_plan_mask.shape == (1, 2, 4, 96)
        assert semantic_plan_relevance.shape == (1, 2, 4, 96)
        assert semantic_plan_times.shape == (2, 4)
        assert semantic_condition_mask.shape == (2,)
        return (super().forward(semantic_plan),)


def test_joint_optimizer_has_five_exhaustive_disjoint_owned_groups() -> None:
    from runner.ge_trainer import build_optimizer_parameter_groups
    from models.ltx_models.vlm_semantic_planner import (
        configure_qwen_top_layers_for_joint_training,
    )

    diffusion = _TinyJointLTX()
    provider = _TinyGroundedProvider()
    ownership = configure_qwen_top_layers_for_joint_training(
        provider,
        top_language_layers=8,
    )
    groups = build_optimizer_parameter_groups(
        diffusion,
        train_mode="all",
        base_lr=2e-5,
        action_lr=1e-4,
        semantic_lr=5e-5,
        provider=provider,
        qwen_top_lr=1e-6,
        qwen_vision_lr=5e-7,
        qwen_ownership=ownership,
    )

    assert {group["name"]: group["lr"] for group in groups} == {
        "ltx_video": 2e-5,
        "action_expert": 1e-4,
        "semantic_adapter": 5e-5,
        "qwen_top8": 1e-6,
        "qwen_vision": 5e-7,
    }
    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    trainable_ids = {
        id(parameter)
        for module in (diffusion, provider)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == trainable_ids
    language = provider.planner.backbone.model.language_model
    assert all(
        not parameter.requires_grad
        for layer in language.layers[:4]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in language.layers[-8:]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in provider.planner.backbone.model.visual.parameters()
    )
    assert not language.embed_tokens.weight.requires_grad
    assert all(not parameter.requires_grad for parameter in language.norm.parameters())


def test_joint_loss_routes_only_ge_originating_qwen_gradient_at_point_one() -> None:
    from qwen35_planx.provider import scale_gradient
    from runner.ge_trainer import compute_joint_loss

    x = torch.tensor(2.0, requires_grad=True)
    y = scale_gradient(x, 0.1)
    loss = compute_joint_loss(
        loss_video=3.0 * y,
        loss_action=torch.tensor(0.0),
        planner_loss=5.0 * x,
        action_loss_scale=1.0,
        planner_aux_weight=1.0,
    )

    assert y.item() == 2.0
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor(5.3))


def test_joint_loss_uses_exact_action_and_planner_coefficients() -> None:
    from runner.ge_trainer import compute_joint_loss

    actual = compute_joint_loss(
        loss_video=torch.tensor(2.0),
        loss_action=torch.tensor(3.0),
        planner_loss=torch.tensor(5.0),
        action_loss_scale=0.4,
        planner_aux_weight=0.25,
    )

    torch.testing.assert_close(actual, torch.tensor(4.45))


class _RecordingGroundedProvider(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(2.0))
        self.relevance_anchor = nn.Parameter(torch.tensor(1.0))
        self.teacher_force_calls = 0
        self.generate_calls = 0
        self.targets = None
        self.last_gradient_scale = None

    def _plan(self):
        return type(
            "Plan",
            (),
            {
                "relevance": self.relevance_anchor.expand(1, 2, 4, 3, 729),
                "loss": 5.0 * self.anchor,
            },
        )()

    def teacher_force(self, current_images, instructions, targets):
        assert current_images.shape == (1, 2, 3, 8, 8)
        assert instructions == ["pick"]
        self.teacher_force_calls += 1
        self.targets = targets
        return self._plan()

    def generate(self, current_images, instructions):
        assert current_images.shape == (1, 2, 3, 8, 8)
        assert instructions == ["pick"]
        self.generate_calls += 1
        return self._plan()

    def fuse(self, _plan, *, qwen_gradient_scale):
        from qwen35_planx.provider import scale_gradient

        self.last_gradient_scale = qwen_gradient_scale
        return scale_gradient(
            self.anchor,
            qwen_gradient_scale,
        ).expand(1, 2, 4, 729, 1)


def _grounded_batch() -> dict[str, object]:
    return {
        "current_images": torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
        "caption": ["pick"],
        "target_codes": torch.zeros(1, 2, 4, 729, dtype=torch.long),
        "target_relevance": torch.full((1, 2, 4, 3, 729), 1.0 / 729),
        "target_relevance_confidence": torch.ones(1, 2, 4, 3),
        "target_flow": torch.zeros(1, 2, 3, 729, 3),
        "target_phrase_embeddings": torch.zeros(1, 3, 1152),
    }


def test_grounded_training_teacher_forces_complete_packed_targets_and_compresses() -> None:
    from qwen35_planx.planner_dataset import CachedPlannerTargets
    from runner.ge_trainer import build_grounded_semantic_condition

    provider = _RecordingGroundedProvider()
    condition, planner_loss = build_grounded_semantic_condition(
        provider,
        _grounded_batch(),
        training=True,
        qwen_gradient_scale=0.1,
    )

    assert provider.teacher_force_calls == 1
    assert provider.generate_calls == 0
    assert isinstance(provider.targets, CachedPlannerTargets)
    assert provider.targets.codes.shape == (1, 2, 4, 729)
    assert provider.last_gradient_scale == 0.1
    assert condition.tokens.shape == (1, 2, 4, 96, 1)
    assert condition.positions.shape == (1, 2, 4, 96, 2)
    assert condition.mask.shape == (1, 2, 4, 96)
    assert condition.relevance.shape == (1, 2, 4, 96)
    torch.testing.assert_close(planner_loss, 5.0 * provider.anchor)


def test_grounded_validation_generates_without_cache_targets() -> None:
    from runner.ge_trainer import build_grounded_semantic_condition

    provider = _RecordingGroundedProvider()
    batch = {
        "current_images": torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
        "caption": ["pick"],
    }
    condition, planner_loss = build_grounded_semantic_condition(
        provider,
        batch,
        training=False,
        qwen_gradient_scale=0.1,
    )

    assert provider.teacher_force_calls == 0
    assert provider.generate_calls == 1
    assert provider.last_gradient_scale == 1.0
    assert planner_loss is None
    assert condition.tokens.shape == (1, 2, 4, 96, 1)


def test_grounded_validation_helper_restores_provider_training_mode() -> None:
    from runner.ge_trainer import generate_grounded_condition_for_validation

    provider = _RecordingGroundedProvider()
    provider.train()
    condition = generate_grounded_condition_for_validation(
        provider,
        {
            "current_images": torch.zeros(
                1,
                2,
                3,
                8,
                8,
                dtype=torch.uint8,
            ),
            "caption": ["pick"],
        },
        qwen_gradient_scale=0.1,
    )

    assert provider.training
    assert provider.generate_calls == 1
    assert condition.tokens.shape == (1, 2, 4, 96, 1)


class _TinySmokeProvider(_TinyGroundedProvider):
    def teacher_force(self, current_images, instructions, targets):
        assert current_images.shape[1] == 2
        assert instructions == ["pick"]
        assert targets.codes.shape == (1, 2, 4, 729)
        batch_size = current_images.shape[0]
        hidden = torch.ones(
            batch_size,
            2,
            4,
            729,
            4,
            device=next(self.parameters()).device,
        )
        language = self.planner.backbone.model.language_model
        for layer in language.layers:
            hidden = torch.tanh(layer(hidden))
        hidden = hidden + self.planner.backbone.model.visual(hidden)
        codes = self.planner.visual_regression(hidden)
        relevance = torch.nn.functional.softplus(
            self.planner.grounding_query(hidden)
        ).movedim(-1, -2)
        planner_loss = (
            codes.square().mean()
            + self.planner.semantic_projection(hidden).square().mean()
            + self.planner.phrase_projection(hidden).square().mean()
            + relevance.square().mean()
            + self.planner.fusion_gate(hidden).square().mean()
        )
        return type(
            "TinyGroundedOutput",
            (),
            {
                "hidden": hidden,
                "codes": codes,
                "relevance": relevance,
                "loss": planner_loss,
            },
        )()

    def fuse(self, plan, *, qwen_gradient_scale):
        from qwen35_planx.provider import scale_gradient

        codes = self.visual_adapter(
            scale_gradient(plan.codes, qwen_gradient_scale)
        )
        hidden = self.hidden_adapter(
            scale_gradient(plan.hidden, qwen_gradient_scale)
        )
        positions = torch.ones_like(plan.hidden)
        position_features = self.position_encoder(positions)
        return self.output_norm(codes + hidden + position_features)


def _joint_forward_kwargs() -> dict[str, object]:
    return {
        "prompt_embeds": torch.zeros(1, 1, 4),
        "prompt_attention_mask": torch.ones(1, 1, dtype=torch.bool),
        "noisy_latents": torch.zeros(2, 1, 4),
        "timesteps": torch.zeros(2, 1, dtype=torch.long),
        "num_frames": 1,
        "height": 1,
        "width": 1,
        "n_view": 2,
        "frame_rate": 20,
        "temporal_compression_ratio": 8,
        "spatial_compression_ratio": 32,
        "semantic_plan_times": torch.ones(2, 4),
        "semantic_condition_mask": torch.ones(2),
    }


def test_joint_wrapper_runs_grounded_provider_and_diffusion_in_one_forward() -> None:
    from runner.ge_trainer import JointGroundedTrainingModel

    diffusion = _JointForwardDiffusion()
    provider = _TinySmokeProvider()
    joint = JointGroundedTrainingModel(
        diffusion_model=diffusion,
        semantic_provider=provider,
    )

    output = joint(
        grounded_batch=_grounded_batch(),
        qwen_gradient_scale=0.1,
        **_joint_forward_kwargs(),
    )

    assert diffusion.forward_calls == 1
    assert output.latents["video"].shape == (1, 4)
    assert output.latents["action"].shape == (1, 4)
    assert output.planner_loss.ndim == 0
    assert output.semantic_condition.tokens.shape == (1, 2, 4, 96, 4)


def _two_rank_joint_forward_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_path: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(123)
        diffusion = _JointForwardDiffusion()
        provider = _TinySmokeProvider()
        __import__(
            "models.ltx_models.vlm_semantic_planner",
            fromlist=["configure_qwen_top_layers_for_joint_training"],
        ).configure_qwen_top_layers_for_joint_training(
            provider,
            top_language_layers=8,
        )
        joint = torch.nn.parallel.DistributedDataParallel(
            __import__(
                "runner.ge_trainer",
                fromlist=["JointGroundedTrainingModel"],
            ).JointGroundedTrainingModel(
                diffusion_model=diffusion,
                semantic_provider=provider,
            ),
            find_unused_parameters=False,
        )
        optimizer = torch.optim.SGD(joint.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        for microbatch in range(2):
            synchronization = (
                joint.no_sync()
                if microbatch == 0
                else contextlib.nullcontext()
            )
            with synchronization:
                output = joint(
                    grounded_batch=_grounded_batch(),
                    qwen_gradient_scale=0.1,
                    **_joint_forward_kwargs(),
                )
                loss = float(rank + 1 + microbatch) * (
                    output.latents["video"].square().mean()
                    + output.latents["action"].square().mean()
                    + 0.25 * output.planner_loss
                )
                loss.backward()
        optimizer.step()
        flattened = torch.cat(
            [
                parameter.detach().reshape(-1)
                for parameter in joint.module.parameters()
            ]
        )
        gathered = [torch.empty_like(flattened) for _ in range(world_size)]
        dist.all_gather(gathered, flattened)
        if rank == 0:
            torch.save(gathered, result_path)
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_joint_forward_synchronizes_provider_and_diffusion(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "ddp.init"
    result_path = tmp_path / "ddp-result.pt"
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_two_rank_joint_forward_worker,
            args=(rank, 2, str(init_file), str(result_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    gathered = torch.load(result_path, weights_only=True)
    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)


def _single_gpu_accelerator_joint_worker(result_path: str) -> None:
    from accelerate import Accelerator
    from runner.ge_trainer import (
        Trainer,
    )

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("single-GPU smoke did not receive a CUDA device")
    torch.manual_seed(321)
    diffusion = _JointForwardDiffusion()
    provider = _TinySmokeProvider()
    __import__(
        "models.ltx_models.vlm_semantic_planner",
        fromlist=["configure_qwen_top_layers_for_joint_training"],
    ).configure_qwen_top_layers_for_joint_training(
        provider,
        top_language_layers=8,
    )
    optimizer = torch.optim.SGD(
        [*diffusion.parameters(), *provider.parameters()],
        lr=0.01,
    )
    dataloader = torch.utils.data.DataLoader(_CursorToyDataset(), batch_size=1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _step: 1.0,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.grounded_training_enabled = True
    trainer.grounded_provider = provider
    trainer.diffusion_model = diffusion
    trainer.optimizer = optimizer
    trainer.train_dataloader = dataloader
    trainer.lr_scheduler = scheduler
    trainer.state = SimpleNamespace(accelerator=accelerator)
    trainer.args = SimpleNamespace(resume_from_checkpoint=None)
    trainer.sampler_seed = 19
    trainer.joint_model = None
    trainer.resume_cursor = None
    trainer.current_cursor = None
    trainer.prepare_for_training()
    model = trainer.diffusion_model
    optimizer = trainer.optimizer
    scheduler = trainer.lr_scheduler
    assert trainer.grounded_provider is None
    assert trainer.resume_cursor.microbatches_per_epoch == len(
        trainer.train_dataloader
    )
    kwargs = {
        key: (
            value.to(accelerator.device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in _joint_forward_kwargs().items()
    }
    output = model(
        grounded_batch=_grounded_batch(),
        qwen_gradient_scale=0.1,
        **kwargs,
    )
    loss = (
        output.latents["video"].square().mean()
        + output.latents["action"].square().mean()
        + 0.25 * output.planner_loss
    )
    accelerator.backward(loss)
    unwrapped = accelerator.unwrap_model(model)
    result = {
        "device": str(accelerator.device),
        "diffusion_grad": float(
            unwrapped.diffusion_model.video_proj.weight.grad.abs().sum()
        ),
        "provider_grad": float(
            unwrapped.semantic_provider.planner.backbone.model.visual[
                0
            ].weight.grad.abs().sum()
        ),
    }
    optimizer.step()
    scheduler.step()
    torch.save(result, result_path)
    accelerator.end_training()


def test_single_gpu_accelerator_joint_step_updates_both_children(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    result_path = tmp_path / "single-gpu.pt"
    context = mp.get_context("spawn")
    process = context.Process(
        target=_single_gpu_accelerator_joint_worker,
        args=(str(result_path),),
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 0
    result = torch.load(result_path, weights_only=True)
    assert result["device"].startswith("cuda")
    assert result["diffusion_grad"] > 0
    assert result["provider_grad"] > 0


def test_tiny_joint_one_step_has_gradients_in_all_five_groups_and_freezes_lower_qwen() -> None:
    from models.ltx_models.vlm_semantic_planner import (
        configure_qwen_top_layers_for_joint_training,
    )
    from runner.ge_trainer import (
        JointGroundedTrainingModel,
        build_optimizer_parameter_groups,
        compute_joint_loss,
    )

    torch.manual_seed(0)
    diffusion = _JointForwardDiffusion()
    provider = _TinySmokeProvider()
    ownership = configure_qwen_top_layers_for_joint_training(
        provider,
        top_language_layers=8,
    )
    groups = build_optimizer_parameter_groups(
        diffusion,
        train_mode="all",
        base_lr=2e-5,
        action_lr=1e-4,
        semantic_lr=5e-5,
        provider=provider,
        qwen_top_lr=1e-6,
        qwen_vision_lr=5e-7,
        qwen_ownership=ownership,
    )
    joint = JointGroundedTrainingModel(
        diffusion_model=diffusion,
        semantic_provider=provider,
    )
    optimizer = torch.optim.AdamW(groups)
    output = joint(
        grounded_batch=_grounded_batch(),
        qwen_gradient_scale=0.1,
        **_joint_forward_kwargs(),
    )
    loss = compute_joint_loss(
        loss_video=output.latents["video"].square().mean(),
        loss_action=output.latents["action"].square().mean(),
        planner_loss=output.planner_loss,
        action_loss_scale=0.4,
        planner_aux_weight=0.25,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_sums = {
        group["name"]: sum(
            float(parameter.grad.abs().sum())
            for parameter in group["params"]
            if parameter.grad is not None
        )
        for group in groups
    }
    assert all(value > 0 for value in gradient_sums.values()), gradient_sums

    language = provider.planner.backbone.model.language_model
    assert all(
        parameter.grad is None
        for layer in language.layers[:4]
        for parameter in layer.parameters()
    )
    assert language.embed_tokens.weight.grad is None
    assert all(parameter.grad is None for parameter in language.norm.parameters())
    assert not provider.planner.codebook.requires_grad
    assert provider.planner.codebook.grad is None
    optimizer.step()


class _CheckpointAccelerator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.is_main_process = True
        self.saved_paths: list[Path] = []
        self.loaded_paths: list[Path] = []

    def wait_for_everyone(self) -> None:
        pass

    def save_state(self, path: str) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), destination / "model.pt")
        self.saved_paths.append(destination)

    def load_state(self, path: str) -> None:
        source = Path(path)
        self.model.load_state_dict(
            torch.load(source / "model.pt", weights_only=True)
        )
        self.loaded_paths.append(source)


def test_joint_container_checkpoint_restores_diffusion_and_provider_atomically(
    tmp_path: Path,
) -> None:
    from runner.ge_trainer import (
        JointGroundedTrainingModel,
        TrainingCursor,
        load_joint_training_checkpoint,
        save_joint_training_checkpoint,
    )

    joint = JointGroundedTrainingModel(
        diffusion_model=nn.Linear(2, 2, bias=False),
        semantic_provider=nn.Linear(2, 2, bias=False),
    )
    with torch.no_grad():
        joint.diffusion_model.weight.fill_(2.0)
        joint.semantic_provider.weight.fill_(3.0)
    accelerator = _CheckpointAccelerator(joint)

    checkpoint = save_joint_training_checkpoint(
        accelerator,
        tmp_path,
        cursor=TrainingCursor(
            global_step=7,
            epoch=1,
            consumed_microbatches=2,
            microbatches_per_epoch=5,
            sampler_seed=42,
        ),
        joint_model=joint,
    )
    assert checkpoint == tmp_path / "step_000007"
    assert checkpoint.is_dir()
    assert not list(tmp_path.glob("*.incomplete-*"))
    with torch.no_grad():
        joint.diffusion_model.weight.zero_()
        joint.semantic_provider.weight.zero_()

    cursor = load_joint_training_checkpoint(
        accelerator,
        checkpoint,
        joint_model=joint,
        expected_microbatches_per_epoch=5,
        expected_sampler_seed=42,
    )

    assert cursor.global_step == 7
    torch.testing.assert_close(
        joint.diffusion_model.weight,
        torch.full_like(joint.diffusion_model.weight, 2.0),
    )
    torch.testing.assert_close(
        joint.semantic_provider.weight,
        torch.full_like(joint.semantic_provider.weight, 3.0),
    )


def test_joint_checkpoint_round_trips_exact_training_cursor(
    tmp_path: Path,
) -> None:
    from runner.ge_trainer import (
        JointGroundedTrainingModel,
        TrainingCursor,
        load_joint_training_checkpoint,
        save_joint_training_checkpoint,
    )

    joint = JointGroundedTrainingModel(
        diffusion_model=nn.Linear(2, 2, bias=False),
        semantic_provider=nn.Linear(2, 2, bias=False),
    )
    accelerator = _CheckpointAccelerator(joint)
    cursor = TrainingCursor(
        global_step=7,
        epoch=2,
        consumed_microbatches=3,
        microbatches_per_epoch=5,
        sampler_seed=42,
    )

    checkpoint = save_joint_training_checkpoint(
        accelerator,
        tmp_path,
        cursor=cursor,
        joint_model=joint,
    )
    metadata = json.loads(
        (checkpoint / "joint_state.json").read_text(encoding="utf-8")
    )
    assert metadata["format_version"] == 2
    assert metadata["cursor"] == {
        "global_step": 7,
        "epoch": 2,
        "consumed_microbatches": 3,
        "microbatches_per_epoch": 5,
        "sampler_seed": 42,
    }

    restored = load_joint_training_checkpoint(
        accelerator,
        checkpoint,
        joint_model=joint,
        expected_microbatches_per_epoch=5,
        expected_sampler_seed=42,
    )
    assert restored == cursor


def test_epoch_seeded_sampler_reconstructs_each_epoch_without_rng_state() -> None:
    from runner.ge_trainer import EpochSeededRandomSampler

    sampler = EpochSeededRandomSampler(range(8), seed=19)
    sampler.set_epoch(3)
    first = list(sampler)
    torch.rand(100)
    sampler.set_epoch(3)
    reconstructed = list(sampler)
    sampler.set_epoch(4)
    next_epoch = list(sampler)

    assert reconstructed == first
    assert sorted(first) == list(range(8))
    assert sorted(next_epoch) == list(range(8))
    assert next_epoch != first


class _CursorToyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"sample_id": torch.tensor(index)}


def _save_tiny_training_result(
    path: str,
    *,
    accelerator,
    model,
    optimizer,
    scheduler,
    sequence: list[int],
) -> None:
    torch.save(
        {
            "sequence": sequence,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def _tiny_cursor_training_worker(
    mode: str,
    checkpoint_root: str,
    result_path: str,
) -> None:
    from accelerate import Accelerator
    from runner.ge_trainer import (
        EpochSeededRandomSampler,
        JointGroundedTrainingModel,
        TrainingCursor,
        load_joint_training_checkpoint,
        prepare_joint_training_components,
        save_joint_training_checkpoint,
        set_dataloader_epoch,
    )

    torch.set_num_threads(1)
    torch.manual_seed(123)
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
    )
    dataset = _CursorToyDataset()
    sampler = EpochSeededRandomSampler(dataset, seed=19)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        generator=torch.Generator().manual_seed(19),
    )
    diffusion = _JointForwardDiffusion()
    provider = _TinySmokeProvider()
    __import__(
        "models.ltx_models.vlm_semantic_planner",
        fromlist=["configure_qwen_top_layers_for_joint_training"],
    ).configure_qwen_top_layers_for_joint_training(
        provider,
        top_language_layers=8,
    )
    joint = JointGroundedTrainingModel(
        diffusion_model=diffusion,
        semantic_provider=provider,
    )
    optimizer = torch.optim.AdamW(joint.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0 / (step + 1),
    )
    model, optimizer, dataloader, scheduler = (
        prepare_joint_training_components(
            accelerator,
            joint,
            optimizer,
            dataloader,
            scheduler,
        )
    )
    cursor = TrainingCursor(
        global_step=0,
        epoch=0,
        consumed_microbatches=0,
        microbatches_per_epoch=4,
        sampler_seed=19,
    )
    if mode == "resume":
        cursor = load_joint_training_checkpoint(
            accelerator,
            Path(checkpoint_root) / "step_000002",
            joint_model=accelerator.unwrap_model(model),
            expected_microbatches_per_epoch=4,
            expected_sampler_seed=19,
        )
    sequence: list[int] = []
    stop_after = 4 if mode == "interrupt" else None
    processed = 0
    start_epoch = cursor.epoch
    for epoch in range(start_epoch, 2):
        set_dataloader_epoch(
            dataloader,
            epoch=epoch,
            sampler_seed=19,
        )
        skipped = cursor.consumed_microbatches if epoch == start_epoch else 0
        active_loader = accelerator.skip_first_batches(
            dataloader,
            num_batches=skipped,
        )
        for offset, batch in enumerate(active_loader):
            absolute_microbatch = skipped + offset + 1
            sequence.append(int(batch["sample_id"].item()))
            with accelerator.accumulate(model):
                output = model(
                    grounded_batch=_grounded_batch(),
                    qwen_gradient_scale=0.1,
                    **_joint_forward_kwargs(),
                )
                stochastic_scale = 0.5 + torch.rand(())
                loss = stochastic_scale * (
                    output.latents["video"].square().mean()
                    + output.latents["action"].square().mean()
                    + 0.25 * output.planner_loss
                )
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step = cursor.global_step + int(accelerator.sync_gradients)
            if absolute_microbatch == 4:
                cursor = TrainingCursor(
                    global_step=global_step,
                    epoch=epoch + 1,
                    consumed_microbatches=0,
                    microbatches_per_epoch=4,
                    sampler_seed=19,
                )
            else:
                cursor = TrainingCursor(
                    global_step=global_step,
                    epoch=epoch,
                    consumed_microbatches=absolute_microbatch,
                    microbatches_per_epoch=4,
                    sampler_seed=19,
                )
            processed += 1
            if stop_after is not None and processed == stop_after:
                assert accelerator.sync_gradients
                save_joint_training_checkpoint(
                    accelerator,
                    checkpoint_root,
                    cursor=cursor,
                    joint_model=accelerator.unwrap_model(model),
                )
                _save_tiny_training_result(
                    result_path,
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sequence=sequence,
                )
                accelerator.end_training()
                return
        cursor = TrainingCursor(
            global_step=cursor.global_step,
            epoch=epoch + 1,
            consumed_microbatches=0,
            microbatches_per_epoch=4,
            sampler_seed=19,
        )
    _save_tiny_training_result(
        result_path,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sequence=sequence,
    )
    accelerator.end_training()


def _assert_nested_state_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_state_equal(left_item, right_item)
    else:
        assert left == right


def test_accelerator_resume_matches_uninterrupted_samples_and_training_state(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    checkpoint_root = tmp_path / "checkpoint"
    paths = {
        name: tmp_path / f"{name}.pt"
        for name in ("baseline", "interrupt", "resume")
    }
    for mode in ("baseline", "interrupt", "resume"):
        process = context.Process(
            target=_tiny_cursor_training_worker,
            args=(mode, str(checkpoint_root), str(paths[mode])),
        )
        process.start()
        process.join(timeout=60)
        assert process.exitcode == 0

    baseline = torch.load(paths["baseline"], weights_only=True)
    interrupted = torch.load(paths["interrupt"], weights_only=True)
    resumed = torch.load(paths["resume"], weights_only=True)
    assert interrupted["sequence"] + resumed["sequence"] == baseline["sequence"]
    _assert_nested_state_equal(baseline["model"], resumed["model"])
    _assert_nested_state_equal(baseline["optimizer"], resumed["optimizer"])
    _assert_nested_state_equal(baseline["scheduler"], resumed["scheduler"])


def test_joint_prepare_passes_one_registered_model_and_all_training_state() -> None:
    from runner.ge_trainer import (
        JointGroundedTrainingModel,
        prepare_joint_training_components,
    )

    joint = JointGroundedTrainingModel(
        diffusion_model=nn.Linear(2, 2),
        semantic_provider=nn.Linear(2, 2),
    )
    optimizer = torch.optim.AdamW(joint.parameters(), lr=1e-4)
    loader = [torch.ones(1)]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _step: 1.0,
    )

    class RecordingAccelerator:
        def __init__(self) -> None:
            self.arguments = None

        def prepare(self, *arguments):
            self.arguments = arguments
            return arguments

    accelerator = RecordingAccelerator()
    prepared = prepare_joint_training_components(
        accelerator,
        joint,
        optimizer,
        loader,
        scheduler,
    )

    assert accelerator.arguments == (
        joint,
        optimizer,
        loader,
        scheduler,
    )
    assert prepared == accelerator.arguments


def test_pipeline_preserves_all_compressed_grounding_fields_for_cfg() -> None:
    from models.pipeline.custom_pipeline import (
        prepare_pipeline_semantic_conditioning,
    )

    plan = torch.randn(1, 2, 4, 96, 8)
    times = torch.tensor([[0.8, 0.875, 0.925, 1.0]]).repeat(2, 1)
    condition_mask = torch.ones(2)
    positions = torch.rand(1, 2, 4, 96, 2)
    mask = torch.ones(1, 2, 4, 96, dtype=torch.bool)
    relevance = torch.rand(1, 2, 4, 96)

    prepared = prepare_pipeline_semantic_conditioning(
        plan,
        times,
        condition_mask,
        semantic_plan_positions=positions,
        semantic_plan_mask=mask,
        semantic_plan_relevance=relevance,
        batch_size=1,
        n_view=2,
        num_keyframes=4,
        do_classifier_free_guidance=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(prepared) == 6
    prepared_plan, prepared_times, prepared_condition, *grounding = prepared
    assert prepared_plan.shape == (2, 2, 4, 96, 8)
    assert prepared_times.shape == (4, 4)
    assert prepared_condition.shape == (4,)
    assert grounding[0].shape == (2, 2, 4, 96, 2)
    assert grounding[1].shape == (2, 2, 4, 96)
    assert grounding[2].shape == (2, 2, 4, 96)
    torch.testing.assert_close(grounding[0][0], grounding[0][1])
    torch.testing.assert_close(grounding[2][0], grounding[2][1])


def test_grounded_joint_config_locks_training_and_geometry_contract() -> None:
    config_path = (
        GE_ACT_ROOT
        / "configs/ltx_model/libero/action_model_libero_qwen35_grounded_hdf5.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert {
        "return_action": config["return_action"],
        "return_video": config["return_video"],
        "train_mode": config["train_mode"],
        "train_steps": config["train_steps"],
        "steps_to_save": config["steps_to_save"],
        "mixed_precision": config["mixed_precision"],
        "allow_tf32": config["allow_tf32"],
        "lr": config["lr"],
        "action_lr": config["action_lr"],
        "semantic_lr": config["semantic_lr"],
        "qwen_top_lr": config["qwen_top_lr"],
        "qwen_vision_lr": config["qwen_vision_lr"],
        "planner_aux_weight": config["planner_aux_weight"],
        "qwen_ge_gradient_scale": config["qwen_ge_gradient_scale"],
    } == {
        "return_action": True,
        "return_video": True,
        "train_mode": "all",
        "train_steps": 30_000,
        "steps_to_save": 5_000,
        "mixed_precision": "bf16",
        "allow_tf32": True,
        "lr": 2e-5,
        "action_lr": 1e-4,
        "semantic_lr": 5e-5,
        "qwen_top_lr": 1e-6,
        "qwen_vision_lr": 5e-7,
        "planner_aux_weight": 0.25,
        "qwen_ge_gradient_scale": 0.1,
    }
    assert {
        key: config["semantic_plan"][key]
        for key in (
            "enabled",
            "source",
            "num_keyframes",
            "tokens_per_frame",
            "dropout",
        )
    } == {
        "enabled": True,
        "source": "qwen35_grounded",
        "num_keyframes": 4,
        "tokens_per_frame": 96,
        "dropout": 0.15,
    }
    model = config["diffusion_model"]["config"]
    assert model["action_expert"] is True
    assert model["num_layers"] == 28
    assert model["semantic_plan_cross_attention_blocks"] == list(range(28))
    assert config["train_data_class"] == "LiberoHindsightHDF5Dataset"
    assert config["data"]["train"]["hindsight_cache"]
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128


def test_grounded_joint_config_passes_shape_only_hdf5_preflight() -> None:
    from ge_act.scripts.preflight_libero_fastwam_hdf5 import (
        collect_hdf5_preflight_errors,
    )

    config_path = (
        GE_ACT_ROOT
        / "configs/ltx_model/libero/action_model_libero_qwen35_grounded_hdf5.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=False,
    ) == []


def test_joint_video_action_training_keeps_the_real_future_clip() -> None:
    from runner.ge_trainer import select_training_future_video

    video = torch.arange(2 * 3 * 13).reshape(2, 3, 13)

    joint = select_training_future_video(
        video,
        n_previous=4,
        chunk=9,
        return_action=True,
        return_video=True,
    )
    action_only = select_training_future_video(
        video,
        n_previous=4,
        chunk=9,
        return_action=True,
        return_video=False,
    )

    torch.testing.assert_close(joint, video[:, :, 4:])
    torch.testing.assert_close(
        action_only,
        video[:, :, 4:5].repeat(1, 1, 9),
    )


def _write_qwen35_model_config(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "vocab_size": 32,
                },
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_heads": 1,
                    "out_hidden_size": 8,
                    "patch_size": 2,
                    "spatial_merge_size": 1,
                    "temporal_patch_size": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_qwen35_runtime_gate_rejects_unsupported_transformers_actionably(
    tmp_path: Path,
) -> None:
    from models.ltx_models.vlm_semantic_planner import (
        validate_qwen35_model_config_runtime,
    )

    model_config = _write_qwen35_model_config(tmp_path / "model_config")
    with pytest.raises(RuntimeError, match="Qwen3.5-capable Transformers"):
        validate_qwen35_model_config_runtime(model_config)


def test_qwen35_runtime_gate_resolves_saved_model_class_in_supported_interpreter(
    tmp_path: Path,
) -> None:
    model_config = _write_qwen35_model_config(tmp_path / "model_config")
    qwen_python = Path("/data/LFT-W02_data/.conda/envs/qwen35/bin/python")
    if not qwen_python.is_file():
        pytest.skip("Qwen3.5-capable reference interpreter is unavailable")
    code = """
import sys
from models.ltx_models.vlm_semantic_planner import validate_qwen35_model_config_runtime
config, model_class = validate_qwen35_model_config_runtime(sys.argv[1])
assert config.model_type == "qwen3_5"
assert model_class.__name__ == "Qwen3_5ForConditionalGeneration"
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{REPOSITORY_ROOT}:{GE_ACT_ROOT}"
        + (
            f":{environment['PYTHONPATH']}"
            if environment.get("PYTHONPATH")
            else ""
        )
    )
    completed = subprocess.run(
        [str(qwen_python), "-c", code, str(model_config)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_sbatch_requires_explicit_coherent_grounded_environment(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("CONDA_ENV", None)
    environment["GE_ACT_ROOT"] = str(GE_ACT_ROOT)
    environment["CONFIG"] = str(config)

    completed = subprocess.run(
        [
            "bash",
            str(
                GE_ACT_ROOT
                / "scripts/sbatch_train_ltx_qwen35_grounded_hpc3.sh"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "CONDA_ENV must point to" in completed.stderr
