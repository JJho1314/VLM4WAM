from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from qwen35_planx.planner_dataset import GroundedPlannerBatch


class RaisingOutputHead(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("the full Qwen vocabulary output head was called")


class FakeBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, sequence_length: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.sequence_states = nn.Parameter(
            torch.randn(sequence_length, hidden_dim)
        )
        self.lm_head = RaisingOutputHead()
        self.calls: list[dict[str, object]] = []
        self.last_hidden_state: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor, **kwargs):
        self.calls.append(kwargs)
        hidden = self.sequence_states.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        hidden = hidden + 0.05 * self.embedding(input_ids)
        self.last_hidden_state = hidden
        return SimpleNamespace(last_hidden_state=hidden)


class FakeConditionalWrapper(nn.Module):
    def __init__(self, model: FakeBackbone) -> None:
        super().__init__()
        self.model = model
        self.lm_head = RaisingOutputHead()
        self.public_forward_calls = 0

    def forward(self, *args, **kwargs):
        self.public_forward_calls += 1
        output = self.model(*args, **kwargs)
        return self.lm_head(output.last_hidden_state)


def make_batch(
    *,
    batch_cameras: int = 2,
    frames: int = 4,
    tokens: int = 4,
    vocab_size: int = 13,
    text_dim: int = 6,
) -> GroundedPlannerBatch:
    code_count = frames * tokens
    sequence_length = (2 * code_count) + 3
    pre_positions = torch.arange(code_count).expand(batch_cameras, -1).clone()
    post_positions = (
        torch.arange(code_count, 2 * code_count)
        .expand(batch_cameras, -1)
        .clone()
    )
    field_positions = torch.arange(
        2 * code_count,
        sequence_length,
    ).expand(batch_cameras, -1).clone()
    relevance = torch.rand(batch_cameras, frames, 3, tokens)
    relevance = relevance / relevance.sum(dim=-1, keepdim=True)
    flow = torch.zeros(batch_cameras, frames - 1, tokens, 3)
    flow[..., 2] = 1
    phrase_embeddings = F.normalize(
        torch.randn(batch_cameras, 3, text_dim),
        dim=-1,
    )
    return GroundedPlannerBatch(
        qwen_inputs={
            "input_ids": torch.arange(sequence_length)
            .remainder(vocab_size)
            .expand(batch_cameras, -1)
            .clone(),
            "attention_mask": torch.ones(
                batch_cameras,
                sequence_length,
                dtype=torch.long,
            ),
        },
        code_targets=torch.arange(batch_cameras * code_count)
        .reshape(batch_cameras, code_count)
        .remainder(vocab_size),
        pre_positions=pre_positions,
        post_positions=post_positions,
        field_positions=field_positions,
        field_mask=torch.ones(batch_cameras, 3, dtype=torch.bool),
        relevance_targets=relevance,
        relevance_confidence=torch.ones(batch_cameras, frames, 3),
        flow_targets=flow,
        phrase_embeddings=phrase_embeddings,
        counterfactual_embeddings=torch.roll(
            phrase_embeddings,
            shifts=1,
            dims=-1,
        ).unsqueeze(2),
        counterfactual_mask=torch.ones(batch_cameras, 3, 1, dtype=torch.bool),
    )


def make_planner_and_batch():
    from qwen35_planx.planner import GroundedQwen35Planner

    torch.manual_seed(11)
    hidden_dim = 8
    text_dim = 6
    code_dim = 7
    vocab_size = 13
    batch = make_batch(
        vocab_size=vocab_size,
        text_dim=text_dim,
    )
    backbone = FakeBackbone(
        vocab_size,
        hidden_dim,
        batch.qwen_inputs["input_ids"].shape[1],
    )
    original_codebook = torch.randn(
        vocab_size,
        code_dim,
        requires_grad=True,
    )
    planner = GroundedQwen35Planner._from_test_components(
        backbone=backbone,
        visual_embedding_weight=backbone.embedding.weight,
        codebook=original_codebook,
        hidden_dim=hidden_dim,
        text_dim=text_dim,
    )
    return planner, backbone, batch, original_codebook


def test_planner_uses_pre_for_code_and_post_for_semantics() -> None:
    planner, backbone, batch, _ = make_planner_and_batch()

    output = planner(batch)
    assert backbone.last_hidden_state is not None
    pre = torch.gather(
        backbone.last_hidden_state,
        1,
        batch.pre_positions.unsqueeze(-1).expand(-1, -1, 8),
    )
    post = torch.gather(
        backbone.last_hidden_state,
        1,
        batch.post_positions.unsqueeze(-1).expand(-1, -1, 8),
    )
    expected_codes = F.linear(
        pre.reshape(-1, 8),
        backbone.embedding.weight,
    ).argmax(dim=-1)
    expected_visual = F.normalize(planner.visual_regression(pre), dim=-1)
    expected_semantic = F.normalize(
        planner.semantic_projection(post),
        dim=-1,
    )

    assert output.codes.shape == (batch.size, 4, 4)
    assert output.code_embeddings.shape == (batch.size, 4, 4, 7)
    assert output.post_hidden.shape == (batch.size, 4, 4, 8)
    assert output.predicted_phrase_embeddings.shape == (batch.size, 3, 6)
    assert output.visual_regression.shape == (batch.size, 16, 7)
    assert output.semantic_features.shape == (batch.size, 4, 4, 6)
    assert output.relevance_logits.shape == (batch.size, 4, 3, 4)
    assert output.relevance.shape == (batch.size, 4, 3, 4)
    assert output.fusion_gate.shape == (batch.size, 4, 4, 1)
    assert output.times.shape == (4,)
    torch.testing.assert_close(
        output.times,
        torch.tensor([0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0]),
    )
    torch.testing.assert_close(
        output.codes.reshape(batch.size, -1),
        expected_codes.reshape(batch.size, -1),
    )
    torch.testing.assert_close(output.visual_regression, expected_visual)
    torch.testing.assert_close(
        output.semantic_features.reshape(batch.size, -1, 6),
        expected_semantic,
    )
    assert (
        output.codes.reshape(batch.size, -1)[:, -1]
        == expected_codes.reshape(batch.size, -1)[:, -1]
    ).all()
    assert output.debug_pre_positions is batch.pre_positions
    assert output.debug_post_positions is batch.post_positions
    assert len(backbone.calls) == 1
    assert backbone.calls[0]["output_hidden_states"] is False
    assert backbone.calls[0]["return_dict"] is True
    assert torch.equal(
        backbone.calls[0]["attention_mask"],
        batch.qwen_inputs["attention_mask"],
    )
    torch.testing.assert_close(
        output.total_loss,
        output.code_loss
        + 0.5 * output.dense_feature_loss
        + 0.5 * output.grounding_loss
        + 0.2 * output.counterfactual_loss
        + 0.1 * output.temporal_loss,
    )
    for value in output.__dict__.values():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all()


def test_planner_visual_logits_are_bounded_and_never_use_qwen_output_head(
    monkeypatch,
) -> None:
    import qwen35_planx.losses as losses

    planner, backbone, batch, _ = make_planner_and_batch()
    seen: list[int] = []
    original = losses.F.linear

    def recording_linear(input: torch.Tensor, weight: torch.Tensor, *args):
        if weight is backbone.embedding.weight:
            seen.append(int(input.shape[0]))
        return original(input, weight, *args)

    monkeypatch.setattr(losses.F, "linear", recording_linear)
    planner(batch)

    assert seen
    assert max(seen) <= 64
    assert sum(seen) == batch.code_targets.numel()


def test_planner_bypasses_a_conditional_generation_wrapper() -> None:
    from qwen35_planx.planner import GroundedQwen35Planner

    _, base_model, batch, _ = make_planner_and_batch()
    wrapper = FakeConditionalWrapper(base_model)
    planner = GroundedQwen35Planner._from_test_components(
        backbone=wrapper,
        visual_embedding_weight=base_model.embedding.weight,
        codebook=torch.randn(13, 7),
        hidden_dim=8,
        text_dim=6,
    )

    output = planner(batch)

    assert output.codes.shape == (batch.size, 4, 4)
    assert wrapper.public_forward_calls == 0
    assert len(base_model.calls) == 1


def test_production_planner_rejects_nonreleased_geometry() -> None:
    from qwen35_planx.planner import GroundedQwen35Planner

    backbone = FakeBackbone(vocab_size=13, hidden_dim=8, sequence_length=35)

    with pytest.raises(ValueError, match="released.*geometry"):
        GroundedQwen35Planner.from_components(
            backbone=backbone,
            visual_embedding_weight=backbone.embedding.weight,
            codebook=torch.randn(13, 7),
            hidden_dim=8,
            text_dim=6,
        )


def test_production_planner_exact_released_shapes_on_meta_device() -> None:
    from qwen35_planx.planner import GroundedQwen35Planner

    class MetaBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(65_536, 2_048)
            self.calls = 0

        def forward(self, input_ids: torch.Tensor, **kwargs):
            self.calls += 1
            assert kwargs["output_hidden_states"] is False
            assert kwargs["return_dict"] is True
            return SimpleNamespace(
                last_hidden_state=torch.empty(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    2_048,
                )
            )

    with torch.device("meta"):
        backbone = MetaBackbone()
        wrapper = FakeConditionalWrapper(backbone)
        planner = GroundedQwen35Planner.from_components(
            backbone=wrapper,
            visual_embedding_weight=backbone.embedding.weight,
            codebook=torch.empty(65_536, 1_536),
            hidden_dim=2_048,
            text_dim=1_152,
        )
        camera_batch = 2
        tokens = 729
        sequence_length = 2 * 4 * tokens + 3
        batch = GroundedPlannerBatch(
            qwen_inputs={
                "input_ids": torch.empty(
                    camera_batch,
                    sequence_length,
                    dtype=torch.long,
                ),
                "attention_mask": torch.empty(
                    camera_batch,
                    sequence_length,
                    dtype=torch.long,
                ),
            },
            code_targets=torch.empty(
                camera_batch,
                4 * tokens,
                dtype=torch.long,
            ),
            pre_positions=torch.empty(
                camera_batch,
                4 * tokens,
                dtype=torch.long,
            ),
            post_positions=torch.empty(
                camera_batch,
                4 * tokens,
                dtype=torch.long,
            ),
            field_positions=torch.empty(camera_batch, 3, dtype=torch.long),
            field_mask=torch.empty(camera_batch, 3, dtype=torch.bool),
            relevance_targets=torch.empty(camera_batch, 4, 3, tokens),
            relevance_confidence=torch.empty(camera_batch, 4, 3),
            flow_targets=torch.empty(camera_batch, 3, tokens, 3),
            phrase_embeddings=torch.empty(camera_batch, 3, 1_152),
            counterfactual_embeddings=torch.empty(
                camera_batch,
                3,
                1,
                1_152,
            ),
            counterfactual_mask=torch.empty(
                camera_batch,
                3,
                1,
                dtype=torch.bool,
            ),
        )

        output = planner(batch)

    assert output.codes.shape == (2, 4, 729)
    assert output.code_embeddings.shape == (2, 4, 729, 1_536)
    assert output.post_hidden.shape == (2, 4, 729, 2_048)
    assert output.predicted_phrase_embeddings.shape == (2, 3, 1_152)
    assert output.visual_regression.shape == (2, 2_916, 1_536)
    assert output.semantic_features.shape == (2, 4, 729, 1_152)
    assert output.relevance_logits.shape == (2, 4, 3, 729)
    assert output.fusion_gate.shape == (2, 4, 729, 1)
    assert output.times.shape == (4,)
    assert output.total_loss.shape == ()
    assert backbone.calls == 1
    assert wrapper.public_forward_calls == 0


def test_output_unflattens_only_per_camera_tensors() -> None:
    planner, _, batch, _ = make_planner_and_batch()

    flat = planner(batch)
    output = flat.unflatten_cameras(batch_size=1)

    assert output.codes.shape == (1, 2, 4, 4)
    assert output.code_embeddings.shape == (1, 2, 4, 4, 7)
    assert output.post_hidden.shape == (1, 2, 4, 4, 8)
    assert output.predicted_phrase_embeddings.shape == (1, 2, 3, 6)
    assert output.debug_pre_positions.shape == (1, 2, 16)
    assert output.times is flat.times
    assert output.total_loss is flat.total_loss


def test_gradients_reach_qwen_visual_rows_and_every_head_but_not_codebook() -> None:
    planner, backbone, batch, original_codebook = make_planner_and_batch()

    output = planner(batch)
    output.total_loss.backward()

    assert backbone.sequence_states.grad is not None
    assert torch.count_nonzero(backbone.sequence_states.grad) > 0
    assert backbone.embedding.weight.grad is not None
    assert torch.count_nonzero(backbone.embedding.weight.grad) > 0
    for module in (
        planner.visual_regression,
        planner.semantic_projection,
        planner.phrase_projection,
        planner.grounding_query,
        planner.fusion_gate,
    ):
        parameters = tuple(module.parameters())
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)
    assert original_codebook.grad is None
    assert planner.codebook.requires_grad is False


def test_all_zero_auxiliary_supervision_remains_finite() -> None:
    planner, _, batch, _ = make_planner_and_batch()
    batch.relevance_targets.fill_(float("nan"))
    batch.relevance_confidence.zero_()
    batch.field_mask.zero_()
    batch.phrase_embeddings.fill_(float("nan"))
    batch.counterfactual_embeddings.fill_(float("nan"))
    batch.counterfactual_mask.zero_()
    batch.flow_targets.zero_()

    output = planner(batch)

    assert torch.isfinite(output.total_loss)
    assert torch.isfinite(output.dense_feature_loss)
    assert torch.isfinite(output.grounding_loss)
    assert torch.isfinite(output.counterfactual_loss)
    assert torch.isfinite(output.temporal_loss)
    torch.testing.assert_close(
        output.grounding_loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        output.counterfactual_loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        output.temporal_loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )


class _CheckpointableLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(5, 5)
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, kwargs=None) -> None:
        self.gradient_checkpointing_kwargs = kwargs


class _GroupedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = nn.Linear(5, 5)
        self.language_model = _CheckpointableLanguage()
        self.embed_tokens = nn.Embedding(20, 5)
        self.lm_head = nn.Linear(5, 20, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens


class _GroupedPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _GroupedBackbone()
        self.visual_regression = nn.Linear(5, 3)
        self.semantic_projection = nn.Linear(5, 4)
        self.phrase_projection = nn.Linear(5, 4)
        self.grounding_query = nn.Linear(5, 4, bias=False)
        self.fusion_gate = nn.Sequential(nn.Linear(8, 2), nn.SiLU(), nn.Linear(2, 1))


class _SaveableArtifact:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "artifact.json").write_text(
            json.dumps({"kind": self.kind}),
            encoding="utf-8",
        )


class _FakeScaler:
    def __init__(self) -> None:
        self.scale = torch.tensor(1024.0)
        self.growth_tracker = torch.tensor(7, dtype=torch.int32)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "scale": self.scale.clone(),
            "growth_tracker": self.growth_tracker.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.scale.copy_(state["scale"])
        self.growth_tracker.copy_(state["growth_tracker"])


def _tiny_checkpoint_metadata() -> dict[str, object]:
    return {
        "format_version": 1,
        "planner_backend": "qwen35_planx_grounded",
        "visual_vocab_size": 13,
        "visual_token_start_id": 7,
        "visual_token_end_id": 20,
        "hindsight_cache_hash": "cache-sha256",
        "ta_tok_hash": "released-ta-sha256",
        "tokenizer_hash": "tokenizer-sha256",
        "base_model_hash": "base-model-sha256",
    }


def test_optimizer_groups_are_exact_exhaustive_and_duplicate_free() -> None:
    from qwen35_planx.cli.train_semantic_planner import build_optimizer_groups

    planner = _GroupedPlanner()
    groups = build_optimizer_groups(
        planner,
        visual_token_start_id=7,
        qwen_language_lr=1e-5,
        qwen_vision_lr=5e-6,
        head_lr=1e-4,
    )

    assert {group["name"]: group["lr"] for group in groups} == {
        "qwen_language": 1e-5,
        "qwen_vision": 5e-6,
        "visual_vocab_and_prediction_head": 1e-4,
        "semantic_phrase_grounding_fusion_heads": 1e-4,
    }
    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    trainable_ids = {
        id(parameter)
        for parameter in planner.parameters()
        if parameter.requires_grad
    }
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == trainable_ids

    names = {
        id(parameter): name
        for name, parameter in planner.named_parameters()
    }
    grouped_names = {
        group["name"]: {names[id(parameter)] for parameter in group["params"]}
        for group in groups
    }
    assert grouped_names["qwen_vision"] == {
        "backbone.vision_model.weight",
        "backbone.vision_model.bias",
    }
    assert "backbone.embed_tokens.weight" in grouped_names[
        "visual_vocab_and_prediction_head"
    ]
    assert planner.backbone.lm_head.weight.requires_grad is False
    assert {
        name
        for name in grouped_names["visual_vocab_and_prediction_head"]
        if name.startswith("visual_regression.")
    } == {"visual_regression.weight", "visual_regression.bias"}
    assert grouped_names["semantic_phrase_grounding_fusion_heads"] == {
        "semantic_projection.weight",
        "semantic_projection.bias",
        "phrase_projection.weight",
        "phrase_projection.bias",
        "grounding_query.weight",
        "fusion_gate.0.weight",
        "fusion_gate.0.bias",
        "fusion_gate.2.weight",
        "fusion_gate.2.bias",
    }


def test_selective_checkpointing_and_effective_batch_contract() -> None:
    from qwen35_planx.cli.train_semantic_planner import (
        enable_selective_qwen_activation_checkpointing,
        validate_effective_global_batch,
    )

    planner = _GroupedPlanner()
    validate_effective_global_batch(
        per_device_batch=4,
        num_processes=8,
        grad_accum=8,
    )
    with pytest.raises(ValueError, match="effective global batch must be 256, got 32"):
        validate_effective_global_batch(
            per_device_batch=2,
            num_processes=8,
            grad_accum=2,
        )

    enable_selective_qwen_activation_checkpointing(planner)

    assert planner.backbone.language_model.gradient_checkpointing_kwargs == {
        "use_reentrant": False
    }
    assert not hasattr(planner.backbone.vision_model, "gradient_checkpointing_kwargs")


def test_atomic_checkpoint_resume_restores_all_training_state(tmp_path: Path) -> None:
    from qwen35_planx.cli.train_semantic_planner import (
        REQUIRED_CHECKPOINT_ENTRIES,
        _capture_rng_state,
        build_optimizer_groups,
        load_planner_checkpoint,
        save_planner_checkpoint,
    )

    torch.manual_seed(27)
    planner = _GroupedPlanner()
    groups = build_optimizer_groups(
        planner,
        visual_token_start_id=7,
        qwen_language_lr=1e-5,
        qwen_vision_lr=5e-6,
        head_lr=1e-4,
    )
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, 1.0 - (step / 10.0)),
    )
    loss = sum(parameter.square().sum() for parameter in planner.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected_model = {
        name: value.detach().clone()
        for name, value in planner.state_dict().items()
    }
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())

    base_dir = tmp_path / "base-qwen"
    released_ta_dir = tmp_path / "released-ta"
    base_dir.mkdir()
    released_ta_dir.mkdir()
    rank_zero_rng = _capture_rng_state()
    torch.manual_seed(902)
    rank_one_rng = _capture_rng_state()
    checkpoint = save_planner_checkpoint(
        output_dir=tmp_path / "runs",
        step=1,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        processor=_SaveableArtifact("processor"),
        tokenizer=_SaveableArtifact("tokenizer"),
        metadata=_tiny_checkpoint_metadata(),
        codebook=torch.randn(13, 3),
        scaler=None,
        optimizer_group_lrs={
            group["name"]: group["lr"] for group in groups
        },
        base_model_dir=base_dir,
        released_ta_dir=released_ta_dir,
        allow_test_artifacts=True,
        rng_states_by_rank=(rank_zero_rng, rank_one_rng),
    )

    assert checkpoint.name == "step_000001"
    assert all((checkpoint / name).exists() for name in REQUIRED_CHECKPOINT_ENTRIES)
    assert (checkpoint / "scaler.pt").is_file()
    assert (checkpoint / "rng_state.pt").is_file()
    assert not tuple(checkpoint.parent.glob(".*.incomplete-*"))
    assert not tuple(base_dir.iterdir())
    assert not tuple(released_ta_dir.iterdir())

    for parameter in planner.parameters():
        parameter.data.zero_()
    optimizer.state.clear()
    scheduler.last_epoch = -1

    resumed_step = load_planner_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_metadata=_tiny_checkpoint_metadata(),
        allow_test_artifacts=True,
        process_index=1,
        world_size=2,
    )

    assert resumed_step == 1
    for name, value in planner.state_dict().items():
        torch.testing.assert_close(value, expected_model[name])
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for key, state in actual_optimizer["state"].items():
        assert state.keys() == expected_optimizer["state"][key].keys()
        for name, value in state.items():
            expected = expected_optimizer["state"][key][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected)
            else:
                assert value == expected
    assert scheduler.state_dict() == expected_scheduler
    torch.testing.assert_close(torch.random.get_rng_state(), rank_one_rng["torch_cpu"])


def test_resume_fails_closed_before_mutation_on_missing_or_mismatch(
    tmp_path: Path,
) -> None:
    from qwen35_planx.cli.train_semantic_planner import (
        _artifact_hash,
        _capture_rng_state,
        build_optimizer_groups,
        load_planner_checkpoint,
        save_planner_checkpoint,
    )

    planner = _GroupedPlanner()
    groups = build_optimizer_groups(planner, visual_token_start_id=7)
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = _FakeScaler()
    with pytest.raises(ValueError, match="current.*optimizer.*scheduler"):
        save_planner_checkpoint(
            output_dir=tmp_path / "bad-step",
            step=3,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            processor=_SaveableArtifact("processor"),
            tokenizer=_SaveableArtifact("tokenizer"),
            metadata=_tiny_checkpoint_metadata(),
            codebook=torch.randn(13, 3),
            scaler=scaler,
            optimizer_group_lrs={
                group["name"]: group["lr"] for group in groups
            },
            allow_test_artifacts=True,
        )
    sum(parameter.square().sum() for parameter in planner.parameters()).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = save_planner_checkpoint(
        output_dir=tmp_path / "runs",
        step=1,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        processor=_SaveableArtifact("processor"),
        tokenizer=_SaveableArtifact("tokenizer"),
        metadata=_tiny_checkpoint_metadata(),
        codebook=torch.randn(13, 3),
        scaler=scaler,
        optimizer_group_lrs={
            group["name"]: group["lr"] for group in groups
        },
        allow_test_artifacts=True,
    )
    original = {
        name: value.detach().clone()
        for name, value in planner.state_dict().items()
    }

    mismatched = dict(_tiny_checkpoint_metadata())
    mismatched["hindsight_cache_hash"] = "different-cache"
    with pytest.raises(ValueError, match="hindsight_cache_hash"):
        load_planner_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_metadata=mismatched,
            allow_test_artifacts=True,
        )
    for name, value in planner.state_dict().items():
        torch.testing.assert_close(value, original[name])

    trainer_state_path = checkpoint / "trainer_state.json"
    original_trainer_state = trainer_state_path.read_text(encoding="utf-8")
    trainer_state = json.loads(original_trainer_state)
    trainer_state["optimizer_groups"]["qwen_language"] = 9e-5
    trainer_state_path.write_text(
        json.dumps(trainer_state),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="optimizer group learning rates"):
        load_planner_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_metadata=_tiny_checkpoint_metadata(),
            allow_test_artifacts=True,
        )
    trainer_state_path.write_text(original_trainer_state, encoding="utf-8")

    # A topology failure discovered after the checkpoint's artifact manifest
    # has passed must still leave every runtime object untouched.
    for parameter in planner.parameters():
        parameter.data.zero_()
    sentinel_model = {
        name: value.detach().clone()
        for name, value in planner.state_dict().items()
    }
    sentinel_optimizer = copy.deepcopy(optimizer.state_dict())
    sentinel_scheduler = copy.deepcopy(scheduler.state_dict())
    sentinel_scaler = copy.deepcopy(scaler.state_dict())
    sentinel_rng = _capture_rng_state()
    optimizer_path = checkpoint / "optimizer.pt"
    corrupted_optimizer = torch.load(
        optimizer_path,
        weights_only=True,
        map_location="cpu",
    )
    corrupted_optimizer["param_groups"][0]["params"].pop()
    torch.save(corrupted_optimizer, optimizer_path)
    trainer_state = json.loads(original_trainer_state)
    trainer_state["artifact_hashes"]["optimizer.pt"] = _artifact_hash(
        optimizer_path
    )
    trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer group topology mismatch"):
        load_planner_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_metadata=_tiny_checkpoint_metadata(),
            allow_test_artifacts=True,
        )
    for name, value in planner.state_dict().items():
        torch.testing.assert_close(value, sentinel_model[name])
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == sentinel_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == sentinel_optimizer["state"].keys()
    for key, state in actual_optimizer["state"].items():
        for name, value in state.items():
            expected = sentinel_optimizer["state"][key][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected)
            else:
                assert value == expected
    assert scheduler.state_dict() == sentinel_scheduler
    actual_scaler = scaler.state_dict()
    assert actual_scaler.keys() == sentinel_scaler.keys()
    for name, value in actual_scaler.items():
        torch.testing.assert_close(value, sentinel_scaler[name])
    actual_rng = _capture_rng_state()
    for name in sentinel_rng:
        expected = sentinel_rng[name]
        actual = actual_rng[name]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected)
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected):
                torch.testing.assert_close(actual_item, expected_item)
        else:
            assert actual == expected

    (checkpoint / "rng_state.pt").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete planner checkpoint"):
        load_planner_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_metadata=_tiny_checkpoint_metadata(),
            allow_test_artifacts=True,
        )


def test_stage_one_schedule_and_batch_candidates_preserve_exact_contract() -> None:
    from qwen35_planx.cli.train_semantic_planner import (
        PlannerTrainingConfig,
        cosine_lr_multiplier,
        estimate_per_gpu_batch_candidates,
    )

    config = PlannerTrainingConfig.from_mapping(
        {
            "tiny_smoke": True,
            "output_dir": "/tmp/planx-tiny",
            "per_device_batch": 256,
            "gradient_accumulation_steps": 1,
        }
    )
    assert config.max_steps == 30_000
    assert config.warmup_steps == 1_000
    assert config.save_every == 5_000
    assert config.validate_every == 5_000
    assert config.log_every == 20
    assert config.mixed_precision == "bf16"
    assert config.gradient_clip_norm == 1.0
    assert config.tf32 is True
    assert cosine_lr_multiplier(0, warmup_steps=1_000, max_steps=30_000) == 0.0
    assert cosine_lr_multiplier(
        1_000,
        warmup_steps=1_000,
        max_steps=30_000,
    ) == 1.0
    assert cosine_lr_multiplier(
        30_000,
        warmup_steps=1_000,
        max_steps=30_000,
    ) == 0.0

    assert estimate_per_gpu_batch_candidates(
        num_processes=8,
        available_bytes=8 * 1024**3,
        estimated_bytes_per_sample=3 * 1024**3,
    ) == ((2, 16), (1, 32))


def test_tiny_accelerate_training_writes_reloadable_one_step_checkpoint(
    tmp_path: Path,
) -> None:
    from qwen35_planx.cli.train_semantic_planner import (
        main,
        validate_planner_checkpoint,
    )

    config_path = tmp_path / "tiny.json"
    output_dir = tmp_path / "run"
    config_path.write_text(
        json.dumps(
            {
                "tiny_smoke": True,
                "output_dir": str(output_dir),
                "per_device_batch": 256,
                "gradient_accumulation_steps": 1,
                "max_steps": 1,
                "warmup_steps": 0,
                "save_every": 1,
                "validate_every": 1,
                "log_every": 1,
                "num_workers": 0,
            }
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--max-steps", "1"]) == 0

    checkpoint = output_dir / "step_000001"
    _, trainer_state = validate_planner_checkpoint(
        checkpoint,
        expected_metadata=_tiny_checkpoint_metadata(),
        allow_test_artifacts=True,
    )
    assert trainer_state["current_step"] == 1
    assert trainer_state["optimizer_step"] == 1
    assert trainer_state["scheduler_step"] == 1


def test_two_process_short_loader_accumulates_and_saves_rank_rng(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tiny-two-process.json"
    output_dir = tmp_path / "run"
    config_path.write_text(
        json.dumps(
            {
                "tiny_smoke": True,
                "output_dir": str(output_dir),
                "per_device_batch": 64,
                "gradient_accumulation_steps": 2,
                "max_steps": 2,
                "warmup_steps": 0,
                "save_every": 2,
                "validate_every": 2,
                "log_every": 1,
                "num_workers": 0,
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(Path.cwd()),
                str(Path.cwd() / "ge_act"),
                environment.get("PYTHONPATH"),
            ),
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "qwen35_planx.cli.train_semantic_planner",
            "--config",
            str(config_path),
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    checkpoint = output_dir / "step_000002"
    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    assert trainer_state["current_step"] == 2
    assert trainer_state["optimizer_step"] == 2
    assert trainer_state["scheduler_step"] == 2
    rank_rng = torch.load(
        checkpoint / "rng_state.pt",
        weights_only=True,
        map_location="cpu",
    )
    assert rank_rng["world_size"] == 2
    assert len(rank_rng["states"]) == 2
    assert not torch.equal(
        rank_rng["states"][0]["torch_cpu"],
        rank_rng["states"][1]["torch_cpu"],
    )


def test_training_preflight_refuses_incomplete_resume(tmp_path: Path) -> None:
    from qwen35_planx.cli.preflight import (
        collect_planner_training_preflight_errors,
    )
    from qwen35_planx.cli.train_semantic_planner import PlannerTrainingConfig

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    config = PlannerTrainingConfig(
        tiny_smoke=True,
        output_dir=str(tmp_path / "run"),
        resume_from=str(incomplete),
        per_device_batch=256,
        gradient_accumulation_steps=1,
    )

    errors = collect_planner_training_preflight_errors(config)

    assert len(errors) == 1
    assert "incomplete planner checkpoint" in errors[0]
