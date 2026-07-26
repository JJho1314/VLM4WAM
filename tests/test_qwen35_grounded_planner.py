from __future__ import annotations

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
