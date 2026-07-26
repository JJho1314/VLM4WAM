from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen35_planx.planner_dataset import (
    CachedPlannerTargets,
    GroundedPlannerBatch,
)
from qwen35_planx.vocabulary import STRUCTURE_TOKENS, VisualVocabularyLayout


def _provider_layout(*, tokenizer_hash: str = "provider-layout"):
    structure = tuple(
        (token, index) for index, token in enumerate(STRUCTURE_TOKENS, start=4)
    )
    visual_start = 4 + len(STRUCTURE_TOKENS)
    return VisualVocabularyLayout(
        original_vocab_size=4,
        visual_start_id=visual_start,
        visual_end_id=visual_start + 65_536,
        structure_token_ids=structure,
        tokenizer_hash=tokenizer_hash,
        base_embedding_hash="base",
        expanded_embedding_hash="expanded",
    )


def _generated_flat(camera_batch: int):
    from qwen35_planx.decoding import GeneratedGroundedPlan

    return GeneratedGroundedPlan._from_test_components(
        codes=torch.arange(camera_batch * 4 * 729)
        .reshape(camera_batch, 4, 729)
        .remainder(65_536),
        code_embeddings=torch.randn(camera_batch, 4, 729, 3),
        post_hidden=torch.randn(camera_batch, 4, 729, 5),
        predicted_phrase_embeddings=torch.randn(camera_batch, 3, 6),
        semantic_features=torch.randn(camera_batch, 4, 729, 6),
        relevance=torch.full((camera_batch, 4, 3, 729), 1 / 729),
        fusion_gate=torch.sigmoid(torch.randn(camera_batch, 4, 729, 1)),
        times=torch.tensor([0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0]),
    )


class _ProviderPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(2.0))
        self.child = nn.Linear(1, 1)
        self.register_buffer("codebook", torch.zeros(8, 3))
        self.hidden_dim = 5


class _Collator:
    def __init__(self) -> None:
        self.processor = object()
        self.received = None

    def build_teacher_forced(self, current_images, instructions, targets):
        self.received = (current_images, tuple(instructions), targets)
        return SimpleNamespace(size=current_images.shape[0] * 2)


@dataclass
class _TeacherOutput:
    value: torch.Tensor
    batch_size: int | None = None

    def unflatten_cameras(self, batch_size: int):
        return _TeacherOutput(self.value, batch_size=batch_size)


class _TeacherPlanner(_ProviderPlanner):
    def forward(self, batch):
        return _TeacherOutput(self.anchor * batch.size)


def _targets(batch: int = 1) -> CachedPlannerTargets:
    return CachedPlannerTargets(
        codes=torch.zeros(batch, 2, 4, 729, dtype=torch.long),
        relevance=torch.full((batch, 2, 4, 3, 729), 1 / 729),
        relevance_confidence=torch.ones(batch, 2, 4, 3),
        flow=torch.zeros(batch, 2, 3, 729, 3),
        phrase_embeddings=torch.zeros(batch, 3, 1152),
    )


def test_generate_duplicates_camera_rows_and_restores_planner_state(
    monkeypatch,
) -> None:
    import qwen35_planx.provider as provider_module
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    planner = _ProviderPlanner()
    planner.train()
    planner.child.eval()
    original_rope_deltas = object()
    planner.child.rope_deltas = original_rope_deltas
    requires_grad = {
        name: parameter.requires_grad for name, parameter in planner.named_parameters()
    }
    collator = _Collator()
    layout = object()
    received = {}

    def fake_generate(
        actual_planner,
        *,
        current_images,
        instructions,
        camera_names,
        layout,
        processor,
    ):
        assert actual_planner is planner
        assert not actual_planner.training
        assert not actual_planner.child.training
        assert not torch.is_grad_enabled()
        actual_planner.child.rope_deltas = object()
        received.update(
            images=current_images.clone(),
            instructions=tuple(instructions),
            camera_names=tuple(camera_names),
            layout=layout,
            processor=processor,
        )
        return _generated_flat(current_images.shape[0])

    monkeypatch.setattr(provider_module, "generate_grounded_plan", fake_generate)
    provider = Qwen35GroundedPlanProvider._from_test_components(
        planner=planner,
        collator=collator,
        layout=layout,
        condition_dim=7,
    )
    images = torch.arange(2 * 2 * 3 * 2 * 2).reshape(2, 2, 3, 2, 2).float()
    output = provider.generate(images, ("first", "second"))

    assert output.codes.shape == (2, 2, 4, 729)
    assert torch.equal(received["images"], images.reshape(4, 3, 2, 2))
    assert received["instructions"] == ("first", "first", "second", "second")
    assert received["camera_names"] == ("main", "wrist", "main", "wrist")
    assert received["layout"] is layout
    assert received["processor"] is collator.processor
    assert planner.training
    assert not planner.child.training
    assert planner.child.rope_deltas is original_rope_deltas
    assert {
        name: parameter.requires_grad for name, parameter in planner.named_parameters()
    } == requires_grad
    assert torch.is_grad_enabled()


def test_teacher_force_requires_complete_targets_and_remains_differentiable() -> None:
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    planner = _TeacherPlanner()
    collator = _Collator()
    provider = Qwen35GroundedPlanProvider._from_test_components(
        planner=planner,
        collator=collator,
        layout=object(),
        condition_dim=7,
    )
    images = torch.zeros(1, 2, 3, 2, 2)
    targets = _targets()
    output = provider.teacher_force(images, ("open the drawer",), targets)
    assert output.batch_size == 1
    output.value.backward()
    torch.testing.assert_close(planner.anchor.grad, torch.tensor(2.0))
    assert collator.received == (images, ("open the drawer",), targets)

    with pytest.raises(TypeError, match="CachedPlannerTargets"):
        provider.teacher_force(images, ("open the drawer",), object())


def test_fuse_is_shared_for_equivalent_generated_and_teacher_forced_inputs() -> None:
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    torch.manual_seed(17)
    provider = Qwen35GroundedPlanProvider._from_test_components(
        planner=_ProviderPlanner(),
        collator=_Collator(),
        layout=object(),
        condition_dim=7,
    )
    plan = _generated_flat(camera_batch=2)
    generated = SimpleNamespace(
        codes=plan.codes.reshape(1, 2, 4, 729),
        code_embeddings=plan.code_embeddings.reshape(1, 2, 4, 729, 3),
        post_hidden=plan.post_hidden.reshape(1, 2, 4, 729, 5),
        fusion_gate=plan.fusion_gate.reshape(1, 2, 4, 729, 1),
        times=plan.times,
    )
    teacher = SimpleNamespace(**generated.__dict__)

    generated_fused = provider.fuse(generated)
    teacher_fused = provider.fuse(teacher)

    assert generated_fused.shape == (1, 2, 4, 729, 7)
    torch.testing.assert_close(generated_fused, teacher_fused)
    assert provider.visual_adapter.in_features == 3
    assert provider.visual_adapter.out_features == 7
    assert provider.hidden_adapter.in_features == 5
    assert provider.hidden_adapter.out_features == 7
    assert provider.position_encoder.spatial_embedding.shape == (729, 7)
    assert provider.position_encoder.time_embedding.shape == (4, 7)


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0])
def test_scale_gradient_is_forward_identity_and_backward_scale(scale: float) -> None:
    from qwen35_planx.provider import scale_gradient

    x = torch.tensor([2.0, -3.0], requires_grad=True)
    y = scale_gradient(x, scale)
    assert torch.equal(y, x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, scale))


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan"), True])
def test_scale_gradient_rejects_invalid_values(scale) -> None:
    from qwen35_planx.provider import scale_gradient

    with pytest.raises((TypeError, ValueError), match="scale"):
        scale_gradient(torch.ones(1), scale)


def test_fuse_scales_only_gradients_crossing_the_planner_boundary() -> None:
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    torch.manual_seed(23)
    provider = Qwen35GroundedPlanProvider._from_test_components(
        planner=_ProviderPlanner(),
        collator=_Collator(),
        layout=object(),
        condition_dim=2,
    )
    code = torch.randn(1, 2, 4, 729, 3, requires_grad=True)
    hidden = torch.randn(1, 2, 4, 729, 5, requires_grad=True)
    gate = torch.rand(1, 2, 4, 729, 1, requires_grad=True)
    plan = SimpleNamespace(
        codes=torch.zeros(1, 2, 4, 729, dtype=torch.long),
        code_embeddings=code,
        post_hidden=hidden,
        fusion_gate=gate,
        times=torch.tensor([0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0]),
    )
    provider.fuse(plan, qwen_gradient_scale=0.25).sum().backward()
    scaled_grads = (code.grad.clone(), hidden.grad.clone(), gate.grad.clone())
    adapter_grad = provider.visual_adapter.weight.grad.clone()

    provider.zero_grad(set_to_none=True)
    for tensor in (code, hidden, gate):
        tensor.grad = None
    provider.fuse(plan, qwen_gradient_scale=1.0).sum().backward()

    torch.testing.assert_close(code.grad * 0.25, scaled_grads[0])
    torch.testing.assert_close(hidden.grad * 0.25, scaled_grads[1])
    torch.testing.assert_close(gate.grad * 0.25, scaled_grads[2])
    torch.testing.assert_close(provider.visual_adapter.weight.grad, adapter_grad)


def test_public_provider_enforces_released_geometry_and_matching_layout() -> None:
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    with pytest.raises(ValueError, match="released"):
        Qwen35GroundedPlanProvider(
            planner=_ProviderPlanner(),
            collator=_Collator(),
            layout=object(),
            condition_dim=2,
        )

    class ReleasedPlanner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "codebook",
                torch.empty(65_536, 1_536, device="meta"),
            )
            self.hidden_dim = 2_048
            self.text_dim = 1_152
            self._enforce_released_geometry = True

    collator = _Collator()
    collator.layout = _provider_layout(tokenizer_hash="collator")
    with pytest.raises(ValueError, match="layout"):
        Qwen35GroundedPlanProvider(
            planner=ReleasedPlanner(),
            collator=collator,
            layout=_provider_layout(tokenizer_hash="provider"),
            condition_dim=2,
        )


def test_teacher_force_refreshes_visual_view_after_provider_to() -> None:
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    layout = _provider_layout()

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(layout.visual_end_id, 1)

        def get_input_embeddings(self):
            return self.embedding

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Base()

    class Planner(_TeacherPlanner):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Wrapper()
            self.visual_embedding_weight = self.backbone.model.embedding.weight[
                layout.visual_start_id : layout.visual_end_id
            ]
            self.hidden_dim = 1

        def _language_backbone(self):
            return self.backbone.model

        def forward(self, batch):
            current = self.backbone.model.embedding.weight[
                layout.visual_start_id : layout.visual_end_id
            ]
            assert self.visual_embedding_weight.dtype == current.dtype
            assert self.visual_embedding_weight.data_ptr() == current.data_ptr()
            return super().forward(batch)

    planner = Planner()
    collator = _Collator()
    provider = Qwen35GroundedPlanProvider._from_test_components(
        planner=planner,
        collator=collator,
        layout=layout,
        condition_dim=2,
    )
    provider.to(dtype=torch.float64)
    output = provider.teacher_force(
        torch.zeros(1, 2, 3, 2, 2),
        ("open the drawer",),
        _targets(),
    )
    assert output.batch_size == 1


def test_teacher_force_moves_every_collated_tensor_to_planner_device() -> None:
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    class Collator(_Collator):
        def build_teacher_forced(self, current_images, instructions, targets):
            del current_images, instructions, targets
            values = {
                "qwen_inputs": {
                    "input_ids": torch.zeros(2, 1, dtype=torch.long),
                    "pixel_values": torch.zeros(2, 3, 1, 1),
                },
                "code_targets": torch.zeros(2, 1, dtype=torch.long),
                "pre_positions": torch.zeros(2, 1, dtype=torch.long),
                "post_positions": torch.zeros(2, 1, dtype=torch.long),
                "field_positions": torch.zeros(2, 3, dtype=torch.long),
                "field_mask": torch.ones(2, 3, dtype=torch.bool),
                "relevance_targets": torch.zeros(2, 1),
                "relevance_confidence": torch.zeros(2, 1),
                "flow_targets": torch.zeros(2, 1),
                "phrase_embeddings": torch.zeros(2, 1),
                "counterfactual_embeddings": torch.zeros(2, 1),
                "counterfactual_mask": torch.zeros(2, 1, dtype=torch.bool),
            }
            return GroundedPlannerBatch(**values)

    class Planner(_ProviderPlanner):
        def __init__(self) -> None:
            super().__init__()
            self.codebook = torch.empty(8, 3, device="meta")

        def forward(self, batch):
            tensors = tuple(batch.qwen_inputs.values()) + tuple(
                getattr(batch, name)
                for name in batch.__dataclass_fields__
                if name != "qwen_inputs"
            )
            assert all(value.device.type == "meta" for value in tensors)
            return _TeacherOutput(torch.empty((), device="meta"))

    provider = Qwen35GroundedPlanProvider._from_test_components(
        planner=Planner(),
        collator=Collator(),
        layout=object(),
        condition_dim=2,
    )
    output = provider.teacher_force(
        torch.zeros(1, 2, 3, 2, 2),
        ("open the drawer",),
        _targets(),
    )
    assert output.value.device.type == "meta"


def test_fuse_rejects_non_tensor_codes_cleanly() -> None:
    from qwen35_planx.provider import Qwen35GroundedPlanProvider

    provider = Qwen35GroundedPlanProvider._from_test_components(
        planner=_ProviderPlanner(),
        collator=_Collator(),
        layout=object(),
        condition_dim=2,
    )
    malformed = SimpleNamespace(
        codes=[],
        code_embeddings=torch.zeros(1),
        post_hidden=torch.zeros(1),
        fusion_gate=torch.zeros(1),
        times=torch.zeros(4),
    )
    with pytest.raises(TypeError, match="codes"):
        provider.fuse(malformed)
    malformed.codes = torch.tensor(0)
    with pytest.raises(ValueError, match=r"\[B,2,4,729\]"):
        provider.fuse(malformed)
