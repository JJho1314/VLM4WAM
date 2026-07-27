from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from qwen35_baton.data import BatonPlannerBatch
from qwen35_baton.model import (
    BatonPlannerOutput,
    BatonQwen35Planner,
    PlanTokenEmbeddingAdapter,
)
from qwen35_baton.ownership import configure_stage1_trainable_modules
from qwen35_baton.query_tower import QueryTowerOutput


ADDED_TOKEN_IDS = (40, 41, 42, 43, 44, 45, 46)
PLAN_PAD_TOKEN_ID = ADDED_TOKEN_IDS[5]


class TinyLanguageModel(nn.Module):
    def __init__(self, embedding: nn.Module, width: int) -> None:
        super().__init__()
        self.embed_tokens = embedding
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(12))
        self.norm = nn.LayerNorm(width)


class TinyMultimodalBase(nn.Module):
    def __init__(self, embedding: nn.Module, width: int) -> None:
        super().__init__()
        self.language_model = TinyLanguageModel(embedding, width)
        self.visual = nn.Linear(width, width)
        self.forward_calls = 0
        self.received_kwargs: dict[str, object] | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        self.forward_calls += 1
        self.received_kwargs = {
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "use_cache": use_cache,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        hidden = self.language_model.embed_tokens(input_ids)
        row_signal = input_ids[:, :1].to(hidden.dtype).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden + row_signal)


class TinyQwen(nn.Module):
    def __init__(self, *, vocab_size: int = 64, width: int = 8) -> None:
        super().__init__()
        embedding = nn.Embedding(vocab_size, width)
        self.model = TinyMultimodalBase(embedding, width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, embedding: nn.Module) -> None:
        self.model.language_model.embed_tokens = embedding

    def forward(self, **_: object) -> None:
        raise AssertionError("conditional-generation LM/logits path must not run")


class TinyQueryTower(nn.Module):
    qwen_dim = 8
    query_dim = 1024
    num_frames = 4
    tokens_per_frame = 256
    num_cameras = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        qwen_states: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        return_attention_maps: bool = False,
    ) -> QueryTowerOutput:
        signal = (
            qwen_states[..., :1] * self.scale
            + camera_ids[:, None, None, None].to(qwen_states) * 1000
        )
        return QueryTowerOutput(
            hidden_states=signal.expand(-1, -1, -1, self.query_dim),
            cross_attention_maps=(
                (torch.zeros(qwen_states.shape[0], 1024, 1024),)
                if return_attention_maps
                else None
            ),
        )


def make_batch(*, batch_size: int = 2) -> BatonPlannerBatch:
    rows = batch_size * 4
    input_ids = torch.full((rows, 1026), PLAN_PAD_TOKEN_ID, dtype=torch.long)
    input_ids[:, 0] = torch.arange(1, rows + 1)
    input_ids[:, -1] = 2
    plan_positions = torch.arange(1, 1025).expand(rows, -1).clone()
    labels = tuple(
        (condition, sample, camera)
        for condition in ("positive", "negative")
        for sample in range(batch_size)
        for camera in ("main", "wrist")
    )
    return BatonPlannerBatch(
        qwen_inputs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        },
        plan_positions=plan_positions,
        current_images=torch.zeros(batch_size, 2, 3, 2, 2, dtype=torch.uint8),
        future_images=torch.zeros(
            batch_size, 2, 4, 3, 2, 2, dtype=torch.uint8
        ),
        instructions=tuple(f"task {index}" for index in range(batch_size)),
        negative_instructions=tuple(
            f"wrong task {index}" for index in range(batch_size)
        ),
        row_labels=labels,
    )


def make_planner() -> tuple[BatonQwen35Planner, TinyQwen]:
    qwen = TinyQwen()
    planner = BatonQwen35Planner(
        qwen,
        added_token_ids=ADDED_TOKEN_IDS,
        query_tower=TinyQueryTower(),
    )
    planner.sem_mlp = nn.Identity()
    return planner, qwen


def make_unmodified_planner() -> tuple[BatonQwen35Planner, TinyQwen]:
    qwen = TinyQwen()
    return (
        BatonQwen35Planner(
            qwen,
            added_token_ids=ADDED_TOKEN_IDS,
            query_tower=TinyQueryTower(),
        ),
        qwen,
    )


def test_model_splits_positive_and_negative_predictions() -> None:
    planner, qwen = make_planner()

    output = planner(make_batch())

    assert isinstance(output, BatonPlannerOutput)
    assert output.positive.shape == (2, 2, 4, 256, 1024)
    assert output.negative is not None
    assert output.negative.shape == (2, 2, 4, 256, 1024)
    assert output.flat.shape == (8, 4, 256, 1024)
    assert qwen.model.forward_calls == 1


def test_model_calls_multimodal_base_without_lm_logits_or_cache() -> None:
    planner, qwen = make_planner()
    batch = make_batch(batch_size=1)
    pixel_values = torch.randn(4, 3, 2, 2)
    image_grid_thw = torch.ones(4, 3, dtype=torch.long)
    mm_token_type_ids = torch.zeros_like(batch.qwen_inputs["input_ids"])
    qwen_inputs = {
        **batch.qwen_inputs,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }

    planner.forward_rows(
        qwen_inputs,
        batch.plan_positions,
        camera_ids=torch.tensor([0, 1, 0, 1]),
    )

    assert qwen.model.forward_calls == 1
    assert qwen.model.received_kwargs == {
        "attention_mask": batch.qwen_inputs["attention_mask"],
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "use_cache": False,
        "output_hidden_states": False,
        "return_dict": True,
    }


def test_plan_token_adapter_changes_only_added_rows() -> None:
    base = nn.Embedding(64, 8)
    adapter = PlanTokenEmbeddingAdapter(base, ADDED_TOKEN_IDS)
    with torch.no_grad():
        adapter.plan_embeddings.weight.add_(10)
    input_ids = torch.tensor([[0, 40, 12, 45, 46]])

    embeddings = adapter(input_ids)
    base_embeddings = base(input_ids)
    ordinary = ~torch.isin(input_ids, torch.tensor(ADDED_TOKEN_IDS))

    torch.testing.assert_close(embeddings[ordinary], base_embeddings[ordinary])
    assert not torch.allclose(embeddings[~ordinary], base_embeddings[~ordinary])
    embeddings.sum().backward()
    assert adapter.plan_embeddings.weight.grad is not None
    assert base.weight.grad is None


def test_permuting_wrist_rows_changes_only_wrist_predictions() -> None:
    planner, _ = make_planner()
    batch = make_batch()
    baseline = planner(batch).positive
    changed_ids = batch.qwen_inputs["input_ids"].clone()
    changed_ids[[1, 3]] = changed_ids[[3, 1]]
    changed_batch = BatonPlannerBatch(
        qwen_inputs={**batch.qwen_inputs, "input_ids": changed_ids},
        plan_positions=batch.plan_positions,
        current_images=batch.current_images,
        future_images=batch.future_images,
        instructions=batch.instructions,
        negative_instructions=batch.negative_instructions,
        row_labels=batch.row_labels,
    )

    changed = planner(changed_batch).positive

    torch.testing.assert_close(changed[:, 0], baseline[:, 0])
    torch.testing.assert_close(changed[0, 1], baseline[1, 1])
    torch.testing.assert_close(changed[1, 1], baseline[0, 1])


@pytest.mark.parametrize("count", (1023, 1025))
def test_wrong_plan_pad_count_fails_before_qwen(count: int) -> None:
    planner, qwen = make_planner()
    batch = make_batch(batch_size=1)
    input_ids = batch.qwen_inputs["input_ids"].clone()
    input_ids[:, 1:] = 3
    input_ids[:, 1 : count + 1] = PLAN_PAD_TOKEN_ID
    positions = torch.arange(1, 1025).expand(input_ids.shape[0], -1)
    malformed = BatonPlannerBatch(
        qwen_inputs={**batch.qwen_inputs, "input_ids": input_ids},
        plan_positions=positions,
        current_images=batch.current_images,
        future_images=batch.future_images,
        instructions=batch.instructions,
        negative_instructions=batch.negative_instructions,
        row_labels=batch.row_labels,
    )

    with pytest.raises(ValueError, match="each Qwen row.*exactly 1024"):
        planner(malformed)

    assert qwen.model.forward_calls == 0


def test_forward_rows_requires_positions_to_match_plan_pad_tokens() -> None:
    planner, qwen = make_planner()
    batch = make_batch(batch_size=1)
    positions = batch.plan_positions.clone()
    positions[:, 0] = 0

    with pytest.raises(ValueError, match="positions.*PLAN_PAD"):
        planner.forward_rows(
            batch.qwen_inputs,
            positions,
            camera_ids=torch.tensor([0, 1, 0, 1]),
        )

    assert qwen.model.forward_calls == 0


def test_forward_can_return_flat_query_tower_attention_maps() -> None:
    planner, _ = make_planner()

    output = planner(make_batch(batch_size=1), return_attention_maps=True)

    assert output.cross_attention_maps is not None
    assert output.cross_attention_maps[0].shape == (4, 1024, 1024)


def test_sem_mlp_has_exact_1024_2048_1024_structure_without_output_norm() -> None:
    planner, _ = make_unmodified_planner()

    assert len(planner.sem_mlp) == 3
    first, activation, output = planner.sem_mlp
    assert isinstance(first, nn.Linear)
    assert (first.in_features, first.out_features) == (1024, 2048)
    assert isinstance(activation, nn.GELU)
    assert isinstance(output, nn.Linear)
    assert (output.in_features, output.out_features) == (2048, 1024)


def test_planner_state_dict_round_trips_overlay_and_frozen_base() -> None:
    source, _ = make_planner()
    with torch.no_grad():
        source.frozen_base_embedding.weight.add_(3)
        source.plan_token_adapter.plan_embeddings.weight.sub_(7)
    adapter_keys = set(source.plan_token_adapter.state_dict())
    state = {name: value.clone() for name, value in source.state_dict().items()}

    assert adapter_keys == {"added_token_ids", "plan_embeddings.weight"}
    assert "frozen_base_embedding.weight" in state
    overlay_key = (
        "backbone.model.language_model.embed_tokens.plan_embeddings.weight"
    )
    assert overlay_key in state
    assert not any(
        "embed_tokens.frozen_base_embedding" in name for name in state
    )

    restored, _ = make_planner()
    incompatible = restored.load_state_dict(state)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(
        restored.frozen_base_embedding.weight,
        source.frozen_base_embedding.weight,
    )
    torch.testing.assert_close(
        restored.plan_token_adapter.plan_embeddings.weight,
        source.plan_token_adapter.plan_embeddings.weight,
    )


def test_stage1_ownership_is_explicit_disjoint_and_exhaustive() -> None:
    planner, _ = make_planner()
    planner.sem_mlp = nn.Linear(1024, 1024)

    ownership = configure_stage1_trainable_modules(planner)

    assert ownership.planner_modules == (
        planner.query_tower,
        planner.sem_mlp,
        planner.plan_token_adapter,
    )
    assert ownership.qwen_top_layers == tuple(
        planner.backbone.model.language_model.layers[-8:]
    )
    assert ownership.qwen_vision_modules == (planner.backbone.model.visual,)
    category_ids = [
        {
            id(parameter)
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        for modules in (
            ownership.planner_modules,
            ownership.qwen_top_layers,
            ownership.qwen_vision_modules,
        )
    ]
    assert not category_ids[0].intersection(category_ids[1])
    assert not category_ids[0].intersection(category_ids[2])
    assert not category_ids[1].intersection(category_ids[2])
    assert set().union(*category_ids) == {
        id(parameter)
        for parameter in planner.parameters()
        if parameter.requires_grad
    }
    assert not planner.frozen_base_embedding.weight.requires_grad
    assert all(
        not parameter.requires_grad
        for layer in planner.backbone.model.language_model.layers[:4]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in planner.backbone.model.language_model.layers[-8:]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in planner.backbone.model.visual.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in planner.backbone.model.language_model.norm.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in planner.backbone.lm_head.parameters()
    )


def test_stage1_ownership_rejects_ambiguous_explicit_vision_paths() -> None:
    planner, _ = make_planner()
    planner.backbone.vision_model = nn.Linear(8, 8)

    with pytest.raises(ValueError, match="exactly one.*vision"):
        configure_stage1_trainable_modules(planner)


def test_stage1_ownership_rejects_parameter_overlap() -> None:
    planner, _ = make_planner()
    shared = planner.backbone.model.language_model.layers[-1]
    planner.backbone.model.visual = shared

    with pytest.raises(ValueError, match="overlap"):
        configure_stage1_trainable_modules(planner)


def test_stage1_ownership_rejects_shared_planner_module_parameters() -> None:
    planner, _ = make_planner()
    planner.sem_mlp = planner.query_tower

    with pytest.raises(ValueError, match="overlap"):
        configure_stage1_trainable_modules(planner)


def test_stage1_ownership_rejects_shared_parameters_between_top_layers() -> None:
    planner, _ = make_planner()
    layers = planner.backbone.model.language_model.layers
    layers[-1].weight = layers[-2].weight

    with pytest.raises(ValueError, match="overlap"):
        configure_stage1_trainable_modules(planner)
