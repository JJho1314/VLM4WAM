from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from qwen35_planx.vocabulary import (
    CAMERA_TOKENS,
    FRAME_END_TOKENS,
    FRAME_START_TOKENS,
    PLAN_END_TOKEN,
    PLAN_START_TOKEN,
    ROLE_QUERY_TOKENS,
    STRUCTURE_TOKENS,
    VisualVocabularyLayout,
)


class _Tokenizer:
    def __init__(self, token_ids: dict[str, int]) -> None:
        self.token_ids = token_ids
        self.eos_token_id = 2
        self.pad_token_id = 0


def _layout(*, eos_token_id: int = 2) -> VisualVocabularyLayout:
    structure = tuple(
        (token, index) for index, token in enumerate(STRUCTURE_TOKENS, start=4)
    )
    token_ids = dict(structure)
    tokenizer = _Tokenizer(token_ids)
    tokenizer.eos_token_id = eos_token_id
    visual_start = 4 + len(STRUCTURE_TOKENS)
    return VisualVocabularyLayout(
        original_vocab_size=4,
        visual_start_id=visual_start,
        visual_end_id=visual_start + 65_536,
        structure_token_ids=structure,
        tokenizer_hash="test-tokenizer",
        base_embedding_hash="base",
        expanded_embedding_hash="expanded",
        _tokenizer=tokenizer,
    )


class _Processor:
    def __init__(self, layout: VisualVocabularyLayout) -> None:
        self.layout = layout
        self.tokenizer = layout._tokenizer
        self.calls: list[tuple[str, torch.Tensor]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"][1]["text"]

    def __call__(
        self,
        *,
        text: list[str],
        images: list[torch.Tensor],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding is False
        prompt = text[0]
        image = images[0]
        self.calls.append((prompt, image.clone()))
        camera = (
            CAMERA_TOKENS[0] if CAMERA_TOKENS[0] in prompt else CAMERA_TOKENS[1]
        )
        ids = torch.tensor(
            [
                1,
                self.layout.token_id(camera),
                *self.layout.role_query_ids,
            ],
            dtype=torch.long,
        ).unsqueeze(0)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": image.float().unsqueeze(0),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_calls = 0
        self.incremental_tokens: dict[int, list[int]] = {1: [], 2: []}
        self.kwargs_seen: list[dict[str, object]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ):
        assert use_cache is True
        assert output_hidden_states is False
        assert return_dict is True
        self.kwargs_seen.append(
            {
                "past_key_values": past_key_values,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        )
        if past_key_values is None:
            assert pixel_values is not None
            assert image_grid_thw is not None
            assert position_ids is None
            self.prefill_calls += 1
            row = int(pixel_values[0, 0, 0, 0].item())
            sign = 1.0 if row == 1 else -1.0
            assert torch.equal(
                cache_position,
                torch.arange(input_ids.shape[1], device=input_ids.device),
            )
            hidden = input_ids.to(torch.float32).unsqueeze(-1) * sign
            cache = {"row": row, "length": input_ids.shape[1]}
        else:
            assert pixel_values is None
            assert image_grid_thw is None
            assert input_ids.shape == (1, 1)
            row = int(past_key_values["row"])
            sign = 1.0 if row == 1 else -1.0
            token = int(input_ids.item())
            assert cache_position.tolist() == [past_key_values["length"]]
            assert position_ids is not None
            assert position_ids.shape == (3, 1, 1)
            assert torch.equal(
                position_ids,
                torch.full_like(
                    position_ids,
                    past_key_values["length"] + 11,
                ),
            )
            self.incremental_tokens[row].append(token)
            hidden = input_ids.to(torch.float32).unsqueeze(-1) * sign
            cache = {
                "row": row,
                "length": int(past_key_values["length"]) + 1,
            }
        assert attention_mask.shape == (1, cache["length"])
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=cache,
            rope_deltas=torch.tensor([[11]], device=input_ids.device),
        )


class _ConditionalWrapper(nn.Module):
    def __init__(self, model: _BaseModel) -> None:
        super().__init__()
        self.model = model
        self.public_forward_calls = 0
        self.lm_head_calls = 0

    def forward(self, *_args, **_kwargs):
        self.public_forward_calls += 1
        self.lm_head_calls += 1
        raise AssertionError("public conditional-generation forward was called")


class _Planner(nn.Module):
    def __init__(self, layout: VisualVocabularyLayout) -> None:
        super().__init__()
        self._enforce_released_geometry = False
        self.layout = layout
        self.processor = _Processor(layout)
        self.base_model = _BaseModel()
        self.backbone = _ConditionalWrapper(self.base_model)
        rows = torch.zeros(65_536, 1)
        rows[5, 0] = 2.0
        rows[7, 0] = -2.0
        self.visual_embedding_weight = nn.Parameter(rows)
        codebook = torch.arange(65_536, dtype=torch.float32).unsqueeze(-1)
        self.register_buffer("codebook", codebook)
        self.hidden_dim = 1
        self.text_dim = 3
        self.semantic_projection = nn.Linear(1, 3, bias=False)
        self.phrase_projection = nn.Linear(1, 3, bias=False)
        self.grounding_query = nn.Linear(1, 3, bias=False)
        self.fusion_gate = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        self.register_buffer(
            "target_times",
            torch.tensor([0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0]),
        )
        with torch.no_grad():
            self.semantic_projection.weight.copy_(torch.tensor([[1.0], [2.0], [3.0]]))
            self.phrase_projection.weight.copy_(torch.tensor([[2.0], [3.0], [4.0]]))
            self.grounding_query.weight.copy_(torch.tensor([[3.0], [4.0], [5.0]]))

    def _language_backbone(self) -> nn.Module:
        return self.backbone.model

    @staticmethod
    def _last_hidden(output) -> torch.Tensor:
        return output.last_hidden_state


@pytest.fixture()
def decoding_components():
    layout = _layout()
    return _Planner(layout), layout


def test_generation_emits_exact_structure_and_records_final_post_state(
    decoding_components,
    monkeypatch,
) -> None:
    real_import = builtins.__import__

    def reject_siglip(name, *args, **kwargs):
        if "siglip" in name.lower():
            raise AssertionError("generation imported SigLIP2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_siglip)
    from qwen35_planx.decoding import generate_grounded_plan

    planner, layout = decoding_components
    images = torch.stack(
        (
            torch.ones(3, 2, 2),
            torch.full((3, 2, 2), 2.0),
        )
    )
    plan = generate_grounded_plan(
        planner,
        current_images=images,
        instructions=(
            "pick up the bowl and place it on the plate",
            "open the drawer",
        ),
        camera_names=("main", "wrist"),
        layout=layout,
    )

    assert plan.codes.shape == (2, 4, 729)
    assert plan.code_embeddings.shape == (2, 4, 729, 1)
    assert plan.post_hidden.shape == (2, 4, 729, 1)
    assert plan.predicted_phrase_embeddings.shape == (2, 3, 3)
    assert plan.semantic_features.shape == (2, 4, 729, 3)
    assert plan.relevance.shape == (2, 4, 3, 729)
    assert plan.fusion_gate.shape == (2, 4, 729, 1)
    assert torch.equal(plan.codes[0], torch.full((4, 729), 5))
    assert torch.equal(plan.codes[1], torch.full((4, 729), 7))
    field_states = torch.tensor(
        [
            list(layout.role_query_ids),
            [-value for value in layout.role_query_ids],
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)
    expected_phrases = F.normalize(
        planner.phrase_projection(field_states),
        dim=-1,
    )
    torch.testing.assert_close(
        plan.predicted_phrase_embeddings,
        expected_phrases,
    )
    torch.testing.assert_close(
        plan.post_hidden[0, -1, -1, 0],
        torch.tensor(float(layout.code_token_id(5))),
    )
    torch.testing.assert_close(
        plan.post_hidden[1, -1, -1, 0],
        torch.tensor(float(-layout.code_token_id(7))),
    )
    assert planner.backbone.public_forward_calls == 0
    assert planner.backbone.lm_head_calls == 0
    assert planner.base_model.prefill_calls == 2

    for row, code in ((1, 5), (2, 7)):
        tokens = planner.base_model.incremental_tokens[row]
        assert tokens[0] == layout.token_id(PLAN_START_TOKEN)
        cursor = 1
        for frame in range(4):
            assert tokens[cursor] == layout.token_id(FRAME_START_TOKENS[frame])
            cursor += 1
            assert tokens[cursor : cursor + 729] == [
                layout.code_token_id(code)
            ] * 729
            cursor += 729
            assert tokens[cursor] == layout.token_id(FRAME_END_TOKENS[frame])
            cursor += 1
        assert tokens[cursor] == layout.token_id(PLAN_END_TOKEN)
        assert cursor + 1 == len(tokens)

    main_prompt, wrist_prompt = (item[0] for item in planner.processor.calls)
    assert "<CAMERA_MAIN>" in main_prompt
    assert "<CAMERA_WRIST>" in wrist_prompt
    for prompt in (main_prompt, wrist_prompt):
        assert "<ACT>" in prompt and "</ACT>" in prompt
        assert "<SRC>" in prompt and "</SRC>" in prompt
        assert "<TGT>" in prompt and "</TGT>" in prompt
        assert all(query in prompt for query in ROLE_QUERY_TOKENS)


def test_generation_fails_closed_for_malformed_layout_and_eos_overlap(
    decoding_components,
) -> None:
    from qwen35_planx.decoding import generate_grounded_plan

    planner, layout = decoding_components
    images = torch.ones(1, 3, 2, 2)
    kwargs = {
        "current_images": images,
        "instructions": ("open the drawer",),
        "camera_names": ("main",),
    }

    malformed = VisualVocabularyLayout(
        original_vocab_size=layout.original_vocab_size,
        visual_start_id=layout.visual_start_id,
        visual_end_id=layout.visual_end_id,
        structure_token_ids=tuple(
            (token, layout.structure_token_ids[0][1])
            for token, _ in layout.structure_token_ids
        ),
        tokenizer_hash=layout.tokenizer_hash,
        base_embedding_hash=layout.base_embedding_hash,
        expanded_embedding_hash=layout.expanded_embedding_hash,
        _tokenizer=layout._tokenizer,
    )
    with pytest.raises(ValueError, match="structural"):
        generate_grounded_plan(planner, layout=malformed, **kwargs)

    eos_layout = _layout(eos_token_id=layout.code_token_id(5))
    planner.layout = eos_layout
    planner.processor.layout = eos_layout
    planner.processor.tokenizer = eos_layout._tokenizer
    with pytest.raises(ValueError, match="EOS"):
        generate_grounded_plan(planner, layout=eos_layout, **kwargs)


def test_generation_rejects_nonfinite_visual_logits(
    decoding_components,
) -> None:
    from qwen35_planx.decoding import generate_grounded_plan

    planner, layout = decoding_components
    with torch.no_grad():
        planner.visual_embedding_weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite visual logits"):
        generate_grounded_plan(
            planner,
            current_images=torch.ones(1, 3, 2, 2),
            instructions=("open the drawer",),
            camera_names=("main",),
            layout=layout,
        )


def test_visual_rows_follow_current_backbone_embedding_after_module_to() -> None:
    from qwen35_planx.decoding import _current_visual_embedding_rows

    layout = _layout()

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(layout.visual_end_id, 2)

        def get_input_embeddings(self):
            return self.embedding

    class Wrapper(nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

    class Planner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Wrapper(Base())
            self.visual_embedding_weight = self.backbone.model.embedding.weight[
                layout.visual_start_id : layout.visual_end_id
            ]

        def _language_backbone(self):
            return self.backbone.model

    planner = Planner()
    stale_rows = planner.visual_embedding_weight
    planner.to(dtype=torch.float64)
    assert stale_rows.dtype == torch.float32
    assert planner.backbone.model.embedding.weight.dtype == torch.float64

    current_rows = _current_visual_embedding_rows(planner, layout)
    expected = planner.backbone.model.embedding.weight[
        layout.visual_start_id : layout.visual_end_id
    ]
    assert current_rows.dtype == torch.float64
    assert current_rows.data_ptr() == expected.data_ptr()


def test_missing_multimodal_rope_state_fails_closed() -> None:
    from qwen35_planx.decoding import _rope_deltas

    with pytest.raises(RuntimeError, match="rope_deltas"):
        _rope_deltas(
            SimpleNamespace(),
            backbone=nn.Linear(1, 1),
            device=torch.device("cpu"),
        )


def test_generated_plan_validation_and_unflatten_fail_closed() -> None:
    from qwen35_planx.decoding import (
        GeneratedGroundedPlan,
        unflatten_generated_plan,
    )

    flat = GeneratedGroundedPlan._from_test_components(
        codes=torch.zeros(4, 4, 729, dtype=torch.long),
        code_embeddings=torch.zeros(4, 4, 729, 2),
        post_hidden=torch.zeros(4, 4, 729, 3),
        predicted_phrase_embeddings=F.normalize(torch.ones(4, 3, 5), dim=-1),
        semantic_features=F.normalize(torch.ones(4, 4, 729, 5), dim=-1),
        relevance=torch.full((4, 4, 3, 729), 1 / 729),
        fusion_gate=torch.full((4, 4, 729, 1), 0.5),
        times=torch.tensor([0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0]),
    )
    output = unflatten_generated_plan(flat, batch_size=2)
    assert output.codes.shape == (2, 2, 4, 729)
    assert output.semantic_features.shape == (2, 2, 4, 729, 5)
    assert output.times is flat.times

    with pytest.raises(ValueError, match=r"batch_size\*2"):
        unflatten_generated_plan(flat, batch_size=1)
    with pytest.raises(ValueError, match="codes"):
        GeneratedGroundedPlan(
            **{
                **flat.__dict__,
                "codes": torch.full((4, 4, 729), 65_536, dtype=torch.long),
            }
        )
    with pytest.raises(ValueError, match="released"):
        GeneratedGroundedPlan(
            codes=torch.zeros(1, 4, 729, dtype=torch.long),
            code_embeddings=torch.zeros(1, 4, 729, 2),
            post_hidden=torch.zeros(1, 4, 729, 3),
            predicted_phrase_embeddings=torch.ones(1, 3, 5),
            semantic_features=torch.ones(1, 4, 729, 5),
            relevance=torch.full((1, 4, 3, 729), 1 / 729),
            fusion_gate=torch.full((1, 4, 729, 1), 0.5),
            times=torch.tensor([0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0]),
        )
