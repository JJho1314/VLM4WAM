from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from qwen35_planx.config import CAMERA_NAMES
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
        self.calls: list[tuple[tuple[str, ...], torch.Tensor]] = []

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
        assert padding is True
        assert len(text) == len(images)
        self.calls.append((tuple(text), torch.stack(images).clone()))
        rows: list[torch.Tensor] = []
        for prompt in text:
            camera = (
                CAMERA_TOKENS[0]
                if CAMERA_TOKENS[0] in prompt
                else CAMERA_TOKENS[1]
            )
            prefix = [1, 3] if "open the drawer" in prompt else [1]
            rows.append(
                torch.tensor(
                    [
                        *prefix,
                        self.layout.token_id(camera),
                        *self.layout.role_query_ids,
                    ],
                    dtype=torch.long,
                )
            )
        width = max(int(row.numel()) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention_mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : row.numel()] = row
            attention_mask[index, : row.numel()] = 1
        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.stack(images).float(),
            "image_grid_thw": torch.ones(len(rows), 3, dtype=torch.long),
        }


class _BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_calls = 0
        self.incremental_tokens: dict[int, list[int]] = {1: [], 2: []}
        self.total_calls = 0
        self.prefill_attention_mask: torch.Tensor | None = None
        self.rope_deltas_seen: torch.Tensor | None = None

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
        self.total_calls += 1
        if past_key_values is None:
            assert pixel_values is not None
            assert image_grid_thw is not None
            assert position_ids is None
            self.prefill_calls += 1
            rows = pixel_values[:, 0, 0, 0].to(dtype=torch.long)
            signs = torch.where(rows.eq(1), 1.0, -1.0)
            assert torch.equal(
                cache_position,
                torch.arange(input_ids.shape[1], device=input_ids.device),
            )
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            hidden = hidden * signs[:, None, None]
            cache = {"rows": rows.clone(), "length": input_ids.shape[1]}
            rope_deltas = torch.where(rows.eq(1), 11, 23).reshape(-1, 1)
            self.prefill_attention_mask = attention_mask.clone()
            self.rope_deltas_seen = rope_deltas.clone()
        else:
            assert pixel_values is None
            assert image_grid_thw is None
            rows = past_key_values["rows"]
            assert input_ids.shape == (rows.numel(), 1)
            signs = torch.where(rows.eq(1), 1.0, -1.0)
            assert cache_position.tolist() == [past_key_values["length"]]
            assert position_ids is not None
            assert position_ids.shape == (3, rows.numel(), 1)
            rope_deltas = torch.where(rows.eq(1), 11, 23)
            expected_positions = (
                attention_mask.to(dtype=torch.long).sum(dim=-1) - 1 + rope_deltas
            )
            assert torch.equal(
                position_ids,
                expected_positions.reshape(1, -1, 1).expand(3, -1, -1),
            )
            for row, token in zip(rows.tolist(), input_ids[:, 0].tolist()):
                self.incremental_tokens[int(row)].append(int(token))
            hidden = input_ids.to(torch.float32).unsqueeze(-1)
            hidden = hidden * signs[:, None, None]
            cache = {
                "rows": rows.clone(),
                "length": int(past_key_values["length"]) + 1,
            }
        assert attention_mask.shape == (input_ids.shape[0], cache["length"])
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=cache,
            rope_deltas=rope_deltas.to(device=input_ids.device),
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


def test_decoding_config_defaults_to_deterministic_greedy_and_is_immutable() -> None:
    from qwen35_planx.decoding import (
        _select_visual_codes,
        GroundedDecodingConfig,
    )

    config = GroundedDecodingConfig()
    logits = torch.zeros(2, 65_536)
    logits[0, 17] = 3.0
    logits[1, 29] = 4.0
    global_state = torch.random.get_rng_state()

    first = _select_visual_codes(logits, config=config, generator=None)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    torch.manual_seed(91)
    second = _select_visual_codes(logits, config=config, generator=None)

    assert first.tolist() == [17, 29]
    assert torch.equal(first, second)
    with pytest.raises(FrozenInstanceError):
        config.top_k = 2  # type: ignore[misc]
    torch.random.set_rng_state(global_state)


def test_top_k_sampling_is_seeded_reproducible_and_candidate_restricted() -> None:
    from qwen35_planx.decoding import (
        _prepare_sampling_generator,
        _select_visual_codes,
        GroundedDecodingConfig,
    )

    logits = torch.full((32, 65_536), -100.0)
    candidates = torch.tensor([5, 7, 11])
    logits[:, candidates] = 0.0
    config = GroundedDecodingConfig(top_k=3, temperature=0.75, seed=123)
    global_state = torch.random.get_rng_state()

    first = _select_visual_codes(
        logits,
        config=config,
        generator=_prepare_sampling_generator(
            config,
            supplied=None,
            device=logits.device,
        ),
    )
    repeated = _select_visual_codes(
        logits,
        config=config,
        generator=_prepare_sampling_generator(
            config,
            supplied=None,
            device=logits.device,
        ),
    )
    other_config = GroundedDecodingConfig(top_k=3, seed=124)
    different_seed = _select_visual_codes(
        logits,
        config=other_config,
        generator=_prepare_sampling_generator(
            other_config,
            supplied=None,
            device=logits.device,
        ),
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different_seed)
    assert set(first.tolist()) <= set(candidates.tolist())
    assert set(different_seed.tolist()) <= set(candidates.tolist())
    assert torch.equal(torch.random.get_rng_state(), global_state)

    external_config = GroundedDecodingConfig(top_k=3)
    first_generator = torch.Generator(device="cpu").manual_seed(77)
    second_generator = torch.Generator(device="cpu").manual_seed(77)
    generator_state = first_generator.get_state()
    external_first = _select_visual_codes(
        logits,
        config=external_config,
        generator=_prepare_sampling_generator(
            external_config,
            supplied=first_generator,
            device=logits.device,
        ),
    )
    external_second = _select_visual_codes(
        logits,
        config=external_config,
        generator=_prepare_sampling_generator(
            external_config,
            supplied=second_generator,
            device=logits.device,
        ),
    )
    assert torch.equal(external_first, external_second)
    assert not torch.equal(first_generator.get_state(), generator_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)

    tiny_temperature = GroundedDecodingConfig(
        top_k=3,
        temperature=1e-300,
        seed=9,
    )
    selected = _select_visual_codes(
        logits[:1].clone().index_fill(1, candidates, 0.0).index_fill(
            1,
            candidates[:1],
            1.0,
        ),
        config=tiny_temperature,
        generator=_prepare_sampling_generator(
            tiny_temperature,
            supplied=None,
            device=logits.device,
        ),
    )
    assert selected.item() == 5


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"top_k": 0}, ValueError),
        ({"top_k": 65_537}, ValueError),
        ({"top_k": True}, TypeError),
        ({"top_k": 1.5}, TypeError),
        ({"top_k": 2, "temperature": 0.0}, ValueError),
        ({"top_k": 2, "temperature": -1.0}, ValueError),
        ({"top_k": 2, "temperature": float("nan")}, ValueError),
        ({"top_k": 2, "temperature": float("inf")}, ValueError),
        ({"top_k": 2, "temperature": True}, TypeError),
        ({"seed": 1}, ValueError),
        ({"top_k": 2, "seed": True}, TypeError),
    ],
)
def test_decoding_config_rejects_invalid_or_ambiguous_values(
    kwargs,
    error,
) -> None:
    from qwen35_planx.decoding import GroundedDecodingConfig

    with pytest.raises(error):
        GroundedDecodingConfig(**kwargs)


def test_top_k_requires_exactly_one_seed_source() -> None:
    from qwen35_planx.decoding import (
        _prepare_sampling_generator,
        GroundedDecodingConfig,
    )

    unseeded = GroundedDecodingConfig(top_k=2)
    with pytest.raises(ValueError, match="seed|generator"):
        _prepare_sampling_generator(
            unseeded,
            supplied=None,
            device=torch.device("cpu"),
        )

    configured = GroundedDecodingConfig(top_k=2, seed=7)
    supplied = torch.Generator(device="cpu").manual_seed(8)
    with pytest.raises(ValueError, match="exactly one"):
        _prepare_sampling_generator(
            configured,
            supplied=supplied,
            device=torch.device("cpu"),
        )


def test_batched_decode_call_count_is_row_independent_and_cache_isolated(
    monkeypatch,
) -> None:
    import qwen35_planx.decoding as decoding

    monkeypatch.setattr(decoding, "_NUM_KEYFRAMES", 1)
    monkeypatch.setattr(decoding, "_TOKENS_PER_FRAME", 3)
    layout = _layout()

    def decode(images: torch.Tensor):
        planner = _Planner(layout)
        values = decoding._decode_camera_batch(
            planner,
            backbone=planner.base_model,
            visual_weight=planner.visual_embedding_weight,
            images=images,
            instructions=tuple("open the drawer" for _ in images),
            cameras=tuple(
                CAMERA_NAMES[index % 2] for index in range(len(images))
            ),
            layout=layout,
            processor=planner.processor,
            config=decoding.GroundedDecodingConfig(),
            generator=None,
        )
        return planner, values

    one_planner, _ = decode(torch.ones(1, 3, 2, 2))
    two_planner, original = decode(
        torch.stack((torch.ones(3, 2, 2), torch.full((3, 2, 2), 2.0)))
    )
    changed_planner, changed = decode(torch.ones(2, 3, 2, 2))

    expected_calls = 1 + 1 + (1 + 3 + 1) + 1
    assert one_planner.base_model.total_calls == expected_calls
    assert two_planner.base_model.total_calls == expected_calls
    assert changed_planner.base_model.total_calls == expected_calls
    assert torch.equal(original[0][0], changed[0][0])
    assert torch.equal(original[1][0], changed[1][0])
    assert not torch.equal(original[0][1], changed[0][1])
    assert not torch.equal(original[1][1], changed[1][1])


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
    from qwen35_planx.decoding import (
        generate_grounded_plan,
        unflatten_generated_plan,
    )

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
    assert planner.base_model.prefill_calls == 1
    assert planner.base_model.total_calls == (
        1 + 1 + 4 * (1 + 729 + 1) + 1
    )
    assert planner.base_model.prefill_attention_mask is not None
    assert planner.base_model.prefill_attention_mask.tolist() == [
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    assert planner.base_model.rope_deltas_seen is not None
    assert planner.base_model.rope_deltas_seen.tolist() == [[11], [23]]

    sample_major = unflatten_generated_plan(plan, batch_size=1)
    assert torch.equal(sample_major.codes[0, 0], plan.codes[0])
    assert torch.equal(sample_major.codes[0, 1], plan.codes[1])

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

    assert len(planner.processor.calls) == 1
    (main_prompt, wrist_prompt), batched_images = planner.processor.calls[0]
    assert torch.equal(batched_images, images)
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
            camera_batch=1,
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
