from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch import nn


class PlannerTokenizer:
    def __init__(self) -> None:
        self.tokens = ["<unk>", "\n"]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        for token in payload["additional_special_tokens"]:
            self.token_to_id[token] = len(self.tokens)
            self.tokens.append(token)
        return len(payload["additional_special_tokens"])

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_id[token]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        specials = sorted(self.token_to_id, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(token) for token in specials))
        result: list[int] = []
        cursor = 0
        for match in pattern.finditer(text):
            result.extend([0] * len(text[cursor : match.start()].encode()))
            result.append(self.token_to_id[match.group()])
            cursor = match.end()
        result.extend([0] * len(text[cursor:].encode()))
        return result


class PlannerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, 4)
        self.output = nn.Linear(4, 2, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output

    def resize_token_embeddings(self, size: int, *, mean_resizing: bool = False):
        del mean_resizing
        old_input = self.embedding.weight.detach()
        old_output = self.output.weight.detach()
        self.embedding = nn.Embedding(size, 4)
        self.output = nn.Linear(4, size, bias=False)
        with torch.no_grad():
            self.embedding.weight[: len(old_input)].copy_(old_input)
            self.output.weight[: len(old_output)].copy_(old_output)
        return self.embedding


class FakeProcessor:
    def __init__(self, tokenizer: PlannerTokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls: list[tuple[str, torch.Tensor]] = []

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
        assert len(text) == len(images) == 1
        self.calls.append((text[0], images[0].clone()))
        identifiers = torch.tensor(
            [self.tokenizer.encode(text[0], add_special_tokens=False)],
            dtype=torch.long,
        )
        return {
            "input_ids": identifiers,
            "attention_mask": torch.ones_like(identifiers),
            "pixel_values": images[0].float().unsqueeze(0),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class ChatTemplateProcessor(FakeProcessor):
    def __init__(self, tokenizer: PlannerTokenizer) -> None:
        super().__init__(tokenizer)
        self.messages = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages.append(messages)
        text = messages[0]["content"][1]["text"]
        return f"<image>\n{text}\nassistant:"


class FakeCache:
    def __init__(self, record, sample) -> None:
        self.records = (record,)
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.sample


@pytest.fixture(scope="module")
def planner_components():
    from qwen35_planx.vocabulary import install_visual_vocabulary

    tokenizer = PlannerTokenizer()
    layout = install_visual_vocabulary(tokenizer, PlannerModel())
    return tokenizer, layout


def _manifest_fixture(tmp_path: Path):
    from qwen35_planx.hindsight_data import build_fixed_windows

    root = tmp_path / "hdf5"
    root.mkdir()
    shard = root / "shard.h5"
    key = "libero_goal:000000"
    caption = "pick up the bowl and place it on the plate"
    length = 40
    with h5py.File(shard, "w") as handle:
        group = handle.create_group(f"episodes/{key}")
        strings = h5py.string_dtype(encoding="utf-8")
        group.create_dataset("caption", data=caption, dtype=strings)
        group.create_dataset("domain", data="libero_goal", dtype=strings)
        group.create_dataset("episode_index", data=0, dtype=np.int64)
        group.create_dataset("length", data=length, dtype=np.int64)
        main = np.empty((length, 256, 256, 3), dtype=np.uint8)
        wrist = np.empty_like(main)
        for index in range(length):
            main[index].fill(index)
            wrist[index].fill(100 + index)
        group.create_dataset("rgb_main", data=main)
        group.create_dataset("rgb_wrist", data=wrist)
        group.create_dataset("action", data=np.zeros((length, 7), np.float32))
        group.create_dataset("state", data=np.zeros((length, 8), np.float32))
    payload = {
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
                "key": key,
                "shard": shard.name,
                "group": f"episodes/{key}",
                "caption": caption,
                "domain": "libero_goal",
                "episode_index": 0,
                "length": length,
            }
        ],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    record = build_fixed_windows(manifest, split_seed=0)[0]
    return manifest, record


def _sample_targets(*, batch: int = 1):
    from qwen35_planx.planner_dataset import CachedPlannerTargets

    codes = torch.arange(batch * 2 * 4 * 729).reshape(batch, 2, 4, 729)
    codes = codes.remainder(65_536).long()
    relevance = torch.full((batch, 2, 4, 3, 729), 1 / 729)
    confidence = torch.ones((batch, 2, 4, 3))
    flow = torch.zeros((batch, 2, 3, 729, 3))
    phrases = torch.arange(batch * 3 * 1152).reshape(batch, 3, 1152).float()
    return CachedPlannerTargets(
        codes=codes,
        relevance=relevance,
        relevance_confidence=confidence,
        flow=flow,
        phrase_embeddings=phrases,
    )


def _phrase_tables():
    vocabulary = {
        "source": ["the bowl", "the mug", "cross suite object"],
        "target": ["on the plate", "in the drawer", "cross suite target"],
        "action": ["pick up and place", "put", "cross suite action"],
    }
    embeddings = {
        role: torch.stack(
            [torch.full((1152,), float(index + 1)) for index in range(3)]
        ).to(torch.float16)
        for role in vocabulary
    }
    suite_vocabulary = {
        "libero_goal": {
            "source": ("the bowl", "the mug"),
            "target": ("on the plate", "in the drawer"),
            "action": ("pick up and place", "put"),
        },
        "other": {
            "source": ("cross suite object",),
            "target": ("cross suite target",),
            "action": ("cross suite action",),
        },
    }
    return vocabulary, embeddings, suite_vocabulary


def test_dataset_reads_current_rgb_from_the_exact_cache_window(tmp_path: Path) -> None:
    from qwen35_planx.planner_dataset import HindsightPlannerDataset

    manifest, record = _manifest_fixture(tmp_path)
    targets = _sample_targets()
    cache_sample = SimpleNamespace(
        record=record,
        codes=targets.codes[0],
        relevance=targets.relevance[0],
        confidence=targets.relevance_confidence[0],
        flow=targets.flow[0],
        phrase_embeddings=targets.phrase_embeddings[0],
    )
    cache = FakeCache(record, cache_sample)

    dataset = HindsightPlannerDataset(cache=cache, hdf5_manifest=manifest)
    sample = dataset[0]
    assert sample["current_images"].shape == (2, 3, 256, 256)
    assert sample["current_images"].dtype == torch.uint8
    assert torch.all(sample["current_images"][0] == record.current_index)
    assert torch.all(sample["current_images"][1] == 100 + record.current_index)
    assert sample["instruction"] == record.caption
    assert sample["suite"] == "libero_goal"
    assert torch.equal(sample["codes"], targets.codes[0])


def test_collator_flattens_cameras_and_builds_exact_target_shapes(
    planner_components,
) -> None:
    from qwen35_planx.planner_dataset import GroundedPlannerCollator

    tokenizer, layout = planner_components
    processor = FakeProcessor(tokenizer)
    vocabulary, embeddings, suite_vocabulary = _phrase_tables()
    instruction = "pick up the bowl and place it on the plate"
    collator = GroundedPlannerCollator(
        processor,
        layout,
        phrase_vocabulary=vocabulary,
        phrase_embeddings=embeddings,
        suite_vocabularies=suite_vocabulary,
        instruction_suites={instruction: "libero_goal"},
        max_negatives=1,
    )
    images = torch.zeros((1, 2, 3, 8, 8), dtype=torch.uint8)
    targets = _sample_targets()
    batch = collator.build_teacher_forced(images, [instruction], targets)

    assert batch.size == 2
    assert batch.qwen_inputs["input_ids"].shape[0] == 2
    assert batch.qwen_inputs["attention_mask"].dtype == torch.long
    assert batch.code_targets.shape == (2, 2916)
    assert batch.pre_positions.shape == batch.post_positions.shape == (2, 2916)
    assert batch.field_positions.shape == batch.field_mask.shape == (2, 3)
    assert batch.relevance_targets.shape == (2, 4, 3, 729)
    assert batch.relevance_confidence.shape == (2, 4, 3)
    assert batch.flow_targets.shape == (2, 3, 729, 3)
    assert batch.phrase_embeddings.shape == (2, 3, 1152)
    assert batch.counterfactual_embeddings.shape == (2, 3, 1, 1152)
    assert batch.counterfactual_mask.all()
    assert torch.all(batch.counterfactual_embeddings[:, 0, 0] == 2)
    assert torch.all(batch.counterfactual_embeddings[:, 1, 0] == 2)
    assert torch.all(batch.counterfactual_embeddings[:, 2, 0] == 2)
    assert len(processor.calls) == 2
    assert "<CAMERA_MAIN>" in processor.calls[0][0]
    assert "<CAMERA_WRIST>" in processor.calls[1][0]
    assert torch.equal(batch.code_targets[0], targets.codes[0, 0].flatten())
    assert torch.equal(batch.code_targets[1], targets.codes[0, 1].flatten())
    assert batch.qwen_inputs["input_ids"].data_ptr() != 0


def test_dataset_call_and_teacher_forced_are_byte_identical(
    planner_components,
) -> None:
    from qwen35_planx.planner_dataset import GroundedPlannerCollator

    tokenizer, layout = planner_components
    vocabulary, embeddings, suite_vocabulary = _phrase_tables()
    instruction = "pick up the bowl and place it on the plate"
    collator = GroundedPlannerCollator(
        FakeProcessor(tokenizer),
        layout,
        phrase_vocabulary=vocabulary,
        phrase_embeddings=embeddings,
        suite_vocabularies=suite_vocabulary,
        instruction_suites={instruction: "libero_goal"},
    )
    images = torch.arange(1 * 2 * 3 * 8 * 8, dtype=torch.uint8).reshape(
        1, 2, 3, 8, 8
    )
    targets = _sample_targets()
    sample = {
        "current_images": images[0],
        "instruction": instruction,
        "suite": "libero_goal",
        "codes": targets.codes[0],
        "relevance": targets.relevance[0],
        "relevance_confidence": targets.relevance_confidence[0],
        "flow": targets.flow[0],
        "phrase_embeddings": targets.phrase_embeddings[0],
    }
    from_samples = collator([sample])
    direct = collator.build_teacher_forced(images, [instruction], targets)
    for name in from_samples.qwen_inputs:
        assert torch.equal(
            from_samples.qwen_inputs[name], direct.qwen_inputs[name]
        ), name
    for item in fields(from_samples):
        if item.name == "qwen_inputs":
            continue
        assert torch.equal(
            getattr(from_samples, item.name), getattr(direct, item.name)
        ), item.name


def test_missing_role_and_empty_suite_vocabulary_fail_closed(
    planner_components,
) -> None:
    from qwen35_planx.planner_dataset import GroundedPlannerCollator

    tokenizer, layout = planner_components
    vocabulary, embeddings, _ = _phrase_tables()
    instruction = "open the drawer"
    collator = GroundedPlannerCollator(
        FakeProcessor(tokenizer),
        layout,
        phrase_vocabulary=vocabulary,
        phrase_embeddings=embeddings,
        suite_vocabularies={
            "libero_goal": {
                "source": (),
                "target": (),
                "action": (),
            }
        },
        instruction_suites={instruction: "libero_goal"},
    )
    batch = collator.build_teacher_forced(
        torch.zeros((1, 2, 3, 8, 8), dtype=torch.uint8),
        [instruction],
        _sample_targets(),
    )
    assert batch.field_mask.tolist() == [
        [True, False, True],
        [True, False, True],
    ]
    assert not batch.counterfactual_mask.any()
    assert torch.count_nonzero(batch.counterfactual_embeddings) == 0


def test_real_processor_path_uses_multimodal_chat_template(
    planner_components,
) -> None:
    from qwen35_planx.planner_dataset import GroundedPlannerCollator

    tokenizer, layout = planner_components
    processor = ChatTemplateProcessor(tokenizer)
    vocabulary, embeddings, suite_vocabulary = _phrase_tables()
    instruction = "pick up the bowl and place it on the plate"
    collator = GroundedPlannerCollator(
        processor,
        layout,
        phrase_vocabulary=vocabulary,
        phrase_embeddings=embeddings,
        suite_vocabularies=suite_vocabulary,
        instruction_suites={instruction: "libero_goal"},
    )
    collator.build_teacher_forced(
        torch.zeros((1, 2, 3, 8, 8), dtype=torch.uint8),
        [instruction],
        _sample_targets(),
    )
    assert len(processor.messages) == 2
    assert processor.messages[0][0]["content"][0]["type"] == "image"
    assert processor.messages[0][0]["content"][1]["type"] == "text"
    assert processor.calls[0][0].startswith("<image>\n<CAMERA_MAIN>")


def test_collator_rejects_missing_suite_provenance(planner_components) -> None:
    from qwen35_planx.planner_dataset import GroundedPlannerCollator

    tokenizer, layout = planner_components
    vocabulary, embeddings, _ = _phrase_tables()
    with pytest.raises(ValueError, match="suite provenance"):
        GroundedPlannerCollator(
            FakeProcessor(tokenizer),
            layout,
            phrase_vocabulary=vocabulary,
            phrase_embeddings=embeddings,
        )


def test_unknown_instruction_suite_has_no_counterfactuals(
    planner_components,
) -> None:
    from qwen35_planx.planner_dataset import GroundedPlannerCollator

    tokenizer, layout = planner_components
    vocabulary, embeddings, suite_vocabulary = _phrase_tables()
    known = "pick up the bowl and place it on the plate"
    unknown = "pick up the mug and place it in the drawer"
    collator = GroundedPlannerCollator(
        FakeProcessor(tokenizer),
        layout,
        phrase_vocabulary=vocabulary,
        phrase_embeddings=embeddings,
        suite_vocabularies=suite_vocabulary,
        instruction_suites={known: "libero_goal"},
    )
    batch = collator.build_teacher_forced(
        torch.zeros((1, 2, 3, 8, 8), dtype=torch.uint8),
        [unknown],
        _sample_targets(),
    )
    assert not batch.counterfactual_mask.any()
    assert torch.count_nonzero(batch.counterfactual_embeddings) == 0
