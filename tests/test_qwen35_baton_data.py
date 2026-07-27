from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from qwen35_baton.config import BatonGeometry


class _Tokenizer:
    def __init__(self) -> None:
        from qwen35_baton.sequence import ADDED_TOKENS

        self.pad_token_id = 0
        self._ids = {token: index + 1 for index, token in enumerate(ADDED_TOKENS)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def encode(self, text: str) -> list[int]:
        pattern = re.compile("|".join(re.escape(token) for token in self._ids))
        identifiers: list[int] = []
        cursor = 0
        for match in pattern.finditer(text):
            identifiers.extend([0] * len(text[cursor : match.start()]))
            identifiers.append(self._ids[match.group()])
            cursor = match.end()
        identifiers.extend([0] * len(text[cursor:]))
        return identifiers


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
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
        ids = torch.tensor([self.tokenizer.encode(text[0])], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": images[0].float().unsqueeze(0),
            "image_grid_thw": torch.tensor([[1, 16, 16]], dtype=torch.long),
        }


class _BaseDataset:
    n_previous = 4

    def __init__(self) -> None:
        self.records = (
            SimpleNamespace(
                key="libero_object:000000",
                caption="put the red mug on the plate",
                domain="libero_object",
            ),
            SimpleNamespace(
                key="libero_object:000001",
                caption="put the blue bowl in the drawer",
                domain="libero_object",
            ),
        )
        video = torch.empty((3, 2, 13, 256, 256), dtype=torch.uint8)
        for camera in range(2):
            for frame in range(13):
                video[:, camera, frame].fill_(camera * 20 + frame)
        self.video = video

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        normalized = self.video.to(torch.float32).div(255).sub(0.5).div(0.5)
        return {"video": normalized, "caption": record.caption}


@pytest.fixture
def dataset():
    from qwen35_baton.data import BatonLiberoDataset

    return BatonLiberoDataset(_BaseDataset(), seed=7)


def test_dataset_selects_current_and_four_future_frames(dataset) -> None:
    sample = dataset[0]
    expected_video = dataset.base_dataset.video.permute(1, 2, 0, 3, 4)
    assert sample["current_images"].shape == (2, 3, 256, 256)
    assert sample["future_images"].shape == (2, 4, 3, 256, 256)
    torch.testing.assert_close(sample["current_images"], expected_video[:, 3])
    torch.testing.assert_close(
        sample["future_images"], expected_video[:, [4, 7, 9, 12]]
    )
    assert sample["suite"] == "libero_object"
    assert sample["instruction"] != sample["negative_instruction"]


def test_dataset_rejects_suites_without_an_alternative_instruction() -> None:
    from qwen35_baton.data import BatonLiberoDataset

    base = _BaseDataset()
    base.records = base.records[:1]
    with pytest.raises(ValueError, match="at least two distinct instructions"):
        BatonLiberoDataset(base)


def test_plan_template_and_positions_fail_closed() -> None:
    from qwen35_baton.sequence import PLAN_PAD, build_plan_text, find_plan_positions

    text = build_plan_text("move the cup")
    assert text.count(PLAN_PAD) == 1024
    assert text.count("<PLAN_START>") == text.count("<PLAN_END>") == 1
    for index in range(4):
        assert text.count(f"<FRAME_{index}>") == 1
    ids = torch.tensor([[5] * 1024, [5] * 1023 + [0]])
    with pytest.raises(ValueError, match="exactly 1024"):
        find_plan_positions(ids, 5)


def test_collator_orders_positive_then_negative_sample_major_rows(dataset) -> None:
    from qwen35_baton.data import BatonPlannerCollator

    processor = _Processor()
    collator = BatonPlannerCollator(processor)
    sample0, sample1 = dataset[0], dataset[1]
    batch = collator([sample0, sample1])
    plan_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<PLAN_PAD>")

    assert batch.row_labels == (
        ("positive", 0, "main"),
        ("positive", 0, "wrist"),
        ("positive", 1, "main"),
        ("positive", 1, "wrist"),
        ("negative", 0, "main"),
        ("negative", 0, "wrist"),
        ("negative", 1, "main"),
        ("negative", 1, "wrist"),
    )
    assert batch.plan_positions.shape == (8, 1024)
    assert torch.all(
        batch.qwen_inputs["input_ids"].gather(1, batch.plan_positions)
        == plan_pad_token_id
    )
    assert batch.current_images.shape == (2, 2, 3, 256, 256)
    assert batch.future_images.shape == (2, 2, 4, 3, 256, 256)
    assert batch.positive_rows == slice(0, 4)
    assert batch.negative_rows == slice(4, 8)
    assert all(
        positive != negative
        for positive, negative in zip(batch.instructions, batch.negative_instructions)
    )

    modified = dict(sample0)
    modified["current_images"] = sample0["current_images"].clone()
    modified["current_images"][1].add_(1)
    changed = collator([modified, sample1])
    torch.testing.assert_close(
        batch.qwen_inputs["pixel_values"][0], changed.qwen_inputs["pixel_values"][0]
    )
    assert not torch.equal(
        batch.qwen_inputs["pixel_values"][1], changed.qwen_inputs["pixel_values"][1]
    )
    torch.testing.assert_close(
        batch.qwen_inputs["input_ids"][0], changed.qwen_inputs["input_ids"][0]
    )


def test_batch_contract_exposes_fixed_geometry(dataset) -> None:
    from qwen35_baton.data import BatonPlannerCollator

    batch = BatonPlannerCollator(_Processor())([dataset[0]])
    geometry = BatonGeometry()
    assert batch.current_images.shape[1:] == (
        2,
        3,
        geometry.image_size,
        geometry.image_size,
    )
    assert batch.future_images.shape[1:3] == (2, len(geometry.future_indices))
