from __future__ import annotations

import re
import random
from types import SimpleNamespace

import numpy as np
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
    assert sample["instruction"] == "put the red mug on the plate"
    assert "negative_instruction" not in sample


def test_dataset_accepts_a_suite_with_one_instruction() -> None:
    from qwen35_baton.data import BatonLiberoDataset

    base = _BaseDataset()
    base.records = base.records[:1]
    dataset = BatonLiberoDataset(base)
    assert len(dataset) == 1
    assert dataset[0]["instruction"] == base.records[0].caption


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


def test_collator_builds_only_positive_sample_major_camera_rows(dataset) -> None:
    from qwen35_baton.data import BatonPlannerCollator

    processor = _Processor()
    collator = BatonPlannerCollator(processor)
    sample0, sample1 = dataset[0], dataset[1]
    batch = collator([sample0, sample1])
    plan_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<PLAN_PAD>")

    assert batch.row_labels == (
        (0, "main"),
        (0, "wrist"),
        (1, "main"),
        (1, "wrist"),
    )
    assert batch.qwen_inputs["input_ids"].shape[0] == 4
    assert batch.plan_positions.shape == (4, 1024)
    assert torch.all(
        batch.qwen_inputs["input_ids"].gather(1, batch.plan_positions)
        == plan_pad_token_id
    )
    assert batch.current_images.shape == (2, 2, 3, 256, 256)
    assert batch.future_images.shape == (2, 2, 4, 3, 256, 256)
    assert not hasattr(batch, "negative_instructions")
    assert not hasattr(batch, "positive_rows")
    assert not hasattr(batch, "negative_rows")

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


def test_persistent_worker_samples_match_a_fresh_epoch_resume() -> None:
    from qwen35_baton.cli.train_semantic_planner import EpochSeededRandomSampler
    from qwen35_baton.data import BatonLiberoDataset

    class _RandomizedBase(_BaseDataset):
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

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
            return {
                "probe": (
                int(random.random() * 10_000)
                + int(np.random.random() * 10_000)
                + int(torch.rand(()).item() * 10_000)
                )
                % 256
            }

    def _loader() -> tuple[
        BatonLiberoDataset, EpochSeededRandomSampler, torch.utils.data.DataLoader
    ]:
        adapted = BatonLiberoDataset(_RandomizedBase(), seed=23)

        class _RngProbe(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(adapted)

            def __getitem__(self, index: int) -> int:
                return int(adapted._load_base_sample(index)["probe"])

        probe = _RngProbe()
        sampler = EpochSeededRandomSampler(probe, seed=23)
        generator = torch.Generator().manual_seed(23)
        return (
            adapted,
            sampler,
            torch.utils.data.DataLoader(
                probe,
                batch_size=None,
                sampler=sampler,
                num_workers=2,
                persistent_workers=True,
                generator=generator,
            ),
        )

    def _values(loader: torch.utils.data.DataLoader) -> list[int]:
        return [int(sample) for sample in loader]

    continuous_dataset, continuous_sampler, continuous_loader = _loader()
    continuous_dataset.set_epoch(0)
    continuous_sampler.set_epoch(0)
    _values(continuous_loader)
    continuous_dataset.set_epoch(1)
    continuous_sampler.set_epoch(1)
    expected_epoch_one = _values(continuous_loader)

    resumed_dataset, resumed_sampler, resumed_loader = _loader()
    resumed_dataset.set_epoch(1)
    resumed_sampler.set_epoch(1)
    resumed_epoch_one = _values(resumed_loader)

    assert resumed_epoch_one == expected_epoch_one
