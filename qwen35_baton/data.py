"""LIBERO HDF5 adapter and independent-camera Qwen batch construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from qwen35_baton.config import BatonGeometry
from qwen35_baton.sequence import PLAN_PAD, build_plan_text, find_plan_positions


@dataclass(frozen=True)
class BatonPlannerBatch:
    """Positive and counterfactual camera rows aligned with teacher RGB."""

    qwen_inputs: Mapping[str, torch.Tensor]
    plan_positions: torch.Tensor
    current_images: torch.Tensor
    future_images: torch.Tensor
    instructions: tuple[str, ...]
    negative_instructions: tuple[str, ...]
    row_labels: tuple[tuple[str, int, str], ...]

    @property
    def batch_size(self) -> int:
        return len(self.instructions)

    @property
    def positive_rows(self) -> slice:
        return slice(0, self.batch_size * 2)

    @property
    def negative_rows(self) -> slice:
        return slice(self.batch_size * 2, self.batch_size * 4)


class BatonLiberoDataset(Dataset[dict[str, Any]]):
    """Adapt normalized FastWAM videos into current and four future RGB frames."""

    def __init__(
        self,
        base_dataset: Any,
        *,
        seed: int = 0,
        geometry: BatonGeometry | None = None,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if geometry is None:
            geometry = BatonGeometry()
        if not isinstance(geometry, BatonGeometry):
            raise TypeError("geometry must be BatonGeometry")
        records = tuple(getattr(base_dataset, "records", ()))
        n_previous = getattr(base_dataset, "n_previous", None)
        if not records:
            raise ValueError("base dataset must expose nonempty manifest records")
        if type(n_previous) is not int or n_previous <= 0:
            raise ValueError("base dataset must expose a positive n_previous")
        for record in records:
            for field in ("key", "caption", "domain"):
                value = getattr(record, field, None)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"manifest records require nonempty {field}")

        captions: dict[str, set[str]] = {}
        for record in records:
            captions.setdefault(record.domain, set()).add(record.caption)
        self.suite_to_captions = {
            suite: tuple(sorted(values)) for suite, values in sorted(captions.items())
        }
        invalid = [
            suite for suite, values in self.suite_to_captions.items() if len(values) < 2
        ]
        if invalid:
            raise ValueError(
                "each suite must contain at least two distinct instructions: "
                + ", ".join(invalid)
            )

        self.base_dataset = base_dataset
        self.records = records
        self.seed = seed
        self.geometry = geometry
        self.n_previous = n_previous
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_epoch(self, epoch: int) -> None:
        """Publish the current sampler epoch to persistent worker processes."""

        if type(epoch) is not int or epoch < 0:
            raise ValueError("dataset epoch must be a non-negative integer")
        self._shared_epoch.fill_(epoch)

    def _source_index(self, index: int) -> int:
        selected = getattr(self.base_dataset, "fix_epiidx", None)
        if selected is None:
            selected = index
        if type(selected) is not int:
            raise TypeError("base dataset selected index must be an integer")
        if selected < 0:
            selected += len(self.records)
        if selected < 0 or selected >= len(self.records):
            raise IndexError("base dataset selected index is outside manifest records")
        return selected

    def _record_for_index(self, index: int) -> Any:
        return self.records[self._source_index(index)]

    def _negative_instruction(self, record: Any) -> str:
        candidates = tuple(
            caption
            for caption in self.suite_to_captions[record.domain]
            if caption != record.caption
        )
        if not candidates:
            raise ValueError("suite has no counterfactual instruction")
        payload = json.dumps(
            (self.seed, record.key, record.caption),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        choice = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        return candidates[choice % len(candidates)]

    def _load_base_sample(self, index: int) -> Any:
        """Run legacy stochastic transforms in an epoch/index-local RNG scope."""

        source_index = self._source_index(index)
        epoch = int(self._shared_epoch.item())
        seed_payload = json.dumps(
            (self.seed, epoch, source_index),
            separators=(",", ":"),
        ).encode("utf-8")
        sample_seed = int.from_bytes(
            hashlib.sha256(seed_payload).digest()[:8], "big"
        )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32))
            torch.manual_seed(sample_seed)
            sample = self.base_dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._load_base_sample(index)
        if not isinstance(sample, Mapping):
            raise TypeError("base dataset samples must be mappings")
        video = sample.get("video")
        if (
            not isinstance(video, torch.Tensor)
            or video.ndim != 5
            or tuple(video.shape[:2]) != (3, 2)
            or tuple(video.shape[-2:])
            != (self.geometry.image_size, self.geometry.image_size)
            or not video.dtype.is_floating_point
            or not bool(torch.isfinite(video).all())
        ):
            raise ValueError(
                "base video must be finite normalized [3,2,T,256,256] tensor"
            )
        required_frames = self.n_previous + max(self.geometry.future_indices) + 1
        if video.shape[2] < required_frames:
            raise ValueError(
                f"base video needs at least {required_frames} frames, got {video.shape[2]}"
            )
        record = self._record_for_index(index)
        caption = sample.get("caption")
        if caption != record.caption:
            raise ValueError("base sample caption differs from manifest record")

        rgb = (
            video.add(1)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
        )
        future_positions = tuple(
            self.n_previous + offset for offset in self.geometry.future_indices
        )
        return {
            "current_images": rgb[:, self.n_previous - 1],
            "future_images": rgb[:, list(future_positions)],
            "instruction": record.caption,
            "negative_instruction": self._negative_instruction(record),
            "suite": record.domain,
            "episode_key": record.key,
        }


class BatonPlannerCollator:
    """Create positive then negative, sample-major independent Qwen camera rows."""

    def __init__(self, processor: Any, *, plan_pad_token_id: int | None = None) -> None:
        self.processor = processor
        if plan_pad_token_id is None:
            tokenizer = getattr(processor, "tokenizer", None)
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if not callable(convert):
                raise ValueError(
                    "processor tokenizer must expose convert_tokens_to_ids"
                )
            plan_pad_token_id = convert(PLAN_PAD)
        if type(plan_pad_token_id) is not int or plan_pad_token_id < 0:
            raise ValueError("plan_pad_token_id must be a non-negative integer")
        self.plan_pad_token_id = plan_pad_token_id

    def _process_row(
        self, image: torch.Tensor, instruction: str
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        text = build_plan_text(instruction)
        apply_chat_template = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            text = apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        processed = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        )
        if not isinstance(processed, Mapping) or "input_ids" not in processed:
            raise ValueError("Qwen processor must return input_ids")
        input_ids = processed["input_ids"]
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise ValueError("Qwen processor must return one rank-2 input_ids row")
        attention = processed.get("attention_mask")
        if attention is not None:
            if (
                not isinstance(attention, torch.Tensor)
                or attention.shape != input_ids.shape
            ):
                raise ValueError("processor attention_mask must match input_ids")
            input_ids = input_ids[0][attention[0].bool()]
        else:
            input_ids = input_ids[0]
        return input_ids, processed

    @staticmethod
    def _merge_processor_values(
        processed_rows: Sequence[Mapping[str, torch.Tensor]],
        sequence_rows: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        keys = set(processed_rows[0]).difference({"input_ids", "attention_mask"})
        if any(
            set(row).difference({"input_ids", "attention_mask"}) != keys
            for row in processed_rows
        ):
            raise ValueError("processor returned inconsistent fields across rows")
        maximum = max(row.numel() for row in sequence_rows)
        merged: dict[str, torch.Tensor] = {}
        for key in sorted(keys):
            values = [row[key] for row in processed_rows]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"processor field {key} must be a tensor")
            sequence_aligned = all(
                value.ndim == 2
                and value.shape[0] == 1
                and value.shape[1] == row["input_ids"].shape[1]
                for value, row in zip(values, processed_rows)
            )
            if sequence_aligned:
                padded = torch.zeros((len(values), maximum), dtype=values[0].dtype)
                for index, (value, row) in enumerate(zip(values, processed_rows)):
                    attention = row.get("attention_mask")
                    values_row = (
                        value[0] if attention is None else value[0][attention[0].bool()]
                    )
                    padded[index, : values_row.numel()] = values_row
                merged[key] = padded
            else:
                try:
                    merged[key] = torch.cat(values, dim=0)
                except RuntimeError as error:
                    raise ValueError(
                        f"processor field {key} cannot be concatenated"
                    ) from error
        return merged

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> BatonPlannerBatch:
        if not samples:
            raise ValueError("collator requires at least one sample")
        current_images = torch.stack([sample["current_images"] for sample in samples])
        future_images = torch.stack([sample["future_images"] for sample in samples])
        if (
            current_images.ndim != 5
            or tuple(current_images.shape[1:3]) != (2, 3)
            or future_images.ndim != 6
            or tuple(future_images.shape[1:4]) != (2, 4, 3)
            or current_images.dtype != torch.uint8
            or future_images.dtype != torch.uint8
        ):
            raise ValueError("samples must contain uint8 [2,3,H,W] and [2,4,3,H,W] RGB")
        instructions = tuple(sample["instruction"] for sample in samples)
        negatives = tuple(sample["negative_instruction"] for sample in samples)
        suites = tuple(sample["suite"] for sample in samples)
        if any(
            not isinstance(value, str) or not value
            for value in instructions + negatives + suites
        ):
            raise ValueError("sample instructions and suites must be nonempty strings")
        if any(
            positive == negative for positive, negative in zip(instructions, negatives)
        ):
            raise ValueError(
                "negative instructions must differ from positive instructions"
            )

        rows: list[tuple[str, int, str, torch.Tensor, str]] = []
        for condition, condition_instructions in (
            ("positive", instructions),
            ("negative", negatives),
        ):
            for sample_index, instruction in enumerate(condition_instructions):
                for camera_index, camera in enumerate(("main", "wrist")):
                    rows.append(
                        (
                            condition,
                            sample_index,
                            camera,
                            current_images[sample_index, camera_index],
                            instruction,
                        )
                    )

        sequences: list[torch.Tensor] = []
        processed_rows: list[Mapping[str, torch.Tensor]] = []
        for _, _, _, image, instruction in rows:
            sequence, processed = self._process_row(image, instruction)
            sequences.append(sequence)
            processed_rows.append(processed)
        pad_token_id = getattr(
            getattr(self.processor, "tokenizer", None), "pad_token_id", 0
        )
        if type(pad_token_id) is not int:
            raise ValueError("processor tokenizer pad_token_id must be an integer")
        maximum = max(sequence.numel() for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), maximum), pad_token_id, dtype=sequences[0].dtype
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, : sequence.numel()] = sequence
            attention_mask[index, : sequence.numel()] = 1
        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **self._merge_processor_values(processed_rows, sequences),
        }
        return BatonPlannerBatch(
            qwen_inputs=qwen_inputs,
            plan_positions=find_plan_positions(input_ids, self.plan_pad_token_id),
            current_images=current_images,
            future_images=future_images,
            instructions=instructions,
            negative_instructions=negatives,
            row_labels=tuple(
                (condition, index, camera) for condition, index, camera, _, _ in rows
            ),
        )
