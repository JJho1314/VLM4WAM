"""LIBERO HDF5 adapter and independent-camera Qwen batch construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from qwen35_baton.config import BatonGeometry
from qwen35_baton.sequence import (
    BATON_TEMPLATE_KIND,
    LEGACY_TEMPLATE_KIND,
    PLAN_PAD,
    STRIP_WORLD_ARENA_INSTRUCTION_KIND,
    VERBATIM_INSTRUCTION_KIND,
    build_baton_conversation,
    build_plan_text,
    find_plan_positions,
    render_instruction,
    validate_source_indices,
)


@dataclass(frozen=True)
class BatonPlannerBatch:
    """Independent positive camera rows aligned with teacher RGB."""

    qwen_inputs: Mapping[str, torch.Tensor]
    plan_positions: torch.Tensor
    current_images: torch.Tensor
    future_images: torch.Tensor | None
    instructions: tuple[str, ...]
    row_labels: tuple[tuple[int, str], ...]
    camera_names: tuple[str, ...] = ("main", "wrist")
    future_pixel_values: torch.Tensor | None = None
    rendered_instructions: tuple[str, ...] = ()
    source_indices: tuple[tuple[int, int, int, int, int], ...] | None = None

    @property
    def batch_size(self) -> int:
        return len(self.instructions)

    @staticmethod
    def _map_tensors(
        values: Mapping[str, torch.Tensor],
        operation: Any,
    ) -> dict[str, torch.Tensor]:
        return {key: operation(value) for key, value in values.items()}

    def pin_memory(self) -> "BatonPlannerBatch":
        """Pin only tensors consumed by the GPU training hot path."""

        return replace(
            self,
            qwen_inputs=self._map_tensors(
                self.qwen_inputs, lambda value: value.pin_memory()
            ),
            plan_positions=self.plan_positions.pin_memory(),
            future_images=(
                None
                if self.future_images is None
                else self.future_images.pin_memory()
            ),
            future_pixel_values=(
                None
                if self.future_pixel_values is None
                else self.future_pixel_values.pin_memory()
            ),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "BatonPlannerBatch":
        """Move model inputs and teacher pixels without copying unused current RGB."""

        resolved = torch.device(device)

        def move(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=resolved, non_blocking=non_blocking)

        return replace(
            self,
            qwen_inputs=self._map_tensors(self.qwen_inputs, move),
            plan_positions=move(self.plan_positions),
            future_images=(
                None if self.future_images is None else move(self.future_images)
            ),
            future_pixel_values=(
                None
                if self.future_pixel_values is None
                else move(self.future_pixel_values)
            ),
        )

    def record_stream(self, stream: Any) -> None:
        """Associate asynchronously transferred tensors with the consumer stream."""

        for value in self.qwen_inputs.values():
            value.record_stream(stream)
        self.plan_positions.record_stream(stream)
        if self.future_images is not None:
            self.future_images.record_stream(stream)
        if self.future_pixel_values is not None:
            self.future_pixel_values.record_stream(stream)


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
            "source_indices": (
                self.n_previous - 1,
                *future_positions,
            ),
            "suite": record.domain,
            "episode_key": record.key,
        }


class BatonPlannerCollator:
    """Create sample-major independent positive Qwen camera rows."""

    def __init__(
        self,
        processor: Any,
        *,
        camera_names: tuple[str, ...] = ("main", "wrist"),
        plan_pad_token_id: int | None = None,
        siglip_processor: Any | None = None,
        siglip_dtype: torch.dtype = torch.bfloat16,
        batch_qwen_rows: bool = False,
        input_template_kind: str = LEGACY_TEMPLATE_KIND,
        instruction_rendering_kind: str = VERBATIM_INSTRUCTION_KIND,
    ) -> None:
        if (
            not camera_names
            or any(not isinstance(name, str) or not name for name in camera_names)
            or len(set(camera_names)) != len(camera_names)
        ):
            raise ValueError("camera_names must contain unique nonempty strings")
        self.camera_names = camera_names
        self.processor = processor
        if siglip_processor is not None and not callable(siglip_processor):
            raise TypeError("siglip_processor must be callable")
        if not isinstance(siglip_dtype, torch.dtype):
            raise TypeError("siglip_dtype must be a torch dtype")
        self.siglip_processor = siglip_processor
        self.siglip_dtype = siglip_dtype
        if input_template_kind not in {
            LEGACY_TEMPLATE_KIND,
            BATON_TEMPLATE_KIND,
        }:
            raise ValueError(
                f"unsupported input template kind: {input_template_kind!r}"
            )
        if instruction_rendering_kind not in {
            VERBATIM_INSTRUCTION_KIND,
            STRIP_WORLD_ARENA_INSTRUCTION_KIND,
        }:
            raise ValueError(
                "unsupported instruction rendering kind: "
                f"{instruction_rendering_kind!r}"
            )
        self.input_template_kind = input_template_kind
        self.instruction_rendering_kind = instruction_rendering_kind
        if type(batch_qwen_rows) is not bool:
            raise TypeError("batch_qwen_rows must be boolean")
        self.batch_qwen_rows = batch_qwen_rows
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
        self,
        image: torch.Tensor,
        instruction: str,
        source_indices: tuple[int, int, int, int, int] | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        text = self._render_row_text(image, instruction, source_indices)
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

    def _render_row_text(
        self,
        image: torch.Tensor,
        instruction: str,
        source_indices: tuple[int, int, int, int, int] | None = None,
    ) -> str:
        apply_chat_template = getattr(self.processor, "apply_chat_template", None)
        if self.input_template_kind == LEGACY_TEMPLATE_KIND:
            text = build_plan_text(instruction)
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
            return text
        if not callable(apply_chat_template):
            raise ValueError(
                "baton_assistant_time_v2 requires processor.apply_chat_template"
            )
        if source_indices is None:
            raise ValueError(
                "baton_assistant_time_v2 requires sample source_indices"
            )
        conversation = build_baton_conversation(instruction, source_indices)
        user_content = conversation[1]["content"]
        if not isinstance(user_content, list) or not isinstance(
            user_content[0], dict
        ):
            raise AssertionError("Baton conversation image content is malformed")
        user_content[0]["image"] = image
        return apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _process_rows_batched(
        self,
        rows: Sequence[
            tuple[
                int,
                str,
                torch.Tensor,
                str,
                tuple[int, int, int, int, int] | None,
            ]
        ],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        texts = [
            self._render_row_text(image, instruction, source_indices)
            for _, _, image, instruction, source_indices in rows
        ]
        images = [image for _, _, image, _, _ in rows]
        processed = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        if not isinstance(processed, Mapping) or "input_ids" not in processed:
            raise ValueError("Qwen processor must return input_ids")
        input_ids = processed["input_ids"]
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] != len(rows)
        ):
            raise ValueError("Qwen processor must return one input_ids row per image")
        attention = processed.get("attention_mask")
        if attention is not None:
            if (
                not isinstance(attention, torch.Tensor)
                or attention.shape != input_ids.shape
            ):
                raise ValueError("processor attention_mask must match input_ids")
            sequences = [
                row[mask.bool()] for row, mask in zip(input_ids, attention)
            ]
        else:
            sequences = list(input_ids)
        maximum = max(sequence.numel() for sequence in sequences)
        canonical_ids = torch.full(
            (len(sequences), maximum),
            self.processor.tokenizer.pad_token_id,
            dtype=input_ids.dtype,
        )
        canonical_attention = torch.zeros_like(canonical_ids)
        for index, sequence in enumerate(sequences):
            canonical_ids[index, : sequence.numel()] = sequence
            canonical_attention[index, : sequence.numel()] = 1
        merged = {
            key: value
            for key, value in processed.items()
            if key not in {"input_ids", "attention_mask"}
        }
        for key, value in tuple(merged.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"processor field {key} must be a tensor")
            if value.ndim == 2 and value.shape == input_ids.shape:
                canonical = torch.zeros(
                    (len(sequences), maximum), dtype=value.dtype
                )
                for index, row in enumerate(value):
                    selected = (
                        row
                        if attention is None
                        else row[attention[index].bool()]
                    )
                    canonical[index, : selected.numel()] = selected
                merged[key] = canonical
        return sequences, {
            "input_ids": canonical_ids,
            "attention_mask": canonical_attention,
            **merged,
        }

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
        camera_count = len(self.camera_names)
        if (
            current_images.ndim != 5
            or tuple(current_images.shape[1:3]) != (camera_count, 3)
            or future_images.ndim != 6
            or tuple(future_images.shape[1:4]) != (camera_count, 4, 3)
            or current_images.dtype != torch.uint8
            or future_images.dtype != torch.uint8
        ):
            raise ValueError(
                "samples must contain uint8 [C,3,H,W] and [C,4,3,H,W] RGB "
                "matching camera_names"
            )
        instructions = tuple(sample["instruction"] for sample in samples)
        suites = tuple(sample["suite"] for sample in samples)
        if any(
            not isinstance(value, str) or not value
            for value in instructions + suites
        ):
            raise ValueError("sample instructions and suites must be nonempty strings")
        rendered_instructions = tuple(
            render_instruction(instruction, self.instruction_rendering_kind)
            for instruction in instructions
        )
        raw_source_indices = tuple(sample.get("source_indices") for sample in samples)
        source_indices: tuple[tuple[int, int, int, int, int], ...] | None
        if self.input_template_kind == BATON_TEMPLATE_KIND or any(
            value is not None for value in raw_source_indices
        ):
            if any(value is None for value in raw_source_indices):
                raise ValueError("all samples must provide source_indices together")
            source_indices = tuple(
                validate_source_indices(value) for value in raw_source_indices
            )
        else:
            source_indices = None

        rows: list[
            tuple[
                int,
                str,
                torch.Tensor,
                str,
                tuple[int, int, int, int, int] | None,
            ]
        ] = []
        for sample_index, instruction in enumerate(rendered_instructions):
            for camera_index, camera in enumerate(self.camera_names):
                rows.append(
                    (
                        sample_index,
                        camera,
                        current_images[sample_index, camera_index],
                        instruction,
                        None if source_indices is None else source_indices[sample_index],
                    )
                )

        sequences: list[torch.Tensor] = []
        processed_rows: list[Mapping[str, torch.Tensor]] = []
        if self.batch_qwen_rows:
            sequences, qwen_inputs = self._process_rows_batched(rows)
        else:
            for _, _, image, instruction, indices in rows:
                sequence, processed = self._process_row(
                    image,
                    instruction,
                    indices,
                )
                sequences.append(sequence)
                processed_rows.append(processed)
        pad_token_id = getattr(
            getattr(self.processor, "tokenizer", None), "pad_token_id", 0
        )
        if type(pad_token_id) is not int:
            raise ValueError("processor tokenizer pad_token_id must be an integer")
        if not self.batch_qwen_rows:
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
        input_ids = qwen_inputs["input_ids"]
        future_pixel_values = None
        retained_future_images: torch.Tensor | None = future_images
        if self.siglip_processor is not None:
            from qwen35_baton.teacher import preprocess_siglip2_future

            future_pixel_values = preprocess_siglip2_future(
                self.siglip_processor,
                future_images,
                dtype=self.siglip_dtype,
            )
            retained_future_images = None
        return BatonPlannerBatch(
            qwen_inputs=qwen_inputs,
            plan_positions=find_plan_positions(input_ids, self.plan_pad_token_id),
            current_images=current_images,
            future_images=retained_future_images,
            instructions=instructions,
            row_labels=tuple(
                (index, camera) for index, camera, _, _, _ in rows
            ),
            camera_names=self.camera_names,
            future_pixel_values=future_pixel_values,
            rendered_instructions=rendered_instructions,
            source_indices=source_indices,
        )
