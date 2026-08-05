"""HDF5/cache join and shared teacher-forced Qwen batch construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import torch

from ge_act.data.libero_fastwam_hdf5_schema import (
    EpisodeRecord,
    load_manifest,
    validate_episode_group,
)
from qwen35_planx.cli.build_hindsight_cache import load_phrase_embedding_table
from qwen35_planx.config import CAMERA_NAMES, GroundedPlannerMetadata
from qwen35_planx.hashing import sha256_file
from qwen35_planx.instruction import format_grounded_prompt, parse_libero_instruction
from qwen35_planx.sequence import CausalPlanSequence, build_plan_sequence
from qwen35_planx.vocabulary import CAMERA_TOKENS, VisualVocabularyLayout


_ROLES = ("source", "target", "action")


def _require_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int | None, ...],
    *,
    floating: bool,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if floating != bool(value.dtype.is_floating_point):
        kind = "floating-point" if floating else "integer"
        raise TypeError(f"{name} must contain {kind} values")
    if not floating and value.dtype == torch.bool:
        raise TypeError(f"{name} must contain integer values")


@dataclass(frozen=True)
class CachedPlannerTargets:
    """Batched, teacher-only arrays read from the finalized hindsight cache."""

    codes: torch.Tensor
    relevance: torch.Tensor
    relevance_confidence: torch.Tensor
    flow: torch.Tensor
    phrase_embeddings: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor("codes", self.codes, (None, 2, 4, 729), floating=False)
        batch = int(self.codes.shape[0])
        if batch <= 0:
            raise ValueError("cached targets must contain at least one sample")
        _require_tensor(
            "relevance", self.relevance, (batch, 2, 4, 3, 729), floating=True
        )
        _require_tensor(
            "relevance_confidence",
            self.relevance_confidence,
            (batch, 2, 4, 3),
            floating=True,
        )
        _require_tensor("flow", self.flow, (batch, 2, 3, 729, 3), floating=True)
        _require_tensor(
            "phrase_embeddings",
            self.phrase_embeddings,
            (batch, 3, 1152),
            floating=True,
        )

    @property
    def batch_size(self) -> int:
        return int(self.codes.shape[0])


@dataclass(frozen=True)
class GroundedPlannerBatch:
    """Flattened dual-camera Qwen inputs and aligned hindsight supervision."""

    qwen_inputs: Mapping[str, torch.Tensor]
    code_targets: torch.Tensor
    pre_positions: torch.Tensor
    post_positions: torch.Tensor
    field_positions: torch.Tensor
    field_mask: torch.Tensor
    relevance_targets: torch.Tensor
    relevance_confidence: torch.Tensor
    flow_targets: torch.Tensor
    phrase_embeddings: torch.Tensor
    counterfactual_embeddings: torch.Tensor
    counterfactual_mask: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.code_targets.shape[0])


class HindsightPlannerDataset:
    """Read current dual-camera RGB from the HDF5 record and targets from cache."""

    def __init__(
        self,
        cache: Any,
        hdf5_manifest: Path | str,
        *,
        metadata: GroundedPlannerMetadata | None = None,
    ) -> None:
        self.cache = cache
        self.hdf5_manifest = Path(hdf5_manifest)
        _, episodes = load_manifest(self.hdf5_manifest)
        self._episodes: dict[str, EpisodeRecord] = {
            episode.key: episode for episode in episodes
        }
        records = tuple(cache.records)
        if not records:
            raise ValueError("hindsight planner dataset requires a nonempty cache")
        for record in records:
            episode = self._episodes.get(record.episode_key)
            if episode is None:
                raise ValueError(
                    f"cache window references unknown HDF5 episode: {record.episode_key}"
                )
            if record.caption != episode.caption:
                raise ValueError(
                    f"cache/HDF5 caption mismatch for {record.episode_key}"
                )
            if not 0 <= record.current_index < episode.length:
                raise ValueError(
                    f"cache current index is outside HDF5 episode: {record.sample_id}"
                )
        cache_metadata = getattr(cache, "metadata", None)
        if cache_metadata is not None:
            actual_manifest_hash = sha256_file(self.hdf5_manifest)
            if cache_metadata.hdf5_manifest_hash != actual_manifest_hash:
                raise ValueError("HDF5 manifest hash differs from hindsight cache")
        if metadata is not None:
            metadata.validate_runtime(
                hindsight_cache_hash=getattr(cache, "cache_hash", None)
            )

        instruction_suites: dict[str, str] = {}
        ambiguous: set[str] = set()
        vocabulary_values: dict[str, dict[str, set[str]]] = {}
        for record in records:
            suite = self._episodes[record.episode_key].domain
            existing = instruction_suites.get(record.caption)
            if existing is not None and existing != suite:
                ambiguous.add(record.caption)
            else:
                instruction_suites[record.caption] = suite
            if record.split != "train":
                continue
            fields = parse_libero_instruction(record.caption)
            suite_values = vocabulary_values.setdefault(
                suite, {role: set() for role in _ROLES}
            )
            for role in _ROLES:
                phrase = getattr(fields, role)
                if phrase:
                    suite_values[role].add(phrase)
        for instruction in ambiguous:
            instruction_suites.pop(instruction, None)
        self.instruction_suites = instruction_suites
        self.suite_vocabularies = {
            suite: {
                role: tuple(sorted(role_values[role]))
                for role in _ROLES
            }
            for suite, role_values in vocabulary_values.items()
        }

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cached = self.cache[index]
        record = cached.record
        episode = self._episodes[record.episode_key]
        with h5py.File(episode.shard_path, "r") as handle:
            if episode.group not in handle:
                raise ValueError(f"missing episode group: {episode.group}")
            group = handle[episode.group]
            validate_episode_group(group, episode)
            main = torch.from_numpy(group["rgb_main"][record.current_index].copy())
            wrist = torch.from_numpy(group["rgb_wrist"][record.current_index].copy())
        current_images = torch.stack((main, wrist)).permute(0, 3, 1, 2).contiguous()
        return {
            "current_images": current_images,
            "instruction": record.caption,
            "suite": episode.domain,
            "codes": cached.codes,
            "relevance": cached.relevance,
            "relevance_confidence": cached.confidence,
            "flow": cached.flow,
            "phrase_embeddings": cached.phrase_embeddings,
        }


def _normalise_vocabulary(
    vocabulary: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(vocabulary) != set(_ROLES):
        raise ValueError(f"phrase vocabulary roles must be exactly {_ROLES!r}")
    result = {}
    for role in _ROLES:
        values = tuple(str(value) for value in vocabulary[role] if value)
        if len(set(values)) != len(values):
            raise ValueError(f"{role} phrase vocabulary contains duplicates")
        result[role] = values
    return result


class GroundedPlannerCollator:
    """Single implementation shared by dataset and provider teacher forcing."""

    def __init__(
        self,
        processor: Any,
        layout: VisualVocabularyLayout,
        *,
        cache_dir: Path | str | None = None,
        phrase_vocabulary: Mapping[str, Sequence[str]] | None = None,
        phrase_embeddings: Mapping[str, torch.Tensor] | None = None,
        suite_vocabularies: Mapping[
            str, Mapping[str, Sequence[str]]
        ] | None = None,
        instruction_suites: Mapping[str, str] | None = None,
        dataset: HindsightPlannerDataset | None = None,
        metadata: GroundedPlannerMetadata | None = None,
        max_negatives: int = 1,
    ) -> None:
        if type(max_negatives) is not int or max_negatives < 0:
            raise ValueError("max_negatives must be a non-negative integer")
        if cache_dir is not None:
            if phrase_vocabulary is not None or phrase_embeddings is not None:
                raise ValueError(
                    "provide cache_dir or injected phrase tables, not both"
                )
            phrase_vocabulary, phrase_embeddings = load_phrase_embedding_table(
                cache_dir
            )
        if phrase_vocabulary is None or phrase_embeddings is None:
            raise ValueError("a verified cache phrase table is required")
        self.processor = processor
        self.layout = layout
        self.max_negatives = max_negatives
        self.phrase_vocabulary = _normalise_vocabulary(phrase_vocabulary)
        if set(phrase_embeddings) != set(_ROLES):
            raise ValueError(f"phrase embedding roles must be exactly {_ROLES!r}")
        self.phrase_embeddings: dict[str, torch.Tensor] = {}
        self._embedding_lookup: dict[str, dict[str, torch.Tensor]] = {}
        for role in _ROLES:
            table = phrase_embeddings[role]
            expected = (len(self.phrase_vocabulary[role]), 1152)
            if (
                not isinstance(table, torch.Tensor)
                or tuple(table.shape) != expected
                or not table.dtype.is_floating_point
                or not bool(torch.isfinite(table).all())
            ):
                raise ValueError(
                    f"{role} phrase embeddings must be finite with shape {expected}"
                )
            table = table.detach().cpu().contiguous()
            self.phrase_embeddings[role] = table
            self._embedding_lookup[role] = dict(
                zip(self.phrase_vocabulary[role], table)
            )

        if dataset is not None:
            if suite_vocabularies is None:
                suite_vocabularies = dataset.suite_vocabularies
            if instruction_suites is None:
                instruction_suites = dataset.instruction_suites
        if not suite_vocabularies or not instruction_suites:
            raise ValueError(
                "explicit train-only suite provenance and instruction suites "
                "are required for counterfactual negatives"
            )
        if "*" in suite_vocabularies:
            raise ValueError("global '*' suite provenance is not permitted")
        self.suite_vocabularies = {
            str(suite): _normalise_vocabulary(vocabulary)
            for suite, vocabulary in suite_vocabularies.items()
        }
        self.instruction_suites = {
            str(instruction): str(suite)
            for instruction, suite in instruction_suites.items()
        }
        unknown_suites = sorted(
            set(self.instruction_suites.values()).difference(self.suite_vocabularies)
        )
        if unknown_suites:
            raise ValueError(
                "instruction suite provenance references unknown suites: "
                + ", ".join(unknown_suites)
            )
        for suite, vocabulary in self.suite_vocabularies.items():
            for role in _ROLES:
                unavailable = sorted(
                    set(vocabulary[role]).difference(self._embedding_lookup[role])
                )
                if unavailable:
                    raise ValueError(
                        f"suite {suite!r} contains unverified {role} phrases: "
                        + ", ".join(unavailable)
                    )
        if metadata is not None:
            if (
                metadata.visual_token_start_id != layout.visual_start_id
                or metadata.visual_token_end_id != layout.visual_end_id
                or metadata.tokenizer_hash != layout.tokenizer_hash
            ):
                raise ValueError("planner metadata differs from visual vocabulary")

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> GroundedPlannerBatch:
        if not samples:
            raise ValueError("collator requires at least one sample")
        current_images = torch.stack(
            [sample["current_images"] for sample in samples]
        )
        instructions = tuple(str(sample["instruction"]) for sample in samples)
        for sample, instruction in zip(samples, instructions):
            suite = str(sample["suite"])
            configured = self.instruction_suites.get(instruction)
            if configured is not None and configured != suite:
                raise ValueError("sample suite differs from the instruction suite")
        targets = CachedPlannerTargets(
            codes=torch.stack([sample["codes"] for sample in samples]).long(),
            relevance=torch.stack([sample["relevance"] for sample in samples]),
            relevance_confidence=torch.stack(
                [sample["relevance_confidence"] for sample in samples]
            ),
            flow=torch.stack([sample["flow"] for sample in samples]),
            phrase_embeddings=torch.stack(
                [sample["phrase_embeddings"] for sample in samples]
            ),
        )
        return self.build_teacher_forced(current_images, instructions, targets)

    def _processor_sequence(
        self,
        *,
        image: torch.Tensor,
        instruction: str,
        camera: str,
        codes: torch.Tensor,
    ) -> tuple[CausalPlanSequence, Mapping[str, torch.Tensor]]:
        fields = parse_libero_instruction(instruction)
        camera_token = CAMERA_TOKENS[CAMERA_NAMES.index(camera)]
        text = f"{camera_token}\n{format_grounded_prompt(fields)}"
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
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        if "input_ids" not in processed:
            raise ValueError("Qwen processor must return input_ids")
        prompt_ids = processed["input_ids"]
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("Qwen processor must return one prompt sequence")
        if "attention_mask" in processed:
            prompt_attention = processed["attention_mask"]
            if prompt_attention.shape != prompt_ids.shape:
                raise ValueError("processor attention_mask must match input_ids")
            prompt_ids = prompt_ids[0][prompt_attention[0].bool()]
        else:
            prompt_ids = prompt_ids[0]
        sequence = build_plan_sequence(
            camera=camera,
            prompt=prompt_ids,
            codes=codes,
            layout=self.layout,
            field_mask=tuple(bool(value) for value in fields.confidences),
        )
        return sequence, processed

    def _suite_for(self, instruction: str) -> str | None:
        return self.instruction_suites.get(instruction)

    def _counterfactuals(
        self,
        instructions: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(instructions)
        dtype = next(iter(self.phrase_embeddings.values())).dtype
        embeddings = torch.zeros(
            (batch, 3, self.max_negatives, 1152), dtype=dtype
        )
        mask = torch.zeros((batch, 3, self.max_negatives), dtype=torch.bool)
        for batch_index, instruction in enumerate(instructions):
            fields = parse_libero_instruction(instruction)
            suite = self._suite_for(instruction)
            if suite is None:
                continue
            vocabulary = self.suite_vocabularies.get(suite)
            if vocabulary is None:
                continue
            for role_index, role in enumerate(_ROLES):
                positive = getattr(fields, role)
                if not positive:
                    continue
                candidates = tuple(
                    phrase
                    for phrase in vocabulary[role]
                    if phrase != positive and phrase in self._embedding_lookup[role]
                )[: self.max_negatives]
                for negative_index, phrase in enumerate(candidates):
                    embeddings[batch_index, role_index, negative_index].copy_(
                        self._embedding_lookup[role][phrase]
                    )
                    mask[batch_index, role_index, negative_index] = True
        return embeddings, mask

    @staticmethod
    def _pad_sequences(
        sequences: Sequence[CausalPlanSequence],
        *,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = max(len(sequence.input_ids) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), maximum), pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            length = len(sequence.input_ids)
            input_ids[index, :length] = sequence.input_ids
            attention_mask[index, :length] = 1
        return input_ids, attention_mask

    @staticmethod
    def _merge_processor_values(
        processed_values: Sequence[Mapping[str, torch.Tensor]],
        *,
        sequences: Sequence[CausalPlanSequence],
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        keys = set(processed_values[0]).difference({"input_ids", "attention_mask"})
        if any(
            set(values).difference({"input_ids", "attention_mask"}) != keys
            for values in processed_values
        ):
            raise ValueError("processor returned inconsistent fields across examples")
        maximum = max(len(sequence.input_ids) for sequence in sequences)
        for key in sorted(keys):
            values = [item[key] for item in processed_values]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"processor field {key} must be a tensor")
            is_sequence_aligned = all(
                value.ndim == 2
                and value.shape[0] == 1
                and value.shape[1] == item["input_ids"].shape[1]
                for value, item in zip(values, processed_values)
            )
            if is_sequence_aligned:
                padded = torch.zeros(
                    (len(values), maximum), dtype=values[0].dtype
                )
                for index, (value, item) in enumerate(
                    zip(values, processed_values)
                ):
                    if "attention_mask" in item:
                        value = value[0][item["attention_mask"][0].bool()]
                    else:
                        value = value[0]
                    padded[index, : len(value)] = value
                merged[key] = padded
            else:
                try:
                    merged[key] = torch.cat(values, dim=0)
                except RuntimeError as error:
                    raise ValueError(
                        f"processor field {key} cannot be concatenated"
                    ) from error
        return merged

    def build_teacher_forced(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
        targets: CachedPlannerTargets,
    ) -> GroundedPlannerBatch:
        """Build the exact same flattened batch used by ``__call__``."""

        if (
            not isinstance(current_images, torch.Tensor)
            or current_images.ndim != 5
            or current_images.shape[1] != 2
            or current_images.shape[2] != 3
        ):
            raise ValueError("current_images must have shape [B,2,3,H,W]")
        batch_size = int(current_images.shape[0])
        if batch_size <= 0:
            raise ValueError("current_images must contain at least one sample")
        if len(instructions) != batch_size or targets.batch_size != batch_size:
            raise ValueError("images, instructions, and targets batch sizes must match")

        sequences: list[CausalPlanSequence] = []
        processed_values: list[Mapping[str, torch.Tensor]] = []
        for batch_index, instruction in enumerate(instructions):
            if type(instruction) is not str or not instruction:
                raise ValueError("instructions must contain nonempty strings")
            for camera_index, camera in enumerate(CAMERA_NAMES):
                sequence, processed = self._processor_sequence(
                    image=current_images[batch_index, camera_index],
                    instruction=instruction,
                    camera=camera,
                    codes=targets.codes[batch_index, camera_index],
                )
                sequences.append(sequence)
                processed_values.append(processed)

        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        input_ids, attention_mask = self._pad_sequences(
            sequences, pad_token_id=int(pad_token_id)
        )
        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **self._merge_processor_values(
                processed_values,
                sequences=sequences,
            ),
        }
        negatives, negative_mask = self._counterfactuals(instructions)
        return GroundedPlannerBatch(
            qwen_inputs=qwen_inputs,
            code_targets=torch.stack(
                [sequence.code_targets for sequence in sequences]
            ),
            pre_positions=torch.stack(
                [sequence.pre_positions for sequence in sequences]
            ),
            post_positions=torch.stack(
                [sequence.post_positions for sequence in sequences]
            ),
            field_positions=torch.stack(
                [sequence.field_positions for sequence in sequences]
            ),
            field_mask=torch.stack([sequence.field_mask for sequence in sequences]),
            relevance_targets=targets.relevance.reshape(
                batch_size * 2, 4, 3, 729
            ),
            relevance_confidence=targets.relevance_confidence.reshape(
                batch_size * 2, 4, 3
            ),
            flow_targets=targets.flow.reshape(batch_size * 2, 3, 729, 3),
            phrase_embeddings=targets.phrase_embeddings.repeat_interleave(2, dim=0),
            counterfactual_embeddings=negatives.repeat_interleave(2, dim=0),
            counterfactual_mask=negative_mask.repeat_interleave(2, dim=0),
        )
