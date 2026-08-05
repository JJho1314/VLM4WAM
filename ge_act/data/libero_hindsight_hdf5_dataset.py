"""Cache-aligned LIBERO HDF5 windows for joint grounded-plan training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ge_act.data.libero_fastwam_hdf5_dataset import (
    LiberoFastWAMHDF5Dataset,
)
from qwen35_planx.hashing import sha256_file
from qwen35_planx.hindsight_schema import HindsightCache
from qwen35_planx.instruction import parse_libero_instruction


_ROLES = ("source", "target", "action")


class LiberoHindsightHDF5Dataset(Dataset):
    """Join finalized hindsight targets to exact normalized GE-Act windows."""

    def __init__(
        self,
        manifest_path: str | Path,
        hindsight_cache: str | Path,
        stat_file: str | Path,
        *,
        train_dataset: bool = True,
        **base_kwargs: Any,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.hindsight_cache = Path(hindsight_cache)
        self.cache = HindsightCache.open(self.hindsight_cache)
        try:
            actual_manifest_hash = sha256_file(self.manifest_path)
            if self.cache.metadata.hdf5_manifest_hash != actual_manifest_hash:
                raise ValueError("HDF5 manifest differs from hindsight cache")
            base_kwargs.pop("previous_pick_mode", None)
            self.base = LiberoFastWAMHDF5Dataset(
                manifest_path=self.manifest_path,
                stat_file=stat_file,
                train_dataset=train_dataset,
                previous_pick_mode="uniform",
                **base_kwargs,
            )
            episode_indexes = {
                record.key: index
                for index, record in enumerate(self.base.records)
            }
            selected_split = "train" if train_dataset else "val"
            self.cache_indexes = tuple(
                index
                for index, record in enumerate(self.cache.records)
                if record.split == selected_split
            )
            if not self.cache_indexes:
                raise ValueError(
                    f"hindsight cache has no {selected_split} windows"
                )
            self._episode_indexes: dict[str, int] = {}
            for cache_index in self.cache_indexes:
                record = self.cache.records[cache_index]
                episode_index = episode_indexes.get(record.episode_key)
                if episode_index is None:
                    raise ValueError(
                        "hindsight cache references unknown HDF5 episode: "
                        f"{record.episode_key}"
                    )
                episode = self.base.records[episode_index]
                if record.caption != episode.caption:
                    raise ValueError(
                        "hindsight cache caption differs from HDF5 episode: "
                        f"{record.episode_key}"
                    )
                if not 0 <= record.current_index < episode.length:
                    raise ValueError(
                        "hindsight cache current index is outside HDF5 episode: "
                        f"{record.sample_id}"
                    )
                self._episode_indexes[record.episode_key] = episode_index
            (
                self.instruction_suites,
                self.suite_vocabularies,
            ) = self._build_phrase_provenance(episode_indexes)
        except Exception:
            self.cache.close()
            raise

    def _build_phrase_provenance(
        self,
        episode_indexes: dict[str, int],
    ) -> tuple[
        dict[str, str],
        dict[str, dict[str, tuple[str, ...]]],
    ]:
        instruction_suites: dict[str, str] = {}
        ambiguous: set[str] = set()
        values: dict[str, dict[str, set[str]]] = {}
        for record in self.cache.records:
            episode = self.base.records[episode_indexes[record.episode_key]]
            suite = episode.domain
            previous = instruction_suites.get(record.caption)
            if previous is not None and previous != suite:
                ambiguous.add(record.caption)
            else:
                instruction_suites[record.caption] = suite
            if record.split != "train":
                continue
            fields = parse_libero_instruction(record.caption)
            suite_values = values.setdefault(
                suite,
                {role: set() for role in _ROLES},
            )
            for role in _ROLES:
                phrase = getattr(fields, role)
                if phrase:
                    suite_values[role].add(phrase)
        for instruction in ambiguous:
            instruction_suites.pop(instruction, None)
        suite_vocabularies = {
            suite: {
                role: tuple(sorted(role_values[role]))
                for role in _ROLES
            }
            for suite, role_values in values.items()
        }
        return instruction_suites, suite_vocabularies

    def __len__(self) -> int:
        return len(self.cache_indexes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        if type(index) is not int:
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"dataset index out of range: {index}")
        cached = self.cache[self.cache_indexes[index]]
        record = cached.record
        base_sample = self.base.read_by_indexes(
            self._episode_indexes[record.episode_key],
            record.frame_indices,
            record.action_indices,
        )
        video = base_sample["video"]
        if not isinstance(video, torch.Tensor):
            raise TypeError("base HDF5 video must be a tensor")
        current_images = (
            video[:, :, 3]
            .permute(1, 0, 2, 3)
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        return {
            **base_sample,
            "current_images": current_images,
            "episode_key": record.episode_key,
            "current_index": record.current_index,
            "sample_id": record.sample_id,
            "suite": self.base.records[
                self._episode_indexes[record.episode_key]
            ].domain,
            "target_codes": cached.codes,
            "target_relevance": cached.relevance,
            "target_relevance_confidence": cached.confidence,
            "target_flow": cached.flow,
            "target_phrase_embeddings": cached.phrase_embeddings,
        }

    def close(self) -> None:
        base = getattr(self, "base", None)
        if base is not None:
            base.close()
        cache = getattr(self, "cache", None)
        if cache is not None:
            cache.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
