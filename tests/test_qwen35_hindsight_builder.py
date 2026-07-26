from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from qwen35_planx.config import HindsightCacheMetadata
from qwen35_planx.hashing import sha256_file, sha256_json
from qwen35_planx.hindsight_data import HDF5Trajectory, HindsightWindowRecord
from qwen35_planx.instruction import parse_libero_instruction
from qwen35_planx.siglip_relevance import PhraseRelevance


def _window(
    *,
    sample_id: str = "libero_goal:000000:000000000",
    split: str = "train",
    caption: str = (
        "pick up the black bowl and place it on the white plate"
    ),
) -> HindsightWindowRecord:
    actions = (1, 1, 1, 1) + tuple(min(index, 11) for index in range(36))
    frames = (1, 1, 1, 1) + actions[7::4]
    return HindsightWindowRecord(
        sample_id=sample_id,
        episode_key="libero_goal:000000",
        split=split,
        caption=caption,
        current_index=1,
        future_indices=(3, 11, 11, 11),
        frame_indices=frames,
        action_indices=actions,
    )


def _trajectory() -> HDF5Trajectory:
    rgb = np.zeros((2, 12, 256, 256, 3), dtype=np.uint8)
    for camera in range(2):
        for frame in range(12):
            rgb[camera, frame, :, :, :] = camera * 40 + frame
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:3, 6] = 1
    actions[3:8, 6] = 0
    actions[8:, 6] = 1
    states = np.zeros((12, 8), dtype=np.float32)
    states[:3, 6:8] = (-1, 1)
    states[3:8, 6:8] = (-0.1, 0.1)
    states[8:, 6:8] = (-1, 1)
    return HDF5Trajectory(rgb=rgb, actions=actions, states=states)


class _FakeDino:
    def __init__(self) -> None:
        self.frames_seen = 0
        self.microbatch_size = None

    def encode(self, rgb: torch.Tensor, *, microbatch_size: int) -> torch.Tensor:
        self.frames_seen = int(rgb.shape[1])
        self.microbatch_size = microbatch_size
        cameras, frames = rgb.shape[:2]
        positions = torch.arange(729, dtype=torch.float32)
        features = torch.stack(
            (
                torch.sin(positions / 31),
                torch.cos(positions / 31),
                torch.sin(positions / 17),
                torch.cos(positions / 17),
            ),
            dim=-1,
        )
        return features.view(1, 1, 729, 4).expand(cameras, frames, -1, -1).clone()


class _FakeSiglip:
    def __init__(self, *, confidence: float = 1.0) -> None:
        self.confidence = confidence

    def encode_fields(
        self,
        rgb: torch.Tensor,
        fields,
        *,
        counterfactual_phrases,
    ) -> PhraseRelevance:
        frame_count = rgb.shape[0]
        maps = torch.full((frame_count, 3, 27, 27), 1 / 729)
        embeddings = torch.zeros(3, 1152)
        embeddings[0, 0] = 1
        embeddings[1, 1] = 1
        embeddings[2, 2] = 1
        return PhraseRelevance(
            phrase_embeddings=embeddings,
            maps=maps,
            confidence=torch.full((frame_count, 3), self.confidence),
        )


class _FakeTATok:
    def encode_codes(self, images: torch.Tensor):
        values = torch.arange(images.shape[0], dtype=torch.long).view(-1, 1)
        return SimpleNamespace(codes=values.expand(-1, 729).clone())


@pytest.fixture
def fake_builder_inputs():
    dino = _FakeDino()
    return SimpleNamespace(
        trajectory=_trajectory(),
        window=_window(),
        dino=dino,
        components={
            "ta_tokenizer": _FakeTATok(),
            "siglip_teacher": _FakeSiglip(),
            "dino_teacher": dino,
            "microbatch_size": 3,
        },
    )


def test_builder_uses_complete_video_but_stores_only_k4(fake_builder_inputs) -> None:
    from qwen35_planx.hindsight_builder import HindsightTargetBuilder

    builder = HindsightTargetBuilder.from_components(
        **fake_builder_inputs.components
    )
    target = builder.build_window(
        fake_builder_inputs.trajectory,
        fake_builder_inputs.window,
    )

    assert (
        fake_builder_inputs.dino.frames_seen
        == fake_builder_inputs.trajectory.rgb.shape[1]
    )
    assert fake_builder_inputs.dino.microbatch_size == 3
    assert target.codes.shape == (2, 4, 729)
    assert target.relevance.shape == (2, 4, 3, 729)
    assert target.confidence.shape == (2, 4, 3)
    assert target.flow.shape == (2, 3, 729, 3)
    assert target.phrase_embeddings.shape == (3, 1152)
    assert target.teacher_only_fields == ()
    assert set(target.__dataclass_fields__) == {
        "codes",
        "relevance",
        "confidence",
        "flow",
        "phrase_embeddings",
    }


def test_builder_is_deterministic_and_masks_missing_fields(
    fake_builder_inputs,
) -> None:
    from qwen35_planx.hindsight_builder import HindsightTargetBuilder

    builder = HindsightTargetBuilder.from_components(
        **fake_builder_inputs.components
    )
    incomplete = replace(
        fake_builder_inputs.window,
        caption="open the wooden drawer",
    )
    first = builder.build_window(fake_builder_inputs.trajectory, incomplete)
    second = builder.build_window(fake_builder_inputs.trajectory, incomplete)

    for field in first.__dataclass_fields__:
        torch.testing.assert_close(getattr(first, field), getattr(second, field))
    assert torch.count_nonzero(first.confidence[:, :, 1]) == 0
    torch.testing.assert_close(
        first.relevance.sum(dim=-1),
        torch.ones(2, 4, 3),
    )


def test_builder_preserves_counterfactual_teacher_zero_confidence(
    fake_builder_inputs,
) -> None:
    from qwen35_planx.hindsight_builder import HindsightTargetBuilder

    components = {
        **fake_builder_inputs.components,
        "siglip_teacher": _FakeSiglip(confidence=0.0),
    }
    builder = HindsightTargetBuilder.from_components(**components)

    target = builder.build_window(
        fake_builder_inputs.trajectory,
        fake_builder_inputs.window,
    )

    assert torch.count_nonzero(target.confidence) == 0


def test_builder_rejects_indices_outside_complete_trajectory(
    fake_builder_inputs,
) -> None:
    from qwen35_planx.hindsight_builder import HindsightTargetBuilder

    actions = (1, 1, 1, 1) + tuple(range(36))
    frames = (1, 1, 1, 1) + actions[7::4]
    invalid = replace(
        fake_builder_inputs.window,
        future_indices=(3, 15, 23, 35),
        frame_indices=frames,
        action_indices=actions,
    )
    builder = HindsightTargetBuilder.from_components(
        **fake_builder_inputs.components
    )

    with pytest.raises(ValueError, match="trajectory bounds"):
        builder.build_window(fake_builder_inputs.trajectory, invalid)


def test_counterfactual_vocabulary_uses_train_split_only() -> None:
    from qwen35_planx.hindsight_builder import build_counterfactual_vocabulary

    train = _window()
    val = replace(
        _window(
            sample_id="libero_goal:000001:000000000",
            split="val",
            caption="open the forbidden validation drawer",
        ),
        episode_key="libero_goal:000001",
    )

    vocabulary = build_counterfactual_vocabulary([val, train])

    parsed = parse_libero_instruction(train.caption)
    assert parsed.source in vocabulary.sources
    assert parsed.target in vocabulary.targets
    assert parsed.action in vocabulary.actions
    assert all("forbidden" not in phrase for phrase in vocabulary.sources)
    assert all("forbidden" not in phrase for phrase in vocabulary.targets)


def test_change_map_compares_cycle_matched_dino_positions() -> None:
    from qwen35_planx.hindsight_builder import _change_map

    generator = torch.Generator().manual_seed(7)
    initial = torch.nn.functional.normalize(
        torch.randn(729, 8, generator=generator),
        dim=-1,
    )
    final = torch.zeros_like(initial)
    flow = torch.zeros(729, 3)
    for y in range(27):
        for x in range(26):
            source = y * 27 + x
            target = source + 1
            final[target] = initial[source]
            flow[source] = torch.tensor((1, 0, 1))

    change, confidence = _change_map(initial, final, flow)

    torch.testing.assert_close(change, torch.full((729,), 1 / 729))
    assert confidence == 0.0


def _write_fake_hdf5_manifest(root: Path) -> Path:
    root.mkdir(parents=True)
    shard = root / "episodes.hdf5"
    trajectory = _trajectory()
    with h5py.File(shard, "w") as handle:
        group = handle.create_group("episodes/libero_goal:000000")
        string_dtype = h5py.string_dtype("utf-8")
        group.create_dataset("caption", data=_window().caption, dtype=string_dtype)
        group.create_dataset("domain", data="libero_goal", dtype=string_dtype)
        group.create_dataset("episode_index", data=np.int64(0))
        group.create_dataset("length", data=np.int64(12))
        group.create_dataset("rgb_main", data=trajectory.rgb[0])
        group.create_dataset("rgb_wrist", data=trajectory.rgb[1])
        group.create_dataset("action", data=trajectory.actions)
        group.create_dataset("state", data=trajectory.states)
    manifest = {
        "schema_version": 1,
        "compression": "none",
        "source_roots": ["/source"],
        "datasets": {
            "rgb_main": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
            "rgb_wrist": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
            "action": {"width": 7, "dtype": "float32"},
            "state": {"width": 8, "dtype": "float32"},
        },
        "converter_fingerprint": "a" * 64,
        "camera_names": ["main", "wrist"],
        "image_size": [256, 256],
        "source_fps": 20,
        "n_previous": 4,
        "chunk": 9,
        "action_chunk": 36,
        "action_type": "absolute",
        "action_space": "eef",
        "episodes": [
            {
                "key": "libero_goal:000000",
                "shard": shard.name,
                "group": "episodes/libero_goal:000000",
                "caption": _window().caption,
                "domain": "libero_goal",
                "episode_index": 0,
                "length": 12,
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _metadata(records: list[HindsightWindowRecord]) -> HindsightCacheMetadata:
    return HindsightCacheMetadata(
        format_version=1,
        hdf5_manifest_hash="hdf5-hash",
        window_manifest_hash=sha256_json([record.to_dict() for record in records]),
        instruction_parser_hash="parser-hash",
        ta_tok_hash="ta-hash",
        siglip2_hash="siglip-hash",
        dinov3_hash="dino-hash",
        preprocessing_hash="preprocessing-hash",
    )


def test_fake_teacher_build_finalize_audit_smoke(
    fake_builder_inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from safetensors.torch import load_file

    from qwen35_planx.cli.build_hindsight_cache import (
        audit_cache,
        build_shards,
        finalize_cache,
    )
    from qwen35_planx.hindsight_builder import HindsightTargetBuilder
    from qwen35_planx.hindsight_schema import HindsightCache

    hdf5_manifest = _write_fake_hdf5_manifest(tmp_path / "hdf5")
    record = fake_builder_inputs.window
    window_manifest = tmp_path / "windows.jsonl"
    window_manifest.write_text(
        json.dumps(record.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_mtime = (tmp_path / "hdf5" / "episodes.hdf5").stat().st_mtime_ns
    builder = HindsightTargetBuilder.from_components(
        **fake_builder_inputs.components
    )
    shard_root = tmp_path / "shards"
    build_shards(
        hdf5_manifest=hdf5_manifest,
        window_manifest=window_manifest,
        output=shard_root,
        shard_index=0,
        num_shards=1,
        builder=builder,
        metadata=_metadata([record]),
    )
    from qwen35_planx.cli import build_hindsight_cache as cache_cli

    real_fsync = os.fsync
    fsynced: list[str] = []

    def tracking_fsync(descriptor: int) -> None:
        try:
            fsynced.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            pass
        real_fsync(descriptor)

    monkeypatch.setattr(cache_cli.os, "fsync", tracking_fsync)
    cache_dir = tmp_path / "cache"
    manifest = finalize_cache(
        window_manifest=window_manifest,
        shard_root=shard_root,
        output=cache_dir,
    )
    metrics_path = cache_dir / "metrics.json"
    metrics = audit_cache(cache=cache_dir, samples=1, output=metrics_path)

    with HindsightCache.open(
        cache_dir,
        expected_cache_hash=manifest["cache_hash"],
    ) as cache:
        assert len(cache) == 1
        assert cache.codes.flags.writeable is False
    assert metrics_path.is_file()
    assert metrics["samples_audited"] == 1
    assert set(metrics["per_camera_phrase"]["main"]) == {
        "source",
        "target",
        "action",
    }
    assert set(load_file(cache_dir / "phrase_embeddings.safetensors")) == {
        "action",
        "source",
        "target",
    }
    vocabulary = json.loads(
        (cache_dir / "phrase_vocabulary.json").read_text(encoding="utf-8")
    )
    assert vocabulary["split"] == "train"
    assert "forbidden validation drawer" not in json.dumps(vocabulary)
    assert (tmp_path / "hdf5" / "episodes.hdf5").stat().st_mtime_ns == source_mtime
    assert any(
        path.endswith("phrase_embeddings.safetensors") for path in fsynced
    )
    assert any(path.endswith("phrase_vocabulary.json") for path in fsynced)
    assert str(tmp_path) in fsynced

    tensor_path = cache_dir / "phrase_embeddings.safetensors"
    tensor_path.chmod(0o644)
    tensors = load_file(tensor_path)
    tensors["source"][0, 0] = -tensors["source"][0, 0]
    tensor_path.unlink()
    from safetensors.torch import save_file

    save_file(tensors, tensor_path)
    vocabulary_path = cache_dir / "phrase_vocabulary.json"
    vocabulary_path.chmod(0o644)
    vocabulary["embedding_sha256"] = sha256_file(tensor_path)
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")

    with pytest.raises(ValueError, match="phrase embedding table content"):
        audit_cache(cache=cache_dir, samples=1, output=metrics_path)


def test_hindsight_preflight_rejects_missing_local_inputs(tmp_path: Path) -> None:
    from qwen35_planx.cli.preflight import (
        collect_hindsight_cache_preflight_errors,
    )

    errors = collect_hindsight_cache_preflight_errors(
        hdf5_manifest=tmp_path / "missing-hdf5.json",
        window_manifest=tmp_path / "missing-windows.jsonl",
        ta_checkpoint=tmp_path / "missing-ta.pt",
        siglip_model=tmp_path / "missing-siglip",
        dinov3_model=tmp_path / "missing-dino",
        output_dir=tmp_path / "output",
        minimum_free_bytes=0,
    )

    assert any("HDF5 manifest" in error for error in errors)
    assert any("window manifest" in error for error in errors)
    assert any("TA-Tok checkpoint" in error for error in errors)
    assert any("SigLIP2" in error for error in errors)
    assert any("DINOv3" in error for error in errors)


def test_dinov3_preflight_rejects_invalid_weights_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen35_planx.cli import preflight

    model = tmp_path / "dino"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"not-safetensors")
    monkeypatch.setattr(
        preflight.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="dinov3_vit"),
    )

    errors = preflight._validate_local_dinov3_model(model)

    assert any("weights are invalid" in error for error in errors)


def test_hindsight_preflight_rejects_window_mismatch_bounds_and_existing_output(
    tmp_path: Path,
) -> None:
    from qwen35_planx.cli.preflight import (
        collect_hindsight_cache_preflight_errors,
    )

    hdf5_manifest = _write_fake_hdf5_manifest(tmp_path / "hdf5")
    actions = (1, 1, 1, 1) + tuple(range(36))
    frames = (1, 1, 1, 1) + actions[7::4]
    invalid = replace(
        _window(),
        caption="open a mismatched drawer",
        future_indices=(3, 15, 23, 35),
        frame_indices=frames,
        action_indices=actions,
    )
    windows = tmp_path / "windows.jsonl"
    windows.write_text(json.dumps(invalid.to_dict()) + "\n", encoding="utf-8")
    output = tmp_path / "existing-output"
    output.mkdir()

    errors = collect_hindsight_cache_preflight_errors(
        hdf5_manifest=hdf5_manifest,
        window_manifest=windows,
        ta_checkpoint=tmp_path / "missing-ta.pt",
        siglip_model=tmp_path / "missing-siglip",
        dinov3_model=tmp_path / "missing-dino",
        output_dir=output,
        minimum_free_bytes=0,
    )

    assert any("caption" in error for error in errors)
    assert any("trajectory bounds" in error for error in errors)
    assert any("already exists" in error for error in errors)


def test_ola_launcher_runs_preflight_before_workers(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    executable = "#!/usr/bin/env bash\nprintf '%s %s\\n' \"$(basename \"$0\")\" \"$*\" >> \"$CALL_LOG\"\n"
    for name in ("python", "torchrun"):
        path = bin_dir / name
        path.write_text(executable, encoding="utf-8")
        path.chmod(0o755)
    inputs = {}
    for name in (
        "HDF5_MANIFEST",
        "WINDOW_MANIFEST",
        "TA_TOK_CHECKPOINT",
        "SIGLIP2_MODEL_DIR",
        "DINOV3_MODEL_DIR",
    ):
        path = tmp_path / name.lower()
        path.touch()
        inputs[name] = str(path)
    env = {
        **os.environ,
        **inputs,
        "OUTPUT_DIR": str(tmp_path / "cache"),
        "NUM_GPUS": "2",
        "CALL_LOG": str(log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }

    subprocess.run(
        ["bash", "qwen35_planx/scripts/build_hindsight_cache_ola.sh"],
        check=True,
        env=env,
    )

    calls = log.read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith(
        "python -m qwen35_planx.cli.preflight hindsight-cache"
    )
    assert calls[1].startswith("torchrun --standalone --nproc_per_node=2")
    assert " build " in f" {calls[1]} "
    assert " finalize " in f" {calls[2]} "
    assert " audit " in f" {calls[3]} "


def test_shard_collision_is_detected_before_teacher_loading(tmp_path: Path) -> None:
    from qwen35_planx.cli.build_hindsight_cache import (
        _collect_shard_assignment_errors,
        _shard_name,
    )

    first = _window()
    second = replace(
        _window(sample_id="libero_goal:000001:000000000"),
        episode_key="libero_goal:000001",
    )
    output = tmp_path / "shards"
    output.mkdir()
    (output / _shard_name(first.episode_key)).touch()

    errors = _collect_shard_assignment_errors(
        records=[first, second],
        output=output,
        shard_index=0,
        num_shards=2,
    )

    assert any("already exists" in error for error in errors)
