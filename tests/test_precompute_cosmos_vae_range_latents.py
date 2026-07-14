import importlib.util
import json
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "cosmos-predict2.5/scripts/precompute_cosmos_vae_range_latents.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("precompute_cosmos_vae_range_latents_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stride_specific_cache_records_keep_sequence_coordinates(tmp_path):
    module = _load_module()

    records = module.make_cache_records_from_frame_ranges(
        {"episode_a": [(10, 120)]},
        output_dir=tmp_path,
        frame_strides=[1, 2],
        min_sequence_length=8,
        cache_num_frames=0,
        cache_step_frames=0,
    )

    stride2 = [record for record in records if record["frame_stride"] == 2][0]
    assert stride2["cache_id"] == "episode_a__r00__fs02__c0000__q000000_000055"
    assert stride2["stem"] == "episode_a"
    assert stride2["range_start"] == 10
    assert stride2["range_end"] == 120
    assert stride2["sequence_start_index"] == 0
    assert stride2["sequence_end_index"] == 55
    assert stride2["sequence_length"] == 55
    assert stride2["first_pixel_frame"] == 10
    assert stride2["last_pixel_frame"] == 118
    assert stride2["latent_num_frames"] == 14
    assert stride2["latent_path"] == "range_latents/episode_a__r00__fs02__c0000__q000000_000055.pt"


def test_long_ranges_are_split_with_overlap_to_keep_window_coverage(tmp_path):
    module = _load_module()

    records = module.make_cache_records_from_frame_ranges(
        {"episode_a": [(0, 30)]},
        output_dir=tmp_path,
        frame_strides=[1],
        min_sequence_length=8,
        cache_num_frames=12,
        cache_step_frames=5,
    )

    assert [(r["sequence_start_index"], r["sequence_end_index"]) for r in records] == [
        (0, 12),
        (5, 17),
        (10, 22),
        (15, 27),
        (18, 30),
    ]


def test_cache_sampling_limits_records_per_range_and_stride(tmp_path):
    module = _load_module()

    first = module.make_cache_records_from_frame_ranges(
        {"episode_a": [(0, 40)]},
        output_dir=tmp_path,
        frame_strides=[1],
        min_sequence_length=8,
        cache_num_frames=8,
        cache_step_frames=1,
        caches_per_range=3,
        cache_seed=2026,
    )
    second = module.make_cache_records_from_frame_ranges(
        {"episode_a": [(0, 40)]},
        output_dir=tmp_path,
        frame_strides=[1],
        min_sequence_length=8,
        cache_num_frames=8,
        cache_step_frames=1,
        caches_per_range=3,
        cache_seed=2026,
    )

    assert first == second
    assert len(first) == 3
    assert all(record["sequence_length"] == 8 for record in first)


def test_window_records_are_aligned_to_wan_vae_temporal_stride(tmp_path):
    module = _load_module()
    cache_records = [
        {
            "cache_id": "episode_a__r00__fs02__c0000__q000000_000020",
            "stem": "episode_a",
            "range_index": 0,
            "range_start": 10,
            "range_end": 100,
            "frame_stride": 2,
            "sequence_start_index": 0,
            "sequence_end_index": 20,
            "sequence_length": 20,
            "latent_path": "range_latents/episode_a__r00__fs02__c0000__q000000_000020.pt",
        }
    ]

    windows = module.make_window_records_from_cache_records(
        cache_records,
        num_frames=8,
        windows_per_cache=0,
        seed=123,
        temporal_alignment=4,
    )

    assert [window["sequence_start_index"] for window in windows] == [0, 4, 8, 12]
    assert [window["latent_offset"] for window in windows] == [0, 1, 2, 3]
    assert windows[1]["video_frame_indices"] == [18, 20, 22, 24, 26, 28, 30, 32]
    assert windows[1]["latent_num_frames"] == 2
    assert windows[1]["latent_path"] == "range_latents/episode_a__r00__fs02__c0000__q000000_000020.pt"


def test_window_sampling_is_deterministic_per_cache(tmp_path):
    module = _load_module()
    cache_records = [
        {
            "cache_id": "episode_a__r00__fs01__c0000__q000000_000040",
            "stem": "episode_a",
            "range_index": 0,
            "range_start": 0,
            "range_end": 40,
            "frame_stride": 1,
            "sequence_start_index": 0,
            "sequence_end_index": 40,
            "sequence_length": 40,
            "latent_path": "range_latents/episode_a__r00__fs01__c0000__q000000_000040.pt",
        }
    ]

    first = module.make_window_records_from_cache_records(
        cache_records,
        num_frames=8,
        windows_per_cache=3,
        seed=2026,
        temporal_alignment=4,
    )
    second = module.make_window_records_from_cache_records(
        cache_records,
        num_frames=8,
        windows_per_cache=3,
        seed=2026,
        temporal_alignment=4,
    )

    assert first == second
    assert len(first) == 3
    assert all(window["sequence_start_index"] % 4 == 0 for window in first)


def test_crop_latent_window_uses_offset_and_length():
    module = _load_module()
    latent = torch.arange(1 * 2 * 6 * 2 * 2).reshape(1, 2, 6, 2, 2)
    window = {"latent_offset": 2, "latent_num_frames": 3}

    cropped = module.crop_latent_window(latent, window)

    assert torch.equal(cropped, latent[:, :, 2:5])


def test_write_manifest_jsonl_roundtrip(tmp_path):
    module = _load_module()
    path = tmp_path / "manifest.jsonl"
    rows = [{"cache_id": "a"}, {"cache_id": "b"}]

    module.write_jsonl(path, rows)

    assert [json.loads(line) for line in path.read_text().splitlines()] == rows
