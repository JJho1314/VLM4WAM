import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "qwen3_vl_semantic_planner/build_siglip2_semantic_plan_labels.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_siglip2_semantic_plan_labels_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_window_manifest_preserves_explicit_video_frame_indices(tmp_path):
    module = _load_module()
    manifest = tmp_path / "window_manifest.jsonl"
    row = {
        "sample_id": "episode_a__r00__fs02__c0000__w0000__s000010_e000025",
        "stem": "episode_a",
        "range_index": 0,
        "window_index": 0,
        "frame_stride": 2,
        "sequence_length": 8,
        "video_frame_indices": [10, 12, 14, 16, 18, 20, 22, 24],
    }
    manifest.write_text(json.dumps(row) + "\n")

    items = module.load_window_manifest(manifest)

    assert len(items) == 1
    assert items[0].sample_id == row["sample_id"]
    assert items[0].stem == "episode_a"
    assert items[0].start == 10
    assert items[0].end == 25
    assert items[0].frame_stride == 2
    assert items[0].sequence_length == 8
    assert items[0].video_frame_indices == tuple(row["video_frame_indices"])


def test_sample_future_from_explicit_clip_matches_cosmos_keyframe_policy():
    module = _load_module()
    clip = [10, 12, 14, 16, 18, 20, 22, 24]

    first, future, full_clip = module.sample_future_from_explicit_clip(clip, video_len=100, num_future=3)

    assert first == 10
    assert future == [12, 18, 24]
    assert full_clip == clip
