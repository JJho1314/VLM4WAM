import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "third_party/cosmos-predict2.5/scripts/precompute_cosmos_vae_latents_pilot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("precompute_cosmos_vae_latents_pilot_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_uint8_video_for_vae_matches_training_range():
    module = _load_module()
    video = torch.tensor([[[[[0, 127, 255]]]]], dtype=torch.uint8)

    normalized = module.normalize_uint8_video_for_vae(video)

    assert normalized.dtype == torch.float32
    assert torch.allclose(normalized.flatten(), torch.tensor([-1.0, -0.0039215689, 1.0]))


def test_latent_path_uses_sample_id_under_latents_dir(tmp_path):
    module = _load_module()

    path = module.latent_path_for_sample(tmp_path, "episode_a__r00__w0001__fs01__s000000_e000093")

    assert path == tmp_path / "latents" / "episode_a__r00__w0001__fs01__s000000_e000093.pt"


def test_make_manifest_record_uses_relative_latent_path(tmp_path):
    module = _load_module()
    latent_path = tmp_path / "latents" / "sample_a.pt"
    latent = torch.zeros(1, 16, 24, 40, 72, dtype=torch.bfloat16)
    source_record = {
        "sample_id": "sample_a",
        "stem": "episode_a",
        "video_frame_indices": [0, 1, 2],
        "frame_stride": 1,
    }

    record = module.make_manifest_record(
        output_dir=tmp_path,
        latent_path=latent_path,
        latent=latent,
        source_record=source_record,
        fps=10.0,
        video_size=(320, 576),
        tokenizer_name="wan2pt1_tokenizer",
    )

    assert record["sample_id"] == "sample_a"
    assert record["stem"] == "episode_a"
    assert record["latent_path"] == "latents/sample_a.pt"
    assert record["latent_shape"] == [1, 16, 24, 40, 72]
    assert record["latent_dtype"] == "torch.bfloat16"
    assert record["video_size"] == [320, 576]
    assert record["fps"] == 10.0
    assert record["tokenizer"] == "wan2pt1_tokenizer"
    assert record["video_frame_indices"] == [0, 1, 2]


def test_select_records_applies_start_index_and_max_samples():
    module = _load_module()
    records = [{"sample_id": f"s{i}"} for i in range(5)]

    assert module.select_records(records, start_index=1, max_samples=2) == [{"sample_id": "s1"}, {"sample_id": "s2"}]
    assert module.select_records(records, start_index=3, max_samples=0) == [{"sample_id": "s3"}, {"sample_id": "s4"}]


def test_write_jsonl_roundtrips_manifest_records(tmp_path):
    module = _load_module()
    path = tmp_path / "manifest.jsonl"
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]

    module.write_jsonl(path, rows)

    assert [json.loads(line) for line in path.read_text().splitlines()] == rows


def test_install_megatron_parallel_state_stub_when_missing(monkeypatch):
    module = _load_module()
    for name in ["megatron", "megatron.core"]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    module.install_megatron_parallel_state_stub()

    from megatron.core import parallel_state

    assert parallel_state.is_initialized() is False
    assert parallel_state.get_data_parallel_rank() == 0
    assert parallel_state.get_data_parallel_world_size() == 1
