import importlib.util
import json
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COSMOS_SRC = REPO_ROOT / "third_party/cosmos-predict2.5/cosmos_predict2/_src/predict2"


def _install_dataset_stubs(monkeypatch):
    decord = types.ModuleType("decord")
    decord.cpu = lambda *_args, **_kwargs: None
    decord.VideoReader = object
    monkeypatch.setitem(sys.modules, "decord", decord)

    parallel_state = types.SimpleNamespace(
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_world_size=lambda: 1,
    )
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_core.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    lazy_config = types.ModuleType("cosmos_predict2._src.imaginaire.lazy_config")
    lazy_config.LazyCall = lambda target: target
    log_module = types.SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)
    utils_module = types.ModuleType("cosmos_predict2._src.imaginaire.utils")
    utils_module.log = log_module

    dataset_utils = types.ModuleType("cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils")
    dataset_utils.ToTensorVideo = lambda: (lambda x: x)
    dataset_utils.ResizePreprocess = lambda _size: (lambda x: x)

    module_names = [
        "cosmos_predict2",
        "cosmos_predict2._src",
        "cosmos_predict2._src.imaginaire",
        "cosmos_predict2._src.predict2",
        "cosmos_predict2._src.predict2.datasets",
        "cosmos_predict2._src.predict2.datasets.local_datasets",
        "cosmos_predict2._src.predict2.networks",
    ]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    semantic_module = _load_file_module(
        COSMOS_SRC / "networks/semantic_plan_conditioning.py",
        "cosmos_predict2._src.predict2.networks.semantic_plan_conditioning",
    )
    monkeypatch.setitem(
        sys.modules,
        "cosmos_predict2._src.predict2.networks.semantic_plan_conditioning",
        semantic_module,
    )
    monkeypatch.setitem(sys.modules, "cosmos_predict2._src.imaginaire.lazy_config", lazy_config)
    monkeypatch.setitem(sys.modules, "cosmos_predict2._src.imaginaire.utils", utils_module)
    monkeypatch.setitem(
        sys.modules,
        "cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils",
        dataset_utils,
    )


def _load_file_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_sample_uses_matching_video_window_and_semantic_plan(tmp_path, monkeypatch):
    _install_dataset_stubs(monkeypatch)
    module = _load_file_module(
        COSMOS_SRC / "datasets/local_datasets/dataset_video.py",
        "dataset_video_under_test",
    )
    dataset_root = tmp_path / "dataset"
    semantic_dir = dataset_root / "semantic"
    (dataset_root / "videos").mkdir(parents=True)
    (dataset_root / "metas").mkdir()
    semantic_dir.mkdir()

    (dataset_root / "videos" / "episode_a.mp4").write_bytes(b"not-used")
    (dataset_root / "metas" / "episode_a.txt").write_text("pick up the yellow carrot")
    plan = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    torch.save({"semantic_plan": plan}, semantic_dir / "episode_a__r00__w0001__fs02__s000010_e000018.pt")
    manifest = {
        "sample_id": "episode_a__r00__w0001__fs02__s000010_e000018",
        "stem": "episode_a",
        "video_frame_indices": [10, 12, 14, 16],
        "frame_stride": 2,
        "future_frame_indices": [12, 16],
    }
    (semantic_dir / "manifest_0.jsonl").write_text(json.dumps(manifest) + "\n")

    captured = {}

    def fake_get_frames_by_ids(self, video_path, frame_ids):
        captured["video_path"] = Path(video_path).name
        captured["frame_ids"] = list(frame_ids)
        return torch.zeros(len(frame_ids), 3, 4, 4, dtype=torch.uint8), 24.0

    monkeypatch.setattr(module.VideoDataset, "_get_frames_by_ids", fake_get_frames_by_ids)

    dataset = module.VideoDataset(
        dataset_dir=str(dataset_root),
        num_frames=4,
        video_size=(4, 4),
        caption_format="text",
        semantic_plan_dir=str(semantic_dir),
        semantic_plan_dim=2,
        semantic_plan_max_tokens=3,
        semantic_plan_manifest="manifest_0.jsonl",
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert captured == {"video_path": "episode_a.mp4", "frame_ids": [10, 12, 14, 16]}
    assert sample["fps"] == 12.0
    assert torch.equal(sample["semantic_plan"], plan)
    assert sample["semantic_plan_meta"]["sample_id"] == manifest["sample_id"]
    assert sample["semantic_plan_meta"]["video_frame_indices"] == manifest["video_frame_indices"]


def test_manifest_glob_must_resolve_when_configured(tmp_path, monkeypatch):
    _install_dataset_stubs(monkeypatch)
    module = _load_file_module(
        COSMOS_SRC / "datasets/local_datasets/dataset_video.py",
        "dataset_video_missing_manifest_under_test",
    )
    dataset_root = tmp_path / "dataset"
    semantic_dir = dataset_root / "semantic"
    (dataset_root / "videos").mkdir(parents=True)
    (dataset_root / "metas").mkdir()
    semantic_dir.mkdir()

    (dataset_root / "videos" / "episode_a.mp4").write_bytes(b"not-used")
    (dataset_root / "metas" / "episode_a.txt").write_text("pick up the yellow carrot")

    try:
        module.VideoDataset(
            dataset_dir=str(dataset_root),
            num_frames=4,
            video_size=(4, 4),
            caption_format="text",
            semantic_plan_dir=str(semantic_dir),
            semantic_plan_dim=2,
            semantic_plan_max_tokens=3,
            semantic_plan_manifest="manifest*.jsonl",
        )
    except FileNotFoundError as exc:
        assert "semantic plan manifest" in str(exc).lower()
    else:
        raise AssertionError("Expected missing semantic-plan manifest to fail fast")


def test_manifest_mode_reraises_sample_errors(tmp_path, monkeypatch):
    _install_dataset_stubs(monkeypatch)
    module = _load_file_module(
        COSMOS_SRC / "datasets/local_datasets/dataset_video.py",
        "dataset_video_reraise_under_test",
    )
    dataset_root = tmp_path / "dataset"
    semantic_dir = dataset_root / "semantic"
    (dataset_root / "videos").mkdir(parents=True)
    (dataset_root / "metas").mkdir()
    semantic_dir.mkdir()

    (dataset_root / "videos" / "episode_a.mp4").write_bytes(b"not-used")
    (dataset_root / "metas" / "episode_a.txt").write_text("pick up the yellow carrot")
    manifest = {
        "sample_id": "missing_plan_sample",
        "stem": "episode_a",
        "video_frame_indices": [0, 1, 2, 3],
    }
    (semantic_dir / "manifest_0.jsonl").write_text(json.dumps(manifest) + "\n")

    def fake_get_frames_by_ids(self, video_path, frame_ids):
        return torch.zeros(len(frame_ids), 3, 4, 4, dtype=torch.uint8), 24.0

    monkeypatch.setattr(module.VideoDataset, "_get_frames_by_ids", fake_get_frames_by_ids)
    dataset = module.VideoDataset(
        dataset_dir=str(dataset_root),
        num_frames=4,
        video_size=(4, 4),
        caption_format="text",
        semantic_plan_dir=str(semantic_dir),
        semantic_plan_dim=2,
        semantic_plan_max_tokens=3,
        semantic_plan_manifest="manifest_0.jsonl",
    )

    try:
        dataset[0]
    except FileNotFoundError as exc:
        assert "missing_plan_sample" in str(exc)
    else:
        raise AssertionError("Expected manifest sample error to be re-raised")


def test_semantic_plan_coords_are_rescaled_to_latent_grid():
    module = _load_file_module(
        COSMOS_SRC / "networks/semantic_plan_conditioning.py",
        "semantic_plan_conditioning_under_test",
    )

    coords = module.build_semantic_plan_rope_positions(
        batch_size=1,
        num_tokens=2 * 3 * 3,
        num_keyframes=2,
        spatial_grid_size=3,
        latent_t=5,
        latent_h=7,
        latent_w=9,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coords.shape == (1, 18, 3)
    assert torch.allclose(coords[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(coords[0, 8], torch.tensor([0.0, 6.0, 8.0]))
    assert torch.allclose(coords[0, 9], torch.tensor([4.0, 0.0, 0.0]))
    assert torch.allclose(coords[0, -1], torch.tensor([4.0, 6.0, 8.0]))


def test_adapter_does_not_encode_coordinates_in_token_content():
    module = _load_file_module(
        COSMOS_SRC / "networks/semantic_plan_conditioning.py",
        "semantic_plan_conditioning_no_content_coords_under_test",
    )
    adapter = module.SemanticPlanContextAdapter(
        in_dim=2,
        hidden_dim=4,
        out_dim=4,
        num_keyframes=1,
        spatial_grid_size=2,
    )
    for parameter in adapter.projection.parameters():
        parameter.data.zero_()
    adapter.type_token.data.zero_()

    semantic_plan = torch.ones(1, 4, 2)
    tokens, valid = adapter(semantic_plan)

    assert valid.all()
    assert torch.allclose(tokens[:, :1], tokens)


def test_trims_all_invalid_semantic_token_columns():
    module = _load_file_module(
        COSMOS_SRC / "networks/semantic_plan_conditioning.py",
        "semantic_plan_conditioning_trim_under_test",
    )
    tokens = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    valid = torch.tensor(
        [
            [True, False, True, False],
            [False, False, True, False],
        ]
    )

    trimmed_tokens, trimmed_valid, keep = module.trim_invalid_semantic_tokens(tokens, valid)

    assert torch.equal(keep, torch.tensor([True, False, True, False]))
    assert trimmed_tokens.shape == (2, 2, 3)
    assert torch.equal(trimmed_valid, torch.tensor([[True, True], [False, True]]))


def test_load_semantic_plan_tensor_from_file(tmp_path):
    module = _load_file_module(
        COSMOS_SRC / "networks/semantic_plan_conditioning.py",
        "semantic_plan_conditioning_load_under_test",
    )
    path = tmp_path / "plan.pt"
    torch.save({"semantic_plan": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)}, path)

    loaded = module.load_semantic_plan_tensor(path, semantic_plan_dim=2, max_tokens=7)

    assert loaded.shape == (7, 2)
    assert torch.equal(loaded[:6], torch.arange(12, dtype=torch.float32).reshape(6, 2))
    assert torch.equal(loaded[6], torch.zeros(2))
