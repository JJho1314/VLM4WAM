import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "qwen3_vl_semantic_planner/downsample_video_dataset_to_fps.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("downsample_video_dataset_to_fps_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_ffmpeg_command_forces_target_fps_and_mp4_compatible_video(tmp_path):
    module = _load_module()
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"

    cmd = module.build_ffmpeg_command(
        src,
        dst,
        target_fps=10.0,
        crf=18,
        preset="medium",
        overwrite=True,
    )

    assert cmd[:4] == ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert "-y" in cmd
    assert cmd[cmd.index("-i") + 1] == str(src)
    assert cmd[cmd.index("-vf") + 1] == "fps=10"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[-1] == str(dst)


def test_build_ffmpeg_command_can_resize_video_after_fps_filter(tmp_path):
    module = _load_module()
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"

    cmd = module.build_ffmpeg_command(
        src,
        dst,
        target_fps=10.0,
        target_height=320,
        target_width=576,
        crf=18,
        preset="",
        overwrite=True,
    )

    assert cmd[cmd.index("-vf") + 1] == "fps=10,scale=576:320:flags=lanczos"


def test_build_ffmpeg_command_can_omit_encoder_preset_for_compatibility(tmp_path):
    module = _load_module()
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"

    cmd = module.build_ffmpeg_command(
        src,
        dst,
        target_fps=10.0,
        target_height=0,
        target_width=0,
        crf=18,
        preset="",
        overwrite=False,
    )

    assert "-preset" not in cmd
    assert "-n" in cmd
    assert cmd[cmd.index("-crf") + 1] == "18"


def test_default_output_root_includes_resolution_when_requested(tmp_path):
    module = _load_module()

    output = module.default_output_root(tmp_path / "dataset_a", 10.0, target_height=320, target_width=576)

    assert output.name == "dataset_a_10hz_320x576"


def test_rewrite_frame_ranges_maps_exclusive_ranges_from_15hz_to_10hz(tmp_path):
    module = _load_module()
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    dst_root.mkdir()
    (src_root / "frame_ranges.json").write_text(
        json.dumps(
            {
                "episode_a": [[0, 150], [3, 18]],
                "episode_b": {"start": 15, "end": 45},
            }
        )
    )

    rewritten = module.rewrite_frame_ranges(
        src_root,
        dst_root,
        target_fps=10.0,
        source_fps_by_stem={"episode_a": 15.0, "episode_b": 15.0},
    )

    assert rewritten == {
        "episode_a": [[0, 100], [2, 12]],
        "episode_b": [[10, 30]],
    }
    assert json.loads((dst_root / "frame_ranges.json").read_text()) == rewritten
    audit = (dst_root / "frame_ranges_10hz_audit.tsv").read_text().splitlines()
    assert audit[0].split("\t") == [
        "stem",
        "source_fps",
        "target_fps",
        "old_start",
        "old_end",
        "new_start",
        "new_end",
    ]
    assert "episode_a\t15.000000\t10.000000\t3\t18\t2\t12" in audit


def test_rewrite_frame_ranges_uses_default_source_fps_for_unprocessed_stems(tmp_path):
    module = _load_module()
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    dst_root.mkdir()
    (src_root / "frame_ranges.json").write_text(json.dumps({"episode_b": [[15, 45]]}))

    rewritten = module.rewrite_frame_ranges(
        src_root,
        dst_root,
        target_fps=10.0,
        source_fps_by_stem={},
        default_source_fps=15.0,
    )

    assert rewritten == {"episode_b": [[10, 30]]}


def test_copy_dataset_sidecars_skips_derived_feature_directories(tmp_path):
    module = _load_module()
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    for name in [
        "metas",
        "captions",
        "siglip2_semantic_plan_k16_g9_cosmos_t93_s123_step24_full",
        "qwen3vl2b_semantic_plan_k6_g9",
        "target_features",
    ]:
        (src_root / name).mkdir(parents=True)
        (src_root / name / "episode_a.txt").write_text(name)

    copied = module.copy_dataset_sidecars(src_root, dst_root)

    assert copied == ["captions", "metas"]
    assert (dst_root / "metas" / "episode_a.txt").read_text() == "metas"
    assert (dst_root / "captions" / "episode_a.txt").read_text() == "captions"
    assert not (dst_root / "siglip2_semantic_plan_k16_g9_cosmos_t93_s123_step24_full").exists()
    assert not (dst_root / "qwen3vl2b_semantic_plan_k6_g9").exists()
    assert not (dst_root / "target_features").exists()


def test_discover_videos_honors_max_videos_without_requiring_full_dataset(tmp_path):
    module = _load_module()
    video_root = tmp_path / "dataset" / "videos"
    video_root.mkdir(parents=True)
    for idx in range(5):
        (video_root / f"episode_{idx}.mp4").write_bytes(b"")

    videos = module.discover_videos(tmp_path / "dataset", "videos", max_videos=2)

    assert len(videos) == 2
    assert all(path.suffix == ".mp4" for path in videos)


def test_discover_videos_defaults_to_flat_directory_scan(tmp_path):
    module = _load_module()
    video_root = tmp_path / "dataset" / "videos"
    nested = video_root / "nested"
    nested.mkdir(parents=True)
    (video_root / "episode_root.mp4").write_bytes(b"")
    (nested / "episode_nested.mp4").write_bytes(b"")

    videos = module.discover_videos(tmp_path / "dataset", "videos")

    assert [path.name for path in videos] == ["episode_root.mp4"]
