from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPO_ROOT / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

import scripts.preflight_ltx_siglip2 as preflight_module  # noqa: E402
from scripts.preflight_ltx_siglip2 import collect_preflight_errors  # noqa: E402


CONFIG_PATH = (
    GE_ACT_ROOT / "configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml"
)
TRAIN_LAUNCHER = GE_ACT_ROOT / "scripts/train_ltx_siglip2.sh"
SBATCH_LAUNCHER = GE_ACT_ROOT / "scripts/sbatch_train_ltx_siglip2_hpc3.sh"
VLM_CONFIG_PATH = (
    GE_ACT_ROOT / "configs/ltx_model/libero/video_model_libero_vlm_planner_hdf5.yaml"
)
VLM_TRAIN_LAUNCHER = GE_ACT_ROOT / "scripts/train_ltx_vlm_planner.sh"
VLM_SBATCH_LAUNCHER = GE_ACT_ROOT / "scripts/sbatch_train_ltx_vlm_planner_hpc3.sh"
JOINT_CONFIG_PATH = (
    GE_ACT_ROOT
    / "configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml"
)
JOINT_TRAIN_LAUNCHER = GE_ACT_ROOT / "scripts/train_joint_vlm_geact_ola.sh"

OLA_PLANNER_CHECKPOINT = (
    "/data/users/junjie/code/VLM4WAM_dual_camera_k4/outputs/"
    "qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa_predecoded_b8_restart/"
    "step_030000"
)
OLA_LTX_COMPONENTS = "/data/users/junjie/Genie-Envisioner-V1/weights/LTX-Video"
OLA_LTX_CHECKPOINT = "/data/users/junjie/Genie-Envisioner-V1/weights/ltx_step_50000"
OLA_SIGLIP2_TEACHER = "/data/users/junjie/vlm4wam_2b/weights/siglip2-large-patch16-256"
OLA_DA3_CHECKPOINT = "/data/users/junjie/vlm4wam_2b/weights/DA3-LARGE-1.1"
OLA_DA3_CODE_ROOT = "/data/users/junjie/vlm4wam_2b/code/Depth-Anything-3"
OLA_DATA_ROOT = "/data/shared/datasets/libero_fastwam"
OLA_PREDECODED_ROOT = "/data/shared/datasets/libero_fastwam-predecoded-rgb"
OLA_DOMAINS = [
    "libero_10_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
]


def _complete_k4_planner_metadata() -> dict[str, object]:
    return {
        "planner_input_layout": "separate_camera_images",
        "camera_names": ["main", "wrist"],
        "num_camera_views": 2,
        "camera_head_sharing": "shared_head_per_view_image_context",
        "semantic_teacher": "siglip2-large-patch16-256",
        "future_keyframe_offsets": [2, 4, 6, 8],
        "num_keyframes": 4,
        "semantic_output_layout": "batch_view_keyframe_token_feature",
        "sequence_length": 9,
        "grid_size": 16,
        "semantic_dim": 1024,
        "target_tokens": 1024,
        "target_tokens_per_keyframe": 256,
        "planner_token_count": 384,
        "video_target_type": "siglip2",
        "use_current_alignment": False,
        "plan_token_strings": [f"<|sem_plan_{index}|>" for index in range(384)],
        "da3_align_strategy": "wsa_multilayer",
        "da3_teacher_layers": [11, 15, 19, 23],
        "depth_feature_dim": 2048,
        "bidirectional_plan_attn": False,
    }


def _write_complete_planner_checkpoint(path: Path) -> dict[str, object]:
    path.mkdir()
    metadata = _complete_k4_planner_metadata()
    (path / "planner_meta.json").write_text(json.dumps(metadata))
    for filename in ("plan_head.pt", "depth_head.pt", "plan_token_embedding.pt"):
        (path / filename).write_bytes(b"checkpoint")
    for directory in ("qwen3vl_lora_or_model", "processor"):
        component = path / directory
        component.mkdir()
        (component / "config.json").write_text("{}")
    return metadata


def _redirect_formal_recipe_to_tmp_paths(
    config: dict,
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, Path]:
    pretrained = tmp_path / "LTX-Video"
    for component in ("tokenizer", "text_encoder", "vae"):
        (pretrained / component).mkdir(parents=True)
    ltx_checkpoint = tmp_path / "ltx_step_50000"
    ltx_checkpoint.mkdir()
    planner = tmp_path / "planner"
    _write_complete_planner_checkpoint(planner)
    siglip = tmp_path / "siglip2"
    da3 = tmp_path / "da3"
    da3_code = tmp_path / "da3-code"
    cache = tmp_path / "predecoded"
    data_root = tmp_path / "data"
    for path in (siglip, da3, da3_code, cache, data_root):
        path.mkdir()
    stat_file = tmp_path / "stats.json"
    stat_file.write_text("{}")

    config["pretrained_model_name_or_path"] = str(pretrained)
    config["diffusion_model"]["model_path"] = str(ltx_checkpoint)
    config["semantic_plan"]["planner_checkpoint"] = str(planner)
    config["joint_training"]["siglip2_model_dir"] = str(siglip)
    config["joint_training"]["da3_ckpt_dir"] = str(da3)
    config["joint_training"]["da3_code_root"] = str(da3_code)
    config["output_dir"] = str(tmp_path / "output")
    for split in ("train", "val"):
        config["data"][split]["data_roots"] = [str(data_root)] * 4
        config["data"][split]["predecoded_video_root"] = str(cache)
        config["data"][split]["stat_file"] = str(stat_file)

    monkeypatch.setattr(preflight_module, "REQUIRED_MODULES", ())
    monkeypatch.setattr(
        preflight_module.importlib.util,
        "find_spec",
        lambda _module_name: object(),
    )
    redirected = {
        "ltx_components": str(pretrained),
        "ltx_checkpoint": str(ltx_checkpoint),
        "planner_checkpoint": str(planner),
        "siglip2_teacher": str(siglip),
        "da3_checkpoint": str(da3),
        "da3_code_root": str(da3_code),
        "data_root": str(data_root),
        "predecoded_root": str(cache),
        "domains": OLA_DOMAINS,
    }
    monkeypatch.setattr(
        preflight_module, "JOINT_FORMAL_RECIPE", redirected, raising=False
    )
    return planner, cache


def test_siglip2_training_config_matches_the_approved_recipe() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["train_steps"] == 30_000
    assert config["save_steps"] == [20_000, 25_000, 30_000]
    assert config["batch_size"] == 8
    assert config["gradient_accumulation_steps"] == 2
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
    assert config["gradient_checkpointing"] is True
    assert config["lr"] == 2e-5
    assert config["semantic_lr"] == 1e-4
    assert config["lr_warmup_steps"] == 1000
    assert config["max_grad_norm"] == 1.0

    semantic = config["semantic_plan"]
    assert semantic["enabled"] is True
    assert semantic["keyframe_indices"] == [0, 3, 5, 8]
    assert semantic["tokens_per_frame"] == 256
    assert semantic["feature_dim"] == 1024
    assert semantic["dropout"] == 0.15
    assert semantic["validation_mode"] == "gt"

    model = config["diffusion_model"]["config"]
    assert model["semantic_plan_context"] is True
    assert model["semantic_plan_in_dim"] == 1024
    assert model["semantic_plan_num_keyframes"] == 4
    assert model["semantic_plan_cross_attention_blocks"] == list(range(28))
    assert config["data"]["train"]["source_fps"] == 20
    for split in ("train", "val"):
        data_config = config["data"][split]
        assert data_config["require_predecoded"] is True
        assert data_config["predecoded_video_root"].endswith(
            "LIBERO-fastwam-predecoded-rgb"
        )


def test_static_preflight_accepts_the_config() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert collect_preflight_errors(config, world_size=8, check_paths=False) == []


def test_launchers_verify_cache_and_constrain_host_threads() -> None:
    train_launcher = TRAIN_LAUNCHER.read_text()
    sbatch_launcher = SBATCH_LAUNCHER.read_text()

    assert "predecode_lerobot_videos.py" in train_launcher
    assert "--verify-only" in train_launcher
    assert "#SBATCH --gres=gpu:8" in sbatch_launcher
    assert "#SBATCH --cpus-per-task=96" in sbatch_launcher
    assert "#SBATCH --mem=512G" in sbatch_launcher
    assert "SLURM_SUBMIT_DIR" in sbatch_launcher
    assert '-f "$GE_ACT_ROOT/main.py"' in sbatch_launcher
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f"export {variable}=1" in sbatch_launcher


def test_preflight_rejects_non_strict_or_inconsistent_cache_config() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["data"]["train"]["require_predecoded"] = False
    config["data"]["val"]["predecoded_video_root"] = "/tmp/other-cache"

    errors = collect_preflight_errors(config, world_size=8, check_paths=False)

    assert "training must require predecoded RGB caches" in errors
    assert "train and validation must use the same predecoded RGB cache" in errors


def test_vlm_planner_hdf5_config_uses_one_endpoint_and_global_batch_128() -> None:
    config = yaml.safe_load(VLM_CONFIG_PATH.read_text())

    assert config["train_steps"] == 30_000
    assert config["save_steps"] == [20_000, 25_000, 30_000]
    assert config["batch_size"] == 4
    assert config["gradient_accumulation_steps"] == 4
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
    assert config["gradient_checkpointing"] is True
    assert config["lr"] == 2e-5
    assert config["semantic_lr"] == 1e-4
    assert config["lr_warmup_steps"] == 1000

    semantic = config["semantic_plan"]
    assert semantic["source"] == "vlm_planner"
    assert semantic["keyframe_indices"] == [8]
    assert semantic["validation_mode"] == "planner"
    assert semantic["tokens_per_frame"] == 256
    assert semantic["feature_dim"] == 1024
    assert semantic["planner_checkpoint"].startswith("/root/nas/junjie/")

    model = config["diffusion_model"]["config"]
    assert model["semantic_plan_num_keyframes"] == 1
    assert model["semantic_plan_num_views"] == 2
    assert model["semantic_plan_cross_attention_blocks"] == list(range(28))
    assert config["train_data_class"] == "LiberoFastWAMHDF5Dataset"
    assert config["data"]["train"]["manifest_path"].startswith("/root/nas/junjie/")


def test_static_preflight_accepts_vlm_planner_hdf5_config() -> None:
    config = yaml.safe_load(VLM_CONFIG_PATH.read_text())
    assert collect_preflight_errors(config, world_size=8, check_paths=False) == []


def test_vlm_preflight_rejects_wrong_endpoint_or_missing_planner() -> None:
    config = yaml.safe_load(VLM_CONFIG_PATH.read_text())
    config["semantic_plan"]["keyframe_indices"] = [0, 8]
    config["semantic_plan"].pop("planner_checkpoint")

    errors = collect_preflight_errors(config, world_size=8, check_paths=False)

    assert "VLM planner semantic keyframes must be [8]" in errors
    assert "semantic_plan.planner_checkpoint is required" in errors


def test_vlm_launchers_expose_bounded_smoke_override() -> None:
    train_launcher = VLM_TRAIN_LAUNCHER.read_text()
    sbatch_launcher = VLM_SBATCH_LAUNCHER.read_text()

    assert "MAX_TRAIN_STEPS" in train_launcher
    assert "--max_train_steps" in train_launcher
    assert "PYTHONPATH" in train_launcher
    assert "#SBATCH --gres=gpu:8" in sbatch_launcher
    assert "train_ltx_vlm_planner.sh" in sbatch_launcher


def test_joint_vlm_geact_config_matches_approved_recipe() -> None:
    config = yaml.safe_load(JOINT_CONFIG_PATH.read_text())

    assert config["train_steps"] == 30_000
    assert config["save_steps"] == [20_000, 25_000, 30_000]
    assert config["batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 16
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
    assert config["mixed_precision"] == "bf16"
    assert config["gradient_checkpointing"] is True
    assert config["lr"] == 2e-5
    assert config["semantic_lr"] == 1e-4
    assert config["lr_warmup_steps"] == 1_000
    assert config["max_grad_norm"] == 1.0

    semantic = config["semantic_plan"]
    assert semantic["source"] == "vlm_planner"
    assert semantic["planner_checkpoint"] == OLA_PLANNER_CHECKPOINT
    assert semantic["keyframe_indices"] == [2, 4, 6, 8]
    assert semantic["tokens_per_frame"] == 256
    assert semantic["feature_dim"] == 1024
    assert semantic["validation_mode"] == "planner"

    joint = config["joint_training"]
    assert joint["enabled"] is True
    assert joint["planner_loss_weight"] == 0.1
    assert joint["lm_plan_loss_weight"] == 1e-3
    assert joint["qwen_lr"] == 1e-6
    assert joint["planner_head_lr"] == 3e-5
    assert joint["qwen_gradient_checkpointing"] is True
    assert joint["bidirectional_plan_attn"] is False
    assert joint["future_keyframe_offsets"] == [2, 4, 6, 8]
    assert joint["num_camera_views"] == 2
    assert joint["tokens_per_keyframe"] == 256
    assert joint["semantic_feature_dim"] == 1024
    assert joint["da3_align_strategy"] == "wsa_multilayer"
    assert joint["da3_teacher_layers"] == [11, 15, 19, 23]
    assert joint["da3_feature_dim"] == 2048
    assert joint["siglip2_model_dir"] == OLA_SIGLIP2_TEACHER
    assert joint["da3_ckpt_dir"] == OLA_DA3_CHECKPOINT
    assert joint["da3_code_root"] == OLA_DA3_CODE_ROOT

    assert config["pretrained_model_name_or_path"] == OLA_LTX_COMPONENTS
    assert config["diffusion_model"]["model_path"] == OLA_LTX_CHECKPOINT
    model = config["diffusion_model"]["config"]
    assert model["semantic_plan_num_keyframes"] == 4
    assert model["semantic_plan_num_views"] == 2
    assert model["semantic_plan_in_dim"] == 1024
    assert model["semantic_plan_cross_attention_blocks"] == list(range(28))

    assert config["use_deepspeed"] is True
    assert config["deepspeed"]["zero_optimization"]["stage"] == 2
    assert config["deepspeed"]["bf16"]["enabled"] is True


def test_joint_vlm_geact_config_uses_verified_predecoded_ola_data() -> None:
    config = yaml.safe_load(JOINT_CONFIG_PATH.read_text())

    assert config["train_data_class"] == "CustomLeRobotDataset"
    assert config["val_data_class"] == "CustomLeRobotDataset"
    for split in ("train", "val"):
        data = config["data"][split]
        assert data["require_predecoded"] is True
        assert data["data_roots"] == [OLA_DATA_ROOT] * 4
        assert data["domains"] == OLA_DOMAINS
        assert data["predecoded_video_root"] == OLA_PREDECODED_ROOT
        assert data["valid_cam"] == [
            "observation.images.image",
            "observation.images.wrist_image",
        ]
        assert data["chunk"] == 9
        assert data["n_previous"] == 4


def test_static_preflight_accepts_joint_vlm_geact_config() -> None:
    config = yaml.safe_load(JOINT_CONFIG_PATH.read_text())
    assert collect_preflight_errors(config, world_size=8, check_paths=False) == []


def test_joint_preflight_rejects_geometry_lr_batch_and_checkpointing_drift() -> None:
    config = yaml.safe_load(JOINT_CONFIG_PATH.read_text())
    config["semantic_plan"]["keyframe_indices"] = [8]
    config["diffusion_model"]["config"]["semantic_plan_num_views"] = 1
    config["lr"] = 3e-5
    config["semantic_lr"] = 2e-4
    config["batch_size"] = 2
    config["gradient_accumulation_steps"] = 8
    config["gradient_checkpointing"] = False
    joint = config["joint_training"]
    joint["qwen_lr"] = 2e-6
    joint["planner_head_lr"] = 4e-5
    joint["qwen_gradient_checkpointing"] = False
    joint["future_keyframe_offsets"] = [1, 3, 5, 7]
    joint["num_camera_views"] = 1
    joint["tokens_per_keyframe"] = 128
    joint["semantic_feature_dim"] = 512
    joint["da3_align_strategy"] = "last_layer"
    joint["da3_teacher_layers"] = [23]
    joint["da3_feature_dim"] = 1024

    errors = collect_preflight_errors(config, world_size=8, check_paths=False)

    for expected in (
        "joint VLM planner keyframe offsets must be [2, 4, 6, 8]",
        "joint training requires per-GPU batch 1 and accumulation 16",
        "joint LTX base lr must be 2e-5",
        "joint LTX semantic lr must be 1e-4",
        "joint Qwen lr must be 1e-6",
        "joint planner head lr must be 3e-5",
        "joint training requires LTX gradient checkpointing",
        "joint training requires Qwen gradient checkpointing",
        "joint training requires two camera views",
        "joint semantic plan must use 256 tokens per keyframe",
        "joint semantic feature width must be 1024",
        "joint DA3 teacher must use four-layer WSA",
        "joint DA3 teacher layers must be [11, 15, 19, 23]",
        "joint DA3 feature width must be 2048",
    ):
        assert expected in errors


def test_joint_preflight_rejects_unsafe_lm_objective_and_missing_owners() -> None:
    config = yaml.safe_load(JOINT_CONFIG_PATH.read_text())
    joint = config["joint_training"]
    joint["lm_plan_loss_weight"] = 0.0
    joint["bidirectional_plan_attn"] = True
    joint["siglip2_model_dir"] = ""
    joint["da3_ckpt_dir"] = ""
    joint["da3_code_root"] = ""

    errors = collect_preflight_errors(config, world_size=8, check_paths=False)

    assert "joint lm_plan_loss_weight must be 1e-3" in errors
    assert "joint planner checkpoint must use causal attention" in errors
    assert "joint SigLIP2 teacher path is required" in errors
    assert "joint DA3 teacher checkpoint is required" in errors
    assert "joint DA3 code root is required" in errors


def test_joint_runtime_preflight_checks_checkpoint_metadata_and_all_paths(
    tmp_path: Path, monkeypatch
) -> None:
    config = copy.deepcopy(yaml.safe_load(JOINT_CONFIG_PATH.read_text()))
    planner, _cache = _redirect_formal_recipe_to_tmp_paths(
        config, tmp_path, monkeypatch
    )
    assert (
        collect_preflight_errors(
            config,
            world_size=8,
            check_paths=True,
            minimum_free_gb=0.0,
        )
        == []
    )

    planner_meta = json.loads((planner / "planner_meta.json").read_text())
    planner_meta["bidirectional_plan_attn"] = True
    planner_meta["da3_teacher_layers"] = [23]
    (planner / "planner_meta.json").write_text(json.dumps(planner_meta))
    errors = collect_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert "planner metadata must set bidirectional_plan_attn=false" in errors
    assert "planner metadata DA3 teacher layers must be [11, 15, 19, 23]" in errors


def test_joint_static_preflight_rejects_any_formal_recipe_path_drift() -> None:
    config = copy.deepcopy(yaml.safe_load(JOINT_CONFIG_PATH.read_text()))
    config["pretrained_model_name_or_path"] = "/tmp/other-components"
    config["diffusion_model"]["model_path"] = "/tmp/other-ltx"
    config["semantic_plan"]["planner_checkpoint"] = "/tmp/other-planner"
    joint = config["joint_training"]
    joint["lm_plan_loss_weight"] = 2e-3
    joint["siglip2_model_dir"] = "/tmp/other-siglip"
    joint["da3_ckpt_dir"] = "/tmp/other-da3"
    joint["da3_code_root"] = "/tmp/other-da3-code"
    config["data"]["train"]["data_roots"] = ["/tmp/other-data"] * 4
    config["data"]["val"]["predecoded_video_root"] = "/tmp/other-cache"
    config["data"]["train"]["domains"] = list(reversed(OLA_DOMAINS))

    errors = collect_preflight_errors(config, world_size=8, check_paths=False)

    for expected in (
        "joint lm_plan_loss_weight must be 1e-3",
        "joint formal LTX components path does not match the approved recipe",
        "joint formal LTX checkpoint path does not match the approved recipe",
        "joint formal planner checkpoint path does not match the approved recipe",
        "joint formal SigLIP2 teacher path does not match the approved recipe",
        "joint formal DA3 checkpoint path does not match the approved recipe",
        "joint formal DA3 code root does not match the approved recipe",
        "joint formal training data roots do not match the approved recipe",
        "joint formal predecoded root does not match the approved recipe",
        "joint formal domains do not match the approved recipe",
    ):
        assert expected in errors


def test_joint_runtime_preflight_rejects_incomplete_loader_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    config = copy.deepcopy(yaml.safe_load(JOINT_CONFIG_PATH.read_text()))
    planner, _cache = _redirect_formal_recipe_to_tmp_paths(
        config, tmp_path, monkeypatch
    )

    (planner / "depth_head.pt").unlink()
    errors = collect_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert "planner checkpoint is missing required file: depth_head.pt" in errors

    (planner / "depth_head.pt").write_bytes(b"checkpoint")
    (planner / "processor" / "config.json").unlink()
    errors = collect_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert "planner checkpoint directory is missing or empty: processor/" in errors


def test_joint_runtime_preflight_reuses_exact_provider_k4_metadata_validator(
    tmp_path: Path, monkeypatch
) -> None:
    config = copy.deepcopy(yaml.safe_load(JOINT_CONFIG_PATH.read_text()))
    planner, _cache = _redirect_formal_recipe_to_tmp_paths(
        config, tmp_path, monkeypatch
    )
    metadata_path = planner / "planner_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("planner_input_layout")
    metadata_path.write_text(json.dumps(metadata))

    errors = collect_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert any(
        "incompatible dual-camera planner metadata field planner_input_layout" in error
        for error in errors
    )


def test_main_smoke_max_steps_is_a_constructor_override(
    monkeypatch, tmp_path: Path
) -> None:
    main_path = GE_ACT_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("ge_act_main_task5", main_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, config_file, *, config_overrides=None):
            captured["config_file"] = config_file
            captured["config_overrides"] = config_overrides
            self.args = SimpleNamespace(train_steps=99)

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    monkeypatch.setattr(module, "import_custom_class", lambda *_args: FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ge_act/main.py",
            "--config_file",
            str(tmp_path / "config.yaml"),
            "--max_train_steps",
            "1",
            "--batch_size_override",
            "1",
            "--gradient_accumulation_steps_override",
            "1",
            "--disable_deepspeed",
        ],
    )

    module.main()

    assert captured["config_overrides"] == {
        "train_steps": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "use_deepspeed": False,
    }
    assert "runner.args.train_steps =" not in main_path.read_text()


def test_joint_ola_launcher_has_formal_and_bounded_smoke_modes() -> None:
    launcher = JOINT_TRAIN_LAUNCHER.read_text()

    assert "RUN_KIND=${RUN_KIND:-formal}" in launcher
    assert '"$RUN_KIND" == "smoke8"' in launcher
    assert "NUM_GPUS=8" in launcher
    assert '--nproc_per_node="$NUM_GPUS"' in launcher
    assert "predecode_lerobot_videos.py" in launcher
    assert "--verify-only" in launcher
    assert "preflight_ltx_siglip2.py" in launcher
    assert "--require-joint-formal" in launcher
    for argument in (
        "--max_train_steps",
        "--batch_size_override",
        "--gradient_accumulation_steps_override",
        "--disable_deepspeed",
        "--enable_8bit_optimizer",
    ):
        assert argument in launcher
    assert "--max_train_steps 10" in launcher
    assert "RUN_KIND must be formal, smoke, or smoke8" in launcher
    for variable in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f"export {variable}=" in launcher


def test_runtime_preflight_requires_deepspeed_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    config = copy.deepcopy(yaml.safe_load(JOINT_CONFIG_PATH.read_text()))
    _redirect_formal_recipe_to_tmp_paths(config, tmp_path, monkeypatch)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(module_name: str):
        if module_name == "deepspeed":
            return None
        return real_find_spec(module_name)

    monkeypatch.setattr(preflight_module.importlib.util, "find_spec", fake_find_spec)
    errors = collect_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )

    assert "missing Python module: deepspeed" in errors


def test_required_joint_formal_mode_rejects_disabled_or_missing_joint_config() -> None:
    disabled = copy.deepcopy(yaml.safe_load(JOINT_CONFIG_PATH.read_text()))
    disabled["joint_training"]["enabled"] = False
    missing = copy.deepcopy(disabled)
    missing.pop("joint_training")

    for config in (disabled, missing):
        errors = collect_preflight_errors(
            config,
            world_size=8,
            check_paths=False,
            require_joint_formal=True,
        )
        assert "formal joint preflight requires joint_training.enabled=true" in errors


def test_default_preflight_mode_preserves_legacy_gt_and_frozen_vlm_configs() -> None:
    for path in (CONFIG_PATH, VLM_CONFIG_PATH):
        config = yaml.safe_load(path.read_text())
        assert (
            collect_preflight_errors(
                config,
                world_size=8,
                check_paths=False,
            )
            == []
        )
