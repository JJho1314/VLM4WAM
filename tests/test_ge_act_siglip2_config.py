from __future__ import annotations

import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPO_ROOT / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from scripts.preflight_ltx_siglip2 import collect_preflight_errors


CONFIG_PATH = GE_ACT_ROOT / "configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml"


def test_siglip2_training_config_matches_the_approved_recipe() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["train_steps"] == 30_000
    assert config["save_steps"] == [20_000, 25_000, 30_000]
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


def test_static_preflight_accepts_the_config() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert collect_preflight_errors(config, world_size=8, check_paths=False) == []
