from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py"
DINO_TARGET = ROOT / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_video_target.py"
DEPTH_TARGET = ROOT / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/depth_target.py"


def load_module(name: str, path: Path):
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dino_teacher_returns_current_and_future_from_one_call():
    module = load_module("k1_dino_target", DINO_TARGET)

    class FakeTeacher:
        def __init__(self):
            self.calls = []

        def get_future_feature(self, video, **kwargs):
            self.calls.append((video.clone(), dict(kwargs)))
            batch = video.shape[0]
            future = torch.full((batch, 256, 1024), 2.0)
            current = torch.full((batch, 256, 1024), 1.0)
            return future, current

    encoder = module.DinoVideoTargetEncoder.__new__(module.DinoVideoTargetEncoder)
    nn.Module.__init__(encoder)
    encoder.input_size = 2
    encoder.effective_fps = 1.0
    encoder.teacher = FakeTeacher()
    encoder.register_buffer("mean", torch.zeros(1, 3, 1, 1, 1))
    encoder.register_buffer("std", torch.ones(1, 3, 1, 1, 1))
    current = torch.zeros(2, 3, 2, 2)
    future = torch.ones(2, 3, 2, 2)

    current_target, future_target = encoder.encode_current_and_future(current, future)

    assert len(encoder.teacher.calls) == 1
    video, kwargs = encoder.teacher.calls[0]
    assert video.shape == (2, 3, 3, 2, 2)
    assert torch.equal(video[:, :, 0], video[:, :, 1])
    assert kwargs == {"return_current": True, "current_index": 1, "fps": 1.0}
    assert torch.all(current_target == 1)
    assert torch.all(future_target == 2)


def test_depth_teacher_batches_current_and_future_once():
    module = load_module("k1_depth_target", DEPTH_TARGET)
    encoder = module.DepthTargetEncoder.__new__(module.DepthTargetEncoder)
    nn.Module.__init__(encoder)
    encoder._prep = lambda tensor: tensor.float()
    calls = []

    def fake_depth_target(batch):
        calls.append(batch.clone())
        values = torch.arange(batch.shape[0], dtype=torch.float32).view(-1, 1, 1)
        return values.expand(-1, 256, 1024)

    encoder._depth_target = fake_depth_target
    current = torch.zeros(2, 3, 2, 2)
    future = torch.ones(2, 3, 2, 2)

    current_target, future_target = encoder.encode_current_and_future(current, future)

    assert len(calls) == 1
    assert calls[0].shape[0] == 4
    assert torch.all(current_target[0] == 0)
    assert torch.all(current_target[1] == 1)
    assert torch.all(future_target[0] == 2)
    assert torch.all(future_target[1] == 3)


def test_k1_query_split_uses_eight_current_and_eight_future_tokens():
    module = load_module("k1_trainer_split", TRAINER)
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.num_task_tokens = 8
    hidden = torch.arange(2 * 16 * 3).reshape(2, 16, 3)

    current, future = wrapper.split_current_future_task_hidden(hidden)

    assert torch.equal(current, hidden[:, :8])
    assert torch.equal(future, hidden[:, 8:])


def test_k1_independent_modality_query_split_uses_four_private_groups():
    module = load_module("k1_trainer_independent_split", TRAINER)
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.num_task_tokens = 8
    hidden = torch.arange(2 * 32 * 3).reshape(2, 32, 3)

    groups = wrapper.split_independent_current_future_task_hidden(hidden)

    assert list(groups) == [
        "current_dino",
        "future_dino",
        "current_depth",
        "future_depth",
    ]
    assert torch.equal(groups["current_dino"], hidden[:, 0:8])
    assert torch.equal(groups["future_dino"], hidden[:, 8:16])
    assert torch.equal(groups["current_depth"], hidden[:, 16:24])
    assert torch.equal(groups["future_depth"], hidden[:, 24:32])


def test_k1_independent_modality_query_split_supports_four_64_token_groups():
    module = load_module("k1_trainer_independent_split_64", TRAINER)
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.num_task_tokens = 64
    hidden = torch.arange(2 * 256 * 3).reshape(2, 256, 3)

    groups = wrapper.split_independent_current_future_task_hidden(hidden)

    assert torch.equal(groups["current_dino"], hidden[:, 0:64])
    assert torch.equal(groups["future_dino"], hidden[:, 64:128])
    assert torch.equal(groups["current_depth"], hidden[:, 128:192])
    assert torch.equal(groups["future_depth"], hidden[:, 192:256])


def test_k1_wrapper_builds_four_independent_heads_from_sixteen_task_tokens():
    module = load_module("k1_trainer_heads", TRAINER)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"image_token_id": 1})()

    wrapper = module.PlannerWrapper(
        model=TinyModel(),
        hidden_size=32,
        semantic_dim=16,
        plan_token_ids=list(range(16)),
        target_len=4,
        num_keyframes=1,
        grid_size=2,
        plan_head_type="lingbot_dino",
        use_depth=True,
        depth_dim=16,
        depth_grid_size=2,
        use_current_alignment=True,
        num_task_tokens=8,
    )

    assert wrapper.latent_len == 16
    assert wrapper.target_len == 4
    assert wrapper.num_latent_per_keyframe == 8
    assert wrapper.plan_head is not wrapper.current_plan_head
    assert wrapper.depth_head is not wrapper.current_depth_head
    assert wrapper.plan_head.num_latent_per_keyframe == 8
    assert wrapper.current_plan_head.num_latent_per_keyframe == 8
    assert wrapper.depth_head.num_latent_per_keyframe == 8
    assert wrapper.current_depth_head.num_latent_per_keyframe == 8


def test_k1_wrapper_can_build_four_independent_task_token_groups():
    module = load_module("k1_trainer_independent_heads", TRAINER)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"image_token_id": 1})()

    wrapper = module.PlannerWrapper(
        model=TinyModel(),
        hidden_size=32,
        semantic_dim=16,
        plan_token_ids=list(range(32)),
        target_len=4,
        num_keyframes=1,
        grid_size=2,
        plan_head_type="lingbot_dino",
        use_depth=True,
        depth_dim=16,
        depth_grid_size=2,
        use_current_alignment=True,
        independent_modality_task_tokens=True,
        num_task_tokens=8,
    )

    assert wrapper.independent_modality_task_tokens is True
    assert wrapper.latent_len == 32
    assert wrapper.total_unique_latent_per_keyframe == 32
    assert wrapper.plan_token_ids == list(range(32))


def test_k1_wrapper_supports_64_tokens_per_independent_group():
    module = load_module("k1_trainer_independent_heads_64", TRAINER)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"image_token_id": 1})()

    wrapper = module.PlannerWrapper(
        model=TinyModel(),
        hidden_size=32,
        semantic_dim=16,
        plan_token_ids=list(range(256)),
        target_len=4,
        num_keyframes=1,
        grid_size=2,
        plan_head_type="lingbot_dino",
        use_depth=True,
        depth_dim=16,
        depth_grid_size=2,
        use_current_alignment=True,
        independent_modality_task_tokens=True,
        num_task_tokens=64,
    )

    assert wrapper.latent_len == 256
    assert wrapper.total_unique_latent_per_keyframe == 256
    assert wrapper.num_latent_per_keyframe == 64
    assert wrapper.plan_head.num_latent_per_keyframe == 64
    assert wrapper.depth_head.num_latent_per_keyframe == 64
    assert wrapper.current_plan_head.num_latent_per_keyframe == 64
    assert wrapper.current_depth_head.num_latent_per_keyframe == 64


def test_depth_warmstart_does_not_accidentally_load_future_depth_head():
    module = load_module(
        "k1_lingbot_head",
        ROOT / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/lingbot_dino_head.py",
    )
    head = module.LingbotDinoPlanHead(
        num_keyframes=1,
        num_latent_per_keyframe=8,
        num_backbone_tokens=1,
        llm_hidden=32,
        dim_out=16,
    )
    state = {}
    for name, tensor in head.resampler.state_dict().items():
        state[f"model.depth_align_head.projector.{name}"] = torch.ones_like(tensor)
        state[f"model.future_depth_align_head.projector.{name}"] = torch.full_like(
            tensor, 2
        )
    state["model.depth_align_embs"] = torch.ones_like(head.query_embs)
    state["model.future_depth_align_embs"] = torch.full_like(head.query_embs, 2)

    report = head.load_lingbot_warmstart(state, head_name="depth_align_head")

    assert report["query_loaded"] is True
    assert torch.all(head.query_embs == 1)
    assert all(torch.all(tensor == 1) for tensor in head.resampler.state_dict().values())


def test_lingbot_four_term_loss_matches_released_weights():
    module = load_module("k1_trainer_loss", TRAINER)
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.current_dino_loss_weight = 0.004
    wrapper.future_dino_loss_weight = 0.004
    wrapper.current_depth_loss_weight = 0.004
    wrapper.future_depth_loss_weight = 0.004
    zeros = torch.zeros(1, 2, 3)
    plans = {
        "current_dino": torch.full_like(zeros, 1.0),
        "future_dino": torch.full_like(zeros, 2.0),
        "current_depth": torch.full_like(zeros, 3.0),
        "future_depth": torch.full_like(zeros, 4.0),
    }
    targets = {name: zeros for name in plans}

    losses = wrapper.compute_current_future_losses(plans, targets)

    current_dino = torch.tensor(1.0)
    future_dino = torch.tensor(4.0)
    current_depth = torch.tensor(2.5)
    future_depth = torch.tensor(3.5)
    expected = 0.004 * (current_dino + future_dino + current_depth + future_depth)
    assert torch.allclose(losses["loss"], expected)
    assert torch.allclose(losses["current_dino_mse"], current_dino)
    assert torch.allclose(losses["future_dino_mse"], future_dino)
    assert torch.allclose(losses["current_depth_smooth_l1"], current_depth)
    assert torch.allclose(losses["future_depth_smooth_l1"], future_depth)
