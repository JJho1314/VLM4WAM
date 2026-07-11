from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner"
    / "train_qwen3vl4b_lingbot_dino_planner.py"
)


def load_trainer_module():
    trainer_dir = str(TRAINER_PATH.parent)
    if trainer_dir not in sys.path:
        sys.path.insert(0, trainer_dir)
    spec = importlib.util.spec_from_file_location(
        "lingbot_planner_trainer", TRAINER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CountingHead(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0
        self.anchor = nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, image_hidden, plan_hidden):
        self.calls += 1
        self.last_plan_hidden = plan_hidden.detach().clone()
        batch = plan_hidden.shape[0]
        return torch.full(
            (batch, 4 * 16 * 16, 1024),
            self.value,
            device=plan_hidden.device,
        )


class RecordingProcessor:
    def __init__(self):
        self.conversations = []
        self.processor_call = None

    def apply_chat_template(self, conversation, **kwargs):
        self.conversations.append((conversation, kwargs))
        return f"rendered-{len(self.conversations)}"

    def __call__(self, **kwargs):
        self.processor_call = kwargs
        return {"input_ids": torch.ones(len(kwargs["text"]), 2)}


def test_build_planner_inputs_uses_one_shared_prompt_contract():
    module = load_trainer_module()
    processor = RecordingProcessor()
    images = [object(), object()]

    result = module.build_planner_inputs(
        processor,
        images,
        ["pick up the cup", 17],
        ["<plan-0>", "<plan-1>"],
    )

    assert result["input_ids"].shape == (2, 2)
    assert processor.processor_call == {
        "text": ["rendered-1", "rendered-2"],
        "images": images,
        "padding": True,
        "return_tensors": "pt",
    }
    assert [entry[0][0]["content"][1]["text"] for entry in processor.conversations] == [
        module.PLANNER_USER_TEMPLATE.format(instruction="pick up the cup"),
        module.PLANNER_USER_TEMPLATE.format(instruction="17"),
    ]
    assert all(
        entry[0][1]["content"] == "<plan-0> <plan-1>"
        for entry in processor.conversations
    )


def test_build_planner_inputs_rejects_batch_mismatch():
    module = load_trainer_module()
    with pytest.raises(ValueError, match="batch mismatch"):
        module.build_planner_inputs(RecordingProcessor(), [object()], [], "<plan>")


def _make_lingbot_wrapper(*, use_depth: bool):
    module = load_trainer_module()
    model = nn.Linear(1, 1)
    model.config = SimpleNamespace(image_token_id=42)
    return module.PlannerWrapper(
        model=model,
        hidden_size=8,
        semantic_dim=4,
        plan_token_ids=[1],
        target_len=2,
        num_keyframes=2,
        grid_size=1,
        num_latent_per_keyframe=7,
        shared_latent_per_keyframe=2,
        private_latent_per_keyframe=3,
        plan_head_type="lingbot_dino",
        use_depth=use_depth,
        depth_dim=4,
        depth_grid_size=1,
    )


def test_lingbot_depth_wrapper_uses_shared_private_query_geometry():
    wrapper = _make_lingbot_wrapper(use_depth=True)

    assert wrapper.num_keyframes == 2
    assert wrapper.shared_latent_per_keyframe == 2
    assert wrapper.private_latent_per_keyframe == 3
    assert wrapper.branch_latent_per_keyframe == 5
    assert wrapper.total_unique_latent_per_keyframe == 8
    assert wrapper.num_latent_per_keyframe == 5
    assert wrapper.latent_len == 16
    assert wrapper.plan_head.num_latent_per_keyframe == 5
    assert wrapper.depth_head.num_latent_per_keyframe == 5


def test_lingbot_without_depth_preserves_legacy_query_geometry():
    wrapper = _make_lingbot_wrapper(use_depth=False)

    assert wrapper.num_latent_per_keyframe == 7
    assert wrapper.latent_len == 14
    assert wrapper.plan_head.num_latent_per_keyframe == 7
    assert wrapper.depth_head is None


def test_split_lingbot_queries_shares_only_the_shared_group():
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.num_keyframes = 4
    wrapper.shared_latent_per_keyframe = 32
    wrapper.private_latent_per_keyframe = 32
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 96
    hidden = torch.arange(4 * 96, dtype=torch.float32).reshape(1, 4 * 96, 1)

    dino_hidden, depth_hidden = wrapper.split_lingbot_query_hidden(hidden)

    grouped = hidden.reshape(1, 4, 96, 1)
    expected_dino = torch.cat(
        [grouped[:, :, :32], grouped[:, :, 32:64]],
        dim=2,
    ).reshape(1, 4 * 64, 1)
    expected_depth = torch.cat(
        [grouped[:, :, :32], grouped[:, :, 64:96]],
        dim=2,
    ).reshape(1, 4 * 64, 1)
    assert torch.equal(dino_hidden, expected_dino)
    assert torch.equal(depth_hidden, expected_depth)
    assert not torch.equal(dino_hidden[:, 32:64], depth_hidden[:, 32:64])


def test_even_future_offsets_cover_every_second_future_frame():
    module = load_trainer_module()
    assert module.keyframe_offsets(9, 4, "even_future", 0.6) == [
        2,
        4,
        6,
        8,
    ]
    with pytest.raises(ValueError, match="divisible"):
        module.keyframe_offsets(9, 3, "even_future", 0.6)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--shared-latent-per-keyframe", "0"),
        ("--private-latent-per-keyframe", "-1"),
    ],
)
def test_parser_rejects_nonpositive_dual_branch_latent_counts(
    monkeypatch,
    flag,
    value,
):
    module = load_trainer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trainer",
            "--model-path",
            "model",
            "--dataset-root",
            "dataset",
            "--output-dir",
            "output",
            flag,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_predict_dino_depth_plan_uses_one_vlm_forward():
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.plan_head = CountingHead(1.0)
    wrapper.depth_head = CountingHead(2.0)
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.num_keyframes = 4
    wrapper.shared_latent_per_keyframe = 32
    wrapper.private_latent_per_keyframe = 32
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 96
    wrapper.vlm_forward_calls = 0

    def fake_forward_hiddens(**_inputs):
        wrapper.vlm_forward_calls += 1
        plan_hidden = torch.arange(
            2 * 4 * 96 * 64,
            dtype=torch.float32,
        ).reshape(2, 4 * 96, 64)
        return torch.zeros(2, 6, 64), plan_hidden

    wrapper._forward_hiddens = fake_forward_hiddens
    dino, depth = wrapper.predict_dino_depth_plan(input_ids=torch.ones(2, 4))

    assert wrapper.vlm_forward_calls == 1
    assert wrapper.plan_head.calls == 1
    assert wrapper.depth_head.calls == 1
    assert wrapper.plan_head.last_plan_hidden.shape == (2, 4 * 64, 64)
    assert wrapper.depth_head.last_plan_hidden.shape == (2, 4 * 64, 64)
    assert dino.shape == depth.shape == (2, 1024, 1024)
    assert torch.all(dino == 1)
    assert torch.all(depth == 2)


def test_predict_semantic_plan_uses_only_the_dino_query_branch():
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.plan_head = CountingHead(1.0)
    wrapper.depth_head = CountingHead(2.0)
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.use_depth = True
    wrapper.num_keyframes = 4
    wrapper.shared_latent_per_keyframe = 32
    wrapper.private_latent_per_keyframe = 32
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 96
    plan_hidden = torch.arange(4 * 96, dtype=torch.float32).reshape(1, 4 * 96, 1)
    wrapper._forward_hiddens = lambda **_: (torch.zeros(1, 6, 1), plan_hidden)

    wrapper.predict_semantic_plan(input_ids=torch.ones(1, 4))

    expected_dino, _ = wrapper.split_lingbot_query_hidden(plan_hidden)
    assert torch.equal(wrapper.plan_head.last_plan_hidden, expected_dino)
    assert wrapper.depth_head.calls == 0


def test_training_routes_branch_specific_queries_after_one_vlm_forward():
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.plan_head = CountingHead(1.0)
    wrapper.depth_head = CountingHead(2.0)
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.use_depth = True
    wrapper.num_keyframes = 4
    wrapper.shared_latent_per_keyframe = 32
    wrapper.private_latent_per_keyframe = 32
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 96
    wrapper.target_len = 1024
    wrapper.depth_loss_weight = 0.004
    wrapper.vlm_forward_calls = 0

    def fake_forward_hiddens(**_inputs):
        wrapper.vlm_forward_calls += 1
        plan_hidden = torch.arange(
            2 * 4 * 96 * 64,
            dtype=torch.float32,
        ).reshape(2, 4 * 96, 64)
        return torch.zeros(2, 6, 64), plan_hidden

    wrapper._forward_hiddens = fake_forward_hiddens
    wrapper.compute_plan_losses = lambda pred, _target: {"loss": pred.mean()}

    wrapper.forward(
        semantic_plan_labels=torch.zeros(2, 1024, 1024),
        depth_plan_labels=torch.zeros(2, 1024, 1024),
        input_ids=torch.ones(2, 4),
    )

    assert wrapper.vlm_forward_calls == 1
    assert wrapper.plan_head.last_plan_hidden.shape == (2, 4 * 64, 64)
    assert wrapper.depth_head.last_plan_hidden.shape == (2, 4 * 64, 64)


def test_predict_dino_depth_plan_requires_depth_head():
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.plan_head = CountingHead(1.0)
    wrapper.depth_head = None
    wrapper.plan_head_type = "lingbot_dino"
    wrapper._forward_hiddens = lambda **_: (
        torch.zeros(1, 6, 64),
        torch.zeros(1, 4 * 96, 64),
    )

    try:
        wrapper.predict_dino_depth_plan(input_ids=torch.ones(1, 4))
    except RuntimeError as error:
        assert "depth head" in str(error).lower()
    else:
        raise AssertionError("missing depth head must fail")


class FakeSaveableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(512, 8)

    def get_input_embeddings(self):
        return self.embedding

    def save_pretrained(self, output_dir, **_kwargs):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")


class FakeProcessor:
    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "processor_config.json").write_text("{}")


def _make_checkpoint_wrapper(module):
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = FakeSaveableModel()
    wrapper.plan_head = nn.Linear(8, 8)
    wrapper.depth_head = nn.Linear(8, 8)
    wrapper.plan_token_ids = list(range(3, 387))
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.num_keyframes = 4
    wrapper.shared_latent_per_keyframe = 32
    wrapper.private_latent_per_keyframe = 32
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 96
    wrapper.num_latent_per_keyframe = 64
    wrapper.latent_len = 384
    wrapper.target_len = 1024
    wrapper.plan_head_num_heads = 16
    wrapper.plan_head_dropout = 0.0
    wrapper.sem_mlp_hidden_size = 0
    wrapper.mse_loss_weight = 1.0
    wrapper.cosine_loss_weight = 0.0
    wrapper.norm_loss_weight = 0.0
    wrapper.variance_loss_weight = 0.0
    wrapper.infonce_loss_weight = 0.0
    wrapper.infonce_temperature = 0.07
    wrapper.depth_loss_weight = 0.004
    return wrapper


def _make_checkpoint_args(**overrides):
    values = {
        "sample_feature_type": "lingbot_dino_depth",
        "plan_label_dir": None,
        "sample_one_window_per_stem": False,
        "online_plan_labels": True,
        "keyframe_scheme": "even_future",
        "keyframe_gamma": 0.6,
        "sequence_length": 9,
        "online_grid_size": 16,
        "siglip2_encoder_path": None,
        "frame_ranges_json": None,
        "fastwam_data_config": Path("configs/data/libero_2cam_cosmos.yaml"),
        "model_path": Path("Qwen3-VL-4B-lingbot-vlm"),
        "num_keyframes": 4,
        "shared_latent_per_keyframe": 32,
        "private_latent_per_keyframe": 32,
        "grid_size": 16,
        "semantic_dim": 1024,
        "train_plan_token_embedding": True,
        "full_finetune": True,
        "freeze_vision": True,
        "freeze_lm_head": True,
        "use_depth": True,
        "depth_dim": 1024,
        "depth_grid_size": 16,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("target", "field", "incompatible_value"),
    [
        ("args", "use_depth", False),
        ("args", "sequence_length", 49),
        ("args", "num_keyframes", 6),
        ("args", "keyframe_scheme", "uniform"),
        ("args", "grid_size", 15),
        ("args", "semantic_dim", 768),
        ("args", "depth_grid_size", 15),
        ("args", "depth_dim", 768),
        ("wrapper", "num_keyframes", 5),
        ("wrapper", "target_len", 1023),
        ("wrapper", "shared_latent_per_keyframe", 31),
        ("wrapper", "private_latent_per_keyframe", 31),
        ("wrapper", "branch_latent_per_keyframe", 63),
        ("wrapper", "total_unique_latent_per_keyframe", 95),
        ("wrapper", "num_latent_per_keyframe", 63),
        ("wrapper", "latent_len", 383),
        ("wrapper", "plan_token_ids", list(range(3, 386))),
        ("wrapper", "plan_token_ids", [3] * 384),
        ("wrapper", "plan_head_type", "covt"),
        ("wrapper", "depth_head", None),
    ],
    ids=[
        "use-depth",
        "sequence-length",
        "args-num-keyframes",
        "keyframe-scheme-and-offsets",
        "grid-size",
        "semantic-dim",
        "depth-grid-size",
        "depth-dim",
        "wrapper-num-keyframes",
        "target-len",
        "shared-latents",
        "private-latents",
        "branch-latents",
        "total-unique-latents",
        "head-latents",
        "latent-len",
        "plan-token-count",
        "unique-plan-token-count",
        "plan-head-type",
        "depth-head",
    ],
)
def test_save_fastwam_checkpoint_preflight_rejects_incompatible_contract(
    tmp_path,
    target,
    field,
    incompatible_value,
):
    module = load_trainer_module()
    wrapper = _make_checkpoint_wrapper(module)
    args = _make_checkpoint_args()
    setattr(args if target == "args" else wrapper, field, incompatible_value)
    checkpoint = tmp_path / "step_000001"

    with pytest.raises(ValueError, match="FastWAM"):
        module.save_checkpoint(
            tmp_path,
            1,
            wrapper,
            FakeProcessor(),
            args,
            rank=0,
        )
    assert not checkpoint.exists()


def test_save_legacy_checkpoint_skips_fastwam_preflight(tmp_path):
    module = load_trainer_module()
    wrapper = _make_checkpoint_wrapper(module)
    wrapper.depth_head = None
    wrapper.plan_head_type = "covt"
    args = _make_checkpoint_args(
        fastwam_data_config=None,
        use_depth=False,
        sequence_length=49,
        num_keyframes=6,
        keyframe_scheme="uniform",
        grid_size=9,
        semantic_dim=768,
        depth_grid_size=8,
        depth_dim=512,
    )

    module.save_checkpoint(
        tmp_path,
        2,
        wrapper,
        FakeProcessor(),
        args,
        rank=0,
    )

    assert (tmp_path / "step_000002/planner_meta.json").is_file()


def test_save_checkpoint_writes_depth_and_fastwam_contract(tmp_path):
    module = load_trainer_module()
    wrapper = _make_checkpoint_wrapper(module)
    args = _make_checkpoint_args()

    module.save_checkpoint(
        tmp_path,
        7,
        wrapper,
        FakeProcessor(),
        args,
        rank=0,
    )

    checkpoint = tmp_path / "step_000007"
    assert (checkpoint / "qwen3vl_lora_or_model/config.json").is_file()
    assert (checkpoint / "processor/processor_config.json").is_file()
    assert (checkpoint / "plan_head.pt").is_file()
    assert (checkpoint / "depth_head.pt").is_file()
    assert (checkpoint / "plan_token_embedding.pt").is_file()
    metadata = json.loads((checkpoint / "planner_meta.json").read_text())
    assert metadata["sequence_length"] == 9
    assert metadata["num_keyframes"] == 4
    assert metadata["grid_size"] == 16
    assert metadata["semantic_dim"] == 1024
    assert metadata["target_tokens"] == 1024
    assert metadata["keyframe_offsets"] == [2, 4, 6, 8]
    assert metadata["has_depth_head"] is True
    assert metadata["token_order"] == "keyframe_major_row_major"
    assert metadata["query_layout"] == (
        "keyframe_major__shared_dino_private_depth_private"
    )
    assert metadata["shared_latent_per_keyframe"] == 32
    assert metadata["private_latent_per_keyframe"] == 32
    assert metadata["branch_latent_per_keyframe"] == 64
    assert metadata["total_unique_latent_per_keyframe"] == 96
    assert metadata["plan_token_strings"] == [
        f"<|sem_plan_{index}|>" for index in range(384)
    ]


def test_fastwam_planner_dataset_uses_composed_nine_frame_video():
    module = load_trainer_module()
    source_pixels = (
        torch.arange(3, dtype=torch.uint8)[:, None, None, None] * 80
        + torch.arange(9, dtype=torch.uint8)[None, :, None, None] * 8
        + torch.arange(2, dtype=torch.uint8)[None, None, :, None] * 3
        + torch.arange(3, dtype=torch.uint8)[None, None, None, :]
    )

    class FakeFastWAMDataset:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {
                'video': source_pixels.to(torch.float32) / 127.5 - 1.0,
                'instruction': 'open the middle drawer',
            }

    dataset = module.FastWAMOnlinePlannerDataset.from_dataset(
        FakeFastWAMDataset(),
        max_samples=0,
    )
    item = dataset[0]

    expected = source_pixels[:, [0, 2, 4, 6, 8]].permute(1, 2, 3, 0)
    assert item['image'].size == (3, 2)
    assert item['prompt'] == 'open the middle drawer'
    assert item['keyframe_images'].shape == (4, 2, 3, 3)
    assert item['current_image'].shape == (2, 3, 3)
    assert np.array_equal(np.asarray(item['image']), expected[0].numpy())
    assert torch.equal(item['current_image'], expected[0])
    assert torch.equal(item['keyframe_images'], expected[1:])
    assert dataset.offsets == [2, 4, 6, 8]


@pytest.mark.parametrize("shape", [(3, 8, 2, 2), (3, 9, 2)])
def test_fastwam_planner_dataset_rejects_malformed_video_shape(shape):
    module = load_trainer_module()
    wrapped = [
        {
            "video": torch.zeros(shape),
            "instruction": "pick up the cup",
        }
    ]
    dataset = module.FastWAMOnlinePlannerDataset.from_dataset(wrapped)

    with pytest.raises(ValueError, match=r"must be \[3, 9, H, W\]"):
        dataset[0]


@pytest.mark.parametrize("instruction", [None, "", "   ", 7])
def test_fastwam_planner_dataset_rejects_malformed_instruction(instruction):
    module = load_trainer_module()
    wrapped = [
        {
            "video": torch.zeros(3, 9, 2, 2),
            "instruction": instruction,
        }
    ]
    dataset = module.FastWAMOnlinePlannerDataset.from_dataset(wrapped)

    with pytest.raises(ValueError, match="non-empty raw instruction"):
        dataset[0]


def test_fastwam_config_preparation_rebases_yaml_paths_without_instantiation():
    module = load_trainer_module()
    config_path = (
        ROOT / "third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml"
    )
    fastwam_root = config_path.resolve().parents[2]

    prepared = module.prepare_fastwam_data_config(config_path)

    expected_dataset_dirs = [
        str((fastwam_root / relative).resolve())
        for relative in (
            "data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot",
            "data/libero_mujoco3.3.2/libero_object_no_noops_lerobot",
            "data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot",
            "data/libero_mujoco3.3.2/libero_10_no_noops_lerobot",
        )
    ]
    assert list(prepared.data.train.dataset_dirs) == expected_dataset_dirs
    assert prepared.data.train.text_embedding_cache_dir == str(
        (fastwam_root / "data/text_embeds_cache/libero_qwen").resolve()
    )

    explicit_dirs = ["machine/relative/data", "/mnt/absolute/data"]
    overridden = module.prepare_fastwam_data_config(
        config_path,
        dataset_dirs=explicit_dirs,
    )

    assert list(overridden.data.train.dataset_dirs) == explicit_dirs
    assert overridden.data.train.text_embedding_cache_dir == str(
        (fastwam_root / "data/text_embeds_cache/libero_qwen").resolve()
    )


def test_fastwam_config_preflight_imports_target_without_instantiating_dataset(
    monkeypatch,
):
    module = load_trainer_module()
    config_path = (
        ROOT / "third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml"
    )
    monkeypatch.syspath_prepend(str(ROOT / "third_party/FastWAM/src"))

    import hydra.utils

    def reject_instantiation(*_args, **_kwargs):
        raise AssertionError("preflight must not instantiate the FastWAM dataset")

    imported_targets = []
    monkeypatch.setattr(hydra.utils, "instantiate", reject_instantiation)
    monkeypatch.setattr(hydra.utils, "get_class", imported_targets.append)

    prepared = module.preflight_fastwam_data_config(config_path)

    assert prepared.data.train._target_ == (
        "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset"
    )
    assert imported_targets == [prepared.data.train._target_]


def test_main_preflights_fastwam_before_loading_qwen(monkeypatch, tmp_path):
    module = load_trainer_module()
    events = []

    class ExpectedPreflightFailure(RuntimeError):
        pass

    class UnexpectedModelLoad(RuntimeError):
        pass

    args = Namespace(
        full_finetune=False,
        lora_r=0,
        plan_head_type="lingbot_dino",
        use_depth=False,
        shared_latent_per_keyframe=32,
        private_latent_per_keyframe=32,
        fastwam_data_config=tmp_path / "data.yaml",
        fastwam_dataset_dir=[],
        seed=1,
        output_dir=tmp_path / "output",
        model_path=tmp_path / "model",
        dtype="bf16",
    )

    def fail_preflight(*_args, **_kwargs):
        events.append("fastwam_preflight")
        raise ExpectedPreflightFailure("bad FastWAM config")

    def fail_model_load(*_args, **_kwargs):
        events.append("qwen_load")
        raise UnexpectedModelLoad("Qwen load happened before preflight")

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "preflight_fastwam_data_config", fail_preflight)
    monkeypatch.setattr(module, "load_qwen3vl_model_and_processor", fail_model_load)
    monkeypatch.setattr(module, "ddp_info", lambda: (0, 1, 0))

    with pytest.raises(ExpectedPreflightFailure, match="bad FastWAM config"):
        module.main()

    assert events == ["fastwam_preflight"]


def _fastwam_parser_argv(*extra: str) -> list[str]:
    return [
        str(TRAINER_PATH),
        "--model-path",
        "/tmp/model",
        "--output-dir",
        "/tmp/output",
        "--fastwam-data-config",
        str(ROOT / "third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml"),
        "--online-plan-labels",
        "--sequence-length",
        "9",
        "--num-keyframes",
        "4",
        "--keyframe-scheme",
        "even_future",
        *extra,
    ]


def test_fastwam_parser_accepts_only_aligned_source_and_geometry(monkeypatch):
    module = load_trainer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        _fastwam_parser_argv(
            "--fastwam-dataset-dir",
            "machine/relative/data",
            "--fastwam-dataset-dir",
            "/mnt/absolute/data",
        ),
    )

    args = module.parse_args()

    assert args.dataset_root is None
    assert args.sequence_length == 9
    assert args.num_keyframes == 4
    assert args.keyframe_scheme == "even_future"
    assert args.fastwam_dataset_dir == [
        "machine/relative/data",
        "/mnt/absolute/data",
    ]


@pytest.mark.parametrize("source_mode", ["both", "neither", "not_online"])
def test_fastwam_parser_rejects_invalid_source_mode(
    monkeypatch,
    capsys,
    source_mode,
):
    module = load_trainer_module()
    argv = _fastwam_parser_argv()
    if source_mode == "both":
        argv.extend(["--dataset-root", "/tmp/legacy"])
    elif source_mode == "neither":
        config_index = argv.index("--fastwam-data-config")
        del argv[config_index : config_index + 2]
    else:
        argv.remove("--online-plan-labels")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as error:
        module.parse_args()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    if source_mode in {"both", "neither"}:
        assert "exactly one of --dataset-root or --fastwam-data-config" in stderr
    else:
        assert "--fastwam-data-config requires --online-plan-labels" in stderr


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--sequence-length", "13"),
        ("--num-keyframes", "3"),
        ("--keyframe-scheme", "uniform"),
    ],
)
def test_fastwam_parser_rejects_misaligned_geometry(
    monkeypatch,
    capsys,
    flag,
    value,
):
    module = load_trainer_module()
    argv = _fastwam_parser_argv()
    value_index = argv.index(flag) + 1
    argv[value_index] = value
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as error:
        module.parse_args()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert (
        "keyframe offsets [2, 4, 6, 8]" in stderr
        or "invalid FastWAM keyframe geometry" in stderr
    )


def test_fastwam_launcher_pins_nine_frame_dual_branch_contract():
    launcher = (
        ROOT
        / 'scripts/qwen3_vl_semantic_planner/lingbot_dino_4b'
        / 'train_lingbot_dino_depth_fastwam_k4.sh'
    ).read_text()
    required_exports = (
        'export USE_DEPTH=1',
        'export SEQUENCE_LENGTH=9',
        'export NUM_KEYFRAMES=4',
        'export GRID_SIZE=16',
        'export SEMANTIC_DIM=1024',
        'export KEYFRAME_SCHEME=even_future',
        'export SHARED_LATENT_PER_KEYFRAME=32',
        'export PRIVATE_LATENT_PER_KEYFRAME=32',
        'export FASTWAM_DATA_CONFIG=',
    )
    for export in required_exports:
        assert export in launcher
    assert 'train_lingbot_dino_4b.sh' in launcher


def test_base_launcher_exposes_in_repo_fastwam_package():
    launcher = (
        ROOT
        / 'scripts/qwen3_vl_semantic_planner/lingbot_dino_4b'
        / 'train_lingbot_dino_4b.sh'
    ).read_text()

    assert (
        'export PYTHONPATH="$REPO_ROOT/third_party/FastWAM/src'
        '${PYTHONPATH:+:$PYTHONPATH}"'
    ) in launcher


def test_fastwam_hydra_target_imports_in_launcher_python():
    fastwam_src = ROOT / "third_party/FastWAM/src"
    starvla_python = Path(
        "/data/LFT-W02_data/.conda/envs/starVLA/bin/python"
    )
    python = starvla_python if starvla_python.is_file() else Path(sys.executable)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fastwam_src), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.machinery, os, shutil, sys, tempfile, types; "
                "from fastwam.datasets.lerobot.robot_video_dataset "
                "import RobotVideoDataset, _get_work_dir; "
                "assert RobotVideoDataset.__name__ == 'RobotVideoDataset'; "
                "os.environ['FASTWAM_WORK_DIR'] = '/tmp/fastwam-fallback'; "
                "assert _get_work_dir() == '/tmp/fastwam-fallback'; "
                "os.environ.pop('FASTWAM_WORK_DIR'); "
                "assert _get_work_dir() == './runs'; "
                "boto3 = types.ModuleType('boto3'); "
                "boto3.__spec__ = importlib.machinery.ModuleSpec("
                "'boto3', loader=None); "
                "sys.modules['boto3'] = boto3; "
                "from fastwam.utils import misc; "
                "work_dir = tempfile.mkdtemp(); "
                "misc.register_work_dir(work_dir); "
                "assert _get_work_dir() == work_dir; "
                "shutil.rmtree(work_dir)"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
