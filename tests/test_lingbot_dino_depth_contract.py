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
        entry[0][1]["content"] == "<plan-0><plan-1>"
        for entry in processor.conversations
    )


def test_build_planner_inputs_rejects_batch_mismatch():
    module = load_trainer_module()
    with pytest.raises(ValueError, match="batch mismatch"):
        module.build_planner_inputs(RecordingProcessor(), [object()], [], "<plan>")


def test_plan_token_embedding_injector_only_trains_selected_rows():
    module = load_trainer_module()
    base_embedding = nn.Embedding(8, 4)
    base_embedding.weight.requires_grad_(False)
    base_before = base_embedding.weight.detach().clone()
    injector = module.PlanTokenEmbeddingInjector(
        base_embedding,
        plan_token_ids=[2, 5],
    )
    input_ids = torch.tensor([[1, 2, 3, 5]])

    output = injector(input_ids, base_embedding(input_ids))

    assert torch.equal(output[0, 0], base_before[1])
    assert torch.equal(output[0, 2], base_before[3])
    assert torch.equal(output[0, 1], injector.weight[0])
    assert torch.equal(output[0, 3], injector.weight[1])

    optimizer = torch.optim.AdamW(injector.parameters(), lr=0.1, weight_decay=0.1)
    output.sum().backward()
    optimizer.step()

    assert base_embedding.weight.grad is None
    assert torch.equal(base_embedding.weight, base_before)
    assert injector.weight.grad is not None
    assert sum(parameter.numel() for parameter in injector.parameters()) == 8


def test_plan_token_embedding_injector_forward_hook_replaces_plan_positions():
    module = load_trainer_module()
    base_embedding = nn.Embedding(8, 4)
    base_embedding.weight.requires_grad_(False)
    injector = module.PlanTokenEmbeddingInjector(
        base_embedding,
        plan_token_ids=[2, 5],
    )
    handle = base_embedding.register_forward_hook(injector.forward_hook)
    input_ids = torch.tensor([[1, 2, 5]])

    try:
        output = base_embedding(input_ids)
    finally:
        handle.remove()

    assert torch.equal(output[0, 0], base_embedding.weight[1])
    assert torch.equal(output[0, 1], injector.weight[0])
    assert torch.equal(output[0, 2], injector.weight[1])


def test_planner_wrapper_installs_independent_plan_query_embeddings():
    module = load_trainer_module()

    class TinyEmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.config = SimpleNamespace(image_token_id=7)

        def get_input_embeddings(self):
            return self.embedding

    model = TinyEmbeddingModel()
    wrapper = module.PlannerWrapper(
        model=model,
        hidden_size=4,
        semantic_dim=2,
        plan_token_ids=[2, 5],
        target_len=2,
        num_keyframes=1,
        grid_size=1,
        plan_head_type="mlp",
        train_plan_token_embedding=True,
    )
    with torch.no_grad():
        wrapper.plan_embedding_injector.weight.fill_(17.0)

    output = model.get_input_embeddings()(torch.tensor([[1, 2, 5]]))

    assert model.embedding.weight.requires_grad is False
    assert torch.equal(output[0, 0], model.embedding.weight[1])
    assert torch.equal(output[0, 1], torch.full((4,), 17.0))
    assert torch.equal(output[0, 2], torch.full((4,), 17.0))


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


def test_save_checkpoint_exports_independent_plan_query_embeddings(tmp_path):
    module = load_trainer_module()
    wrapper = _make_checkpoint_wrapper(module)
    wrapper.plan_embedding_injector = module.PlanTokenEmbeddingInjector(
        wrapper.model.get_input_embeddings(),
        wrapper.plan_token_ids,
    )
    with torch.no_grad():
        wrapper.model.get_input_embeddings().weight.fill_(-3.0)
        wrapper.plan_embedding_injector.weight.fill_(11.0)

    module.save_checkpoint(
        tmp_path,
        8,
        wrapper,
        FakeProcessor(),
        _make_checkpoint_args(),
        rank=0,
    )

    exported = torch.load(
        tmp_path / "step_000008/plan_token_embedding.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(
        exported,
        wrapper.plan_embedding_injector.weight.detach().cpu(),
    )


def test_save_checkpoint_writes_dynamic_64_token_independent_contract(tmp_path):
    module = load_trainer_module()
    wrapper = _make_checkpoint_wrapper(module)
    wrapper.num_keyframes = 1
    wrapper.target_len = 256
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 256
    wrapper.num_latent_per_keyframe = 64
    wrapper.latent_len = 256
    wrapper.plan_token_ids = list(range(3, 259))
    wrapper.use_current_alignment = True
    wrapper.independent_modality_task_tokens = True
    wrapper.num_task_tokens = 64
    wrapper.current_plan_head = nn.Linear(8, 8)
    wrapper.current_depth_head = nn.Linear(8, 8)
    args = _make_checkpoint_args(num_keyframes=1)

    module.save_checkpoint(
        tmp_path,
        64,
        wrapper,
        FakeProcessor(),
        args,
        rank=0,
    )

    metadata = json.loads(
        (tmp_path / "step_000064/planner_meta.json").read_text()
    )
    assert metadata["num_task_tokens"] == 64
    assert metadata["latent_len"] == 256
    assert metadata["total_unique_latent_per_keyframe"] == 256
    assert metadata["query_layout"] == (
        "current_dino_64_then_future_dino_64_then_"
        "current_depth_64_then_future_depth_64"
    )
    assert len(metadata["plan_token_strings"]) == 256


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
        text_embedding_cache_dir=Path("machine/relative/cache"),
        pretrained_norm_stats=Path("machine/relative/stats.json"),
    )

    assert list(overridden.data.train.dataset_dirs) == [
        str((Path.cwd() / explicit_dirs[0]).resolve()),
        explicit_dirs[1],
    ]
    assert overridden.data.train.text_embedding_cache_dir == str(
        (Path.cwd() / "machine/relative/cache").resolve()
    )
    assert overridden.data.train.pretrained_norm_stats == str(
        (Path.cwd() / "machine/relative/stats.json").resolve()
    )


def _write_fastwam_data_config(tmp_path, *, target="builtins.dict"):
    fastwam_root = tmp_path / "FastWAM"
    config_dir = fastwam_root / "configs/data"
    dataset_dir = fastwam_root / "data/dataset"
    cache_dir = fastwam_root / "cache"
    stats_path = fastwam_root / "stats.json"
    config_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    cache_dir.mkdir()
    stats_path.write_text("{}\n", encoding="utf-8")
    config_path = config_dir / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "train:",
                f"  _target_: {target}",
                "  dataset_dirs:",
                "    - ./data/dataset",
                "  text_embedding_cache_dir: ./cache",
                "  pretrained_norm_stats: ./stats.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "root": fastwam_root,
        "config": config_path,
        "dataset": dataset_dir,
        "cache": cache_dir,
        "stats": stats_path,
    }


def test_fastwam_config_preparation_rebases_yaml_pretrained_stats(tmp_path):
    module = load_trainer_module()
    paths = _write_fastwam_data_config(tmp_path)

    prepared = module.prepare_fastwam_data_config(paths["config"])

    assert list(prepared.data.train.dataset_dirs) == [str(paths["dataset"])]
    assert prepared.data.train.text_embedding_cache_dir == str(paths["cache"])
    assert prepared.data.train.pretrained_norm_stats == str(paths["stats"])


def test_fastwam_config_preflight_imports_target_without_instantiating_dataset(
    monkeypatch,
    tmp_path,
):
    module = load_trainer_module()
    paths = _write_fastwam_data_config(tmp_path)
    config_path = paths["config"]
    monkeypatch.syspath_prepend(str(ROOT / "third_party/FastWAM/src"))

    import hydra.utils

    def reject_instantiation(*_args, **_kwargs):
        raise AssertionError("preflight must not instantiate the FastWAM dataset")

    imported_targets = []
    monkeypatch.setattr(hydra.utils, "instantiate", reject_instantiation)
    monkeypatch.setattr(hydra.utils, "get_class", imported_targets.append)

    prepared = module.preflight_fastwam_data_config(config_path)

    assert prepared.data.train._target_ == "builtins.dict"
    assert imported_targets == [prepared.data.train._target_]


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ("dataset", "FastWAM dataset directory"),
        ("cache", "FastWAM text-embedding cache"),
        ("stats", "FastWAM pretrained normalization stats"),
    ],
)
def test_fastwam_config_preflight_rejects_missing_assets(
    tmp_path,
    asset,
    expected,
):
    module = load_trainer_module()
    paths = _write_fastwam_data_config(tmp_path)
    path = paths[asset]
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()

    with pytest.raises(FileNotFoundError, match=expected):
        module.preflight_fastwam_data_config(paths["config"])


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ("dataset", "FastWAM dataset directory"),
        ("cache", "FastWAM text-embedding cache"),
        ("stats", "FastWAM pretrained normalization stats"),
    ],
)
def test_fastwam_config_preflight_rejects_wrong_asset_types(
    tmp_path,
    asset,
    expected,
):
    module = load_trainer_module()
    paths = _write_fastwam_data_config(tmp_path)
    path = paths[asset]
    if path.is_dir():
        path.rmdir()
        path.write_text("not a directory\n", encoding="utf-8")
    else:
        path.unlink()
        path.mkdir()

    with pytest.raises(ValueError, match=expected):
        module.preflight_fastwam_data_config(paths["config"])


def test_fastwam_dataset_config_overrides_reach_hydra_instantiation(
    monkeypatch,
    tmp_path,
):
    module = load_trainer_module()
    paths = _write_fastwam_data_config(tmp_path)
    caller = tmp_path / "caller"
    caller.mkdir()
    for relative in ("data-a", "data-b", "cache"):
        (caller / relative).mkdir()
    (caller / "stats.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(caller)

    import hydra.utils

    instantiated = []

    def record_instantiation(config):
        instantiated.append(config)
        return []

    monkeypatch.setattr(hydra.utils, "instantiate", record_instantiation)

    module.FastWAMOnlinePlannerDataset.from_config(
        paths["config"],
        dataset_dirs=["data-a", "data-b"],
        text_embedding_cache_dir=Path("cache"),
        pretrained_norm_stats=Path("stats.json"),
    )

    assert len(instantiated) == 1
    config = instantiated[0]
    assert list(config.dataset_dirs) == [
        str((caller / "data-a").resolve()),
        str((caller / "data-b").resolve()),
    ]
    assert config.text_embedding_cache_dir == str((caller / "cache").resolve())
    assert config.pretrained_norm_stats == str((caller / "stats.json").resolve())


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
        fastwam_text_embedding_cache_dir=tmp_path / "cache",
        fastwam_pretrained_norm_stats=tmp_path / "stats.json",
        seed=1,
        output_dir=tmp_path / "output",
        model_path=tmp_path / "model",
        dtype="bf16",
    )

    def fail_preflight(*_args, **kwargs):
        events.append("fastwam_preflight")
        assert kwargs == {
            "dataset_dirs": [],
            "text_embedding_cache_dir": tmp_path / "cache",
            "pretrained_norm_stats": tmp_path / "stats.json",
        }
        raise ExpectedPreflightFailure("bad FastWAM config")

    def fail_model_load(*_args, **_kwargs):
        events.append("qwen_load")
        raise UnexpectedModelLoad("Qwen load happened before preflight")

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "preflight_fastwam_data_config", fail_preflight)
    monkeypatch.setattr(module, "load_qwen3vl_model_and_processor", fail_model_load)

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


def test_gradient_checkpointing_is_opt_in_and_configures_model(monkeypatch):
    module = load_trainer_module()
    monkeypatch.setattr(sys, "argv", _fastwam_parser_argv())

    args = module.parse_args()

    assert args.gradient_checkpointing is False

    class FakeModel:
        def __init__(self):
            self.events = []

        def gradient_checkpointing_enable(self, **kwargs):
            self.events.append(("enable", kwargs))

        def gradient_checkpointing_disable(self):
            self.events.append(("disable", {}))

        def enable_input_require_grads(self):
            self.events.append(("input_grads", {}))

    disabled_model = FakeModel()
    module.configure_gradient_checkpointing(disabled_model, enabled=False)
    assert disabled_model.events == [("disable", {})]

    enabled_model = FakeModel()
    module.configure_gradient_checkpointing(enabled_model, enabled=True)
    assert enabled_model.events == [
        ("enable", {"gradient_checkpointing_kwargs": {"use_reentrant": False}}),
        ("input_grads", {}),
    ]


def test_gradient_checkpointing_cli_flag_enables_opt_in(monkeypatch):
    module = load_trainer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        _fastwam_parser_argv("--gradient-checkpointing"),
    )

    assert module.parse_args().gradient_checkpointing is True


def test_expected_global_batch_cli_defaults_to_unconstrained_and_accepts_128(monkeypatch):
    module = load_trainer_module()
    monkeypatch.setattr(sys, "argv", _fastwam_parser_argv())
    assert module.parse_args().expected_global_batch == 0

    monkeypatch.setattr(
        sys,
        "argv",
        _fastwam_parser_argv("--expected-global-batch", "128"),
    )
    assert module.parse_args().expected_global_batch == 128


def test_independent_modality_task_token_cli_flag_is_opt_in(monkeypatch):
    module = load_trainer_module()
    monkeypatch.setattr(sys, "argv", _fastwam_parser_argv())
    assert module.parse_args().independent_modality_task_tokens is False

    monkeypatch.setattr(
        sys,
        "argv",
        _fastwam_parser_argv(
            "--use-depth",
            "--use-current-alignment",
            "--num-keyframes",
            "1",
            "--independent-modality-task-tokens",
        ),
    )
    assert module.parse_args().independent_modality_task_tokens is True


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
            "--fastwam-text-embedding-cache-dir",
            "/tmp/text-cache",
            "--fastwam-pretrained-norm-stats",
            "/tmp/stats.json",
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
    assert args.fastwam_text_embedding_cache_dir == Path("/tmp/text-cache")
    assert args.fastwam_pretrained_norm_stats == Path("/tmp/stats.json")


@pytest.mark.parametrize(
    "option",
    [
        "--fastwam-text-embedding-cache-dir",
        "--fastwam-pretrained-norm-stats",
    ],
)
def test_fastwam_parser_rejects_overrides_without_fastwam_config(
    monkeypatch,
    capsys,
    option,
):
    module = load_trainer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAINER_PATH),
            "--model-path",
            "/tmp/model",
            "--output-dir",
            "/tmp/output",
            "--dataset-root",
            "/tmp/data",
            option,
            "/tmp/override",
        ],
    )

    with pytest.raises(SystemExit) as error:
        module.parse_args()

    assert error.value.code == 2
    assert f"{option} requires --fastwam-data-config" in capsys.readouterr().err


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


def _capture_base_launcher_args(tmp_path, *, cache, stats):
    launcher = (
        ROOT
        / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
        / "train_lingbot_dino_4b.sh"
    )
    fake_python = tmp_path / "fake-python"
    captured = tmp_path / "args.txt"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$FAKE_ARGS_FILE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PY": str(fake_python),
            "FAKE_ARGS_FILE": str(captured),
            "NUM_GPUS": "1",
            "USE_DEEPSPEED": "0",
            "BATCH_SIZE": "1",
            "GRAD_ACCUM": "1",
            "EXPECTED_GLOBAL_BATCH": "1",
            "FULL_FINETUNE": "0",
            "HEAD_WARMSTART_CKPT": "",
            "DATASET_ROOT": "",
            "FASTWAM_DATA_CONFIG": str(
                ROOT / "third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml"
            ),
            "FASTWAM_TEXT_EMBEDDING_CACHE_DIR": cache,
            "FASTWAM_PRETRAINED_NORM_STATS": stats,
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return captured.read_text(encoding="utf-8").splitlines()


def test_base_launcher_emits_nonempty_fastwam_cache_and_stats_overrides(tmp_path):
    args = _capture_base_launcher_args(
        tmp_path,
        cache="relative/text-cache",
        stats="relative/stats.json",
    )

    assert args[args.index("--fastwam-text-embedding-cache-dir") + 1] == (
        "relative/text-cache"
    )
    assert args[args.index("--fastwam-pretrained-norm-stats") + 1] == (
        "relative/stats.json"
    )


def test_base_launcher_omits_empty_fastwam_cache_and_stats_overrides(tmp_path):
    args = _capture_base_launcher_args(tmp_path, cache="", stats="")

    assert "--fastwam-text-embedding-cache-dir" not in args
    assert "--fastwam-pretrained-norm-stats" not in args


def test_fastwam_hydra_target_imports_in_launcher_python(tmp_path):
    fastwam_src = ROOT / "third_party/FastWAM/src"
    starvla_python = Path(
        "/data/LFT-W02_data/.conda/envs/starVLA/bin/python"
    )
    python = starvla_python if starvla_python.is_file() else Path(sys.executable)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fastwam_src), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    fallback_dir = tmp_path / "missing" / "nested" / "runs"
    env["FASTWAM_WORK_DIR"] = str(fallback_dir)
    assert not fallback_dir.exists()

    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import builtins, importlib.machinery, json, os, shutil, sys, "
                "tempfile, types; from pathlib import Path; "
                "from fastwam.datasets.lerobot.robot_video_dataset "
                "import RobotVideoDataset, _get_work_dir, "
                "save_dataset_stats_to_json; "
                "assert RobotVideoDataset.__name__ == 'RobotVideoDataset'; "
                "real_import = builtins.__import__; "
                "builtins.__import__ = lambda name, *args, **kwargs: "
                "(_ for _ in ()).throw(ModuleNotFoundError("
                "\"No module named 'boto3'\", name='boto3')) "
                "if name == 'boto3' else real_import(name, *args, **kwargs); "
                "work_dir = _get_work_dir(); "
                "stats_path = Path(work_dir) / 'dataset_stats.json'; "
                "save_dataset_stats_to_json({'count': 1}, stats_path); "
                "assert json.loads(stats_path.read_text()) == {'count': 1}; "
                "builtins.__import__ = real_import; "
                "boto3 = types.ModuleType('boto3'); "
                "boto3.__spec__ = importlib.machinery.ModuleSpec("
                "'boto3', loader=None); "
                "sys.modules['boto3'] = boto3; "
                "from fastwam.utils import misc; "
                "registered_dir = tempfile.mkdtemp(); "
                "misc.register_work_dir(registered_dir); "
                "assert _get_work_dir() == registered_dir; "
                "shutil.rmtree(registered_dir)"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

def test_fastwam_default_work_dir_is_created_before_stats_write(tmp_path):
    fastwam_src = ROOT / "third_party/FastWAM/src"
    starvla_python = Path("/data/LFT-W02_data/.conda/envs/starVLA/bin/python")
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
                "import importlib.machinery, sys, types; "
                "from pathlib import Path; "
                "boto3 = types.ModuleType('boto3'); "
                "boto3.__spec__ = importlib.machinery.ModuleSpec("
                "'boto3', loader=None); "
                "sys.modules['boto3'] = boto3; "
                "from fastwam.datasets.lerobot.robot_video_dataset "
                "import _get_work_dir, save_dataset_stats_to_json; "
                "work_dir = Path(_get_work_dir()); "
                "assert work_dir.is_dir(), work_dir; "
                "save_dataset_stats_to_json({'count': 1}, "
                "work_dir / 'dataset_stats.json')"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
