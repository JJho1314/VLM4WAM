from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset

from qwen3_vl_semantic_planner.ge_act_dual_camera import (
    DualCameraPlannerCollator,
    GEActDualCameraPlannerDataset,
    build_dual_camera_planner_inputs,
)

PLANNER_ROOT = Path(__file__).resolve().parents[1] / "qwen3_vl_semantic_planner"
if str(PLANNER_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANNER_ROOT))

from qwen3_vl_semantic_planner import train_qwen3vl4b_lingbot_dino_planner as planner
from qwen3_vl_semantic_planner import qwen3vl_wrapper as qwen_helper

PlannerWrapper = planner.PlannerWrapper


class FakeDataset(Dataset):
    def __init__(self, sample: dict[str, Any]):
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        return self.sample


class RecordingProcessor:
    def __init__(self) -> None:
        self.images: list[Image.Image] = []
        self.texts: list[str] = []
        self.rendered_conversations: list[str] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        rendered: list[str] = []
        for message in conversation:
            content = message["content"]
            if isinstance(content, str):
                rendered.append(content)
                continue
            for part in content:
                if part["type"] == "image":
                    rendered.append("<|image_pad|>")
                else:
                    rendered.append(part["text"])
        rendered_conversation = "\n".join(rendered)
        self.rendered_conversations.append(rendered_conversation)
        return rendered_conversation

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding
        assert return_tensors == "pt"
        self.texts = text
        self.images = images
        return {"input_ids": torch.arange(len(text)).unsqueeze(1)}


class ViewAwareHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        image_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch = image_hidden.shape[0]
        view_value = image_hidden.mean(dim=(1, 2)).reshape(batch, 1, 1)
        return self.scale * view_value.expand(batch, 256, 1024)


class CheckpointHead(nn.Module):
    def __init__(
        self,
        *,
        rows: int = 256,
        hidden_size: int = 2,
        dim_out: int = 1024,
    ) -> None:
        super().__init__()
        self.query_embs = nn.Parameter(torch.randn(rows, hidden_size))
        self.num_keyframes = 1
        self.num_latent_per_keyframe = 64
        self.num_backbone_tokens = 256
        self.dim_out = int(dim_out)


def valid_initialization_metadata() -> dict[str, Any]:
    return {
        "use_current_alignment": True,
        "independent_modality_task_tokens": True,
        "num_task_tokens": 64,
        "num_latent_per_keyframe": 64,
        "branch_latent_per_keyframe": 64,
        "num_keyframes": 1,
        "grid_size": 16,
        "semantic_dim": 1024,
        "target_len": 256,
        "target_tokens": 256,
        "video_target_type": "siglip2",
        "has_depth_head": True,
        "depth_grid_size": 16,
        "depth_feature_dim": 2048,
        "depth_target_type": "da3",
        "plan_head_type": "lingbot_dino",
        "sequence_length": 9,
        "keyframe_offsets": [8],
        "latent_len": 4 * 64,
        "total_unique_latent_per_keyframe": 4 * 64,
        "query_layout": (
            "current_dino_64_then_current_depth_64_then_"
            "future_dino_64_then_future_depth_64"
        ),
        "plan_token_ids": list(range(4 * 64)),
        "plan_token_strings": [f"<|sem_plan_{index}|>" for index in range(4 * 64)],
    }


def make_four_head_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    source = CheckpointHead()
    for filename in (
        "plan_head.pt",
        "depth_head.pt",
        "current_plan_head.pt",
        "current_depth_head.pt",
    ):
        torch.save(source.state_dict(), checkpoint / filename)
    (checkpoint / "planner_meta.json").write_text(
        json.dumps(valid_initialization_metadata()), encoding="utf-8"
    )
    return checkpoint


def make_fake_dual_camera_checkpoint_wrapper() -> PlannerWrapper:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.use_current_alignment = True
    wrapper.independent_modality_task_tokens = True
    wrapper.num_task_tokens = 64
    wrapper.num_latent_per_keyframe = 64
    wrapper.branch_latent_per_keyframe = 64
    wrapper.num_keyframes = 1
    wrapper.target_len = 256
    wrapper.latent_len = 4 * 64
    wrapper.num_camera_views = 2
    wrapper.use_depth = True
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.plan_token_ids = list(range(4 * 64))
    wrapper.plan_head = CheckpointHead()
    wrapper.depth_head = CheckpointHead(dim_out=2048)
    wrapper.current_plan_head = CheckpointHead()
    wrapper.current_depth_head = CheckpointHead(dim_out=2048)
    return wrapper


class SaveableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(512, 2)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True)


class SaveableProcessor:
    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True)


def make_saveable_dual_camera_wrapper() -> PlannerWrapper:
    wrapper = make_fake_dual_camera_checkpoint_wrapper()
    wrapper.model = SaveableModel()
    wrapper.plan_token_ids = list(range(4 * 64))
    wrapper.plan_embedding_injector = None
    wrapper.uses_pooled_head_query_embeddings = False
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.plan_head_num_heads = 16
    wrapper.plan_head_dropout = 0.0
    wrapper.num_keyframes = 1
    wrapper.num_latent_per_keyframe = 64
    wrapper.target_len = 256
    wrapper.sem_mlp_hidden_size = 0
    wrapper.mse_loss_weight = 1.0
    wrapper.cosine_loss_weight = 1.0
    wrapper.norm_loss_weight = 0.2
    wrapper.variance_loss_weight = 0.1
    wrapper.infonce_loss_weight = 0.1
    wrapper.infonce_temperature = 0.07
    wrapper.shared_latent_per_keyframe = 32
    wrapper.private_latent_per_keyframe = 32
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 256
    wrapper.depth_loss_weight = 0.004
    wrapper.bidirectional_plan_attn = True
    wrapper.current_dino_loss_weight = 0.004
    wrapper.future_dino_loss_weight = 0.004
    wrapper.current_depth_loss_weight = 0.004
    wrapper.future_depth_loss_weight = 0.004
    wrapper.da3_align_strategy = "last_layer"
    wrapper.da3_num_layers = 1
    wrapper.da3_layer_weights = None
    return wrapper


def make_checkpoint_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        fastwam_data_config=None,
        plan_label_dir=None,
        sample_feature_type="siglip2",
        num_keyframes=1,
        grid_size=16,
        semantic_dim=1024,
        model_path=tmp_path / "base-model",
        sample_one_window_per_stem=False,
        online_plan_labels=True,
        keyframe_scheme="even_future",
        keyframe_gamma=0.6,
        sequence_length=9,
        online_grid_size=16,
        siglip2_encoder_path=None,
        frame_ranges_json=None,
        train_plan_token_embedding=True,
        full_finetune=True,
        freeze_vision=True,
        freeze_lm_head=True,
        video_target_type="siglip2",
        depth_target_type="da3",
        depth_dim=2048,
        depth_grid_size=16,
    )


def valid_dual_camera_metadata() -> dict[str, Any]:
    return {
        "planner_input_layout": "separate_camera_images",
        "camera_names": ["main", "wrist"],
        "num_camera_views": 2,
        "camera_head_sharing": "shared_head_per_view_image_context",
        "semantic_output_layout": "batch_view_token_feature",
        "semantic_teacher": "siglip2-large-patch16-256",
        "future_keyframe_offsets": [8],
    }


def make_fake_alignment_wrapper(*, num_camera_views: int) -> PlannerWrapper:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.use_current_alignment = True
    wrapper.independent_modality_task_tokens = True
    wrapper.num_task_tokens = 64
    wrapper.latent_len = 4 * wrapper.num_task_tokens
    wrapper.num_camera_views = num_camera_views
    wrapper.da3_align_strategy = "last_layer"
    wrapper.current_plan_head = ViewAwareHead()
    wrapper.plan_head = ViewAwareHead()
    wrapper.current_depth_head = ViewAwareHead()
    wrapper.depth_head = ViewAwareHead()

    def forward_hiddens(**inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch = inputs["input_ids"].shape[0]
        if num_camera_views == 1:
            image_hidden = torch.zeros(batch, 2, 4)
        else:
            image_hidden = torch.stack(
                [torch.zeros(batch, 2, 4), torch.full((batch, 2, 4), 10.0)],
                dim=1,
            )
        task_hidden = torch.zeros(batch, wrapper.latent_len, 4)
        return image_hidden, task_hidden

    wrapper._forward_hiddens = forward_hiddens
    return wrapper


def make_loss_only_wrapper_with_unit_branch_weights() -> PlannerWrapper:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.da3_align_strategy = "last_layer"
    wrapper.current_dino_loss_weight = 1.0
    wrapper.future_dino_loss_weight = 1.0
    wrapper.current_depth_loss_weight = 1.0
    wrapper.future_depth_loss_weight = 1.0
    return wrapper


def make_four_branch_plans(
    *,
    main_value: float,
    wrist_value: float,
) -> dict[str, torch.Tensor]:
    values = torch.tensor([main_value, wrist_value]).reshape(1, 2, 1, 1)
    plans = values.expand(1, 2, 256, 4).clone()
    return {
        "current_dino": plans.clone(),
        "future_dino": plans.clone(),
        "current_depth": plans.clone(),
        "future_depth": plans.clone(),
    }


def test_ge_act_adapter_selects_current_and_future_endpoint_without_concat() -> None:
    video = torch.zeros(3, 2, 13, 2, 2)
    video[:, 0, 3].fill_(-0.5)
    video[:, 1, 3].fill_(0.0)
    video[:, 0, 12].fill_(0.5)
    video[:, 1, 12].fill_(1.0)
    wrapped = GEActDualCameraPlannerDataset(
        FakeDataset({"video": video, "caption": "pick the cup"}),
        n_previous=4,
        future_offset=8,
    )

    item = wrapped[0]

    assert item["current_camera_images"].shape == (2, 2, 2, 3)
    assert item["future_camera_images"].shape == (2, 2, 2, 3)
    torch.testing.assert_close(
        item["future_camera_images"][0],
        torch.full((2, 2, 3), 0.5),
    )
    torch.testing.assert_close(
        item["future_camera_images"][1],
        torch.full((2, 2, 3), 1.0),
    )
    assert item["images"][0].getpixel((0, 0))[0] == 64
    assert item["images"][1].getpixel((0, 0))[0] == 128
    assert item["prompt"] == "pick the cup"


def test_dual_camera_input_builder_flattens_images_main_then_wrist() -> None:
    processor = RecordingProcessor()
    main = Image.new("RGB", (2, 2), "red")
    wrist = Image.new("RGB", (2, 2), "blue")

    build_dual_camera_planner_inputs(
        processor,
        [(main, wrist)],
        ["pick"],
        ["<|sem_plan_0|>"],
    )

    assert processor.images == [main, wrist]
    assert processor.texts[0].count("<|image_pad|>") == 2
    assert "Main camera" in processor.rendered_conversations[0]
    assert "Wrist camera" in processor.rendered_conversations[0]


def test_dual_camera_collator_stacks_targets_and_keeps_sample_major_image_order() -> None:
    processor = RecordingProcessor()
    main_0 = Image.new("RGB", (2, 2), "red")
    wrist_0 = Image.new("RGB", (2, 2), "blue")
    main_1 = Image.new("RGB", (2, 2), "green")
    wrist_1 = Image.new("RGB", (2, 2), "yellow")
    batch = [
        {
            "images": (main_0, wrist_0),
            "current_camera_images": torch.full((2, 2, 2, 3), 1.0),
            "future_camera_images": torch.full((2, 2, 2, 3), 2.0),
            "prompt": "pick",
            "stem": "geact_000000000",
        },
        {
            "images": (main_1, wrist_1),
            "current_camera_images": torch.full((2, 2, 2, 3), 3.0),
            "future_camera_images": torch.full((2, 2, 2, 3), 4.0),
            "prompt": "place",
            "stem": "geact_000000001",
        },
    ]

    result = DualCameraPlannerCollator(
        processor=processor,
        plan_sequence=["<|sem_plan_0|>"],
    )(batch)

    assert processor.images == [main_0, wrist_0, main_1, wrist_1]
    assert result["current_camera_images"].shape == (2, 2, 2, 2, 3)
    assert result["future_camera_images"].shape == (2, 2, 2, 2, 3)
    assert result["current_camera_images"][:, 0, 0, 0, 0].tolist() == [1.0, 3.0]
    assert result["future_camera_images"][:, 1, 0, 0, 0].tolist() == [2.0, 4.0]
    assert result["stems"] == ["geact_000000000", "geact_000000001"]


def test_collect_image_hidden_keeps_two_contiguous_camera_spans() -> None:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    wrapper.image_token_id = 99
    hidden = torch.arange(1 * 10 * 3).reshape(1, 10, 3).float()
    input_ids = torch.tensor([[1, 99, 99, 2, 3, 99, 99, 4, 5, 6]])

    actual = wrapper.collect_image_hidden_by_view(hidden, input_ids, num_views=2)

    assert actual.shape == (1, 2, 2, 3)
    torch.testing.assert_close(actual[0, 0], hidden[0, 1:3])
    torch.testing.assert_close(actual[0, 1], hidden[0, 5:7])


def test_dual_camera_wrapper_reuses_four_query_groups_for_both_views() -> None:
    wrapper = make_fake_alignment_wrapper(num_camera_views=2)

    plans = wrapper.predict_current_future_plans(
        input_ids=torch.ones(2, 1, dtype=torch.long)
    )

    assert wrapper.latent_len == 4 * 64
    assert set(plans) == {
        "current_dino",
        "future_dino",
        "current_depth",
        "future_depth",
    }
    assert all(value.shape == (2, 2, 256, 1024) for value in plans.values())
    assert not torch.equal(plans["future_dino"][:, 0], plans["future_dino"][:, 1])


def test_single_camera_wrapper_keeps_legacy_output_shape() -> None:
    wrapper = make_fake_alignment_wrapper(num_camera_views=1)

    plans = wrapper.predict_current_future_plans(
        input_ids=torch.ones(2, 1, dtype=torch.long)
    )

    assert all(value.shape == (2, 256, 1024) for value in plans.values())


def test_dual_camera_loss_detects_swapped_teacher_views() -> None:
    wrapper = make_loss_only_wrapper_with_unit_branch_weights()
    plans = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    aligned = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    swapped = {name: value.flip(1) for name, value in aligned.items()}

    aligned_loss = wrapper.compute_current_future_losses(plans, aligned)["loss"]
    swapped_loss = wrapper.compute_current_future_losses(plans, swapped)["loss"]

    assert aligned_loss == 0
    assert swapped_loss > 0


def test_dual_camera_forward_validates_target_tokens_before_view_dimension() -> None:
    wrapper = make_loss_only_wrapper_with_unit_branch_weights()
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.use_current_alignment = True
    wrapper.target_len = 256
    plans = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    wrapper.predict_current_future_plans = lambda **_inputs: plans

    result = wrapper(
        semantic_plan_labels=plans["future_dino"],
        depth_plan_labels=plans["future_depth"],
        current_dino_labels=plans["current_dino"],
        current_depth_labels=plans["current_depth"],
    )

    assert result["loss"] == 0


def test_wrapper_rejects_unsupported_camera_view_count() -> None:
    with pytest.raises(ValueError, match="num_camera_views must be 1 or 2"):
        PlannerWrapper(
            model=nn.Module(),
            hidden_size=4,
            semantic_dim=2,
            plan_token_ids=[1],
            target_len=1,
            num_keyframes=1,
            grid_size=1,
            num_camera_views=3,
        )


def test_legacy_checkpoint_initializes_four_shared_heads_without_expansion(
    tmp_path: Path,
) -> None:
    source = make_four_head_checkpoint(tmp_path)
    wrapper = make_fake_dual_camera_checkpoint_wrapper()

    report = planner.load_planner_initialization(wrapper, source)

    assert report["loaded_heads"] == [
        "plan_head",
        "depth_head",
        "current_plan_head",
        "current_depth_head",
    ]
    assert wrapper.plan_head.query_embs.shape == (
        256,
        wrapper.plan_head.query_embs.shape[1],
    )
    assert wrapper.latent_len == 256


def test_legacy_checkpoint_rejects_missing_head_file(tmp_path: Path) -> None:
    source = make_four_head_checkpoint(tmp_path)
    (source / "current_depth_head.pt").unlink()

    with pytest.raises(FileNotFoundError, match="current_depth_head.pt"):
        planner.load_planner_initialization(
            make_fake_dual_camera_checkpoint_wrapper(), source
        )


def test_legacy_checkpoint_rejects_non_independent_64_token_metadata(
    tmp_path: Path,
) -> None:
    source = make_four_head_checkpoint(tmp_path)
    metadata = valid_initialization_metadata()
    metadata["independent_modality_task_tokens"] = False
    (source / "planner_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="independent_modality_task_tokens"):
        planner.load_planner_initialization(
            make_fake_dual_camera_checkpoint_wrapper(), source
        )


def test_legacy_checkpoint_rejects_reordered_plan_token_ids(tmp_path: Path) -> None:
    source = make_four_head_checkpoint(tmp_path)
    metadata = valid_initialization_metadata()
    metadata["plan_token_ids"] = metadata["plan_token_ids"][1:] + [0]
    (source / "planner_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="plan_token_ids"):
        planner.load_planner_initialization(
            make_fake_dual_camera_checkpoint_wrapper(), source
        )


@pytest.mark.parametrize(
    ("field", "corrupted"),
    [
        ("num_keyframes", 2),
        ("grid_size", 8),
        ("semantic_dim", 512),
        ("target_len", 128),
        ("target_tokens", 128),
        ("video_target_type", "dinov3"),
        ("has_depth_head", False),
        ("depth_grid_size", 8),
        ("depth_feature_dim", 1024),
        ("depth_target_type", "morgbd"),
        ("sequence_length", 8),
        ("keyframe_offsets", [7]),
    ],
)
def test_legacy_checkpoint_rejects_corrupted_output_geometry_metadata(
    tmp_path: Path,
    field: str,
    corrupted: Any,
) -> None:
    source = make_four_head_checkpoint(tmp_path)
    metadata = valid_initialization_metadata()
    metadata[field] = corrupted
    (source / "planner_meta.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=field):
        planner.load_planner_initialization(
            make_fake_dual_camera_checkpoint_wrapper(), source
        )


def test_legacy_checkpoint_rejects_incompatible_wrapper_target_geometry(
    tmp_path: Path,
) -> None:
    source = make_four_head_checkpoint(tmp_path)
    wrapper = make_fake_dual_camera_checkpoint_wrapper()
    wrapper.target_len = 128

    with pytest.raises(ValueError, match="target_len"):
        planner.load_planner_initialization(wrapper, source)


def test_legacy_checkpoint_rejects_incompatible_head_output_geometry(
    tmp_path: Path,
) -> None:
    source = make_four_head_checkpoint(tmp_path)
    wrapper = make_fake_dual_camera_checkpoint_wrapper()
    wrapper.current_depth_head.dim_out = 1024

    with pytest.raises(ValueError, match="current_depth_head.dim_out"):
        planner.load_planner_initialization(wrapper, source)


def test_legacy_checkpoint_strictly_rejects_head_shape_mismatch(
    tmp_path: Path,
) -> None:
    source = make_four_head_checkpoint(tmp_path)
    torch.save(
        CheckpointHead(rows=257).state_dict(),
        source / "plan_head.pt",
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        planner.load_planner_initialization(
            make_fake_dual_camera_checkpoint_wrapper(), source
        )


def test_dual_camera_metadata_rejects_legacy_composite_inference() -> None:
    metadata = valid_dual_camera_metadata()
    metadata["planner_input_layout"] = "fastwam_current_multicamera_composite"

    with pytest.raises(ValueError, match="separate_camera_images"):
        planner.validate_dual_camera_export_metadata(metadata)


def test_dual_camera_metadata_accepts_exact_separate_view_contract() -> None:
    metadata = valid_dual_camera_metadata()

    assert planner.validate_dual_camera_export_metadata(metadata) == metadata


def test_dual_camera_exported_checkpoint_rejects_composite_before_model_build(
    tmp_path: Path,
) -> None:
    metadata = valid_dual_camera_metadata()
    metadata["planner_input_layout"] = "fastwam_current_multicamera_composite"
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))
    )

    with pytest.raises(ValueError, match="separate_camera_images"):
        PlannerWrapper.from_exported_checkpoint(
            model=model,
            checkpoint_dir=tmp_path,
            metadata=metadata,
        )


def test_dual_camera_checkpoint_exports_exact_separate_view_metadata(
    tmp_path: Path,
) -> None:
    planner.save_checkpoint(
        tmp_path,
        1,
        make_saveable_dual_camera_wrapper(),
        SaveableProcessor(),
        make_checkpoint_args(tmp_path),
        rank=0,
    )

    metadata = json.loads(
        (tmp_path / "step_000001" / "planner_meta.json").read_text(
            encoding="utf-8"
        )
    )
    for field, expected in valid_dual_camera_metadata().items():
        assert metadata[field] == expected
    assert metadata["planner_input_frame"] == "separate_camera_images"
    planner.validate_dual_camera_export_metadata(metadata)


def test_single_camera_checkpoint_keeps_legacy_input_metadata(tmp_path: Path) -> None:
    wrapper = make_saveable_dual_camera_wrapper()
    wrapper.num_camera_views = 1

    planner.save_checkpoint(
        tmp_path,
        1,
        wrapper,
        SaveableProcessor(),
        make_checkpoint_args(tmp_path),
        rank=0,
    )

    metadata = json.loads(
        (tmp_path / "step_000001" / "planner_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["planner_input_frame"] == "legacy_single_current_frame"
    assert "planner_input_layout" not in metadata


def test_init_checkpoint_cli_does_not_require_separate_model_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "planner",
            "--init-planner-checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    args = planner.parse_args()

    assert args.init_planner_checkpoint == checkpoint
    assert args.model_path is None


def test_qwen_loader_can_read_processor_from_separate_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Path] = {}

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(padding_side="right")

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **_kwargs: Any) -> FakeProcessor:
            calls["processor"] = Path(path)
            return FakeProcessor()

    class FakeQwenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=2)
            )

        @classmethod
        def from_pretrained(cls, path: str, **_kwargs: Any) -> "FakeQwenModel":
            calls["model"] = Path(path)
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeAutoProcessor,
            Qwen3VLForConditionalGeneration=FakeQwenModel,
        ),
    )
    model_dir = tmp_path / "model"
    processor_dir = tmp_path / "processor"

    _, processor = qwen_helper.load_qwen3vl_model_and_processor(
        model_dir,
        processor_path=processor_dir,
        dtype="fp32",
    )

    assert calls == {"model": model_dir, "processor": processor_dir}
    assert processor.tokenizer.padding_side == "left"


def test_qwen_loader_defaults_processor_to_model_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    class FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(padding_side="right")

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **_kwargs: Any) -> FakeProcessor:
            calls.append(Path(path))
            return FakeProcessor()

    class FakeQwenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=2)
            )

        @classmethod
        def from_pretrained(cls, path: str, **_kwargs: Any) -> "FakeQwenModel":
            calls.append(Path(path))
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeAutoProcessor,
            Qwen3VLForConditionalGeneration=FakeQwenModel,
        ),
    )
    model_dir = tmp_path / "model"

    qwen_helper.load_qwen3vl_model_and_processor(model_dir, dtype="fp32")

    assert calls == [model_dir, model_dir]
