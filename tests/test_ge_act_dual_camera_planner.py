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
from qwen3_vl_semantic_planner.dinov3_da3_2b.depth_anything3_target import (
    DepthAnything3TargetEncoder,
)

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


class K4ViewAwareHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        image_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
    ) -> torch.Tensor:
        assert task_hidden.shape[1] == 4 * 64
        batch = image_hidden.shape[0]
        view_value = image_hidden.mean(dim=(1, 2)).reshape(batch, 1, 1)
        return self.scale * view_value.expand(batch, 4 * 256, self.feature_dim)


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


class CameraValueTeacher:
    def __init__(self, *, tokens: int, feature_dim: int) -> None:
        self.tokens = tokens
        self.feature_dim = feature_dim
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def encode_current_and_future(
        self,
        current: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inputs = (current.clone(), future.clone())

        def encode(frames: torch.Tensor) -> torch.Tensor:
            values = frames.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
            return values.expand(-1, self.tokens, self.feature_dim).clone()

        return encode(current), encode(future)


class FutureAppearanceTeacher:
    def __init__(self, *, tokens: int = 256, feature_dim: int = 1024) -> None:
        self.tokens = tokens
        self.feature_dim = feature_dim
        self.current: torch.Tensor | None = None
        self.keyframes: list[torch.Tensor] = []

    def encode_future_keyframes(
        self,
        current: torch.Tensor,
        keyframes: list[torch.Tensor],
        effective_fps: Any = None,
    ) -> torch.Tensor:
        assert effective_fps is None
        self.current = current.clone()
        self.keyframes = [frame.clone() for frame in keyframes]
        encoded = []
        for frame in keyframes:
            values = frame.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)
            encoded.append(
                values.expand(-1, self.tokens, self.feature_dim).clone()
            )
        return torch.cat(encoded, dim=1)


class FutureDepthTeacher:
    align_strategy = "wsa_multilayer"
    num_layers = 4

    def __init__(self, *, tokens: int = 256, feature_dim: int = 8) -> None:
        self.tokens = tokens
        self.feature_dim = feature_dim
        self.keyframes: list[torch.Tensor] = []

    def encode_future_keyframes(
        self,
        keyframes: list[torch.Tensor],
    ) -> torch.Tensor:
        self.keyframes = [frame.clone() for frame in keyframes]
        per_keyframe = []
        for frame in keyframes:
            values = frame.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
            per_keyframe.append(
                values.expand(
                    -1,
                    self.num_layers,
                    self.tokens,
                    self.feature_dim,
                ).clone()
            )
        return torch.cat(per_keyframe, dim=2)


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


def make_fake_future_k4_wrapper() -> PlannerWrapper:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.use_current_alignment = False
    wrapper.num_keyframes = 4
    wrapper.num_camera_views = 2
    wrapper.shared_latent_per_keyframe = 32
    wrapper.private_latent_per_keyframe = 32
    wrapper.branch_latent_per_keyframe = 64
    wrapper.total_unique_latent_per_keyframe = 96
    wrapper.latent_len = 4 * 96
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.da3_align_strategy = "wsa_multilayer"
    wrapper.da3_num_layers = 4
    wrapper.depth_per_layer_dim = 8
    wrapper.plan_head = K4ViewAwareHead(feature_dim=8)
    wrapper.depth_head = K4ViewAwareHead(feature_dim=4 * 8)

    def forward_hiddens(**inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch = inputs["input_ids"].shape[0]
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


def test_dual_camera_k4_dataset_preserves_view_then_keyframe_order() -> None:
    video = torch.empty(3, 2, 13, 2, 2)
    for view in range(2):
        for frame in range(13):
            video[:, view, frame].fill_(-0.9 + view * 0.4 + frame * 0.05)
    wrapped = GEActDualCameraPlannerDataset(
        FakeDataset({"video": video, "caption": "pick the cup"}),
        n_previous=4,
        future_offsets=(2, 4, 6, 8),
    )

    item = wrapped[0]

    assert item["future_camera_images"].shape == (2, 4, 2, 2, 3)
    torch.testing.assert_close(
        item["future_camera_images"][:, :, 0, 0, 0],
        torch.tensor(
            [
                [-0.6, -0.5, -0.4, -0.3],
                [-0.2, -0.1, 0.0, 0.1],
            ]
        ),
    )
    assert wrapped.future_offsets == (2, 4, 6, 8)


@pytest.mark.parametrize(
    "offsets",
    [(), (0, 2, 4, 8), (-1, 2, 4, 8), (2, 2, 4, 8), (4, 2, 6, 8)],
)
def test_dual_camera_k4_dataset_rejects_invalid_future_offsets(
    offsets: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="strictly increasing positive"):
        GEActDualCameraPlannerDataset(
            FakeDataset({"video": torch.zeros(3, 2, 13, 2, 2)}),
            future_offsets=offsets,
        )


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


def test_future_only_k4_wsa_wrapper_uses_384_queries() -> None:
    wrapper = PlannerWrapper(
        model=nn.Module(),
        hidden_size=8,
        semantic_dim=8,
        plan_token_ids=list(range(384)),
        target_len=4 * 256,
        num_keyframes=4,
        grid_size=16,
        plan_head_type="lingbot_dino",
        use_depth=True,
        depth_dim=8,
        depth_grid_size=16,
        shared_latent_per_keyframe=32,
        private_latent_per_keyframe=32,
        use_current_alignment=False,
        da3_align_strategy="wsa_multilayer",
        da3_num_layers=4,
        num_camera_views=2,
    )

    assert wrapper.latent_len == 384
    assert wrapper.num_latent_per_keyframe == 64
    assert wrapper.plan_head is not wrapper.depth_head


def test_future_only_k4_wsa_prediction_preserves_views_and_layers() -> None:
    wrapper = make_fake_future_k4_wrapper()

    dino, depth = wrapper.predict_dino_depth_plan(
        input_ids=torch.ones(2, 1, dtype=torch.long)
    )

    assert dino.shape == (2, 2, 4 * 256, 8)
    assert depth.shape == (2, 2, 4 * 256, 4, 8)
    assert not torch.equal(dino[:, 0], dino[:, 1])
    assert not torch.equal(depth[:, 0], depth[:, 1])
    torch.testing.assert_close(
        wrapper.predict_semantic_plan(input_ids=torch.ones(2, 1, dtype=torch.long)),
        dino,
    )


def test_wsa_depth_target_reshape_preserves_dual_camera_dimension() -> None:
    wrapper = make_fake_future_k4_wrapper()
    target = torch.arange(2 * 2 * 4 * 1024 * 8).reshape(2, 2, 4, 1024, 8)

    reshaped = wrapper._reshape_depth_target(target)

    assert reshaped.shape == (2, 2, 1024, 4, 8)
    torch.testing.assert_close(reshaped[:, :, 0, 3], target[:, :, 3, 0])


def test_da3_wsa_future_keyframes_concatenate_time_inside_each_layer() -> None:
    encoder = DepthAnything3TargetEncoder.__new__(DepthAnything3TargetEncoder)
    nn.Module.__init__(encoder)
    encoder.align_strategy = "wsa_multilayer"
    encoder._prep = lambda frames: frames

    def patch_tokens(batch: torch.Tensor) -> torch.Tensor:
        values = batch[:, 0, 0, 0].reshape(-1, 1, 1, 1)
        layer_offsets = torch.arange(4).reshape(1, 4, 1, 1) * 100
        token_offsets = torch.arange(2).reshape(1, 1, 2, 1) * 10
        return (values + layer_offsets + token_offsets).expand(-1, -1, -1, 3)

    encoder._patch_tokens = patch_tokens
    keyframes = [
        torch.full((2, 3, 1, 1), float(keyframe * 10 + batch))
        for keyframe in range(4)
        for batch in [0]
    ]
    for keyframe, frames in enumerate(keyframes):
        frames[1].fill_(float(keyframe * 10 + 1))

    encoded = encoder.encode_future_keyframes(keyframes).float()

    assert encoded.shape == (2, 4, 4 * 2, 3)
    assert encoded[0, 0, :, 0].tolist() == [0, 10, 10, 20, 20, 30, 30, 40]
    torch.testing.assert_close(
        encoded[1, 3, :, 0],
        torch.tensor([301, 311, 311, 321, 321, 331, 331, 341], dtype=torch.float32),
        rtol=0,
        atol=1,
    )


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


@pytest.mark.parametrize(
    ("dataset_root", "fastwam_config", "ge_act_config"),
    [
        (None, None, None),
        (Path("legacy"), Path("fastwam.yaml"), None),
        (Path("legacy"), None, Path("ge_act.yaml")),
        (None, Path("fastwam.yaml"), Path("ge_act.yaml")),
    ],
)
def test_dataset_selection_requires_exactly_one_source(
    dataset_root: Path | None,
    fastwam_config: Path | None,
    ge_act_config: Path | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        planner.validate_dataset_source_selection(
            dataset_root=dataset_root,
            fastwam_data_config=fastwam_config,
            ge_act_data_config=ge_act_config,
        )


def test_dataset_selection_accepts_ge_act_config_without_loading_models() -> None:
    assert (
        planner.validate_dataset_source_selection(
            dataset_root=None,
            fastwam_data_config=None,
            ge_act_data_config=Path("ge_act.yaml"),
        )
        == "ge_act"
    )


def test_dataset_selection_cli_accepts_ge_act_config_without_loading_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "ge_act.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "planner",
            "--model-path",
            str(tmp_path / "model"),
            "--ge-act-data-config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    args = planner.parse_args()

    assert args.ge_act_data_config == config_path
    assert args.dataset_root is None
    assert args.fastwam_data_config is None


def test_ge_act_dataset_selection_loads_configured_train_dataset(
    tmp_path: Path,
) -> None:
    dataset_module = tmp_path / "fake_ge_act_dataset.py"
    dataset_module.write_text(
        "from pathlib import Path\n"
        "class FakeGEActDataset:\n"
        "    def __init__(self, marker, **kwargs):\n"
        "        self.marker = marker\n"
        "        self.constructor_cwd = Path.cwd()\n"
        "    def __len__(self):\n"
        "        return 1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "ge_act.yaml"
    config_path.write_text(
        f"train_data_class_path: {dataset_module}\n"
        "train_data_class: FakeGEActDataset\n"
        "data:\n"
        "  train:\n"
        "    marker: configured-train-split\n"
        "    valid_cam: [observation.images.image, observation.images.wrist_image]\n"
        "    source_fps: 20\n"
        "    chunk: 9\n"
        "    action_chunk: 36\n"
        "    n_previous: 4\n"
        "    ignore_seek: false\n",
        encoding="utf-8",
    )

    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)

    dataset = planner.load_ge_act_dual_camera_planner_dataset(config_path)

    assert isinstance(dataset, GEActDualCameraPlannerDataset)
    assert dataset.dataset.marker == "configured-train-split"
    assert dataset.dataset.constructor_cwd == PLANNER_ROOT.parent / "ge_act"
    assert dataset.n_previous == 4
    assert dataset.future_offset == 8
    assert Path.cwd() == original_cwd
    assert sys.path == original_sys_path


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "valid_cam: [observation.images.wrist_image, observation.images.image]",
            "valid_cam",
        ),
        ("source_fps: 10", "source_fps"),
        ("chunk: 13", "chunk"),
        ("action_chunk: 54", "action_chunk"),
        ("n_previous: 3", "n_previous"),
        ("ignore_seek: true", "ignore_seek"),
    ],
)
def test_ge_act_dataset_rejects_incompatible_camera_or_temporal_contract(
    tmp_path: Path,
    override: str,
    message: str,
) -> None:
    field = override.split(":", 1)[0]
    contract = {
        "valid_cam": (
            "valid_cam: [observation.images.image, observation.images.wrist_image]"
        ),
        "source_fps": "source_fps: 20",
        "chunk": "chunk: 9",
        "action_chunk": "action_chunk: 36",
        "n_previous": "n_previous: 4",
        "ignore_seek": "ignore_seek: false",
    }
    contract[field] = override
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "train_data_class_path: unused.py\n"
        "train_data_class: Unused\n"
        "data:\n"
        "  train:\n"
        + "".join(f"    {line}\n" for line in contract.values()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        planner.load_ge_act_dual_camera_planner_dataset(config_path)


def test_ge_act_dataset_restores_process_state_when_constructor_fails(
    tmp_path: Path,
) -> None:
    dataset_module = tmp_path / "broken_ge_act_dataset.py"
    dataset_module.write_text(
        "class BrokenGEActDataset:\n"
        "    def __init__(self, **kwargs):\n"
        "        raise RuntimeError('constructor failed')\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "broken.yaml"
    config_path.write_text(
        f"train_data_class_path: {dataset_module}\n"
        "train_data_class: BrokenGEActDataset\n"
        "data:\n"
        "  train:\n"
        "    valid_cam: [observation.images.image, observation.images.wrist_image]\n"
        "    source_fps: 20\n"
        "    chunk: 9\n"
        "    action_chunk: 36\n"
        "    n_previous: 4\n",
        encoding="utf-8",
    )
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)

    with pytest.raises(RuntimeError, match="constructor failed"):
        planner.load_ge_act_dual_camera_planner_dataset(config_path)

    assert Path.cwd() == original_cwd
    assert sys.path == original_sys_path


def test_flatten_two_camera_frames_for_online_teachers_preserves_order() -> None:
    frames = torch.zeros(2, 2, 4, 4, 3, dtype=torch.uint8)
    frames[:, 0].fill_(10)
    frames[:, 1].fill_(20)

    flat = planner.flatten_camera_teacher_frames(frames)

    assert flat.shape == (4, 3, 4, 4)
    assert flat[0].float().mean() == 10
    assert flat[1].float().mean() == 20
    assert flat[2].float().mean() == 10
    assert flat[3].float().mean() == 20


def test_teacher_features_restore_batch_view_layout() -> None:
    encoded = torch.arange(4 * 256 * 1024).reshape(4, 256, 1024)

    restored = planner.restore_camera_teacher_features(
        encoded,
        batch_size=2,
        num_views=2,
    )

    assert restored.shape == (2, 2, 256, 1024)
    torch.testing.assert_close(restored[0, 1], encoded[1])


def test_encode_dual_camera_teachers_preserves_current_future_and_view_order() -> None:
    current = torch.empty(2, 2, 4, 4, 3)
    future = torch.empty_like(current)
    current[0, 0].fill_(-1.0)
    current[0, 1].fill_(-0.5)
    current[1, 0].fill_(0.0)
    current[1, 1].fill_(0.5)
    future[0, 0].fill_(-0.75)
    future[0, 1].fill_(-0.25)
    future[1, 0].fill_(0.25)
    future[1, 1].fill_(1.0)
    appearance = CameraValueTeacher(tokens=256, feature_dim=1024)
    depth = CameraValueTeacher(tokens=256, feature_dim=2048)

    labels = planner.encode_dual_camera_teacher_targets(
        current,
        future,
        appearance_encoder=appearance,
        depth_encoder=depth,
    )

    assert labels["current_dino_labels"].shape == (2, 2, 256, 1024)
    assert labels["semantic_plan_labels"].shape == (2, 2, 256, 1024)
    assert labels["current_depth_labels"].shape == (2, 2, 256, 2048)
    assert labels["depth_plan_labels"].shape == (2, 2, 256, 2048)
    assert labels["current_dino_labels"][:, :, 0, 0].tolist() == [
        [0.0, 0.25],
        [0.5, 0.75],
    ]
    assert labels["semantic_plan_labels"][:, :, 0, 0].tolist() == [
        [0.125, 0.375],
        [0.625, 1.0],
    ]
    assert appearance.inputs is not None
    assert appearance.inputs[0].shape == (4, 3, 4, 4)


def test_encode_dual_camera_k4_future_targets_preserves_view_and_time() -> None:
    current = torch.zeros(2, 2, 4, 4, 3)
    future = torch.empty(2, 2, 4, 4, 4, 3)
    for batch in range(2):
        for view in range(2):
            for keyframe in range(4):
                future[batch, view, keyframe].fill_(
                    -1.0 + 0.1 * (8 * batch + 4 * view + keyframe)
                )
    appearance = FutureAppearanceTeacher()
    depth = FutureDepthTeacher()

    labels = planner.encode_dual_camera_future_targets(
        current,
        future,
        appearance_encoder=appearance,
        depth_encoder=depth,
    )

    assert labels["semantic_plan_labels"].shape == (2, 2, 4 * 256, 1024)
    assert labels["depth_plan_labels"].shape == (2, 2, 4, 4 * 256, 8)
    torch.testing.assert_close(
        labels["semantic_plan_labels"][0, :, ::256, 0],
        torch.tensor(
            [
                [0.0, 0.05, 0.1, 0.15],
                [0.2, 0.25, 0.3, 0.35],
            ]
        ),
    )
    assert appearance.current is not None
    assert appearance.current.shape == (4, 3, 4, 4)
    assert len(appearance.keyframes) == 4
    assert all(frame.shape == (4, 3, 4, 4) for frame in appearance.keyframes)
    assert len(depth.keyframes) == 4


@pytest.mark.parametrize(
    ("current_shape", "future_shape"),
    [
        ((2, 1, 4, 4, 3), (2, 1, 4, 4, 3)),
        ((2, 2, 4, 4, 3), (2, 2, 5, 4, 3)),
    ],
)
def test_encode_dual_camera_teachers_rejects_bad_camera_batches(
    current_shape: tuple[int, ...],
    future_shape: tuple[int, ...],
) -> None:
    appearance = CameraValueTeacher(tokens=256, feature_dim=1024)
    depth = CameraValueTeacher(tokens=256, feature_dim=2048)

    with pytest.raises(ValueError, match=r"\[B,2,H,W,3\]|same shape"):
        planner.encode_dual_camera_teacher_targets(
            torch.zeros(current_shape),
            torch.zeros(future_shape),
            appearance_encoder=appearance,
            depth_encoder=depth,
        )


def test_encode_dual_camera_teachers_rejects_wrong_teacher_shape() -> None:
    appearance = CameraValueTeacher(tokens=81, feature_dim=1024)
    depth = CameraValueTeacher(tokens=256, feature_dim=2048)

    with pytest.raises(ValueError, match="appearance current.*256, 1024"):
        planner.encode_dual_camera_teacher_targets(
            torch.zeros(1, 2, 4, 4, 3),
            torch.zeros(1, 2, 4, 4, 3),
            appearance_encoder=appearance,
            depth_encoder=depth,
        )


def test_ge_act_launchers_record_dual_camera_training_contract() -> None:
    legacy_launcher = (
        PLANNER_ROOT / "lingbot_dino_4b" / "train_lingbot_dino_4b.sh"
    ).read_text(encoding="utf-8")
    ge_act_launcher = (
        PLANNER_ROOT
        / "dinov3_da3_2b"
        / "train_ge_act_dual_camera_siglip2da3.sh"
    ).read_text(encoding="utf-8")

    assert "GE_ACT_DATA_CONFIG" in legacy_launcher
    assert "--ge-act-data-config" in legacy_launcher
    assert "INIT_PLANNER_CHECKPOINT" in legacy_launcher
    assert "--init-planner-checkpoint" in legacy_launcher
    assert "NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-64}" in legacy_launcher
    assert "MAX_STEPS=${MAX_STEPS:-30000}" in ge_act_launcher
    assert "SIGLIP2_INPUT_SIZE=${SIGLIP2_INPUT_SIZE:-256}" in ge_act_launcher
    assert "DA3_ALIGN_STRATEGY=${DA3_ALIGN_STRATEGY:-last_layer}" in ge_act_launcher
    assert "FULL_FINETUNE=${FULL_FINETUNE:-1}" in ge_act_launcher
    assert "FUTURE_KEYFRAME_OFFSET=${FUTURE_KEYFRAME_OFFSET:-8}" in ge_act_launcher


def test_ola_k4_config_and_launcher_are_fresh_and_fail_closed() -> None:
    import yaml

    config_path = (
        PLANNER_ROOT.parent
        / "ge_act"
        / "configs"
        / "ltx_model"
        / "libero"
        / "planner_data_libero_fastwam_ola.yaml"
    )
    launcher_path = (
        PLANNER_ROOT
        / "dinov3_da3_2b"
        / "train_ge_act_dual_camera_k4_siglip2da3_ola.sh"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = config["data"]["train"]
    launcher = launcher_path.read_text(encoding="utf-8")

    assert config["train_data_class"] == "CustomLeRobotDataset"
    assert train["valid_cam"] == [
        "observation.images.image",
        "observation.images.wrist_image",
    ]
    assert train["source_fps"] == 20
    assert train["chunk"] == 9
    assert train["n_previous"] == 4
    assert train["sample_size"] == [256, 256]
    assert train["require_predecoded"] is False
    assert len(train["domains"]) == 4
    assert all(
        root == "/data/shared/datasets/libero_fastwam"
        for root in train["data_roots"]
    )
    for required in (
        "NUM_KEYFRAMES=${NUM_KEYFRAMES:-4}",
        "USE_CURRENT_ALIGNMENT=${USE_CURRENT_ALIGNMENT:-0}",
        "BATCH_SIZE=${BATCH_SIZE:-8}",
        "GRAD_ACCUM=${GRAD_ACCUM:-2}",
        "EXPECTED_GLOBAL_BATCH=${EXPECTED_GLOBAL_BATCH:-128}",
        "MAX_STEPS=${MAX_STEPS:-30000}",
        "SAVE_STEPS=${SAVE_STEPS:-5000}",
        "SAVE_START_STEP=${SAVE_START_STEP:-20000}",
        "LR=${LR:-3e-5}",
        "HEAD_LR=${HEAD_LR:-3e-4}",
        "WARMUP_STEPS=${WARMUP_STEPS:-2500}",
        "DA3_ALIGN_STRATEGY=${DA3_ALIGN_STRATEGY:-wsa_multilayer}",
        "INIT_PLANNER_CHECKPOINT=",
        "HEAD_WARMSTART_CKPT=",
        'export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
    ):
        assert required in launcher
    assert "step_020000" not in launcher


def test_ge_act_k4_loader_uses_all_four_future_offsets(tmp_path: Path) -> None:
    dataset_module = tmp_path / "fake_ge_act_k4_dataset.py"
    dataset_module.write_text(
        "class FakeGEActDataset:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def __len__(self): return 1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "ge_act_k4.yaml"
    config_path.write_text(
        f"train_data_class_path: {dataset_module}\n"
        "train_data_class: FakeGEActDataset\n"
        "data:\n"
        "  train:\n"
        "    valid_cam: [observation.images.image, observation.images.wrist_image]\n"
        "    source_fps: 20\n"
        "    chunk: 9\n"
        "    action_chunk: 36\n"
        "    n_previous: 4\n"
        "    ignore_seek: false\n",
        encoding="utf-8",
    )

    dataset = planner.load_ge_act_dual_camera_planner_dataset(
        config_path,
        future_offsets=(2, 4, 6, 8),
    )

    assert dataset.future_offsets == (2, 4, 6, 8)


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


def test_dual_camera_k4_metadata_is_geometry_derived_and_strict() -> None:
    metadata = planner.build_dual_camera_export_metadata(
        future_keyframe_offsets=(2, 4, 6, 8),
        num_keyframes=4,
        target_tokens_per_keyframe=256,
        planner_token_count=384,
    )

    assert metadata == {
        "planner_input_layout": "separate_camera_images",
        "camera_names": ["main", "wrist"],
        "num_camera_views": 2,
        "camera_head_sharing": "shared_head_per_view_image_context",
        "semantic_output_layout": "batch_view_keyframe_token_feature",
        "semantic_teacher": "siglip2-large-patch16-256",
        "future_keyframe_offsets": [2, 4, 6, 8],
        "num_keyframes": 4,
        "target_tokens_per_keyframe": 256,
        "planner_token_count": 384,
    }
    assert planner.validate_dual_camera_export_metadata(metadata) == metadata

    corrupted = dict(metadata)
    corrupted["planner_token_count"] = 256
    with pytest.raises(ValueError, match="planner_token_count"):
        planner.validate_dual_camera_export_metadata(corrupted)


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
        def from_pretrained(cls, path: str, **kwargs: Any) -> "FakeQwenModel":
            calls["model"] = Path(path)
            calls["model_kwargs"] = kwargs
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

    assert calls["model"] == model_dir
    assert calls["processor"] == processor_dir
    assert calls["model_kwargs"]["torch_dtype"] is torch.float32
    assert "dtype" not in calls["model_kwargs"]
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
