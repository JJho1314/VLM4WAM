"""Research opt-in: GE-Act loss trains the baton planner head (joint KI)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import sys

import pytest
from safetensors.torch import save_file
import torch
import torch.nn as nn

from qwen35_baton.model import BatonQwen35Planner
from qwen35_baton.provider import FrozenBatonPlanner
from qwen35_baton.query_tower import BatonVisualAlignmentTower
from qwen35_baton.research_provider import (
    ResearchPlannerHead,
    predict_with_research_head,
    research_plan_states,
)
from qwen35_baton.sequence import ADDED_TOKENS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPOSITORY_ROOT / "ge_act"
for path in (REPOSITORY_ROOT, GE_ACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runner import ge_trainer  # noqa: E402


ADDED_TOKEN_IDS = tuple(range(100, 107))
PLAN_PAD_ID = ADDED_TOKEN_IDS[5]
WIDTH = 8
ENV = (
    "BATON_RESEARCH_PLANNER_PROB",
    "BATON_RESEARCH_PLANNER_JOINT",
    "BATON_RESEARCH_PLANNER_ANCHOR",
    "BATON_RESEARCH_PLANNER_HEAD_LR",
)


class _Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return dict(zip(ADDED_TOKENS, ADDED_TOKEN_IDS, strict=True))[token]


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, Any]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        return str(messages[0]["content"][1]["text"])

    def __call__(
        self, *, text: Sequence[str], images: Sequence[torch.Tensor], return_tensors: str, padding: bool
    ) -> dict[str, torch.Tensor]:
        image_code = int(images[0][0, 0, 0].item()) % 50 + 2
        input_ids = torch.tensor([[image_code, 1, *([PLAN_PAD_ID] * 1024)]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class _TinyLanguageModel(nn.Module):
    def __init__(self, embedding: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = embedding
        self.mix = nn.Linear(WIDTH, WIDTH)


class _TinyBase(nn.Module):
    def __init__(self, embedding: nn.Module) -> None:
        super().__init__()
        self.language_model = _TinyLanguageModel(embedding)
        self.grad_enabled: list[bool] = []

    def forward(self, *, input_ids, attention_mask=None, use_cache, output_hidden_states, return_dict):
        self.grad_enabled.append(torch.is_grad_enabled())
        hidden = self.language_model.embed_tokens(input_ids)
        hidden = self.language_model.mix(hidden + input_ids[:, :1, None].to(hidden.dtype))
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBase(nn.Embedding(128, WIDTH))

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, embedding: nn.Module) -> None:
        self.model.language_model.embed_tokens = embedding


def _provider(seed: int = 0) -> FrozenBatonPlanner:
    torch.manual_seed(seed)
    tower = BatonVisualAlignmentTower._from_test_config(
        qwen_dim=WIDTH, num_frames=4, tokens_per_frame=256, num_heads=2
    )
    planner = BatonQwen35Planner(_TinyQwen(), added_token_ids=ADDED_TOKEN_IDS, query_tower=tower)
    return FrozenBatonPlanner(planner=planner, processor=_Processor(), added_token_ids=ADDED_TOKEN_IDS)


class _TinyLTX(nn.Module):
    """Stands in for the LTX transformer: semantic tokens -> one scalar loss."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(4, 16)
        self.semantic_proj = nn.Linear(1024, 1)
        self.action_out = nn.Linear(16, 4)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.semantic_proj(tokens.float()).square().mean()


def _images(batch_size: int) -> torch.Tensor:
    images = torch.zeros(batch_size, 2, 3, 8, 8, dtype=torch.uint8)
    images[:, 0, 0, 0, 0] = 3
    images[:, 1, 0, 0, 0] = 7
    return images


def _video(batch_size: int, n_previous: int = 2) -> torch.Tensor:
    # [B,3,2,T,H,W] normalized to [-1,1].
    return torch.rand(batch_size, 3, 2, n_previous + 1, 8, 8) * 2 - 1


def _trainer(provider, head=None) -> SimpleNamespace:
    return SimpleNamespace(_research_planner=provider, _research_planner_head=head)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


def test_default_env_is_the_unchanged_code_path() -> None:
    trainer = SimpleNamespace()
    tokens = torch.randn(2, 2, 4, 256, 1024)
    out, extra = ge_trainer.apply_research_planner_mixing(trainer, tokens, None, ["a", "b"], 2)
    assert out is tokens and extra is None
    assert not hasattr(trainer, "_research_planner")
    model = _TinyLTX()
    ge_trainer.apply_research_planner_head(trainer, model, None)
    assert not hasattr(model, "research_planner_head")
    groups = ge_trainer.build_optimizer_parameter_groups(
        model, "all", 1e-4, 2e-4, action_lr=3e-4, baton_source=ge_trainer.BATON_TEACHER_SOURCE
    )
    assert [group["name"] for group in groups] == ["ltx_video", "action_expert", "semantic_adapter"]


def test_frozen_mixing_without_joint_is_detached(monkeypatch) -> None:
    monkeypatch.setenv("BATON_RESEARCH_PLANNER_PROB", "1")
    provider = _provider()
    tokens = torch.zeros(2, 2, 4, 256, 1024)
    video = _video(2)
    out, extra = ge_trainer.apply_research_planner_mixing(
        _trainer(provider), tokens, video, ["a", "b"], 2
    )
    assert extra is None and not out.requires_grad
    current = ge_trainer._normalized_video_to_uint8(video[:, :, :, 1].permute(0, 2, 1, 3, 4))
    torch.testing.assert_close(out, provider.predict(current.contiguous(), ["a", "b"]).tokens)


def test_refactored_forward_rows_matches_head_on_plan_states() -> None:
    provider = _provider()
    images = _images(2)
    reference = provider.predict(images, ["pick", "place"]).tokens
    head = ResearchPlannerHead(provider.planner)
    with torch.no_grad():
        joint = predict_with_research_head(provider, head, images, ["pick", "place"])
    torch.testing.assert_close(joint, reference)


def test_joint_head_gets_gradients_and_backbone_none(monkeypatch) -> None:
    monkeypatch.setenv("BATON_RESEARCH_PLANNER_PROB", "1")
    monkeypatch.setenv("BATON_RESEARCH_PLANNER_JOINT", "1")
    monkeypatch.setenv("BATON_RESEARCH_PLANNER_ANCHOR", "0.5")
    provider = _provider()
    model = _TinyLTX()
    restored = ge_trainer.attach_research_planner_head(model, provider, None)
    assert restored == 0
    head = model.research_planner_head
    tokens = torch.zeros(2, 2, 4, 256, 1024)
    out, extra = ge_trainer.apply_research_planner_mixing(
        _trainer(provider, head), tokens, _video(2), ["a", "b"], 2
    )
    assert out.requires_grad and extra is not None and float(extra) > 0
    assert provider.planner.backbone.model.grad_enabled == [False]
    (model.loss(out) + extra).backward()
    head_grads = [parameter.grad for parameter in head.parameters()]
    assert all(grad is not None for grad in head_grads)
    assert sum(float(grad.abs().sum()) for grad in head_grads) > 0
    assert all(parameter.grad is None for parameter in provider.planner.parameters())
    assert not any(parameter.requires_grad for parameter in provider.parameters())


def test_joint_step_without_picks_still_touches_every_head_parameter(monkeypatch) -> None:
    monkeypatch.setenv("BATON_RESEARCH_PLANNER_PROB", "1e-12")
    monkeypatch.setenv("BATON_RESEARCH_PLANNER_JOINT", "1")
    provider = _provider()
    model = _TinyLTX()
    ge_trainer.attach_research_planner_head(model, provider, None)
    tokens = torch.zeros(2, 2, 4, 256, 1024)
    out, extra = ge_trainer.apply_research_planner_mixing(
        _trainer(provider, model.research_planner_head), tokens, _video(2), ["a", "b"], 2
    )
    assert out is tokens and float(extra) == 0
    extra.backward()
    assert all(p.grad is not None for p in model.research_planner_head.parameters())


def test_joint_optimizer_group_and_checkpoint_round_trip(tmp_path) -> None:
    provider = _provider()
    model = _TinyLTX()
    ge_trainer.attach_research_planner_head(model, provider, None)
    with pytest.raises(ValueError, match="planner_head_lr"):
        ge_trainer.build_optimizer_parameter_groups(
            model, "all", 1e-4, 2e-4, action_lr=3e-4, baton_source=ge_trainer.BATON_TEACHER_SOURCE
        )
    groups = ge_trainer.build_optimizer_parameter_groups(
        model, "all", 1e-4, 2e-4, action_lr=3e-4,
        baton_source=ge_trainer.BATON_TEACHER_SOURCE, planner_head_lr=1e-5,
    )
    by_name = {group["name"]: group for group in groups}
    assert list(by_name)[-1] == "research_planner_head"
    assert by_name["research_planner_head"]["lr"] == 1e-5
    assert {id(p) for p in by_name["research_planner_head"]["params"]} == {
        id(p) for p in model.research_planner_head.parameters()
    }

    with torch.no_grad():
        for parameter in model.research_planner_head.parameters():
            parameter.add_(torch.randn_like(parameter))
    checkpoint = tmp_path / "diffusion_model"
    checkpoint.mkdir()
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(checkpoint / "diffusion_pytorch_model.safetensors"))
    assert ge_trainer.checkpoint_has_research_planner_head(str(checkpoint))

    fresh = _TinyLTX()
    restored = ge_trainer.attach_research_planner_head(fresh, _provider(seed=1), str(checkpoint))
    assert restored == len(model.research_planner_head.state_dict())
    for key, value in model.research_planner_head.state_dict().items():
        torch.testing.assert_close(fresh.research_planner_head.state_dict()[key], value)

    plain = tmp_path / "plain"
    plain.mkdir()
    save_file(
        {key: value.contiguous() for key, value in _TinyLTX().state_dict().items()},
        str(plain / "diffusion_pytorch_model.safetensors"),
    )
    assert not ge_trainer.checkpoint_has_research_planner_head(str(plain))


def _eval_module():
    path = REPOSITORY_ROOT / "qwen35_baton/scripts/research/research_eval_stage2_modes.py"
    spec = importlib.util.spec_from_file_location("research_eval_stage2_modes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(model, model_path) -> SimpleNamespace:
    return SimpleNamespace(
        _unwrapped_diffusion_model=lambda accelerator: model,
        state=SimpleNamespace(accelerator=None),
        args=SimpleNamespace(diffusion_model={"model_path": model_path}),
    )


def test_eval_planner_modes_pick_up_the_fine_tuned_head(tmp_path) -> None:
    evaluator = _eval_module()
    provider = _provider()
    trained = _TinyLTX()
    ge_trainer.attach_research_planner_head(trained, provider, None)
    with torch.no_grad():
        for parameter in trained.research_planner_head.parameters():
            parameter.mul_(1.5)
    checkpoint = tmp_path / "diffusion_model"
    checkpoint.mkdir()
    save_file(
        {key: value.contiguous() for key, value in trained.state_dict().items()},
        str(checkpoint / "diffusion_pytorch_model.safetensors"),
    )

    images = _images(1)
    loaded = _TinyLTX()
    head = evaluator._planner_head(_runner(loaded, str(checkpoint)), provider)
    assert head is loaded.research_planner_head and not head.training
    tokens = evaluator._planner_tokens(provider, head, images, ("pick",))
    with torch.no_grad():
        expected = predict_with_research_head(provider, trained.research_planner_head, images, ("pick",))
    torch.testing.assert_close(tokens, expected)
    assert not torch.allclose(tokens, provider.predict(images, ("pick",)).tokens)

    plain = tmp_path / "plain"
    plain.mkdir()
    save_file(
        {key: value.contiguous() for key, value in _TinyLTX().state_dict().items()},
        str(plain / "diffusion_pytorch_model.safetensors"),
    )
    assert evaluator._planner_head(_runner(_TinyLTX(), str(plain)), provider) is None
    torch.testing.assert_close(
        evaluator._planner_tokens(provider, None, images, ("pick",)),
        provider.predict(images, ("pick",)).tokens,
    )
