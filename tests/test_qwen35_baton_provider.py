from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch
import torch.nn as nn

from qwen35_baton.checkpoint import (
    BatonTrainingCursor,
    capture_rank_rng_state,
    save_baton_checkpoint,
)
from qwen35_baton.cli.train_semantic_planner import BatonCosineWarmupScheduler
from qwen35_baton.config import BatonCheckpointMetadata
from qwen35_baton.hashing import sha256_artifact, sha256_file, sha256_json
from qwen35_baton.model import BatonPlannerOutput
from qwen35_baton.provider import FrozenBatonPlanner
from qwen35_baton.sequence import ADDED_TOKENS


PLAN_PAD_ID = 105
ADDED_TOKEN_IDS = tuple(range(100, 107))


class _Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return dict(zip(ADDED_TOKENS, ADDED_TOKEN_IDS, strict=True))[token]


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.seen_texts: list[str] = []

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return str(messages[0]["content"][1]["text"])

    def __call__(
        self,
        *,
        text: Sequence[str],
        images: Sequence[torch.Tensor],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert not padding
        self.seen_texts.append(text[0])
        image_code = int(images[0][0, 0, 0].item())
        instruction_code = 2 if "counterfactual" in text[0] else 1
        input_ids = torch.tensor(
            [[image_code, instruction_code, *([PLAN_PAD_ID] * 1024)]],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class _FakePlanner(nn.Module):
    def __init__(self, *, malformed: str | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.malformed = malformed
        self.forward_calls = 0
        self.attention_flags: list[bool] = []
        self.grad_enabled: list[bool] = []

    def forward_rows(
        self,
        qwen_inputs: Mapping[str, torch.Tensor],
        plan_positions: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        return_attention_maps: bool = False,
    ) -> BatonPlannerOutput:
        self.forward_calls += 1
        self.attention_flags.append(return_attention_maps)
        self.grad_enabled.append(torch.is_grad_enabled())
        rows = qwen_inputs["input_ids"].shape[0]
        flat = torch.zeros(
            rows,
            4,
            256,
            1024,
            device=qwen_inputs["input_ids"].device,
        )
        flat[..., 0] = qwen_inputs["input_ids"][:, 0, None, None].float()
        flat[..., 1] = qwen_inputs["input_ids"][:, 1, None, None].float()
        flat = flat * self.weight
        if self.malformed == "shape":
            flat = flat[..., :-1]
        elif self.malformed == "nonfinite":
            flat = flat.clone()
            flat[0, 0, 0, 0] = float("nan")
        maps = None
        if return_attention_maps:
            base = torch.arange(rows, dtype=torch.float32, device=flat.device)
            maps = tuple(
                base[:, None, None].expand(rows, 1024, 1024) + layer
                for layer in range(4)
            )
        return BatonPlannerOutput(
            flat=flat,
            positive=flat,
            negative=None,
            cross_attention_maps=maps,
        )


def _images(batch_size: int = 2) -> torch.Tensor:
    images = torch.zeros(batch_size, 2, 3, 8, 8, dtype=torch.uint8)
    images[:, 0, 0, 0, 0] = 3
    images[:, 1, 0, 0, 0] = 7
    return images


def test_provider_returns_full_independent_camera_grids_and_patch_centers() -> None:
    planner = _FakePlanner()
    provider = FrozenBatonPlanner(
        planner=planner,
        processor=_Processor(),
        added_token_ids=ADDED_TOKEN_IDS,
    )

    plan = provider.predict(_images(), ("pick cup", "open drawer"))

    assert plan.tokens.shape == (2, 2, 4, 256, 1024)
    assert plan.future_indices == (0, 3, 5, 8)
    assert plan.positions_xy.shape == (2, 2, 4, 256, 2)
    torch.testing.assert_close(
        plan.positions_xy[0, 0, 0, 0],
        torch.tensor([1 / 32, 1 / 32]),
    )
    torch.testing.assert_close(
        plan.positions_xy[0, 0, 0, -1],
        torch.tensor([31 / 32, 31 / 32]),
    )
    assert plan.relevance is None
    assert plan.cross_attention_maps is None
    assert plan.instruction_sensitivity is None
    assert plan.tokens[0, 0, 0, 0, 0].item() == 3
    assert plan.tokens[0, 1, 0, 0, 0].item() == 7
    assert planner.forward_calls == 1
    assert planner.attention_flags == [False]


def test_counterfactual_sensitivity_uses_one_combined_no_grad_forward() -> None:
    planner = _FakePlanner()
    provider = FrozenBatonPlanner(
        planner=planner,
        processor=_Processor(),
        added_token_ids=ADDED_TOKEN_IDS,
    )

    plan = provider.predict(
        _images(batch_size=1),
        ("pick cup",),
        counterfactual_instructions=("counterfactual instruction",),
        return_attention=True,
    )

    expected_main = 1 - torch.nn.functional.cosine_similarity(
        torch.tensor([3.0, 1.0]),
        torch.tensor([3.0, 2.0]),
        dim=0,
    )
    expected_wrist = 1 - torch.nn.functional.cosine_similarity(
        torch.tensor([7.0, 1.0]),
        torch.tensor([7.0, 2.0]),
        dim=0,
    )
    assert plan.instruction_sensitivity is not None
    assert plan.instruction_sensitivity.shape == (1, 2, 4, 256)
    torch.testing.assert_close(
        plan.instruction_sensitivity[0, 0],
        torch.full((4, 256), expected_main),
    )
    torch.testing.assert_close(
        plan.instruction_sensitivity[0, 1],
        torch.full((4, 256), expected_wrist),
    )
    assert plan.cross_attention_maps is not None
    assert len(plan.cross_attention_maps) == 4
    assert all(value.shape == (1, 2, 1024, 1024) for value in plan.cross_attention_maps)
    assert planner.forward_calls == 1
    assert planner.attention_flags == [True]
    assert planner.grad_enabled == [False]
    assert not plan.tokens.requires_grad
    assert not plan.instruction_sensitivity.requires_grad


def test_provider_freezes_every_module_and_cannot_be_switched_to_training() -> None:
    provider = FrozenBatonPlanner(
        planner=_FakePlanner(),
        processor=_Processor(),
        added_token_ids=ADDED_TOKEN_IDS,
    )

    provider.train()

    assert all(not module.training for module in provider.modules())
    assert all(not parameter.requires_grad for parameter in provider.parameters())


@pytest.mark.parametrize(
    ("images", "instructions", "message"),
    [
        (torch.zeros(1, 3, 8, 8, dtype=torch.uint8), ("pick",), r"\[B,2,3,H,W\]"),
        (torch.zeros(1, 2, 3, 8, 8), ("pick",), "uint8"),
        (torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8), (), "batch sizes"),
        (
            torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8),
            ("",),
            "nonempty strings",
        ),
    ],
)
def test_provider_rejects_malformed_inputs(
    images: torch.Tensor,
    instructions: tuple[str, ...],
    message: str,
) -> None:
    provider = FrozenBatonPlanner(
        planner=_FakePlanner(),
        processor=_Processor(),
        added_token_ids=ADDED_TOKEN_IDS,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        provider.predict(images, instructions)


@pytest.mark.parametrize("malformed", ["shape", "nonfinite"])
def test_provider_rejects_malformed_or_nonfinite_planner_output(
    malformed: str,
) -> None:
    provider = FrozenBatonPlanner(
        planner=_FakePlanner(malformed=malformed),
        processor=_Processor(),
        added_token_ids=ADDED_TOKEN_IDS,
    )

    with pytest.raises(RuntimeError, match="finite.*\\[rows,4,256,1024\\]"):
        provider.predict(_images(batch_size=1), ("pick",))


def _qwen_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "intermediate_size": 6144,
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "out_hidden_size": 2048,
        },
    }


def _write_artifacts(root: Path) -> tuple[Path, Path, Path]:
    model = root / "qwen"
    tokenizer = root / "tokenizer"
    processor = root / "processor"
    model.mkdir()
    tokenizer.mkdir()
    processor.mkdir()
    (model / "config.json").write_text(json.dumps(_qwen_config()))
    (tokenizer / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"content": token, "id": token_id}
                    for token, token_id in zip(
                        ADDED_TOKENS, ADDED_TOKEN_IDS, strict=True
                    )
                ]
            }
        )
    )
    (processor / "processor_config.json").write_text("{}")
    return model, tokenizer, processor


def _write_checkpoint(
    checkpoint: Path,
    *,
    metadata: BatonCheckpointMetadata,
    planner: nn.Module,
) -> None:
    optimizer = torch.optim.AdamW(
        [{"name": "planner", "params": list(planner.parameters()), "lr": 5e-5}]
    )
    scheduler = BatonCosineWarmupScheduler(
        optimizer,
        warmup_steps=0,
        max_steps=10,
    )
    save_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        cursor=BatonTrainingCursor(
            global_step=0,
            epoch=0,
            consumed_microbatches=0,
            microbatches_per_epoch=1,
            sampler_seed=0,
        ),
        rank_rng_state={0: capture_rank_rng_state(distributed_rank=0)},
    )


def _metadata_for_artifacts(
    model_path: Path,
    tokenizer_path: Path,
    processor_path: Path,
) -> BatonCheckpointMetadata:
    return replace(
        BatonCheckpointMetadata.example(),
        qwen_config_hash=sha256_file(model_path / "config.json"),
        tokenizer_hash=sha256_artifact(tokenizer_path),
        processor_hash=sha256_artifact(processor_path),
        input_template_hash=sha256_json(
            "Instruction: {instruction}\n"
            "<PLAN_START>\n"
            + "\n".join(
                f"<FRAME_{index}> " + " ".join(["<PLAN_PAD>"] * 256)
                for index in range(4)
            )
            + "\n<PLAN_END>"
        ),
        added_token_ids=ADDED_TOKEN_IDS,
    )


def test_checkpoint_tokenizer_hash_tamper_fails_before_component_loading(
    tmp_path: Path,
) -> None:
    model_path, tokenizer_path, processor_path = _write_artifacts(tmp_path)
    metadata = replace(
        _metadata_for_artifacts(model_path, tokenizer_path, processor_path),
        tokenizer_hash="0" * 64,
    )
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, metadata=metadata, planner=_FakePlanner())
    calls = 0

    def forbidden_loader(**_: Any) -> tuple[Any, nn.Module]:
        nonlocal calls
        calls += 1
        raise AssertionError("components must not load after provenance failure")

    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        FrozenBatonPlanner.from_checkpoint(
            checkpoint,
            qwen_model_path=model_path,
            qwen_tokenizer_path=tokenizer_path,
            qwen_processor_path=processor_path,
            _component_loader=forbidden_loader,
        )

    assert calls == 0


def test_checkpoint_file_hash_tamper_fails_before_component_loading(
    tmp_path: Path,
) -> None:
    model_path, tokenizer_path, processor_path = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint,
        metadata=_metadata_for_artifacts(
            model_path,
            tokenizer_path,
            processor_path,
        ),
        planner=_FakePlanner(),
    )
    with (checkpoint / "planner.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    calls = 0

    def forbidden_loader(**_: Any) -> tuple[Any, nn.Module]:
        nonlocal calls
        calls += 1
        raise AssertionError("components must not load after hash failure")

    with pytest.raises(ValueError, match="hash mismatch.*planner.safetensors"):
        FrozenBatonPlanner.from_checkpoint(
            checkpoint,
            qwen_model_path=model_path,
            qwen_tokenizer_path=tokenizer_path,
            qwen_processor_path=processor_path,
            _component_loader=forbidden_loader,
        )

    assert calls == 0


def test_checkpoint_topology_hash_tamper_fails_before_component_loading(
    tmp_path: Path,
) -> None:
    model_path, tokenizer_path, processor_path = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint,
        metadata=_metadata_for_artifacts(
            model_path,
            tokenizer_path,
            processor_path,
        ),
        planner=_FakePlanner(),
    )
    optimizer_state = torch.load(
        checkpoint / "optimizer.pt",
        weights_only=True,
        map_location="cpu",
    )
    optimizer_state["param_groups"][0]["name"] = "rewritten"
    torch.save(optimizer_state, checkpoint / "optimizer.pt")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["files"]["optimizer.pt"] = sha256_file(checkpoint / "optimizer.pt")
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    calls = 0

    def forbidden_loader(**_: Any) -> tuple[Any, nn.Module]:
        nonlocal calls
        calls += 1
        raise AssertionError("components must not load after topology hash failure")

    with pytest.raises(ValueError, match="optimizer topology hash"):
        FrozenBatonPlanner.from_checkpoint(
            checkpoint,
            qwen_model_path=model_path,
            qwen_tokenizer_path=tokenizer_path,
            qwen_processor_path=processor_path,
            _component_loader=forbidden_loader,
        )

    assert calls == 0


def test_valid_checkpoint_loads_state_then_returns_frozen_provider(
    tmp_path: Path,
) -> None:
    model_path, tokenizer_path, processor_path = _write_artifacts(tmp_path)
    source = _FakePlanner()
    source.weight.data.fill_(3.0)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint,
        metadata=_metadata_for_artifacts(
            model_path,
            tokenizer_path,
            processor_path,
        ),
        planner=source,
    )
    runtime = _FakePlanner()
    runtime.weight.data.zero_()
    component_calls = 0

    def component_loader(**kwargs: Any) -> tuple[Any, nn.Module]:
        nonlocal component_calls
        component_calls += 1
        assert kwargs["qwen_model_path"] == model_path
        assert kwargs["qwen_tokenizer_path"] == tokenizer_path
        assert kwargs["qwen_processor_path"] == processor_path
        return _Processor(), runtime

    provider = FrozenBatonPlanner.from_checkpoint(
        checkpoint,
        qwen_model_path=model_path,
        qwen_tokenizer_path=tokenizer_path,
        qwen_processor_path=processor_path,
        _component_loader=component_loader,
    )

    assert component_calls == 1
    assert provider.planner.weight.item() == 3.0
    assert all(not module.training for module in provider.modules())
    assert all(not parameter.requires_grad for parameter in provider.parameters())
