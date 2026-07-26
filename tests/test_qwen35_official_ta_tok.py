from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
from easydict import EasyDict
from safetensors.torch import load_file
from torch import nn
from transformers import Siglip2VisionConfig, Siglip2VisionModel


class _FakeEncoder(nn.Module):
    def forward(
        self, images: torch.Tensor, *, output_hidden_states: bool
    ) -> SimpleNamespace:
        assert output_hidden_states
        hidden = images.new_zeros((images.shape[0], 729, 1152))
        return SimpleNamespace(hidden_states=(hidden, hidden, hidden))


@dataclass(frozen=True)
class _FakeReleasedCheckpoint:
    path: Path
    encoder_factory: Callable[[], nn.Module]


def _released_args() -> EasyDict:
    return EasyDict(
        {
            "bottleneck": {
                "name": "bottleneck",
                "args": {
                    "bottleneck_dim": 1536,
                    "norm": "none",
                    "regularizer": {
                        "name": "simvq",
                        "args": {
                            "codebook_size": 65_536,
                            "commitment_loss_weight": 0.25,
                            "codebook_loss_weight": 1.0,
                            "entropy_loss_weight": 0.0,
                            "entropy_loss_temperature": 0.01,
                            "l2_normalized": True,
                            "stochastic": True,
                            "stochastic_temperature": 0.03,
                            "top_k": 4,
                            "top_k_prob": 0.5,
                            "residual_weight": 0.1,
                        },
                    },
                },
            },
            "bottleneck_token_num": 729,
            "input_size": 384,
            "teacher": "google/siglip2-so400m-patch14-384",
            "ckpt_path": "google/siglip2-so400m-patch14-384",
            "pool_scale": 1,
            "rand_scale": True,
        }
    )


def _expanded(shape: torch.Size, value: float = 0.0) -> torch.Tensor:
    scalar = torch.tensor(value, dtype=torch.float32)
    return scalar if len(shape) == 0 else scalar.expand(shape)


def _fake_released_state() -> dict[str, torch.Tensor]:
    decoder_config = Siglip2VisionConfig()
    decoder_config.update(
        {
            "patch_size": 1,
            "num_hidden_layers": 3,
            "num_channels": 1536,
            "hidden_size": 1152,
        }
    )
    decoder = Siglip2VisionModel(decoder_config)
    state = {
        f"decoder.{name}": _expanded(value.shape)
        for name, value in decoder.state_dict().items()
    }
    state.update(
        {
            "scale_layer.shift": _expanded(torch.Size((1, 3, 1, 1)), 0.5),
            "scale_layer.scale": _expanded(torch.Size((1, 3, 1, 1)), 0.5),
            "encode_task_layer.0.weight": _expanded(torch.Size((1152, 1152))),
            "encode_task_layer.0.bias": _expanded(torch.Size((1152,))),
            "decode_task_layer.0.weight": _expanded(torch.Size((1152, 1152))),
            "decode_task_layer.0.bias": _expanded(torch.Size((1152,))),
            "decode_task_layer.2.weight": _expanded(torch.Size((1152, 1152))),
            "decode_task_layer.2.bias": _expanded(torch.Size((1152,))),
            "bottleneck.in_linear.weight": _expanded(torch.Size((1536, 1152))),
            "bottleneck.in_linear.bias": _expanded(torch.Size((1536,))),
            "bottleneck.out_linear.weight": _expanded(torch.Size((1536, 1536))),
            "bottleneck.out_linear.bias": _expanded(torch.Size((1536,))),
            "bottleneck.regularizer.embedding.weight": _expanded(
                torch.Size((65_536, 1536))
            ),
            "bottleneck.regularizer.embedding_proj.weight": _expanded(
                torch.Size((1536, 1536))
            ),
            "bottleneck.regularizer.embedding_proj.bias": _expanded(
                torch.Size((1536,))
            ),
        }
    )
    return state


def _write_checkpoint(
    path: Path,
    *,
    args: EasyDict | None = None,
    state: dict[str, torch.Tensor] | None = None,
) -> Path:
    torch.save(
        {"model": {"args": args or _released_args(), "sd": state or {}}},
        path,
    )
    return path


@pytest.fixture(scope="module")
def fake_released_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> _FakeReleasedCheckpoint:
    path = tmp_path_factory.mktemp("released-ta") / "ta_tok.pth"
    _write_checkpoint(path, state=_fake_released_state())
    return _FakeReleasedCheckpoint(path=path, encoder_factory=_FakeEncoder)


def test_released_adapter_exposes_codes_and_codebook(
    fake_released_checkpoint: _FakeReleasedCheckpoint,
) -> None:
    from qwen35_planx.official_ta_tok import ReleasedTATok

    tokenizer = ReleasedTATok.from_checkpoint(
        fake_released_checkpoint.path,
        encoder_factory=fake_released_checkpoint.encoder_factory,
        weights_only=True,
    )
    images = torch.zeros(2, 3, 384, 384)
    output = tokenizer.encode_codes(images)
    assert output.codes.shape == (2, 729)
    assert output.codes.dtype == torch.long
    assert tokenizer.codebook.shape == (65_536, 1536)
    assert not any(parameter.requires_grad for parameter in tokenizer.parameters())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_size", 256, "input_size"),
        ("bottleneck_token_num", 256, "bottleneck_token_num"),
    ],
)
def test_released_adapter_rejects_superseded_geometry_before_model_construction(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    from qwen35_planx.official_ta_tok import ReleasedTATok

    args = _released_args()
    args[field] = value
    checkpoint = _write_checkpoint(tmp_path / "bad.pth", args=args)
    constructed = False

    def factory() -> nn.Module:
        nonlocal constructed
        constructed = True
        return _FakeEncoder()

    with pytest.raises(ValueError, match=message):
        ReleasedTATok.from_checkpoint(checkpoint, encoder_factory=factory)
    assert not constructed


def test_released_adapter_rejects_wrong_codebook_shape_before_model_construction(
    tmp_path: Path,
) -> None:
    from qwen35_planx.official_ta_tok import ReleasedTATok

    state = {
        "bottleneck.regularizer.embedding.weight": torch.zeros(1024, 1536),
        "bottleneck.regularizer.embedding_proj.weight": torch.zeros(1536, 1536),
        "bottleneck.regularizer.embedding_proj.bias": torch.zeros(1536),
    }
    checkpoint = _write_checkpoint(tmp_path / "bad.pth", state=state)

    with pytest.raises(ValueError, match="codebook.*shape"):
        ReleasedTATok.from_checkpoint(
            checkpoint,
            encoder_factory=lambda: pytest.fail("constructed model too early"),
        )


def test_released_adapter_rejects_missing_checkpoint_keys_before_construction(
    tmp_path: Path,
) -> None:
    from qwen35_planx.official_ta_tok import ReleasedTATok

    checkpoint = _write_checkpoint(tmp_path / "bad.pth", state={})

    with pytest.raises(ValueError, match="missing.*embedding.weight"):
        ReleasedTATok.from_checkpoint(
            checkpoint,
            encoder_factory=lambda: pytest.fail("constructed model too early"),
        )


def test_released_adapter_rejects_unsafe_loading_before_reading_checkpoint(
    tmp_path: Path,
) -> None:
    from qwen35_planx.official_ta_tok import ReleasedTATok

    with pytest.raises(ValueError, match="weights_only=True"):
        ReleasedTATok.from_checkpoint(
            tmp_path / "does-not-need-to-exist.pth",
            encoder_factory=_FakeEncoder,
            weights_only=False,
        )


def test_released_adapter_rejects_nonfinite_codebook_before_construction(
    tmp_path: Path,
) -> None:
    from qwen35_planx.official_ta_tok import ReleasedTATok

    state = {
        "bottleneck.regularizer.embedding.weight": _expanded(
            torch.Size((65_536, 1536)), float("nan")
        ),
        "bottleneck.regularizer.embedding_proj.weight": _expanded(
            torch.Size((1536, 1536))
        ),
        "bottleneck.regularizer.embedding_proj.bias": _expanded(torch.Size((1536,))),
    }
    checkpoint = _write_checkpoint(tmp_path / "bad.pth", state=state)

    with pytest.raises(ValueError, match="non-finite"):
        ReleasedTATok.from_checkpoint(
            checkpoint,
            encoder_factory=lambda: pytest.fail("constructed model too early"),
        )


def test_lookup_decode_and_atomic_codebook_export(
    fake_released_checkpoint: _FakeReleasedCheckpoint, tmp_path: Path
) -> None:
    from qwen35_planx.official_ta_tok import (
        ReleasedTATok,
        export_codebook_safetensors,
    )

    tokenizer = ReleasedTATok.from_checkpoint(
        fake_released_checkpoint.path,
        encoder_factory=fake_released_checkpoint.encoder_factory,
    )
    codes = torch.zeros(1, 729, dtype=torch.long)
    assert tokenizer.lookup_codes(codes).shape == (1, 729, 1536)
    assert tokenizer.decode_features(codes).shape == (1, 729, 1152)

    tensor_path, metadata_path = export_codebook_safetensors(tokenizer, tmp_path)

    assert set(load_file(tensor_path)) == {"codebook"}
    metadata = json.loads(metadata_path.read_text())
    assert metadata["checkpoint_sha256"] == tokenizer.checkpoint_hash
    assert metadata["state_sha256"] == tokenizer.state_hash
    assert metadata["geometry"]["tokens_per_frame"] == 729
    assert metadata["teacher"]["selected_layer"] == -2
    assert not list(tmp_path.glob("*.tmp"))


def test_released_ta_preflight_reports_shape_and_checkpoint_hash(
    fake_released_checkpoint: _FakeReleasedCheckpoint,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from qwen35_planx.cli.preflight import main

    siglip_model = tmp_path / "siglip"
    siglip_model.mkdir()
    (siglip_model / "model.safetensors").touch()
    (siglip_model / "config.json").write_text("{}")

    result = main(
        [
            "released-ta",
            "--ta-checkpoint",
            str(fake_released_checkpoint.path),
            "--siglip-model",
            str(siglip_model),
            "--output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "(729, 65536, 1536)" in output
    assert "checkpoint_sha256=" in output


def test_released_ta_preflight_rejects_missing_weights_and_low_space(
    fake_released_checkpoint: _FakeReleasedCheckpoint,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen35_planx.cli import preflight

    siglip_model = tmp_path / "siglip"
    siglip_model.mkdir()
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=100, used=100, free=0),
    )

    errors = preflight.collect_released_ta_preflight_errors(
        fake_released_checkpoint.path,
        siglip_model,
        tmp_path,
    )

    assert any("SigLIP2 model weights" in error for error in errors)
    assert any("free output space" in error for error in errors)
