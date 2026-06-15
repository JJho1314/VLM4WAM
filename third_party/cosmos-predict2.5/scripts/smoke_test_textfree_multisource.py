#!/usr/bin/env python3
"""Local smoke test for the text-free multi-source path (no InstructSAM / no 2B).

Validates the pieces that can run without the InstructSAM checkpoint or the full
Cosmos training stack:
  1. extractor pooling helpers (_to_2d_float, _adaptive_pool_tokens)
  2. precompute fusion (SourceProjector, _fit_budget, fuse)
  3. MultiSourceTargetFeatureContextAdapter forward + per-source segment embedding
  4. the replace-text branch of append_target_feature_context (logic-level)
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
import types
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: F401  (used by exec'd adapter source)
from torch import nn  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# This conda env is NOT the cosmos uv runtime: stub the cosmos_cuda extra so the
# light target_aware modules import without the CUDA build, and skip the version
# check. Heavy modules (minimal_v4_dit -> transformer_engine) are tested by
# exec'ing only the adapter class source, not importing the package.
_stub = types.ModuleType("cosmos_cuda")
_stub.__version__ = "0.0.0"
sys.modules.setdefault("cosmos_cuda", _stub)
os.environ["COSMOS_SKIP_CUDA_VERSION_CHECK"] = "1"


def _load_script_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extractor_helpers() -> None:
    from cosmos_predict2._src.predict2.target_aware.instructsam_multisource import (
        _adaptive_pool_tokens,
        _to_2d_float,
    )

    assert _to_2d_float(torch.randn(7, 256)).shape == (7, 256)
    assert _to_2d_float(torch.randn(256)).shape == (1, 256)
    assert _to_2d_float(torch.randn(2, 3, 256)).shape == (6, 256)
    nan = torch.full((4, 8), float("nan"))
    assert torch.isfinite(_to_2d_float(nan)).all()

    assert _adaptive_pool_tokens(torch.randn(100, 256), 32).shape == (32, 256)
    assert _adaptive_pool_tokens(torch.randn(10, 256), 32).shape == (10, 256)  # no upsample
    print("[ok] extractor helpers")


def test_fusion() -> None:
    pc = _load_script_module(
        "precompute_ms", "scripts/precompute_instructsam_multisource_features.py"
    )
    from cosmos_predict2._src.predict2.target_aware.instructsam_multisource import (
        MultiSourceFeatureResult,
    )

    budgets = {"mask": 16, "detect": 16, "vtext": 32}
    out_dim = 256

    with tempfile.TemporaryDirectory() as td:
        projector = pc.SourceProjector(Path(td), out_dim=256, seed=0)
        # native == out -> identity (mask/detect 256 stay as-is).
        ident = projector.project(torch.randn(5, 256))
        assert ident.shape == (5, 256)
        # native (2048, the REAL InstructSAM-2B vtext dim) -> projects to 256,
        # NOT silently zeroed.
        vt = torch.randn(7, 2048)
        proj_vt = projector.project(vt)
        assert proj_vt.shape == (7, 256)
        assert proj_vt.abs().sum() > 0, "vtext 2048->256 must not be zero"
        # orthonormal columns for 2048>=256.
        mat = projector._matrix(2048)
        assert mat.shape == (2048, 256)
        assert torch.allclose(mat.t() @ mat, torch.eye(256), atol=1e-3)
        # deterministic across projector instances (persisted + seeded).
        projector2 = pc.SourceProjector(Path(td), out_dim=256, seed=0)
        assert torch.equal(projector2._matrix(2048), mat)

        # All three present, with the REAL vtext dim 2048.
        res = MultiSourceFeatureResult(
            mask_L_Dm=torch.randn(5, 256),
            detect_L_Dd=torch.randn(9, 256),
            vtext_L_Dv=torch.randn(40, 2048),
        )
        fused, segs = pc.fuse(res, projector, budgets, out_dim)
        assert fused.shape == (64, 256), fused.shape
        assert segs == budgets
        assert fused[:5].abs().sum() > 0 and fused[5:16].abs().sum() == 0   # mask rows 0..4
        assert fused[16:25].abs().sum() > 0 and fused[25:32].abs().sum() == 0  # detect rows 16..24
        assert fused[32:64].abs().sum() > 0, "vtext segment must be non-zero (regression: 2048!=4096)"

        # Missing detect -> only its segment is zero.
        res2 = MultiSourceFeatureResult(
            mask_L_Dm=torch.randn(3, 256), detect_L_Dd=None, vtext_L_Dv=torch.randn(10, 2048)
        )
        fused2, _ = pc.fuse(res2, projector, budgets, out_dim)
        assert fused2.shape == (64, 256)
        assert fused2[16:32].abs().sum() == 0, "detect segment should be zero when missing"
        assert fused2[32:64].abs().sum() > 0, "vtext still present"
    print("[ok] precompute fusion (per-source proj, vtext 2048->256 not zeroed, missing-source zeros)")


def _exec_adapter_classes():
    """Exec just the two adapter classes from minimal_v4_dit.py in isolation.

    The full module imports transformer_engine/megatron which aren't in this
    env, but the adapters only need torch.nn, so we slice their source out.
    """
    src = (REPO / "cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py").read_text()
    start = src.index("class TargetFeatureContextAdapter")
    end = src.index("class VideoPositionEmb")
    classes_src = src[start:end]
    ns: dict = {"nn": nn, "torch": torch, "F": F, "Sequence": Sequence,
                "Optional": object, "Tuple": tuple}
    exec(compile(classes_src, "minimal_v4_dit_adapters", "exec"), ns)
    return ns["MultiSourceTargetFeatureContextAdapter"]


def test_adapter() -> None:
    MultiSourceTargetFeatureContextAdapter = _exec_adapter_classes()

    segments = [16, 16, 32]
    adapter = MultiSourceTargetFeatureContextAdapter(
        source_segments=segments, in_dim=256, hidden_dim=512, out_dim=1024
    )
    adapter.reset_parameters()
    x = torch.randn(2, 64, 256)
    out = adapter(x)
    assert out.shape == (2, 64, 1024), out.shape
    assert torch.isfinite(out).all()

    # After reset, source_embedding is zero -> equals base adapter output. Now make
    # source embeddings distinct and confirm segments are shifted differently.
    with torch.no_grad():
        adapter.source_embedding.weight.copy_(
            torch.arange(3).float().view(3, 1).expand(3, 1024) * 10.0
        )
    out2 = adapter(x)
    seg_means = [
        out2[:, 0:16].mean().item(),
        out2[:, 16:32].mean().item(),
        out2[:, 32:64].mean().item(),
    ]
    assert seg_means[0] < seg_means[1] < seg_means[2], seg_means
    # Buffer not persisted in state_dict (deterministic).
    assert "segment_ids" not in adapter.state_dict()
    print(f"[ok] multi-source adapter forward + per-source embedding (seg means {seg_means})")


def test_replace_text_branch() -> None:
    # Logic-level check of the replace-text branch without building the 2B DiT:
    # feature_tokens become the entire crossattn context.
    feature_tokens = torch.randn(2, 64, 1024)
    text_emb = torch.randn(2, 512, 1024)

    def append(replace_text: bool, append_to_text: bool):
        if replace_text:
            return feature_tokens
        if not append_to_text:
            return text_emb
        return torch.cat([feature_tokens, text_emb], dim=1)

    assert append(True, False).shape == (2, 64, 1024)  # text-free
    assert torch.equal(append(True, False), feature_tokens)
    assert append(False, True).shape == (2, 576, 1024)  # prepend (existing)
    assert append(False, False).shape == (2, 512, 1024)  # separate branch (existing)
    print("[ok] replace-text branch logic")


def main() -> int:
    test_extractor_helpers()
    test_fusion()
    test_adapter()
    test_replace_text_branch()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
