"""Bridge to the vendored GE-Act (Genie-Envisioner) WAM code.

GE-Act uses absolute imports rooted at its repo top (``from models...``, ``from utils...``,
``from data...``). The code is vendored at ``VLM4WAM/ge_act`` — put that dir on
sys.path before importing, mirroring how the planner bridges to lingbot-vla-v2 via LINGBOT_SRC_ROOT.

Override the vendored root with env GE_ACT_ROOT if needed.
"""
import os
import sys
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
GE_ACT_ROOT = Path(os.environ.get("GE_ACT_ROOT", str(_DEFAULT_ROOT)))


def ensure_on_path() -> Path:
    """Put the vendored GE-Act root on sys.path (idempotent); return the root."""
    root = str(GE_ACT_ROOT)
    if not (GE_ACT_ROOT / "models" / "action_patches" / "patches.py").exists():
        raise RuntimeError(f"vendored GE-Act not found under {root} (set GE_ACT_ROOT)")
    if root not in sys.path:
        sys.path.insert(0, root)
    return GE_ACT_ROOT


def load_action_dit(ckpt_dir, dtype=None, device="cuda", **action_overrides):
    """Load MultiViewCosmosTransformer3DModel from a diffusers-format ckpt dir.

    from_pretrained alone fails (action-expert kwargs live in the YAML, not config.json, and
    diffusers drops unregistered **kwargs) — instantiate directly with merged config + load_state_dict.
    """
    import json
    import torch
    from safetensors.torch import load_file
    ensure_on_path()
    from models.cosmos_models.models.transformers.transformer_cosmos_multiview import (
        MultiViewCosmosTransformer3DModel,
    )
    ckpt_dir = Path(ckpt_dir)
    cfg = json.load(open(ckpt_dir / "config.json"))
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    # action-expert geometry for the LIBERO-cosmos ckpt (from action_model_libero_cosmos*.yaml)
    cfg.setdefault("action_in_channels", 15)
    cfg.setdefault("action_out_channels", 15)
    cfg.setdefault("action_num_attention_heads", 16)
    cfg.setdefault("action_attention_head_dim", 32)
    cfg.setdefault("action_rope_dim", 32)
    cfg.update(action_overrides)
    model = MultiViewCosmosTransformer3DModel(**cfg)
    miss, unexp = model.load_state_dict(load_file(ckpt_dir / "diffusion_pytorch_model.safetensors"), strict=False)
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device).eval()
    return model, (len(miss), len(unexp))
