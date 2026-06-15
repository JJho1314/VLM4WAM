#!/usr/bin/env python3
"""Extract fused multi-source InstructSAM features for arbitrary (image, query) pairs.

Used for the target-query ablation: feed the SAME first frame with the real
target query vs a distractor query, producing the fused [L, out_dim] feature the
text-free multisource model consumes. Reuses the exact SourceProjector + fuse
from the precompute pipeline (and loads the SAME saved projection matrix) so the
features are identical in construction to training.

Run in the InstructSAM conda env (see sbatch_precompute_instructsam_multisource).

Manifest JSON: a list of {"image": <path>, "query": <str>, "out": <path>}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cosmos_predict2._src.predict2.target_aware.instructsam_multisource import (  # noqa: E402
    InstructSAMMultiSourceGenerator,
)


def _load_precompute_module():
    path = Path(__file__).resolve().parent / "precompute_instructsam_multisource_features.py"
    spec = importlib.util.spec_from_file_location("precompute_ms", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="JSON list of {image, query, out}.")
    ap.add_argument("--proj-dir", required=True,
                    help="Dir holding the training _proj_*.pt (use the train target_features_multisource dir so the projection matches training exactly).")
    ap.add_argument("--model-path", default=os.environ.get("INSTRUCTSAM_MODEL_PATH"))
    ap.add_argument("--source-root", default=os.environ.get("INSTRUCTSAM_SOURCE_ROOT"))
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--mask-tokens", type=int, default=16)
    ap.add_argument("--detect-tokens", type=int, default=16)
    ap.add_argument("--vtext-tokens", type=int, default=32)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = ap.parse_args()

    pc = _load_precompute_module()
    budgets = {"mask": args.mask_tokens, "detect": args.detect_tokens, "vtext": args.vtext_tokens}
    projector = pc.SourceProjector(Path(args.proj_dir), args.out_dim, args.proj_seed)

    device_map = {"": "cuda:0"} if torch.cuda.is_available() else "cpu"
    gen = InstructSAMMultiSourceGenerator(
        args.model_path,
        source_root=args.source_root,
        device_map=device_map,
        torch_dtype=pc.torch_dtype_from_name(args.torch_dtype),
        detect_max_tokens=max(args.detect_tokens * 4, 64),
        vtext_max_tokens=max(args.vtext_tokens * 4, 64),
    )

    items = json.loads(Path(args.manifest).read_text())
    total_tokens = sum(budgets.values())
    for i, it in enumerate(items):
        image, query, out = it["image"], it["query"], it["out"]
        result = gen.predict_multi_source_from_input(image, query)
        def _sh(t):
            return None if t is None else list(t.shape)
        print(f"[{i+1}/{len(items)}] {Path(image).name} q={query!r} "
              f"mask={_sh(result.mask_L_Dm)} detect={_sh(result.detect_L_Dd)} vtext={_sh(result.vtext_L_Dv)} "
              f"text={result.text!r}", flush=True)
        fused, segments = pc.fuse(result, projector, budgets, args.out_dim)
        assert fused.shape == (total_tokens, args.out_dim), fused.shape
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        # Save the InstructSAM segmentation mask alongside the feature, as a
        # record of what the model was pointed at (PNG for eyeballing).
        mask_png = None
        mask_area = None
        if result.mask_HW is not None:
            from PIL import Image as PILImage

            mask_np = (result.mask_HW.numpy() > 0.5)
            mask_area = float(mask_np.mean())
            mask_png = str(Path(out).with_suffix("")) + "_mask.png"
            PILImage.fromarray((mask_np * 255).astype("uint8")).save(mask_png)

        torch.save({
            "target_feature": fused,
            "source_segments": segments,
            "source_order": list(pc.SOURCES),
            "query": query,
            "image": str(image),
            "instructsam_text": result.text,
            "score": result.score,
            "mask_png": mask_png,
            "mask_area_fraction": mask_area,
            "feature_mode": "multisource",
        }, out)
        print(f"    saved -> {out}  shape={tuple(fused.shape)}  mask={mask_png} area={mask_area}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
