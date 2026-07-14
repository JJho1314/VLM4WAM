#!/usr/bin/env python3
"""Qualitative visualization for the 4B lingbot-DINO planner (lingbot-paper style).

One PNG per sample. Rows = the 5 FUTURE keyframes the planner predicts; columns:

  Current (input RGB, reference only) | Future RGB | Depth-Target | Depth-Pred |
  DINO-Target | DINO-Pred

Everything predicted is FUTURE-only (no current alignment exists in this planner line).

DINO panels: patch features have no decoder, so we render PCA(3) fit JOINTLY on each
(target, pred) pair -> shared RGB basis (this is also what the lingbot figure shows — their
DINO panels are 16x16 mosaics too).

Depth panels: rendered DENSE like the official figure. The MDM/LingBot-Depth checkpoint ships
its ConvStack depth decoder (neck -> depth_head, 16x16 -> 256x256, exp remap), so we push the
16x16x1024 feature tokens — teacher's for Depth-Target, the planner head's for Depth-Pred —
through that frozen decoder to get continuous depth maps. The cls token is not distilled, so
BOTH decodes borrow the teacher cls of the same future frame (shared context; the comparison
then isolates exactly what the head predicts: the patch features). Displayed as jointly
min-max-normalized disparity (1/depth) with the turbo colormap.

Depth-Pred needs a depth head in the checkpoint (depth_head.pt from newer runs, or
depth_head_refit.pt from refit_depth_head_from_ckpt.py); without one the column is omitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_qwen3vl4b_lingbot_dino_planner as T  # noqa: E402
from evaluate_qwen3vl4b_lingbot_dino_planner import load_planner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--frame-ranges-json", type=Path, default=None)
    p.add_argument("--dino-teacher-ckpt", type=Path, required=True)
    p.add_argument("--dino-teacher-config", type=Path, required=True)
    p.add_argument("--dino-input-size", type=int, default=256)
    p.add_argument("--depth-moge-path", type=Path, default=None)
    p.add_argument("--depth-morgbd-path", type=Path, default=None)
    p.add_argument("--num-keyframes", type=int, default=5)
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--num-latent-per-keyframe", type=int, default=8)
    p.add_argument("--semantic-dim", type=int, default=1024)
    p.add_argument("--sequence-length", type=int, default=49)
    p.add_argument("--keyframe-scheme", default="uniform")
    p.add_argument("--keyframe-gamma", type=float, default=0.6)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def pca_rgb_pair(target_tokens: np.ndarray, pred_tokens: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA(3) on the concatenated pair -> two (grid, grid, 3) images sharing one color basis."""
    from sklearn.decomposition import PCA

    both = np.concatenate([target_tokens, pred_tokens], axis=0)
    comp = PCA(n_components=3).fit_transform(both)
    lo, hi = comp.min(axis=0, keepdims=True), comp.max(axis=0, keepdims=True)
    comp = (comp - lo) / np.maximum(hi - lo, 1e-8)
    n = grid * grid
    return comp[:n].reshape(grid, grid, 3), comp[n:].reshape(grid, grid, 3)


def depth_feats_with_cls(enc, kfs: list[torch.Tensor]) -> tuple[np.ndarray, torch.Tensor]:
    """Teacher depth features + cls for K future keyframes of ONE sample (batch 1).

    kfs: list of K (1,3,H,W) uint8/float frames. Returns tokens (K, N, C) float numpy and
    cls (K, C) float cpu tensor (needed to run the dense decoder)."""
    prepped = torch.cat([enc._prep(kf) for kf in kfs], dim=0).to(enc.device)  # (K,3,S,S) [0,255]
    inp = prepped / 255.0
    out = enc.moge.infer(inp, resolution_level=enc.resolution_level,
                         num_tokens=enc.num_tokens, apply_mask=False)
    d = torch.nan_to_num(out["depth"].detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
    if d.dim() == 4:
        d = d.squeeze(1)
    feat, cls = enc.morgbd.infer_feat(inp, d, depth_down_scale=1,
                                      resolution_level=enc.resolution_level,
                                      num_tokens=enc.num_tokens, enable_depth_mask=False)
    tokens = feat.permute(0, 2, 3, 1)
    tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])  # (K, N, C)
    return tokens.float().cpu().numpy(), cls.detach().float().cpu()


@torch.no_grad()
def decode_depth_dense(mdm, tokens: torch.Tensor, cls: torch.Tensor, grid: int, out_size: int = 256) -> torch.Tensor:
    """Official-style dense decode: (N, C) feature tokens + (C,) cls -> (out_size, out_size) depth.

    Replicates MDMModel.forward's head path (cls add -> UV pyramid -> neck -> ConvStack depth
    head -> bilinear resize -> exp remap) with the encoder output swapped for `tokens`."""
    from mdm.utils.geo import normalized_view_plane_uv  # on sys.path once DepthTargetEncoder built

    device = next(mdm.parameters()).device
    dtype = next(mdm.parameters()).dtype
    feat = tokens.reshape(1, grid, grid, -1).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    feat = feat + cls.reshape(1, -1, 1, 1).to(device=device, dtype=dtype)
    features: list[torch.Tensor | None] = [feat, None, None, None, None]
    for level in range(5):
        uv = normalized_view_plane_uv(width=grid * 2 ** level, height=grid * 2 ** level,
                                      aspect_ratio=1.0, dtype=dtype, device=device)
        uv = uv.permute(2, 0, 1).unsqueeze(0)
        features[level] = uv if features[level] is None else torch.cat([features[level], uv], dim=1)
    features = mdm.neck(features)
    depth = mdm.depth_head(features)[-1]
    depth = F.interpolate(depth, (out_size, out_size), mode="bilinear", align_corners=False)
    remap = getattr(mdm, "remap_depth_out", getattr(mdm, "remap_output", "exp"))
    if remap == "exp":
        depth = depth.exp()
    return depth.squeeze().float().cpu()


def disparity_pair(dt: torch.Tensor, dp: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Jointly normalized disparity maps for display (shared scale => comparable colors)."""
    it, ip = 1.0 / dt.clamp_min(1e-4), 1.0 / dp.clamp_min(1e-4)
    lo = torch.minimum(it.min(), ip.min())
    hi = torch.maximum(it.max(), ip.max())
    scale = (hi - lo).clamp_min(1e-8)
    return ((it - lo) / scale).numpy(), ((ip - lo) / scale).numpy()


def _imshow(ax, img, title: str | None = None, cmap=None, vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.frame_ranges_json is None:
        args.frame_ranges_json = args.dataset_root / "frame_ranges.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with_depth = args.depth_moge_path is not None and args.depth_morgbd_path is not None
    wrapper, processor, meta = load_planner(args.checkpoint_dir, args, device, use_depth=with_depth)
    depth_pred_ok = with_depth and wrapper.depth_head is not None and any(
        (args.checkpoint_dir / n).exists() for n in ("depth_head.pt", "depth_head_refit.pt")
    )
    current_ok = False
    if getattr(wrapper, "current_plan_head", None) is not None and (args.checkpoint_dir / "current_plan_head.pt").exists():
        wrapper.current_plan_head.load_state_dict(
            torch.load(args.checkpoint_dir / "current_plan_head.pt", map_location="cpu", weights_only=False), strict=True)
        current_ok = True
        if getattr(wrapper, "current_depth_head", None) is not None and (args.checkpoint_dir / "current_depth_head.pt").exists():
            wrapper.current_depth_head.load_state_dict(
                torch.load(args.checkpoint_dir / "current_depth_head.pt", map_location="cpu", weights_only=False), strict=True)
        wrapper.eval()
        print("[viz] current heads loaded", flush=True)

    dino_encoder = T.DinoVideoTargetEncoder(
        ckpt_path=args.dino_teacher_ckpt, config_path=args.dino_teacher_config,
        input_size=args.dino_input_size, device=device,
    )
    depth_encoder = None
    if with_depth:
        depth_encoder = T.DepthTargetEncoder(
            moge_path=args.depth_moge_path, morgbd_path=args.depth_morgbd_path,
            input_size=256, num_tokens=args.grid_size * args.grid_size, device=device,
        )

    latent_len = wrapper.latent_len  # from ckpt meta (handles grouped shared+own latents)
    plan_sequence = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
    _offs = str(meta.get("keyframe_offsets", "") or "")
    offsets_override = [int(x) for x in _offs.split(",") if x.strip()] or None
    dataset = T.OnlineSemanticPlanDataset(
        dataset_root=args.dataset_root, frame_ranges_json=args.frame_ranges_json,
        num_keyframes=args.num_keyframes, sequence_length=args.sequence_length,
        keyframe_scheme=args.keyframe_scheme, keyframe_gamma=args.keyframe_gamma,
        max_samples=args.num_samples, seed=args.seed, offsets_override=offsets_override,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=T.Collator(processor=processor, plan_sequence=plan_sequence), pin_memory=True,
    )

    K, G = args.num_keyframes, args.grid_size
    model_dtype = next(wrapper.model.parameters()).dtype
    head_dtype = next(wrapper.plan_head.parameters()).dtype
    saved = []

    with torch.no_grad():
        for si, batch in enumerate(loader):
            stems = batch.pop("stems", ["?"])
            keyframes = batch.pop("keyframe_images")      # (1, K, H, W, 3) uint8
            current = batch.pop("current_image")          # (1, H, W, 3) uint8
            inputs = T.move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)

            image_hidden, plan_hidden = wrapper._forward_hiddens(**inputs)
            current_lat, future_hidden = wrapper._split_current(plan_hidden)
            video_lat, depth_lat = wrapper._split_latents(future_hidden)
            dino_pred = wrapper.plan_head(
                image_hidden.to(head_dtype), video_lat.to(head_dtype)
            ).float().cpu()[0].reshape(K, G * G, -1).numpy()
            depth_pred = None
            if depth_pred_ok:
                depth_pred = wrapper.depth_head(
                    image_hidden.to(head_dtype), depth_lat.to(head_dtype)
                ).float().cpu()[0].reshape(K, G * G, -1).numpy()

            cur = current.permute(0, 3, 1, 2).contiguous()
            kfs = [keyframes[:, j].permute(0, 3, 1, 2).contiguous() for j in range(K)]
            dino_tgt = dino_encoder.encode_future_keyframes(cur, kfs).float().cpu()[0]
            dino_tgt = dino_tgt.reshape(K, G * G, -1).numpy()
            depth_tgt = depth_cls = None
            if depth_encoder is not None:
                depth_tgt, depth_cls = depth_feats_with_cls(depth_encoder, kfs)  # (K,N,C), (K,C)
            # official-style CURRENT row: targets/preds for the current frame itself
            cur_row = None
            if current_ok:
                c_dino_pred = wrapper.current_plan_head(
                    image_hidden.to(head_dtype), current_lat.to(head_dtype)).float().cpu()[0].numpy()
                c_dino_tgt = dino_encoder.encode_future_keyframes(cur, [cur]).float().cpu()[0].numpy()
                c_depth_tgt = c_depth_cls = c_depth_pred = None
                if depth_encoder is not None:
                    c_depth_tgt, c_depth_cls = depth_feats_with_cls(depth_encoder, [cur])
                    if getattr(wrapper, "current_depth_head", None) is not None:
                        c_depth_pred = wrapper.current_depth_head(
                            image_hidden.to(head_dtype), current_lat.to(head_dtype)).float().cpu()[0].numpy()
                cur_row = (c_dino_tgt, c_dino_pred, c_depth_tgt, c_depth_cls, c_depth_pred)

            cols = ["Current (input)", "Future RGB"]
            if depth_tgt is not None:
                cols.append("Depth-Target")
                if depth_pred is not None:
                    cols.append("Depth-Pred")
            cols += ["DINO-Target", "DINO-Pred"]
            nc = len(cols)
            nrows = K + (1 if cur_row is not None else 0)
            fig, axes = plt.subplots(nrows, nc, figsize=(2.1 * nc, 2.1 * nrows))
            axes = np.atleast_2d(axes)
            if cur_row is not None:
                c_dt, c_dp, c_dep_t, c_dep_cls, c_dep_p = cur_row
                r, c = 0, 0
                _imshow(axes[r, c], current[0].numpy(), cols[c]); c += 1
                _imshow(axes[r, c], current[0].numpy(), "Current (target=self)"); c += 1
                if c_dep_t is not None:
                    if c_dep_p is not None:
                        dtm = decode_depth_dense(depth_encoder.morgbd, torch.from_numpy(c_dep_t[0]), c_dep_cls[0], G)
                        dpm = decode_depth_dense(depth_encoder.morgbd, torch.from_numpy(c_dep_p), c_dep_cls[0], G)
                        it, ip = disparity_pair(dtm, dpm)
                        _imshow(axes[r, c], it, "Depth-Target", cmap="turbo", vmin=0, vmax=1); c += 1
                        _imshow(axes[r, c], ip, "Depth-Pred", cmap="turbo", vmin=0, vmax=1); c += 1
                    else:
                        dtm = decode_depth_dense(depth_encoder.morgbd, torch.from_numpy(c_dep_t[0]), c_dep_cls[0], G)
                        it, _ = disparity_pair(dtm, dtm)
                        _imshow(axes[r, c], it, "Depth-Target", cmap="turbo", vmin=0, vmax=1); c += 1
                t_rgb, p_rgb = pca_rgb_pair(c_dt, c_dino_pred, G)
                _imshow(axes[r, c], t_rgb, "DINO-Target"); c += 1
                _imshow(axes[r, c], p_rgb, "DINO-Pred")
                axes[r, 0].set_ylabel("current", fontsize=10)
            _row0 = 1 if cur_row is not None else 0
            for k in range(K):
                r = _row0 + k
                show_t = (k == 0 and _row0 == 0)
                c = 0
                _imshow(axes[r, c], current[0].numpy(), cols[c] if show_t else None); c += 1
                _imshow(axes[r, c], keyframes[0, k].numpy(), cols[c] if show_t else None); c += 1
                if depth_tgt is not None:
                    dt_map = decode_depth_dense(
                        depth_encoder.morgbd, torch.from_numpy(depth_tgt[k]), depth_cls[k], G)
                    if depth_pred is not None:
                        dp_map = decode_depth_dense(
                            depth_encoder.morgbd, torch.from_numpy(depth_pred[k]), depth_cls[k], G)
                        it, ip = disparity_pair(dt_map, dp_map)
                        _imshow(axes[r, c], it, cols[c] if show_t else None, cmap="turbo", vmin=0, vmax=1); c += 1
                        _imshow(axes[r, c], ip, cols[c] if show_t else None, cmap="turbo", vmin=0, vmax=1); c += 1
                    else:
                        it, _ = disparity_pair(dt_map, dt_map)
                        _imshow(axes[r, c], it, cols[c] if show_t else None, cmap="turbo", vmin=0, vmax=1); c += 1
                t_rgb, p_rgb = pca_rgb_pair(dino_tgt[k], dino_pred[k], G)
                _imshow(axes[r, c], t_rgb, cols[c] if show_t else None); c += 1
                _imshow(axes[r, c], p_rgb, cols[c] if show_t else None)
                axes[r, 0].set_ylabel(f"kf{k + 1} (future)", fontsize=10)
            stem = Path(str(stems[0])).name
            fig.suptitle(f"{stem} — {Path(str(args.checkpoint_dir)).name} (future-only prediction)", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            out = args.output_dir / f"sample{si}_{stem[:24]}.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            saved.append(str(out))
            print(f"[viz] saved {out}", flush=True)

    print(json.dumps({"saved": saved, "depth_pred": bool(depth_pred_ok)}), flush=True)


if __name__ == "__main__":
    main()
