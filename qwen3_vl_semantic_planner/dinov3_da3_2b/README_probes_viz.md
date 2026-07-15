# 2B DINOv3+DA3 planner — feature probes & visualization toolkit

Qualitative visualization for the Qwen3-VL-2B DINOv3+DA3 LIBERO planner (the k1_zero2
FastWAM line). The planner aligns, in feature space, to **DINOv3** (last-layer patch
features, 1280-d) and **Depth-Anything-3** (last-layer encoder features, 2048-d). These
scripts decode those features back to something viewable and compare the planner's
PREDICTED features against the frozen teachers' TARGET features.

> **Dependency note:** these scripts import the **k1_zero2 trainer**
> (`train_qwen3vl4b_lingbot_dino_planner.py` with `PlannerWrapper.from_exported_checkpoint`,
> `predict_current_future_plans`, pooled head-query embeddings) which lives on the
> pod/Ola copy, **not** the DROID-line trainer of the same name in this repo. Run them on
> Ola (`/data/users/junjie/code/VLM4WAM_k1_zero2_bidir/scripts/qwen3_vl_semantic_planner/`),
> not locally, unless you point them at that trainer.

## Files
- `train_feature_probes.py` — trains the decoder probes (`--which dino|da3|dino_up|da3_v2|both`).
  Defines the probe architectures (`ProbeDecoder`, `FeatureUpsampler`, `MiniDPTProbe`) that
  the viz imports. Frozen backbones; probes train on the LIBERO frame cache.
- `visualize_qwen3vl2b_dinov3_da3_split.py` — **the canonical viz.** Per sample, two PNGs
  (`_main` / `_wrist`), each a 3-row × 6-col grid, all square 224×224 panels:
  - Row0 **DINO hi-res PCA** — features upsampled (FeatUp-style, `FeatureUpsampler`) then PCA(3)
    fit jointly per (target,pred) pair.
  - Row1 **Depth (probe v2)** — `MiniDPTProbe` decode of the DA3 features → turbo disparity.
  - Row2 **DA3-full GT** — real 4-layer DPT depth reference.
- `diag_depth_single_layer.py`, `diag_v2_compare.py` — depth diagnostics (single-layer
  bottleneck: probe-from-real vs DA3-full GT; probe_v2 vs v1 error tables).

## Probe checkpoints (binaries — not in git)
On Ola: `/data/users/junjie/probes_2b/`. Local backup of the keepers (gitignored):
`outputs/probes_2b_backup/`.
- `dino_upsample_probe.pt` (55M) — DINOv3 16×16→32×32 feature upsampler (loss 0.073). **keeper**
- `da3_depth_v2_probe.pt` (27M) — mini-DPT single-layer→depth, SILog+grad, distilled from
  DA3-full (loss 0.0154; probe-from-real vs GT AbsRel 0.127, corr 0.994). **keeper**
- `dino_rgb_probe.pt`, `da3_depth_probe.pt` (v1) — superseded (early RGB-recon / plain
  upsampler iterations), kept on Ola only.

## Key facts
- LIBERO is **2 cameras** concatenated horizontally into `[224,448]`; the teacher `_prep`
  squishes that to square, so the 16×16 feature grid's left half = main cam, right half =
  wrist cam. The split viz crops each panel at the midpoint → un-squished 224×224 per camera.
- The depth MAP is a **post-hoc single-layer probe decode**, not what the planner trains on
  (it trains on the raw 2048-d last-layer features via smooth-L1). Depth softness vs the
  4-layer DA3-full GT is a viz limitation, not a training-target defect.
- Checkpoint reload uses `from_exported_checkpoint`; its pooled-mode plan-token check was
  fixed to compare in fp32 (rtol/atol 2e-2, warn-only) so bf16 drift no longer blocks load.
</content>
