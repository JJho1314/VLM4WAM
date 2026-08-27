# Semantic-guided LTX cross-attention (local, ge-act env)

Answers: does the semantic-plan-conditioned LTX (GE-Act line) show the object-focusing effect?

- **Model**: trained semantic LTX `step_25000` (25k steps, pulled from HPC3
  /data/user/jhe724/junjie/outputs/libero_fastwam_ltx_siglip2/2026_07_16_19_17_17/step_25000),
  has `semantic_adapter` + all 28 blocks with `semantic_attn` (video->SigLIP-plan cross-attn).
  Local copy: /data/LFT-W02_data/junjie/ltx_semantic_ckpt/ (4.7G, outside repo).
- **Deps** (all local): LTX-Video base (VAE AutoencoderKLLTXVideo + T5), SigLIP2, ge-act conda env
  (diffusers 0.35.2). Runs on A6000 (use GPU1 to avoid the shared box's OOM kills on GPU0).
- `m1_ltx_load.py` load milestone; `ltx_capture.py` single scene; `ltx_poster.py` -> ../figs/ltx_semantic_attn.png.
- **Method**: VAE-encode a LIBERO clip (4 mem + 9 future frames, 2 views, 256^2) -> latent tokens
  (8x8 spatial); semantic_plan = SigLIP2 of future keyframes [0,3,5,8]; capture video->plan attention
  (max-over-plan-tokens, blocks 8/12/16/20) via a wrapped LTXVideoSemanticAttentionProcessor.
- **Finding**: attention is object/manipulation-oriented (gripper, table objects, drawers), NOT uniform
  -> the focusing effect exists. But coarser than FastWAM (8x8 latent) and the matched-vs-shuffled-plan
  contrast is modest -> broadly object-driven rather than sharply plan-specific at 25k steps.

## Dual-conditioning fusion (updated)
LTX has BOTH text (attn2 -> T5 instruction) and semantic (semantic_attn -> SigLIP goal plan)
cross-attention. `ltx_poster.py` now captures both and FUSES them (geometric mean = where both agree):
- text attn: FastWAM-style fraction-on-instruction (softmax over caption, sum over valid tokens);
- semantic attn: per-keyframe goal (last keyframe [KFSEL=3]);
- fused = sqrt(norm(text) * norm(semantic)).
Figure ../figs/ltx_semantic_attn.png = 3 rows (text / semantic / fused). Text pathway gives the
sharpest instruction-driven object focus; fused concentrates on manipulation target/region.
Tuning notes: RES=256 (512->16x16 latent triggers rope grid artifacts), SS=0.15, per-keyframe goal.
