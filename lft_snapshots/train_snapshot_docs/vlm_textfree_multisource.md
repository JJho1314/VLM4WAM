# Text-free multi-source VLM guidance for Cosmos

Experiment: replace Cosmos's T5 text conditioning with InstructSAM's own
representations and see whether the VLM signal guides robot-video generation
better than free-text. This is a *validation* run on the lightweight
`droid_success_v21_..._scene_cap200_tasktarget` holdout (HPC3).

## What changes vs. the existing InstructSAM bridge

The current bridge (`..._instructsam_feature_context`) keeps the T5 text stream
and *prepends* a single InstructSAM `mask_query` feature (256-d) to it. Here we:

1. Extract **three** InstructSAM representations per sample instead of one:
   - **mask**   — `mask_hidden_fcs[0](seg_output_embeddings)`  → `[Lm, 256]`
     (SAM mask-decoder space; "what the target looks like as a mask").
   - **detect** — SAM3 `detr_decoder.intermediate_hidden_states[-1]`, best query
     per object by `pred_logits` → `[Ld, 256]` ("where / which object").
   - **vtext**  — Qwen3VL `language_model.last_hidden_state` at prefill,
     adaptive-pooled over the sequence → `[Lv, 4096]` ("instruction grounded in
     the image" — the actual VLM guidance signal).
2. **Fuse** them into one `[L, 256]` tensor at precompute time with fixed
   per-source budgets (default mask 16 / detect 16 / vtext 32 = 64 tokens).
   `vtext` is reduced 4096→256 by a fixed, seeded, orthogonal projection saved
   to disk (reproducible across shards / train / val). Mask & detect are already
   256-d. Storing a single `[L,256]` tensor means the existing dataloader and
   collate path are reused unchanged.
3. **Drop the text**: a new DiT flag `target_feature_context_replace_text=True`
   makes `crossattn_emb` *become* the fused feature tokens (the T5 stream is
   discarded inside the net). Cross-attention has no positional encoding on the
   context side, so a new `MultiSourceTargetFeatureContextAdapter` adds a learned
   per-source embedding by fixed segment so the model can tell mask vs. detect
   vs. vtext apart. TAVID alignment uses `token_source="feature"`.

## Why these choices

- **Single fused [L,256] tensor**: the dataloader (`VideoDataset`) loads exactly
  one `target_feature` per sample, normalizes to `target_feature_dim` and
  pads/truncates to `target_feature_max_tokens`. Storing the fusion as one tensor
  reuses all of that with zero collate changes. `target_feature_dim=256`,
  `target_feature_max_tokens=64`.
- **Fixed seeded vtext projection** (not learned): there is no trained 4096→256
  head for the full VLM hidden state. A seeded orthogonal/gaussian projection is
  information-preserving enough (JL) for a "does it help" validation, and the
  learnable DiT adapter (256→1024) still adapts the fused stream. Saved to
  `target_features_multisource/_vtext_proj_4096x256.pt` and reused everywhere.
- **Replace at the DiT, keep T5 running**: the conditioner still produces a T5
  `crossattn_emb` (from the caption); the net ignores it when `replace_text`.
  This avoids touching the conditioner's hard dependency on `t5_text_embeddings`
  and the caption-dropout logic. The model is genuinely text-free in
  conditioning; the only cost is a wasted T5 forward (cheap; embeddings are
  precomputed for these datasets).
- **Per-source segment embedding**: because context tokens have no positions,
  without a source tag the model sees an unordered bag and cannot route the three
  signals. Fixed budgets give a deterministic `[mask | detect | vtext]` layout so
  a learned `nn.Embedding(3, 1024)` can mark each segment.

## Files touched

- `cosmos_predict2/_src/predict2/target_aware/instructsam_multisource.py` (new)
  — extractor: registers forward hooks (by module type / attribute) for detect &
  vtext, assembles the three native-dim reps from one inference call.
- `scripts/precompute_instructsam_multisource_features.py` (new) — fuse → one
  `[L,256]` `target_feature` + metadata; seeded vtext projection.
- `cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py` — new
  `MultiSourceTargetFeatureContextAdapter`, new ctor flags
  `target_feature_context_replace_text`, `target_feature_context_source_segments`,
  replace-text branch in `append_target_feature_context`.
- `cosmos_predict2/experiments/base/robointer.py` — new dataloaders
  (`target_feature_dir="target_features_multisource"`) and experiment
  `predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource`.
- `scripts/sbatch_precompute_instructsam_multisource_*.sh`,
  `scripts/sbatch_train_droid_success_v21_instructsam_textfree_multisource.sh` —
  HPC3 launchers, dataset path via env var
  `DROID_SUCCESS_V21_TAVID_DIR` (default points at the cap200_tasktarget set).

## HPC3 run order

1. `sbatch scripts/sbatch_precompute_instructsam_multisource_strict_holdout_v3.sh`
   (writes `target_features_multisource/*.pt` into the dataset dir).
2. `sbatch scripts/sbatch_train_droid_success_v21_instructsam_textfree_multisource.sh`.
3. Compare validation samples / target-attention IoU against the
   `..._instructsam_feature_context` (text+mask) and `..._baseline` (text-only)
   runs already in `robointer.py`.

## Tunables (env / config)

- Per-source budgets: `--mask-tokens 16 --detect-tokens 16 --vtext-tokens 32`.
- `target_feature_context_replace_text` — set False to A/B against "text + 3 reps".
- vtext pooling: adaptive mean-pool to `--vtext-tokens` over the prefill sequence.
