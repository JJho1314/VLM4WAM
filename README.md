# VLM4WAM

Workspace for semantic-plan guided Cosmos Predict 2.5 robot video generation.

The current Cosmos copy has been reset from the clean upstream checkout at:

```text
/data/LFT-W02_data/junjie/VLA_WM/cosmos-predict2.5
```

Old target-aware experiment switches, explicit mask paths, and prior
branch-specific Cosmos paths have been removed from the active Cosmos tree. The
active Stage-2 route is now only:

```text
semantic_plan [B, L, 1152]
-> SemanticPlanContextAdapter
-> semantic cross-attention in Cosmos DiT blocks
-> video prediction
```

## Semantic planner (Qwen3-VL)

The `semantic_plan` fed to the world model is produced by a Qwen3-VL planner trained under
[`qwen3_vl_semantic_planner/`](qwen3_vl_semantic_planner/README.md). Three independent
lines: **CoVT·SigLIP·2B** (baseline), **tasktoken·SigLIP·2B** (rich-KV head variant), and
**lingbot-DINO·4B** (`lingbot_dino_4b/`, aligns to DINO-video via the open `robbyant/lingbot-vla-v2-6b`
weights — needs a matching DINO-conditioned WM). See that README for details.

## World model (Cosmos)

Main files:

- `cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/semantic_plan_conditioning.py`
- `cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`
- `cosmos-predict2.5/cosmos_predict2/experiments/base/semantic_plan.py`
- `cosmos-predict2.5/scripts/sbatch_train_semantic_plan_cosmos_2b_320x576_93f.sh`

Training entry:

```bash
cd cosmos-predict2.5
sbatch scripts/sbatch_train_semantic_plan_cosmos_2b_320x576_93f.sh
```

The script defaults to 93 frames, 320x576, SigLIP2 semantic plans with
`k=6, grid=9`, and global batch size 128 on 8 GPUs. Override paths and
hyperparameters with environment variables such as `DATASET_ROOT`,
`SEMANTIC_PLAN_DIR`, `CHECKPOINT_LOAD_PATH`, `BATCH_SIZE`,
`GRAD_ACCUM_ITER`, and `MAX_ITER`.

Conditioning behavior:

- `SEMANTIC_PLAN_DROPOUT_PROB` (default `0.15`): training-time probability of
  dropping the semantic-plan conditioning for a micro-batch, so the CFG
  unconditional branch (`semantic_plan=None` at inference) is a trained
  configuration.
- Keyframe times: the dataset reads `future_frame_indices` /
  `video_frame_indices` from the semantic-plan manifest and passes normalized
  keyframe times through to the DiT, so semantic-token RoPE/coord temporal
  positions match the true keyframe locations (labels sample keyframes from
  window positions `round(linspace(1, T-1, k))`, and k16->k8 selection is
  non-uniform). Manifests without frame indices fall back to the previous
  uniform-spacing assumption.
- Native-grid plans: with `SEMANTIC_PLAN_SPATIAL_GRID=0` the per-keyframe
  token count is inferred from `SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES`, so
  keyframe selection also works for native SigLIP2 grids (27x27 = 729
  tokens/frame) built with `--grid-size 0`.
- Online encoding (`SEMANTIC_PLAN_ONLINE=1`): SigLIP2 plans are encoded on the
  fly from the training video window by a frozen per-rank encoder
  (`OnlineSemanticPlanEncoder`, outside state_dict/EMA/FSDP), Baton-style — no
  .pt features are read, so native grids need no label storage. The manifest
  under `SEMANTIC_PLAN_DIR` still defines windows and VAE-latent pairing.
  Keyframe indices/times match the offline builder exactly; features match the
  offline teacher space (token cosine ~0.999, difference is only the resize
  implementation). Training-only: inference still takes
  `--semantic-plan-path`. The dropout decision is broadcast from rank 0 —
  per-rank divergence would desync FSDP all-gathers (the adapter is its own
  FSDP unit) and hang NCCL.
