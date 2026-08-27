# GE-Act LTX SigLIP2 Semantic Guidance Implementation Plan

> Execute this plan inline on branch `semantic-guidance`. Preserve existing untracked artifacts and tests.

**Goal:** Restore the stable Genie-Envisioner LIBERO/LIBERO-Plus support files and add online, per-camera SigLIP2 semantic cross-attention to the LTX video model without changing the baseline data semantics.

**Architecture:** Keep the original two-camera GE-Act pipeline and LTX view attention intact. Extract four future-frame SigLIP2 16x16 token grids online, adapt them to the LTX width with explicit `(t,y,x)` coordinates, and inject same-camera semantic context through a zero-gated cross-attention branch in every transformer block. The base LTX remains trainable; SigLIP2, T5, and VAE remain frozen.

**Environment:** Python 3.10, PyTorch, Diffusers LTX modules, Transformers SigLIP2, Accelerate/DeepSpeed, PyAV, pytest. Use `/data/LFT-W02_data/.conda/envs/ge-act/bin/python` for checks.

---

## Task 1: Restore stable LIBERO source files

**Files:**
- Modify: `.gitignore`
- Add: `ge_act/data/__init__.py`
- Add: `ge_act/data/lerobot_like_dataset.py`
- Add: `ge_act/data/utils/__init__.py`
- Add: `ge_act/data/utils/statistics.py`
- Add: `ge_act/data/utils/utils.py`
- Add: `ge_act/experiments/eval_libero.py`
- Add: `ge_act/experiments/eval_libero_plus.py`
- Add: `ge_act/experiments/__init__.py`
- Add: `ge_act/scripts/train.sh`
- Add: `ge_act/requirements.txt`
- Test: `tests/test_ge_act_source_completeness.py`

1. Write an import/completeness test that asserts the stable dataset, LIBERO and LIBERO-Plus entrypoints are present and the dataset returns the established `[C,V,T,H,W]` contract.
2. Run the test and confirm it fails because `ge_act/data` is absent.
3. Vendor only the required files from `/data/LFT-W02_data/junjie/VLA_WM/Genie-Envisioner-V1`, preserving camera order, frame sampling, normalization and PyAV decoding.
4. Add narrow `.gitignore` negations for `ge_act/data/**`.
5. Run the test and import/compile checks.

## Task 2: Add semantic conditioning primitives

**Files:**
- Add: `ge_act/models/ltx_models/semantic_conditioning.py`
- Modify: `ge_act/models/ltx_models/transformer_ltx_multiview.py`
- Test: `tests/test_ge_act_ltx_semantic_guidance.py`

1. Write failing tests for future keyframes `[0,3,5,8]`, normalized LTX times, 16x16 token geometry, same-camera attention isolation and zero-gate base parity.
2. Implement the online frozen SigLIP2 encoder using `AutoModel`, penultimate vision hidden states, no pooling, bf16/no-grad and configurable microbatching.
3. Implement the shared 1024-to-2048 semantic adapter, `(t,y,x)` coordinate MLP and camera/type embeddings.
4. Add a dedicated semantic attention processor accepting separate query/key RoPE tensors and never rearranging camera views.
5. Add low-rank timestep-conditioned semantic AdaLN and zero-initialized residual gate to configured LTX blocks.
6. Extend LTX rotary embedding to generate frequencies from explicit semantic positions with the same interpolation/frequency formula as video queries.
7. Run focused tests until green.

## Task 3: Wire training, optimizer and checkpoint contracts

**Files:**
- Modify: `ge_act/runner/ge_trainer.py`
- Modify: `ge_act/utils/model_utils.py`
- Modify: `ge_act/utils/__init__.py`
- Test: `tests/test_ge_act_semantic_training_contract.py`

1. Write failing tests for frozen encoders, differential optimizer groups, frame-rate propagation, semantic dropout view sharing, gradient flow and single-file/sharded checkpoint resolution.
2. Preserve an unjittered video clone for SigLIP2 targets; continue color jitter only on the VAE training input.
3. Extract four future frames per camera, encode online and pass semantic tokens/times/mask through `forward_pass`.
4. Set the effective training frame rate from source FPS and action/video stride (20/4 = 5 FPS for the LIBERO config), removing the hard-coded 30 FPS path.
5. Use base LTX LR `2e-5`, semantic LR `1e-4`, AdamW betas `(0.9,0.95)`, weight decay `1e-5`, warmup 1000, clipping 1.0; explicitly freeze SigLIP2/T5/VAE.
6. Make checkpoint-directory loading support either a shard index or one safetensors file; fix rank-zero logging.
7. Run focused tests and a small backward smoke test.

## Task 4: Wire inference/validation and CFG

**Files:**
- Modify: `ge_act/models/pipeline/custom_pipeline.py`
- Modify: `ge_act/runner/ge_trainer.py`
- Test: `tests/test_ge_act_semantic_pipeline.py`

1. Write failing tests for canonical `[B,V,L,D]` semantic input flattening, explicit semantic times and classifier-free-guidance duplication order.
2. Add optional `semantic_plan`, `semantic_plan_times`, and `semantic_condition_mask` pipeline arguments.
3. Validate shapes, flatten cameras consistently with video latents and duplicate semantic conditioning for CFG while keeping it identical in unconditional/conditional text branches.
4. Add validation modes for no-semantic ablation, online GT oracle, and externally supplied semantic tensors; default the semantic training config to GT oracle validation.
5. Run pipeline contract tests.

## Task 5: Add a conservative training config and launcher

**Files:**
- Add: `ge_act/configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml`
- Add: `ge_act/scripts/train_ltx_siglip2.sh`
- Add: `ge_act/scripts/preflight_ltx_siglip2.py`
- Modify: `ge_act/VENDORED_FROM.md`
- Test: `tests/test_ge_act_siglip2_config.py`

1. Write a failing config test covering 4 keyframes, 256 tokens/frame, all 28 layers, gradient checkpointing, 30k steps and effective global batch 128 on 8 GPUs.
2. Derive the new config from the stable FastWAM LIBERO config instead of changing the baseline config.
3. Start with per-GPU batch 2 and accumulation 8; keep a documented path to test 4/8 after memory smoke tests.
4. Add preflight checks for dataset metadata, SigLIP2 weights, LTX checkpoint layout, required Python modules and writable output space.
5. Add an eight-GPU launcher that runs preflight before training.
6. Run config/preflight unit tests without launching a remote training job.

## Task 6: Full verification and clean handoff

**Files:** all modified files above.

1. Run all newly added GE-Act tests.
2. Run `compileall` and import smoke checks in the GE-Act environment.
3. Run a tiny CPU/CUDA model forward/backward smoke test where resources permit.
4. Inspect `git diff --check`, `git status --short` and the final diff; ensure no generated assets, caches or unrelated user files are staged.
5. Report remaining environment/data prerequisites separately from code readiness.
