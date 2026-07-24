# OLA Dual-Camera K4 VLM Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a fresh Qwen3-VL-2B planner on OLA that consumes separate main/wrist images and predicts four view-aligned future SigLIP2 grids for GE-Act.

**Architecture:** One Qwen conversation contains two ordered image slots. The wrapper preserves two image-token spans, applies shared K4 SigLIP2/DA3 heads once per view, and exports `[B,2,4,256,1024]` future SigLIP2 features under a strict dual-camera K4 contract.

**Tech Stack:** Python, PyTorch, Transformers Qwen3-VL, Accelerate, DeepSpeed ZeRO-2, SigLIP2, Depth Anything 3, pytest, Bash, OLA 8xH100.

## Global Constraints

- Start from `/data/users/junjie/vlm4wam_2b/weights/Qwen3-VL-2B-Instruct`; do not load any existing planner checkpoint.
- Camera order is `main=0`, `wrist=1`; no horizontal concatenation or camera-stream copying is allowed.
- Future offsets are exactly `[2,4,6,8]` within a nine-frame future window.
- SigLIP2 output is exactly `[B,2,4,256,1024]`.
- Use 384 planner tokens: per keyframe, 32 shared + 32 SigLIP-private + 32 depth-private.
- Freeze the Qwen vision tower; train the language backbone, planner-token embeddings, and prediction heads.
- Use eight H100 GPUs, per-GPU batch 8, gradient accumulation 2, global batch 128, bf16, ZeRO-2.
- Train 30,000 optimizer steps; save only 20k, 25k, and 30k.
- Preserve unrelated untracked files in the source checkout.

---

### Task 1: Generalize the dual-camera dataset and teacher batch to K4

**Files:**
- Modify: `qwen3_vl_semantic_planner/ge_act_dual_camera.py`
- Modify: `qwen3_vl_semantic_planner/train_semantic_planner.py`
- Test: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Produces: `GEActDualCameraPlannerDataset(..., future_offsets: Sequence[int])`.
- Produces: sample field `future_camera_images: Tensor[2,K,H,W,3]`.
- Produces: `encode_dual_camera_future_targets(...) -> {semantic_plan_labels, depth_plan_labels}` with flattened target length `K*256`.

- [ ] **Step 1: Write failing dataset tests**

Add tests that construct a synthetic video whose value encodes camera and time, then require:

```python
dataset = GEActDualCameraPlannerDataset(
    FakeDataset(sample), n_previous=4, future_offsets=(2, 4, 6, 8)
)
item = dataset[0]
assert item["current_camera_images"].shape == (2, height, width, 3)
assert item["future_camera_images"].shape == (2, 4, height, width, 3)
assert torch.equal(item["future_camera_images"][:, 0], expected_offset_2)
assert torch.equal(item["future_camera_images"][:, 3], expected_offset_8)
```

Also assert duplicate, non-increasing, zero, or out-of-window offsets raise `ValueError`.

- [ ] **Step 2: Run the new dataset tests and verify red**

Run:

```bash
PYTHONPATH=$PWD pytest -q tests/test_ge_act_dual_camera_planner.py -k 'k4 or future_offsets'
```

Expected: failures because the current constructor accepts only `future_offset=8` and returns no K dimension.

- [ ] **Step 3: Implement K4 sampling**

Replace the scalar offset with a validated tuple and gather frames without losing view order:

```python
def __init__(
    self,
    dataset: Dataset,
    *,
    n_previous: int = 4,
    future_offsets: Sequence[int] = (2, 4, 6, 8),
):
    offsets = tuple(int(value) for value in future_offsets)
    if not offsets or tuple(sorted(set(offsets))) != offsets or offsets[0] <= 0:
        raise ValueError(f"future_offsets must be strictly increasing positive integers: {offsets}")
    self.future_offsets = offsets
```

In `__getitem__`, form indices as `self.n_previous + offset` and return:

```python
future = (
    video[:, :, future_indices]
    .permute(1, 2, 3, 4, 0)
    .contiguous()
)  # [2,K,H,W,3]
```

- [ ] **Step 4: Write failing K4 teacher-target tests**

Use recording fake teachers and assert the teacher receives four frames for each of the two views, main/wrist values remain distinct, and outputs have:

```python
assert targets["semantic_plan_labels"].shape == (batch, 2, 4 * 256, 1024)
assert targets["depth_plan_labels"].shape == (batch, 2, layers, 4 * 256, depth_width)
```

- [ ] **Step 5: Implement future-only K4 teacher encoding**

Add `encode_dual_camera_future_targets` that converts `[B,V,K,H,W,3]` to a list of K tensors `[B*V,3,H,W]`, calls each teacher's `encode_future_keyframes`, then restores the view dimension. Do not produce current-alignment labels in this route.

- [ ] **Step 6: Run Task 1 tests and commit**

Run:

```bash
PYTHONPATH=$PWD pytest -q tests/test_ge_act_dual_camera_planner.py
```

Expected: all tests in the file pass.

Commit:

```bash
git add qwen3_vl_semantic_planner/ge_act_dual_camera.py \
  qwen3_vl_semantic_planner/train_semantic_planner.py \
  tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): load dual-camera K4 targets"
```

---

### Task 2: Support future-only K4 WSA heads and strict export metadata

**Files:**
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/depth_anything3_target.py`
- Modify: `qwen3_vl_semantic_planner/train_semantic_planner.py`
- Test: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Produces: DA3 `encode_future_keyframes(...) -> Tensor[B,L,K*256,D]` for `wsa_multilayer`.
- Produces: `predict_dino_depth_plan` outputs with view shape `[B,2,K*256,*]`.
- Produces: `validate_dual_camera_export_metadata` for K4 offsets and 384 planner tokens.

- [ ] **Step 1: Write failing future-only WSA tests**

Add tests requiring `PlannerWrapper` to accept:

```python
PlannerWrapper(
    ..., num_keyframes=4, num_camera_views=2,
    use_depth=True, use_current_alignment=False,
    shared_latent_per_keyframe=32,
    private_latent_per_keyframe=32,
    da3_align_strategy="wsa_multilayer",
    da3_num_layers=4,
)
```

Require `latent_len == 384`, plan/depth predictions preserve view dimension, and `_reshape_depth_target` accepts `[B,2,L,K*256,D]`.

- [ ] **Step 2: Run the tests and verify red**

Run:

```bash
PYTHONPATH=$PWD pytest -q tests/test_ge_act_dual_camera_planner.py -k 'wsa and k4'
```

Expected: constructor fails with `wsa_multilayer requires ... current alignment`.

- [ ] **Step 3: Implement K4 WSA teacher encoding**

In `DA3DepthTargetEncoder.encode_future_keyframes`, batch all K preprocessed frames. When `_patch_tokens` returns `[K*B,L,256,D]`, restore keyframe-major output as:

```python
features = features.view(k, b, layers, tokens, width)
features = features.permute(1, 2, 0, 3, 4)
return features.reshape(b, layers, k * tokens, width).detach().to(torch.bfloat16)
```

Keep the existing last-layer return contract unchanged.

- [ ] **Step 4: Generalize wrapper validation and view-aware reshaping**

Allow WSA when depth is enabled and either current alignment is used or the planner is future-only K4. Make `_reshape_depth_target` preserve every leading batch/view dimension and convert `[...,L,T,D]` to `[...,T,L,D]` using `transpose(-3, -2)`.

Ensure `predict_semantic_plan` routes through the view-aware K4 head path when `num_camera_views == 2`; it must not call `self.plan_head` once on a four-dimensional image tensor.

- [ ] **Step 5: Replace fixed K1 metadata with geometry-derived K4 metadata**

Build the dual-camera metadata from wrapper and offsets at save time. Validate these exact values:

```python
{
    "planner_input_layout": "separate_camera_images",
    "camera_names": ["main", "wrist"],
    "num_camera_views": 2,
    "camera_head_sharing": "shared_head_per_view_image_context",
    "semantic_output_layout": "batch_view_keyframe_token_feature",
    "semantic_teacher": "siglip2-large-patch16-256",
    "future_keyframe_offsets": [2, 4, 6, 8],
    "num_keyframes": 4,
    "target_tokens_per_keyframe": 256,
    "planner_token_count": 384,
}
```

Keep K1 loader compatibility only when its own metadata explicitly describes K1; never accept the OLA composite checkpoint as dual-camera.

- [ ] **Step 6: Run Task 2 tests and commit**

Run:

```bash
PYTHONPATH=$PWD pytest -q tests/test_ge_act_dual_camera_planner.py
```

Commit:

```bash
git add qwen3_vl_semantic_planner/dinov3_da3_2b/depth_anything3_target.py \
  qwen3_vl_semantic_planner/train_semantic_planner.py \
  tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): predict dual-camera K4 semantic plans"
```

---

### Task 3: Generalize the frozen GE-Act provider to dual-camera K4

**Files:**
- Modify: `ge_act/models/ltx_models/vlm_semantic_planner.py`
- Test: `tests/test_ge_act_vlm_semantic_planner.py`

**Interfaces:**
- Produces: `DualCameraSemanticPlan.semantic_tokens: Tensor[B,2,4,256,1024]`.
- Produces: `DualCameraSemanticPlan.times: Tensor[B*2,4]` with normalized values `[0.25,0.5,0.75,1.0]`.

- [ ] **Step 1: Write failing K4 provider tests**

Create a fake wrapper returning `future_dino=[B,2,4*256,1024]`. Require the provider to reshape without copying views:

```python
plan = provider.predict(images, instructions)
assert plan.semantic_tokens.shape == (batch, 2, 4, 256, 1024)
assert torch.equal(
    plan.times[0],
    torch.tensor([0.25, 0.5, 0.75, 1.0]),
)
```

Require K1, missing-view, reordered-camera, and composite metadata to fail.

- [ ] **Step 2: Run provider tests and verify red**

Run:

```bash
PYTHONPATH=$PWD pytest -q tests/test_ge_act_vlm_semantic_planner.py -k k4
```

- [ ] **Step 3: Implement metadata-driven K4 reshape**

Read `num_keyframes`, `future_keyframe_offsets`, `sequence_length`, and `target_tokens_per_keyframe` from validated metadata. Convert `[B,2,1024,1024]` into `[B,2,4,256,1024]` and construct normalized times from `offset/(sequence_length-1)`.

Select required checkpoint files from the exported geometry: future-only K4
requires `plan_head.pt`, `depth_head.pt`, `plan_token_embedding.pt`, model, and
processor, but must not require the K1-only `current_plan_head.pt` or
`current_depth_head.pt` files.

- [ ] **Step 4: Run provider tests and commit**

Run:

```bash
PYTHONPATH=$PWD pytest -q tests/test_ge_act_vlm_semantic_planner.py
```

Commit:

```bash
git add ge_act/models/ltx_models/vlm_semantic_planner.py \
  tests/test_ge_act_vlm_semantic_planner.py
git commit -m "feat(ge-act): load dual-camera K4 planner exports"
```

---

### Task 4: Add the OLA data contract, preflight, and production launcher

**Files:**
- Create: `ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml`
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_k4_siglip2da3_ola.sh`
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/preflight_dual_camera_k4.py`
- Modify: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Produces: an OLA-only data YAML for `CustomLeRobotDataset` with two ordered cameras.
- Produces: a launcher with `RUN_KIND=smoke|formal` and no checkpoint warm start.

- [ ] **Step 1: Write failing launcher/config contract tests**

Parse the YAML and shell script as text and require:

```python
assert train["valid_cam"] == [
    "observation.images.image",
    "observation.images.wrist_image",
]
assert train["chunk"] == 9
assert train["n_previous"] == 4
assert "NUM_KEYFRAMES=${NUM_KEYFRAMES:-4}" in launcher
assert "USE_CURRENT_ALIGNMENT=${USE_CURRENT_ALIGNMENT:-0}" in launcher
assert "BATCH_SIZE=${BATCH_SIZE:-8}" in launcher
assert "GRAD_ACCUM=${GRAD_ACCUM:-2}" in launcher
assert "EXPECTED_GLOBAL_BATCH=${EXPECTED_GLOBAL_BATCH:-128}" in launcher
assert "SAVE_START_STEP=${SAVE_START_STEP:-20000}" in launcher
assert "INIT_PLANNER_CHECKPOINT=" in launcher
```

- [ ] **Step 2: Add the OLA data YAML**

Configure the four LIBERO domains under `/data/shared/datasets/libero_fastwam`, `source_fps=20`, `chunk=9`, `n_previous=4`, `valid_cam` in main/wrist order, `sample_size=[256,256]`, and raw decoding without requiring a predecoded cache.

- [ ] **Step 3: Add a fail-closed preflight**

The Python preflight must instantiate one sample, assert source video shape `[3,2,T,H,W]`, build K4 planner input, validate 384 plan tokens, confirm the base Qwen/SigLIP2/DA3 paths, and verify the configured output filesystem has at least 100 GiB free.

- [ ] **Step 4: Add the production launcher**

The formal preset exports:

```bash
MODEL_PATH=/data/users/junjie/vlm4wam_2b/weights/Qwen3-VL-2B-Instruct
NUM_GPUS=8
NUM_KEYFRAMES=4
SEQUENCE_LENGTH=9
KEYFRAME_SCHEME=even_future
USE_CURRENT_ALIGNMENT=0
SHARED_LATENT_PER_KEYFRAME=32
PRIVATE_LATENT_PER_KEYFRAME=32
BATCH_SIZE=8
GRAD_ACCUM=2
EXPECTED_GLOBAL_BATCH=128
LR=3e-5
HEAD_LR=3e-4
WARMUP_STEPS=2500
MAX_STEPS=30000
SAVE_STEPS=5000
SAVE_START_STEP=20000
FULL_FINETUNE=1
USE_DEEPSPEED=1
VIDEO_TARGET_TYPE=siglip2
DEPTH_TARGET_TYPE=da3
DA3_ALIGN_STRATEGY=wsa_multilayer
```

Set `INIT_PLANNER_CHECKPOINT=` and `HEAD_WARMSTART_CKPT=` explicitly. The smoke preset uses one GPU, batch 1, accumulation 1, one optimizer step, and a separate output directory.

- [ ] **Step 5: Run Task 4 tests and commit**

Run:

```bash
PYTHONPATH=$PWD pytest -q tests/test_ge_act_dual_camera_planner.py
bash -n qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_k4_siglip2da3_ola.sh
```

Commit:

```bash
git add ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml \
  qwen3_vl_semantic_planner/dinov3_da3_2b/preflight_dual_camera_k4.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_k4_siglip2da3_ola.sh \
  tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): add OLA dual-camera K4 launcher"
```

---

### Task 5: Run full local verification and publish the implementation

**Files:**
- Verify only; no new files expected.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=$PWD pytest -q \
  tests/test_ge_act_dual_camera_planner.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_siglip2_config.py
```

Expected: zero failures.

- [ ] **Step 2: Run syntax and whitespace checks**

```bash
python -m py_compile \
  qwen3_vl_semantic_planner/ge_act_dual_camera.py \
  qwen3_vl_semantic_planner/train_semantic_planner.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/depth_anything3_target.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/preflight_dual_camera_k4.py \
  ge_act/models/ltx_models/vlm_semantic_planner.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Push the feature branch**

```bash
git push origin ge-act-dual-camera-planner
```

Record the pushed commit SHA for the OLA deployment log.

---

### Task 6: Deploy to OLA, smoke test, and launch the formal 30k run

**Files:**
- Remote checkout: `/data/users/junjie/code/VLM4WAM_dual_camera_k4`
- Remote smoke log: `/data/users/junjie/logs/vlm_dual_camera_k4_smoke.log`
- Remote formal log: `/data/users/junjie/logs/vlm_dual_camera_k4_30k.log`
- Remote formal output: `/data/users/junjie/code/VLM4WAM_dual_camera_k4/outputs/qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa`

- [ ] **Step 1: Verify OLA resources and preserve the current checkpoint**

Check all GPUs, current processes, free disk, and the existing composite
`step_020000`. Do not stop the current run until the new CPU preflight and tests pass.

- [ ] **Step 2: Create a fresh remote checkout and run CPU preflight**

Clone or fetch `ge-act-dual-camera-planner` into the dedicated directory, verify its SHA equals the pushed SHA, then run focused tests and:

```bash
/data/users/junjie/envs/vlm4wam/bin/python \
  qwen3_vl_semantic_planner/dinov3_da3_2b/preflight_dual_camera_k4.py \
  --config ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml
```

Expected: preflight reports two cameras, offsets `[2,4,6,8]`, 384 planner tokens, and no errors.

- [ ] **Step 3: Stop only the old composite K4 training after preflight passes**

Match the exact output directory `qwen3vl2b_siglip2_da3_libero_future_k4_wsa`, send SIGTERM to its Accelerate launcher, wait for all eight workers to exit, and verify the existing `step_020000` files remain intact. Do not delete its output directory.

- [ ] **Step 4: Run one-GPU smoke training**

```bash
RUN_KIND=smoke CUDA_VISIBLE_DEVICES=0 \
  bash qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_k4_siglip2da3_ola.sh \
  > /data/users/junjie/logs/vlm_dual_camera_k4_smoke.log 2>&1
```

Require exit 0, one optimizer update, finite SigLIP/DA3 losses, exported K4 metadata, and a provider prediction with shape `[1,2,4,256,1024]`.

- [ ] **Step 5: Launch the eight-GPU formal run**

Use a detached session if available, otherwise `nohup`, and record the launcher PID:

```bash
nohup env RUN_KIND=formal \
  bash qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_k4_siglip2da3_ola.sh \
  > /data/users/junjie/logs/vlm_dual_camera_k4_30k.log 2>&1 &
```

- [ ] **Step 6: Verify launch evidence**

Require eight trainer workers, nonzero utilization on all eight H100s, runtime JSON reporting batch 8 / accumulation 2 / global 128 / ZeRO-2 / bf16 / gradient checkpointing false, and at least one finite training-loss record. Report PID, log, output directory, step time, and estimated completion time.
