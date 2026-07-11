# LingBot K1 Current/Future Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and retrain a FastWAM-compatible Qwen3-VL 4B planner that faithfully uses LingBot-VLA 2.0 current/future alignment with eight current task queries, eight final-future task queries, current/future DINO and depth heads, and one future semantic frame.

**Architecture:** One Qwen forward consumes the observed image, instruction, eight current task tokens, and eight future task tokens. Four warm-started LingBot resampler heads predict current/future DINO and depth targets; only the two future predictions cross the provider boundary into FastWAM as `[B,256,1024]` plans at normalized time `1.0`.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers/Qwen3-VL, safetensors, Hydra/OmegaConf, pytest, bash, Slurm, 8×H100.

## Global Constraints

- Use exactly one future frame: FastWAM sampled-video offset `[8]`, the last image of the 33-record action window.
- Keep the FastWAM video sequence at nine RGB frames: current plus eight future frames at 5 FPS.
- Set `num_task_tokens=8` for current and `8` for future, for exactly `16` unique Qwen task tokens.
- Share current task hidden states across current DINO/depth heads and future task hidden states across future DINO/depth heads.
- Keep four independent heads and output query banks, warm-started from LingBot `current_video_align`, `future_video_align`, `depth_align`, and `future_depth_align` tensors.
- Use exactly `0.004` for current DINO MSE, future DINO MSE, current depth Smooth-L1, and future depth Smooth-L1.
- Keep cosine, norm, variance, InfoNCE, and CLS loss weights at zero.
- Export only future DINO/depth plans through the provider; each is `[B,256,1024]` with semantic time `[1.0]`.
- Preserve frozen teachers, frozen Qwen vision encoder, bf16, full language-backbone fine-tuning, gradient checkpointing, and clip norm `1.0`.
- Train on HPC3/jhe724 with global batch 128, 12,000 optimizer steps, backbone LR `3e-5`, head LR `3e-4`, 1,000 warmup steps, and cosine decay.
- Preserve unrelated dirty-worktree changes and stage only the files named by each task.

---

## File and Responsibility Map

- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_video_target.py`
  returns current and one future DINO target from one teacher call.
- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/depth_target.py`
  batches current and future depth target encoding.
- `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`
  owns the 16-token query layout, four heads, loss, warm start, optimizer, checkpoint, and data loop.
- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_depth_plan_provider.py`
  validates/loads the strict K1 checkpoint and returns only future plans.
- `third_party/FastWAM/src/fastwam/models/cosmos/online_semantic_planner.py`
  independently validates the external K1 provider metadata before model allocation.
- `third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py`
  accepts one semantic time and 256 fused semantic tokens.
- `third_party/FastWAM/configs/model/fastwam_cosmos.yaml`
  pins semantic source/consumer keyframes to one and max tokens to 256.
- `third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml`
  pins file/online semantic capacity to 256 tokens.
- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1.sh`
  pins the strict LingBot K1 training contract.
- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1_hpc3.sbatch`
  launches exact-config smoke/formal runs and enforces global batch 128.

---

### Task 1: Produce Current and Final-Future Teacher Targets

**Files:**
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_video_target.py`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/depth_target.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`

**Interfaces:**
- Produces: `DinoVideoTargetEncoder.encode_current_and_future(current_b3hw, future_b3hw) -> tuple[Tensor, Tensor]`, ordered `(current, future)`, each `[B,256,1024]`.
- Produces: `DepthTargetEncoder.encode_current_and_future(current_b3hw, future_b3hw) -> tuple[Tensor, Tensor]`, ordered `(current, future)`, each `[B,256,1024]`.
- Produces: `FastWAMOnlinePlannerDataset.offsets == (8,)` and samples `[current, future@8]`.

- [ ] **Step 1: Write failing teacher and dataset tests**

Add lightweight fake-teacher tests that require one DINO call with `return_current=True`, `current_index=1`, and a three-frame `[current,current,future]` clip. Add a fake depth encoder test that verifies `_depth_target` receives one concatenated `2*B` batch and splits current/future correctly. Change the FastWAM dataset assertion to:

```python
assert dataset.offsets == (8,)
item = dataset[0]
assert item["keyframe_images"].shape == (1, 2, 3, 3)
assert torch.equal(item["current_image"], source_video[0])
assert torch.equal(item["keyframe_images"][0], source_video[8])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_lingbot_dino_depth_contract.py \
  -k 'current_and_future_target or fastwam_online_dataset'
```

Expected: failures because both `encode_current_and_future` methods are absent and dataset offsets are still `[2,4,6,8]`.

- [ ] **Step 3: Implement the minimal DINO API**

Implement one clip and one teacher call:

```python
@torch.no_grad()
def encode_current_and_future(self, current_b3hw, future_b3hw):
    current = self._prep(current_b3hw)
    future = self._prep(future_b3hw)
    video = torch.stack([current, current, future], dim=2)
    video = (video - self.mean) / self.std
    future_patch, current_patch = self.teacher.get_future_feature(
        video,
        return_current=True,
        current_index=1,
        fps=1.0,
    )
    return current_patch.detach(), future_patch.detach()
```

Store `effective_fps` on the encoder and pass that value rather than a second hardcoded constant.

- [ ] **Step 4: Implement the minimal depth API and K1 dataset selection**

```python
@torch.no_grad()
def encode_current_and_future(self, current_b3hw, future_b3hw):
    batch = torch.cat([self._prep(current_b3hw), self._prep(future_b3hw)], dim=0)
    features = self._depth_target(batch).detach().to(torch.bfloat16)
    batch_size = current_b3hw.shape[0]
    return features[:batch_size], features[batch_size:]
```

Set `FastWAMOnlinePlannerDataset.offsets = (8,)`; retain the existing `selected = video[:, [0, *self.offsets]]` conversion.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit teacher/data changes**

```bash
git add tests/test_lingbot_dino_depth_contract.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_video_target.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/depth_target.py \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py
git commit -m "feat: add lingbot current future targets"
```

---

### Task 2: Build the 16-Token Four-Head Planner

**Files:**
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`

**Interfaces:**
- Produces: `PlannerWrapper.split_current_future_task_hidden(plan_hidden) -> tuple[current_hidden, future_hidden]` with each `[B,8,H]`.
- Produces: `PlannerWrapper.predict_current_future_plans(...) -> dict[str, Tensor]` containing `current_dino`, `future_dino`, `current_depth`, `future_depth`, each `[B,256,1024]`.
- Preserves: `predict_dino_depth_plan(...) -> tuple[future_dino, future_depth]` for the provider.

- [ ] **Step 1: Write failing query-layout and one-forward tests**

Construct a wrapper without loading Qwen, assign four counting fake heads, and require:

```python
wrapper.num_task_tokens = 8
wrapper.latent_len = 16
hidden = torch.arange(2 * 16 * 4).reshape(2, 16, 4)
current, future = wrapper.split_current_future_task_hidden(hidden)
assert torch.equal(current, hidden[:, :8])
assert torch.equal(future, hidden[:, 8:])

plans = wrapper.predict_current_future_plans(input_ids=torch.ones(2, 4))
assert wrapper.vlm_forward_calls == 1
assert set(plans) == {
    "current_dino", "future_dino", "current_depth", "future_depth"
}
assert all(value.shape == (2, 256, 1024) for value in plans.values())
```

Also assert `predict_dino_depth_plan` returns only `future_dino` and `future_depth`.

- [ ] **Step 2: Run tests and verify RED**

```bash
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_lingbot_dino_depth_contract.py \
  -k 'split_current_future or four_head or predict_dino_depth_plan'
```

Expected: failures because the split API and current heads do not exist.

- [ ] **Step 3: Replace shared/private geometry with LingBot task segments**

For strict `lingbot_dino + use_depth` mode, set:

```python
self.num_task_tokens = 8
self.num_temporal_query_groups = 2
self.latent_len = self.num_task_tokens * self.num_temporal_query_groups
self.target_len = 16 * 16
```

Build four `LingbotDinoPlanHead` instances with `num_keyframes=1` and `num_latent_per_keyframe=8`. Keep future names `plan_head` and `depth_head`; add `current_plan_head` and `current_depth_head`.

- [ ] **Step 4: Implement split and prediction methods**

```python
def split_current_future_task_hidden(self, plan_hidden):
    expected = 2 * self.num_task_tokens
    if plan_hidden.shape[1] != expected:
        raise RuntimeError(f"expected {expected} current/future task tokens")
    return plan_hidden[:, :self.num_task_tokens], plan_hidden[:, self.num_task_tokens:]
```

Call `_forward_hiddens` once, run current/future hidden states through the corresponding DINO/depth heads, and validate all four output shapes before returning the dictionary.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 6: Commit wrapper changes**

```bash
git add tests/test_lingbot_dino_depth_contract.py \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py
git commit -m "feat: add lingbot current future planner heads"
```

---

### Task 3: Match LingBot Losses, Warm Start, Optimizer, and Checkpoints

**Files:**
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`

**Interfaces:**
- Consumes four target tensors named `current_dino_labels`, `semantic_plan_labels`, `current_depth_labels`, and `depth_plan_labels`.
- Produces metadata fields `current_alignment`, `num_task_tokens`, `num_temporal_query_groups`, `current_dino_loss_weight`, `future_dino_loss_weight`, `current_depth_loss_weight`, and `future_depth_loss_weight`.
- Produces checkpoint files `current_plan_head.pt`, `plan_head.pt`, `current_depth_head.pt`, and `depth_head.pt`.

- [ ] **Step 1: Write failing exact-loss tests**

Use constant fake predictions/targets and assert exact arithmetic:

```python
expected = 0.004 * (
    future_dino_mse + current_dino_mse
    + future_depth_smooth_l1 + current_depth_smooth_l1
)
assert torch.allclose(output["loss"], expected)
assert output["future_dino_weighted"] == 0.004 * future_dino_mse
assert output["current_dino_weighted"] == 0.004 * current_dino_mse
assert output["future_depth_weighted"] == 0.004 * future_depth_smooth_l1
assert output["current_depth_weighted"] == 0.004 * current_depth_smooth_l1
```

Add parser tests requiring all four defaults to equal `0.004` and rejecting non-finite or negative weights.

- [ ] **Step 2: Write failing four-head warm-start and checkpoint tests**

Require `_load_lingbot_head_state` to select all four marker families from a fake safetensors index. Require warm-start reports for `current_video_align_head`, `future_video_align_head`, `depth_align_head`, and `future_depth_align_head`. Require all four head files and 16 plan embeddings in exported checkpoints.

- [ ] **Step 3: Run tests and verify RED**

```bash
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_lingbot_dino_depth_contract.py \
  -k 'loss_weight or four_head_warmstart or checkpoint_current_heads or metadata_k1'
```

Expected: failures for missing arguments, current warm-start keys, and current checkpoint files.

- [ ] **Step 4: Implement exact four-term loss**

Add explicit CLI flags with default `0.004`. In `PlannerWrapper.forward`, require all four targets and compute DINO with `F.mse_loss` and depth with `F.smooth_l1_loss`; multiply each raw term by its own weight. Return raw, weighted, norm-ratio, and total metrics with stable names.

- [ ] **Step 5: Extend warm start and optimizer groups**

Include these markers in `_load_lingbot_head_state`:

```python
markers = (
    "current_video_align_head.", "future_video_align_head.",
    "depth_align_head.", "future_depth_align_head.",
)
query_suffixes = (
    "current_video_align_embs", "future_video_align_embs",
    "depth_align_embs", "future_depth_align_embs",
)
```

Warm-start each corresponding head. Include all four heads in the head-LR optimizer group and exclude all four from the backbone group.

- [ ] **Step 6: Implement strict K1 export**

Require `sequence_length=9`, `num_keyframes=1`, offset `[8]`, target length `256`, task-token count `8`, temporal groups `2`, latent length `16`, and four heads. Save all four heads plus the 16 token embeddings. Serialize the four exact loss weights and normalized time `[1.0]`.

- [ ] **Step 7: Run focused and full contract tests**

```bash
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_lingbot_dino_depth_contract.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit loss/export changes**

```bash
git add tests/test_lingbot_dino_depth_contract.py \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py
git commit -m "feat: match lingbot k1 alignment loss"
```

---

### Task 4: Load the K1 Checkpoint and Export Only Future Plans

**Files:**
- Modify: `tests/test_dino_depth_plan_provider.py`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_depth_plan_provider.py`

**Interfaces:**
- Produces: `PlannerContract(num_keyframes=1, target_tokens=256, keyframe_offsets=(8,), normalized_keyframe_times=(1.0,), num_task_tokens=8, latent_len=16)`.
- Preserves: `FrozenDinoDepthPlanProvider.__call__` returning detached future DINO/depth plans and `[1.0]` times.

- [ ] **Step 1: Rewrite provider fixtures for the strict K1 metadata**

Fixtures must contain `current_alignment=True`, `num_keyframes=1`, `target_tokens=256`, `keyframe_offsets=[8]`, `normalized_keyframe_times=[1.0]`, `num_task_tokens=8`, `num_temporal_query_groups=2`, `latent_len=16`, the new query layout, four exact `0.004` weights, and 16 unique plan tokens.

- [ ] **Step 2: Add failing completeness/output tests**

Require current head files during checkpoint preflight even though the provider returns only future outputs. Fake wrapper predictions must be `[B,256,1024]`; reject 255 or 257 tokens. Assert the provider invokes one wrapper call and returns times shaped `[B,1]` containing `1.0`.

- [ ] **Step 3: Run tests and verify RED**

```bash
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_dino_depth_plan_provider.py
```

Expected: failures because provider metadata and shape checks still require K4/1024 tokens.

- [ ] **Step 4: Implement the strict K1 contract**

Replace the K4 metadata table with the exact K1 fields. Require `current_plan_head.pt` and `current_depth_head.pt`. Construct the wrapper from metadata, load all four heads, restore 16 plan embeddings, and keep `predict_dino_depth_plan` as the public future-only prediction method.

- [ ] **Step 5: Run provider tests and verify GREEN**

Run the Step 3 command. Expected: all tests pass.

- [ ] **Step 6: Commit provider changes**

```bash
git add tests/test_dino_depth_plan_provider.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_depth_plan_provider.py
git commit -m "feat: load lingbot k1 planner checkpoints"
```

---

### Task 5: Consume One Semantic Frame in FastWAM

**Files:**
- Modify: `tests/test_fastwam_online_semantic_planner.py`
- Modify: `tests/test_fastwam_dino_depth_fusion.py`
- Modify: `tests/test_fastwam_cosmos_semantic_plan.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/online_semantic_planner.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py`
- Modify: `third_party/FastWAM/configs/model/fastwam_cosmos.yaml`
- Modify: `third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml`

**Interfaces:**
- Consumes future DINO/depth `[B,256,1024]` and semantic times `[B,1]`.
- Produces fused semantic plan `[B,256,1024]` without changing feature width.

- [ ] **Step 1: Rewrite metadata/config tests for K1**

Require FastWAM's independent metadata preflight to match Task 4 exactly. Set runtime/config expectations to:

```python
semantic_plan_max_tokens = 256
semantic_plan_num_keyframes = 1
semantic_plan_source_num_keyframes = 1
expected_times = torch.ones(batch, 1)
```

Change fusion tests to use `[B,256,1024]` branches and assert same-position fusion preserves that shape.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_fastwam_online_semantic_planner.py \
  tests/test_fastwam_dino_depth_fusion.py \
  tests/test_fastwam_cosmos_semantic_plan.py
```

Expected: K4/1024-token validation failures.

- [ ] **Step 3: Update independent metadata preflight**

Set K1/16-task-token expected metadata in `online_semantic_planner.py`, including exact loss weights and `[1.0]`. Preserve the rule that metadata is validated before importing/loading Qwen.

- [ ] **Step 4: Remove the hardcoded four-keyframe runtime gate**

Replace `semantic_plan_num_keyframes != 4` with a strict equality check against the configured/validated K1 source. Require online plans to match `(batch,256,1024)` and times `(batch,1)`. Keep all non-finite and source-exclusivity checks.

- [ ] **Step 5: Pin Hydra configs to one keyframe**

In `fastwam_cosmos.yaml`, set both keyframe counts to `1` and max tokens to `256`. In `libero_2cam_cosmos.yaml`, set semantic max tokens to `256`. Do not change the video frame count, action-video ratio, or output horizon.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 7: Commit FastWAM K1 changes**

```bash
git add tests/test_fastwam_online_semantic_planner.py \
  tests/test_fastwam_dino_depth_fusion.py \
  tests/test_fastwam_cosmos_semantic_plan.py \
  third_party/FastWAM/src/fastwam/models/cosmos/online_semantic_planner.py \
  third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py \
  third_party/FastWAM/configs/model/fastwam_cosmos.yaml \
  third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml
git commit -m "feat: consume one lingbot semantic frame"
```

---

### Task 6: Add Strict Launchers and Retrain on HPC3

**Files:**
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1.sh`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1_hpc3.sbatch`

**Interfaces:**
- Produces a launcher that fixes K1/offset8/16-token/four-0.004-loss geometry.
- Produces an HPC3 Slurm job chain with b4 smoke, b4 formal, b2 fallback smoke, and b2 fallback formal.

- [ ] **Step 1: Write failing launcher tests**

Require the shell launcher to export:

```text
USE_DEPTH=1
SEQUENCE_LENGTH=9
NUM_KEYFRAMES=1
NUM_LATENT_PER_KEYFRAME=8
KEYFRAME_SCHEME=even_future
CURRENT_DINO_LOSS_WEIGHT=0.004
FUTURE_DINO_LOSS_WEIGHT=0.004
CURRENT_DEPTH_LOSS_WEIGHT=0.004
FUTURE_DEPTH_LOSS_WEIGHT=0.004
```

Require the sbatch launcher to default to 12,000 steps, enforce `NUM_GPUS * BATCH_SIZE * GRAD_ACCUM == 128`, use the full `lingbot-vla-v2-6b` directory for four-head warm start, and write to a K1-current output directory.

- [ ] **Step 2: Run launcher tests and verify RED**

```bash
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_lingbot_dino_depth_contract.py -k 'k1_launcher or hpc3_k1'
```

Expected: failures because the K1 launchers do not exist.

- [ ] **Step 3: Implement the strict shell launcher**

Export the values from Step 1 and exec `train_lingbot_dino_4b.sh`. Extend the base launcher to forward the four explicit loss weights while preserving generic non-K1 callers.

- [ ] **Step 4: Implement the HPC3 launcher**

Copy the verified resource/path checks from the prior GBS128 launcher. Use a distinct output root ending in `lingbot_current_future_k1_offset8_*_gbs128_12000step`; print the 16-token query contract and all four weights before launch.

- [ ] **Step 5: Run launcher/full CPU verification**

```bash
bash -n \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1.sh \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1_hpc3.sbatch
/data/LFT-W02_data/.conda/envs/starVLA/bin/python -m pytest -q \
  tests/test_lingbot_dino_depth_contract.py \
  tests/test_dino_depth_plan_provider.py \
  tests/test_fastwam_online_semantic_planner.py \
  tests/test_fastwam_dino_depth_fusion.py \
  tests/test_fastwam_cosmos_semantic_plan.py
```

Expected: all selected tests pass with no collection errors.

- [ ] **Step 6: Commit launchers**

```bash
git add tests/test_lingbot_dino_depth_contract.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1.sh \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_k1_hpc3.sbatch \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
git commit -m "feat: launch lingbot k1 planner training"
```

- [ ] **Step 7: Deploy to an isolated HPC3 code directory**

Sync only source/config files to:

```text
/data/user/jhe724/workspace/VLM4WAM_lingbot_k1_current_20260712
```

Reuse the already-synced LIBERO dataset, text cache, stats, LingBot source, and weights. Verify rsync dry-run differences are zero and instantiate the 277,713-sample dataset on the login node.

- [ ] **Step 8: Submit and monitor exact-config smoke**

Submit b4/acc4/MAX_STEPS2. If it exits nonzero, automatically run b2/acc8/MAX_STEPS2. Confirm the smoke log reports 16 task tokens, offset8, four heads loaded, four weights `0.004`, finite loss terms, and an exported strict checkpoint.

- [ ] **Step 9: Submit the 12k formal dependency**

Submit the matching 12,000-step formal job only after successful smoke. Verify the selected batch decomposition, node allocation, output path, first optimizer step, and absence of OOM/Traceback/NCCL errors. Preserve checkpoints every 1,000 steps.

---

## Final Verification

- [ ] Run `git diff --check` on every touched path.
- [ ] Run `python -m compileall -q scripts/qwen3_vl_semantic_planner third_party/FastWAM/src/fastwam`.
- [ ] Run the five-file pytest command from Task 6 Step 5.
- [ ] Inspect the exported `planner_meta.json` and assert K1, offset8, time1.0, latent16, four heads, and four exact weights.
- [ ] Run the checkpoint-backed FastWAM smoke and assert future DINO/depth/fused shapes `[1,256,1024]`.
- [ ] Confirm the formal HPC3 job is RUNNING or truthfully report its Slurm queue reason.
