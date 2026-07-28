# Qwen3.5 Strict Baton Stage-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current positive/negative Plan-X-derived planner training with Baton's positive-only continuous SigLIP2 visual blueprint pretraining and run the fastest stable ZeRO-2 configuration on eight OLA H100 GPUs.

**Architecture:** Each predecoded LIBERO sample becomes two independent Qwen3.5-VL rows, one per camera, containing the current observation, instruction, and 1,024 assistant plan placeholders. A shared Baton visual alignment tower uses 1,024 learned queries to cross-attend the gathered Qwen plan states; Sem-MLP maps them to four 16×16 SigLIP2 feature grids. The full Qwen planner trains against frozen online SigLIP2 penultimate features using only pointwise MSE, while ZeRO-2 and fused Qwen3.5 linear-attention kernels provide the systems optimization.

**Tech Stack:** Python 3.10, PyTorch, Transformers Qwen3.5-VL, Accelerate/DeepSpeed ZeRO-2, SigLIP2, HDF5, pytest, eight NVIDIA H100 80GB GPUs.

## Global Constraints

- Use Qwen3.5-2B-VL, the approved SigLIP2 patch16-256 artifact, four future keyframes, and 256 tokens per keyframe.
- Main and wrist cameras remain independent Qwen inputs and independent future-feature predictions.
- Use only Baton's learned-query cross-attention, Sem-MLP, and pointwise continuous-feature MSE in Stage 1.
- Do not add negative instructions, counterfactual ranking, cosine loss, delta loss, change weighting, query self-attention, query causal masks, or query-tower positional encoding.
- Read RGB only from `/data/users/junjie/data/LIBERO-fastwam-hdf5/manifest.json`; do not decode source videos and do not cache SigLIP2 target features.
- Train the full MLLM, visual alignment tower, and Sem-MLP at BF16 with AdamW `β1=0.9`, `β2=0.999`, learning rate `1e-5`; keep SigLIP2 frozen.
- Use ZeRO Stage 2, no gradient accumulation, 30,000 optimizer steps, and checkpoints every 5,000 steps.
- Require the fused Qwen3.5 linear-attention path; a Torch fallback is a preflight failure.
- Benchmark batch two without checkpointing and batch four with checkpointing only if required; select by samples/second, then average power.

---

### Task 1: Positive-Only Dual-Camera Batch Contract

**Files:**
- Modify: `qwen35_baton/data.py`
- Modify: `tests/test_qwen35_baton_data.py`

**Interfaces:**
- Produces: `BatonPlannerBatch` with `row_labels: tuple[tuple[int, str], ...]`, two rows per sample, `current_images: [B,2,3,H,W]`, and `future_images: [B,2,4,3,H,W]`.
- Produces: `BatonPlannerCollator.__call__(samples) -> BatonPlannerBatch` with sample-major `main`, `wrist` ordering.
- Removes: `negative_instructions`, `positive_rows`, `negative_rows`, `_negative_instruction`, and suite-level negative-caption bookkeeping.

- [ ] **Step 1: Replace negative-data tests with the desired two-row contract**

```python
def test_collator_builds_only_positive_sample_major_camera_rows(dataset) -> None:
    batch = BatonPlannerCollator(FakeProcessor())([dataset[0], dataset[1]])
    assert batch.row_labels == (
        (0, "main"), (0, "wrist"), (1, "main"), (1, "wrist")
    )
    assert batch.qwen_inputs["input_ids"].shape[0] == 4
    assert batch.plan_positions.shape == (4, 1024)
    assert not hasattr(batch, "negative_instructions")
```

- [ ] **Step 2: Run the new data test and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_data.py::test_collator_builds_only_positive_sample_major_camera_rows`

Expected: FAIL because the current collator emits positive and negative rows.

- [ ] **Step 3: Implement the minimal positive-only dataset and collator**

Remove wrong-instruction construction and build rows with:

```python
rows = [
    (sample_index, camera, current_images[sample_index, camera_index], instruction)
    for sample_index, instruction in enumerate(instructions)
    for camera_index, camera in enumerate(("main", "wrist"))
]
```

- [ ] **Step 4: Run the complete data tests**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_data.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qwen35_baton/data.py tests/test_qwen35_baton_data.py
git commit -m "refactor(baton): use positive-only camera rows"
```

### Task 2: Baton Visual Alignment Tower

**Files:**
- Modify: `qwen35_baton/query_tower.py`
- Modify: `qwen35_baton/model.py`
- Modify: `tests/test_qwen35_baton_query_tower.py`
- Modify: `tests/test_qwen35_baton_model.py`

**Interfaces:**
- Produces: `BatonVisualAlignmentTower(qwen_dim: int, num_heads: int = 16)`.
- Produces: `forward(plan_states: Tensor, *, return_attention_maps: bool = False) -> QueryTowerOutput`, whose `hidden_states` map `[R,4,256,D]` to `[R,4,256,D]`.
- Keeps: `BatonQwen35Planner.forward(batch) -> BatonPlannerOutput` with `positive: [B,2,4,256,1024]`.
- Removes: negative output, query self-attention, context projection, camera embeddings, block-causal mask, and 3D query RoPE.

- [ ] **Step 1: Write failing structural and numerical tests**

```python
def test_visual_alignment_tower_is_baton_query_cross_attention_only() -> None:
    tower = BatonVisualAlignmentTower._from_test_config(
        qwen_dim=16, num_frames=2, tokens_per_frame=4, num_heads=4
    )
    assert tower.learned_queries.shape == (2, 4, 16)
    assert isinstance(tower.cross_attention, nn.MultiheadAttention)
    assert not hasattr(tower, "blocks")
    assert not hasattr(tower, "allowed_mask")
    assert not hasattr(tower, "positions")
    output = tower(torch.randn(3, 2, 4, 16))
    assert output.hidden_states.shape == (3, 2, 4, 16)
```

Add a model test asserting one sample requires exactly two rows and
`BatonPlannerOutput` has no `negative` attribute.

- [ ] **Step 2: Run the new tower/model tests and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_query_tower.py tests/test_qwen35_baton_model.py`

Expected: FAIL on the missing Baton tower and current four-row/negative contract.

- [ ] **Step 3: Implement Baton's learned-query cross-attention**

Use shared queries initialized with `nn.init.normal_(learned_queries, std=0.02)`:

```python
context = plan_states.flatten(1, 2)
queries = self.learned_queries.flatten(0, 1)[None].expand(rows, -1, -1)
aligned, attention = self.cross_attention(
    query=queries, key=context, value=context,
    need_weights=return_attention_maps,
    average_attn_weights=True,
)
return aligned.reshape(rows, self.num_frames, self.tokens_per_frame, self.qwen_dim)
```

Change Sem-MLP to map Qwen width to SigLIP2 width:

```python
self.sem_mlp = nn.Sequential(
    nn.Linear(qwen_dim, qwen_dim),
    nn.GELU(),
    nn.Linear(qwen_dim, 1024),
)
```

- [ ] **Step 4: Run tower and model tests**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_query_tower.py tests/test_qwen35_baton_model.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qwen35_baton/query_tower.py qwen35_baton/model.py \
  tests/test_qwen35_baton_query_tower.py tests/test_qwen35_baton_model.py
git commit -m "refactor(baton): align visual queries with paper"
```

### Task 3: Pure Baton Continuous-Feature MSE

**Files:**
- Modify: `qwen35_baton/losses.py`
- Modify: `qwen35_baton/config.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Modify: `tests/test_qwen35_baton_losses.py`
- Modify: `tests/test_qwen35_baton_config.py`
- Modify: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Produces: `BatonPlannerLoss(mse: Tensor, total: Tensor)`.
- Produces: `compute_baton_planner_loss(prediction, future_teacher) -> BatonPlannerLoss`.
- Training calls only `teacher.encode_future`; current-frame SigLIP2 targets are not computed.

- [ ] **Step 1: Write the failing Equation-8 loss test**

```python
def test_baton_loss_is_only_pointwise_continuous_feature_mse() -> None:
    prediction = torch.tensor([0.0, 2.0]).reshape(1, 1, 1, 1, 2)
    target = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2)
    loss = compute_baton_planner_loss(prediction, target)
    assert loss.mse.item() == pytest.approx(2.5)
    assert loss.total.item() == pytest.approx(2.5)
    assert set(vars(loss)) == {"mse", "total"}
```

Update the tiny training test so the fake teacher raises if
`encode_current` is called and metrics contain only total MSE plus timing,
throughput, and per-camera/frame MSE.

- [ ] **Step 2: Run loss and one-step training tests and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_losses.py tests/test_qwen35_baton_training.py::test_one_tiny_stage1_step_updates_only_owned_parameters`

Expected: FAIL because the current loss requires negative/current features and emits auxiliary terms.

- [ ] **Step 3: Implement pure MSE and remove auxiliary training metrics**

Use:

```python
mse = (prediction.float() - future_teacher.float()).square().mean()
return BatonPlannerLoss(mse=mse, total=mse)
```

Delete `BatonLossWeights`, counterfactual ranking metrics, cosine metrics,
changed-patch weighting, delta loss, and current-teacher extraction.

- [ ] **Step 4: Run loss, config, and training tests**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_losses.py tests/test_qwen35_baton_config.py tests/test_qwen35_baton_training.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qwen35_baton/losses.py qwen35_baton/config.py \
  qwen35_baton/cli/train_semantic_planner.py \
  tests/test_qwen35_baton_losses.py tests/test_qwen35_baton_config.py \
  tests/test_qwen35_baton_training.py
git commit -m "refactor(baton): use equation 8 feature mse"
```

### Task 4: Full VA-Planner Ownership and Paper Optimizer

**Files:**
- Modify: `qwen35_baton/ownership.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Modify: `qwen35_baton/configs/libero_stage1.json`
- Modify: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Produces: `configure_stage1_trainable_modules(planner) -> Stage1Ownership` whose owned parameter IDs equal every planner parameter ID.
- Produces: one AdamW group at learning rate `1e-5`, betas `(0.9, 0.999)`, and weight decay from config.
- Removes: planner/top-eight/vision learning-rate split and partial language freezing.

- [ ] **Step 1: Write failing full-ownership and optimizer tests**

```python
def test_stage1_trains_the_entire_va_planner() -> None:
    planner = _TinyPlanner()
    ownership = configure_stage1_trainable_modules(planner)
    assert all(parameter.requires_grad for parameter in planner.parameters())
    assert ownership.trainable_modules == (planner,)

def test_stage1_uses_paper_optimizer_contract(tmp_path) -> None:
    config = _config(tmp_path)
    optimizer = build_stage1_optimizer(_TinyPlanner(), config)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert {group["lr"] for group in optimizer.param_groups} == {1e-5}
```

- [ ] **Step 2: Run ownership/optimizer tests and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_training.py -k 'entire_va_planner or paper_optimizer'`

Expected: FAIL because current ownership freezes lower Qwen layers and uses three learning rates.

- [ ] **Step 3: Implement full ownership and one optimizer group**

Set `planner.requires_grad_(True)`, expose `(planner,)` as the exhaustive
owner, and construct `torch.optim.AdamW` with `lr=config.learning_rate`,
`betas=(0.9, 0.999)`.

- [ ] **Step 4: Run training and checkpoint tests**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_training.py tests/test_qwen35_baton_checkpoint.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qwen35_baton/ownership.py qwen35_baton/cli/train_semantic_planner.py \
  qwen35_baton/configs/libero_stage1.json tests/test_qwen35_baton_training.py
git commit -m "refactor(baton): train the full va planner"
```

### Task 5: ZeRO-2 and Fast-Path Preflight

**Files:**
- Create: `qwen35_baton/configs/deepspeed_zero2.json`
- Modify: `qwen35_baton/cli/preflight.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Modify: `qwen35_baton/scripts/train_semantic_planner.sh`
- Modify: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Produces: `require_qwen35_fast_path() -> dict[str, str]`, importing `fla` and `causal_conv1d` and rejecting absence.
- Produces: `Accelerator(..., deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=...))`.
- Produces: launcher contract `PER_DEVICE_BATCH`, `GRAD_ACCUM=1`, `DEEPSPEED_CONFIG`.
- Produces: `gradient_checkpointing: bool` runtime config that calls the released Qwen backbone's public enable/disable API before distributed wrapping.
- Removes: the old validation that forced an effective global batch of exactly 128.

- [ ] **Step 1: Write failing ZeRO-2 and dependency preflight tests**

Assert the JSON has:

```python
assert config["zero_optimization"]["stage"] == 2
assert config["bf16"]["enabled"] is True
assert config["gradient_accumulation_steps"] == 1
```

Monkeypatch imports so `require_qwen35_fast_path()` fails with an actionable
message when either `fla` or `causal_conv1d` is missing. Assert real
preflight requires it while `tiny_test=True` bypasses compiled dependencies.
Assert accumulation values other than one are rejected and batch size is
reported rather than forced to produce global batch 128.

- [ ] **Step 2: Run the new preflight tests and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_training.py -k 'zero2 or fast_path'`

Expected: FAIL because the config and preflight do not exist.

- [ ] **Step 3: Implement explicit ZeRO-2 and fast-path contracts**

Create a ZeRO-2 config with BF16, overlap communication, contiguous gradients,
and no CPU/NVMe offload. Load it through `DeepSpeedPlugin`. Require the imports
before model loading so Transformers cannot silently print and use the slow
fallback. Add the explicit gradient-checkpointing switch and remove the old
global-batch-128 invariant.

- [ ] **Step 4: Validate launcher and tests**

Run: `bash -n qwen35_baton/scripts/train_semantic_planner.sh`

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_training.py`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add qwen35_baton/configs/deepspeed_zero2.json \
  qwen35_baton/cli/preflight.py qwen35_baton/cli/train_semantic_planner.py \
  qwen35_baton/scripts/train_semantic_planner.sh \
  tests/test_qwen35_baton_training.py
git commit -m "feat(baton): add zero2 fused training runtime"
```

### Task 6: Predecoded HDF5 Integration and Full Regression

**Files:**
- Modify: `tests/test_qwen35_baton_end_to_end.py`
- Modify: `tests/test_libero_fastwam_hdf5.py`
- Modify: `qwen35_baton/cli/smoke_pipeline.py`
- Modify: `qwen35_baton/provider.py`

**Interfaces:**
- Consumes: positive-only batch, Baton tower, pure MSE, and full ownership.
- Proves: no source-video decoder is called and checkpoint/provider inference still produces `[B,2,4,256,1024]`.

- [ ] **Step 1: Write a failing no-video-decode integration test**

Construct a temporary HDF5 manifest, monkeypatch `av.open` to raise, load one
`LiberoFastWAMHDF5Dataset` sample through `BatonLiberoDataset`, and assert
future/current RGB shapes. This proves the Stage-1 loader never reaches the
source-video decoder.

- [ ] **Step 2: Run the end-to-end test and verify RED where old contracts remain**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_end_to_end.py tests/test_libero_fastwam_hdf5.py -k 'baton or predecoded'`

Expected: FAIL on stale negative/loss/tower assumptions, never on HDF5 decoding.

- [ ] **Step 3: Update smoke/provider consumers without changing inference geometry**

Change `smoke_pipeline.py` fake batches and fake planner outputs to the
positive-only `BatonPlannerBatch`/`BatonPlannerOutput` signatures. Change
`provider.py` where it constructs planner outputs and retain only direct Baton
cross-attention tracing as the inference analysis path.

- [ ] **Step 4: Run the complete Baton suite**

Run: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_*.py`

Expected: PASS with no failures.

- [ ] **Step 5: Commit**

```bash
git add qwen35_baton/cli/smoke_pipeline.py qwen35_baton/provider.py \
  tests/test_qwen35_baton_end_to_end.py tests/test_libero_fastwam_hdf5.py
git commit -m "test(baton): verify strict predecoded pipeline"
```

### Task 7: OLA Environment, Throughput Trials, and Production Launch

**Files:**
- Create remotely: `/data/users/junjie/code/VLM4WAM_qwen35_baton_strict/runtime/ola_stage1_b2.json`
- Create remotely: `/data/users/junjie/code/VLM4WAM_qwen35_baton_strict/runtime/ola_stage1_b4.json`
- Create remotely: `/data/users/junjie/logs/qwen35_baton/strict_b2.log`
- Create remotely: `/data/users/junjie/logs/qwen35_baton/strict_b4.log`

**Interfaces:**
- Produces: measured `samples/s`, peak MiB, average utilization, and average watts for batch two and batch four.
- Produces: one detached `screen` session `baton_stage1` running the selected 30,000-step configuration.

- [ ] **Step 1: Stop the obsolete positive/negative OLA run after local verification**

Run: `ssh Ola_H100 'screen -S baton_stage1 -X quit || true'`

Expected: the old `stage1_855aa28_b1a16` process tree exits and all eight GPUs become free.

- [ ] **Step 2: Sync the exact committed source state and record its revision**

Use `rsync` from the clean worktree and write the exact `git rev-parse HEAD` to
remote `CODE_REVISION`. Exclude outputs, caches, logs, temporary tests, and
model/data artifacts.

- [ ] **Step 3: Install and verify fused dependencies in the existing environment**

Run:

```bash
/data/users/junjie/envs/qwen35-planx/bin/pip install 'flash-linear-attention[cuda]' causal-conv1d
/data/users/junjie/envs/qwen35-planx/bin/python -c \
  'import fla, causal_conv1d; print(fla.__version__)'
```

Expected: imports succeed. This follows the official FLA CUDA installation
route and the causal-conv1d import contract.

- [ ] **Step 4: Run remote static preflight**

Run the Stage-1 preflight with the local augmented Qwen artifact, exact
SigLIP2 hashes, HDF5 manifest hash, ZeRO stage two, eight GPUs, batch two, and
accumulation one.

Expected: preflight prints `tiny_test=false`, `zero_stage=2`,
`gradient_accumulation_steps=1`, and fast-path module versions.

- [ ] **Step 5: Benchmark batch two without checkpointing**

Launch eight ranks into a disposable output root and run at least 20 optimizer
steps. Sample `nvidia-smi` once per second and record mean power/utilization.
Reject OOM, nonfinite loss, skipped updates, or any fast-path fallback warning.

- [ ] **Step 6: Benchmark batch four**

Try batch four without checkpointing first. If it OOMs, enable selective Qwen
activation checkpointing and rerun the same 20-step window. Do not change the
Baton objective, token geometry, precision, or ZeRO stage.

- [ ] **Step 7: Select by throughput and launch production**

Choose the stable trial with highest samples/second; use power only to diagnose
underutilization. Create a fresh output directory, retain 30,000 steps and
5,000-step saves, and launch it as detached screen `baton_stage1`.

- [ ] **Step 8: Verify the production run**

Confirm all eight ranks remain alive, fused fast path is active, no traceback
or OOM exists, and the first optimizer-step metric is finite. Report screen,
log, output directory, batch, ZeRO stage, throughput, peak memory, average
utilization, and average power.
