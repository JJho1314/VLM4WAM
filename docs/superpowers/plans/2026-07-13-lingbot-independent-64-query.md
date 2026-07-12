# LingBot Independent 64-Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the LIBERO Qwen3-VL 4B planner with four private 64-token VLM task groups, totaling 256 task tokens.

**Architecture:** Reuse the existing `num_task_tokens` parameter as the per-group width and remove the remaining 8/32 assumptions from metadata validation and launchers. Keep old checkpoints compatible by defaulting a missing `num_task_tokens` field to 8. Submit an eight-GPU HPC3 two-step smoke followed by an `afterok` 12,000-step formal run.

**Tech Stack:** Python, PyTorch, pytest, Bash, Slurm, rsync, HPC3.

## Global Constraints

- Group order is current DINO, future DINO, current depth, future depth.
- Each new-run group has exactly 64 task tokens; total `latent_len` is 256.
- Each prediction head continues to emit 256 teacher-feature tokens.
- Existing shared 16-token and independent 32-token checkpoints remain loadable.
- Training uses 8 GPUs, batch 16/GPU, accumulation 1, global batch 128, gradient checkpointing off, and 12,000 steps.
- Preserve unrelated dirty-worktree changes; do not create an implementation commit containing pre-existing edits.

---

### Task 1: Parameterize the four-group token geometry

**Files:**
- Modify: `tests/test_lingbot_k1_current_future.py`
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`

**Interfaces:**
- Consumes: `PlannerWrapper` keyword arguments `independent_modality_task_tokens: bool` and `num_task_tokens: int`.
- Produces: four slices of width `num_task_tokens`, `latent_len == 4 * num_task_tokens`, and dynamic checkpoint metadata.

- [ ] **Step 1: Write failing 64-token geometry tests**

```python
def test_independent_query_split_supports_four_64_token_groups():
    wrapper.num_task_tokens = 64
    hidden = torch.arange(2 * 256 * 3).reshape(2, 256, 3)
    groups = wrapper.split_independent_current_future_task_hidden(hidden)
    assert torch.equal(groups["current_dino"], hidden[:, 0:64])
    assert torch.equal(groups["future_dino"], hidden[:, 64:128])
    assert torch.equal(groups["current_depth"], hidden[:, 128:192])
    assert torch.equal(groups["future_depth"], hidden[:, 192:256])

def test_independent_wrapper_supports_64_tokens_per_group():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"image_token_id": 1})()

    wrapper = PlannerWrapper(
        model=TinyModel(), hidden_size=32, semantic_dim=16,
        plan_token_ids=list(range(256)), target_len=4,
        num_keyframes=1, grid_size=2, plan_head_type="lingbot_dino",
        use_depth=True, depth_dim=16, depth_grid_size=2,
        use_current_alignment=True, independent_modality_task_tokens=True,
        num_task_tokens=64,
    )
    assert wrapper.latent_len == 256
    assert wrapper.total_unique_latent_per_keyframe == 256
```

- [ ] **Step 2: Run tests and verify the metadata assertion fails**

Run: `pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py`

Expected: FAIL because the exported query-layout string is fixed to `_8_` and the CLI/launcher does not yet pin 64.

- [ ] **Step 3: Make metadata describe the configured group width**

Use `module.num_task_tokens` when producing `query_layout`, `num_task_tokens`, `latent_len`, and token strings. Keep `metadata.get("num_task_tokens", 8)` in `from_exported_checkpoint` for compatibility.

- [ ] **Step 4: Run the two test files again**

Run: `pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py`

Expected: PASS.

### Task 2: Make both checkpoint consumers accept 4×64

**Files:**
- Modify: `tests/test_dino_depth_plan_provider.py`
- Modify: `tests/test_fastwam_online_semantic_planner.py`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_depth_plan_provider.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/online_semantic_planner.py`

**Interfaces:**
- Consumes: metadata fields `independent_modality_task_tokens`, `num_task_tokens`, `latent_len`, `query_layout`, `plan_token_ids`, and `plan_token_strings`.
- Produces: strict dynamic validation of `4 * num_task_tokens`, with missing `num_task_tokens` interpreted as 8.

- [ ] **Step 1: Write failing provider and FastWAM loader tests**

```python
metadata.update({
    "independent_modality_task_tokens": True,
    "num_task_tokens": 64,
    "total_unique_latent_per_keyframe": 256,
    "latent_len": 256,
    "query_layout": "current_dino_64_then_future_dino_64_then_current_depth_64_then_future_depth_64",
    "plan_token_ids": list(range(256)),
    "plan_token_strings": [f"<|sem_plan_{i}|>" for i in range(256)],
})
assert validate_planner_metadata(metadata).total_unique_latent_per_keyframe == 256
```

- [ ] **Step 2: Run tests and verify strict validators reject 256**

Run: `pytest -q tests/test_dino_depth_plan_provider.py tests/test_fastwam_online_semantic_planner.py`

Expected: FAIL with expected length/layout 32.

- [ ] **Step 3: Compute independent-mode expectations dynamically**

Validate `num_task_tokens` as a positive integer, default it to 8 when absent, and set independent expectations to `4 * num_task_tokens` plus the matching dynamic layout string. Add `num_task_tokens: int` to `PlannerContract`.

- [ ] **Step 4: Re-run consumer tests**

Run: `pytest -q tests/test_dino_depth_plan_provider.py tests/test_fastwam_online_semantic_planner.py`

Expected: PASS for new 256-token and legacy 32-token metadata.

### Task 3: Pin HPC3 to 64 tokens and deploy

**Files:**
- Modify: `tests/test_lingbot_k1_current_future.py`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch`

**Interfaces:**
- Consumes: environment variable `NUM_TASK_TOKENS`.
- Produces: HPC3 command with `NUM_TASK_TOKENS=64`, 8 GPUs, batch 16, accumulation 1, and GBS128.

- [ ] **Step 1: Add a failing launcher contract test**

```python
assert "export NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-8}" in current_future_launcher
assert "NUM_TASK_TOKENS=64" in hpc3_launcher
assert "latent_len=256" in hpc3_launcher
```

- [ ] **Step 2: Verify the launcher test fails**

Run: `pytest -q tests/test_lingbot_k1_current_future.py::test_independent_query_hpc3_launcher_pins_64_tokens_per_feature`

Expected: FAIL because the current/future wrapper hardcodes `NUM_TASK_TOKENS=8`.

- [ ] **Step 3: Allow the wrapper override and pin the HPC3 run**

Change the wrapper to `export NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-8}`. Set `NUM_TASK_TOKENS=64` in the sbatch environment and update its contract/output naming to `4x64`, `latent_len=256`.

- [ ] **Step 4: Run full local verification**

Run: `pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py tests/test_dino_depth_plan_provider.py tests/test_fastwam_online_semantic_planner.py tests/test_fastwam_semantic_timing_routing.py tests/test_fastwam_cosmos_semantic_plan.py`

Expected: all tests pass. Also run `python -m py_compile` on the three Python implementation files, `bash -n` on both launchers, and `git diff --check`.

- [ ] **Step 5: Sync and submit a dependent HPC3 job chain**

Sync only the modified implementation/launcher files with `rsync -avR`. On HPC3, run:

```bash
REMOTE_ROOT=/data/user/jhe724/workspace/VLM4WAM_lingbot_k1_current_20260712
SBATCH=$REMOTE_ROOT/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch
SMOKE_JOB_ID=$(sbatch --parsable --export=ALL,RUN_KIND=smoke,MAX_STEPS=2,SAVE_STEPS=2 "$SBATCH")
FORMAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$SMOKE_JOB_ID --export=ALL,RUN_KIND=formal,MAX_STEPS=12000,SAVE_STEPS=1000 "$SBATCH")
```

- [ ] **Step 6: Verify Slurm state**

Run in the same HPC3 shell:

```bash
squeue -j "$SMOKE_JOB_ID,$FORMAL_JOB_ID" -o "%.18i %.28j %.2t %.10M %R"
```

Expected: smoke is `R` or `PD`; formal is `PD` with dependency reason until smoke succeeds.
