# Baton Early Checkpoint Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the repaired DDP checkpoint path at step 20, then keep the ACD1-18 Qwen3.5 Stage-1 run training to step 30,000.

**Architecture:** Extend the existing deterministic checkpoint schedule with one early step while keeping the 5,000-step periodic cadence. Use the same `_save_training_checkpoint` path for the probe and normal checkpoints so the probe exercises the production model, AdamW, scheduler, cursor, RNG, hashes, and atomic publication behavior.

**Tech Stack:** Python 3.10, PyTorch 2.7, Accelerate DDP, pytest, Slurm allocation ACD1-18, 8×H100 80GB.

## Global Constraints

- The early probe is exactly optimizer step 20.
- Periodic checkpoints remain steps 5,000 through 30,000 at 5,000-step intervals.
- Training remains BF16 with global batch 128, learning rate `1e-5`, four future keyframes, two cameras, and no gradient checkpointing.
- A failed step-20 save or load stops the run for diagnosis; a successful probe leaves the long run active.

---

### Task 1: Add the early production checkpoint

**Files:**
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Modify: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Consumes: `checkpoint_steps(max_steps: int, save_every: int) -> tuple[int, ...]`
- Produces: `checkpoint_steps(..., initial_save_step: int | None = None) -> tuple[int, ...]`

- [ ] **Step 1: Write the failing schedule test**

```python
def test_checkpoint_cadence_probes_step_20_then_saves_every_5000() -> None:
    assert checkpoint_steps(
        max_steps=30_000,
        save_every=5_000,
        initial_save_step=20,
    ) == (20, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_qwen35_baton_training.py::test_checkpoint_cadence_probes_step_20_then_saves_every_5000
```

Expected: fail because `checkpoint_steps` does not accept `initial_save_step`.

- [ ] **Step 3: Implement and route the schedule**

Add `initial_save_step: int | None = 20` to `Stage1TrainingConfig`, validate that it is strictly between zero and `max_steps`, make `checkpoint_steps` return the sorted union of the early and periodic steps, and replace the modulo save condition with membership in the configured schedule.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_qwen35_baton_checkpoint.py tests/test_qwen35_baton_training.py tests/test_qwen35_baton_provider.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add qwen35_baton/cli/train_semantic_planner.py tests/test_qwen35_baton_training.py
git commit -m "feat(baton): add early checkpoint probe"
```

### Task 2: Deploy and prove the checkpoint on ACD1-18

**Files:**
- Sync source tree to: `/data/user/jhe724/VLM4WAM_qwen35_baton_strict`
- Create runtime config outside Git: `/data/user/jhe724/VLM4WAM_qwen35_baton_strict/runtime/acd1_18_baton_ddp_probe.json`
- Write log to: `/data/user/jhe724/logs/qwen35_baton_strict/acd1_18_ddp_30k.log`
- Write checkpoints to: `/data/user/jhe724/outputs/qwen35_baton_strict_acd1_18_ddp_30k`

**Interfaces:**
- Consumes: committed DDP checkpoint code and ACD1-18 local model/data paths
- Produces: a verified `step_000020` checkpoint and the active 30,000-step run

- [ ] **Step 1: Check resources and paths**

Run `nvidia-smi` on ACD1-18 and verify all eight GPUs are below 500 MiB before launch. Locate the Qwen3.5 base, SigLIP2, HDF5 manifest, dataset statistics, and the `qwen35-planx` Python environment on shared storage.

- [ ] **Step 2: Sync code and run preflight**

Use rsync without data/checkpoint outputs, then run:

```bash
PYTHONPATH=. /data/user/jhe724/.conda/envs/qwen35/bin/python \
  -m qwen35_baton.cli.preflight \
  --config runtime/acd1_18_baton_ddp_probe.json \
  --world-size 8 --per-device-batch 4 --gradient-accumulation-steps 4
```

Expected: DDP, global batch 128, dense Qwen3.5-2B, SigLIP2 256/16/1024, and matching artifact hashes.

- [ ] **Step 3: Start the 8-GPU run**

Launch in a detached `screen` with `torch.distributed.run --nproc_per_node=8`, per-device batch 4, accumulation 4, and `--no-gradient-checkpointing`.

- [ ] **Step 4: Verify the early checkpoint**

Wait for step 20, require a complete `step_000020/manifest.json`, run the checkpoint loader in a fresh process, and confirm no traceback, OOM, NCCL error, or incomplete staging directory.

- [ ] **Step 5: Continue and monitor**

Leave the process running to step 30,000. On each continuation, inspect the screen, metrics tail, GPU memory/power, checkpoint directories, and error signatures. Diagnose and repair any failure before restarting from the newest validated checkpoint.

- [ ] **Step 6: Verify completion**

Require the training process to exit successfully at step 30,000, the metrics tail to report step 30,000 with finite loss, and `step_030000` to pass the same manifest/load validation as the early probe.
