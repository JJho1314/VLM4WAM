# OLA B16 Planner Throughput Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark per-GPU batch 16 at unchanged global batch 128 and select the faster safe launch configuration.

**Architecture:** Use environment overrides already supported by the OLA launcher, so no training code changes are required. Run one isolated 60-step eight-GPU candidate, compare its measured speed and memory with the recorded B8 baseline, then launch the selected 30k configuration fresh.

**Tech Stack:** Bash, Accelerate, DeepSpeed ZeRO-2, PyTorch, NVIDIA H100, predecoded NumPy RGB cache.

## Global Constraints

- Use exactly 8 GPUs and global batch 128.
- Candidate is `BATCH_SIZE=16`, `GRAD_ACCUM=1`.
- Keep gradient checkpointing disabled and verify the actual model state in the log.
- Require all 3,424 predecoded caches; never fall back to MP4 decoding.
- Candidate passes only below 75 GiB/GPU, with finite loss and at least 10% lower seconds per step than the 1.09-second baseline.

---

### Task 1: Run the isolated B16 candidate

**Files:**
- Runtime output: `/data/users/junjie/code/VLM4WAM_dual_camera_k4/outputs/benchmark_qwen3vl2b_dual_camera_k4_b16_ga1`
- Runtime log: `/data/users/junjie/logs/vlm_dual_camera_k4_b16_ga1_benchmark.log`

**Interfaces:**
- Consumes: the existing `train_ge_act_dual_camera_k4_siglip2da3_ola.sh` environment-variable interface.
- Produces: 60 optimizer steps with progress, losses, and GPU telemetry.

- [ ] **Step 1: Confirm all eight GPUs are free**

Run: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader`

Expected: every GPU has less than 500 MiB allocated.

- [ ] **Step 2: Launch the 60-step candidate**

```bash
RUN_KIND=formal \
BATCH_SIZE=16 GRAD_ACCUM=1 EXPECTED_GLOBAL_BATCH=128 \
MAX_STEPS=60 SAVE_START_STEP=100000 \
OUTPUT_DIR=/data/users/junjie/code/VLM4WAM_dual_camera_k4/outputs/benchmark_qwen3vl2b_dual_camera_k4_b16_ga1 \
bash qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_k4_siglip2da3_ola.sh
```

Expected: runtime contract reports batch 16, accumulation 1, global batch 128, and gradient checkpointing false.

- [ ] **Step 3: Record peak memory and completion**

Run `nvidia-smi` during active forward/backward and inspect the benchmark log after exit.

Expected: all 60 steps complete, all logged losses are finite, and every GPU remains below 75 GiB.

### Task 2: Compare and launch the selected formal run

**Files:**
- Selected output: `/data/users/junjie/code/VLM4WAM_dual_camera_k4/outputs/qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa_predecoded_b16`
- Selected log: `/data/users/junjie/logs/vlm_dual_camera_k4_predecoded_b16_30k.log`

**Interfaces:**
- Consumes: baseline 1.09 seconds/step and the Task 1 benchmark results.
- Produces: one fresh 30k training process using the winning configuration.

- [ ] **Step 1: Compute candidate steady-state speed**

Use progress elapsed time after initialization and report seconds per optimizer step, peak memory, and delta versus baseline.

- [ ] **Step 2: Apply the decision rule**

Select B16 only if it is finite, below 75 GiB/GPU, and at least 10% faster; otherwise select B8.

- [ ] **Step 3: Launch and verify the selected 30k run**

Run the existing launcher with the selected `BATCH_SIZE` and `GRAD_ACCUM`, then verify the runtime contract, actual checkpointing state, first finite loss, process identity, output directory, and log path.

