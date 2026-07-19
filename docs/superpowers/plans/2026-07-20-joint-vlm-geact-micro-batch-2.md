# Joint VLM + GE-Act Micro-Batch 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the 30,000-step joint VLM + GE-Act experiment on 30332 with a per-GPU batch size of 2 while preserving global batch size 128.

**Architecture:** Change only the formal batch geometry and its preflight contract: 2 samples per GPU across 8 GPUs with 8 gradient-accumulation microsteps. The eight-GPU smoke path inherits the production batch geometry so it exercises the same memory demand before the formal run is restarted.

**Tech Stack:** YAML, Python 3.10, pytest, Bash, PyTorch torchrun, Accelerate, DeepSpeed ZeRO-2, 8×H100 80GB.

## Global Constraints

- Keep the effective global batch size at exactly 128.
- Keep `train_steps: 30000` and checkpoint steps 20,000, 25,000, and 30,000.
- Keep the predecoded RGB dataset with `require_predecoded: true`.
- Preserve both LTX and Qwen gradient checkpointing, bf16, and DeepSpeed ZeRO-2.
- Reject the rollout on OOM, non-finite loss, worker failure, or a sustained throughput regression.

---

### Task 1: Update the formal batch contract

**Files:**
- Modify: `ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml:53-55`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py:183-188`
- Modify: `ge_act/scripts/train_joint_vlm_geact_ola.sh:52-57`
- Test: `tests/test_ge_act_siglip2_config.py:283-289`
- Test: `tests/test_ge_act_siglip2_config.py:583-605`
- Test: `tests/test_ge_act_siglip2_config.py:365-388`

**Interfaces:**
- Consumes: the existing YAML keys `batch_size` and `gradient_accumulation_steps` and the existing eight-GPU launcher mode `RUN_KIND=smoke8`.
- Produces: a formal recipe with `batch_size=2`, `gradient_accumulation_steps=8`, and an eight-GPU smoke path that inherits those values.

- [ ] **Step 1: Change the contract tests first**

Update `test_joint_vlm_geact_config_matches_approved_recipe` to assert:

```python
assert config["batch_size"] == 2
assert config["gradient_accumulation_steps"] == 8
assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
```

Update the preflight error expectation to:

```python
"joint training requires per-GPU batch 2 and accumulation 8"
```

In `test_joint_ola_launcher_has_formal_and_bounded_smoke_modes`, isolate the `smoke8` branch and assert that it contains `--max_train_steps 10` but contains neither batch override:

```python
smoke8_branch = launcher.split('elif [[ "$RUN_KIND" == "smoke8" ]]', 1)[1]
smoke8_branch = smoke8_branch.split('elif [[ "$RUN_KIND" != "formal" ]]', 1)[0]
assert "--max_train_steps 10" in smoke8_branch
assert "--batch_size_override" not in smoke8_branch
assert "--gradient_accumulation_steps_override" not in smoke8_branch
```

- [ ] **Step 2: Run the focused tests and verify they fail against the old recipe**

Run:

```bash
pytest -q tests/test_ge_act_siglip2_config.py -k 'joint_vlm_geact_config_matches_approved_recipe or joint_ola_launcher_has_formal_and_bounded_smoke_modes or joint_formal'
```

Expected: failures report the old `batch_size: 1`, old accumulation 16, old preflight message, and smoke8 overrides.

- [ ] **Step 3: Implement the batch geometry and preflight contract**

Change the YAML block to:

```yaml
# 2 samples/GPU * 8 GPUs * 8 accumulation = global batch 128.
batch_size: 2
gradient_accumulation_steps: 8
```

Change the joint preflight check to:

```python
if (
    config.get("batch_size") != 2
    or config.get("gradient_accumulation_steps") != 8
):
    errors.append("joint training requires per-GPU batch 2 and accumulation 8")
```

Change the `smoke8` launcher branch to:

```bash
elif [[ "$RUN_KIND" == "smoke8" ]]; then
  MAIN_ARGS+=(
    --max_train_steps 10
  )
```

- [ ] **Step 4: Run focused and affected tests**

Run:

```bash
pytest -q tests/test_ge_act_siglip2_config.py tests/test_joint_vlm_geact_training.py tests/test_ge_act_semantic_training_contract.py
```

Expected: all selected tests pass.

Run:

```bash
python -m py_compile ge_act/scripts/preflight_ltx_siglip2.py
bash -n ge_act/scripts/train_joint_vlm_geact_ola.sh
git diff --check
```

Expected: all commands exit 0 without output indicating errors.

- [ ] **Step 5: Commit the batch change**

```bash
git add ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml \
  ge_act/scripts/preflight_ltx_siglip2.py \
  ge_act/scripts/train_joint_vlm_geact_ola.sh \
  tests/test_ge_act_siglip2_config.py
git commit -m "perf(ge-act): raise joint per-device batch to two"
```

Expected: one commit containing only the batch-contract change and its tests.

### Task 2: Validate and deploy on 30332

**Files:**
- Deploy: the four tracked files changed in Task 1 to `/data/users/junjie/code/VLM4WAM_joint_geact_30332`
- Observe: `/data/users/junjie/logs/joint_vlm_geact_k4_b2_smoke8_30332.log`
- Observe: `/data/users/junjie/logs/joint_vlm_geact_k4_b2_30k_30332.log`

**Interfaces:**
- Consumes: the committed formal recipe and launcher from Task 1, the existing predecoded cache, model weights, Python environment, and eight H100 GPUs on port 30332.
- Produces: a detached formal 30,000-step process with a persistent PID and log after a successful production-geometry smoke test.

- [ ] **Step 1: Stop the old formal process and verify all GPUs are released**

Read `/data/users/junjie/logs/joint_vlm_geact_k4_30k_30332.pid`, send `TERM` to its process group, and wait up to 30 seconds. Use `KILL` only if the group remains alive.

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Expected: every GPU has less than 500 MiB allocated before redeployment.

- [ ] **Step 2: Synchronize the committed runtime files**

Use `rsync` to update the YAML, preflight script, and launcher under `/data/users/junjie/code/VLM4WAM_joint_geact_30332`, preserving the existing compatibility symlink `/data/users/junjie/code/VLM4WAM_dual_camera_k4`.

Expected: local and remote SHA256 hashes match for all three runtime files.

- [ ] **Step 3: Run formal preflight**

Run on 30332:

```bash
cd /data/users/junjie/code/VLM4WAM_dual_camera_k4/ge_act
/data/users/junjie/envs/vlm4wam/bin/python scripts/preflight_ltx_siglip2.py \
  --config configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml \
  --world-size 8 \
  --require-joint-formal
```

Expected: `GE-Act SigLIP2 preflight passed`.

- [ ] **Step 4: Run the detached production-geometry smoke test**

Launch `RUN_KIND=smoke8` with `nohup setsid`, recording its process ID and tee log at `/data/users/junjie/logs/joint_vlm_geact_k4_b2_smoke8_30332.log`.

Expected after 10 optimizer steps: exit code 0, finite loss, no OOM or traceback, and peak allocated memory below 80GB per GPU. Compare `samples_per_second` with the previous micro-batch 1 value of approximately 8.41.

- [ ] **Step 5: Restart the detached 30,000-step formal run**

Launch `RUN_KIND=formal` with `nohup setsid`. Record the wrapper PID at `/data/users/junjie/logs/joint_vlm_geact_k4_b2_30k_30332.pid` and tee output at `/data/users/junjie/logs/joint_vlm_geact_k4_b2_30k_30332.log`.

Expected: the log reports `train batch size: 128`, `batches per device: 2`, `gradient accumulation steps: 8`, and advances beyond optimizer step 1 without OOM or non-finite loss.

- [ ] **Step 6: Report the running experiment**

Report the current optimizer step, loss, samples per second, peak allocated memory, wrapper PID, output directory, and log path. If smoke throughput regresses, restore the original batch geometry and report the measurements instead of launching the formal run.
