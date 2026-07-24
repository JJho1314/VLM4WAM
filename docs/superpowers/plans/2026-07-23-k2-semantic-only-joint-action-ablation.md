# K2 Semantic-Only Joint Action Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a full 25,000-step HPC3 joint VLM/LTX/action run that supervises and injects only future offsets `[4, 8]`, never constructs or executes DA3, and preserves enough metadata for a guided/off/baseline LIBERO comparison.

**Architecture:** Keep the warm-started planner's internal K4 query sequence and add a semantic-only prediction API that executes Qwen and the SigLIP2 head but not the depth head. Select K4 output indices `(1, 3)` into a differentiable K2 tensor before planner loss and LTX cross-attention. Add a separate fail-closed HPC3 recipe and launcher, leaving the existing K4+DA3 recipe unchanged.

**Tech Stack:** Python, PyTorch, Qwen3-VL, SigLIP2, LTX/GE-Act, Accelerate, DeepSpeed ZeRO-2, pytest, YAML, Slurm.

## Global Constraints

- Planner warm start remains the existing dual-camera K4 checkpoint.
- Runtime semantic offsets are exactly `[4, 8]`, corresponding to K4 indices `[1, 3]`.
- Runtime semantic shape is `[B, 2, 2 * 256, 1024]`.
- DA3 is not instantiated, its head is not executed, and no depth loss is added.
- Qwen still consumes the checkpoint's K4 query sequence; native K2 query conversion is out of scope.
- The full objective is `video_loss + action_loss + 0.1 * semantic_planner_loss`.
- Qwen vision and language layers 16–27 are trainable at `1e-4`.
- Qwen embeddings, language layers 0–15, LM head, and depth head are frozen.
- Global batch is `4 * 8 * 8 = 256`.
- The formal run is 25,000 optimizer steps and saves every 5,000 steps.
- BF16, DeepSpeed ZeRO-2, seed 2026, gradient checkpointing disabled.
- Formal training is submitted only after a successful eight-GPU smoke run.

---

### Task 1: Add a Semantic-Only Planner Forward

**Files:**
- Modify: `qwen3_vl_semantic_planner/train_semantic_planner.py`
- Test: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Consumes: K4 `PlannerWrapper` with `plan_head_type == "lingbot_dino"`.
- Produces: `select_flat_keyframes(plan, *, num_keyframes, tokens_per_keyframe, indices)` and `PlannerWrapper.predict_semantic_plan_with_losses(...) -> tuple[Tensor, dict[str, Tensor]]`.

- [ ] **Step 1: Write failing selector and no-depth tests**

Add tests that require ordered differentiable selection:

```python
def test_select_flat_keyframes_keeps_order_and_gradient() -> None:
    plan = torch.arange(4.0).reshape(1, 1, 4, 1, 1).repeat(1, 2, 1, 2, 3)
    plan.requires_grad_()
    selected = planner.select_flat_keyframes(
        plan.reshape(1, 2, 8, 3),
        num_keyframes=4,
        tokens_per_keyframe=2,
        indices=(1, 3),
    )
    assert selected.shape == (1, 2, 4, 3)
    torch.testing.assert_close(
        selected.reshape(1, 2, 2, 2, 3)[:, :, :, 0, 0],
        torch.tensor([[[1.0, 3.0], [1.0, 3.0]]]),
    )
    selected.sum().backward()
    assert plan.grad is not None
```

Add a `FailIfCalledDepthHead` and assert the semantic-only method returns
`[1,2,512,8]`, produces a differentiable loss, and never calls the depth
head.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q \
  tests/test_ge_act_dual_camera_planner.py::test_select_flat_keyframes_keeps_order_and_gradient \
  tests/test_ge_act_dual_camera_planner.py::test_semantic_only_k2_forward_never_calls_depth_head
```

Expected: both fail because the selector and semantic-only method do not
exist.

- [ ] **Step 3: Implement the flat keyframe selector**

Add:

```python
def select_flat_keyframes(
    plan: torch.Tensor,
    *,
    num_keyframes: int,
    tokens_per_keyframe: int,
    indices: Sequence[int],
) -> torch.Tensor:
    if plan.ndim < 3:
        raise ValueError("plan must have batch, token, and feature dimensions")
    resolved = tuple(int(index) for index in indices)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("selected keyframe indices must be unique and non-empty")
    if min(resolved) < 0 or max(resolved) >= int(num_keyframes):
        raise ValueError("selected keyframe index exceeds planner geometry")
    expected_tokens = int(num_keyframes) * int(tokens_per_keyframe)
    if plan.shape[-2] != expected_tokens:
        raise ValueError(
            f"planner token count must be {expected_tokens}, got {plan.shape[-2]}"
        )
    grouped = plan.reshape(
        *plan.shape[:-2],
        int(num_keyframes),
        int(tokens_per_keyframe),
        plan.shape[-1],
    )
    selected = grouped.index_select(
        -3,
        torch.as_tensor(resolved, device=plan.device),
    )
    return selected.flatten(-3, -2)
```

- [ ] **Step 4: Implement semantic-only prediction and loss**

Refactor only the SigLIP2 branch of `predict_dino_depth_plan` into
`predict_semantic_plan`. It must run `_forward_hiddens`, split the DINO query
hidden, execute only `plan_head`, and retain the existing future-image detach.
Then add:

```python
def predict_semantic_plan_with_losses(
    self,
    semantic_plan_labels: torch.Tensor,
    *,
    selected_keyframe_indices: Sequence[int],
    **inputs: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    semantic = self.predict_semantic_plan(**inputs)
    semantic = select_flat_keyframes(
        semantic,
        num_keyframes=self.num_keyframes,
        tokens_per_keyframe=self.target_len // self.num_keyframes,
        indices=selected_keyframe_indices,
    )
    target = semantic_plan_labels.to(semantic.device, torch.float32)
    if semantic.shape != target.shape:
        raise ValueError(
            "semantic prediction/target shape mismatch: "
            f"{tuple(semantic.shape)} != {tuple(target.shape)}"
        )
    losses = self.compute_plan_losses(semantic, target)
    if self.lm_plan_loss_weight > 0:
        lm_plan_ce = self._last_lm_plan_ce
        losses["loss"] = losses["loss"] + self.lm_plan_loss_weight * lm_plan_ce
        losses["lm_plan_ce"] = lm_plan_ce.detach()
    return semantic, losses
```

Keep `predict_dino_depth_plan_with_losses` behavior unchanged for the K4+DA3
recipe.

- [ ] **Step 5: Run planner tests**

Run:

```bash
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q tests/test_ge_act_dual_camera_planner.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add qwen3_vl_semantic_planner/train_semantic_planner.py \
  tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): add K2 semantic-only prediction"
```

### Task 2: Route Differentiable K2 Semantics Through Joint GE-Act

**Files:**
- Modify: `ge_act/models/ltx_models/joint_vlm_geact.py`
- Modify: `ge_act/runner/ge_trainer.py`
- Modify: `qwen3_vl_semantic_planner/train_semantic_planner.py`
- Test: `tests/test_ge_act_dual_camera_planner.py`
- Test: `tests/test_joint_vlm_geact_training.py`

**Interfaces:**
- Consumes: `PlannerWrapper.predict_semantic_plan_with_losses`.
- Produces: `JointVLMGEActModel(..., semantic_only=True, selected_planner_keyframe_indices=(1,3))` and `encode_dual_camera_future_semantic_targets(...)`.

- [ ] **Step 1: Write failing K2 composite and target-encoder tests**

Add a tiny semantic-only planner whose depth method raises. Instantiate:

```python
joint = JointVLMGEActModel(
    planner,
    ltx,
    num_keyframes=2,
    tokens_per_keyframe=2,
    planner_num_keyframes=4,
    selected_planner_keyframe_indices=(1, 3),
    semantic_only=True,
)
```

Assert:

- semantic labels are `[B,2,4,1024]`;
- `depth_labels=None` is accepted;
- LTX receives `[B*2,2,2,1024]`;
- total loss reaches the planner and LTX parameters;
- calling the depth path raises the test sentinel.

In `tests/test_ge_act_dual_camera_planner.py`, add a target-encoder test with
a fake appearance teacher. Require output `[B,2,512,1024]` and prove the API
accepts no depth encoder.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q tests/test_joint_vlm_geact_training.py -k \
  "semantic_only_k2"
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q \
  tests/test_ge_act_dual_camera_planner.py::test_encode_dual_camera_future_semantic_targets_has_no_depth_dependency
```

Expected: fail on the missing constructor arguments and target encoder.

- [ ] **Step 3: Extend the composite model without changing K4 behavior**

Change `JointVLMGEActOutput.depth_plan` to `torch.Tensor | None`. Extend the
constructor:

```python
def __init__(
    self,
    planner: nn.Module,
    ltx: nn.Module,
    *,
    num_keyframes: int = 4,
    tokens_per_keyframe: int = 256,
    planner_num_keyframes: int | None = None,
    selected_planner_keyframe_indices: tuple[int, ...] | None = None,
    semantic_only: bool = False,
) -> None:
```

For semantic-only mode, validate that `len(selected indices) ==
num_keyframes`, call `predict_semantic_plan_with_losses`, skip all depth shape
checks, and pass the live K2 semantic tensor to LTX. Preserve the current
K4+depth branch byte-for-byte where possible.

- [ ] **Step 4: Add semantic-only target encoding**

Add beside `encode_dual_camera_future_targets` in
`train_semantic_planner.py`:

```python
def encode_dual_camera_future_semantic_targets(
    current_camera_images: torch.Tensor,
    future_camera_images: torch.Tensor,
    *,
    appearance_encoder: Any,
) -> dict[str, torch.Tensor]:
    batch_size, num_views, num_keyframes = future_camera_images.shape[:3]

    def normalize(frames: torch.Tensor) -> torch.Tensor:
        flattened = flatten_camera_teacher_frames(frames)
        return (flattened.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)

    current_bv = normalize(current_camera_images)
    future_bv = [
        normalize(future_camera_images[:, :, index])
        for index in range(num_keyframes)
    ]
    with torch.no_grad():
        semantic = appearance_encoder.encode_future_keyframes(
            current_bv,
            future_bv,
            effective_fps=None,
        )
    expected = (batch_size * num_views, num_keyframes * 256, 1024)
    if semantic.shape != expected:
        raise ValueError(
            f"semantic teacher features must be {expected}, got {tuple(semantic.shape)}"
        )
    return {
        "semantic_plan_labels": semantic.reshape(
            batch_size, num_views, num_keyframes * 256, 1024
        ).float().detach()
    }
```

In `ge_trainer.py`, import this function for semantic-only mode and assign it
to `self.joint_target_encoder`.

- [ ] **Step 5: Make depth-head freezing explicit**

Extend `configure_joint_planner_trainability` with
`freeze_depth_head: bool = False`. When true, require `planner.depth_head`,
set `requires_grad_(False)`, and put it in eval mode. Pass
`semantic_only` at all three trainability-policy call sites so recursive
`.train()` cannot reactivate it.

- [ ] **Step 6: Run joint tests**

Run:

```bash
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q tests/test_joint_vlm_geact_training.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add ge_act/models/ltx_models/joint_vlm_geact.py \
  ge_act/runner/ge_trainer.py \
  qwen3_vl_semantic_planner/train_semantic_planner.py \
  tests/test_ge_act_dual_camera_planner.py \
  tests/test_joint_vlm_geact_training.py
git commit -m "feat(joint): train GE-Act with K2 semantics only"
```

### Task 3: Add the Fail-Closed K2 HPC3 Recipe

**Files:**
- Create: `ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k2_semantic_only_hpc3.yaml`
- Create: `ge_act/scripts/sbatch_train_joint_vlm_geact_action_k2_hpc3.sh`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py`
- Modify: `ge_act/runner/ge_trainer.py`
- Test: `tests/test_ge_act_siglip2_config.py`
- Test: `tests/test_joint_vlm_geact_training.py`

**Interfaces:**
- Consumes: semantic-only composite and target encoder from Task 2.
- Produces: formal profile `hpc3_action_k2_semantic_only`, K2 checkpoint metadata, smoke/formal Slurm launcher.

- [ ] **Step 1: Write failing formal-config tests**

Copy the K4 HPC3 config test and require:

```python
assert config["tracker_name"] == "joint_vlm_geact_action_k2_semantic_only_25k"
assert config["semantic_plan"]["keyframe_indices"] == [4, 8]
assert config["diffusion_model"]["config"]["semantic_plan_num_keyframes"] == 2
joint = config["joint_training"]
assert joint["formal_recipe"] == "hpc3_action_k2_semantic_only"
assert joint["semantic_only"] is True
assert joint["planner_num_keyframes"] == 4
assert joint["selected_planner_keyframe_indices"] == [1, 3]
assert joint["selected_future_keyframe_offsets"] == [4, 8]
assert "da3_ckpt_dir" not in joint
assert "da3_code_root" not in joint
```

Also retain all LaWAM batch/LR/save assertions from the K4 action test.

- [ ] **Step 2: Write failing preflight and launcher tests**

Require the new profile to reject:

- any selected offsets other than `[4,8]`;
- any selected indices other than `[1,3]`;
- `semantic_plan_num_keyframes != 2`;
- `semantic_only != true`;
- any DA3 path;
- global batch other than 256.

Require the launcher to request eight GPUs, use the K2 config, run predecode
verification and formal preflight, and set `--max_train_steps 1` for
`RUN_KIND=smoke8`.

- [ ] **Step 3: Run config tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q tests/test_ge_act_siglip2_config.py -k \
  "k2_semantic_only"
```

Expected: fail because the config, profile, and launcher do not exist.

- [ ] **Step 4: Create the K2 YAML**

Copy the reviewed K4 LaWAM YAML into the new path and change only:

```yaml
tracker_name: joint_vlm_geact_action_k2_semantic_only_25k
output_dir: /data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k2_semantic_only_25k

semantic_plan:
  keyframe_indices: [4, 8]

joint_training:
  formal_recipe: hpc3_action_k2_semantic_only
  semantic_only: true
  planner_num_keyframes: 4
  selected_planner_keyframe_indices: [1, 3]
  selected_future_keyframe_offsets: [4, 8]
  num_keyframes: 2

diffusion_model:
  config:
    semantic_plan_num_keyframes: 2
```

Remove every `da3_*` field. Keep the K4 planner checkpoint path and all
optimizer/training values unchanged.

- [ ] **Step 5: Route the trainer from configuration**

In model preparation:

- validate the provider remains K4;
- construct only `Siglip2TargetEncoder` in semantic-only mode;
- leave `self.depth_teacher = None`;
- assign `self.joint_target_encoder =
  encode_dual_camera_future_semantic_targets`;
- construct `JointVLMGEActModel` with configured K2 selection.

In the train loop:

- select future frames from `selected_future_keyframe_offsets`;
- call the semantic-only target encoder without a depth argument;
- compute semantic times from `[4,8]` and reshape to `[B*2,2]`;
- pass `depth_labels=None`;
- add depth metrics only when they are tensors.

- [ ] **Step 6: Persist the runtime K2 contract**

In `joint_meta.json`, write:

```python
"semantic_only": True,
"planner_num_keyframes": 4,
"selected_planner_keyframe_indices": [1, 3],
"future_keyframe_offsets": [4, 8],
"num_keyframes": 2,
```

The exported standalone planner remains K4-compatible and retains its
original depth head; `joint_meta.json` is the authoritative runtime
conditioning contract.

- [ ] **Step 7: Implement the K2 preflight profile**

Branch on:

```python
semantic_only_action_profile = (
    formal_recipe_name == "hpc3_action_k2_semantic_only"
)
```

Reuse HPC3 path checks for LTX, planner, SigLIP2, and data. Skip DA3 required
paths for this profile and reject DA3 configuration fields. Enforce the exact
K2 geometry and all existing LaWAM action optimizer values.

- [ ] **Step 8: Create the K2 launcher**

Copy the current HPC3 action launcher, set:

```bash
#SBATCH --job-name=jvga_k2_sem
CONFIG=${CONFIG:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k2_semantic_only_hpc3.yaml}
```

For `smoke8`, pass `--max_train_steps 1`. Preserve predecode verification,
formal preflight, eight-process torchrun, offline model loading, and log
capture.

- [ ] **Step 9: Run local formal tests**

Run:

```bash
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q \
  tests/test_ge_act_dual_camera_planner.py \
  tests/test_joint_vlm_geact_training.py \
  tests/test_ge_act_siglip2_config.py
python -m compileall -q \
  qwen3_vl_semantic_planner/train_semantic_planner.py \
  ge_act/models/ltx_models/joint_vlm_geact.py \
  ge_act/runner/ge_trainer.py \
  ge_act/scripts/preflight_ltx_siglip2.py
bash -n ge_act/scripts/sbatch_train_joint_vlm_geact_action_k2_hpc3.sh
git diff --check
```

Expected: all tests and static checks pass.

- [ ] **Step 10: Commit**

```bash
git add \
  ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k2_semantic_only_hpc3.yaml \
  ge_act/scripts/sbatch_train_joint_vlm_geact_action_k2_hpc3.sh \
  ge_act/scripts/preflight_ltx_siglip2.py \
  ge_act/runner/ge_trainer.py \
  tests/test_ge_act_siglip2_config.py \
  tests/test_joint_vlm_geact_training.py
git commit -m "feat(train): add K2 semantic-only HPC3 recipe"
```

### Task 4: Deploy Smoke and Formal Training to HPC3

**Files:**
- Verify only; do not modify remote data or checkpoints.

**Interfaces:**
- Consumes: committed Tasks 1–3 and existing HPC3 assets.
- Produces: one completed smoke job and one active formal 25k Slurm job.

- [ ] **Step 1: Verify local commit and worktree**

Run:

```bash
git status --short
git log --oneline -6
git diff HEAD^ --check
```

Expected: clean worktree and no whitespace errors.

- [ ] **Step 2: Check HPC3 queue and storage**

Run:

```bash
ssh hpc3 'sinfo -p acd_u -o "%P %a %l %D %G"; squeue -u jhe724 -o "%.18i %.9P %.24j %.2t %.10M %R"; df -h /data/user/jhe724'
```

Expected: `acd_u` accepts eight-GPU jobs and the output filesystem has at
least 500 GB free.

- [ ] **Step 3: Synchronize committed code without deleting remote state**

Run:

```bash
git archive HEAD | ssh hpc3 \
  'mkdir -p /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af &&
   tar -xf - -C /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af'
```

Then verify:

```bash
ssh hpc3 'cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af &&
  /data/user/jhe724/.venvs/vlm4wam_joint/bin/python \
    ge_act/scripts/preflight_ltx_siglip2.py \
    --config ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k2_semantic_only_hpc3.yaml \
    --world-size 8 --require-joint-formal'
```

Expected: preflight exits zero.

- [ ] **Step 4: Submit the eight-GPU smoke**

Run:

```bash
SMOKE_JOB=$(
  ssh hpc3 'cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act &&
    mkdir -p logs &&
    RUN_KIND=smoke8 sbatch --parsable \
      scripts/sbatch_train_joint_vlm_geact_action_k2_hpc3.sh'
)
test -n "$SMOKE_JOB"
printf '%s\n' "$SMOKE_JOB"
```

Expected: one numeric Slurm job ID.

- [ ] **Step 5: Verify smoke completion**

Inspect:

```bash
ssh hpc3 "sacct -j '$SMOKE_JOB' --format=JobID,State,ExitCode,Elapsed,MaxRSS"
ssh hpc3 "tail -120 /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act/logs/slurm-joint-vlm-geact-action-k2-${SMOKE_JOB}.out"
```

Expected: `COMPLETED`, exit code `0:0`, one optimizer step, no DA3 load
message, and finite video/action/planner losses.

- [ ] **Step 6: Submit formal training**

Only after Step 5 succeeds:

```bash
FORMAL_JOB=$(
  ssh hpc3 'cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act &&
    RUN_KIND=formal sbatch --parsable \
      scripts/sbatch_train_joint_vlm_geact_action_k2_hpc3.sh'
)
test -n "$FORMAL_JOB"
printf '%s\n' "$FORMAL_JOB"
```

- [ ] **Step 7: Verify formal launch**

Run:

```bash
ssh hpc3 "squeue -j '$FORMAL_JOB' -o '%.18i %.9P %.24j %.2t %.10M %R'"
ssh hpc3 "tail -80 /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act/logs/slurm-joint-vlm-geact-action-k2-${FORMAL_JOB}.out"
```

Expected: the job is `R` or legitimately queued, and once running the log
shows global batch 256, K2 offsets `[4,8]`, semantic-only mode, and no DA3
initialization.

### Task 5: Evaluate Whether VLM Guidance Is Harmful

**Files:**
- Modify after a usable checkpoint exists: `ge_act/experiments/eval_libero_joint.py`
- Modify after a usable checkpoint exists: `ge_act/configs/eval/libero_joint.yaml`
- Test: `tests/test_joint_vlm_geact_libero_eval.py`

**Interfaces:**
- Consumes: K2 `joint_meta.json` and a checkpoint selected from 5k–25k.
- Produces: guided, guidance-off, and original-baseline success-rate tables.

- [ ] **Step 1: Add K2 runtime-metadata tests**

Require the evaluator to read `semantic_only`, selected offsets, selected
indices, and runtime K2 geometry from `joint_meta.json`, while loading the
exported planner as K4.

- [ ] **Step 2: Add a guidance-off evaluation switch**

Add `semantic_guidance_mode: predicted|off`. In `off` mode, preserve the
trained LTX/action weights but set the semantic condition mask to zero for
every rollout.

- [ ] **Step 3: Run evaluator tests**

Run:

```bash
PYTHONPATH="$PWD:$PWD/ge_act" pytest -q tests/test_joint_vlm_geact_libero_eval.py
```

Expected: all tests pass.

- [ ] **Step 4: Run the three evaluation conditions**

Use identical task order, seeds, rollout count, preprocessing, and action
horizon for:

1. K2 checkpoint with predicted guidance;
2. the same K2 checkpoint with guidance off;
3. the original unguided GE-Act baseline.

Write per-suite and aggregate success rates under a single timestamped result
directory so the negative-optimization interpretation is auditable.
