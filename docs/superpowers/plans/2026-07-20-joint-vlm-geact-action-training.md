# Joint VLM + GE-Act + Action Expert Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dual-camera K4 VLM planner, semantic-guided LTX video model, and GE-Act action expert optimize together on LIBERO for 50,000 steps, then launch a verified eight-GPU HPC3 run.

**Architecture:** Keep the existing composite `JointVLMGEActModel` and differentiable semantic-token path, but add the omitted action objective to the joint loss. Give action parameters an explicit optimizer group at the official GE-Act `5e-5`, freeze the Qwen vision encoder and LM head, and fail closed in preflight when the HPC3 action recipe drifts. Deploy a separate HPC3 config and launcher so the existing OLA video-only recipe remains unchanged.

**Tech Stack:** Python 3, PyTorch, Accelerate, DeepSpeed ZeRO-2, Transformers/Qwen3-VL, LTX-Video, GE-Act, pytest, YAML, Bash, Slurm.

## Global Constraints

- Train LTX video backbone at `2e-5`, semantic LTX modules at `1e-4`, GE-Act action modules at `5e-5`, Qwen language backbone at `3e-6`, and planner heads/query/token embeddings at `3e-5`.
- Optimize `loss_video + 1.0 * loss_action + 0.1 * planner_loss`.
- Freeze T5, VAE, SigLIP2 teacher, DA3 teacher, Qwen vision encoder, and Qwen LM head.
- Preserve separate main/wrist inputs, four memory frames, nine future frames, four semantic keyframes at offsets `[2, 4, 6, 8]`, 256 SigLIP2 tokens per keyframe, and a 36-step 15-channel action/state sequence.
- Use only the required predecoded LIBERO RGB cache during training; online video decoding must remain a hard error.
- Use eight GPUs, BF16, DeepSpeed ZeRO-2, global batch 128, 50,000 optimizer steps, 1,000 warmup steps, and checkpoints only at 40,000, 45,000, and 50,000.
- Prefer batch 4 per GPU with accumulation 4. If the eight-GPU smoke OOMs, use batch 2 per GPU with accumulation 8 without changing global batch or losses.
- Keep LTX and Qwen gradient checkpointing disabled in the formal recipe.
- Do not modify `ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml`; it is the OLA video-only recipe.
- Do not add or modify the user-owned untracked files `ge_act/data/agibotworld_dataset.py`, `ge_act/data/utils/domain_table.py`, or `ge_act/data/utils/get_actions.py`.
- Deploy only to `/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af` on HPC3 and do not touch `/data/user/jhe724/workspace/VLM4WAM` or unrelated Slurm jobs.

---

### Task 1: Put action loss into the joint objective and telemetry

**Files:**
- Modify: `tests/test_joint_vlm_geact_training.py`
- Modify: `ge_act/models/ltx_models/joint_vlm_geact.py:320-406`
- Modify: `ge_act/runner/ge_trainer.py:210-220,1970-2075`

**Interfaces:**
- Consumes: `loss_video: torch.Tensor`, `loss_action: torch.Tensor`, `planner_losses["loss"]: torch.Tensor`.
- Produces: `combine_joint_training_loss(loss_video, loss_action, planner_losses, *, action_loss_scale, planner_loss_weight) -> torch.Tensor`, `is_action_parameter_name(name: str) -> bool`, and joint logs containing `loss_action` and `action_grad_norm`.

- [ ] **Step 1: Write the failing loss-composition regression test**

Replace `test_joint_loss_uses_configured_planner_weight` with a test that makes every coefficient observable:

```python
def test_joint_loss_uses_action_and_configured_weights() -> None:
    contracts = _load_ge_trainer_symbols("combine_joint_training_loss")
    video_loss = torch.tensor(2.0, requires_grad=True)
    action_loss = torch.tensor(4.0, requires_grad=True)
    planner_loss = torch.tensor(3.0, requires_grad=True)

    total = contracts.combine_joint_training_loss(
        video_loss,
        action_loss,
        {"loss": planner_loss},
        action_loss_scale=1.0,
        planner_loss_weight=0.1,
    )

    torch.testing.assert_close(total, torch.tensor(6.3))
    total.backward()
    torch.testing.assert_close(video_loss.grad, torch.tensor(1.0))
    torch.testing.assert_close(action_loss.grad, torch.tensor(1.0))
    torch.testing.assert_close(planner_loss.grad, torch.tensor(0.1))
```

- [ ] **Step 2: Run the focused test and verify the old signature fails**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py::test_joint_loss_uses_action_and_configured_weights
```

Expected: `FAIL` because `combine_joint_training_loss` does not accept `loss_action` or `action_loss_scale`.

- [ ] **Step 3: Implement the three-term joint objective**

Change the helper to:

```python
def combine_joint_training_loss(
    loss_video: torch.Tensor,
    loss_action: torch.Tensor,
    planner_losses: Dict[str, torch.Tensor],
    *,
    action_loss_scale: float,
    planner_loss_weight: float,
) -> torch.Tensor:
    planner_loss = planner_losses.get("loss")
    losses = (loss_video, loss_action, planner_loss)
    if not all(torch.is_tensor(value) for value in losses):
        raise TypeError("joint video, action, and planner losses must be tensors")
    return (
        loss_video
        + float(action_loss_scale) * loss_action
        + float(planner_loss_weight) * planner_loss
    )
```

At the training call site, pass `loss_action` and `action_loss_scale`. Always add `loss_action` to `loss_components` in joint `train_mode: all`, so `require_finite_training_loss` rejects a non-finite action objective.

- [ ] **Step 4: Add separate action-gradient ownership and joint logging**

Add one reusable classifier beside the optimizer-group helper in `joint_vlm_geact.py` and import it in `ge_trainer.py`:

```python
def is_action_parameter_name(name: str) -> bool:
    return name.startswith("action_") or ".action_" in name
```

At the synchronized backward boundary, compute `action_grad_norm` from action-named LTX parameters and compute `ltx_grad_norm` from the remaining LTX parameters. Extend joint logs with:

```python
"loss_action": loss_action.detach().item(),
"action_grad_norm": joint_grad_metrics["action_grad_norm"].item(),
```

Add `action_grad_norm` and `loss_action` to the existing default/log-key assertions. Do not detach any loss before constructing the total.

- [ ] **Step 5: Run the joint unit test file**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py
```

Expected: all tests pass and the action-loss regression proves gradients `1.0`, `1.0`, and `0.1` for video, action, and planner losses.

- [ ] **Step 6: Commit the loss fix**

```bash
git add ge_act/models/ltx_models/joint_vlm_geact.py ge_act/runner/ge_trainer.py tests/test_joint_vlm_geact_training.py
git commit -m "fix(joint): optimize GE-Act action loss"
```

### Task 2: Add the official action LR group and enforce Qwen freezing

**Files:**
- Modify: `tests/test_joint_vlm_geact_training.py`
- Modify: `ge_act/models/ltx_models/joint_vlm_geact.py:327-406`
- Modify: `ge_act/runner/ge_trainer.py:1260-1460`

**Interfaces:**
- Consumes: `is_action_parameter_name(name: str) -> bool` from Task 1 and joint config keys `action_lr`, `freeze_qwen_vision`, and `freeze_qwen_lm_head`.
- Produces: five disjoint optimizer groups in order `base_ltx`, `semantic_ltx`, `action_ltx`, `qwen`, `planner_heads`; `configure_joint_planner_trainability(planner, *, freeze_qwen_vision, freeze_qwen_lm_head) -> None`.

- [ ] **Step 1: Extend the tiny model and write failing five-group assertions**

Add an action parameter to `TinyGateOpenLTX`:

```python
self.action_proj = nn.Linear(2, 2, bias=False)
```

Pass `action_lr=5e-5` to every `build_joint_optimizer_parameter_groups` call and assert:

```python
assert [group["name"] for group in groups] == [
    "base_ltx",
    "semantic_ltx",
    "action_ltx",
    "qwen",
    "planner_heads",
]
assert [group["lr"] for group in groups] == [2e-5, 1e-4, 5e-5, 3e-6, 3e-5]
assert id(joint.ltx.action_proj.weight) in ids_by_group["action_ltx"]
```

Change Qwen ownership assertions so `model.proj` remains in `qwen`, while `model.visual` and `model.lm_head` are frozen and absent after trainability configuration.

- [ ] **Step 2: Run optimizer/trainability tests and verify failure**

Run:

```bash
pytest -q \
  tests/test_joint_vlm_geact_training.py::test_joint_optimizer_groups_are_disjoint_complete_and_ordered \
  tests/test_joint_vlm_geact_training.py::test_joint_teacher_parameters_are_frozen_and_excluded
```

Expected: `FAIL` because there is no `action_ltx` group and the current trainer re-enables all Qwen parameters.

- [ ] **Step 3: Implement explicit action optimizer ownership**

Update the signature and grouping logic:

```python
def build_joint_optimizer_parameter_groups(
    model: JointVLMGEActModel,
    ltx_lr: float,
    semantic_lr: float,
    action_lr: float,
    qwen_lr: float,
    planner_head_lr: float,
) -> list[dict[str, Any]]:
    ...
    for name, parameter in _named_trainable_parameters(model.ltx):
        if is_action_parameter_name(name):
            add("action_ltx", parameter)
        elif "semantic_" in name:
            add("semantic_ltx", parameter)
        else:
            add("base_ltx", parameter)
```

Use group order and LRs:

```python
group_order = (
    "base_ltx", "semantic_ltx", "action_ltx", "qwen", "planner_heads"
)
learning_rates = {
    "base_ltx": ltx_lr,
    "semantic_ltx": semantic_lr,
    "action_ltx": action_lr,
    "qwen": qwen_lr,
    "planner_heads": planner_head_lr,
}
```

- [ ] **Step 4: Implement fail-closed Qwen trainability**

Add:

```python
def configure_joint_planner_trainability(
    planner: torch.nn.Module,
    *,
    freeze_qwen_vision: bool,
    freeze_qwen_lm_head: bool,
) -> None:
    planner.requires_grad_(True)
    qwen = getattr(planner, "model", None)
    if not isinstance(qwen, torch.nn.Module):
        raise TypeError("joint planner must expose Qwen as planner.model")
    for enabled, attribute in (
        (freeze_qwen_vision, "visual"),
        (freeze_qwen_lm_head, "lm_head"),
    ):
        module = getattr(qwen, attribute, None)
        if enabled:
            if not isinstance(module, torch.nn.Module):
                raise ValueError(f"Qwen module is missing required {attribute}")
            module.requires_grad_(False)
            module.eval()
    planner.train()
```

Call this helper both after checkpoint loading and in `prepare_trainable_parameters`, replacing both broad `wrapper.requires_grad_(True)` calls. Pass `action_lr=float(joint_config["action_lr"])` to the optimizer builder.

- [ ] **Step 5: Verify five groups, exact LRs, complete ownership, and frozen modules**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py
```

Expected: all tests pass; no trainable parameter is duplicated or omitted; Qwen `visual` and `lm_head` are frozen; Qwen language parameters and planner heads remain trainable.

- [ ] **Step 6: Commit optimizer and freeze behavior**

```bash
git add ge_act/models/ltx_models/joint_vlm_geact.py ge_act/runner/ge_trainer.py tests/test_joint_vlm_geact_training.py
git commit -m "feat(joint): separate action optimizer ownership"
```

### Task 3: Make joint checkpoints self-describing for action training

**Files:**
- Modify: `tests/test_joint_vlm_geact_training.py:1340-1450`
- Modify: `ge_act/runner/ge_trainer.py:513-650`

**Interfaces:**
- Consumes: the five optimizer groups from Task 2 and config fields `action_loss_scale`, `train_mode`, and `joint_training.action_lr`.
- Produces: `joint_meta.json` with the action objective, action LR, mode, and trainable action parameter count.

- [ ] **Step 1: Write failing metadata assertions**

Add an `action_ltx` group to the fake optimizer and assert:

```python
assert joint_meta["optimizer_group_lrs"]["action_ltx"] == 5e-5
assert joint_meta["action_loss_scale"] == 1.0
assert joint_meta["train_mode"] == "all"
assert joint_meta["trainable_parameters"]["action_ltx"] > 0
```

Set `args.action_loss_scale = 1.0`, `args.train_mode = "all"`, and add a named `action_proj` parameter to the test LTX module.

- [ ] **Step 2: Run the checkpoint test and verify failure**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py::test_joint_checkpoint_exports_both_models_metadata_and_training_state
```

Expected: `FAIL` because `action_ltx`, `action_loss_scale`, `train_mode`, and the action parameter count are missing.

- [ ] **Step 3: Export complete action metadata**

Require all five optimizer names and add:

```python
"action_loss_scale": float(args.action_loss_scale),
"train_mode": str(args.train_mode),
"trainable_parameters": {
    "ltx": sum(p.numel() for p in model.ltx.parameters() if p.requires_grad),
    "action_ltx": sum(
        p.numel()
        for name, p in model.ltx.named_parameters()
        if p.requires_grad and is_action_parameter_name(name)
    ),
    "planner": sum(p.numel() for p in model.planner.parameters() if p.requires_grad),
},
```

Keep `accelerator.save_state` unchanged so optimizer, scheduler, RNG, and distributed state continue to resume exactly.

- [ ] **Step 4: Verify checkpoint and resume contracts**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py -k 'checkpoint or resume'
```

Expected: all checkpoint and resume tests pass.

- [ ] **Step 5: Commit checkpoint metadata**

```bash
git add ge_act/runner/ge_trainer.py tests/test_joint_vlm_geact_training.py
git commit -m "feat(joint): record action training metadata"
```

### Task 4: Add a fail-closed HPC3 action recipe

**Files:**
- Create: `ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py`
- Modify: `tests/test_ge_act_siglip2_config.py`

**Interfaces:**
- Consumes: the approved HPC3 paths and five learning rates.
- Produces: a static/runtime preflight-valid 50k action recipe with global batch 128 and no online decoding.

- [ ] **Step 1: Add a failing HPC3 config-contract test**

Define `JOINT_ACTION_HPC3_CONFIG_PATH` and assert these exact values:

```python
assert config["return_video"] is True
assert config["return_action"] is True
assert config["train_mode"] == "all"
assert config["action_loss_scale"] == 1.0
assert config["add_state"] is True
assert config["rand_init_action"] is False
assert config["train_steps"] == 50_000
assert config["save_steps"] == [40_000, 45_000, 50_000]
assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
assert config["gradient_checkpointing"] is False
assert config["joint_training"]["action_lr"] == 5e-5
assert config["joint_training"]["qwen_lr"] == 3e-6
assert config["joint_training"]["planner_head_lr"] == 3e-5
assert config["joint_training"]["freeze_qwen_vision"] is True
assert config["joint_training"]["freeze_qwen_lm_head"] is True
assert config["diffusion_model"]["config"]["action_expert"] is True
assert config["diffusion_model"]["config"]["action_in_channels"] == 15
assert config["diffusion_model"]["config"]["action_out_channels"] == 15
for split in ("train", "val"):
    assert config["data"][split]["pack_action_state"] is True
    assert config["data"][split]["require_predecoded"] is True
```

- [ ] **Step 2: Run the new test and verify missing config failure**

Run:

```bash
pytest -q tests/test_ge_act_siglip2_config.py -k joint_action_hpc3
```

Expected: `FAIL` because the HPC3 action YAML does not exist.

- [ ] **Step 3: Create the HPC3 YAML without modifying the OLA YAML**

Copy the declarative geometry from the OLA joint recipe, then set:

```yaml
output_dir: /data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k4_50k
pretrained_model_name_or_path: /data/user/jhe724/junjie/weights/LTX-Video
return_action: true
return_video: true
train_mode: all
action_loss_scale: 1.0
add_state: true
rand_init_action: false
train_steps: 50000
save_steps: [40000, 45000, 50000]
batch_size: 4
gradient_accumulation_steps: 4
gradient_checkpointing: false
lr: 2.0e-5
semantic_lr: 1.0e-4
joint_training:
  enabled: true
  formal_recipe: hpc3_action
  planner_loss_weight: 0.1
  action_lr: 5.0e-5
  qwen_lr: 3.0e-6
  planner_head_lr: 3.0e-5
  freeze_qwen_vision: true
  freeze_qwen_lm_head: true
  qwen_gradient_checkpointing: false
```

Use these exact approved paths:

```yaml
semantic_plan:
  planner_checkpoint: /data/user/jhe724/junjie/vlm4wam_joint_assets/planner_step_030000
joint_training:
  siglip2_model_dir: /data/user/jhe724/junjie/weights/siglip2-large-patch16-256
  da3_ckpt_dir: /data/user/jhe724/junjie/vlm4wam_joint_assets/DA3-LARGE-1.1
  da3_code_root: /data/user/jhe724/junjie/vlm4wam_joint_assets/Depth-Anything-3
diffusion_model:
  model_path: /data/user/jhe724/junjie/vlm4wam_joint_assets/ltx_step_50000
data:
  train:
    data_roots: [/data/user/jhe724/junjie/datasets/LIBERO-fastwam, /data/user/jhe724/junjie/datasets/LIBERO-fastwam, /data/user/jhe724/junjie/datasets/LIBERO-fastwam, /data/user/jhe724/junjie/datasets/LIBERO-fastwam]
    predecoded_video_root: /data/user/jhe724/junjie/datasets/LIBERO-fastwam-predecoded-rgb
    pack_action_state: true
```

Repeat the same roots and `pack_action_state` contract for validation. Preserve `action_chunk: 36`, `state_key: observation.state`, and `action_key: action`. Set the model action geometry to `action_expert: true`, 16 action heads, head dimension 32, and 15 input/output channels, matching the loaded `ltx_step_50000` checkpoint.

- [ ] **Step 4: Extend preflight with an HPC3 action profile**

Add `hpc3_action` to an approved recipe map while retaining the existing OLA defaults. For the action profile require:

```python
if config.get("return_action") is not True:
    errors.append("joint action training requires return_action=true")
if config.get("train_mode") != "all":
    errors.append("joint action training requires train_mode=all")
if config.get("action_loss_scale") != 1.0:
    errors.append("joint action loss scale must be 1.0")
if joint.get("action_lr") != 5e-5:
    errors.append("joint action lr must be 5e-5")
if joint.get("qwen_lr") != 3e-6:
    errors.append("joint action Qwen lr must be 3e-6")
if not joint.get("freeze_qwen_vision") or not joint.get("freeze_qwen_lm_head"):
    errors.append("joint action training must freeze Qwen vision and LM head")
if (config.get("batch_size"), config.get("gradient_accumulation_steps")) not in ((4, 4), (2, 8)):
    errors.append("joint action training requires batch/accumulation 4/4 or 2/8")
```

Also require `train_steps == 50_000`, `save_steps == [40_000, 45_000, 50_000]`, action model geometry, state packing, and the HPC3 path contract. Keep the existing OLA profile checks at 30k, Qwen `1e-6`, and batch/accumulation `4/4` so this task cannot silently alter the old run.

- [ ] **Step 5: Run config and preflight tests**

Run:

```bash
pytest -q tests/test_ge_act_siglip2_config.py
```

Expected: all existing OLA tests and new HPC3 action tests pass.

- [ ] **Step 6: Commit the HPC3 action recipe**

```bash
git add \
  ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml \
  ge_act/scripts/preflight_ltx_siglip2.py \
  tests/test_ge_act_siglip2_config.py
git commit -m "feat(joint): add HPC3 action training recipe"
```

### Task 5: Add isolated smoke/formal Slurm launch support

**Files:**
- Modify: `ge_act/main.py`
- Create: `ge_act/scripts/sbatch_train_joint_vlm_geact_action_hpc3.sh`
- Modify: `tests/test_ge_act_siglip2_config.py`

**Interfaces:**
- Consumes: the HPC3 YAML from Task 4 and environment `/data/user/jhe724/.venvs/vlm4wam_joint`.
- Produces: `RUN_KIND=smoke8|formal` Slurm entry point with isolated outputs and a CLI `--output_dir_override`.

- [ ] **Step 1: Write failing CLI and launcher contract tests**

Extend the existing main override test with:

```python
"--output_dir_override",
str(tmp_path / "smoke-output"),
```

and assert:

```python
assert captured["config_overrides"]["output_dir"] == str(tmp_path / "smoke-output")
```

For the new launcher assert eight GPUs, `acd_u`, 64 CPUs, the joint venv, the HPC3 config, strict predecoded verification, formal preflight, and a two-step `smoke8` override.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
pytest -q tests/test_ge_act_siglip2_config.py -k 'main_smoke or joint_action_hpc3_launcher'
```

Expected: `FAIL` because the CLI flag and launcher do not exist.

- [ ] **Step 3: Add the output-directory override**

Add to `ge_act/main.py`:

```python
parser.add_argument(
    "--output_dir_override",
    type=str,
    default=None,
    help="isolated output directory override for smoke or deployment runs",
)
```

and:

```python
if args.output_dir_override is not None:
    config_overrides["output_dir"] = args.output_dir_override
```

- [ ] **Step 4: Create the HPC3 Slurm launcher**

The checked-in script must start with:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=jvga_k4_50k
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm-joint-vlm-geact-action-%j.out
set -euo pipefail
```

Use:

```bash
GE_ACT_ROOT=${GE_ACT_ROOT:-/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act}
PY=${PY:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/python}
TORCHRUN=${TORCHRUN:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/torchrun}
CONFIG=${CONFIG:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml}
RUN_KIND=${RUN_KIND:-formal}
```

Export offline Hugging Face flags, `PYTHONPATH`, `TOKENIZERS_PARALLELISM=false`, and all BLAS/OpenMP thread counts as 1. Before training run:

```bash
"$PY" scripts/predecode_lerobot_videos.py --config "$CONFIG" --verify-only
"$PY" scripts/preflight_ltx_siglip2.py \
  --config "$CONFIG" --world-size 8 --require-joint-formal
```

For `smoke8`, invoke two optimizer steps and an isolated directory:

```bash
MAIN_ARGS+=(
  --max_train_steps 2
  --output_dir_override "${SMOKE_OUTPUT_DIR:-/data/user/jhe724/junjie/outputs/smoke_joint_vlm_geact_action_${SLURM_JOB_ID:-manual}}"
)
```

For `formal`, do not override output or batch settings. Execute torchrun with `--nproc_per_node=8`.

- [ ] **Step 5: Verify Python and shell contracts**

Run:

```bash
pytest -q tests/test_ge_act_siglip2_config.py
bash -n ge_act/scripts/sbatch_train_joint_vlm_geact_action_hpc3.sh
```

Expected: pytest passes and `bash -n` exits 0.

- [ ] **Step 6: Commit deployment support**

```bash
git add ge_act/main.py ge_act/scripts/sbatch_train_joint_vlm_geact_action_hpc3.sh tests/test_ge_act_siglip2_config.py
git commit -m "feat(joint): add HPC3 smoke and formal launcher"
```

### Task 6: Run complete local verification and synchronize only tracked code

**Files:**
- Verify: all files committed in Tasks 1-5
- Preserve: the three user-owned untracked data files listed in Global Constraints

**Interfaces:**
- Consumes: complete local implementation.
- Produces: a clean reviewed commit set deployed to the isolated HPC3 workspace.

- [ ] **Step 1: Run the focused training/config suite**

```bash
pytest -q \
  tests/test_joint_vlm_geact_training.py \
  tests/test_ge_act_siglip2_config.py \
  tests/test_ge_act_semantic_training_contract.py
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and compile checks**

```bash
python -m py_compile \
  ge_act/main.py \
  ge_act/runner/ge_trainer.py \
  ge_act/models/ltx_models/joint_vlm_geact.py \
  ge_act/scripts/preflight_ltx_siglip2.py
bash -n ge_act/scripts/sbatch_train_joint_vlm_geact_action_hpc3.sh
```

Expected: both commands exit 0 without output.

- [ ] **Step 3: Audit the diff and tracked file scope**

```bash
git status --short
git diff --check HEAD~4..HEAD
git diff --stat HEAD~4..HEAD
```

Expected: `git diff --check` is empty; the three pre-existing data files are still untracked and absent from every commit.

- [ ] **Step 4: Synchronize committed files to HPC3 without deleting remote state**

From the repository root:

```bash
git archive HEAD | ssh hpc3 \
  'mkdir -p /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af && tar -xf - -C /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af'
```

Expected: archive extraction completes successfully; no `--delete` operation is used.

- [ ] **Step 5: Verify the deployed revision and assets**

```bash
ssh hpc3 'cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af && \
  /data/user/jhe724/.venvs/vlm4wam_joint/bin/python ge_act/scripts/preflight_ltx_siglip2.py \
    --config ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml \
    --world-size 8 --require-joint-formal'
```

Expected: preflight exits 0 and reports no missing checkpoint, teacher, dataset, cache, or geometry path.

### Task 7: Prove the full gradient path with an eight-GPU smoke job

**Files:**
- Runtime output: `/data/user/jhe724/junjie/outputs/smoke_joint_vlm_geact_action_<job-id>`
- Runtime log: `/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act/logs/slurm-joint-vlm-geact-action-<job-id>.out`

**Interfaces:**
- Consumes: deployed code/config and verified assets from Task 6.
- Produces: at least one completed optimizer update with finite nonzero video/action/planner losses and LTX/action/Qwen gradient norms.

- [ ] **Step 1: Submit the isolated smoke job**

```bash
ssh hpc3 'cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act && \
  mkdir -p logs && RUN_KIND=smoke8 sbatch --parsable scripts/sbatch_train_joint_vlm_geact_action_hpc3.sh'
```

Expected: Slurm returns one numeric job ID.

- [ ] **Step 2: Wait for completion without polling more often than once per minute**

```bash
ssh hpc3 'squeue -j <job-id> -o "%.18i %.2t %.10M %.6D %R"'
```

Expected: the job progresses from `PD`/`R` to leaving the queue. Replace `<job-id>` with the numeric ID returned by Step 1; this is a runtime value, not a code placeholder.

- [ ] **Step 3: Inspect the completed smoke log and accounting status**

```bash
ssh hpc3 'sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS && \
  tail -200 /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act/logs/slurm-joint-vlm-geact-action-<job-id>.out'
```

Expected: Slurm state `COMPLETED`, exit code `0:0`, and two completed optimizer steps.

- [ ] **Step 4: Validate the smoke metrics**

Read the final synchronized-step log record and confirm:

```text
isfinite(loss_video) and loss_video > 0
isfinite(loss_action) and loss_action > 0
isfinite(planner_loss) and planner_loss > 0
isfinite(ltx_grad_norm) and ltx_grad_norm > 0
isfinite(action_grad_norm) and action_grad_norm > 0
isfinite(vlm_grad_norm) and vlm_grad_norm > 0
```

Expected: all six checks are true. If batch 4 OOMs, change only the HPC3 YAML to `batch_size: 2` and `gradient_accumulation_steps: 8`, rerun Task 4 tests/preflight, commit that fallback, resynchronize, and repeat the smoke.

### Task 8: Launch and verify the formal 50k run

**Files:**
- Runtime output: `/data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k4_50k`
- Runtime log: `/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act/logs/slurm-joint-vlm-geact-action-<job-id>.out`

**Interfaces:**
- Consumes: a passing eight-GPU smoke result.
- Produces: a running 50,000-step formal job with global batch 128 and the complete video/action/planner objective.

- [ ] **Step 1: Ensure no formal output collision exists**

```bash
ssh hpc3 'test ! -e /data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k4_50k'
```

Expected: exit code 0. If the directory exists, stop and report it instead of overwriting or deleting it.

- [ ] **Step 2: Submit the formal job**

```bash
ssh hpc3 'cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act && \
  RUN_KIND=formal sbatch --parsable scripts/sbatch_train_joint_vlm_geact_action_hpc3.sh'
```

Expected: Slurm returns one numeric job ID.

- [ ] **Step 3: Verify allocation and first synchronized update**

```bash
ssh hpc3 'squeue -j <job-id> -o "%.18i %.2t %.10M %.6D %R" && \
  tail -120 /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act/logs/slurm-joint-vlm-geact-action-<job-id>.out'
```

Expected: eight GPUs on one `acd_u` node; log shows train step progress, five LR groups, finite three-term losses, and finite nonzero three-way gradient norms.

- [ ] **Step 4: Record the handoff facts**

Report the Slurm job ID, node, effective batch equation, output directory, log path, five LRs, loss equation, current step/throughput, and the expected wall-clock estimate derived from measured post-warmup seconds per optimizer step. Do not claim completion while the 50k job is merely running.
