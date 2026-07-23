# LaWAM-Aligned Qwen Vision Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the joint VLM + GE-Act HPC3 recipe train Qwen vision with LaWAM's LIBERO optimizer/freeze policy and an effective global batch of 256.

**Architecture:** Extend the existing joint planner freeze policy instead of adding a second model wrapper. Keep Qwen vision and language parameters in distinct optimizer groups for auditable membership, while applying LaWAM's common `1e-4` VLM learning rate. Preserve GE-Act-specific model geometry and loss weights.

**Tech Stack:** Python, PyTorch, Hugging Face Qwen3-VL, Diffusers schedulers, YAML, pytest, DeepSpeed ZeRO-2.

## Global Constraints

- Qwen vision is fully trainable, including merger modules.
- Qwen language layers 0 through 15, token embeddings, and LM head are frozen.
- Remaining Qwen language layers are trainable.
- Trainable Qwen vision and language parameters use learning rate `1e-4`.
- SigLIP2 and DA3 teachers remain frozen.
- Global batch is `4 per GPU * 8 GPUs * 8 accumulation = 256`.
- Training runs for 25000 optimizer steps and saves at 5000-step intervals.
- Warmup is 1500 steps; scheduler is cosine decay with minimum LR `5e-7`.
- AdamW uses betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-8`.
- Seed is `2026`, max gradient norm is `1.0`, and Qwen/LTX gradient checkpointing remains disabled.

---

### Task 1: Qwen Trainability Policy

**Files:**
- Modify: `tests/test_joint_vlm_geact_training.py`
- Modify: `ge_act/runner/ge_trainer.py`

**Interfaces:**
- Consumes: a planner exposing Qwen as `planner.model`
- Produces: `configure_joint_planner_trainability(planner, *, freeze_qwen_vision, freeze_qwen_lm_head, freeze_qwen_embeddings, keep_qwen_first_n_layers)`

- [ ] **Step 1: Extend the tiny Qwen fixture and write the failing policy test**

Add a tiny language model with `embed_tokens` and 20 transformer layers, expose
it through `TinyQwenModel.model.language_model`, and add:

```python
def test_lawam_qwen_policy_trains_vision_and_freezes_lower_language() -> None:
    symbols = _load_ge_trainer_symbols(
        "_resolve_qwen_language_model",
        "configure_joint_planner_trainability",
    )
    planner = TinyPlanner()
    symbols.configure_joint_planner_trainability(
        planner,
        freeze_qwen_vision=False,
        freeze_qwen_lm_head=True,
        freeze_qwen_embeddings=True,
        keep_qwen_first_n_layers=16,
    )

    assert all(p.requires_grad for p in planner.model.visual.parameters())
    assert all(
        not p.requires_grad
        for p in planner.model.model.language_model.embed_tokens.parameters()
    )
    assert all(
        not p.requires_grad
        for layer in planner.model.model.language_model.layers[:16]
        for p in layer.parameters()
    )
    assert all(
        p.requires_grad
        for layer in planner.model.model.language_model.layers[16:]
        for p in layer.parameters()
    )
    assert all(not p.requires_grad for p in planner.model.lm_head.parameters())
```

- [ ] **Step 2: Run the policy test and verify RED**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py::test_lawam_qwen_policy_trains_vision_and_freezes_lower_language
```

Expected: FAIL because the resolver and new keyword arguments do not exist.

- [ ] **Step 3: Implement the minimal freeze policy**

Add a resolver that accepts Qwen's supported language-model paths:

```python
def _resolve_qwen_language_model(qwen: torch.nn.Module) -> torch.nn.Module:
    nested = getattr(qwen, "model", None)
    for candidate in (
        getattr(nested, "language_model", None),
        getattr(qwen, "language_model", None),
    ):
        if isinstance(candidate, torch.nn.Module):
            return candidate
    raise ValueError("Qwen module is missing its language model")
```

Extend `configure_joint_planner_trainability` to reset all planner parameters
trainable, optionally freeze vision, freeze embeddings returned by
`get_input_embeddings()` or `language_model.embed_tokens`, freeze the first
configured `language_model.layers`, and freeze the LM head. Frozen modules
must also be placed in eval mode.

Pass the two new configuration fields at all three trainer call sites:

```python
freeze_qwen_embeddings=bool(
    joint_config.get("freeze_qwen_embeddings", False)
),
keep_qwen_first_n_layers=int(
    joint_config.get("keep_qwen_first_n_layers", 0)
),
```

- [ ] **Step 4: Run focused policy and trainer tests**

Run:

```bash
pytest -q \
  tests/test_joint_vlm_geact_training.py::test_lawam_qwen_policy_trains_vision_and_freezes_lower_language \
  tests/test_joint_vlm_geact_training.py::test_joint_teacher_parameters_are_frozen_and_excluded \
  tests/test_joint_vlm_geact_training.py::test_joint_train_reapplies_qwen_freeze_after_composite_train_mode
```

Expected: 3 passed.

- [ ] **Step 5: Commit the policy**

```bash
git add ge_act/runner/ge_trainer.py tests/test_joint_vlm_geact_training.py
git commit -m "feat(train): apply LaWAM Qwen freeze policy"
```

### Task 2: Explicit Qwen Vision Optimizer Group

**Files:**
- Modify: `tests/test_joint_vlm_geact_training.py`
- Modify: `ge_act/models/ltx_models/joint_vlm_geact.py`
- Modify: `ge_act/runner/ge_trainer.py`

**Interfaces:**
- Consumes: `JointVLMGEActModel` with an exposed Qwen visual module
- Produces: optimizer groups `qwen_vision` and `qwen`, each with explicit LR

- [ ] **Step 1: Update the optimizer test for a distinct vision group**

Change the group expectation to:

```python
assert [group["name"] for group in groups] == [
    "base_ltx",
    "semantic_ltx",
    "action_ltx",
    "qwen_vision",
    "qwen",
    "planner_heads",
]
assert id(joint.planner.model.visual.weight) in ids_by_group["qwen_vision"]
assert id(joint.planner.model.proj.weight) in ids_by_group["qwen"]
```

Invoke the builder with `qwen_vision_lr=1e-4` and `qwen_lr=1e-4`. Continue to
assert disjointness and exact coverage of all trainable parameters.

- [ ] **Step 2: Run the optimizer test and verify RED**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py::test_joint_optimizer_groups_are_disjoint_complete_and_ordered
```

Expected: FAIL because `qwen_vision_lr` and the group do not exist.

- [ ] **Step 3: Implement vision parameter classification**

Extend `build_joint_optimizer_parameter_groups` with:

```python
qwen_vision_lr: float,
```

Resolve the Qwen visual module from `planner_model.visual` or
`planner_model.model.visual`, collect its parameter IDs, and classify trainable
planner parameters as:

```python
if name.startswith("model."):
    add(
        "qwen_vision" if id(parameter) in qwen_vision_parameter_ids else "qwen",
        parameter,
    )
else:
    add("planner_heads", parameter)
```

Add `qwen_vision` before `qwen` in group order and pass
`joint_config["qwen_vision_lr"]` from `Trainer.prepare_optimizer`.

- [ ] **Step 4: Run optimizer and joint trainer tests**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py -k \
  "optimizer_groups_are_disjoint_complete_and_ordered or teacher_parameters_are_frozen_and_excluded"
```

Expected: 2 passed.

- [ ] **Step 5: Commit optimizer grouping**

```bash
git add \
  ge_act/models/ltx_models/joint_vlm_geact.py \
  ge_act/runner/ge_trainer.py \
  tests/test_joint_vlm_geact_training.py
git commit -m "feat(train): separate Qwen vision optimizer group"
```

### Task 3: LaWAM Scheduler and HPC3 Recipe

**Files:**
- Modify: `tests/test_joint_vlm_geact_training.py`
- Modify: `tests/test_ge_act_siglip2_config.py`
- Modify: `ge_act/runner/ge_trainer.py`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py`
- Modify: `ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml`

**Interfaces:**
- Consumes: existing joint trainer CLI/config loading
- Produces: a preflight-enforced LaWAM-aligned 25k recipe

- [ ] **Step 1: Rewrite config contract assertions before production changes**

Update `test_joint_action_hpc3_config_matches_approved_50k_recipe` to
`test_joint_action_hpc3_config_matches_lawam_recipe` and assert:

```python
assert config["seed"] == 2026
assert config["train_steps"] == 25_000
assert config["save_steps"] == [5_000, 10_000, 15_000, 20_000, 25_000]
assert config["batch_size"] == 4
assert config["gradient_accumulation_steps"] == 8
assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 256
assert config["lr_scheduler"] == "cosine_with_min_lr"
assert config["lr_warmup_steps"] == 1500
assert config["lr_min"] == 5e-7
assert config["weight_decay"] == 1e-8
assert joint["qwen_vision_lr"] == 1e-4
assert joint["qwen_lr"] == 1e-4
assert joint["freeze_qwen_vision"] is False
assert joint["freeze_qwen_embeddings"] is True
assert joint["keep_qwen_first_n_layers"] == 16
assert joint["freeze_qwen_lm_head"] is True
```

Add a scheduler unit test with two optimizer groups (`1e-4` and `2e-5`) and
assert that both reach the same absolute `5e-7` floor after warmup and cosine
decay.

- [ ] **Step 2: Run focused config tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_ge_act_siglip2_config.py::test_joint_action_hpc3_config_matches_lawam_recipe \
  tests/test_ge_act_siglip2_config.py::test_joint_action_hpc3_preflight_rejects_objective_and_geometry_drift \
  tests/test_joint_vlm_geact_training.py::test_joint_teacher_parameters_are_frozen_and_excluded
```

Expected: FAIL on the old 50k/batch-128/frozen-vision recipe.

- [ ] **Step 3: Update recipe and scheduler construction**

Set the YAML values from the global constraints, including:

```yaml
train_steps: 25000
save_steps: [5000, 10000, 15000, 20000, 25000]
seed: 2026
batch_size: 4
gradient_accumulation_steps: 8
weight_decay: 1.0e-8
lr_scheduler: cosine_with_min_lr
lr_warmup_steps: 1500
lr_min: 5.0e-7
joint_training:
  qwen_vision_lr: 1.0e-4
  qwen_lr: 1.0e-4
  freeze_qwen_vision: false
  freeze_qwen_embeddings: true
  keep_qwen_first_n_layers: 16
  freeze_qwen_lm_head: true
```

The installed Diffusers scheduler factory does not implement
`cosine_with_min_lr`. Add a local `torch.optim.lr_scheduler.LambdaLR` helper
that computes one lambda per optimizer group. Each group must decay to the
same absolute floor, so the minimum ratio is `min_lr / group["lr"]`. Select
this helper only for the joint `cosine_with_min_lr` recipe and preserve the
existing Diffusers scheduler path for all other recipes.

Update preflight to enforce the same values and precise error messages.

- [ ] **Step 4: Run config, preflight, and joint trainer suites**

Run:

```bash
pytest -q \
  tests/test_ge_act_siglip2_config.py \
  tests/test_joint_vlm_geact_training.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the formal recipe**

```bash
git add \
  ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml \
  ge_act/runner/ge_trainer.py \
  ge_act/scripts/preflight_ltx_siglip2.py \
  tests/test_ge_act_siglip2_config.py \
  tests/test_joint_vlm_geact_training.py
git commit -m "feat(train): align joint HPC3 recipe with LaWAM"
```

### Task 4: Full Verification and Smoke-Test Readiness

**Files:**
- Verify only

**Interfaces:**
- Consumes: completed implementation
- Produces: evidence that the branch is ready to synchronize and run

- [ ] **Step 1: Run static and unit verification**

Run:

```bash
python -m compileall -q \
  ge_act/runner/ge_trainer.py \
  ge_act/models/ltx_models/joint_vlm_geact.py \
  ge_act/scripts/preflight_ltx_siglip2.py
pytest -q \
  tests/test_ge_act_siglip2_config.py \
  tests/test_joint_vlm_geact_training.py
PYTHONPATH="$PWD:$PWD/ge_act" python - <<'PY'
from pathlib import Path
import yaml
from ge_act.scripts.preflight_ltx_siglip2 import collect_preflight_errors

path = Path(
    "ge_act/configs/ltx_model/libero/"
    "video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml"
)
errors = collect_preflight_errors(
    yaml.safe_load(path.read_text()),
    world_size=8,
    check_paths=False,
    require_joint_formal=True,
)
assert not errors, errors
PY
git diff --check
git status --short
```

Expected: compile exit 0, all tests pass, preflight succeeds, no whitespace
errors, and only intentional commits are present.

- [ ] **Step 2: Inspect the committed diff**

Run:

```bash
git log --oneline -5
git show --stat --oneline HEAD
```

Expected: the LaWAM-alignment commits contain only the planned source, config,
test, and documentation files.
