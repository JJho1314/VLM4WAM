# Joint Qwen3-VL and GE-Act Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jointly fine-tune the complete dual-camera K4 Qwen3-VL planner and GE-Act LTX model so the video loss backpropagates into Qwen while frozen SigLIP2/DA3 teachers preserve planner alignment.

**Architecture:** A single `JointVLMGEActModel` contains `PlannerWrapper` and the LTX transformer and is the only model prepared by Accelerate/DeepSpeed. It runs Qwen once, returns planner losses, reshapes differentiable K4 SigLIP2 tokens, and calls LTX with those tokens in the same forward pass.

**Tech Stack:** Python 3.10, PyTorch, Transformers Qwen3-VL, Diffusers LTX, Accelerate, DeepSpeed ZeRO-2, pytest, YAML, 8x H100.

## Global Constraints

- Initialize VLM from dual-camera K4 WSA `step_030000` and LTX from OLA `ltx_step_50000`.
- Preserve legacy frozen K1/K4 providers and configs; add a separate joint path.
- Use K4 offsets `2,4,6,8`, two camera views, 256 tokens/keyframe, and width 1024.
- Compute `total_loss = video_loss + 0.1 * planner_alignment_loss` from one Qwen forward.
- Keep SigLIP2, DA3, VAE, and T5 frozen and outside the optimizer.
- Train all Qwen3-VL parameters, including visual tower and LM head.
- Use LR groups: LTX `2e-5`, semantic LTX `1e-4`, Qwen `1e-6`, planner heads/queries `3e-5`.
- Formal batch contract: 8 GPUs, batch/GPU 1, accumulation 16, global batch 128, bf16, ZeRO-2.
- Do not commit weights, caches, logs, smoke outputs, or unrelated untracked files.

---

### Task 1: Metadata-driven K1/K4 provider geometry

**Files:**
- Modify: `ge_act/runner/ge_trainer.py`
- Test: `tests/test_ge_act_semantic_training_contract.py`

**Interfaces:**
- Consumes: `provider.num_keyframes`, `provider.target_tokens_per_keyframe`, and `provider.predict(...)`.
- Produces: `build_vlm_semantic_condition(provider, video, instructions, n_previous) -> tuple[Tensor, Tensor]` for both K1 and K4.

- [ ] **Step 1: Write failing K4 geometry test**

```python
class RecordingK4PlannerProvider(RecordingPlannerProvider):
    num_keyframes = 4
    target_tokens_per_keyframe = 256

    def predict(self, images, instructions):
        self.images = images.clone()
        self.instructions = list(instructions)
        return type("Plan", (), {
            "semantic_tokens": torch.zeros(2, 2, 4, 256, 1024),
            "times": torch.tensor([0.25, 0.5, 0.75, 1.0]).repeat(4, 1),
        })()


def test_build_vlm_semantic_condition_accepts_metadata_driven_k4() -> None:
    video = torch.zeros(2, 3, 2, 13, 2, 2)
    tokens, times = build_vlm_semantic_condition(
        RecordingK4PlannerProvider(), video, ["pick", "place"], n_previous=4
    )
    assert tokens.shape == (2, 2, 4, 256, 1024)
    assert times.shape == (4, 4)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `env PYTHONPATH=. pytest -q tests/test_ge_act_semantic_training_contract.py -k metadata_driven_k4`

Expected: FAIL because the function hardcodes `(B,2,1,256,1024)`.

- [ ] **Step 3: Replace K1 constants with provider geometry**

```python
num_keyframes = int(provider.num_keyframes)
tokens_per_keyframe = int(provider.target_tokens_per_keyframe)
expected_tokens = (
    video.shape[0], 2, num_keyframes, tokens_per_keyframe, 1024
)
expected_times = (video.shape[0] * 2, num_keyframes)
```

Keep the existing current-observation-only assertion and error messages, but include the dynamic expected shapes.

- [ ] **Step 4: Run K1 and K4 provider tests**

Run: `env PYTHONPATH=. pytest -q tests/test_ge_act_semantic_training_contract.py tests/test_ge_act_vlm_semantic_planner.py`

Expected: PASS, including the existing K1 test.

- [ ] **Step 5: Commit**

```bash
git add ge_act/runner/ge_trainer.py tests/test_ge_act_semantic_training_contract.py
git commit -m "fix(ge-act): make VLM conditioning geometry metadata-driven"
```

---

### Task 2: One-pass planner predictions and alignment losses

**Files:**
- Modify: `qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`
- Modify: `ge_act/models/ltx_models/vlm_semantic_planner.py`
- Test: `tests/test_ge_act_dual_camera_planner.py`
- Test: `tests/test_ge_act_vlm_semantic_planner.py`

**Interfaces:**
- Produces: `PlannerWrapper.predict_dino_depth_plan_with_losses(...) -> tuple[Tensor, Tensor, dict[str, Tensor]]`.
- Produces: `FrozenDualCameraVLMPlanner.prepare_inputs(current_images, instructions) -> dict[str, Tensor]` without running Qwen.
- Preserves: frozen `predict(...)` behavior and detached outputs.

- [ ] **Step 1: Write failing one-pass API test**

Use a tiny `PlannerWrapper` fixture whose `predict_dino_depth_plan` increments a counter, then assert:

```python
semantic, depth, losses = wrapper.predict_dino_depth_plan_with_losses(
    semantic_plan_labels=semantic_target,
    depth_plan_labels=depth_target,
    input_ids=input_ids,
)
assert calls == 1
assert semantic.shape == semantic_target.shape
assert depth.shape == (1, 2, 1024, 4, depth_dim)
assert torch.isfinite(losses["loss"])
```

- [ ] **Step 2: Run the test and verify RED**

Run: `env PYTHONPATH=. pytest -q tests/test_ge_act_dual_camera_planner.py -k with_losses`

Expected: FAIL with missing `predict_dino_depth_plan_with_losses`.

- [ ] **Step 3: Add the one-pass method and reuse it from `forward`**

```python
def predict_dino_depth_plan_with_losses(
    self,
    semantic_plan_labels: torch.Tensor,
    depth_plan_labels: torch.Tensor,
    **inputs: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    semantic, depth = self.predict_dino_depth_plan(**inputs)
    losses = self.compute_plan_losses(
        semantic,
        semantic_plan_labels.to(device=semantic.device, dtype=torch.float32),
    )
    depth_target = self._reshape_depth_target(
        depth_plan_labels.to(device=depth.device, dtype=torch.float32)
    )
    if self.da3_align_strategy == "wsa_multilayer":
        depth_loss, depth_cos, depth_lnmse = self._wsa_layer_loss(depth, depth_target)
        losses["depth_cos"] = depth_cos.detach()
        losses["depth_lnmse"] = depth_lnmse.detach()
    else:
        depth_loss = F.smooth_l1_loss(depth, depth_target)
    losses["loss"] = losses["loss"] + self.depth_loss_weight * depth_loss
    losses["depth_smooth_l1"] = depth_loss.detach()
    return semantic, depth, losses
```

Refactor the non-current-alignment `lingbot_dino` branch of `forward` to call this method and return its loss dictionary.

- [ ] **Step 4: Separate provider preprocessing from frozen prediction**

Add:

```python
def prepare_inputs(self, current_images, instructions):
    image_pairs = normalized_bvchw_to_pil_pairs(current_images)
    return self.input_mover(self.input_builder(
        self.processor,
        image_pairs,
        [str(value) for value in instructions],
        self.plan_tokens,
    ))
```

Keep `predict` decorated with `@torch.no_grad()` and make it call `prepare_inputs`; do not remove its final detach.

- [ ] **Step 5: Run planner/provider regressions**

Run: `env PYTHONPATH=. pytest -q tests/test_ge_act_dual_camera_planner.py tests/test_ge_act_vlm_semantic_planner.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py ge_act/models/ltx_models/vlm_semantic_planner.py tests/test_ge_act_dual_camera_planner.py tests/test_ge_act_vlm_semantic_planner.py
git commit -m "feat(planner): return K4 predictions with alignment losses"
```

---

### Task 3: Composite joint model and four optimizer groups

**Files:**
- Create: `ge_act/models/ltx_models/joint_vlm_geact.py`
- Modify: `ge_act/runner/ge_trainer.py`
- Create: `tests/test_joint_vlm_geact_training.py`

**Interfaces:**
- Produces: `JointVLMGEActModel(planner, ltx, num_keyframes=4, tokens_per_keyframe=256)`.
- Produces: `JointVLMGEActModel.forward(planner_inputs, semantic_labels, depth_labels, ltx_inputs) -> JointVLMGEActOutput`.
- Produces: `build_joint_optimizer_parameter_groups(model, ltx_lr, semantic_lr, qwen_lr, planner_head_lr) -> list[dict]`.

- [ ] **Step 1: Write failing gradient-path and optimizer-group tests**

```python
def test_video_loss_reaches_qwen_and_ltx_semantic_parameters():
    joint = make_tiny_joint_model()
    out = joint(
        planner_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
        semantic_labels=torch.zeros(1, 2, 4 * 2, 4),
        depth_labels=torch.zeros(1, 2, 4, 4 * 2, 3),
        ltx_inputs=tiny_ltx_inputs(),
    )
    out.ltx_predictions["video"].square().mean().backward()
    assert finite_nonzero(joint.planner.model.proj.weight.grad)
    assert finite_nonzero(joint.ltx.semantic_attn.weight.grad)


def test_joint_optimizer_groups_are_disjoint_and_complete():
    groups = build_joint_optimizer_parameter_groups(
        make_tiny_joint_model(),
        ltx_lr=2e-5, semantic_lr=1e-4, qwen_lr=1e-6,
        planner_head_lr=3e-5,
    )
    assert [group["name"] for group in groups] == [
        "base_ltx", "semantic_ltx", "qwen", "planner_heads"
    ]
    assert [group["lr"] for group in groups] == [2e-5, 1e-4, 1e-6, 3e-5]
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in joint.parameters() if p.requires_grad}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `env PYTHONPATH=.:ge_act pytest -q tests/test_joint_vlm_geact_training.py`

Expected: collection fails because `joint_vlm_geact.py` does not exist.

- [ ] **Step 3: Implement the composite forward**

```python
@dataclass
class JointVLMGEActOutput:
    ltx_predictions: dict[str, torch.Tensor]
    semantic_plan: torch.Tensor
    depth_plan: torch.Tensor
    planner_losses: dict[str, torch.Tensor]


class JointVLMGEActModel(nn.Module):
    def __init__(self, planner, ltx, *, num_keyframes=4, tokens_per_keyframe=256):
        super().__init__()
        self.planner = planner
        self.ltx = ltx
        self.num_keyframes = int(num_keyframes)
        self.tokens_per_keyframe = int(tokens_per_keyframe)

    def forward(self, *, planner_inputs, semantic_labels, depth_labels, ltx_inputs):
        semantic, depth, planner_losses = (
            self.planner.predict_dino_depth_plan_with_losses(
                semantic_plan_labels=semantic_labels,
                depth_plan_labels=depth_labels,
                **planner_inputs,
            )
        )
        batch = semantic.shape[0]
        semantic = semantic.reshape(
            batch, 2, self.num_keyframes, self.tokens_per_keyframe, 1024
        )
        ltx_predictions = forward_pass(
            model=self.ltx,
            semantic_plan=semantic,
            **ltx_inputs,
        )["latents"]
        return JointVLMGEActOutput(
            ltx_predictions=ltx_predictions,
            semantic_plan=semantic,
            depth_plan=depth,
            planner_losses=planner_losses,
        )
```

Validate all K4 shapes before calling LTX; never detach `semantic`.

- [ ] **Step 4: Implement explicit parameter classification**

Classify `joint.ltx` names containing `semantic_` into `semantic_ltx`; classify planner modules `plan_head`, `depth_head`, `plan_embedding_injector`, and pooled query banks into `planner_heads`; put all remaining `joint.planner.model` parameters in `qwen`. Reject duplicate or missing trainable parameter IDs.

- [ ] **Step 5: Run focused and semantic tests**

Run: `env PYTHONPATH=.:ge_act pytest -q tests/test_joint_vlm_geact_training.py tests/test_ge_act_semantic_training_contract.py`

Expected: all tests pass and the video-only gradient assertion is non-zero.

- [ ] **Step 6: Commit**

```bash
git add ge_act/models/ltx_models/joint_vlm_geact.py ge_act/runner/ge_trainer.py tests/test_joint_vlm_geact_training.py
git commit -m "feat(ge-act): add joint VLM LTX training module"
```

---

### Task 4: Online K4 teachers and joint Trainer integration

**Files:**
- Modify: `ge_act/runner/ge_trainer.py`
- Modify: `ge_act/main.py`
- Modify: `tests/test_joint_vlm_geact_training.py`

**Interfaces:**
- Produces: `select_joint_planner_frames(video, n_previous, offsets) -> tuple[current, future]`.
- Produces: a `Trainer` joint mode that prepares only `self.joint_model` with DeepSpeed.
- Produces: `Trainer(config_file, config_overrides=None)` so smoke-only overrides are applied before Accelerate/DeepSpeed initialization.
- Consumes: `encode_dual_camera_future_targets(...)` and frozen `Siglip2TargetEncoder`/`DepthAnything3TargetEncoder`.

- [ ] **Step 1: Write failing frame-order and teacher-freeze tests**

```python
def test_joint_teacher_frames_match_planner_training_offsets():
    video = torch.arange(1 * 3 * 2 * 13 * 2 * 2).reshape(1, 3, 2, 13, 2, 2)
    current, future = select_joint_planner_frames(
        video, n_previous=4, offsets=(2, 4, 6, 8)
    )
    torch.testing.assert_close(current, video[:, :, :, 3].permute(0, 2, 3, 4, 1))
    for keyframe, source_index in enumerate((6, 8, 10, 12)):
        torch.testing.assert_close(
            future[:, :, keyframe],
            video[:, :, :, source_index].permute(0, 2, 3, 4, 1),
        )


def test_joint_teacher_parameters_are_frozen_and_excluded():
    trainer = make_tiny_joint_trainer()
    trainer.prepare_trainable_parameters()
    assert all(not p.requires_grad for p in trainer.semantic_teacher.parameters())
    assert all(not p.requires_grad for p in trainer.depth_teacher.parameters())
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `env PYTHONPATH=.:ge_act pytest -q tests/test_joint_vlm_geact_training.py -k 'teacher_frames or teacher_parameters'`

Expected: FAIL with missing joint frame selector/teacher fields.

- [ ] **Step 3: Load trainable planner and frozen teachers in joint mode**

When `joint_training.enabled=true`:

```python
self.semantic_planner = FrozenDualCameraVLMPlanner.from_checkpoint(...)
self.semantic_planner.wrapper.requires_grad_(True)
self.semantic_planner.wrapper.train()
self.semantic_teacher = Siglip2TargetEncoder(...)
self.depth_teacher = DepthAnything3TargetEncoder(
    align_strategy="wsa_multilayer",
    teacher_layers=(11, 15, 19, 23),
    layer_weights=(1.0, 1.2, 1.4, 1.6),
    ...,
)
self.joint_model = JointVLMGEActModel(
    self.semantic_planner.wrapper,
    self.diffusion_model,
    num_keyframes=4,
    tokens_per_keyframe=256,
)
```

Enable Qwen and LTX gradient checkpointing; do not pass the planner wrapper to `freeze_conditioning_modules` in joint mode.

- [ ] **Step 4: Integrate targets and combined loss into the train step**

Build planner inputs outside the neural forward, encode online targets under `torch.no_grad()`, and call `self.joint_model(...)` once. Compute:

```python
planner_loss = joint_output.planner_losses["loss"]
loss = loss_video + float(self.args.joint_training["planner_loss_weight"]) * planner_loss
```

Use semantic dropout only in `ltx_inputs["semantic_condition_mask"]`. Log `loss_video`, `planner_loss`, semantic MSE, depth WSA loss, both gradient norms, all group LRs, and peak memory.

- [ ] **Step 5: Prepare one composite model and clip all gradients**

```python
self.joint_model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
    self.state.accelerator.prepare(
        self.joint_model,
        self.optimizer,
        self.train_dataloader,
        self.lr_scheduler,
    )
)
```

Use `accelerator.accumulate(self.joint_model)` and clip `self.joint_model.parameters()`.

- [ ] **Step 6: Add joint checkpoint export and exact resume state**

At keeper steps, unwrap the composite, save `model.ltx.save_pretrained(step_dir / "ltx")`, export planner model/head/processor under `step_dir / "planner"`, write `joint_meta.json` and `trainer_state.json`, then call `accelerator.save_state(step_dir / "training_state")` on every rank. After `accelerator.prepare`, an explicit `--resume_from_checkpoint` loads distributed state and restores global step, epoch, and prepared-dataloader position after validating world size, per-device batch size, accumulation, dataset length, sampler seed, and batches per epoch. In joint mode, use an epoch-seeded sampler that emits `(sample_index, epoch)`; derive each sample's Python and NumPy RNGs from `(sampler_seed, epoch, sample_index)` so shuffled order, random frames, memory-frame selection, retries, and crops reproduce the same remaining suffix regardless of worker scheduling. Give the DataLoader a dedicated generator so reconstructing an iterator does not consume the restored global CPU Torch RNG used by subsequent training stochasticity. Preserve the existing LTX-only save path when joint mode is disabled.

- [ ] **Step 7: Apply smoke overrides before distributed initialization**

Add `--batch_size_override`, `--gradient_accumulation_steps_override`, and `--disable_deepspeed` to `ge_act/main.py`. Build an override dictionary before constructing `Trainer`:

```python
config_overrides = {}
if args.batch_size_override is not None:
    config_overrides["batch_size"] = args.batch_size_override
if args.gradient_accumulation_steps_override is not None:
    config_overrides["gradient_accumulation_steps"] = (
        args.gradient_accumulation_steps_override
    )
if args.disable_deepspeed:
    config_overrides["use_deepspeed"] = False
runner = Runner(args.config_file, config_overrides=config_overrides)
```

Apply `config_overrides` to the loaded YAML dictionary before constructing `args` and before calling `_init_distributed`. Add a test that constructs a tiny `Trainer` subclass and observes the overridden values during `_init_distributed`.

- [ ] **Step 8: Run trainer and regression tests**

Run: `env PYTHONPATH=.:ge_act pytest -q tests/test_joint_vlm_geact_training.py tests/test_ge_act_semantic_training_contract.py tests/test_ge_act_vlm_semantic_planner.py tests/test_ge_act_dual_camera_planner.py`

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add ge_act/runner/ge_trainer.py ge_act/main.py tests/test_joint_vlm_geact_training.py
git commit -m "feat(ge-act): train VLM and LTX in one distributed step"
```

---

### Task 5: Joint config, preflight, and OLA launcher

**Files:**
- Create: `ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml`
- Create: `ge_act/scripts/train_joint_vlm_geact_ola.sh`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py`
- Modify: `tests/test_ge_act_siglip2_config.py`

**Interfaces:**
- Produces: a standalone joint config; existing frozen VLM config remains unchanged.
- Produces: preflight errors for invalid K4, LR, batch, teacher, and checkpoint contracts.

- [ ] **Step 1: Write failing config contract test**

```python
def test_joint_vlm_geact_config_matches_approved_recipe():
    config = yaml.safe_load(JOINT_CONFIG_PATH.read_text())
    assert config["batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 16
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
    assert config["joint_training"] == {
        **config["joint_training"],
        "enabled": True,
        "planner_loss_weight": 0.1,
        "qwen_lr": 1e-6,
        "planner_head_lr": 3e-5,
        "qwen_gradient_checkpointing": True,
    }
    assert config["lr"] == 2e-5
    assert config["semantic_lr"] == 1e-4
    assert config["semantic_plan"]["keyframe_indices"] == [2, 4, 6, 8]
    assert config["diffusion_model"]["config"]["semantic_plan_num_keyframes"] == 4
```

- [ ] **Step 2: Run the config test and verify RED**

Run: `env PYTHONPATH=.:ge_act pytest -q tests/test_ge_act_siglip2_config.py -k joint_vlm_geact`

Expected: FAIL because the config does not exist.

- [ ] **Step 3: Add the exact formal YAML**

Use the verified OLA predecoded RGB cache with `CustomLeRobotDataset`, ordered
main/wrist cameras, and `require_predecoded=true`; OLA has no compatible FastWAM
HDF5 manifest. Then set:

```yaml
batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
lr: 2.0e-5
semantic_lr: 1.0e-4
lr_warmup_steps: 1000
semantic_plan:
  enabled: true
  source: vlm_planner
  keyframe_indices: [2, 4, 6, 8]
  tokens_per_frame: 256
  feature_dim: 1024
joint_training:
  enabled: true
  planner_loss_weight: 0.1
  qwen_lr: 1.0e-6
  planner_head_lr: 3.0e-5
  qwen_gradient_checkpointing: true
  siglip2_model_dir: /data/users/junjie/vlm4wam_2b/weights/siglip2-large-patch16-256
  da3_ckpt_dir: /data/users/junjie/vlm4wam_2b/weights/DA3-LARGE-1.1
  da3_code_root: /data/users/junjie/vlm4wam_2b/code/Depth-Anything-3
diffusion_model:
  model_path: /data/users/junjie/Genie-Envisioner-V1/weights/ltx_step_50000
  config:
    semantic_plan_num_keyframes: 4
```

Set the planner checkpoint to the approved OLA dual-camera K4 step30k path and keep all 28 semantic cross-attention blocks.

- [ ] **Step 4: Extend static/runtime preflight**

Require K4 offsets, WSA metadata, two views, width 1024, batch contract 128, all four LRs, both checkpointing flags, and existing paths. Keep the old K1 frozen-planner validation branch unchanged.

- [ ] **Step 5: Add the OLA launcher**

The launcher exports offline HF settings, constrains CPU math threads, verifies all
predecoded caches, runs preflight, and launches `torchrun`. `RUN_KIND=smoke` selects
one GPU and passes `--max_train_steps 1 --batch_size_override 1
--gradient_accumulation_steps_override 1 --disable_deepspeed
--enable_8bit_optimizer` so the full 4.67B trainable parameters fit for the
functional smoke; `RUN_KIND=smoke8`
keeps eight GPUs and ZeRO-2 enabled but bounds the run to ten optimizer steps with
per-GPU batch and accumulation both set to one. Neither mode edits the formal YAML.

- [ ] **Step 6: Run config/preflight tests**

Run: `env PYTHONPATH=.:ge_act pytest -q tests/test_ge_act_siglip2_config.py tests/test_ge_act_source_completeness.py`

Expected: all tests pass and existing configs remain accepted.

- [ ] **Step 7: Commit**

```bash
git add ge_act/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml ge_act/scripts/train_joint_vlm_geact_ola.sh ge_act/scripts/preflight_ltx_siglip2.py tests/test_ge_act_siglip2_config.py
git commit -m "feat(joint): configure K4 VLM GE-Act training on OLA"
```

---

### Task 6: Full verification and OLA smoke gates

**Files:**
- Runtime artifacts only under ignored output/log directories.

**Interfaces:**
- Consumes: committed joint source/config and OLA checkpoints/data.
- Produces: one-GPU and 8-GPU smoke evidence; does not start the 30k formal run without a separate user request.

- [ ] **Step 1: Run the complete affected test suite locally**

Run:

```bash
env PYTHONPATH=.:ge_act pytest -q \
  tests/test_joint_vlm_geact_training.py \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_ge_act_dual_camera_planner.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_siglip2_config.py
```

Expected: zero failures.

- [ ] **Step 2: Compile and check the source diff**

Run:

```bash
python -m py_compile \
  ge_act/models/ltx_models/joint_vlm_geact.py \
  ge_act/models/ltx_models/vlm_semantic_planner.py \
  ge_act/runner/ge_trainer.py
git diff --check
```

Expected: exit 0 and no whitespace errors.

- [ ] **Step 3: Sync only committed source to OLA and run preflight**

Push the branch and fast-forward the clean OLA checkout. Run the joint launcher preflight against the real K4 metadata, LTX step50k, all verified predecoded RGB caches, SigLIP2, and DA3 paths.

- [ ] **Step 4: Run one-GPU one-step smoke**

Run: `RUN_KIND=smoke NPROC_PER_NODE=1 bash ge_act/scripts/train_joint_vlm_geact_ola.sh`

Expected: finite total/video/planner losses, finite non-zero Qwen and LTX semantic gradient norms, frozen-teacher gradient count zero, and a saved reloadable joint checkpoint.

- [ ] **Step 5: Run 8-GPU ten-step smoke**

Run: `RUN_KIND=smoke8 bash ge_act/scripts/train_joint_vlm_geact_ola.sh`.
Verify every rank reaches step 10, DeepSpeed reports one composite engine, loss
remains finite, and peak memory stays below 80 GB/GPU.

- [ ] **Step 6: Verify checkpoint round trip**

Reload `ltx/`, `planner/`, and `training_state/`; compare fixed-input semantic and LTX outputs before/after reload with bf16 tolerances `rtol=2e-2, atol=2e-2` and confirm optimizer step/scheduler state is 10.

- [ ] **Step 7: Final source-only commit if smoke fixes were required**

Run the affected tests again, commit only source/config/test changes, and leave all generated checkpoints/logs untracked and ignored.
