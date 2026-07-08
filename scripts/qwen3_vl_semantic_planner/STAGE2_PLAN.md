# Stage-2: Planner (Qwen3-VL) ⊗ Cosmos WM joint training — preparation

**Goal.** Close the train/infer loop. Today the Cosmos world model (WM) is trained on the
**GT SigLIP semantic plan** (cosine≈1 quality, online-encoded from the actual future frames),
but at inference it is fed the **planner's predicted plan** (cosine≈0.55, noisier). Stage-2
makes the WM consume the planner's predicted plan during training — and, at the top tier,
lets the WM's diffusion loss flow back into the planner (DIAL-style e2e).

This mirrors DIAL's `golden → end2end` switch: swap the downstream conditioning from
ground-truth features to the VLM's *predicted* features, resume from the decoupled checkpoints.

---

## 0. Validated facts (from a 3-codebase investigation, 2026-07-07)

**Planner side** (`train_qwen3vl_semantic_planner.py`):
- Drop-in output = `PlannerWrapper.predict_semantic_plan(**inputs) → [B, 3645, 1152]`
  (5 keyframes × 729 native-grid × 1152), **fully differentiable** end-to-end (Qwen3-VL last
  hidden → `<|sem_plan_i|>` gather → `CoVTLatentDecoderHead`). Raw SigLIP2 penultimate feature
  space — **no target normalization** — same space the WM consumes.
- Inputs are built exactly like `Collator`: user turn = image + instruction, assistant turn =
  the 20 distinct `<|sem_plan_i|>` tokens (teacher-forced). Wrapped in `planner_plan_provider.py`.
- Checkpoint (job 392562, uniform/online/16000, HPC3-only) layout: `qwen3vl_lora_or_model/`,
  `plan_head.pt`, `plan_token_embedding.pt`, `planner_meta.json`.

**WM side** (`third_party/cosmos-predict2.5`):
- Injection point = `data_batch["semantic_plan"]`, read in
  `models/video2world_model_rectified_flow.py:106` (`get_data_and_condition`). Shape `[B,3645,1152]`.
- **Fully differentiable** from that tensor to the flow-matching loss — no VAE/tokenizer/detach
  on the plan branch. So e2e gradient into the planner is architecturally possible.
- The pod WM run is **ONLINE mode** (`SEMANTIC_PLAN_ONLINE=1`, `EPISODE_SAMPLING=1`): it samples
  a random 49-frame window per episode per epoch and SigLIP-encodes the GT plan on the fly
  (`OnlineSemanticPlanEncoder`, `@torch.no_grad`, runs only when `semantic_plan is None`). There
  are **no offline plan .pt files**.
- Raw instruction text is available as `data_batch["ai_caption"]`; first frame = `raw_state[:,:,0]`.
- FSDP (shard 8); the semantic-plan adapter is its own FSDP unit; plan cross-attn is a
  zero-init gated residual. Config env: `SEMANTIC_PLAN_SPATIAL_GRID=0`, `NUM_KEYFRAMES=5`,
  `SOURCE_NUM_KEYFRAMES=5`, `SEMANTIC_PLAN_DROPOUT_PROB=0.15`.

**DIAL recipe** (`/data/LFT-W02_data/junjie/VLA_WM/DIAL`):
- `golden→end2end` = route GT-future features vs VLM predicted bridge tokens into the downstream.
- One `nn.Module`, one AdamW, DDP; freeze vision encoders, tune LLM + bridge-token rows + downstream.
- Gradient throttle `x*α + x.detach()*(1-α)` on the VLM output (constant α, default off; α≈0.1 if unstable).
- Loss = downstream loss + **decaying** bridge feature-matching regularizer (weight 1.0→0.1),
  `(action + w·bridge)/(1+w)`. lr 1e-4 cosine, bf16, 8×H100, stage-2 resumes stage-1, 80k steps.

**Environment (verified GO, low risk):**
- cosmos hard-pins `transformers==4.51.3` (`packages/cosmos-oss/pyproject.toml:74`, `uv.lock`),
  which lacks `Qwen3VLForConditionalGeneration`.
- The semantic_plan WM path only uses stable high-level APIs (`AutoModel` for SigLIP2,
  `T5EncoderModel`). Upgrading to transformers 4.57 is **low risk**. The only fragile code
  (reason1 vendored Qwen2.5-VL, `_flash_attention_forward` deep imports) is **off-path** for
  semantic_plan + an external Qwen3-VL planner.
- **Solution: one unified venv** `.venv-qwen3` = copy of the cosmos venv + `transformers==4.57.6`.
  Build script `build_venv_qwen3.sh`; smoke tests included. (uv/tuna mirror both have 4.57.)
  ⚠ Caveat: never `uv sync` this env — it would revert transformers to the 4.51.3 lock. Install
  with `uv pip install` (pip interface), not `uv sync`.

---

## 1. The integration hook (shared by Tier 1b and Tier 2)

Both tiers plug a `PlannerPlanProvider` into the WM's online-encoder slot. In
`get_data_and_condition` (`video2world_model_rectified_flow.py`, ~line 114-121), replace the
GT online-encode branch when a planner is configured:

```python
planner = self._semantic_plan_planner()          # PlannerPlanProvider | None (env: SEMANTIC_PLAN_PLANNER_CKPT)
if planner is not None and semantic_plan is None and self.training \
        and raw_state.ndim == 5 and raw_state.shape[2] > 1:
    first_frames = raw_state[:, :, 0]             # [B,C,H,W]; convert to PIL at planner resolution
    prompts = data_batch["ai_caption"]            # list[str] instructions
    images  = _to_pil(first_frames)               # de-normalize raw_state → uint8 PIL
    if self.config.planner_trainable:             # Tier-2
        plan = planner.predict(images, prompts)               # [B,3645,1152], grad ON
        a = self.config.planner_grad_alpha
        plan = plan * a + plan.detach() * (1.0 - a)           # DIAL throttle
        self._last_planner_plan = plan            # stash for the regularizer loss
    else:                                          # Tier-1b (frozen)
        plan = planner.predict_frozen(images, prompts)        # no_grad
    semantic_plan = plan
    semantic_plan_times = None                     # uniform keyframe times filled by adapter
```

- **Tier-1b** (`planner_trainable=False`): planner frozen, WM adapts to the predicted-plan
  distribution. No planner optimizer, low extra memory. This is the recommended first run now
  that the env is GO — it needs no WM dataset rework and shares the code path with Tier-2.
- **Tier-2** (`planner_trainable=True`): gradient flows planner←WM through the throttle. Add the
  **decaying regularizer** in the loss: keep the planner's SigLIP MSE/cosine vs the GT plan so it
  can't drift off-manifold. That needs the GT plan too — run the original `OnlineSemanticPlanEncoder`
  in parallel to get `gt_plan`, then add `w * plan_provider.plan_loss(self._last_planner_plan, gt_plan)`
  with `w` linearly 1.0→0.1. (Reuse `PlannerWrapper.compute_plan_losses`.)

`_to_pil` must invert the WM's frame normalization — confirm whether `raw_state` is `[-1,1]`,
`[0,1]`, or `[0,255]` before conversion (open item O1).

---

## 2. Memory plan (two 2B models, one process)

Two full-FT 2B models + FSDP WM + VAE + SigLIP will not fit at full-FT for both. Options,
in increasing capability:

| Tier | Planner state | WM state | Notes |
|---|---|---|---|
| 1b | **frozen** (bf16, no optim) | full-FT (FSDP) | cheapest; only a Qwen3-VL forward per step |
| 2-lite | **LoRA** (CoVT head + LoRA + bridge rows) | full-FT (FSDP) | small planner optimizer; DIAL-compatible (CoVT already supports LoRA) |
| 2-full | full-FT (FSDP) | full-FT (FSDP) | needs planner also FSDP-sharded + grad-ckpt on both; heaviest |

Recommended: **1b → 2-lite**. Enable grad-ckpt on the Qwen3-VL backbone (already used in training),
keep `SEMANTIC_PLAN_DROPOUT_PROB=0` for pure e2e (dropout kills the planner gradient on ~15% of steps),
and zero the planner's `infonce`/`variance` loss terms in-loop (their `[B,L,L]` with L=3645 is the
memory-heavy part) — keep only MSE+cosine for the regularizer.

---

## 3. Run sequence (pod 30282 primary)

**Prereqs (once):**
1. `bash build_venv_qwen3.sh` → `.venv-qwen3` with transformers 4.57 (RUNNING / verify smoke tests).
2. Sync the planner checkpoint (job 392562 `step_016000/`) from HPC3 → pod NAS
   `/root/nas_data/junjie/weight/planner_covt_uniform_16000/`.
3. Smoke-test the provider in `.venv-qwen3`:
   `PYTHONPATH=... python -c "from planner_plan_provider import PlannerPlanProvider; p=PlannerPlanProvider('<ckpt>'); import PIL; ..."`.

**Tier-1b (WM adapts to predicted plans):**
4. Add the §1 hook (guarded by `SEMANTIC_PLAN_PLANNER_CKPT`; inert when unset — safe for the
   current GT-oracle training).
5. Launcher = the uniform WM launcher + `SEMANTIC_PLAN_PLANNER_CKPT=<ckpt>`,
   `SEMANTIC_PLAN_ONLINE=0`-for-GT-but-planner-on, resume from `2b_semplan_uniform_..._6000`.
   Run on `.venv-qwen3`, 8×H100.
6. Eval closed-loop (planner plan → WM) vs the current GT-oracle numbers.

**Tier-2 (joint e2e):** flip `planner_trainable=True`, set α (start 0.1), add the decaying
regularizer, resume from the Tier-1b checkpoint. Same launcher + `PLANNER_TRAINABLE=1 PLANNER_GRAD_ALPHA=0.1`.

**Fallback if the unified env ever breaks (Tier-1a offline):** `gen_predicted_plans.py` dumps
predicted plans for a fixed (stem,start) window manifest in the planner env; point the WM at them
in offline mode. Loses random-window augmentation; only needed if in-process coupling fails.

---

## 4. Open items to verify on the pod

- **O1** `raw_state` normalization for `_to_pil` (planner expects standard RGB PIL).
- **O2** `data_batch["ai_caption"]` text matches the planner's training prompt format (incl. the
  `[TGT]` tag). If the WM captions differ, wrap them into the planner's `_USER_TEMPLATE` verbatim.
- **O3** Qwen3-VL forward at WM batch size (BATCH_SIZE=2/GPU): fits alongside the FSDP WM? Measure.
- **O4** keyframe-times: the planner predicts uniform k5; confirm the adapter's
  `SOURCE_NUM_KEYFRAMES=5`/`NUM_KEYFRAMES=5` path leaves them unsubsampled (it should).
- **O5** provider import under `.venv-qwen3` (transformers 4.57 API drift vs the training code).

## Files
- `planner_plan_provider.py` — keystone loader/predictor (DONE).
- `build_venv_qwen3.sh` (on pod) — unified env build + smoke tests.
- `gen_predicted_plans.py` — Tier-1a offline fallback (TODO if needed).
- WM hook patch — §1, applied to `video2world_model_rectified_flow.py` on the pod (guarded).
