# Research Log: Does a Qwen3.5 Semantic Planner Help GE-Act LTX?

Started 2026-09-28. Loop: summarize context -> state hypotheses with falsifiable
predictions -> run the cheapest decisive experiment -> record results ->
update hypotheses. Code is the ground truth; the paper text is revised to match
what the experiments show.

## 0. Question

Under which conditions does guidance predicted by a Qwen3.5-2B planner
(future SigLIP2 feature maps) improve GE-Act LTX (video + action) on LIBERO and
LIBERO-Plus, and why has it not reliably done so far?

## 1. Context (verified facts, 2026-09-28)

| # | Fact | Evidence |
|---|---|---|
| C1 | Qwen3-VL-2B joint run started LTX from plain GE-Act: `ltx_step_50000` has 0 `semantic_*` keys; semantic modules were zero-initialized at the joint stage and never saw teacher features. | `vlm4wam_joint_assets/ltx_step_50000` key scan |
| C2 | Joint planner never converged: `planner_semantic_mse` 1.81 -> 1.29 over 50k steps, still falling; `lm_plan_ce` 9-15. | `joint_vlm_geact_action_k4_50k` rank-0 log |
| C3 | After joint training, semantic cross-attn output projection Frobenius norm is 39-48 vs 92-187 for text cross-attn (1/4-1/2). | `step_40000/ltx` weights, blocks 0/7/14/21/27 |
| C4 | Qwen3.5 baton planner (`qwen35_baton_strict_acd1_18_ddp_30k`) stopped at 22.6k/30k; last ckpt step_020000. MSE 5.13 -> 1.73 (main ~1.3, wrist ~2.2). Per-keyframe MSE is flat across keyframes 0..3. | `training_metrics.jsonl` |
| C5 | Keyframe mismatch in the Qwen3-VL line: `ltx_siglip2` used [0,3,5,8]; planner/joint used [2,4,6,8]. The baton line uses [0,3,5,8] everywhere. | configs |
| C6 | Baton three-stage pipeline (Stage 2 teacher LTX, Stage 3 frozen-planner LTX) is implemented on `qwen35-video-hindsight-grounding` but was never run. | HPC3 outputs |
| C7 | Unified env `qwen35` (transformers 5.14.1, diffusers 0.35.1, deepspeed 0.16.4) runs both Qwen3.5 and GE-Act LTX; preflights pass. | HPC3 preflight 2026-09-28 |
| C8 | Planner sees only the current frame (no memory); LTX sees 4 memory frames. LIBERO training used all episodes (no held-out split). | dataset / trainer code |
| C9 | The 98.7 LIBERO result came from a standalone GE-Act WAM; its exact checkpoint and eval log are not yet located. | user, 2026-09-28 |

## 2. Hypotheses

| ID | Hypothesis | Prediction if true | Kill criterion |
|---|---|---|---|
| H1 | Oracle future semantics help LTX. | Stage 2 val: `teacher` beats `semantic_disabled` on video loss and action MSE. | No gap after 10k steps -> guidance design, not the planner, is the bottleneck (go to H4). |
| H2 | The planner ignores the current observation and outputs a near-mean plan. | Planner MSE >= copy-current-frame MSE on keyframe 0, and the gap is flat across keyframes. | Planner beats copy baseline clearly on all keyframes. **Refuted by E2.** |
| H3 | Exposure bias: LTX trained on teacher features degrades with predicted features. | Stage-2 ckpt val with predicted features falls between teacher and disabled; Stage 3 closes the gap. | Predicted ~= teacher already. |
| H4 | Guidance is under-weighted relative to text. | Semantic out-proj/gate norms stay << text; raising inference CFG on guidance improves action MSE. | Norms comparable, CFG sweep flat. |
| H5 | A residual planner `F = SigLIP2(current) + delta` (zero-init delta) beats the absolute planner. | Lower MSE than baton step_20000 on every keyframe, and never worse than copy baseline. | No MSE gain after 5k steps. **Deprioritized: E2 shows the copy baseline is far worse than the planner.** |
| H5b | Feeding the current SigLIP2 grid as extra tower context (no skip) improves the absolute planner. | Lower MSE than the same-budget control at every keyframe. | No gain after 5k steps. |
| H6 | DA3 (WSA 4-layer) auxiliary loss improves the SigLIP2 prediction. | Lower SigLIP2 MSE at equal steps vs lambda_spa=0. | No gain or worse. |
| H7 | LIBERO is saturated; gains show on LIBERO-Plus. | Larger delta on LIBERO-Plus (Layout/Language) than on LIBERO. | - |
| H8 | Wrist-view guidance is mostly noise (R^2 ~0.23 vs main ~0.53) and can hurt LTX. | Masking wrist guidance at validation does not raise, or lowers, video/action loss. | Wrist masking clearly hurts. |

## 3. Experiments

| ID | Tests | Setup | Cost | Status |
|---|---|---|---|---|
| E1 | H1, H4, H8 | Stage 2: GE base + online SigLIP2 teacher at [0,3,5,8], 20k steps, val modes teacher / semantic_disabled; log semantic vs text branch norms. | 8 GPU x ~2-3 days | resubmitted as 658617 (first attempt failed: Slurm spool path broke GE_ACT_ROOT) |
| E2 | H2 | Baton step_020000 on LIBERO windows: per-keyframe planner MSE vs copy-current-frame MSE vs dataset-mean MSE, plus instruction swap. | 1 GPU x 15 min | **done** (job 658491) |
| E3 | H3 | Stage-2 ckpt (E1) val with teacher / predicted (E2 planner) / disabled. | 1 GPU x ~2 h | after E1 10k |
| E4 | H5b, H6 | Warm start from step_020000, 5k steps, gbs 128 (2/GPU x 8 accum x 8 GPU), lr 1e-5 backbone / 1e-4 heads, gradient checkpointing: current {none, context} x lambda_spa {0, 0.1}. A = none/0 is the same-budget control. | 4 runs x 8 GPU x ~8 h | resubmitted as 658621-658624 after OOM at 4/GPU; 2-GPU smoke peaks at 49.4 GiB |
| E5 | H3 | Stage 3: E1 ckpt + frozen best planner (E4), 30k steps. | 8 GPU x ~3 days | after E1, E4 |
| E6 | H7 + final | LIBERO 4x500 and LIBERO-Plus, sharded over HPC3 GPUs: GE-Act base, E1 disabled, E5. | ~0.5 day | after E5 |

## 4. Results

(append newest first)

### 2026-09-28 E2: planner vs trivial baselines (400 LIBERO windows, in-distribution)

Teacher-space MSE per token (SigLIP2 penultimate, 1024-d). R^2 = 1 - planner / dataset-mean.

| cam | k | planner | copy current | dataset mean | R^2 | swap-instr shift | swapped vs truth |
|---|---|---|---|---|---|---|---|
| main | 0 | 1.268 | 1.622 | 2.758 | 0.54 | 0.483 | 1.714 |
| main | 1 | 1.295 | 2.320 | 2.760 | 0.53 | 0.476 | 1.768 |
| main | 2 | 1.299 | 2.543 | 2.794 | 0.53 | 0.522 | 1.806 |
| main | 3 | 1.302 | 2.770 | 2.775 | 0.53 | 0.606 | 1.910 |
| wrist | 0 | 2.253 | 3.723 | 2.940 | 0.23 | 0.467 | 2.691 |
| wrist | 1 | 2.224 | 4.802 | 2.914 | 0.24 | 0.493 | 2.709 |
| wrist | 2 | 2.176 | 4.876 | 2.837 | 0.23 | 0.563 | 2.724 |
| wrist | 3 | 2.216 | 5.085 | 2.897 | 0.24 | 0.589 | 2.816 |

Findings:
- H2 refuted: the planner beats copy-current on every keyframe, so it uses the
  observation. The flat per-keyframe MSE comes from SigLIP2 features changing
  a lot even one keyframe ahead (copy MSE 1.62 at k0, grows to 2.77 by k3).
- Wrist copy is worse than the dataset mean; wrist prediction explains only
  ~23% of variance vs ~53% for main -> new H8.
- The prediction is instruction-conditioned: a same-suite instruction swap
  moves it by 0.48-0.61 and raises error vs the true future by 35% (main) /
  20% (wrist). Usable for rebuttal item #2.
- The residual design (H5) starts from the much worse copy baseline, so E4
  tests current-frame context without the skip instead (H5b).

## 5. Decisions

- 2026-09-28: planner frozen after Stage 1; LTX consumes SigLIP2 only; DA3 is
  a planner-side auxiliary target (WSA 4-layer). Paper to be aligned later.
