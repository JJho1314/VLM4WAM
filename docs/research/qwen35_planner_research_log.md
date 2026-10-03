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
| H1 | Oracle future semantics help LTX. **Rejected for per-block cross-attention (E3/E5/E5b/E6); supported with additive injection (E7, no text: -6.6% action MSE).** | Stage 2 val: `teacher` beats `semantic_disabled` on video loss and action MSE. | No gap after 10k steps -> guidance design, not the planner, is the bottleneck (go to H4). |
| H2 | The planner ignores the current observation and outputs a near-mean plan. | Planner MSE >= copy-current-frame MSE on keyframe 0, and the gap is flat across keyframes. | Planner beats copy baseline clearly on all keyframes. **Refuted by E2.** |
| H3 | Exposure bias: LTX trained on teacher features degrades with predicted features. | Stage-2 ckpt val with predicted features falls between teacher and disabled; Stage 3 closes the gap. | Predicted ~= teacher already. |
| H4 | Guidance is under-weighted relative to text. **Confirmed (E3): zero-init gate starves the semantic branch.** | Semantic out-proj/gate norms stay << text; raising inference CFG on guidance improves action MSE. | Norms comparable, CFG sweep flat. |
| H5 | A residual planner `F = SigLIP2(current) + delta` (zero-init delta) beats the absolute planner. | Lower MSE than baton step_20000 on every keyframe, and never worse than copy baseline. | No MSE gain after 5k steps. **Deprioritized: E2 shows the copy baseline is far worse than the planner.** |
| H5b | Feeding the current SigLIP2 grid as extra tower context (no skip) improves the absolute planner. | Lower MSE than the same-budget control at every keyframe. | No gain after 5k steps. |
| H6 | DA3 (WSA 4-layer) auxiliary loss improves the SigLIP2 prediction. | Lower SigLIP2 MSE at equal steps vs lambda_spa=0. | No gain or worse. |
| H7 | LIBERO is saturated; gains show on LIBERO-Plus. | Larger delta on LIBERO-Plus (Layout/Language) than on LIBERO. | - |
| H9 | TIPSv2 (google-deepmind/tips; patch 14, 224 px -> 16x16 = 256 tokens) is a better teacher than SigLIP2 (+DA3): text-aligned and spatially aware in one space. | Under a working gate, the Stage-2 oracle gap (teacher vs disabled, E3) is larger with TIPSv2 targets than with SigLIP2. | Gap equal or smaller. Test only after the gate fix works; download weights on HPC3 directly. |
| H10 | Knowledge-insulated co-training (KI, Physical Intelligence 2025: stop-grad from downstream experts into the VLM, VLM trained only on its own objectives plus VL data) keeps the planner's instruction sensitivity in Stage 3 without losing prediction accuracy. | Under KI, swap-instruction shift stays at the Stage-1 level (0.48-0.61) while planner MSE matches the frozen planner; end-to-end backprop lowers the shift. | No difference from frozen or end-to-end. Test only after LTX uses guidance; compare frozen / KI / end-to-end on LIBERO-Plus Language and swap sensitivity. Evidence so far: 5k extra planner steps already cut swap shift 12-20% (E4 B/C). |
| H8 | Wrist-view guidance is mostly noise (R^2 ~0.23 vs main ~0.53) and can hurt LTX. | Masking wrist guidance at validation does not raise, or lowers, video/action loss. | Wrist masking clearly hurts. |

## 3. Experiments

| ID | Tests | Setup | Cost | Status |
|---|---|---|---|---|
| E1 | H1, H4, H8 | Stage 2: GE base + online SigLIP2 teacher at [0,3,5,8], 20k steps, val modes teacher / semantic_disabled; log semantic vs text branch norms. | 8 GPU x ~2-3 days | running as 658685 via `sbatch_stage2.sh` (8 GPU, ~12.5 s/step, first val at 5k ~17 h); earlier attempts failed on the Slurm spool path and on peft 0.17 / diffusers 0.35.1 vs transformers 5, fixed by the overlay below |
| E2 | H2 | Baton step_020000 on LIBERO windows: per-keyframe planner MSE vs copy-current-frame MSE vs dataset-mean MSE, plus instruction swap. | 1 GPU x 15 min | **done** (job 658491) |
| E3 | H3 | Stage-2 ckpt (E1) val with teacher / predicted (E2 planner) / disabled. | 1 GPU x ~2 h | after E1 10k |
| E4 | H5b, H6 | Warm start from step_020000 incl. its AdamW moments, 5k steps, gbs 128 (2/GPU x 8 accum x 8 GPU), lr 2e-6 pretrained / 1e-4 new modules, gradient checkpointing, worker malloc_trim, GPU SigLIP2 preprocessing: current {none, context} x lambda_spa {0, 0.1}. A = none/0 is the same-budget control. | 4 runs x 8 GPU x ~6.5 h | submitted 659872-659875 (starts 09-29 20:42 to 09-30 10:55) |
| E5 | H3 | Stage 3: E1 ckpt + frozen best planner (E4), 30k steps. | 8 GPU x ~3 days | after E1, E4 |
| E6 | H7 + final | LIBERO 4x500 and LIBERO-Plus, sharded over HPC3 GPUs: GE-Act base, E1 disabled, E5. | ~0.5 day | after E5 |

## 4. Results

(append newest first)

### 2026-10-03 CORRECTION: planner mixing was never applied (bug, fixed in 3cc8c36)

`BATON_RESEARCH_PLANNER_PROB` (d1370f2) replaced `semantic_plan`, but the
Baton training path pops `semantic_plan` from the forward kwargs and
`forward_baton_ge_act` reads `baton_condition.tokens`, i.e. the teacher.
Every run with PROB > 0 before 3cc8c36 trained on teacher tokens only:
S3a, and the first s2planner20k / s2mix050_20k / s3planA_p050 submissions.
The S3a entry below is therefore just 3k more teacher steps on B, not a
planner fine-tune; its "short fine-tune is insufficient" conclusion is
withdrawn. The three runs were stopped. The fix replaces the condition
tokens (`dataclasses.replace(baton_condition, tokens=...)`) and 53fe8d5 logs
once per run how many samples were replaced and the mean |planner - teacher|
so the wiring is visible in every log. The runs are resubmitted under new
RUN_IDs (suffix _v2).

### 2026-10-03 Matched trend A vs control C (5k, 10k, 15k)

| step | C oracle / disabled | A oracle | A vs C (oracle) | A disabled |
|---|---|---|---|---|
| 5k | 0.1872 / 0.1889 | 0.1065 | -43% | 0.1949 |
| 10k | 0.0225 / 0.0234 | 0.0177 | -22% | 0.0418 |
| 15k | 0.0074 / 0.0088 | 0.0053 | -28% | 0.0198 |

With oracle future semantics the both-view additive model stays ~25% below
the matched control from 10k to 15k (the margin is not closing), but it
depends on the guidance: without it the error is ~2x the control. This is
the robustness gap the planner-trained run (s2planner20k) has to close.

### 2026-10-03 A (both views) at 15k

oracle 0.0053, disabled 0.0198: -0.0145 +- 0.0026 (-73%, win 1.00); main
only 0.0240, wrist only 0.0123. Control C at 15k is due shortly.

### 2026-10-02 S3a: 3k-step planner fine-tune of B@20k (p=1)

| S3a (B@20k + 3k steps, planner tokens p=1) | action_all | vs disabled |
|---|---|---|
| oracle | 0.0070 | -0.0049 +- 0.0006 |
| disabled | 0.0119 | - |
| planner | 0.0275 | +0.0155 +- 0.0040 (win 0.16) |

Planner mixing was active for all 3,000 steps (log confirms p=1.0). Planner
guidance went from 0.1134 (B@20k) to 0.0275 but is still worse than no
guidance, and oracle guidance still helps: a short fine-tune does not undo
20k steps of oracle dependence. Next: train the A configuration from the GE
base with planner tokens throughout (p=1, 20k steps, 8 GPUs) and compare
planner-mode error against control C at matched steps; S3b (A@10k, p=0.5)
remains queued.

### 2026-10-02 Matched comparison at 10k: control C vs A vs B

| @10k (action_all) | oracle | disabled |
|---|---|---|
| C: control (no additive) | 0.0225 | 0.0234 |
| **A: additive, both views** | **0.0177 (-22% vs C)** | 0.0418 (+79% vs C) |
| B: additive, main view | 0.0282 (+25% vs C) | 0.0468 |

- A with oracle guidance still beats the matched control, but the margin
  shrinks from -43% (5k) to -22% (10k) as the control catches up.
- A without guidance falls far behind the control (+79%; it was on par at
  5k): the model grows dependent on the guidance, consistent with the
  planner-prediction collapse.
- B (main view only) is worse than the control at matched steps, so the
  main-view-only variant is dropped; its large late gains were mostly
  training length.

### 2026-10-02 A (both views) at 10k

| @10k (action_all) | oracle | disabled | paired diff |
|---|---|---|---|
| A: additive, both views | **0.0177** | 0.0418 | -0.0241 +- 0.0041 (-58%, win 0.95) |
| B: additive, main view | 0.0282 | 0.0468 | -0.0186 +- 0.0018 (-40%) |

Main only (wrist masked) 0.0434, wrist only 0.0324: both views remain
complementary. Control C at 10k is due within the hour.

### 2026-10-02 Planner predictions instead of oracle guidance (negative)

E3 with `planner*` modes: the frozen Qwen3.5 baton planner (step_020000)
predicts the four keyframe grids from the last memory frame and the
instruction, replacing the SigLIP2 teacher tokens (same windows and seeds).

| model | oracle | disabled | planner | planner vs disabled |
|---|---|---|---|---|
| A @5k (both views) | 0.1065 | 0.1949 | 0.2074 | +0.0125 +- 0.0080 (win 0.48) |
| B @20k (main view) | 0.0064 | 0.0162 | 0.1134 | +0.0972 +- 0.0138 (win 0.02, ~7x worse) |

Masking either view with planner guidance is worse still (A: +0.036 / +0.043).

Reading: models trained on oracle teacher tokens rely on them, and the
planner's predictions (E2: R2 0.53 main / 0.23 wrist) are off-distribution
for them; the more a model relies on the guidance (B at 20k), the larger the
damage. Oracle gains do not transfer to deployment as is. The train/test
mismatch has to be handled in training: Stage 3 (fine-tune on planner
predictions), or teacher-token corruption / mixing during Stage 2, and/or a
better planner.

### 2026-10-02 Matched comparison at 5k: control C vs A vs B

| @ 5k steps (action_all) | teacher (oracle guidance) | disabled |
|---|---|---|
| C: control (no additive, original mainline config) | 0.1872 | 0.1889 |
| **A: additive, both views** | **0.1065 (-43% vs C)** | 0.1949 (= C) |
| B: additive, main view only | 0.2447 (+31% vs C) | 0.3569 |

- H1 confirmed at matched steps for A: oracle future-semantic guidance cuts
  action MSE by 43% against an identically trained model without it, and A
  without guidance is no worse than the control (no cost when the guidance
  is absent).
- B (main view only) is worse than the control at 5k even with guidance, so
  dropping wrist guidance hurts early training. B's later gains (10k, 15k)
  still need C at the same steps.
- C reproduces the original mainline (0.2128 / 0.2138), so the control is
  valid; its semantic path is unused (-0.0017).

### 2026-10-02 A (additive, both views) at step 5000

| @ 5k steps | teacher | disabled | main only (wrist masked) | wrist only (main masked) |
|---|---|---|---|---|
| original mainline (no additive) | 0.2128 | 0.2138 | - | - |
| B: additive, main view only | 0.2447 | 0.3569 | 0.2447 | 0.3569 |
| **A: additive, both views** | **0.1065** | 0.1949 | 0.1879 | 0.1712 |

A paired: -0.0884 +- 0.0098 (-45%, win 0.95); semantic_ratio 0.0095;
video_motion -0.0044. Each view alone recovers only a small part of the
gain, so when guidance is trained in from the start the two views are
complementary; this reverses E8, where wrist guidance was useless after a
1.5k-step fine-tune. A with oracle guidance halves the error of the original
mainline at the same step (0.106 vs 0.213, unpaired); the matched control C
at 5k is due shortly.

### 2026-10-01 B (additive, main view only) at step 10000

| B @ step | teacher | disabled | paired diff | win | semantic_ratio |
|---|---|---|---|---|---|
| 5000 | 0.2447 | 0.3569 | -0.1122 +- 0.0080 | 0.92 | 0.036 |
| 10000 | 0.0282 | 0.0468 | -0.0186 +- 0.0018 (-40%) | 0.93 | 0.034 |
| 15000 | 0.0098 | 0.0197 | -0.0099 +- 0.0009 (-50%) | 0.98 | 0.033 |
| 20000 (final) | **0.0064** | 0.0162 | -0.0099 +- 0.0009 (-61%) | 0.99 | 0.032 |

Update at 15k: the relative gain keeps growing (-31% -> -40% -> -50%) while
both conditions keep improving, i.e. the model does not learn to ignore the
guidance as it trains longer.

Video and motion-region video stay flat (-0.0004 / -0.0018). Between 5k and
10k both conditions improve sharply (the learning rate is still near peak
and the schedule runs to 20k), so step 10000 is the first point where B is
clearly better than every earlier model (E5-E8, ~0.079 after 6.5k steps).
Guidance still removes 40% of the remaining error. Whether B with oracle
guidance beats a model trained without guidance needs the matched control C
at 10k (about 15 h away); A (both views) is restarting after the HDF5-lock
crash and is at ~2k steps.

### 2026-10-01 Long runs on Qianhai ACP; first checkpoint (B, step 5000)

HPC3 is down for maintenance; training moved to Qianhai ACP. Two 20k-step
Stage 2 runs from the GE base with additive injection from the start
(zero_out gate, global batch 128, steps_to_save 5000):
A = both views (olabots-cci, 4x H800, pt-o6qzf3bu), B = main view only
(`BATON_RESEARCH_SEMANTIC_VIEWS=main`, olabots, 8x H800, pt-v3of5lse).
B's first attempt died at step 1057: quarkfs intermittently fails HDF5 file
locks (errno 9) -> dataloader worker crash -> NCCL all-reduce timeout. Fixed
with `HDF5_USE_FILE_LOCKING=FALSE` in the ACP launcher (77559c7).

E3 on B step_005000 (210 windows, same indices as every earlier E3):

| model @ 5k steps | teacher | disabled | paired diff |
|---|---|---|---|
| original Stage 2 mainline (no additive) | 0.2128 | 0.2138 | -0.0010 |
| B: additive, main view only | 0.2447 | 0.3569 | -0.1122 +- 0.0080 (win 0.92) |

semantic_ratio 0.036 (E8: 0.0006). Video and motion-region video ~unchanged.

Reading: trained with additive injection from the start, the model leans on
the guidance heavily (removing it costs 31%), but with oracle guidance it is
not better than the mainline was at the same step (0.245 vs 0.213; separate
models, so not a paired comparison). So far the guidance replaces information
the model would otherwise learn, rather than adding to it. A matched control
past 5k steps is missing (the mainline stopped at 5.7k), so a no-additive
control C with the identical schedule is launched to 20k steps.

### 2026-10-01 E8: additive injection with text (run on Qianhai CCI, 2x H800)

Same as E7 but the prompt is kept (step_001500, zero_out gate + additive
injection). E3, 210 paired windows (same linspace indices as E7), diff =
mode - semantic_disabled:

| metric | E7 no text | E8 with text |
|---|---|---|
| disabled action_all | 0.3716 | 0.0858 |
| teacher action_all | -0.0245 +- 0.0043 (-6.6%, win 0.83) | -0.0019 +- 0.0006 (-2.2%, win 0.62) |
| main only (wrist masked) | -0.0119 | -0.0019 +- 0.0005 (win 0.70) |
| wrist only (main masked) | -0.0036 | +0.0015 +- 0.0005 (win 0.30, worse) |
| video / video_motion | ~0 | -0.0003 / -0.0004 +- 0.0003 |

Reading:
- H1 holds with text but the effect is small: oracle future semantics buys
  2% action MSE when the instruction is present, 6.6% (13x larger absolute)
  when it is absent. Text already carries most of what the keyframes add.
- H8: the whole gain comes from the main view; wrist guidance alone hurts.
- Video (including the motion-region metric) does not use the guidance.
- The planner is weaker than the oracle (E2 R2 0.53 main), so planner-driven
  gains under normal text would be smaller still. The case for the planner
  is robustness when the instruction is missing or perturbed (H7), not
  in-distribution LIBERO accuracy.

Next: E9 continues E8 (warm start from its diffusion_model) to test whether
the with-text gain grows with training; H7 evaluation under language
perturbations follows.

### 2026-09-30 E4 planner variants and HPC3 access

Planner diagnostic (400 windows, means over keyframes):

| variant | main MSE | wrist MSE | swap shift main | swap shift wrist |
|---|---|---|---|---|
| base step_020000 | 1.291 | 2.217 | 0.522 | 0.528 |
| B current context | 1.278 | 2.196 | 0.469 | 0.455 |
| C DA3 aux | 1.276 | 2.196 | 0.476 | 0.456 |
| D context + DA3 | 1.282 | 2.195 | 0.465 | 0.452 |

- B, C and D are within ~0.5% of each other and ~1% better than base; D does
  not add up. The gain and the lower instruction sensitivity most likely come
  from the extra 5k steps (control A pending). H5b and H6 show no specific
  benefit.
- E8 (additive injection with text) finished training
  (`s2_additive_text/2026_09_30_18_49_14/step_001500`); its E3 could not be
  submitted: at 23:30 all HPC3 partitions became restricted to the `admin`
  group ("User's group not permitted to use this partition").

### 2026-09-30 E7: additive injection makes LTX use oracle guidance (no text)

Same setting as E6 (no text, zero_out gate, semantic lr 5e-4, 1500 steps from
step-5000 weights) plus the additive, spatially aligned injection. E3 with an
empty prompt, 210 windows, paired vs `semantic_disabled`:

| mode | action MSE | diff vs disabled | win |
|---|---|---|---|
| disabled | 0.3716 | - | - |
| teacher | 0.3471 | **-0.0245 +/- 0.0043 (-6.6%, ~5.7 SE)** | 83% |
| wrist masked (main only) | 0.3597 | -0.0119 +/- 0.0044 | 73% |
| main masked (wrist only) | 0.3680 | -0.0036 +/- 0.0013 | 56% |

All four horizons improve (-0.023 to -0.040). E6 (cross-attention only) gave
+0.0014 under the same conditions.

- The bottleneck was the injection mechanism, not the semantics: H1 holds with
  additive injection.
- H8 partly supported: the main view carries about half the gain, wrist alone
  ~15%; both together are best.
- Decoded full-frame video MSE barely moves (static background dominates); a
  motion-region video metric is needed.
- Next: E8 = E7 with the instruction (normal setting), the case the paper needs.

### 2026-09-30 E6: without text the model needs task information but still ignores guidance

| condition | action MSE | teacher - disabled (action) | video diff | semantic ratio |
|---|---|---|---|---|
| with text (E5b) | 0.079 | +0.0002 +/- 0.0001 | 0.0000 | 4e-4 |
| no text (E6, 1500 steps, caption dropout 1.0, E3 with empty prompt) | 0.455 | +0.0014 +/- 0.0008 | 0.0000 | 7e-4 |

- Removing the instruction raises action MSE ~6x, so the model is starved of
  task information, yet oracle future semantics (which encode the task) do not
  help at all. Redundancy with text is NOT the explanation.
- H1 is rejected for the current injection design: per-block cross-attention to
  LayerNorm-ed adapter tokens cannot carry the signal within 1500 steps even
  under strong incentive.
- Next (E7): additive, spatially aligned injection. Pool each keyframe's 16x16
  grid to the 8x8 latent grid, project with a zero-initialized linear layer, and
  add it to the input tokens of the matching future latent frame. If the oracle
  gap opens, the cross-attention injection is the problem; if not, revisit
  whether future semantics can help LTX at all.

### 2026-09-30 E5b: 10x semantic LR does not help; next, remove the text (E6)

- E5b (zero_out gate, semantic lr 5e-4, 1500 steps from step-5000 weights):
  per-block semantic residual ratio 4e-4 (E5: 2.9e-4); E3 paired diffs still
  ~0 (action +0.0002 +/- 0.0001, video 0.0000).
- The branch is not LR-bound: the model keeps it small because the guidance
  adds little over 4 memory frames + instruction on LIBERO (redundancy).
- E6 (running, job 662076): same as E5b but caption dropout 1.0, so the
  semantic plan is the only task cue (many LIBERO scenes host several tasks).
  E3 runs with an empty prompt. If the oracle gap opens, the injection works
  and text redundancy explains the null result -> move the evaluation to
  settings where text/priors fail (LIBERO-Plus language/layout, H7). If it
  stays ~0, the injection architecture itself is inadequate.

### 2026-09-30 E5: the zero_out gate fix does not make LTX use guidance (yet)

- E5 (zero_out gate, from Stage-2 step-5000 weights, 1500 steps, fast config):
  E3 paired diffs vs disabled are still ~0 (action -0.0001 +/- 0.0004, video
  0.0000). Action MSE overall dropped 0.214 -> 0.079 from the extra training,
  so training works; the semantic branch just does not matter.
- Wiring probe (20 windows): large random tokens change the output as little
  as true-future tokens (action -0.00046 vs -0.00026, video ~0).
- Residual trace: per-block ||semantic residual|| / ||hidden|| = 2.9e-4 with
  teacher tokens, exactly 0 when disabled. The path is live but ~0.03% of the
  residual stream per block; output projections grew from 0 to Frobenius
  0.4-1.7 (text cross-attn ~130).
- Reading: with 4 memory frames + text, LTX predicts LIBERO futures well and
  gains little from extra conditioning, and at lr 5e-5 the zero-initialized
  branch stays tiny. E5b tests the learning-rate part (semantic lr 5e-4).
- Planner side: diag C (DA3 aux) ~= diag B (current context), both ~1% better
  than step_020000 and both less instruction-sensitive (swap shift 0.48-0.61 ->
  ~0.42-0.56). Since C has no current context, that reduction comes from the
  extra 5000 steps, not from context. Control A is on hold because planner
  gains are moot while LTX ignores guidance.

### 2026-09-30 E3: Stage-2 step 5000 ignores even oracle guidance (210 windows, paired)

Paired differences vs `semantic_disabled` (same inputs and noise; negative = better):

| mode | action MSE (all horizons) | video MSE |
|---|---|---|
| teacher (true-future SigLIP2) | -0.0010 +/- 0.0006 (-0.5%, win 60%) | +0.0001 (no change) |
| teacher, wrist masked | -0.0008 +/- 0.0005 | +0.0001 |
| teacher, main masked | -0.0005 +/- 0.0003 | 0.0000 |

- H1 not supported at step 5000: oracle future semantics leave future-video
  error unchanged, so the model does not use the guidance. The earlier 1-sample
  "1-3% better" signal was noise. H8 cannot be tested until guidance is used.
- Cause (H4, confirmed in weights): deep-block semantic output projections are
  still at their random init (Frobenius 26.13 in blocks 7/14/21/27, identical
  to init), and the zero-initialized gate rows grew only to ~0.33. The residual
  `semantic_out * gate` with a zero-init gate is a bilinear saddle: the gate is
  ~0 so the attention weights get ~0 gradient (below Adam eps), and the gate
  grows slowly. The Qwen3-VL joint run showed the same symptom.
- E4-B diagnostic (400 windows): 7/8 keyframes ~1% better than step_020000 and
  less instruction-sensitive (swap shift 0.48-0.61 -> 0.42-0.56); needs the
  same-budget control A before any claim.
- Next (E5): `zero_out` gate mode (zero-init semantic output projection,
  (1 + gate) scale) from step-5000 weights, 1500 steps, then E3 again.
- Stage-2 speed: per-device batch 4 + GPU SigLIP2 preprocessing is 3.5x faster
  (13.7 vs 48.0 s/step on 2 GPUs, same 40 GB peak); GPU util was ~42% before.

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

- 2026-09-29: research-trainer infrastructure fixes, each verified by a
  2-GPU probe before relaunching:
  - Host OOM at ~440 steps: DataLoader worker heap grew ~9 MiB/sample
    (main process flat at ~7.8 GiB). `gc.collect`, `MALLOC_ARENA_MAX=2`,
    GPU preprocessing and `max_open_shards=1` did not help;
    `malloc_trim(0)` after each worker collate keeps total RSS flat at 23.4 GiB.
  - Loss rose after warm start (1.76 -> 2.05) with lr 1e-5/1e-4 and a fresh
    AdamW. Now lr 2e-6 for trained weights (source ended at 2.66e-6), 1e-4 only
    for new modules, and the source AdamW moments restored by name (627/627):
    loss stays at 1.74-1.78.
  - Stage 2 on 8 GPUs needed accelerate 1.14.0 (ZeRO-2 `no_sync`), added to the
    overlay; always smoke-test on >= 2 GPUs.

- 2026-09-28: GE-Act LTX on transformers 5.14.1 needs an import overlay,
  `/data/user/jhe724/envs/overlay_peft_tf5` (peft 0.21.0, diffusers 0.35.2,
  tensorboard 2.17.1, grpcio, markdown; all `--no-deps`), prepended through
  PYTHONPATH. The shared `qwen35` env is left untouched.

- 2026-09-28: planner frozen after Stage 1; LTX consumes SigLIP2 only; DA3 is
  a planner-side auxiliary target (WSA 4-layer). Paper to be aligned later.
