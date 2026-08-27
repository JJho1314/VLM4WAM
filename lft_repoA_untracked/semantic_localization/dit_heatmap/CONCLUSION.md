# Does the semantic plan make the world model attend to the target object?

**Model**: iter3000 SG-WAM (`2b_semplan_gt_online_native_k5_1fps_dp015_320x576_49f_epsample_3000`,
which is the LAST checkpoint of that run — it was configured for 3000 steps).
**Input**: the yellowcarrot oracle case (`yc74616`), oracle plan = GT SigLIP2 of the real future frames.
Prompt names the target (yellow carrot) and a distractor to leave alone (banana).

## Answer: no target-specific spatial selectivity at this checkpoint.

### 1. Plan attention is near-uniform
Entropy of each video position's distribution over the 3645 plan tokens:
mean **7.815** vs the uniform maximum ln(3645) = **8.201** → 95.3% of maximum.
Effective number of plan tokens attended ≈ exp(7.815) ≈ **2477 of 3645**.

### 2. No plan region is preferentially read
Attention received per plan token peaks at only **1.5–2.2x uniform**; top-10% of tokens hold 12.7–13.5%
of the mass (uniform = 10%). Reshaped to the plan's 5 keyframes x 27x27 grid the map is salt-and-pepper
noise with no spatial structure (`figs/plan_token_attention.png`).

### 3. The target is inside the noise floor
Sharpness focus ratio (mean attention sharpness inside a CLIPSeg mask / outside), target vs 7 control nouns:
- block20: target 0.858, ranked 4/8, controls mean 0.902 sd 0.161 → **z = -0.28**
- block27: target 1.038, ranked 5/8, controls mean 1.026 sd 0.068 → **z = +0.19**
"black pot" (1.253) and "robot arm" (1.089) outscore the target. Random area-matched masks: 1.003 +- 0.035.

## Confounds ruled out
- **Plan format**: an earlier run of this analysis used this repo's defaults
  (`SEMANTIC_PLAN_NUM_KEYFRAMES=6`, `SEMANTIC_PLAN_SPATIAL_GRID=9` -> 486 tokens), which silently
  resamples the plan into a layout iter3000 never saw. Those numbers were invalid and are retracted.
  The results above use the trained format (5 keyframes, native 27x27, **3645 tokens**), verified in the
  emitted `gen/config.yaml` (`semantic_plan_num_keyframes: '5'`, `semantic_plan_spatial_grid: '0'`).
- **CFG branch**: SemanticRopeCrossAttention fires 36 times for 35 steps with B=1, i.e. the
  unconditional pass does not call it. Every captured tensor is the conditional, plan-carrying branch.
- **Metric degeneracy**: a hard top-20% metric degenerated (the mask covers ~4 of 720 cells); replaced
  by a soft mask-weighted inside/outside ratio with a control-noun noise floor.

## What this does and does not mean
- It does NOT say the plan is useless: the oracle video does follow the plan's target (carrot, not
  banana), so the plan carries usable information and the conditioning path works.
- It DOES say the mechanism is not spatial attention routing at iter3000 — the plan is consumed as a
  global/holistic signal rather than a spatial map. Consistent with the independent probe result in
  `../sg_probe/RESULT_honest.md` (no significant target-localizability gain, p>0.1).
- Untested: whether more training induces spatial selectivity. iter3000 is only 3000 steps AND is the
  last checkpoint of that run, so answering this needs a longer training run, not another analysis.

## Reproduce
```
python dit_heatmap/capture_attn.py     # plan-ON, hooks SemanticRopeCrossAttention.compute_qkv
python dit_heatmap/analyze_attn.py     # entropy maps + plan-token attention
python dit_heatmap/control_nouns.py    # control-noun noise floor
```
Outputs: `attn_result.txt`, `control_result.txt`, `figs/plan_attn_sharpness.png`,
`figs/plan_token_attention.png`.
