# semantic_localization — what the planner predicts, and what the world model does with it

Two separate questions, answered with separate experiments:

1. **Does the planner's predicted plan carry usable information?** → yes, strongly (§1)
2. **Does the world model consume it spatially?** → no (§2)

Together: the bottleneck is on the world-model side, not the planner. Anything aimed at making the WM
"look at" the target needs explicit spatial supervision; improving the planner is not the lever.

---

## 1. Planner side — the predicted plan is good

Checkpoint: `qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa_predecoded_b8_restart/step_030000`
(K=4 keyframes, 16x16 grid, 1024-d, 2 cameras, offsets [2,4,6,8], future-only).

| what | number | where |
|---|---|---|
| depth prediction, cosine vs DA3 teacher | **0.988** | `figs/depth_k4_main/manifest.json` |
| SigLIP2 plan, cosine vs teacher | **0.861** | same |
| target localisation from the plan, peak-hit | **0.791** (teacher 0.786) | `figs/dualcam_probe/probe_trained_result.txt` |

### 1a. Localisation probe — `figs/dualcam_probe/`
A FiLM(text)+conv head reads a target-noun heatmap out of the plan. Three arms share architecture,
split and schedule; only the input features differ. Split is BY EPISODE, so no keyframe of a training
episode appears in the test set. Main camera only (the wrist view is far harder — objects leave frame —
and dragged the mixed number down to 0.562).

| arm | peak-hit | AP |
|---|---|---|
| teacher SigLIP2 (upper bound) | 0.786 | 0.784 |
| **predicted plan** | **0.791** | 0.679 |
| shuffled (chance for this metric) | 0.321 | 0.310 |

Figures: 10 held-out samples, each 4 keyframes x 3 rows (real frame / teacher readout / predicted
readout). Trained heads kept as `head_teacher.pt`, `head_predicted.pt`; split in `split.json`.

### 1b. Depth — `figs/depth_k4_main/`
32 main-view PNGs, each a 6-panel sheet (RGB, SigLIP2 target/pred, depth target/pred, error maps),
decoded with the WSA 4-layer probe. Produced by the repo's own
`qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_qwen3vl2b_siglip2_da3_dual_camera_k4.py`.

### Reproduce
```bash
# Ola: predict + dump (hand-feeds the model, bypassing the GE-Act dataset config, which points at a
# pod path that does not exist there; DualCameraPlannerCollator only needs 4 fields per sample)
PYTHONPATH=$R/third_party/FastWAM/src:$R LIBERO_ROOT=/data/shared/datasets/libero_fastwam \
  PER_SUITE=30 python dump_dualcam_manual.py

# local: train the probe and score it (CAM_IDX=0 main only, 0,1 both)
CAM_IDX=0 python train_probe_dualcam.py
CAM_IDX=0 MAXN=10 python viz_dualcam_probe.py
```

---

## 2. World-model side — the plan is NOT used spatially

Checkpoint: Cosmos SG-WAM `converted_iter3000` (the last checkpoint of a 3000-step run).

### 2a. Plan cross-attention is uniform in every block — `dit_heatmap/CONCLUSION.md`
Swept all 28 blocks, both attention paths (video→text and video→plan). Best inside/outside ratio is
**1.14** (text, block 1) and **1.019** (plan, block 9); every measurement sits in 0.83–1.14 with
normalised entropy 0.98–0.997. The target ranks 4th/5th of 8 control nouns (z = −0.28 / +0.19) —
inside the noise floor.

Reference point: the mask-supervised `2b_mgv3_target_context_allblocks_iou50` run reaches ratio
**37.7 / 46.0** at blocks 12 / 16 (and ~1.0 everywhere else). That model was trained with target masks
and an IoU loss; ours has no spatial supervision at all, which is the whole difference.

Data `dit_heatmap/allblocks/`, table `per_block.txt`, figures `figs/allblocks_{text,plan}_best_block*.png`.

### 2b. But the plan does change the output — `oracle_repro/plan_on_off/`
Same weights, first frame, prompt, seed, 35 steps; only `semantic_plan_path` differs.
`|plan_on − plan_off| = 6.74/255`, versus 5.38/255 of within-clip frame-to-frame motion → **1.25x**,
and plan-ON tracks the ground-truth trajectory more closely. So the plan matters — as a global
semantic signal, not as a spatial map.

---

## 3. Attention posters and panels (FastWAM RGB-only vs SG-WAM)

`wam_attention/make_wam_poster.py` renders the poster; `export_panels.py` writes each scene as its own
image; `contact_sheet.py` builds a browsing sheet.

All panels are MAIN camera; the main|wrist composite sets were deleted as unused. `VIEW=composite` /
`OUT_SUFFIX=""` still works if they are ever wanted again — the attention `.npz` files are untouched.

| folder | scenes | contents |
|---|---|---|
| `figs/panels_main_regen/` | 8 | the line-up of the original poster, individual images only |
| `figs/panels_main_all/` | 200 (40 tasks x 5 layouts) | + `_contact_sheet.png` for browsing |
| `figs/wam_sg_best_main_panels/` | 12 | top-12 by concentration gain |

```bash
VIEW=main TOPK=12 python make_wam_poster.py
MATCH_PROMPTS='task a|task b|...' VIEW=main python make_wam_poster.py   # rebuild a given line-up
OUT_SUFFIX=_main PER_TASK=5 PANEL_DIR=<dir> python export_panels.py     # PAIR=1 adds a side-by-side
```

**Read the numbers, not the poster.** Posters show the best-gain scenes out of 240–360 and apply a
60–99 percentile + gamma 1.6 stretch. Over ALL samples the concentration means are SG 0.185 vs
RGB 0.162 (**1.14x**, main view) — far milder than the poster looks. Also the RGB-only baseline is a
generic Cosmos model with no in-domain fine-tuning, so its gap to SG mixes "semantic guidance" with
"15k steps of domain training"; it is a qualitative illustration, not an ablation.

---

## 4. Config traps (this cost several invalidated rounds)

One codebase, checkpoints from different months, many implicit defaults. Always cross-check against
the `config.yaml` the original run emitted — never trust the repo default.

| setting | what the checkpoint needs | repo default | symptom if wrong |
|---|---|---|---|
| `SEMANTIC_PLAN_NUM_KEYFRAMES` / `SPATIAL_GRID` | 5 / 0 → 3645 plan tokens | 6 / 9 → 486 | plan silently resampled; attention collapses to uniform (a convincing but fake negative) |
| `COSMOS_NUM_FRAMES` | 49 → `state_t`=13 | 93 → `state_t`=24 | 93-frame clips; the model extrapolates to double length |
| `resolution` in a direct `generate_vid2world` call | `"320,576"` | `"none"` falls back to 720p | latent grid 44x80 instead of 20x36 |
| blocks probed | sweep ALL | — | blocks 20/27 read uniform even in a model that does focus |

Also: `pkill -f <name>` matches the ssh command that also *launches* `<name>` — kill and launch must be
separate calls, the `[b]racket` trick alone is not enough.

---

## 5. Files

**Planner probe**: `dump_dualcam_manual.py` (Ola), `train_probe_dualcam.py`, `viz_dualcam_probe.py`
**WM analysis**: `dit_heatmap/{capture_allblocks,analyze_allblocks,control_nouns,capture_attn,diag_sharpen,temp_sharpen,plan_ablate}.py`
**Oracle**: `oracle_repro/{reproduce_oracle_yc.sh,gen_plan_on_off.sh}`
**Posters**: `wam_attention/{generate_many,make_wam_poster,export_panels,contact_sheet}.py`
**Earlier probes** (LTX line, honest negative): `sg_probe/RESULT_honest.md`
**Improvement code** (not yet trained): `sg_improve/` — README there covers the target-mask auxiliary
loss (Option 2), task-level eval (Option 5) and the discrete-plan VQ head (Option 6).
