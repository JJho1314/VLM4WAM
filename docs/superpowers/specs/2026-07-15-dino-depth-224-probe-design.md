# DINO and Depth 224×224 Probe Visualization Design

## Goal

Train lightweight probes that expose the spatial information contained in the
planner's current/future DINO and depth tokens. Produce exact 224×224 outputs
without placing query-token boxes, metrics, observation, or instruction in one
combined figure.

The probe is diagnostic: it must not be powerful enough to invent scene detail
from LIBERO priors.

## Inputs and Evaluation Split

- Planner checkpoint: the completed `step_030000` Qwen3-VL-4B LingBot planner.
- Suites: `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
- Use 64 deterministic training windows and 16 disjoint evaluation windows per
  suite.
- Each window contributes a current frame and the single future frame at
  offset 8.
- Disable video augmentation and use the existing predecoded 224-pixel frame
  cache.
- Fit probes only on frozen teacher features. The planner and all teachers stay
  frozen.

## DINO Probe

Fit one global three-component PCA projection on the training-set teacher DINO
tokens. The fitted mean, basis, and robust 1st/99th-percentile display bounds
form the DINO probe artifact.

For teacher or planner tokens:

1. Apply the same centering and PCA basis (`1024 → 3`).
2. Reshape 256 tokens into a 16×16×3 map.
3. Normalize with the fixed training-set display bounds.
4. Bicubic-resize to exactly 224×224 and clamp to `[0, 1]`.

The current/future and teacher/planner maps therefore share one coordinate and
color system. No per-image PCA or per-image contrast normalization is allowed.

## Depth Probe

Train one shared per-token linear projection (`1024 → 1`) on frozen
LingBot-Depth teacher features. The target is scale-invariant relative log depth
derived from the corresponding MoGe dense depth map.

- Train at the native 16×16 token grid.
- Optimize Smooth-L1 plus a weighted spatial-gradient loss.
- Track and restore the lowest finite training-loss state.
- Bicubic-resize the 16×16 prediction to exactly 224×224.
- For metric computation only, align the predicted log-depth shift to the
  target median and decode back to positive relative depth.
- Use fixed per-sample target-derived display bounds for the paired target and
  planner images so their colors remain directly comparable.

The existing lightweight depth-probe principle is retained; no convolutional
or U-Net decoder is introduced.

## Planner Evaluation and Metrics

Run one frozen planner forward per evaluation batch and consume all four
outputs:

- `current_dino`
- `future_dino`
- `current_depth`
- `future_depth`

Report DINO projected-map MSE and mean cosine similarity at 224×224. Report
Depth AbsRel, RMSE, and delta1 for teacher-token oracle, planner current,
planner future, and future persistence. Summaries are written overall and per
suite. The report must state that the planner itself was trained on all LIBERO
episodes, so this is not a held-out policy benchmark.

## Output Layout

Each evaluated sample gets its own directory. Do not create a query-token
contact sheet. Save observation and instruction separately from decoded maps:

```text
<suite>_index#########/
  observation_current.png
  observation_future.png
  instruction.txt
  dino_teacher_current_224.png
  dino_planner_current_224.png
  dino_teacher_future_224.png
  dino_planner_future_224.png
  depth_target_current_224.png
  depth_planner_current_224.png
  depth_target_future_224.png
  depth_planner_future_224.png
```

Global artifacts:

```text
dino_pca_probe.pt
depth_linear_probe.pt
training_history.json
summary.json
summary.csv
probe_training_curve.png
```

All PNG outputs must have pixel dimensions exactly 224×224. The completed
directory is copied from the Pod to the A6000 workspace.

## Failure Handling and Verification

- Fail before training if checkpoint, teacher, dataset, or frame-cache assets
  are missing.
- Validate token geometry as `[B, 256, 1024]` for both modalities.
- Reject non-finite PCA statistics, probe losses, decoded maps, and metrics.
- Unit-test the global PCA transform, fixed normalization, 224×224 resize,
  depth probe shapes, disjoint split, and separate output layout.
- Run a one-window end-to-end smoke test before the full probe job.
- Verify expected evaluation sample count, output-file count, PNG dimensions,
  summaries, and remote process exit before reporting completion.

