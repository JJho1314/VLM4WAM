# MoRGBD MiniDPT v2 Depth Visualization Design

## Goal

Replace the structurally weak per-token linear depth visualization probe with a
spatial MiniDPT decoder adapted to the current 4B LingBot MoGe-to-MoRGBD depth
features. Keep the trained planner, query-token layout, teachers, and planner
checkpoint frozen. Generate reference-style, per-camera `224x224` visualizations
for current and future predictions.

## Root Cause

The existing probe maps each `1024-d` MoRGBD token independently to one scalar
and supervises only a downsampled `16x16` relative-log-depth grid. It has no
spatial decoder and never sees the dense object boundaries during training.
Upsampling its `16x16` output to `224x224` therefore enlarges coarse errors; it
cannot reconstruct geometry that was removed from the target.

The reference 2B visualization instead trains a MiniDPT-style spatial decoder
against the full dense depth head. Its convolutional multi-scale fusion and
dense gradient supervision are the material differences.

## Scope

In scope:

- Train a visualization-only MiniDPT probe for current 4B MoRGBD features.
- Decode both frozen teacher features and planner-predicted features.
- Generate separate main-camera and wrist-camera reference-style figures.
- Compare the new decoder with the existing linear probe on disjoint windows.

Out of scope:

- Retraining or changing the 4B planner.
- Replacing MoGe/MoRGBD with Depth Anything 3.
- Feeding RGB pixels into the probe or otherwise leaking the ground-truth frame.
- Claiming that the reconstructed image has native `224x224` token resolution.

## Architecture and Data Flow

Each LIBERO sample provides a current and one future two-camera composite frame
with native layout `main | wrist` and shape `224x448`.

1. The frozen depth teacher follows the planner-training path: resize the
   composite to `256x256`, run MoGe, and pass RGB plus MoGe depth through MoRGBD.
2. MoRGBD produces `256x1024` row-major tokens, reshaped to `16x16x1024`.
3. A MiniDPT probe projects the token map to 256 channels, reassembles four
   spatial scales, fuses coarse-to-fine features, and emits one `224x224`
   dense log-depth map for the squashed composite.
4. The training target is the frozen MoGe dense depth from the same teacher
   invocation, resized to `224x224` and converted to log depth.
5. For display, decoded disparity is stretched back to `224x448`, then split at
   the horizontal midpoint into main and wrist `224x224` views. The split occurs
   after dense decoding, matching the reference implementation.

The same decoder is applied without RGB guidance to teacher features and to the
planner's predicted current/future MoRGBD features.

## Training Protocol

- Freeze the planner, MoGe, and MoRGBD.
- Build a reusable CPU cache containing teacher features and dense log-depth
  targets from LIBERO current/future frames.
- Use disjoint deterministic train and validation window indices across all four
  LIBERO suites.
- Optimize the MiniDPT probe with AdamW and cosine learning-rate decay.
- Use scale-invariant log loss plus `0.5 *` multi-scale depth-gradient matching,
  following the reference v2 probe.
- Select the checkpoint with the best validation loss rather than the final
  optimization step.
- Save architecture configuration, state dict, training history, split metadata,
  and validation metrics with the probe.

The initial run uses the reference architecture (`feat=256`) and comparable
optimization exposure. Batch size may be raised to the largest stable value on
the H100, while effective sample count and scheduler length remain recorded.

## Visualization Contract

For every selected sample, write two figures:

- `sample_XX_main.png`
- `sample_XX_wrist.png`

Each is a three-row, six-column layout modeled on the supplied reference:

- Row 1: current/future RGB and DINO teacher/planner views.
- Row 2: current/future RGB and MiniDPT teacher/planner depth views.
- Row 3: full MoGe current/future depth references below the corresponding
  target columns.

Within each current or future depth comparison, teacher and planner use one
shared disparity color range. The MoGe-full panel is explicitly labeled as a
reference. Main and wrist figures remain separate square views.

Individual `224x224` panels are also saved alongside the grid so results can be
inspected without a contact sheet.

## Validation and Acceptance

Evaluation must report the MiniDPT teacher decode and planner decode separately.
Metrics include scale-aligned AbsRel, RMSE, delta-1, log-depth Pearson
correlation, and multi-scale gradient error, split by camera and time.

The new visualization is accepted only when all of the following hold:

1. Teacher MiniDPT decoding improves spatial correlation and gradient error over
   the existing linear probe on the same disjoint evaluation windows.
2. Teacher-decoded object and robot boundaries visually align with MoGe-full
   references in both cameras.
3. Planner panels are decoded solely from planner features and do not access the
   corresponding RGB or target depth.
4. All saved panels are `224x224`, camera labels are correct, and current/future
   frame indices match the planner inputs and targets.

If teacher features themselves cannot support a faithful dense decode, the run
must be reported as a representation bottleneck rather than hidden by color
normalization or interpolation.

## Deliverables

- Production MiniDPT training and visualization code.
- Probe checkpoint and metrics kept as experiment artifacts, not committed.
- Per-camera figures and individual panels kept as local/remote artifacts.
- Only production source code pushed to the GitHub feature branch; temporary
  tests, generated images, caches, checkpoints, and this process document stay
  out of the pushed commit.
