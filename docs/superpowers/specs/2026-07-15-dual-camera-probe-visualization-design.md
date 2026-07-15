# Dual-camera 224px probe visualization design

## Problem

FastWAM supplies one horizontal 224x448 composite in the order
`main camera | wrist camera`. The planner and frozen teachers consume that
composite and emit one 16x16 token grid. The current visualization crops only
the left RGB camera, but stretches the complete 16x16 two-camera feature grid
to 224x224. The displayed RGB and semantic maps therefore do not describe the
same field of view.

The current depth comparison is also asymmetric: `depth_target` is dense MoGe
depth, while `depth_planner` is a 16x16 token prediction decoded by the linear
probe. A sharp dense target next to a low-resolution probe reconstruction mixes
probe reconstruction error with planner error.

## Scope

This is a visualization and evaluation correction only. It does not change the
planner geometry or retrain the planner. The current 256-token composite is
interpreted honestly as two 16x8 camera grids. Each half is then independently
upsampled to 224x224.

The existing global DINO PCA probe and shared Depth linear probe are reused.
No probe retraining is required.

## Camera split

- RGB input: split 224x448 at column 224.
- Camera order: left is `main`; right is `wrist`, matching the data config order
  `image`, then `wrist_image` and horizontal concatenation.
- DINO: project the 256 tokens with the fixed global PCA, reshape to 16x16x3,
  split into main/wrist 16x8 grids, then bicubic-upsample each grid to 224x224.
- Depth token maps: apply the shared 1024-to-1 probe at 16x16, split into two
  16x8 maps, then bicubic-upsample each map to 224x224.
- Dense MoGe target: split its width in half before resizing each camera to
  224x224.

Splitting before interpolation prevents information from bleeding across the
camera boundary.

## Depth alignment and colors

For each sample, time (`current` or `future`), and camera:

1. Resize that camera's MoGe target to 224x224.
2. Center each probe prediction in relative log-depth space.
3. Align its median log scale to the same camera's MoGe target.
4. Derive one 2nd/98th-percentile display range from that camera's MoGe target.
5. Render MoGe, teacher-probe, and planner-probe with the exact same Viridis
   range.

This makes color differences within a camera meaningful. The three outputs
separate dense-target detail, information retained by the Depth teacher tokens,
and error introduced by the planner.

## Per-sample output contract

Each sample directory contains one instruction text file and 24 independent
224x224 PNG files:

- 4 RGB observations: main/wrist x current/future.
- 8 DINO maps: teacher/planner x main/wrist x current/future.
- 12 Depth maps: MoGe/teacher-probe/planner-probe x main/wrist x
  current/future.

No contact sheet, bounding boxes, query-token frames, or combined-camera image
is produced.

## Metrics and diagnostics

- Report DINO MSE and cosine similarity separately for main and wrist cameras.
- Report Depth AbsRel, RMSE, and delta1 separately for MoGe-vs-teacher-probe,
  MoGe-vs-planner-probe, and the future persistence baseline per camera.
- Record the per-image Depth display bounds in a JSON sidecar so color mapping
  can be audited.
- Reject odd-width RGB, feature, or dense-depth maps rather than guessing a
  split point.

## Validation

- Unit tests first verify camera order, pre-interpolation token splitting,
  per-camera median alignment, shared color ranges, exact filenames, and exact
  224x224 output size.
- A four-suite smoke run reuses the saved probes and creates one sample per
  suite.
- The full visualization run creates two sample directories per suite and is
  copied back to the A6000 artifact directory.
- Representative main/wrist current/future outputs are inspected separately.

