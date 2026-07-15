# MoRGBD Reference-Style Visualization Design

## Goal

Regenerate the current Qwen3-VL 4B planner visualization in the exact visual structure of the supplied 2B reference figure without changing model predictions, probes, or evaluation data.

## Scope

- Keep the current 4B planner checkpoint and frozen feature teachers.
- Keep the trained MiniDPT depth probe and DINO PCA projection.
- Keep truthful labels for the current pipeline: `MoGe-full` and `depth_absrel`.
- Change only the composite-grid renderer and rerun visualization inference.
- Produce one grid for the main camera and one grid for the wrist camera per sample.

## Layout Contract

Each output is a 3-by-6 grid on a white `20 × 10.5` inch canvas, saved at 110 DPI with tight bounding-box cropping.

1. Row 1: current RGB, future RGB, current DINO target, current DINO prediction, future DINO target, future DINO prediction.
2. Row 2: current RGB, future RGB, current depth target, current depth prediction, future depth target, future depth prediction.
3. Row 3: blank, blank, current MoGe dense-depth ground truth, blank, future MoGe dense-depth ground truth, blank.

All visible panels are square. Blank cells are opaque white rather than transparent. Panel titles use 9-point text, and the two-line figure title uses 11-point text. The instruction is truncated to 80 characters to match the reference implementation.

## Data and Metrics

The renderer consumes the existing 224-by-224 per-camera panels. DINO and depth values are not recomputed or cosmetically altered by the renderer. The title reports full-composite current/future DINO MSE and current/future depth AbsRel from the current pipeline.

## Output Contract

- Grid filenames remain `sample_XX_main.png` and `sample_XX_wrist.png`.
- Individual 224-by-224 panels remain available beside each grid.
- Output PNGs must have an opaque white background and the same pixel dimensions as the supplied reference under the same Matplotlib version.
- The visualization summary continues to record `rgb_guidance: false`.

## Validation

- Unit-test the exact row/column panel mapping, titles, instruction truncation, canvas size, DPI, and opaque background.
- Run the existing visualization tests and script compilation checks.
- Regenerate representative main/wrist grids and compare them visually with the supplied reference.
- Confirm that only rendering changes; model/probe checkpoints and numerical predictions remain unchanged.

## Delivery

The design record and tests remain local. Only the production visualization code is included in the final pushed commit, matching the user's repository-cleanliness preference.
