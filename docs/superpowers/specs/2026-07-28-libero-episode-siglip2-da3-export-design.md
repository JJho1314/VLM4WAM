# LIBERO Episode SigLIP2 and DA3 Export Design

## Goal

Export independently viewable RGB, SigLIP2 feature, and DA3 depth images from
one complete LIBERO episode. The export must cover the trajectory rather than
one randomly selected training window, and it must not create composite panel
figures.

## Input

- Dataset suite: `libero_10_no_noops_lerobot`
- Episode: `episode_000288`
- Episode length: 224 frames at 20 FPS
- Cameras:
  - `observation.images.image` as `main`
  - `observation.images.wrist_image` as `wrist`
- Sampling stride: 16 frames
- Sampling always includes frame 0 and the final frame, frame 223.

The expected sampled frame indices are:

`0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 223`

## Feature Extraction

### SigLIP2

Use the configured `siglip2-large-patch16-256` teacher and retain its spatial
token grid. Fit one three-component PCA basis jointly across all sampled
frames and both cameras. Use that shared basis and shared per-channel display
range for every exported SigLIP image, so color changes are comparable over
time and between cameras.

### DA3

Use `DA3-LARGE-1.1` and its full depth head to predict depth for every sampled
frame. Convert depth to disparity for visualization. Use one robust global
display range across all sampled frames and both cameras, then apply the same
`turbo` color map to every output.

The export intentionally excludes planner predictions. It visualizes the
actual SigLIP2 and DA3 teacher outputs requested by the experiment.

## Output Layout

```text
episode_000288_stride16/
  manifest.json
  main/
    frame_000000/
      rgb.png
      siglip_pca.png
      da3_depth.png
    ...
  wrist/
    frame_000000/
      rgb.png
      siglip_pca.png
      da3_depth.png
    ...
```

`manifest.json` records the suite, episode, prompt, FPS, episode length,
sampling stride, sampled indices, model paths, camera mapping, and every
exported file.

## Error Handling

- Fail before model loading if the episode metadata or either camera video is
  missing.
- Verify that both videos contain every requested frame.
- Reject non-positive strides.
- Fail if SigLIP2 does not return a square spatial token grid.
- Do not report completion unless every sampled frame has all three files for
  both cameras.

## Verification

- Unit-test the inclusive sampling rule, especially final-frame inclusion.
- Unit-test the output path mapping for both cameras and all three modalities.
- Run the exporter on HPC3 with one GPU.
- Verify:
  - 15 sampled frames;
  - 2 cameras;
  - 3 PNG files per camera/frame;
  - 90 PNG files total;
  - manifest paths all exist;
  - representative RGB, SigLIP PCA, and DA3 depth images are visually valid.
