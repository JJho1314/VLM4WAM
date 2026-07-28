# SigLIP Target-Highlight Palette Comparison Design

## Goal

Generate one deterministic comparison image that makes the current
instruction target easy to locate inside the learned SigLIP PCA feature
visualization. The comparison must include both the main and wrist cameras,
and it must let the user choose among three display-only PCA channel
permutations.

This is a visualization change only. It does not retrain the SigLIP PCA
upsampling probe, alter its checkpoint, or change the underlying feature
values.

## Episode and Phase Contract

The source episode is:

`libero_10_no_noops_lerobot/episode_000288`

The instruction is:

`put the white mug on the left plate and put the yellow and white mug on the right plate`

The active target is fixed by frame index:

- `frame_index < 128`: `the white textured mug`
- `frame_index >= 128`: `the yellow and white mug`

The main and wrist cameras use the same active-target phase at a given frame,
but their relevance maps are computed from their own images.

## Target Relevance

Target relevance is computed with the complete frozen
`siglip2-large-patch16-256` image-and-text model.

For each image and active-target phrase:

1. Encode the phrase with the checkpoint's own tokenizer.
2. Run the native 256-pixel image through the vision tower.
3. Compute the model's normalized image-text similarity score.
4. Backpropagate that scalar score to the penultimate 16-by-16 spatial vision
   tokens.
5. Form a Grad-CAM map from the channel-wise product of the token activation
   and its gradient.
6. Apply ReLU, reshape to 16 by 16, and bilinearly resize to 256 by 256.

The design intentionally does not take a cosine similarity between raw
penultimate patch tokens and text features. Those tensors have matching
widths but have not both passed through the checkpoint's pooling/projection
path, so such a cosine map would not be a valid SigLIP similarity map.

Relevance normalization is deterministic. Values are clipped at the 5th and
95th percentiles within each image, mapped to `[0, 1]`, and smoothed only by
the bilinear resize. A zero-range map produces an all-zero mask.

## Combined Visualization

The learned `siglip_probe.png` is the base feature visualization. The target
map changes display only:

- low-relevance regions are desaturated to 40 percent and darkened to
  75 percent;
- high-relevance regions retain the selected PCA colors;
- the relevance value controls a translucent warm-yellow overlay;
- an amber contour marks the high-relevance boundary.

The original RGB, `siglip_pca.png`, `siglip_probe.png`, DA3 output, keeper
checkpoint, and manifest remain untouched.

## Palette Candidates

Each option is an exact permutation of the same three normalized PCA output
channels, so no feature information is discarded:

- A, current: `RGB = [PC1, PC2, PC3]`
- B, warm-balanced: `RGB = [PC2, PC3, PC1]`
- C, cool-balanced: `RGB = [PC3, PC1, PC2]`

The combined comparison applies the same target relevance map to all three
palette candidates.

## Comparison Figure

The initial selection figure uses two representative phase frames:

- frame 112 for the first target;
- frame 160 for the second target.

It contains four rows:

1. frame 112, main camera;
2. frame 112, wrist camera;
3. frame 160, main camera;
4. frame 160, wrist camera.

Its columns are:

1. RGB reference;
2. A with target highlight;
3. B with target highlight;
4. C with target highlight.

Every panel is labeled with camera, frame, active target, and palette. The
figure is saved as a readable PNG beside the existing episode export and is
returned directly to the user; no browser companion is required.

## Runtime and Files

The Grad-CAM computation runs on Ola using the existing SigLIP2 weights and
Python environment. Only the minimal source needed for the diagnostic is
synced. The resulting PNG and a compact JSON sidecar containing the model
path, phrases, phase boundary, frames, camera names, normalization rule, and
palette mappings are synced back to:

`outputs/libero_episode_000288_siglip2_da3_stride16_probe/target_highlight_comparison/`

The generator is deterministic for fixed model weights, inputs, and phrases.

## Validation

The implementation must verify:

- both cameras appear for both selected frames;
- frames below and above 128 use the correct target phrase;
- all three palettes are channel permutations of the same base feature;
- the combined panels are 256-by-256 RGB images;
- the comparison PNG and JSON sidecar are readable;
- the original 120 exported PNG files are unchanged;
- no model or probe parameters receive optimizer updates;
- the full SigLIP model score, rather than raw patch/text cosine, drives
  Grad-CAM.

Visual inspection must confirm that the highlighted region follows the
currently manipulated cup in both cameras and switches target after the fixed
boundary.

## Non-Goals

- Automatically infer task completion or phase transitions.
- Add Grounded-SAM or another external localization model.
- Retrain or fine-tune SigLIP2 or the PCA upsampling probe.
- Replace the original feature, RGB, or depth files.
- Change the episode exporter artifact count.
