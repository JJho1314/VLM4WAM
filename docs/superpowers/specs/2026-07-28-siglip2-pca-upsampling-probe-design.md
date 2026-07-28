# SigLIP2 PCA Upsampling Probe Design

## Goal

Add a learned, feature-only visualization probe that converts the native
SigLIP2 spatial tokens used by the current planner
(`16 x 16 x 1024`) into a dense `256 x 256 x 3` PCA visualization. The result
should have clearer spatial boundaries than directly resizing a `16 x 16` PCA
map, while remaining an honest visualization of SigLIP2 features.

The probe must not read the source RGB image at inference time. It must not
replace or modify the existing `siglip_pca.png`; the episode exporter will add
`siglip_probe.png` beside the existing RGB, PCA, and DA3 depth artifacts.

## Scope

This change covers:

- fitting one deterministic, global three-component PCA transform;
- training a SigLIP2 PCA upsampling probe on LIBERO frames;
- validating the learned probe against the current interpolation baseline;
- loading the probe in the sparse LIBERO episode exporter;
- rerunning episode 288 for both cameras at stride 16.

It does not change planner training, planner predictions, SigLIP2 teacher
features, DA3 inference, or the existing three exported images.

## Feature and Teacher Contract

The low-resolution input is produced by the frozen
`siglip2-large-patch16-256` vision tower:

- input image size: `256 x 256`;
- feature layer: penultimate spatial vision layer;
- token grid: `16 x 16`;
- feature dimension: `1024`;
- input to the probe: `[B, 256, 1024]`.

The dense training target comes from the same frozen checkpoint evaluated at
`512 x 512` with interpolated position embeddings:

- high-resolution token grid: `32 x 32`;
- feature dimension: `1024`;
- target before PCA: `[B, 1024, 1024]`.

The target uses the same model and feature layer as the low-resolution input.
Only the spatial sampling density differs. No RGB reconstruction objective is
used.

## Global PCA Contract

PCA is fitted once using only high-resolution SigLIP2 tokens from the training
split. Fitting is deterministic and uses a bounded token sample so it does not
require holding the full training corpus in memory.

The checkpoint stores:

- the `1024`-dimensional feature mean;
- three ordered PCA components;
- a deterministic sign convention for each component;
- robust per-channel display limits computed from the 2nd and 98th
  percentiles;
- the maximum sampled-token count and random seed.

Every training target, validation target, and exported visualization uses this
same transform and display range. Therefore PCA colors are comparable across
episodes, frames, cameras, and runs that use the checkpoint. PCA is never
refitted on episode 288.

The `32 x 32 x 3` projected target is resized to `256 x 256 x 3` with bilinear
interpolation and clamped to `[0, 1]`.

## Probe Architecture

`SiglipPCAUpsampler` accepts only the low-resolution SigLIP2 tokens:

1. reshape `[B, 256, 1024]` to `[B, 1024, 16, 16]`;
2. use a `1 x 1` projection from 1024 channels to 256 channels;
3. apply four residual upsampling stages:
   `16 -> 32 -> 64 -> 128 -> 256`;
4. refine each scale using convolution, GroupNorm, and GELU;
5. produce three channels with a `1 x 1` head and sigmoid.

There is no RGB input, skip connection from the image, or image-guided
refinement. The same probe can therefore be applied later to planner-predicted
SigLIP2 features without leaking the ground-truth frame.

## Dataset and Split

Training uses the existing LIBERO frame-cache datasets across the available
LIBERO suites. Frames are split by episode, never by individual frame, so a
video cannot appear in both training and validation.

- episode 288 of `libero_10_no_noops_lerobot` is always excluded;
- the validation split is deterministic and recorded in the checkpoint;
- frames are sampled deterministically from the selected training episodes;
- the frozen low- and high-resolution SigLIP2 features are computed under
  `torch.inference_mode()`.

The initial recipe trains for 5,000 optimizer steps with AdamW and a cosine
learning-rate schedule. Batch size is chosen by the launcher for the available
GPU and is recorded in the run metadata.

## Objective

The optimization target is the fixed-PCA high-resolution teacher image:

- pixel loss: L1 between probe output and target;
- boundary loss: multiscale finite-difference L1 on horizontal and vertical
  gradients.

The total loss is:

`L = L1 + 0.25 * gradient_loss`.

No loss compares the output to RGB. This prevents the probe from becoming an
RGB decoder disguised as a feature visualization.

## Validation Gate

The reference baseline is:

1. project native `16 x 16 x 1024` tokens with the stored PCA;
2. normalize with the stored display limits;
3. bilinearly resize the resulting three-channel map to `256 x 256`.

On the held-out validation episodes, report for both the baseline and probe:

- mean L1 to the high-resolution PCA target;
- mean multiscale gradient error to the target.

The trained checkpoint is accepted for episode export only if the probe
improves both aggregate metrics over the baseline. A checkpoint that fails
either metric remains a training artifact and is not used to generate the
final episode result.

## Checkpoint Contract

The keeper checkpoint is named:

`siglip2_pca_upsample_probe.pt`

It contains:

- probe `state_dict` and architecture configuration;
- PCA mean, components, signs, and robust display limits;
- SigLIP2 model identity and expected feature layer;
- input size, teacher input size, grid sizes, and feature dimension;
- train/validation split metadata;
- final validation metrics and baseline metrics;
- training seed and optimizer recipe.

Loading fails clearly when the checkpoint does not match
`1024D / 16 x 16 / siglip2-large-patch16-256`, when PCA metadata is missing,
or when the validation gate is not marked as passed.

## Exporter Integration

The sparse episode exporter gains an optional
`--siglip-pca-probe` checkpoint argument.

When supplied:

- the exporter loads and validates the probe checkpoint;
- the already-computed native SigLIP2 features are passed through the probe;
- each camera/frame folder gains `siglip_probe.png`;
- the original `siglip_pca.png` remains unchanged;
- the manifest records the probe checkpoint, its validation metrics, and the
  additional file.

The episode 288 rerun uses the existing frame indices:

`[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 223]`.

For 15 frames and two cameras, the new export contains 30 copies of each of:

- `rgb.png`;
- `siglip_pca.png`;
- `siglip_probe.png`;
- `da3_depth.png`.

The expected total is 120 PNG files and 30 manifest records. Results go to a
new output directory so the validated 90-file export is preserved.

## Testing

Unit tests cover:

- PCA fitting shape, deterministic component signs, and bounded sampling;
- PCA projection using stored rather than per-image statistics;
- probe input/output shapes and absence of an RGB argument;
- loss behavior on identical and edge-mismatched inputs;
- checkpoint round-trip and contract rejection;
- baseline/probe validation-gate calculations;
- exporter artifact paths with and without the optional probe;
- manifest contents and expected PNG counts.

Runtime verification covers:

- a short overfit smoke run on a tiny frame subset;
- a full HPC training run and held-out validation report;
- a sparse episode 288 export on HPC;
- PNG readability and exact `120`-file count;
- visual inspection of main and wrist cameras at early, middle, and final
  sampled frames.

## Operational Safety

- Teacher and probe training never modify model checkpoints.
- The current 90-image episode export is not overwritten.
- Probe training artifacts and generated images remain under ignored runtime
  output directories.
- Failed jobs may leave logs but must not be treated as accepted checkpoints.
- The exporter rejects incomplete or incompatible checkpoints before creating
  image artifacts.
