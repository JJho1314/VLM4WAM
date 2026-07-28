# Qwen3.5 Strict Baton Stage-1 Design

## Decision

Implement the visual branch of Baton Stage 1 without auxiliary objectives or
Plan-X-derived query operations. Each LIBERO sample produces exactly two Qwen
rows:

1. correct instruction with the main-camera observation;
2. correct instruction with the wrist-camera observation.

The deterministic wrong-instruction rows, counterfactual loss, changed-patch
weighting, cosine loss, delta loss, query self-attention, block-causal query
mask, and query-tower 3D RoPE are removed. None is part of Baton's Stage-1
visual alignment equation.

## Paper Contract

This design follows Section 3.1, Equation 3, Section 3.3, Equation 8, and
Appendix A.5 of *Baton: Explicit Semantic Blueprints for Joint Video-Audio
Generation*:

1. put one placeholder per target perceptual token in the assistant planning
   region;
2. run the MLLM causally and gather hidden states at those placeholder
   positions;
3. initialize one learnable query per planned token with
   `Normal(0, 0.02)`;
4. cross-attend the learnable queries to the gathered MLLM planning states;
5. map the aligned states to the frozen perceptual encoder width with Sem-MLP;
6. supervise them only with pointwise continuous-feature L2 regression.

Baton applies timestamp RoPE only to video/audio cross-modal alignment and
RS-RoPE only when DiT latents attend to planned tokens. This visual-only Stage 1
has no audio tower, so neither positional operation belongs inside its
single-modality alignment tower.

## Necessary LIBERO Adaptations

Only task-required substitutions are retained:

- Qwen3.5-2B-VL replaces paper Qwen3-8B so current observations can condition
  the robot planner.
- The already approved SigLIP2 patch16-256 target replaces paper
  SigLIP2-so400m-patch14-384, giving `256` spatial tokens per keyframe.
- Four LIBERO future keyframes give `Lv = 4 × 256 = 1024` planned tokens.
- Main and wrist observations enter Qwen independently and predict their own
  future feature grids. This is two rows, ordered `main`, then `wrist`.
- The audio branch is omitted because LIBERO/GE-Act has no audio modality.

## Objective

For predicted visual blueprint `H_sem` and the frozen SigLIP2 penultimate-layer
target `F_gt`, Stage 1 uses only:

`L_plan = mean((H_sem - F_gt)²)`

The loss is computed over sample, camera, keyframe, spatial token, and feature
dimensions. No counterfactual, cosine, delta, change weighting, discrete
codebook, or auxiliary text-alignment loss is used.

## Data Path

Training reads only the existing predecoded HDF5 dataset:

`/data/users/junjie/data/LIBERO-fastwam-hdf5/manifest.json`

RGB frames are already decoded and stored in HDF5. The loader must not open or
decode source video files. Frozen SigLIP2 features remain computed online from
those predecoded frames, matching Baton rather than introducing a feature
cache.

## Training Configuration

Following Appendix A.5:

- train the full MLLM, visual alignment tower, and Sem-MLP;
- frozen SigLIP2 teacher;
- AdamW with `β1=0.9`, `β2=0.999`;
- learning rate `1e-5`;
- BF16;
- DeepSpeed Stage 3;
- per-device batch `1`;
- no gradient accumulation;
- query initialization `Normal(0, 0.02)`.

The user-selected run length and save cadence remain `30,000` steps and every
`5,000` steps because the LIBERO dataset scale is not the paper's 1.5-million
clip corpus. This schedule change does not alter the Baton method.

## Validation

Tests must first fail against the current behavior and then prove:

- one sample collates to exactly two positive main/wrist rows;
- the Qwen assistant plan region contains exactly `1024` placeholder states per
  camera row;
- the planner accepts two rows per sample and has no negative output;
- the alignment tower implements learnable-query cross-attention followed by
  Sem-MLP, without query self-attention, causal masks, or query-tower RoPE;
- the loss and metrics contain only pointwise feature MSE;
- a tiny Stage-1 optimizer step remains finite and updates only owned modules;
- all full MLLM and alignment parameters are trainable while SigLIP2 remains
  frozen;
- the training loader reads the predecoded HDF5 path without source-video
  decoding;
- the OLA launcher selects DeepSpeed Stage 3, batch one, and accumulation one.

After deployment, a fresh OLA output directory will be used. The launch is
accepted only after all eight ranks enter training without OOM and the first
optimizer step completes.
