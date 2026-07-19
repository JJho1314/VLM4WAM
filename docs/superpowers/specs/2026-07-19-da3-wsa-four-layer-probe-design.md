# DA3 WSA Four-Layer Probe Design

## Goal

Make the planner depth visualization consume the same four DA3 feature layers used by
the WSA auxiliary objective (`11, 15, 19, 23`) instead of discarding three layers and
decoding only layer 23. The resulting visualization must compare teacher and planner
features through one frozen, fair decoder and must preserve the legacy last-layer path.

## Probe contract

Add a `WSAMultiLayerDPTProbe` with input shape `[B, 4, 256, 2048]` and output shape
`[B, 1, 224, 224]` in log-depth space.

- Normalize every token independently across its 2048 feature channels with
  non-affine LayerNorm. This matches the WSA training objective and removes the large
  raw-scale mismatch between teacher and planner features.
- Project each DA3 layer independently to a shared feature width.
- Reassemble the four 16x16 token maps at 8x8, 16x16, 32x32, and 64x64 resolutions,
  then fuse them coarse-to-fine with DPT-style residual refinement blocks.
- Decode the fused map to 224x224 log depth.
- Reject the wrong layer count, token count, or feature width with explicit errors.

The probe is trained only on frozen, real DA3 four-layer teacher features against the
full DA3 depth-head output using the existing SILog plus multi-scale gradient loss.
Planner predictions are never used to train the probe, so the visualization cannot
learn to hide planner errors.

## Training interface

Extend `train_feature_probes.py` with `--which da3_wsa` while leaving all existing
choices and checkpoint names unchanged. The new artifact is
`da3_depth_wsa_probe.pt`; its checkpoint records the architecture config, DA3 teacher
layers, normalization contract, loss, and final training loss.

Use the existing LIBERO frame cache, frozen DA3 backbone/full head, optimizer, and
training loop. Default WSA layers are `11,15,19,23`, overridable through the existing
DA3 environment/configuration contract.

## Visualization interface

Extend the SigLIP2/DA3 visualization path to select the decoder from checkpoint
metadata:

- `wsa_multilayer`: load `da3_depth_wsa_probe.pt`, pass all four target/predicted
  layers, and label panels `Depth TARGET/PRED (WSA 4-layer)`.
- `last_layer` or legacy metadata: retain `da3_depth_v2_probe.pt` and the current
  single-layer behavior.

For the dual-camera K4 checkpoint, preserve camera and keyframe dimensions until the
final render. Produce one 3x6 PNG per sample, camera, and future keyframe, with current
RGB, future RGB, SigLIP2 target/prediction, WSA depth target/prediction, error maps, and
DA3-full reference depth.

## Verification

Tests cover:

1. Four-layer input produces finite `[B,1,224,224]` output.
2. Invalid layer/token/feature dimensions fail clearly.
3. Per-token positive scaling and channel-independent offset do not change probe
   output beyond numerical tolerance because normalization is part of the contract.
4. Checkpoint save/load preserves outputs.
5. Visualization routing selects the WSA probe for WSA metadata and retains the legacy
   probe for last-layer metadata.

After unit tests, train the WSA probe on OLA, compare its teacher-feature validation
quality with the existing last-layer probe, regenerate the 24 dual-camera K4 panels,
and inspect representative main/wrist and near/far frames.
