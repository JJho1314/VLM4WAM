# Frame 80 Action-to-Video Attention Design

## Goal

Generate real, paired action-to-video attention visualizations for:

`outputs/libero_episode_000288_siglip2_da3_stride16_probe/main/frame_000080/rgb.png`

The results must use the trained SG-WAM action expert, the matching LIBERO
episode's real action/state at frame 80, and must be written into the same
`frame_000080` directory.

## Model Comparison

Run the same trained checkpoint, frame, instruction, action, state, timestep,
noise seed, and attention hook twice:

1. `plan_off`: semantic plan disabled.
2. `plan_on`: semantic plan enabled.

This isolates the semantic-plan contribution without comparing differently
trained models. Capture `action_blocks[*].attn2`, where action queries attend to
video tokens. Do not substitute video-to-text attention or Grad-CAM.

## Input

- Dataset: `libero_10_no_noops_lerobot`
- Episode: `288`
- Frame: `80`
- Main RGB: the user-provided `rgb.png`
- Instruction: read from episode metadata.
- Action/state: read from the matching episode parquet at frame 80, normalized
  with the configured LIBERO statistics, padded to the model's 14-D contract,
  and perturbed only by the existing fixed light action noise.
- Semantic context: build from the episode camera frames using the trained
  semantic encoder and the checkpoint's four-keyframe configuration.

## Outputs

Write these files beside `rgb.png`:

- `action_attn_plan_off.png`
- `action_attn_plan_on.png`
- `action_attn_sg_gain.png`
- `action_attn_comparison.png`
- `action_attn_layers.png`
- `action_attn_maps.npz`
- `action_attn_metadata.json`

The individual overlays use the same normalization and color scale. The gain
panel visualizes only positive normalized `plan_on - plan_off` attention so it
shows where semantic guidance adds focus. The layer sheet exposes the captured
per-layer maps rather than hiding layer selection.

## Runtime and Validation

- Run on local GPU 1, which is currently idle.
- Use the existing GE-Act environment and local checkpoint/weights only.
- Verify that all images are readable, maps are finite and non-constant, both
  paired runs use identical non-plan inputs, and the source `rgb.png` checksum
  is unchanged.
- Inspect the final comparison image and report honestly if the real attention
  remains diffuse rather than fabricating sharper focus.
