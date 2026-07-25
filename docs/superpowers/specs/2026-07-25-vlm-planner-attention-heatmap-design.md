# VLM Planner Attention Heatmap Design

## Goal

Add a reproducible visualization for the dual-camera K4 VLM planner that
shows where the future semantic prediction head attends in each observation.
The result must use the trained planner's real Perceiver attention weights,
not Grad-CAM or an output-feature similarity proxy.

## Scope

The visualization targets exported checkpoints with:

- `plan_head_type: lingbot_dino`
- two independent camera inputs in `main`, `wrist` order
- four future keyframes with offsets `[2, 4, 6, 8]`
- one-layer LingBot-style `TaskTokenResampler`

Other planner geometries must fail with a clear compatibility error rather
than silently producing a misleading figure.

## Attention Definition

For each camera and future keyframe, reconstruct the attention computed by
the semantic plan head's `PerceiverAttention`:

1. apply the module's trained normalizations and Q/K projections;
2. use the same split-head layout and double-square-root scaling as training;
3. apply softmax over `cat(image_hidden, semantic_latents)`;
4. retain only the image-token keys;
5. reduce over attention heads and 256 output-grid queries.

Mean reduction is the default and scientifically primary output. A
`--query-reduction max` option may be provided for exploratory visualization,
but its use must be recorded in the output manifest.

The Qwen merged image-token grid is derived from `image_grid_thw` and checked
against the captured image-token count. Main and wrist tokens are never
concatenated into one heatmap.

## Implementation

Create:

`qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py`

The script will:

- load an exported planner checkpoint and its processor locally;
- load LIBERO samples through the existing GE-Act dual-camera dataset path;
- register a temporary hook on the future semantic plan head's Perceiver
  attention module;
- run one normal frozen planner prediction;
- capture calls in view-major, keyframe-major order and validate that exactly
  `2 * 4` maps were produced;
- remove the semantic-latent columns and restore each image-token vector to
  its Qwen spatial grid;
- jointly normalize the four keyframe maps within each sample/camera using
  robust 2nd/98th percentiles;
- bilinearly resize to the source camera resolution and blend a Turbo
  colormap with RGB.

The hook is removed after every prediction, including when inference raises.
No model source or checkpoint is modified.

## Outputs

For every selected sample, write:

- `sample_XX_planner_attention.png`: paper-style 2-by-5 panel;
- `sample_XX_main_k{1..4}_heatmap.png` and wrist equivalents: unblended maps;
- `sample_XX_main_k{1..4}_overlay.png` and wrist equivalents;
- `sample_XX_attention.npz`: raw reduced token-grid attention;
- one JSON manifest entry containing sample index, instruction, camera order,
  keyframe offsets, token-grid geometry, reduction, normalization, and files.

The composite layout is:

| Camera | Observation | t+2 | t+4 | t+6 | t+8 |
| --- | --- | --- | --- | --- | --- |
| Main | RGB | overlay | overlay | overlay | overlay |
| Wrist | RGB | overlay | overlay | overlay | overlay |

It uses rounded row containers, compact labels, equal square panels, and a
single title so its visual language matches the supplied MaskWAM reference.

## Validation

Unit tests will verify:

- reconstructed attention reproduces the Perceiver module output before the
  final projection;
- latent-token columns are excluded from the spatial heatmap;
- camera/keyframe call ordering and expected `2 * 4` capture count;
- `image_grid_thw` restoration rejects inconsistent token counts;
- reduction and robust normalization return finite maps.

An end-to-end smoke run will generate one LIBERO sample from the step-30000
planner checkpoint. The generated PNG will be inspected for correct camera
order, four distinct keyframe panels, and spatial alignment with the RGB
frames.

