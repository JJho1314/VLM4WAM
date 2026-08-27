# FastWAM Cosmos Semantic Plan Design

## Goal

Integrate the existing VLM4WAM semantic-token injection path into the vendored
Cosmos-backbone FastWAM implementation under `third_party/FastWAM`.

## Scope

The integration reuses the existing Cosmos Predict semantic-plan mechanism:

- `SemanticPlanContextAdapter`
- semantic-plan keyframe selection and keyframe-time handling
- optional semantic-token RoPE
- semantic cross-attention inside selected Cosmos DiT blocks
- synchronized semantic-plan dropout where enabled by training code

The semantic tokens condition only the Cosmos video DiT. The FastWAM action path
receives semantic guidance through the existing video/action couplings:

- `mot`: action tokens joint-attend the video tokens
- `cross_attn`: action tokens cross-attend video hidden features
- `agra`: the action head reads the video foresight features

The action expert and GR00T action head are not modified to directly attend to
semantic tokens.

## Defaults

Semantic conditioning is opt-in. Existing FastWAM-Cosmos training remains
unchanged unless `semantic_plan_context: true` is set in the model config and a
batch provides `semantic_plan`.

When enabled, expected defaults match the existing Cosmos WM path:

- semantic plan dimension: `1152`
- source keyframes: `16`
- target keyframes: `6`
- spatial grid: `9`
- hidden dim: `2048`
- coordinate hidden dim: `256`
- semantic cross-attention blocks: all 28 Cosmos video blocks unless overridden
- RoPE enabled unless overridden

## Model Flow

`FastWAMCosmos.training_loss()` reads `semantic_plan` and optional
`semantic_plan_times` from the batch. It stores those tensors on the model for the
current forward call and dispatches through the active coupling.

`CosmosVideoExpert.prepare()` prepares semantic context alongside video tokens and
returns it in the video stream state. Couplings that execute video blocks manually
pass semantic context into the video blocks; standalone paths pass
`semantic_plan_B_L_D` and `semantic_plan_times_B_N` to `MiniTrainDIT.forward`.

## Data Flow

`RobotVideoDataset` gains optional semantic-plan config fields:

- `semantic_plan_dir`
- `semantic_plan_manifest`
- `semantic_plan_dim`
- `semantic_plan_max_tokens`
- `semantic_plan_default_to_zero`

If `semantic_plan_manifest` is set, dataset length and sampling come from the
manifest records. Each record supplies `sample_id`, optional `stem`, and optional
`future_frame_indices` / `video_frame_indices`; the loader reads
`<semantic_plan_dir>/<sample_id>.pt|.pth|.npy|.npz` and emits
`semantic_plan` plus `semantic_plan_times` when frame indices are available.

If no semantic plan config is provided, the dataset behaves as it does today.

## Non-Goals

- No direct semantic-token cross-attention inside the action expert.
- No online SigLIP/Qwen planner inference inside FastWAM in this pass.
- No changes to the existing `third_party/cosmos-predict2.5` semantic WM code.
