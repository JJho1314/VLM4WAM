# Joint VLM + GE-Act + Action Expert Training Design

## Objective

Train one 30,000-step LIBERO model in which the dual-camera K4 Qwen planner,
the semantic-guided LTX video model, and the GE-Act action expert optimize in
the same distributed step. The run must preserve the existing video/planner
recipe while adding a real action objective; merely constructing or logging the
action branch is not sufficient.

## Trainable and frozen modules

Train these modules jointly:

- LTX video backbone at `2e-5`;
- LTX semantic adapter, semantic cross-attention, and semantic gates at `1e-4`;
- GE-Act action expert at `5e-5`;
- Qwen language backbone at `3e-6`;
- planner query banks, token embeddings, DINO/SigLIP2 head, and DA3 head at
  `3e-5`.

Keep condition-only or checkpoint-frozen modules frozen:

- LTX T5 text encoder and VAE;
- online SigLIP2 and DA3 teachers;
- Qwen vision encoder and LM head, matching `planner_meta.json`.

The existing `ltx_step_50000` checkpoint is the common initialization. It
already contains all 711 action parameters, including the 15-channel action
input/output projections, so the action expert must not be reset randomly.

## Data and model flow

Each LIBERO sample uses the existing ordered main/wrist observations, four
memory frames, nine future frames, 36 action steps, and the required predecoded
RGB cache. The planner predicts four future semantic keyframes at offsets
`[2, 4, 6, 8]` for both cameras. Its differentiable semantic tokens enter all
28 LTX blocks through the existing same-camera cross-attention and positional
encoding path.

The LTX video features produced at each block condition the corresponding
action block through the existing GE-Act video-to-action cross-attention. Action
tokens contain the normalized 7-D end-effector action and 8-D state, for 15
channels total.

## Objective

The optimizer minimizes:

```text
loss = video_loss
     + 1.0 * action_loss
     + 0.1 * planner_loss
```

`planner_loss` retains its existing semantic, four-layer DA3 WSA, and LM-plan
components. `action_loss` must participate in the joint total, finite-loss
checks, distributed reduction, TensorBoard/Accelerate logs, and checkpoint
metadata. A regression test must fail if the joint total omits action loss.

## Optimization and checkpointing

- 8 GPUs, BF16, DeepSpeed ZeRO-2;
- target global batch 128;
- 30,000 optimizer steps with 1,000 warmup steps;
- save only steps 20,000, 25,000, and 30,000;
- no LTX or Qwen gradient checkpointing in the preferred configuration;
- preserve exact distributed sampler and resume state.

The action parameters receive their own named optimizer group and LR instead
of being hidden inside `base_ltx`. Exported metadata records the five learning
rates, action loss scale, train mode, and trainable parameter counts. Resume
must restore all model, optimizer, scheduler, RNG, and sampler state.

## HPC3 deployment

Use the verified code deployment and environment:

- code: `/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af`;
- environment: `/data/user/jhe724/.venvs/vlm4wam_joint`;
- data: `/data/user/jhe724/junjie/datasets/LIBERO-fastwam`;
- predecoded RGB: `/data/user/jhe724/junjie/datasets/LIBERO-fastwam-predecoded-rgb`;
- synchronized weights: `/data/user/jhe724/junjie/vlm4wam_joint_assets`.

Create a separate HPC3 YAML and Slurm launcher; do not modify or overwrite the
OLA recipe. Run fail-closed preflight and an 8-GPU bounded smoke job first. The
smoke job must prove nonzero finite gradients for LTX, action expert, and Qwen,
and must log finite video, action, and planner losses.

Start the formal run only after smoke passes. Attempt batch 4 per GPU with
accumulation 4 first (global batch 128). If it OOMs because of the added action
branch, use batch 2 per GPU with accumulation 8 while keeping the same global
batch and all other objective settings unchanged.

## Acceptance criteria

- Unit tests cover loss composition, action optimizer ownership/LR, freezing,
  preflight, logging, and checkpoint metadata.
- The HPC3 preflight reports no missing paths or incompatible geometry.
- The 8-GPU smoke run completes at least one optimizer update with finite
  `loss_video`, `loss_action`, `planner_loss`, `ltx_grad_norm`,
  `action_grad_norm`, and `vlm_grad_norm`.
- The formal Slurm job is running on eight GPUs and writes to a new output
  directory without touching the existing OLA/pod run.
