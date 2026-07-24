# Joint VLM–GE-Act LIBERO Evaluation Design

## Goal

Evaluate the 40,000-step joint VLM–GE-Act checkpoint on the local A6000 using the same two-camera semantic conditioning used during training. The result must measure the action policy with the jointly trained planner enabled, not the unconditioned LTX fallback.

The evaluation will first run one smoke episode, then run the four standard suites (`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`) with 50 trials per task. It will report per-task and per-suite success rates and retain rollout logs and videos.

## Scope

This work adds a dedicated joint-checkpoint LIBERO evaluator, its tests, a local A6000 evaluation configuration, and a launcher. It also transfers the inference subset of the step-40,000 checkpoint from HPC3.

It does not retrain either model, run the SigLIP2 or DA3 teachers online, change action normalization, or modify the existing unconditioned evaluator. DA3 and SigLIP2 are training targets; online rollout uses the planner's predicted future SigLIP2 tokens only.

## Checkpoint Contract

The evaluator accepts one `step_40000` directory and requires all of the following before loading any model:

- `joint_meta.json` with `global_step=40000`, two camera views, four future keyframes, and 256 tokens per keyframe.
- `ltx/`, containing the jointly trained LTX video/action transformer export.
- `planner/`, containing `planner_meta.json`, the Qwen model and processor, `plan_head.pt`, `depth_head.pt`, and `plan_token_embedding.pt`.
- Planner metadata describing separate `main` and `wrist` images in that order, offsets `[2, 4, 6, 8]`, feature width 1024, and a 16-by-16 token grid per keyframe.

The evaluator fails with a precise error for a missing file, metadata mismatch, incompatible LTX semantic configuration, non-finite tensor, or unexpected planner output shape. It never falls back to semantic-free inference.

Only inference exports (`joint_meta.json`, `ltx/`, and `planner/`) are transferred to the A6000 machine. Optimizer and distributed training state are not needed. Transfer is resumable and verified after completion.

## Architecture and Data Flow

A new `eval_libero_joint.py` evaluator reuses the simulator, history buffer, action/state normalization, action execution, and reporting behavior of `eval_libero.py`. The original evaluator remains available unchanged for checkpoints that do not use a planner.

At every policy replan:

1. LIBERO provides the main-camera and wrist-camera RGB observations.
2. The evaluator preserves the ordered tensor `[main, wrist]`, normalizes it to `[-1, 1]`, and forms planner input `[1, 2, 3, H, W]` with the current observation and task instruction.
3. `FrozenDualCameraVLMPlanner.from_checkpoint()` predicts future semantic features under `torch.no_grad()` in evaluation mode.
4. The evaluator validates semantic tokens as finite `[1, 2, 4, 256, 1024]` and times as `[2, 4]`, equal to `[2, 4, 6, 8] / 8` for each view.
5. Existing observation-history handling forms the LTX input `[2, 3, T, H, W]` in the same main/wrist order.
6. The evaluator calls `CustomPipeline.infer()` with `semantic_plan`, `semantic_plan_times`, and an all-ones `semantic_condition_mask`, together with the existing instruction, state, and action arguments.
7. The pipeline performs the same semantic cross-attention and positional/time conditioning used by joint training and returns the GE-Act action chunk.
8. The existing FastWAM statistics and gripper conversion denormalize the action before execution in LIBERO.

The planner is invoked again whenever the policy computes a new action chunk, so its condition follows the current scene rather than being fixed for the whole episode.

## Model Loading and Memory

The dedicated evaluator loads the LTX transformer from `step_40000/ltx` and the planner from `step_40000/planner` in bfloat16 on local A6000 GPU 1. It uses the existing local LTX tokenizer, T5 encoder, and VAE assets, with a local evaluation YAML that preserves the training architecture and LIBERO statistics while replacing HPC3-only paths.

Text conditions are computed with the existing pipeline. If all models do not fit concurrently, the evaluator may move the T5 encoder off GPU after caching the finite set of LIBERO instruction embeddings. It may not reduce the planner, discard a camera, change semantic token geometry, or disable semantic conditioning to avoid an out-of-memory error.

## Evaluation Protocol

The launcher has two stages:

1. Smoke: one task and one trial on GPU 1. It verifies checkpoint loading, two-view order, planner tensor geometry, finite actions, and completion of a simulator rollout.
2. Full evaluation: all ten tasks in each of the four standard LIBERO suites, with 50 trials per task and the existing deterministic initial-state ordering.

Each suite writes to its own output directory. The outputs include the exact resolved configuration, checkpoint identity, log, per-episode outcome, per-task success rate, aggregate suite success rate, and rollout videos. The launcher starts the full stage only after the smoke stage exits successfully.

## Error Handling and Observability

Startup logs record the checkpoint step, LTX path, planner path, camera order, semantic shape, keyframe offsets, dtype, and CUDA device. The first replan logs the actual semantic tensor and time shapes. Repeated per-step shape logging is avoided.

Checkpoint or tensor-contract violations terminate evaluation before reporting metrics. Simulator episode failures are recorded with task and trial identifiers and follow the existing rollout error handling. A partial suite run keeps completed logs and videos so the failure can be diagnosed without losing prior results.

## Verification

Automated tests cover:

- The checkpoint contract, including rejection of missing or mismatched joint exports.
- Main/wrist view ordering and planner input normalization.
- Exact propagation of `semantic_plan`, `semantic_plan_times`, and the all-ones mask into `CustomPipeline.infer()`.
- Rejection of non-finite or incorrectly shaped planner outputs, with no unconditioned fallback.
- CLI parsing and smoke-task limiting.

Runtime verification on the A6000 consists of model-load preflight followed by the one-episode smoke rollout. The four-suite run begins only when both pass.

