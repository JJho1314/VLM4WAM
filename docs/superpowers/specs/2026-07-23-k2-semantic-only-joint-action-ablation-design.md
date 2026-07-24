# K2 Semantic-Only Joint Action Ablation

## Goal

Determine whether VLM-predicted future semantic features improve or degrade
LIBERO action performance when jointly training the semantic planner, LTX
video expert, and GE-Act action expert.

## Training Scope

- Run on HPC3 with the full joint action objective.
- Start from the existing K4 planner and GE-Act/LTX checkpoints.
- Reuse the K4 planner internally, but select only its future offsets `4` and
  `8` for supervision and GE-Act cross-attention.
- Train against online SigLIP2 targets for those two future frames.
- Do not construct the DA3 teacher, do not run the depth head for training,
  and do not include a depth loss.
- Train Qwen vision, Qwen language layers 16–27, planner semantic heads, LTX
  video modules, semantic injection modules, and the action expert.
- Keep Qwen embeddings, language layers 0–15, and the LM head frozen.

This is intentionally a fast K2 ablation rather than a native K2 planner:
the Qwen input still contains the K4 query sequence from the warm-start
checkpoint, while only two semantic predictions affect the objective and
GE-Act. A native 128-query K2 checkpoint is outside this experiment.

## Objective

The total loss is:

```text
loss = video_loss + action_loss + 0.1 * semantic_planner_loss
```

The semantic planner loss contains the existing SigLIP2 feature alignment and
causal plan-token CE terms. It contains no DA3/depth term.

## Formal Recipe

- Keyframe offsets: `[4, 8]`
- Semantic output geometry: two cameras × two keyframes × 256 tokens × 1024
- Per-device batch: `4`
- GPUs: `8`
- Gradient accumulation: `8`
- Effective global batch: `256`
- Optimizer steps: `25,000`
- Save checkpoints: `5k`, `10k`, `15k`, `20k`, `25k`
- Mixed precision: BF16
- DeepSpeed: ZeRO-2
- Qwen vision/language LR: `1e-4`
- Base LTX LR: `2e-5`
- Semantic LTX LR: `1e-4`
- Action LR: `5e-5`
- Planner head LR: `3e-5`
- Warmup: `1,500`
- Scheduler: cosine with absolute minimum LR `5e-7`
- Gradient checkpointing: disabled
- Seed: `2026`

## Negative-Optimization Evaluation

Use the same LIBERO evaluator and task set for three conditions:

1. The trained checkpoint with predicted K2 semantic guidance enabled.
2. The same trained checkpoint with semantic guidance masked off.
3. The original unguided GE-Act baseline checkpoint.

Interpretation:

- Condition 1 worse than condition 2 means the predicted VLM features are
  directly harmful at inference.
- Conditions 1 and 2 both worse than condition 3 means joint training damaged
  the video/action backbone even without using guidance.
- Condition 1 better than conditions 2 and 3 means VLM guidance is helpful.

Report success rate per LIBERO suite and the aggregate success rate. Use the
same seeds, rollout count, observation preprocessing, and action horizon for
all three conditions.

## Safety and Launch Gates

Before the formal launch:

1. Unit tests must prove K2 selection and that the depth teacher/head are not
   called.
2. Formal preflight must validate K2 geometry, semantic-only mode, and global
   batch 256.
3. A one-step eight-GPU smoke run must complete forward, backward, optimizer,
   scheduler, and checkpoint setup.
4. The formal run starts only after the smoke process exits cleanly.

Training logs must include total, video, action, semantic planner, Qwen vision
LR, Qwen language LR, and gradient-norm metrics.
