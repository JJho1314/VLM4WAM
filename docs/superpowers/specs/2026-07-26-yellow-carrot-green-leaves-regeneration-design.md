# Yellow-Carrot Green-Leaves Regeneration Design

## Goal

Regenerate the iter-3000 yellow-carrot oracle video while reducing the failure
where the carrot body separates from its green leafy top.

## Fixed Experiment Contract

Reuse `semantic_localization/oracle_repro/reproduce_oracle_yc.sh` and keep these
inputs unchanged:

- The original `yc74616` first frame.
- The original yellow-carrot GT-future SigLIP2 oracle plan.
- The iter-3000 SG-WAM checkpoint.
- Seed `0`, guidance `7`, `35` denoising steps, and `49` output frames.
- Five semantic-plan keyframes on the native spatial grid.

This is a prompt-only regeneration of the matching carrot oracle case. It does
not change the checkpoint, plan, initial image, or sampling parameters.

## Prompt Changes

Use this positive prompt:

> A Franka robotic arm with a parallel-jaw gripper carefully grasp only the
> [TGT] whole yellow carrot with its fresh green leafy top firmly attached,
> located in the sink basin, and place the intact carrot into the black pot
> next to the banana, without moving the banana. The green leafy top remains
> continuously attached to the carrot and moves together with it as one intact
> object throughout the entire video.

Keep the standard Cosmos negative prompt and append:

> detached green leaves, separated leafy top, broken carrot leaves, floating
> leaves, disconnected carrot and leaves

## Deployment

Submit the existing reproduction script through Slurm on HPC3 using one GPU.
Create a separate input-spec staging directory and write results to:

```text
/data/user/jhe724/workspace/VLM4WAM/eval_results_oracle_iter3000_yellowcarrot_greenleaves_REPRO
```

Do not overwrite the earlier reproduction.

## Verification

The run is accepted only if:

1. Slurm reports `COMPLETED` with exit code `0:0`.
2. The emitted sample JSON contains the approved positive and negative prompts.
3. The emitted config retains five semantic-plan keyframes and native-grid
   spatial layout.
4. The MP4 decodes fully as exactly 49 video frames.
5. Beginning, middle, and ending frames are extracted and visually inspected
   for whether the green leafy top remains connected to the carrot.

Visual inspection is qualitative: prompt constraints can reduce the detachment
failure but cannot guarantee object topology in a diffusion-generated video.
