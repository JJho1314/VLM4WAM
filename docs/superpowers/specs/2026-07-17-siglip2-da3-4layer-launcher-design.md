# SigLIP2 + DA3 Four-Layer Launcher Design

## Goal

Add a dedicated launcher for the existing DA3 four-layer alignment path while preserving the current SigLIP2 + DA3 last-layer launcher as an exact experiment record.

## Scope

- Keep `qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3.sh` unchanged.
- Add `qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3_4layer.sh`.
- Use `qwen3vl2b_siglip2_da3_4layer_libero_cur_k1` as the new default output directory so it cannot overwrite the last-layer run.
- Keep the SigLIP2 teacher, data, optimization, current/future alignment, and task-token settings identical to the last-layer baseline.
- Do not change visualization code or model implementation in this task.

## Four-Layer Configuration

The new launcher will explicitly pass:

```bash
DA3_ALIGN_STRATEGY=wsa_multilayer
DA3_TEACHER_LAYERS=11,15,19,23
DA3_LAYER_WEIGHTS=1.0,1.2,1.4,1.6
```

The launcher filename and output-directory name intentionally use `4layer`, not `wsa`. The internal strategy value remains `wsa_multilayer` because that is the trainer's existing configuration API for activating four-layer alignment.

## Data Flow

The launcher delegates to `lingbot_dino_4b/train_lingbot_dino_4b.sh`, which converts the three environment variables above into trainer arguments. The trainer constructs four DA3 targets from backbone layers 11, 15, 19, and 23 and applies the configured per-layer weights.

## Failure Handling

The new launcher inherits the baseline's strict shell mode and path preflight checks. A separate output directory prevents accidental checkpoint mixing between last-layer and four-layer experiments.

## Verification

1. Confirm the original launcher is byte-for-byte unchanged.
2. Run `bash -n` on the new launcher.
3. Statically verify the new filename/output name contain `4layer` and do not contain `wsa`.
4. Verify the strategy, layer list, and weights are explicitly passed to the inner launcher.
5. Inspect the final Git diff to ensure only the new launcher and planning documentation are added.
