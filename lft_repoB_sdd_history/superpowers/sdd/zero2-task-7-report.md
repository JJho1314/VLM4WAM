# Task 7 remote smoke report

**Final result: PASS**

The isolated preferred `b8/a2` smoke completed successfully on attempt 3.
Attempt 1 exposed an Accelerate external-config incompatibility before runtime
initialization. After an independently approved fix, attempt 2 passed the
external-config and ZeRO-2 runtime contracts but exposed a CPU/rank-local-CUDA
mismatch in DINO target normalization on the first batch. A second independently
approved, locally TDD-verified fix was then synced in exactly two files.
Attempt 3 completed both optimizer steps, exited 0, exported `step_000002`, and
passed exact inventory, metadata, and production-provider validation. Neither
earlier failure was CUDA OOM, so `b4/a4` was never needed.

## Initial preflight and isolated deployment

- Target: `root@182.242.159.145:30282`; remote host
  `cci-511532cb-9176-4f81-acf6-c8a5dacb5ed3-0`.
- Initial preflight: `2026-07-13T09:48:44Z`.
- Source repository
  `/root/nas/junjie/code/VLM4WAM_k1_fastwam_20260712` was present, while
  isolated destination `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713` was
  absent.
- All eight NVIDIA H100 80GB HBM3 devices reported `0 MiB / 81559 MiB` and
  `0%` utilization. The compute-app query was empty, and no pre-existing
  planner/Accelerate process was present.
- Neither `tmux` nor `screen` exists on the POD. With explicit task-owner
  authorization, all attempts used a unique `setsid` + `nohup` wrapper with a
  timestamped `tee` log and PID file under the isolated repository. No package
  or session tool was installed.
- The isolated repository was created by the guarded brief command only:
  `test ! -e ... && cp -a VLM4WAM_k1_fastwam_20260712 VLM4WAM_k1_zero2_20260713`.

## Initial scoped sync manifest

The initial `rsync -avR` transferred exactly the ten Task 7 files (`202,428`
total source bytes). Local and remote SHA-256 values matched:

| SHA-256 | Relative path |
|---|---|
| `4d6efb19841c312bfd30c5755363e86679b5c1baafe6cf78b2ce12f1d55a3119` | `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py` |
| `454260dfc7432237173356504a21f8da7daabbfcae8bb7fac329e211784051f5` | `scripts/qwen3_vl_semantic_planner/README.md` |
| `411934ce6ee0e50a29985d97e5478bf72b4a1cfe26692393b47cd7fe3618a62c` | `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py` |
| `65a46ef2ba0bebba3adec69c379e710e4838e807fb08e38ea695204a00909e83` | `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py` (pre-fix) |
| `60fb7d86140c75518d2c031923ca442e07127039b91bae7d2388d748a2ee3394` | `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh` |
| `dfb4693aaa575a1ec8ff3a9479fc43b40e37e29e86a97146f816df37863ed50d` | `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh` |
| `5e8afe0a88af81915ef2157d5b79e67f7bd1dd84ae0f99a3fea8ad6cd8bce8ac` | `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch` |
| `02ffb4d5b20a5262ffe73e9a5ed32b7978a54604700262178e5566cd94d0fcf1` | `tests/test_lingbot_zero2_runtime.py` (pre-fix) |
| `bc80c18c5288b1b55b246f57d39c3450f11e631ee74bab6c22a3e6cf2f2b2672` | `tests/test_lingbot_dino_depth_contract.py` |
| `a0108618558e1ae7aeec0603303377385a9891114a9e5c709a6d5d06ccdbb9d1` | `tests/test_lingbot_k1_current_future.py` |

The seven obsolete wrapper paths named in the brief were removed only under
the isolated destination and verified absent there. No output, log, cache,
`.git`, or unrelated dirty file was synced.

## Attempt 1: external-config failure

- Session: `zero2_smoke_b8a2_20260713T095214Z`.
- Start/end: `2026-07-13T09:52:14Z` to `2026-07-13T09:52:27Z`, about 13 seconds.
- Launcher PID/PGID: `2837690` / `2837690`.
- Rank PIDs reported by Torch Elastic: `2837753` through `2837760`.
- Log:
  `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/logs/zero2_smoke_b8a2_20260713T095214Z.log`.
- Terminal result: `SMOKE_EXIT_CODE=1`; all recorded processes were gone and
  all GPUs returned to `0 MiB`.
- Peak sampled GPU memory was only `4 MiB / 81559 MiB` per device; the model
  was never allocated.

POD versions were:

```text
accelerate=1.0.0
deepspeed=0.15.3
torch=2.7.1+cu126
```

The pre-fix generated Accelerate YAML combined a
`deepspeed_config_file` with top-level `mixed_precision: bf16`. Accelerate
1.0.0 added `mixed_precision` to `ACCELERATE_CONFIG_DS_FIELDS`, after which
`DeepSpeedPlugin._deepspeed_config_checks` rejected it on all ranks:

```text
ValueError: When using `deepspeed_config_file`, the following accelerate config
variables will be ignored: [..., 'mixed_precision'].
```

This was a non-OOM configuration failure, so no memory fallback was launched.
Attempt 1 produced only the two runtime-config files and no `step_000002`.

## Approved fix sync and retry preflight

Before the retry at `2026-07-13T10:09:28Z`:

- all eight GPUs again reported `0 MiB`, `0%`, and no compute applications;
- old PGID `2837690` had no processes;
- the isolated repository existed and
  `outputs/smoke_zero2_b8a2/step_000002` did not exist;
- the original source repository remained outside every write target.

Exactly two approved files were re-synced into the existing isolated copy:

| SHA-256 | Relative path |
|---|---|
| `b8447bf8e45e36b7b07ffd6e183b916ee21aa2d2281e667355493a600ba4e6c8` | `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py` |
| `7e331db3220b6d91daa99dcc3d5bb200a9cebbe9fa47bee21246670f5d79a4a9` | `tests/test_lingbot_zero2_runtime.py` |

Remote hashes matched local hashes. A fresh generated config was parsed before
launch and asserted:

```text
external-config contract valid: yaml_mixed_precision=absent ds_bf16=true grad_accum=2 zero_stage=2
```

The actual retry's generated YAML SHA-256 was
`fe6e9e88111486998a73e9cb2c78848b15523353fa847f65700a9d4ac4c8f1ed`;
the DeepSpeed JSON SHA-256 was
`a9dd371caf6dad7c864d522752f5778b3b6c5e2c8e61a597d43457d5c4ea5164`.
The YAML has no top-level `mixed_precision`; the JSON owns
`bf16.enabled=true`, `fp16.enabled=false`,
`gradient_accumulation_steps=2`, and `zero_optimization.stage=2`.

## Attempt 2: session, ranks, runtime, and memory

- Session: `zero2_smoke_b8a2_retry_20260713T101013Z`.
- Start/end: `2026-07-13T10:10:13Z` to `2026-07-13T10:13:18Z`, about 3 minutes 5 seconds.
- Log:
  `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/logs/zero2_smoke_b8a2_retry_20260713T101013Z.log`.
- PID file:
  `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/logs/zero2_smoke_b8a2_retry_20260713T101013Z.pid`.
- Status file contains the exact terminal code `1`.
- Wrapper launcher PID and PGID/SID: `2838680`.
- Pipeline wrapper `2838684`, `tee` `2838685`, Accelerate launcher `2838686`.
- Eight observed rank PIDs: `2838743`, `2838744`, `2838745`, `2838746`,
  `2838747`, `2838748`, `2838749`, `2838750`. Torch Elastic placed each rank
  in its own same-numbered PGID/SID; this was recorded in the PID file.
- All eight ranks were simultaneously alive and appeared one-to-one in the
  NVIDIA compute-app query at about `1496 MiB` each during initialization.

The requested launch values were captured exactly:

```text
RUN_KIND=smoke
MAX_STEPS=2
SAVE_STEPS=2
BATCH_SIZE=8
GRAD_ACCUM=2
EXPECTED_GLOBAL_BATCH=128
```

The production runtime contract was reached and logged as:

```json
{"distributed_type": "DEEPSPEED", "world_size": 8, "batch_size_per_gpu": 8, "gradient_accumulation_steps": 2, "global_batch_size": 128, "zero_stage": 2, "dtype": "bf16", "gradient_checkpointing": false}
```

DeepSpeed also logged `zero_config ... stage=2` and `step=0` initialization.
The progress bar remained `0/2`; optimizer step 1 was not completed.

GPU memory observations per device were:

| UTC | Memory used | Note |
|---|---:|---|
| `10:09:28` | `0 MiB` | retry preflight |
| `10:10:39` | `1506 MiB` | eight ranks initialized |
| `10:12:36` | `2698 MiB` | teacher loading |
| `10:13:08` | `17010 MiB` | highest sampled value, all eight ranks alive |
| post-terminal | `0 MiB` | all processes gone |

The observed peak was therefore `17010 MiB / 81559 MiB` per GPU, far below
the device limit. The complete log contains zero matches for CUDA OOM,
out-of-memory, `NCCL error`, NaN, accumulation mismatch, or global-batch
mismatch. It does contain rank/device-mapping NCCL warnings and the terminal
tracebacks below; no NCCL error caused the exit.

## Attempt 2 terminal failure and read-only root cause

Each reporting rank failed at the same first-batch boundary:

```text
train_qwen3vl4b_lingbot_dino_planner.py:2460
  current_dino, future_dino = dino_encoder.encode_current_and_future(...)
dino_video_target.py:139
  video = (video - self.mean) / self.std
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:<local-rank> and cpu!
```

Read-only source inspection traced the mismatch as follows:

- `current_image` and `keyframe_images` are popped from the batch before
  `move_qwen_inputs_to_device` moves the remaining Qwen inputs.
- The popped tensors are permuted and passed directly to
  `DinoVideoTargetEncoder.encode_current_and_future`.
- `DinoVideoTargetEncoder` stacks those tensors and immediately normalizes with
  its registered `mean`/`std` buffers, without first explicitly harmonizing
  the stacked video's device with the buffers.
- The runtime proves those normalization operands resolved to CPU and the
  rank-local CUDA device respectively.

The first observed Torch Elastic root cause was rank 1, PID `2838744`, exit
code 1 at `2026-07-13 18:13:15` local POD time. Torch Elastic then sent its own
closing SIGTERM to the other ranks. No signal or kill command was issued by
this Task 7 executor. The detached wrapper logged:

```text
SMOKE_EXIT_CODE=1
SESSION_END_UTC=2026-07-13T10:13:18Z
```

Postflight `ps` found none of the wrapper, launcher, or eight recorded rank
PIDs. Every GPU was `0 MiB / 81559 MiB` and `0%` utilization. Because this was
a deterministic CPU/CUDA device mismatch rather than OOM, `b4/a4` was not
launched and no code/configuration was changed.

## Approved DINO fix sync and attempt-3 preflight

Before attempt 3 at `2026-07-13T10:35:07Z`:

- all eight H100s reported `0 MiB / 81559 MiB`, `0%`, and no compute apps;
- all attempt-2 wrapper, launcher, and rank PIDs were gone;
- the isolated output still contained only the two runtime-config files and no
  `step_000002`;
- the original source repository remained outside every write target.

Exactly two approved files were re-synced into the isolated copy:

| SHA-256 | Relative path |
|---|---|
| `56ad7d76e320643d44019e127dc6d57919d32eb687a6c5d83ad19f53a5ff8962` | `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_video_target.py` |
| `e90ccf8a888df6e413436c21105841f398007ba4952025112850f7aa98fd5503` | `tests/test_lingbot_k1_current_future.py` |

Remote hashes matched local hashes. Read-only inspection verified that
`_normalize_video` moves `mean` and `std` to the input video's device and dtype,
and both DINO encode paths call that helper.

## Attempt 3: successful b8/a2 smoke

- Session: `zero2_smoke_b8a2_attempt3_20260713T103545Z`.
- Start/end: `2026-07-13T10:35:45Z` to `2026-07-13T10:38:39Z`, about 2 minutes 54 seconds.
- Log:
  `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/logs/zero2_smoke_b8a2_attempt3_20260713T103545Z.log`.
- PID file:
  `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/logs/zero2_smoke_b8a2_attempt3_20260713T103545Z.pid`.
- Status file and log both contain terminal code `0` / `SMOKE_EXIT_CODE=0`.
- Wrapper PID/PGID/SID `3462142`; pipeline wrapper `3462146`, `tee` `3462147`,
  Accelerate launcher `3462149`.
- Eight rank PIDs `3462205` through `3462212`, each recorded with its own
  same-numbered PGID/SID and observed simultaneously in the NVIDIA compute-app
  query.

The launch header again captured exactly:

```text
RUN_KIND=smoke
MAX_STEPS=2
SAVE_STEPS=2
BATCH_SIZE=8
GRAD_ACCUM=2
EXPECTED_GLOBAL_BATCH=128
```

The runtime contract was reached exactly:

```json
{"distributed_type": "DEEPSPEED", "world_size": 8, "batch_size_per_gpu": 8, "gradient_accumulation_steps": 2, "global_batch_size": 128, "zero_stage": 2, "dtype": "bf16", "gradient_checkpointing": false}
```

DeepSpeed resolved `train_micro_batch_size_per_gpu=8`, `train_batch_size=128`,
`gradient_accumulation_steps=2`, BF16 enabled, and ZeRO stage 2. The progress
bar reached step 1 at about 14.74 seconds and step 2 at about 19 seconds, then
completed checkpoint export/finalization at about 50 seconds of loop time.

GPU memory samples were:

| UTC | Memory used | Note |
|---|---:|---|
| `10:35:07` | `0 MiB` | attempt-3 preflight |
| `10:36:06` | `1506 MiB` | eight ranks initialized |
| `10:37:48` | `17010–17020 MiB` | model/teacher ready |
| `10:38:11` | `62294–63312 MiB` | steps 1 and 2 completed |
| post-terminal | `0 MiB` | all recorded processes gone |

The highest sampled value was `63312 MiB / 81559 MiB`, leaving 18247 MiB of
device capacity. The complete attempt-3 log contains zero matches for CUDA
OOM, out-of-memory, traceback, `NCCL error`, NaN, accumulation mismatch, or
global-batch mismatch. Rank/device-mapping NCCL warnings were informational;
no NCCL error occurred. All recorded wrapper, launcher, and rank PIDs were gone
after exit, and every GPU returned to `0 MiB` and `0%` utilization.

## Checkpoint, metadata, and provider result

Attempt 3 created a `9.3G` checkpoint at
`outputs/smoke_zero2_b8a2/step_000002`. Every required top-level item exists:

```text
qwen3vl_lora_or_model/    model config, generation config, index, 3 safetensor shards
processor/                tokenizer and image/video processor assets
plan_head.pt              246628535 bytes
depth_head.pt             246628542 bytes
current_plan_head.pt      246628591 bytes
current_depth_head.pt     246628598 bytes
plan_token_embedding.pt   1312388 bytes
planner_meta.json         11012 bytes
```

The three model shards are `777856152`, `4970230136`, and `3127576656` bytes.
The metadata SHA-256 is
`9b2ff76564f6c99d75dcb51790127ef767467bfc52231667cd0ef9840aace99f`.

The exact required metadata values were parsed and asserted:

```json
{"independent_modality_task_tokens": true, "keyframe_offsets": [8], "latent_len": 256, "num_keyframes": 1, "num_task_tokens": 64, "target_tokens": 256, "total_unique_latent_per_keyframe": 256}
```

`dino_depth_plan_provider.validate_planner_metadata` accepted the parsed
metadata, its returned contract reported `num_task_tokens == 64`, and the exact
production-consumer validation output was:

```text
step-2 export valid
```

## Scope audit

- No executor-issued `kill`, `pkill`, or signal command was used.
- Neither non-OOM failure triggered the allowed memory fallback; attempt 3
  completed with the preferred `b8/a2` contract.
- No command wrote to or deleted from
  `/root/nas/junjie/code/VLM4WAM_k1_fastwam_20260712`; copy, sync, deletion,
  configs, outputs, logs, status, and PID files were scoped to the isolated
  repository.
- Each retry synced only its two explicitly approved files.
- No local source file was edited, staged, or committed by Task 7. This report
  is the only local Task 7 artifact.
