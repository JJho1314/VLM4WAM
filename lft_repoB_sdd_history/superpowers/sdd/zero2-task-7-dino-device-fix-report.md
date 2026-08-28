# Task 7 Report: DINO Target Normalization Device Fix

## Status

PASS for the scoped local correction and verification. The production POD was
not relaunched as part of this fix, so remote end-to-end confirmation remains
outstanding.

## Smoke Evidence

The second preferred `b8/a2` POD smoke reached the intended distributed
runtime contract before failing:

```text
distributed_type=DEEPSPEED
world_size=8
batch_size_per_gpu=8
gradient_accumulation_steps=2
global_batch_size=128
zero_stage=2
gradient_checkpointing=false
```

All eight ranks were alive through model and teacher initialization. Peak
sampled memory was `17010 MiB / 81559 MiB` per GPU, and the log contained no
CUDA OOM, NCCL error, NaN, or batch/accumulation mismatch. The first batch then
failed before optimizer step 1 on every reporting rank:

```text
train_qwen3vl4b_lingbot_dino_planner.py:2460
  current_dino, future_dino = dino_encoder.encode_current_and_future(...)
dino_video_target.py:139
  video = (video - self.mean) / self.std
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:<local-rank> and cpu!
```

This was a deterministic target-encoding device failure, not an OOM. The
`b4/a4` memory fallback was therefore neither applicable nor attempted.

## Root Cause

Accelerate's prepared DataLoader supplied rank-local CUDA `current_image` and
`keyframe_images` tensors. `DinoVideoTargetEncoder` stacked those inputs and
normalized them against `mean` and `std` buffers registered on CPU. That
normalization happens before `DinoVideoTeacher.get_future_feature` performs
its internal conversion to the teacher's device and BF16 dtype, so the teacher
could not repair the earlier operand mismatch. The depth encoder already keeps
its model and input on the same device and was not part of this failure.

The reference implementation in StarVLA's `Wan2_fastwam.py` locally converts
normalization statistics to the input tensor's device and dtype. The scoped
fix follows that pattern through one `_normalize_video` helper:

```python
mean = self.mean.to(device=video.device, dtype=video.dtype)
std = self.std.to(device=video.device, dtype=video.dtype)
return (video - mean) / std
```

Both `encode_future_keyframes` and `encode_current_and_future` now use the
helper. It does not mutate the registered buffers, so CPU behavior remains
unchanged. It also leaves the teacher call contract and output reshaping
unchanged.

## Strict TDD Evidence

### RED

Before any production edit, a real-tensor regression test constructed the
encoder without heavy initialization, kept `mean`/`std` on CPU, replaced only
heavy `_prep` and teacher work, and supplied BF16 current/future tensors on the
meta device.

Command:

```text
pytest -q tests/test_lingbot_k1_current_future.py::test_dino_normalization_uses_input_device_and_dtype
```

Exit code: `1`.

```text
dino_video_target.py:139: in encode_current_and_future
    video = (video - self.mean) / self.std
RuntimeError: Tensor on device cpu is not on the expected device meta!
1 failed in 2.46s
```

The fake teacher was not reached; the failure came from the real normalization
arithmetic at the production failure boundary.

### GREEN: focused

The identical focused command was rerun after the helper was added.

Exit code: `0`.

```text
.                                                                        [100%]
1 passed in 2.20s
```

The test additionally proves the registered buffers stay on CPU while the
teacher receives a BF16 meta-device video with shape `[2,3,3,2,2]`, and both
returned tensors preserve device, dtype, and `[2,256,1024]` geometry.

### GREEN: requested test pair

Command:

```text
pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py
```

Exit code: `0`.

```text
........................................................................ [ 87%]
..........                                                               [100%]
82 passed in 10.80s
```

This includes the pre-existing CPU DINO teacher-contract test, confirming its
normalization values, call count, keyword contract, and output geometry remain
unchanged.

### Source compilation

Command:

```text
python -m py_compile scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_video_target.py
```

Exit code: `0`; no stdout/stderr.

### Diff hygiene

Command:

```text
git diff --check
```

Exit code: `0`; no stdout/stderr after both evidence artifacts were created.
The snapshot review patch also passed
`git apply --check --reverse` against the current scoped files, proving it
describes their exact reversible delta from the before snapshot.

## Scope and Review Artifact

Production/test edits are limited to:

- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_video_target.py`
- `tests/test_lingbot_k1_current_future.py`

No trainer, depth, configuration, launcher, or unrelated file was modified by
this fix. No file was staged, committed, reset, checked out, or sent remotely.
The snapshot-based review artifact is
`.superpowers/sdd/zero2-task-7-dino-device-fix-review.diff`; it compares the
two scoped files with
`.superpowers/sdd/snapshots/zero2-task-7-dino-device-before`.
