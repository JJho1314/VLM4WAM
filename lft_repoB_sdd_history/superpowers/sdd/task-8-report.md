# Task 8 Report: checkpoint-backed online DINO+depth smoke

Base: `3a6c010`

## Implemented

- Added `third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py`.
  - Dependency-light module import; heavyweight Torch/Hydra/PIL imports are lazy.
  - Required checkpoint/config/task/device/instruction/image/FPS CLI contract.
  - Optional canonical DiT, VAE, text-cache, inference-step, and action-horizon overrides.
  - Strict positive/non-boolean FPS and positive-integer runtime validation.
  - Public provider `validate_checkpoint_files` + `validate_planner_metadata` gate before FastWAM import.
  - FastWAM runtime checkpoint/geometry preflight before Hydra model allocation.
  - Cwd-independent repo, config, planner-code, model-asset, and text-cache handling.
  - Prompt-specific text-cache preflight before model allocation.
  - RGB `[-1, 1]` BCHW conversion and exact configured image geometry validation.
  - One `infer_action` call under `torch.inference_mode()`, with raw instruction, formatted prompt, and explicit FPS.
  - Fusion hook requires exactly one `(1, 1024, 1024)` result and is always removed.
  - Finite/non-empty action validation and one JSON summary line containing only the required fields.
- Added `tests/test_fastwam_checkpoint_smoke.py` with 39 dependency-light/stub tests.

## TDD evidence

- Initial RED: missing smoke script (`FileNotFoundError`).
- Subsequent RED cycles covered missing CLI validation, provider preflight, Hydra/runtime preflight, image/model/text-cache preflight, inference/hook cleanup, summary output, FPS type validation, and blank-string rejection.
- Final focused result: `39 passed`.

## Verification

- Full CPU gate:
  - `459 passed in 8.44s`
  - Suites: planner contract/provider/fusion/sample timing/online runtime/semantic timing/legacy Cosmos/stage2/new checkpoint smoke.
- `python -m compileall -q scripts/qwen3_vl_semantic_planner third_party/FastWAM/src/fastwam/models/cosmos third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py`: exit 0.
- `bash -n` for the K4 wrapper and base 4B launcher: exit 0.
- Hydra `train + task=libero_cosmos_2cam224_online_dino_depth` composition verified online source, K4, 1024 tokens, and 5 FPS.
- Smoke `--help` verified from both the outer repository root and `third_party/FastWAM` cwd.
- Cached scope contains exactly:
  - `tests/test_fastwam_checkpoint_smoke.py`
  - `third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py`
- Cached `git diff --check`: clean; new-file placeholder scan: clean.

## Pre-existing untracked legacy test

`tests/test_fastwam_cosmos_semantic_plan.py` initially had two stale fixtures. It was minimally updated in the shared working tree only:

- file-backed dataset fixture now passes `semantic_plan_source="file"`;
- offline model fixture now configures `semantic_plan_dim=4`, `semantic_plan_max_tokens=3`, and `semantic_plan_num_keyframes=2`;
- offline sample now passes `video_fps=5.0`.

The whole user-owned untracked file was deliberately not staged or committed. Its focused result is `4 passed`.

## Real checkpoint/GPU smoke status

Not run in Task 8:

- Both A6000 GPUs remained occupied at final preflight (GPU 0: 36.7/49.1 GB, 93%; GPU 1: 28.9/49.1 GB, 72%), so allocating the 4B planner plus Cosmos was unsafe.
- The only discovered exported planner is legacy `sequence_length=49`, K5, latent length 40, with no strict K4/depth metadata; it is intentionally rejected by the new preflight.
- No production K4 DINO+depth checkpoint exists yet, so checkpoint-backed GPU inference depends on training/export completion.

Task-8-discovered launch follow-up was reported to root: expose FastWAM text-cache and pretrained-normalization-stats overrides through the trainer/launcher and validate them before loading Qwen 4B. This was kept out of the Task 8 smoke-only commit.

## Review follow-up: vendored Cosmos import path

The Task 8 review identified that the default Python environment cannot discover
`cosmos_predict2`, even though its source is vendored under
`third_party/cosmos-predict2.5`. The smoke now:

- accepts `--cosmos-repo`, with CLI > `COSMOS_REPO` environment > cwd-independent vendored default precedence;
- validates the repository directory and `cosmos_predict2/__init__.py` before Hydra/model allocation;
- puts the selected checkout first on `sys.path` before the FastWAM runtime/model import, ahead of stale Cosmos checkout paths;
- keeps Cosmos unimported during dependency-light smoke-module import;
- includes a real local `find_spec("cosmos_predict2")` probe.

Follow-up verification:

- focused smoke suite: `44 passed in 1.55s`;
- complete CPU gate: `464 passed in 8.31s`;
- `ruff format --check` reports both Task 8 files already formatted;
- `ruff check` reports all checks passed (including removal of the review's E731 assignment-lambda finding);
- compileall and help from both supported working directories pass;
- `git diff --check` is clean.

## Real-smoke follow-up: top-level `scripts` collision

The first real checkpoint-backed preflight in the `starVLA` environment exposed
a regular site-packages package named `scripts`. It shadowed the repository's
namespace-style `scripts/` directory, so the previous dotted provider import
failed before GPU allocation.

The smoke now loads the provider directly from the canonical absolute
`PLANNER_CODE_DIR/dino_depth_plan_provider.py` using a private path-hashed module
name and `spec_from_file_location`. It registers the module during execution so
the provider's dataclasses initialize correctly, preserves the conflicting
top-level `scripts` mapping, validates the source file before execution, and
restores/removes its private mapping on import failure.

Verification:

- focused smoke suite: `47 passed in 1.48s`;
- complete ten-suite CPU gate: `481 passed in 8.52s`;
- compileall, Ruff format/check, dual-cwd help, and diff check pass;
- exact `starVLA` probe reports site-packages `scripts/__init__.py`, loads the
  expected repository provider file, preserves the original `scripts` mapping,
  and exposes both public validators.

## Final real checkpoint/GPU smoke (2026-07-11)

GPU 1 later became free, so the deferred gate was executed without touching the
busy GPU 0.

1. A one-step, head-only strict K4 DINO+depth planner was trained with the real
   four-directory LIBERO dataset, real Qwen cache, and pretrained normalization
   stats. The run completed on GPU 1 with loss `0.2357904315` and exported:
   `/tmp/vlm4wam-k4-depth-smoke-20260711/step_000001`.
2. Export metadata reports sequence length 9, K=4, offsets `[2,4,6,8]`, times
   `[0.25,0.5,0.75,1.0]`, 32 shared + 32 private queries, 64 queries per branch,
   96 unique queries/keyframe, latent length 384, both heads, and 1024 dense
   tokens of dimension 1024.
3. Because `starVLA` lacks the official Cosmos CUDA dependencies, an isolated
   `/tmp/vlm4wam-cosmos-cu128-venv` was created with Torch 2.7/cu128,
   Transformer Engine 2.2, NATTEN 0.21, Cosmos 1.5.0, and Transformers 4.57.0.
   Existing conda environments were not modified.
4. The checkpoint-backed smoke completed with exit code 0 and emitted:

   ```json
   {"action_shape":[1,7],"fused_plan_shape":[1,1024,1024],"planner_checkpoint":"/tmp/vlm4wam-k4-depth-smoke-20260711/step_000001","video_fps":5.0}
   ```

The one-step export proves the complete online runtime path but is not a
converged production planner. Final fresh HEAD verification reports 483 CPU
tests passing plus compileall, both launchers' `bash -n`, Ruff, full-range
`git diff --check`, and Hydra K4/source/FPS composition all passing.
