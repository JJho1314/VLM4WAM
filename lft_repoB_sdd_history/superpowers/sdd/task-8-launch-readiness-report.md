# Task 8 Launch-Readiness Follow-up Report

## Result

- Commit: `1e0d4b3` (`fix: preflight fastwam planner training data`)
- Python: `/data/LFT-W02_data/.conda/envs/starVLA/bin/python`
- GPU/model allocation: none; verification was CPU-only.

## Strict TDD evidence

Initial focused RED:

```text
18 failed, 52 passed in 5.83s
```

The failures covered the missing dependency-light Column helper, all six
unconverted Hugging Face column stack sites, missing CLI options, absent path
rebasing and asset validation, missing override propagation to preflight/Hydra,
and absent shell-launcher arguments.

Focused GREEN after the minimal implementation:

```text
70 passed in 5.36s
```

Final full CPU-safe FastWAM/planner gate:

```text
480 passed in 8.52s
```

## Real-data verification

The real `libero_2cam_cosmos.yaml` train dataset was instantiated from all four
LIBERO directories with the real Qwen text cache and pretrained normalization
statistics. `FASTWAM_WORK_DIR` pointed under `/tmp`. Qwen, DINO, depth, and GPU
models were not loaded.

Observed sample:

```json
{
  "dataset_len": 277713,
  "video_shape": [3, 9, 224, 448],
  "video_fps": 5.0,
  "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
  "planner_current_shape": [224, 448, 3],
  "planner_keyframes_shape": [4, 224, 448, 3],
  "keyframe_offsets": [2, 4, 6, 8]
}
```

This verifies one current composed RGB frame plus exactly four future planner
keyframes selected at offsets `[2, 4, 6, 8]`.

## Additional verification

- `python -m compileall -q` on the planner scripts, Cosmos package, LeRobot
  package, and checkpoint smoke script: exit 0.
- `bash -n` on `train_lingbot_dino_4b.sh` and
  `train_lingbot_dino_depth_fastwam_k4.sh`: exit 0.
- Cached `git diff --check`: clean.
- New dependency-light helper/test files: `ruff format --check` and
  `ruff check` pass.

## Committed paths

- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
- `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`
- `tests/test_fastwam_lerobot_column_compat.py`
- `tests/test_lingbot_dino_depth_contract.py`
- `third_party/FastWAM/src/fastwam/datasets/lerobot/lerobot/column_compat.py`
- `third_party/FastWAM/src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py`

The pre-existing untracked `tests/test_fastwam_cosmos_semantic_plan.py` and all
other user-owned dirty files were not staged or committed.
