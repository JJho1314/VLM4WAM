# OLA Predecoded VLM Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OLA dual-camera K4 planner train only from verified predecoded RGB arrays.

**Architecture:** The existing LeRobot dataset already supports strict episode-level NumPy caches. This change enables that strict path in the OLA config, verifies every cache before launch, and keeps DataLoader workers alive between iterator cycles.

**Tech Stack:** Python, PyTorch DataLoader, NumPy `.npy`, PyTest, Bash, YAML.

## Global Constraints

- Cache root is `/data/shared/datasets/libero_fastwam-predecoded-rgb`.
- All 3,424 main/wrist episode caches must verify before training.
- No missing/corrupt cache may fall back to MP4/PyAV decoding.
- Preserve the existing 8-GPU, per-GPU batch 8, accumulation 2, global batch 128 training contract.

---

### Task 1: Enforce the strict predecoded launch contract

**Files:**
- Modify: `tests/test_ge_act_dual_camera_planner.py`
- Modify: `ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml`
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_k4_siglip2da3_ola.sh`

**Interfaces:**
- Consumes: `ge_act/scripts/predecode_lerobot_videos.py --verify-only --config PATH`
- Produces: a launcher that exits before model initialization unless every configured cache is valid.

- [ ] **Step 1: Write the failing config/launcher assertions**

```python
assert train["predecoded_video_root"] == "/data/shared/datasets/libero_fastwam-predecoded-rgb"
assert train["require_predecoded"] is True
assert "predecode_lerobot_videos.py" in launcher
assert "--verify-only" in launcher
```

- [ ] **Step 2: Run the focused test and verify it fails on the old online-decode config**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py::test_ola_k4_config_and_launcher_are_fresh_and_fail_closed`

Expected: FAIL because `require_predecoded` is currently false.

- [ ] **Step 3: Enable the cache and add full-cache verification to the launcher**

```yaml
predecoded_video_root: /data/shared/datasets/libero_fastwam-predecoded-rgb
require_predecoded: true
```

```bash
"$PY" "$REPO_ROOT/ge_act/scripts/predecode_lerobot_videos.py" \
  --config "$GE_ACT_DATA_CONFIG" --verify-only
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py::test_ola_k4_config_and_launcher_are_fresh_and_fail_closed`

Expected: PASS.

### Task 2: Keep planner data workers persistent

**Files:**
- Modify: `tests/test_ge_act_dual_camera_planner.py`
- Modify: `qwen3_vl_semantic_planner/train_semantic_planner.py`

**Interfaces:**
- Consumes: CLI `args.num_workers: int`.
- Produces: `DataLoader(..., persistent_workers=args.num_workers > 0)`.

- [ ] **Step 1: Add a source-level regression assertion for persistent workers**

```python
source = (PLANNER_ROOT / "train_semantic_planner.py").read_text()
assert "persistent_workers=args.num_workers > 0" in source
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py::test_ola_k4_config_and_launcher_are_fresh_and_fail_closed`

Expected: FAIL because the planner DataLoader does not set `persistent_workers`.

- [ ] **Step 3: Set persistent workers on the planner DataLoader**

```python
persistent_workers=args.num_workers > 0,
```

- [ ] **Step 4: Run the focused and planner test suites**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py tests/test_ge_act_predecode_videos.py`

Expected: all tests pass.

### Task 3: Build, verify, and smoke-test the OLA cache

**Files:**
- Runtime data: `/data/shared/datasets/libero_fastwam-predecoded-rgb`

**Interfaces:**
- Consumes: 3,424 source MP4 camera episodes.
- Produces: 3,424 validated `[T,H,W,3] uint8` NumPy arrays and `manifest.json`.

- [ ] **Step 1: Push and pull the tested branch on OLA**

Run: `git push origin ge-act-dual-camera-planner` and `ssh olabots 'cd /data/users/junjie/code/VLM4WAM_dual_camera_k4 && git pull --ff-only'`

Expected: OLA reaches the new commit.

- [ ] **Step 2: Generate caches with the repository predecoder**

Run: `python ge_act/scripts/predecode_lerobot_videos.py --config ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml --workers 32`

Expected: `failed=0` and 3,424 total caches.

- [ ] **Step 3: Verify every cache**

Run: `python ge_act/scripts/predecode_lerobot_videos.py --config ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml --verify-only`

Expected: `predecode verification passed: 3424 caches`.

- [ ] **Step 4: Run one-step smoke then start the 30k formal run**

Run smoke with `RUN_KIND=smoke`, verify step 1 checkpoint, then launch the unchanged formal 8-GPU/global-128 contract into a new `_predecoded` output directory.

Expected: smoke loss is finite and formal training advances without any PyAV decode fallback.
