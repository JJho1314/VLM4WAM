## Task 8 Follow-up: Make the Real FastWAM K4 Training Input Launch-Ready

### Goal

Close the two CPU-side blockers found while preparing the real K4 DINO+depth
training run:

1. FastWAM YAML-relative text-cache paths cannot be overridden with the real
   cache, and pretrained normalization statistics cannot be injected, so the
   launcher either fails late or recomputes statistics after allocating Qwen 4B.
2. Hugging Face `datasets==4.1.1` returns `datasets.arrow_dataset.Column` for
   column access. `torch.stack(column)` raises because Torch requires a tuple or
   list, preventing the real LIBERO dataset from constructing.

Use strict TDD. Preserve all unrelated and user-owned working-tree changes.

### Files

- Modify:
  `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`
- Modify:
  `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
- Modify only if needed for portable documentation/validation:
  `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh`
- Modify:
  `third_party/FastWAM/src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py`
- Create a small dependency-light compatibility helper under the same
  `lerobot` package if that makes real behavior testable without importing all
  optional Hugging Face/video dependencies.
- Modify:
  `tests/test_lingbot_dino_depth_contract.py`
- Create:
  `tests/test_fastwam_lerobot_column_compat.py`

Do not stage the pre-existing untracked
`tests/test_fastwam_cosmos_semantic_plan.py` or unrelated vendor files.

### Part A: explicit FastWAM cache/stats overrides

1. Add parser options:

   - `--fastwam-text-embedding-cache-dir`
   - `--fastwam-pretrained-norm-stats`

   Both are optional `Path` values and are legal only with
   `--fastwam-data-config`. Add parser-error tests for invalid combinations.

2. Thread both overrides through:

   - `prepare_fastwam_data_config`
   - `preflight_fastwam_data_config`
   - `FastWAMOnlinePlannerDataset.from_config`
   - the preflight call in `main`
   - the actual dataset construction call after model creation

3. Path semantics must be cwd-independent and explicit:

   - YAML-relative dataset/cache/stats paths are anchored to the vendored
     FastWAM root.
   - explicit CLI paths are expanded and made absolute from the caller's cwd.
   - explicit values replace the YAML values.
   - `pretrained_norm_stats` may need to be added to the OmegaConf node when it
     is absent from the YAML.

4. Before `load_qwen3vl_model_and_processor` is called, the preflight must
   validate:

   - every selected dataset directory exists and is a directory;
   - the selected text-embedding cache exists and is a directory;
   - when pretrained stats is selected, it exists and is a regular file;
   - the Hydra `_target_` remains importable.

   Error messages must identify the failed asset. Tests must prove preflight
   fails before the 4B loader is reached.

5. The base shell launcher maps environment variables to the new options:

   - `FASTWAM_TEXT_EMBEDDING_CACHE_DIR`
   - `FASTWAM_PRETRAINED_NORM_STATS`

   Only append an option when the corresponding value is non-empty. Do not
   hard-code this machine's absolute paths into the portable K4 wrapper.

6. Add tests that exercise config override propagation all the way to the
   Hydra instantiate call and verify the shell launcher emits both arguments.

### Part B: Hugging Face Column compatibility

1. Reproduce RED with a Column-like iterable of scalar tensors and another of
   vector tensors: direct `torch.stack(column)` is invalid, while converting
   the iterable to a tuple first produces the expected tensors.

2. Add one narrow helper with this contract:

   - if the input is already a `torch.Tensor`, return it unchanged;
   - otherwise materialize it as a tuple and call `torch.stack`;
   - preserve dtype and support both scalar and vector tensor elements.

3. Replace all six direct stack sites in
   `lerobot_dataset.py` that consume Hugging Face columns/lists:

   - initial timestamps and episode indices;
   - `_get_query_timestamps`;
   - `_query_hf_dataset`;
   - `_query_hf_dataset_fast`;
   - `get_episode_data`.

4. Unit tests must exercise the helper's behavior and guard that all relevant
   call sites route through it. Keep the normal root test environment free of a
   hard dependency on `datasets`, PyArrow, Hub, or video decoders.

### Real-data verification

Use the `starVLA` environment and the actual assets after unit tests pass:

```text
Python: /data/LFT-W02_data/.conda/envs/starVLA/bin/python
Data config: third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml
Dataset dirs:
  /data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot
  /data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_object_no_noops_lerobot
  /data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_goal_no_noops_lerobot
  /data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_10_no_noops_lerobot
Text cache:
  /data/LFT-W02_data/junjie/_ola_stage/libero_qwen
Stats:
  /data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean/runs/dataset_stats.json
```

Instantiate the real FastWAM train dataset without loading Qwen/DINO/depth,
fetch one sample, and assert:

- `video.shape[:2] == (3, 9)`;
- `video_fps == 5.0` (scalar/tensor numerically);
- `instruction` is a non-empty raw LIBERO instruction;
- wrapping it in `FastWAMOnlinePlannerDataset` yields the current frame plus
  exactly four future keyframes at offsets `[2,4,6,8]`.

Do not modify the real data, cache, or stats. Temporary FastWAM work output must
go under `/tmp`.

### Verification and commit

- Run focused new/changed tests.
- Run the full CPU-safe FastWAM/planner gate used by Task 8.
- Run compileall, `bash -n` on both planner launchers, and `git diff --check`.
- Stage only the exact implementation/tests listed above (force-add only exact
  vendored files if required).
- Commit as `fix: preflight fastwam planner training data`.
- Write `.superpowers/sdd/task-8-launch-readiness-report.md` with RED/GREEN
  evidence, real-data sample results, exact staged paths, and commit SHA.
