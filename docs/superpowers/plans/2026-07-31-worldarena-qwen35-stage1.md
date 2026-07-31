# WorldArena Qwen3.5 Stage-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing strict Baton Qwen3.5-2B semantic planner on the 509 generated WorldArena/RoboTwin episodes using one independent head-camera row and four future SigLIP2 feature grids.

**Architecture:** Generalize only the per-batch camera axis in the existing Baton batch, model reshape, teacher, metrics, and checkpoint metadata while preserving the LIBERO two-camera defaults. Add a separate WorldArena manifest/HDF5 adapter that stores resized RGB frames per episode, samples one current plus four normalized-horizon future frames, and plugs into the common Stage-1 trainer through an explicit `dataset_type`. Deploy the verified code and predecoded cache to ACD1-58, run a one-GPU finite step, then an eight-GPU step-20 checkpoint probe before the 30,000-step run.

**Tech Stack:** Python 3.10, PyTorch 2.7, Transformers Qwen3.5-2B, frozen SigLIP2 patch16-256, h5py, OpenCV, Accelerate/DeepSpeed ZeRO-2, pytest, Slurm/H100.

## Global Constraints

- Training data is only `episodes/` from `DavidxWang/worldarena2026-robotwin-data`; `official_episodes/` and all official WorldArena validation/test roots are rejected.
- Each sample contains exactly one camera named `head`, one current RGB frame, and four strictly future RGB frames.
- Future indices use `f_k = c + round((k + 1) * (120 - c) / 4)` for `k=0..3`, with `c` in `[0,116]` for the 121-frame release.
- Source MP4 lengths are not trusted as canonical. The downloaded 509-episode release has 0/509 MP4s with exactly 121 decoded frames (observed range 76–787), while all metadata/actions declare 121. For every decodable source with `N >= 1`, map canonical index `i=0..120` to `round(i * (N - 1) / 120)`, allowing repeats for short videos and uniform downsampling for long videos while covering both endpoints. HDF5 and all future-frame sampling remain in canonical 121-frame coordinates.
- Qwen receives the current head image and unchanged instruction only; actions and calibration paths remain metadata.
- SigLIP2 remains frozen and online; the only loss is pointwise MSE against penultimate `[256,1024]` patch features.
- Qwen3.5-2B, the visual alignment tower, and Sem-MLP remain trainable under the existing strict Stage-1 ownership contract.
- LIBERO keeps sample-major `main,wrist` rows, dual-camera checkpoint metadata, and all existing tests unchanged.
- Effective global batch is exactly 128. ACD1-58 uses eight H100 80GB GPUs and starts with per-device batch 2, accumulation 8.
- Production uses predecoded HDF5, BF16, learning rate `1e-5`, step-20 probe checkpoint, saves every 5,000 steps, and 30,000 total optimizer steps.
- Existing `runtime/` files are never committed.

---

### Task 1: Make the Baton training camera axis batch-driven

**Files:**
- Modify: `qwen35_baton/data.py`
- Modify: `qwen35_baton/model.py`
- Modify: `qwen35_baton/teacher.py`
- Modify: `qwen35_baton/config.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Test: `tests/test_qwen35_baton_data.py`
- Test: `tests/test_qwen35_baton_model.py`
- Test: `tests/test_qwen35_baton_teacher.py`
- Test: `tests/test_qwen35_baton_config.py`

**Interfaces:**
- Produces: `BatonPlannerBatch.camera_names: tuple[str, ...]`.
- Produces: `BatonPlannerCollator(processor, *, camera_names=("main", "wrist"), plan_pad_token_id=None)`.
- Produces: `BatonCheckpointMetadata.example(*, camera_names=("main", "wrist"))`.
- Preserves: LIBERO batch shape `[B,2,4,3,H,W]` and model output `[B,2,4,256,1024]`.
- Adds: WorldArena-compatible batch shape `[B,1,4,3,H,W]` and output `[B,1,4,256,1024]`.

- [ ] **Step 1: Write failing one-camera batch/model/teacher/metadata tests**

```python
def test_collator_builds_one_head_row_per_worldarena_sample() -> None:
    collator = BatonPlannerCollator(_Processor(), camera_names=("head",))
    sample = {
        "current_images": torch.zeros((1, 3, 256, 256), dtype=torch.uint8),
        "future_images": torch.zeros((1, 4, 3, 256, 256), dtype=torch.uint8),
        "instruction": "pick up the green bottle",
        "suite": "worldarena",
    }
    batch = collator([sample, sample])
    assert batch.camera_names == ("head",)
    assert batch.row_labels == ((0, "head"), (1, "head"))
    assert batch.qwen_inputs["input_ids"].shape[0] == 2


def test_model_reshapes_one_head_row_per_sample() -> None:
    batch = _batch(batch_size=2, camera_names=("head",))
    output = _planner().forward(batch)
    assert output.positive.shape == (2, 1, 4, 256, 1024)


def test_teacher_accepts_one_camera_future_rgb() -> None:
    teacher = _teacher()
    features = teacher.encode_future(
        torch.zeros((2, 1, 4, 3, 256, 256), dtype=torch.uint8)
    )
    assert features.shape == (2, 1, 4, 256, 1024)


def test_checkpoint_metadata_accepts_truthful_head_camera_shape() -> None:
    metadata = BatonCheckpointMetadata.example(camera_names=("head",))
    assert metadata.camera_names == ("head",)
    assert metadata.target_shape == (1, 4, 256, 1024)
    assert BatonCheckpointMetadata.from_dict(metadata.to_dict()) == metadata
```

- [ ] **Step 2: Run the new tests and verify fixed dual-camera assumptions fail**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_data.py -k one_head \
  tests/test_qwen35_baton_model.py -k one_head \
  tests/test_qwen35_baton_teacher.py -k one_camera \
  tests/test_qwen35_baton_config.py -k head_camera
```

Expected: failures because the batch, model, teacher, and metadata require exactly two cameras.

- [ ] **Step 3: Add a validated camera contract to the batch and collator**

```python
@dataclass(frozen=True)
class BatonPlannerBatch:
    qwen_inputs: Mapping[str, torch.Tensor]
    plan_positions: torch.Tensor
    current_images: torch.Tensor
    future_images: torch.Tensor
    instructions: tuple[str, ...]
    row_labels: tuple[tuple[int, str], ...]
    camera_names: tuple[str, ...] = ("main", "wrist")


class BatonPlannerCollator:
    def __init__(
        self,
        processor: Any,
        *,
        camera_names: tuple[str, ...] = ("main", "wrist"),
        plan_pad_token_id: int | None = None,
    ) -> None:
        if (
            not camera_names
            or any(not isinstance(name, str) or not name for name in camera_names)
            or len(set(camera_names)) != len(camera_names)
        ):
            raise ValueError("camera_names must contain unique nonempty strings")
        self.camera_names = camera_names
```

Validate sample tensors against `len(self.camera_names)`, create rows by enumerating `self.camera_names`, and store the tuple in the returned batch.

- [ ] **Step 4: Generalize model, teacher, metrics, and checkpoint metadata without changing defaults**

```python
# model.py
camera_count = len(batch.camera_names)
expected_labels = tuple(
    (sample, camera)
    for sample in range(batch_size)
    for camera in batch.camera_names
)
if batch.row_labels != expected_labels:
    raise ValueError("Baton rows must be sample-major in camera_names order")
positive = row_output.flat.reshape(
    batch_size, camera_count, _NUM_FRAMES, _TOKENS_PER_FRAME, _FEATURE_DIM
)

# teacher.py encode_future
if images.ndim != 6 or images.shape[1] <= 0 or images.shape[2:4] != (4, 3):
    raise ValueError("future images must be [B,C,4,3,H,W]")
batch_size, camera_count = images.shape[:2]
return self._encode_frames(images.reshape(-1, *images.shape[-3:])).reshape(
    batch_size, camera_count, 4, 256, 1024
)

# config.py
@classmethod
def example(
    cls, *, camera_names: tuple[str, ...] = ("main", "wrist")
) -> "BatonCheckpointMetadata":
    target_shape = (len(camera_names), 4, 256, 1024)
```

Metadata validation requires either `("main", "wrist")` or `("head",)` and requires `target_shape == (len(camera_names),4,256,1024)`. `_loss_metrics` receives `camera_names` from the batch rather than a hardcoded tuple.

- [ ] **Step 5: Run focused and legacy regressions**

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_data.py \
  tests/test_qwen35_baton_model.py \
  tests/test_qwen35_baton_teacher.py \
  tests/test_qwen35_baton_config.py \
  tests/test_qwen35_baton_training.py
```

Expected: all tests pass, including unchanged dual-camera assertions.

- [ ] **Step 6: Commit**

```bash
git add qwen35_baton/data.py qwen35_baton/model.py qwen35_baton/teacher.py \
  qwen35_baton/config.py qwen35_baton/cli/train_semantic_planner.py \
  tests/test_qwen35_baton_data.py tests/test_qwen35_baton_model.py \
  tests/test_qwen35_baton_teacher.py tests/test_qwen35_baton_config.py
git commit -m "feat(baton): support one-camera training batches"
```

### Task 2: Add WorldArena manifest, HDF5 predecode, and dataset

**Files:**
- Create: `qwen35_baton/worldarena_data.py`
- Create: `qwen35_baton/cli/predecode_worldarena.py`
- Create: `tests/test_qwen35_baton_worldarena_data.py`

**Interfaces:**
- Produces: `WorldArenaRecord` and `load_worldarena_source_manifest(path, dataset_root)`.
- Produces: `canonical_source_frame_indices(source_frame_count) -> tuple[int, ...]` with exactly 121 endpoint-covering indices.
- Produces: `future_frame_indices(current_index, frame_count=121) -> tuple[int, int, int, int]`.
- Produces: `WorldArenaMP4Dataset(records, *, seed, split="train")` for correctness comparison.
- Produces: `WorldArenaHDF5Dataset(manifest_path, *, seed, split="train")`.
- Produces: `predecode_worldarena(records, *, output_root, seed, validation_fraction=0.1) -> Path` returning the published manifest path.
- Produces: CLI `python -m qwen35_baton.cli.predecode_worldarena --dataset-root ... --output-root ... --seed 42 --validation-fraction 0.1`.
- HDF5 dataset: `rgb [T,256,256,3] uint8`, chunked per frame with LZF.

- [ ] **Step 1: Write failing manifest, leakage, sampling, and cache tests**

```python
def test_future_indices_are_strict_unique_and_cover_remaining_horizon() -> None:
    assert future_frame_indices(0) == (30, 60, 90, 120)
    assert future_frame_indices(116) == (117, 118, 119, 120)
    for current in range(117):
        future = future_frame_indices(current)
        assert current < future[0] < future[1] < future[2] < future[3] <= 120


def test_source_manifest_rejects_official_episode_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata_train_a2v.jsonl"
    manifest.write_text(json.dumps({
        "video": "official_episodes/task/episode0/video.mp4",
        "prompt": "pick up the cup",
    }) + "\n")
    with pytest.raises(ValueError, match="official"):
        load_worldarena_source_manifest(manifest, tmp_path)


def test_source_manifest_localizes_stale_training_data_prefix(tmp_path: Path) -> None:
    local = tmp_path / "episodes/task__episode0/actions_16d.npy"
    local.parent.mkdir(parents=True)
    local.touch()
    resolved = localize_source_path(
        "/mnt/afs/user/WorldArena/training_data/episodes/task__episode0/actions_16d.npy",
        dataset_root=tmp_path,
        required=True,
    )
    assert resolved == local.resolve()


def test_hdf5_dataset_returns_one_head_camera_and_metadata(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path, frame_count=121)
    dataset = WorldArenaHDF5Dataset(manifest, seed=42, split="train")
    sample = dataset[0]
    assert sample["current_images"].shape == (1, 3, 256, 256)
    assert sample["future_images"].shape == (1, 4, 3, 256, 256)
    assert sample["camera_names"] == ("head",)
    assert sample["instruction"] == "pick up the cup"
    assert sample["source_indices"][0] < sample["source_indices"][1]
    assert "actions_16d_path" in sample["metadata"]


def test_hdf5_selected_rgb_matches_online_mp4_decode(tmp_path: Path) -> None:
    records = _write_source_episode(tmp_path, frame_count=137)
    online = WorldArenaMP4Dataset(records, seed=42, split="validation")
    manifest = predecode_worldarena(records, output_root=tmp_path / "cache", seed=42)
    cached = WorldArenaHDF5Dataset(manifest, seed=42, split="validation")
    assert cached[0]["source_indices"] == online[0]["source_indices"]
    torch.testing.assert_close(cached[0]["current_images"], online[0]["current_images"])
    torch.testing.assert_close(cached[0]["future_images"], online[0]["future_images"])
```

- [ ] **Step 2: Verify the tests fail because the module is absent**

```bash
PYTHONPATH=. pytest -q tests/test_qwen35_baton_worldarena_data.py
```

Expected: collection failure for missing `qwen35_baton.worldarena_data`.

- [ ] **Step 3: Implement strict source manifest and sampling primitives**

```python
@dataclass(frozen=True)
class WorldArenaRecord:
    episode_id: str
    task_name: str
    instruction: str
    video_path: Path
    actions_16d_path: Path | None
    intrinsic_path: Path | None
    extrinsic_path: Path | None


def future_frame_indices(
    current_index: int, frame_count: int = 121
) -> tuple[int, int, int, int]:
    if frame_count < 5 or current_index < 0 or current_index > frame_count - 5:
        raise ValueError("current index must leave four strictly future frames")
    last = frame_count - 1
    future = tuple(
        current_index + round((step + 1) * (last - current_index) / 4)
        for step in range(4)
    )
    if len(set(future)) != 4 or tuple(sorted(future)) != future:
        raise ValueError("normalized future indices must be unique and ordered")
    return future
```

Manifest loading resolves every relative path under `dataset_root`, requires `video` and nonblank `prompt`, rejects any resolved path containing an `official_episodes` component, and derives `episode_id` from the episode directory. For absolute metadata paths from the publisher's `/mnt/afs/.../training_data/` tree, `localize_source_path` discards everything through the `training_data` component and resolves the remaining suffix under `dataset_root`; it rejects absolute paths without that component and required localized files that do not exist. The task name is the episode directory prefix before `__episode`.

- [ ] **Step 4: Implement deterministic predecode and atomic manifest publication**

The CLI reads `metadata_train_a2v.jsonl`, decodes every video with OpenCV, rejects videos with zero decodable frames, and normalizes every actual source length `N >= 1` to the canonical 121-frame timeline using `canonical_to_source[i] = round(i * (N - 1) / 120)`. This contract is required by the downloaded release: all 509 MP4 lengths differ from 121 (observed 76–787), although all 509 metadata/action records declare 121. `WorldArenaMP4Dataset` probes actual decodable `N` rather than trusting `CAP_PROP_FRAME_COUNT`. Convert selected BGR frames to RGB, resize to `256x256` with `INTER_AREA`, and write one temporary HDF5 file before `os.replace`:

```python
with h5py.File(temporary, "w") as handle:
    handle.create_dataset(
        "rgb",
        data=canonical_frames,
        dtype=np.uint8,
        chunks=(1, 256, 256, 3),
        compression="lzf",
    )
```

Assign validation membership by the first eight bytes of `sha256(f"{seed}:{episode_id}")`; publish `manifest.json` containing sorted relative HDF5 paths, source SHA-256 values, split, task, instruction, canonical `frame_count=121`, actual `source_frame_count=N`, and optional metadata paths. Write `stats.json` with train/validation counts, task counts, image size, canonical frame count, per-episode source frame counts, seed, source repository, and manifest SHA-256.

Complete-generation publication is fail-closed: `output_root` must not exist or be empty. Build every shard plus `manifest.json` and `stats.json` in a sibling staging directory on the same filesystem, fsync files and directories, then publish the complete cache with one final `os.replace(staging, output_root)`. Any failure removes staging and leaves the original output untouched; a nonempty existing cache is never overwritten.

- [ ] **Step 5: Implement epoch-deterministic HDF5 loading**

`WorldArenaHDF5Dataset` stores the epoch in shared memory. For each sample, select current frame with a local SHA-256 seed of `(seed, epoch, episode_id)` in training and `(seed, episode_id)` in validation, read only the five indexed frames inside a context-managed HDF5 handle, transpose RGB to CHW, and return one leading head-camera axis.

`WorldArenaMP4Dataset` shares the same canonical index-selection helper and output schema, maps those five canonical indices through the same `canonical_to_source` rule, and decodes only the resulting source indices from the localized MP4. It exists for tests and one-episode smoke comparison; the production trainer selects HDF5 only.

- [ ] **Step 6: Run tests and a real one-video predecode smoke**

```bash
PYTHONPATH=. pytest -q tests/test_qwen35_baton_worldarena_data.py
PYTHONPATH=. python -m qwen35_baton.cli.predecode_worldarena \
  --dataset-root /path/to/worldarena2026-robotwin-data \
  --output-root /tmp/worldarena-cache-smoke \
  --seed 42 --validation-fraction 0.1 --limit 1
```

Expected: tests pass; smoke manifest contains one record and its HDF5 RGB shape is `[121,256,256,3]`.

- [ ] **Step 7: Commit**

```bash
git add qwen35_baton/worldarena_data.py qwen35_baton/cli/predecode_worldarena.py \
  tests/test_qwen35_baton_worldarena_data.py
git commit -m "feat(baton): add WorldArena HDF5 data adapter"
```

### Task 3: Route WorldArena through the common Stage-1 trainer

**Files:**
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Modify: `qwen35_baton/cli/preflight.py`
- Create: `qwen35_baton/configs/worldarena_stage1.json`
- Create: `qwen35_baton/scripts/train_worldarena_semantic_planner.sh`
- Modify: `tests/test_qwen35_baton_training.py`
- Modify: `tests/test_qwen35_baton_config.py`

**Interfaces:**
- Adds config: `dataset_type: str = "libero_hdf5"`, allowed values `libero_hdf5` and `worldarena_hdf5`.
- Adds CLI: `--stop-at-step` and launcher environment `STOP_AT_STEP`, used only for bounded probes.
- Selects `WorldArenaHDF5Dataset` plus `BatonPlannerCollator(..., camera_names=("head",))` only for `worldarena_hdf5`.
- Produces: production recipe with global batch 128 and the same model/provenance paths as strict Baton.

- [ ] **Step 1: Write failing config and artifact-routing tests**

```python
def test_stage1_accepts_only_explicit_dataset_types(tmp_path: Path) -> None:
    assert replace(_config(tmp_path), dataset_type="worldarena_hdf5").dataset_type == "worldarena_hdf5"
    with pytest.raises(ValueError, match="dataset_type"):
        replace(_config(tmp_path), dataset_type="worldarena")


def test_worldarena_artifacts_use_one_head_camera(monkeypatch, tmp_path: Path) -> None:
    config = replace(_config(tmp_path), dataset_type="worldarena_hdf5")
    artifacts = load_local_artifacts(config)
    batch = next(iter(artifacts.train_batches))
    assert batch.camera_names == ("head",)
    assert batch.future_images.shape[1:3] == (1, 4)
    assert artifacts.metadata.camera_names == ("head",)


def test_cli_stop_at_step_is_forwarded_to_training(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(training_module, "run_training", lambda config, stop_at_step=None: captured.update(stop_at_step=stop_at_step) or _result())
    assert training_module.main(["--config", str(_config_json(tmp_path)), "--stop-at-step", "20"]) == 0
    assert captured["stop_at_step"] == 20
```

- [ ] **Step 2: Verify routing tests fail**

```bash
PYTHONPATH=. pytest -q tests/test_qwen35_baton_training.py -k worldarena
```

Expected: failure because `dataset_type` and WorldArena routing do not exist.

- [ ] **Step 3: Add explicit dataset routing and truthful metadata**

```python
if config.dataset_type == "libero_hdf5":
    dataset = BatonLiberoDataset(base_dataset, seed=config.seed)
    camera_names = ("main", "wrist")
elif config.dataset_type == "worldarena_hdf5":
    dataset = WorldArenaHDF5Dataset(
        config.hdf5_manifest_path, seed=config.seed, split="train"
    )
    camera_names = ("head",)
else:
    raise AssertionError("validated dataset_type is unreachable")

collator = BatonPlannerCollator(processor, camera_names=camera_names)
metadata = BatonCheckpointMetadata.example(camera_names=camera_names)
```

The preflight parses the manifest JSON, requires its declared dataset type to match the config, validates the existing manifest SHA-256, and includes `dataset_type` and `camera_names` in its report. LIBERO continues to require its statistics file; WorldArena requires its cache `stats.json`.

- [ ] **Step 4: Add recipe and launcher**

`worldarena_stage1.json` contains `dataset_type="worldarena_hdf5"`, `per_device_batch=2`, `gradient_accumulation_steps=8`, `max_steps=30000`, `initial_save_step=20`, `save_every=5000`, `num_workers=8`, `persistent_workers=true`, and `worker_restart_interval_epochs=100`.

`train_worldarena_semantic_planner.sh` delegates to the common launcher with:

```bash
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export CONFIG="${CONFIG:-qwen35_baton/configs/worldarena_stage1.json}"
exec "${SCRIPT_DIR}/train_semantic_planner.sh" "${CONFIG}"
```

The common launcher builds an optional array only when `STOP_AT_STEP` is nonempty:

```bash
stop_args=()
if [[ -n "${STOP_AT_STEP:-}" ]]; then
  stop_args=(--stop-at-step "${STOP_AT_STEP}")
fi
```

The Python parser accepts a positive integer `--stop-at-step`, passes it to `run_training`, and keeps the default `None` for ordinary 30,000-step runs.

- [ ] **Step 5: Run all Baton regressions**

```bash
PYTHONPATH=. pytest -q tests/test_qwen35_baton_*.py
bash -n qwen35_baton/scripts/train_worldarena_semantic_planner.sh
python -m compileall -q qwen35_baton
git diff --check
```

Expected: all Baton tests pass and legacy LIBERO recipe assertions remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add qwen35_baton/cli/train_semantic_planner.py qwen35_baton/cli/preflight.py \
  qwen35_baton/configs/worldarena_stage1.json \
  qwen35_baton/scripts/train_semantic_planner.sh \
  qwen35_baton/scripts/train_worldarena_semantic_planner.sh \
  tests/test_qwen35_baton_training.py tests/test_qwen35_baton_config.py
git commit -m "feat(baton): train Qwen3.5 on WorldArena cache"
```

### Task 4: Deploy and launch on ACD1-58

**Files:**
- Create remotely: `/data/user/jhe724/workspace/VLM4WAM_qwen35_worldarena/`
- Create remotely: `/data/user/jhe724/junjie/datasets/worldarena2026-robotwin-hdf5/`
- Create remotely: `/data/user/jhe724/workspace/VLM4WAM_qwen35_worldarena/runtime/acd1_58_worldarena_30k.json`
- Produce remotely: `/data/user/jhe724/junjie/outputs/qwen35_worldarena_acd1_58_30k/`

**Interfaces:**
- Uses Slurm allocation `456990` on ACD1-58 with eight H100 80GB GPUs.
- Uses `/data/user/jhe724/.conda/envs/qwen35/bin/python`.
- Uses local Qwen artifact `/data/user/jhe724/junjie/weights/Qwen3.5-2B-baton-v1` and SigLIP2 artifact `/data/user/jhe724/junjie/weights/siglip2-large-patch16-256`.

- [ ] **Step 1: Finish data download and sync source data to HPC3**

```bash
hf download DavidxWang/worldarena2026-robotwin-data --repo-type dataset \
  --local-dir /data/LFT-W02_data/junjie/datasets/worldarena2026-robotwin-data
rsync -a --info=progress2 \
  /data/LFT-W02_data/junjie/datasets/worldarena2026-robotwin-data/ \
  HPC3_jhe724:/data/user/jhe724/junjie/datasets/worldarena2026-robotwin-data/
```

Verify exactly 509 `episodes/*/video.mp4` files and keep `official_episodes` outside the training manifest.

- [ ] **Step 2: Sync code without runtime or outputs**

```bash
rsync -a --delete \
  --exclude='.git/' --exclude='runtime/' --exclude='outputs/' \
  /data/LFT-W02_data/junjie/workspace/VLM4WAM/.worktrees/qwen35-video-hindsight-grounding/ \
  HPC3_jhe724:/data/user/jhe724/workspace/VLM4WAM_qwen35_worldarena/
```

- [ ] **Step 3: Predecode all training episodes on ACD1-58**

```bash
python -m qwen35_baton.cli.predecode_worldarena \
  --dataset-root /data/user/jhe724/junjie/datasets/worldarena2026-robotwin-data \
  --output-root /data/user/jhe724/junjie/datasets/worldarena2026-robotwin-hdf5 \
  --seed 42 --validation-fraction 0.1
```

Verify 509 records total, no official path, all HDF5 shapes `[121,256,256,3]`, and record the manifest SHA-256 in the runtime config.

- [ ] **Step 4: Run CPU preflight and one-GPU finite-step smoke**

Materialize the runtime JSON with real artifact paths and hashes. Run preflight for world size 8, then launch one process with a temporary `max_steps=1`, `initial_save_step=None`, `tiny_test=true` smoke fixture using per-device batch 2. Require finite loss, a completed optimizer step, and no trainable SigLIP2 parameters.

- [ ] **Step 5: Run the eight-GPU step-20 checkpoint probe**

Launch detached session `qwen35_worldarena_probe`:

```bash
NUM_GPUS=8 PER_DEVICE_BATCH=2 GRAD_ACCUM=8 \
PYTHON_BIN=/data/user/jhe724/.conda/envs/qwen35/bin/python \
CONFIG=runtime/acd1_58_worldarena_30k.json \
STOP_AT_STEP=20 bash qwen35_baton/scripts/train_worldarena_semantic_planner.sh
```

Confirm the launcher exits normally at step 20, `step_000020` is atomically published, metrics are finite, all eight ranks agree on cursor state, and no OOM occurs.

- [ ] **Step 6: Start or continue the 30,000-step run**

After the probe, set `resume_from` to `step_000020` and launch detached session `qwen35_worldarena_30k` on all eight GPUs. Verify within the first five optimizer steps after resume:

```text
global_batch=128
camera_names=[head]
prediction/target=[B,1,4,256,1024]
loss finite
8 GPU processes alive
GPU utilization nonzero on every device
```

- [ ] **Step 7: Record launch evidence**

Report the node, allocation, session name, output directory, log path, exact global batch, current step, samples/s, GPU memory/power, checkpoint cadence, and estimated completion time.
