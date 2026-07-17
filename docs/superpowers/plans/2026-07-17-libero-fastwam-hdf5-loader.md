# LIBERO-FastWAM Sharded HDF5 Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional fixed-contract, sharded HDF5 backend for LIBERO-FastWAM and validate it with a 64-episode correctness/performance pilot without changing or deleting the existing loader.

**Architecture:** A standalone converter writes immutable 32-episode HDF5 shards plus an atomic manifest. A separate dataset class lazily opens and reuses read-only shard handles per DataLoader worker while preserving the current frame/action sampling and output contract. Separate configuration, preflight, launcher, and benchmark entry points keep the existing NumPy/MP4 path untouched.

**Tech Stack:** Python 3.10, PyTorch/torchvision, h5py 3.12, NumPy, pandas/pyarrow, PyYAML, pytest, Bash, Accelerate/GE-Act.

## Global Constraints

- Do not delete, rename, wrap, or change the behavior of `ge_act/data/lerobot_like_dataset.py` or its existing YAML files.
- Camera order is exactly `main=0`, `wrist=1`, sourced from `observation.images.image` then `observation.images.wrist_image`.
- RGB is stored as lossless uint8 at 256x256; no JPEG or other lossy codec.
- Fixed temporal contract: source FPS 20, `n_previous=4`, future `chunk=9`, `action_chunk=36`, video stride 4.
- Fixed action contract: absolute EEF actions and the existing LIBERO normalization statistics.
- Shards contain at most 32 episodes and are immutable after atomic publication.
- DataLoader workers open shards lazily, reuse handles, never glob in `__getitem__`, and never inherit live HDF5 handles through pickle/fork.
- The original NumPy loader remains the fallback and no existing config selects HDF5.
- Pilot scope is 64 episodes in both `none` and `lzf` compression modes.
- Correctness requires exact identity/order/action/state/caption parity and normalized RGB max error `<= 1/255 + 1e-6`.
- Full conversion is forbidden until pilot DataLoader throughput is at least 1.5x, aggregate worker RSS increases by at most 25%, and a 200-step model smoke has no throughput regression.
- Production code is written only after its focused test fails for the intended reason.

---

### Task 1: Versioned manifest and shard validation

**Files:**
- Create: `ge_act/data/libero_fastwam_hdf5_schema.py`
- Create: `tests/test_libero_fastwam_hdf5.py`

**Interfaces:**
- Produces: `EpisodeRecord`, `load_manifest(path)`, `validate_manifest(payload, root)`, `validate_episode_group(group, record)`, and `atomic_write_manifest(path, payload)`.
- Consumes later: converter publishes exactly this schema; dataset and preflight reject everything else.

- [ ] **Step 1: Write failing manifest tests**

```python
def test_manifest_accepts_fixed_libero_contract(tmp_path):
    payload = make_manifest(tmp_path, camera_names=["main", "wrist"])
    records = schema.validate_manifest(payload, tmp_path)
    assert records[0].key == "libero_goal:000010"
    assert records[0].shard_path == tmp_path / "shard_00000.h5"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_names", ["wrist", "main"], "camera_names"),
        ("image_size", [512, 512], "image_size"),
        ("source_fps", 30, "source_fps"),
        ("n_previous", 3, "n_previous"),
        ("chunk", 8, "chunk"),
        ("action_chunk", 32, "action_chunk"),
    ],
)
def test_manifest_rejects_wrong_fixed_contract(tmp_path, field, value, message):
    payload = make_manifest(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_duplicate_episode_keys(tmp_path):
    payload = make_manifest(tmp_path)
    payload["episodes"].append(dict(payload["episodes"][0]))
    with pytest.raises(ValueError, match="duplicate episode key"):
        schema.validate_manifest(payload, tmp_path)
```

- [ ] **Step 2: Verify RED**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k manifest`

Expected: collection fails because `ge_act.data.libero_fastwam_hdf5_schema` does not exist.

- [ ] **Step 3: Implement the schema boundary**

```python
SCHEMA_VERSION = 1
FIXED_CONTRACT = {
    "camera_names": ["main", "wrist"],
    "image_size": [256, 256],
    "source_fps": 20,
    "n_previous": 4,
    "chunk": 9,
    "action_chunk": 36,
    "action_type": "absolute",
    "action_space": "eef",
}


@dataclass(frozen=True)
class EpisodeRecord:
    key: str
    shard_path: Path
    group: str
    domain: str
    episode_index: int
    length: int


def validate_manifest(payload: dict[str, Any], root: Path) -> list[EpisodeRecord]:
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for field, expected in FIXED_CONTRACT.items():
        if payload.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}, got {payload.get(field)!r}")
    compression = payload.get("compression")
    if compression not in ("none", "lzf"):
        raise ValueError("compression must be 'none' or 'lzf'")
    raw_records = payload.get("episodes")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("episodes must be a non-empty list")
    seen: set[str] = set()
    records = []
    for item in raw_records:
        key = str(item["key"])
        if key in seen:
            raise ValueError(f"duplicate episode key: {key}")
        seen.add(key)
        shard_path = root / str(item["shard"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing HDF5 shard: {shard_path}")
        length = int(item["length"])
        if length < 2:
            raise ValueError(f"episode {key} has invalid length {length}")
        records.append(EpisodeRecord(
            key=key,
            shard_path=shard_path,
            group=str(item["group"]),
            domain=str(item["domain"]),
            episode_index=int(item["episode_index"]),
            length=length,
        ))
    return records
```

Implement `validate_episode_group` to require `rgb_main/rgb_wrist` uint8
`[T,256,256,3]`, float32 action/state with the same `T`, matching scalar
metadata, and `T == record.length`. Implement `atomic_write_manifest` using a
same-directory temporary file, `flush`, `os.fsync`, and `os.replace`.

- [ ] **Step 4: Verify GREEN**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k manifest`

Expected: all manifest/schema tests pass.

- [ ] **Step 5: Commit**

```bash
git add ge_act/data/libero_fastwam_hdf5_schema.py tests/test_libero_fastwam_hdf5.py
git commit -m "feat(data): define LIBERO HDF5 shard schema"
```

---

### Task 2: Atomic episode-to-shard converter

**Files:**
- Create: `ge_act/scripts/convert_libero_fastwam_hdf5.py`
- Modify: `tests/test_libero_fastwam_hdf5.py`

**Interfaces:**
- Consumes: LeRobot `meta/*.jsonl`, episode parquet, and strict predecoded camera `.npy` files.
- Produces: `discover_source_episodes(...)`, `resize_rgb_uint8(...)`, `convert_dataset(args)`, immutable shards, and schema-v1 manifest.

- [ ] **Step 1: Add failing converter tests**

```python
def test_resize_rgb_uint8_matches_float_resize_with_quantization_bound():
    frames = synthetic_rgb_frames(t=5, height=8, width=10)
    actual = converter.resize_rgb_uint8(frames, size=(256, 256), microbatch=2)
    reference = torchvision.transforms.functional.resize(
        torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0,
        [256, 256],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    restored = torch.from_numpy(actual).permute(0, 3, 1, 2).float() / 255.0
    assert actual.shape == (5, 256, 256, 3)
    assert actual.dtype == np.uint8
    assert (restored - reference).abs().max() <= 0.5 / 255.0 + 1e-6


@pytest.mark.parametrize("compression", ["none", "lzf"])
def test_converter_writes_two_camera_atomic_shards_and_manifest(tmp_path, compression):
    source = make_tiny_lerobot_source(tmp_path / "source", episodes=3)
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source)
    output = tmp_path / f"out-{compression}"
    report = converter.convert_dataset(make_convert_args(
        source=source, cache=cache, output=output,
        max_episodes=3, episodes_per_shard=2, compression=compression,
    ))
    assert report == {"episodes": 3, "shards": 2, "compression": compression}
    payload, records = schema.load_manifest(output / "manifest.json")
    assert [record.episode_index for record in records] == [0, 1, 2]
    assert not list(output.glob("*.tmp"))


def test_converter_can_decode_source_video_when_predecoded_cache_is_absent(tmp_path):
    source = make_tiny_lerobot_source(tmp_path / "source", episodes=1, write_mp4=True)
    output = tmp_path / "out"
    converter.convert_dataset(make_convert_args(
        source=source, cache=None, output=output,
        max_episodes=1, episodes_per_shard=32, compression="none",
    ))
    _, records = schema.load_manifest(output / "manifest.json")
    assert len(records) == 1
```

- [ ] **Step 2: Verify RED**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k 'resize_rgb or converter'`

Expected: import fails because the converter module does not exist.

- [ ] **Step 3: Implement deterministic source discovery and resize**

```python
@dataclass(frozen=True)
class SourceEpisode:
    key: str
    domain: str
    episode_index: int
    length: int
    caption: str
    parquet_path: Path
    main_cache_path: Path
    wrist_cache_path: Path


def resize_rgb_uint8(frames: np.ndarray, *, size=(256, 256), microbatch=16) -> np.ndarray:
    validate_source_rgb(frames)
    outputs = []
    for start in range(0, len(frames), microbatch):
        tensor = torch.from_numpy(np.asarray(frames[start:start + microbatch])).permute(0, 3, 1, 2)
        resized = torchvision.transforms.functional.resize(
            tensor.float() / 255.0,
            list(size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        outputs.append((resized * 255.0).round().clamp_(0, 255).byte().permute(0, 2, 3, 1).numpy())
    return np.concatenate(outputs, axis=0)
```

`discover_source_episodes` reads domains in CLI order and episodes in numeric
episode-index order, verifies the fixed LIBERO source has exactly one task per
episode (the deployed dataset has 1,712/1,712 single-task episodes), resolves
the caption from `tasks.jsonl`, verifies parquet, and emits unique keys
`<domain>:<episode_index:06d>`. Repeated source roots do not duplicate episodes.
When `--predecoded-root` is present, require and read both camera `.npy` files;
when it is absent, decode both source MP4s sequentially with PyAV. A missing
camera in either mode fails the episode rather than falling back silently.

- [ ] **Step 4: Implement atomic shard publication**

For each group, write datasets with:

```python
rgb_kwargs = {
    "chunks": (1, 256, 256, 3),
    "compression": None if args.compression == "none" else "lzf",
}
group.create_dataset("rgb_main", data=main_rgb, dtype=np.uint8, **rgb_kwargs)
group.create_dataset("rgb_wrist", data=wrist_rgb, dtype=np.uint8, **rgb_kwargs)
group.create_dataset("action", data=action, dtype=np.float32,
                     chunks=(min(64, len(action)), action.shape[1]))
group.create_dataset("state", data=state, dtype=np.float32,
                     chunks=(min(64, len(state)), state.shape[1]))
string_dtype = h5py.string_dtype(encoding="utf-8")
group.create_dataset("caption", data=episode.caption, dtype=string_dtype)
group.create_dataset("domain", data=episode.domain, dtype=string_dtype)
group.create_dataset("episode_index", data=episode.episode_index, dtype=np.int64)
group.create_dataset("length", data=episode.length, dtype=np.int64)
```

Write each `shard_N.h5.tmp.<pid>`, reopen read-only, validate every group, flush
and fsync the underlying file before `os.replace`. Publish `manifest.json` only
after all shards validate. Existing matching shards are skipped; an existing
manifest with a different source/config fingerprint fails unless `--overwrite`
is passed.

- [ ] **Step 5: Verify GREEN and CLI syntax**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k 'resize_rgb or converter'
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m ge_act.scripts.convert_libero_fastwam_hdf5 --help
```

Expected: tests pass and CLI exits 0.

- [ ] **Step 6: Commit**

```bash
git add ge_act/scripts/convert_libero_fastwam_hdf5.py tests/test_libero_fastwam_hdf5.py
git commit -m "feat(data): convert LIBERO episodes to HDF5 shards"
```

---

### Task 3: Independent persistent-handle HDF5 dataset

**Files:**
- Create: `ge_act/data/libero_fastwam_hdf5_dataset.py`
- Modify: `tests/test_libero_fastwam_hdf5.py`

**Interfaces:**
- Consumes: schema-v1 manifest and the existing LIBERO statistics JSON.
- Produces: `LiberoFastWAMHDF5Dataset`, returning `video`, `actions`, `caption`, and `state` with the same shapes/dtypes as the original loader.

- [ ] **Step 1: Add failing reader tests**

```python
def test_hdf5_dataset_returns_main_wrist_video_and_normalized_control(tmp_path):
    manifest, expected = make_reader_fixture(tmp_path)
    dataset = LiberoFastWAMHDF5Dataset(
        manifest_path=manifest,
        stat_file=expected.stat_file,
        fix_epiidx=0,
        fix_sidx=12,
        fix_mem_idx=[1, 4, 8, 11],
    )
    sample = dataset[0]
    assert sample["video"].shape == (3, 2, 13, 256, 256)
    assert sample["actions"].shape[0] == 40
    assert sample["state"].shape[0] == 1
    assert sample["caption"] == expected.caption
    torch.testing.assert_close(sample["video"][:, 0], expected.main_normalized)
    torch.testing.assert_close(sample["video"][:, 1], expected.wrist_normalized)


def test_worker_pickle_drops_live_hdf5_handles(tmp_path):
    dataset = make_open_reader(tmp_path, max_open_shards=2)
    dataset[0]
    assert len(dataset._handles) == 1
    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored._handles) == 0


def test_reader_lru_reuses_and_bounds_shard_handles(tmp_path):
    dataset = make_three_shard_reader(tmp_path, max_open_shards=2)
    dataset[0]
    first = dataset._handles[next(iter(dataset._handles))]
    dataset[0]
    assert dataset._handles[next(iter(dataset._handles))] is first
    dataset[1]
    dataset[2]
    assert len(dataset._handles) == 2
```

- [ ] **Step 2: Verify RED**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k 'dataset or handle or reader'`

Expected: import fails because the independent dataset module does not exist.

- [ ] **Step 3: Implement lazy bounded handles**

```python
def __getstate__(self):
    state = dict(self.__dict__)
    state["_handles"] = OrderedDict()
    return state


def _get_handle(self, path: Path) -> h5py.File:
    handle = self._handles.pop(path, None)
    if handle is None:
        try:
            handle = h5py.File(path, "r")
        except OSError as error:
            info = get_worker_info()
            worker = info.id if info is not None else "main"
            raise OSError(f"worker={worker} cannot open HDF5 shard {path}") from error
    self._handles[path] = handle
    while len(self._handles) > self.max_open_shards:
        _, evicted = self._handles.popitem(last=False)
        evicted.close()
    return handle
```

Implement `close()` and a best-effort `__del__`. Constructor loads and validates
the manifest only; it opens no HDF5 file.

- [ ] **Step 4: Implement fixed sampling and output contract**

Copy the current fixed `get_frame_indexes` behavior into this independent class,
including validation-mode fixed indexes, random history selection, future stride,
and clipping. Add `read_by_indexes(index, frame_indexes, action_indexes)` for
deterministic comparison. It must:

```python
main = np.asarray(group["rgb_main"][np.asarray(frame_indexes, dtype=np.int64)])
wrist = np.asarray(group["rgb_wrist"][np.asarray(frame_indexes, dtype=np.int64)])
video = torch.from_numpy(np.stack([main, wrist], axis=0)).permute(4, 0, 1, 2, 3).float() / 255.0
video = (video - 0.5) / 0.5
action = torch.from_numpy(np.asarray(group["action"][action_indexes], dtype=np.float32))
state_seq = torch.from_numpy(np.asarray(group["state"][action_indexes], dtype=np.float32))
state = state_seq[self.n_previous - 1:self.n_previous]
action = (action - self.action_mean[record.domain]) / self.action_std[record.domain]
state = (state - self.state_mean[record.domain]) / self.state_std[record.domain]
return {"video": video, "actions": action, "caption": read_utf8(group["caption"]), "state": state}
```

Because h5py fancy indexes must be increasing and unique, implement
`read_rows_preserving_order(dataset, indexes)` by reading sorted unique clipped
indexes once and gathering back to the original repeated/order layout. Reader
exceptions include worker, shard, episode key, and requested indexes; it never
silently samples a different episode.

- [ ] **Step 5: Verify GREEN and original-loader regression**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_ge_act_source_completeness.py tests/test_ge_act_predecode_videos.py
```

Expected: all pass; original loader tests remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add ge_act/data/libero_fastwam_hdf5_dataset.py tests/test_libero_fastwam_hdf5.py
git commit -m "feat(data): load LIBERO from persistent HDF5 shards"
```

---

### Task 4: Separate HDF5 configuration, preflight, and launcher

**Files:**
- Create: `ge_act/configs/ltx_model/libero/video_model_libero_fastwam_siglip2_hdf5.yaml`
- Create: `ge_act/scripts/preflight_libero_fastwam_hdf5.py`
- Create: `ge_act/scripts/train_ltx_siglip2_hdf5.sh`
- Modify: `tests/test_libero_fastwam_hdf5.py`

**Interfaces:**
- Consumes: a published HDF5 manifest and existing model/stat paths.
- Produces: an opt-in production config and launch path; existing YAML/launcher remain byte-for-byte unchanged in this task.

- [ ] **Step 1: Add failing config/preflight tests**

```python
def test_hdf5_yaml_selects_only_independent_dataset():
    config = load_yaml(HDF5_CONFIG)
    assert config["train_data_class_path"] == "data/libero_fastwam_hdf5_dataset.py"
    assert config["train_data_class"] == "LiberoFastWAMHDF5Dataset"
    assert config["data"]["train"]["manifest_path"].endswith("manifest.json")
    assert "predecoded_video_root" not in config["data"]["train"]


def test_hdf5_preflight_rejects_manifest_contract_mismatch(tmp_path):
    config = make_hdf5_training_config(tmp_path)
    corrupt_manifest(config, camera_names=["wrist", "main"])
    errors = collect_hdf5_preflight_errors(config, world_size=8, check_paths=True)
    assert any("camera_names" in error for error in errors)


def test_original_yaml_and_loader_source_are_unchanged():
    assert sha256(ORIGINAL_YAML) == "14fd689abc9813cd962886776c5a89c06c036a3920247cb078cca9b84003daad"
    assert sha256(ORIGINAL_LOADER) == "35dbcaa7746344f789d1be26a5b67b323296c4d48702aa260abde51c409261a4"
```

The fixed hashes protect the user's explicit requirement that the old path is
not modified.

- [ ] **Step 2: Verify RED**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k 'yaml or preflight or original'`

Expected: new config/preflight paths do not exist.

- [ ] **Step 3: Add separate YAML and preflight**

Copy the active model/optimizer/semantic settings into the HDF5 YAML, changing
only output/tracker names and the dataset class/data blocks:

```yaml
train_data_class_path: data/libero_fastwam_hdf5_dataset.py
train_data_class: LiberoFastWAMHDF5Dataset
val_data_class_path: data/libero_fastwam_hdf5_dataset.py
val_data_class: LiberoFastWAMHDF5Dataset
data:
  train:
    manifest_path: /data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5/manifest.json
    stat_file: configs/ltx_model/libero/libero_fastwam_mix.json
    source_fps: 20
    sample_n_frames: 500
    valid_cam: [observation.images.image, observation.images.wrist_image]
    chunk: 9
    action_chunk: 36
    n_previous: 4
    previous_pick_mode: random
    action_type: absolute
    action_space: eef
    train_dataset: true
  val:
    manifest_path: /data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5/manifest.json
    stat_file: configs/ltx_model/libero/libero_fastwam_mix.json
    source_fps: 20
    sample_n_frames: 500
    valid_cam: [observation.images.image, observation.images.wrist_image]
    chunk: 9
    action_chunk: 36
    n_previous: 4
    previous_pick_mode: random
    action_type: absolute
    action_space: eef
    train_dataset: false
```

`collect_hdf5_preflight_errors` validates schema via `load_manifest`, train/val
manifest equality, global batch 128, 30k steps, semantic plan settings, required
model/stat paths, free output storage, and the exact fixed data contract. It does
not call the old predecoded-cache preflight.

- [ ] **Step 4: Add opt-in launcher**

`train_ltx_siglip2_hdf5.sh` resolves `CONFIG`, runs the new preflight, then uses
the same Accelerate/DeepSpeed command as `train_ltx_siglip2.sh`. It accepts
`CONFIG`, `NUM_PROCESSES`, and standard distributed environment overrides, and
never edits or invokes the old cache verifier.

- [ ] **Step 5: Verify GREEN and shell syntax**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k 'yaml or preflight or original'
bash -n ge_act/scripts/train_ltx_siglip2_hdf5.sh
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ge_act/configs/ltx_model/libero/video_model_libero_fastwam_siglip2_hdf5.yaml ge_act/scripts/preflight_libero_fastwam_hdf5.py ge_act/scripts/train_ltx_siglip2_hdf5.sh tests/test_libero_fastwam_hdf5.py
git commit -m "feat(data): add opt-in LIBERO HDF5 training config"
```

---

### Task 5: Parity and DataLoader benchmark harness

**Files:**
- Create: `ge_act/scripts/benchmark_libero_fastwam_hdf5.py`
- Modify: `tests/test_libero_fastwam_hdf5.py`

**Interfaces:**
- Consumes: old YAML, HDF5 YAML, deterministic episode/index stream, worker counts, warmup/measurement batches.
- Produces: JSON report containing parity, samples/s, median/p95 batch time, worker RSS, and compression winner.

- [ ] **Step 1: Add failing benchmark tests**

```python
def test_parity_accepts_only_uint8_rounding_bound():
    old = make_old_loader_sample()
    new = quantize_like_hdf5(old)
    report = compare_samples(old, new)
    assert report["exact_fields"] == ["actions", "state", "caption", "shape", "dtype"]
    assert report["max_normalized_rgb_error"] <= 1 / 255 + 1e-6


def test_parity_rejects_camera_swap():
    old = make_old_loader_sample()
    new = dict(old, video=old["video"].flip(1))
    with pytest.raises(AssertionError, match="camera"):
        compare_samples(old, new)


def test_compression_winner_prefers_lzf_within_five_percent():
    assert choose_compression({"none": 100.0, "lzf": 97.0}) == "lzf"
    assert choose_compression({"none": 100.0, "lzf": 80.0}) == "none"
    assert choose_compression({"none": 100.0, "lzf": 140.0}) == "lzf"
```

- [ ] **Step 2: Verify RED**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py -k 'parity or compression_winner'`

Expected: benchmark module/functions do not exist.

- [ ] **Step 3: Implement deterministic parity and throughput modes**

The CLI accepts:

```text
--old-config PATH --hdf5-config PATH --output-json PATH
--mode parity|throughput --episodes 64 --samples 1024
--workers 0 2 4 8 --batch-size 8 --warmup-batches 20 --measure-batches 100
```

Parity maps `(domain, episode_index)` across loaders, uses explicit memory/frame
and action indexes, checks camera order independently, exact action/state/caption,
and the RGB bound. Throughput constructs real DataLoaders with
`persistent_workers=workers>0`, production prefetch, pinned memory, a fixed
sampler, and records per-batch wall time after warmup. Capture aggregate child
RSS with `psutil` when available and `/proc/<pid>/status` otherwise.

```python
def choose_compression(results: dict[str, float]) -> str:
    none = float(results["none"])
    lzf = float(results["lzf"])
    if lzf >= 0.95 * none:
        return "lzf"
    return "none"
```

Atomic JSON output contains environment/host/filesystem, git SHA, loader
arguments, parity result, every worker-count measurement, and selected format.

- [ ] **Step 4: Verify GREEN and complete local suite**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_libero_fastwam_hdf5.py
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_ge_act_source_completeness.py tests/test_ge_act_predecode_videos.py tests/test_ge_act_siglip2_config.py
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m py_compile ge_act/data/libero_fastwam_hdf5_schema.py ge_act/data/libero_fastwam_hdf5_dataset.py ge_act/scripts/convert_libero_fastwam_hdf5.py ge_act/scripts/benchmark_libero_fastwam_hdf5.py ge_act/scripts/preflight_libero_fastwam_hdf5.py
```

Expected: all pass with no new warnings.

- [ ] **Step 5: Commit**

```bash
git add ge_act/scripts/benchmark_libero_fastwam_hdf5.py tests/test_libero_fastwam_hdf5.py
git commit -m "test(data): benchmark LIBERO HDF5 loader parity"
```

---

### Task 6: HPC3 64-episode pilot and go/no-go report

**Files:**
- Generated, not committed: `/data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5-pilot-none/`
- Generated, not committed: `/data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5-pilot-lzf/`
- Generated, not committed: `/data/user/jhe724/junjie/benchmarks/libero-fastwam-hdf5-pilot.json`

**Interfaces:**
- Consumes: reviewed code, HPC3 source/predecoded data, both pilot manifests.
- Produces: measured compression winner and an explicit full-conversion go/no-go; it does not start full conversion automatically.

- [ ] **Step 1: Sync reviewed branch to HPC3 and preflight storage**

Run read-only storage check and require at least 30 GiB free at both pilot and
temporary-output parents. Sync only tracked code; do not copy local outputs,
worktree metadata, or test artifacts.

- [ ] **Step 2: Convert 64 episodes twice**

Run from the GE-Act environment:

```bash
python -m ge_act.scripts.convert_libero_fastwam_hdf5 \
  --data-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam \
  --predecoded-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam-predecoded-rgb \
  --domains libero_10_no_noops_lerobot libero_goal_no_noops_lerobot libero_object_no_noops_lerobot libero_spatial_no_noops_lerobot \
  --output-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5-pilot-none \
  --max-episodes 64 --episodes-per-shard 32 --compression none

python -m ge_act.scripts.convert_libero_fastwam_hdf5 \
  --data-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam \
  --predecoded-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam-predecoded-rgb \
  --domains libero_10_no_noops_lerobot libero_goal_no_noops_lerobot libero_object_no_noops_lerobot libero_spatial_no_noops_lerobot \
  --output-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5-pilot-lzf \
  --max-episodes 64 --episodes-per-shard 32 --compression lzf
```

Expected: each output contains two validated shards and one manifest.

- [ ] **Step 3: Run parity and DataLoader benchmarks**

Run parity against both formats, then throughput for worker counts 0/2/4/8.
Require zero parity failures. Record filesystem type, cache state caveat, median,
p95, samples/s, and RSS. Repeat throughput once after dropping no caches; the
second run is explicitly labeled warm-cache rather than replacing the first.

- [ ] **Step 4: Run 200-step model smoke only for the selected format**

Create a temporary config override pointing at the winning manifest and limiting
training to 200 steps with checkpoint saving disabled. Record DataLoader wait,
step time, GPU utilization/power, peak memory, and loss finiteness. Do not modify
the production YAML.

- [ ] **Step 5: Produce go/no-go**

Report:

- selected compression and storage ratio;
- old vs HDF5 samples/s for every worker count;
- production worker-count recommendation;
- aggregate worker RSS delta;
- 200-step old vs HDF5 throughput and GPU utilization;
- correctness status;
- `GO` only if all global thresholds pass, otherwise `NO-GO` with the failed
  threshold and original NumPy loader retained.

Do not start the full conversion without a new user confirmation after this
report.
