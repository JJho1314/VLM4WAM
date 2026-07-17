# LIBERO-FastWAM Sharded HDF5 Loader Design

## Goal

Add an optional, independent HDF5 data path for the fixed LIBERO-FastWAM
training contract. The existing `CustomLeRobotDataset`, MP4 path, predecoded
NumPy cache path, and their YAML files remain present and behaviorally
unchanged.

The first milestone is a 64-episode pilot. Full conversion is allowed only
after the pilot is lossless at the model-input boundary and improves sustained
DataLoader throughput by at least 1.5x on the target HPC3 training setup.

## Fixed scope

The initial backend supports exactly:

- camera order `main=0`, `wrist=1`, sourced from
  `observation.images.image` and `observation.images.wrist_image`;
- RGB output resolution 256x256;
- source rate 20 FPS;
- `n_previous=4`;
- future `chunk=9`;
- `action_chunk=36`, hence video temporal stride 4;
- absolute EEF actions;
- the four LIBERO-FastWAM domains already used by the GE-Act configuration;
- GE-Act video training and the dual-camera VLM planner adapter.

It is not a general LeRobot-to-HDF5 framework. It does not store expanded
training windows, precomputed SigLIP2/DA3 features, VLM outputs, or LTX latents.

## Architecture

### Converter

Create a standalone converter under `ge_act/scripts/`. It reads the existing
LeRobot episode metadata, parquet action/state arrays, and either the strict
predecoded RGB cache or the source videos. The strict predecoded cache is the
preferred source when present.

The converter resizes RGB offline using the same deterministic resize operation
as the current `random_crop=false`, `preprocess=resize`, 256x256 training path.
Color jitter and normalization remain online.

Episodes are assigned deterministically to shards in domain and episode-index
order. A shard contains at most 32 episodes. Conversion writes to a temporary
file, validates it, fsyncs it, and atomically renames it. Existing valid shards
are skipped unless overwrite is explicitly requested.

The pilot converts the first 64 valid episodes using both uncompressed RGB and
HDF5 LZF compression. The benchmark selects the faster format; ties within 5%
prefer LZF for lower NFS traffic and storage use.

### HDF5 layout

Each immutable shard uses this structure:

```text
shard_00000.h5
  attrs/
    schema_version = 1
    compression = "none" | "lzf"
    image_height = 256
    image_width = 256
    camera_names = ["main", "wrist"]
  episodes/<global_episode_key>/
    rgb_main       uint8   [T, 256, 256, 3]
    rgb_wrist      uint8   [T, 256, 256, 3]
    action         float32 [T, action_dim]
    state          float32 [T, state_dim]
    caption        UTF-8 scalar
    domain         UTF-8 scalar
    episode_index  int64 scalar
    length         int64 scalar
```

RGB datasets use temporal chunks `(1, 256, 256, 3)` because the current
history sampler can select non-contiguous frames and future frames are spaced
by a stride of four. Action and state use chunks no larger than 64 timesteps.
No lossy image codec is used.

Alongside the shards, write an atomic JSON manifest containing schema version,
source roots, shard paths, every episode key, domain, episode index, length,
compression, shape, dtype, and a converter-configuration fingerprint. Dataset
construction reads only this manifest; `__getitem__` never scans directories.

### Independent dataset

Create `LiberoFastWAMHDF5Dataset` in a new module under `ge_act/data/`. It
implements the same returned sample contract as `CustomLeRobotDataset` but is
selected through a separate `train_data_class_path`/`train_data_class` pair.

The existing loader is not imported, wrapped, renamed, or deleted. Shared pure
sampling utilities may be extracted only if tests prove the original loader is
unchanged; otherwise the HDF5 loader keeps a small fixed-contract
implementation.

Each DataLoader worker lazily opens required shards after worker creation and
reuses read-only handles. Handles are excluded from pickle state and closed on
destruction. Shards are immutable during training, so writer-side SWMR is not
required. A bounded per-worker handle cache prevents every worker from opening
all shards at once.

For each sample, the loader reads only selected RGB frames and selected
action/state rows. It preserves the current random previous-frame selection,
future stride, endpoint clamping, normalization, caption selection, and output
layout `[C, V, T, H, W]` with main before wrist.

### Configuration

Add a separate HDF5 YAML derived from the active LIBERO-FastWAM configuration.
It changes only the dataset class/path and HDF5 manifest settings. The original
YAML continues to require the predecoded NumPy cache.

The HDF5 YAML declares the fixed camera and temporal contract explicitly.
Preflight rejects a schema, camera order, resolution, source FPS, history,
future chunk, or action chunk mismatch before model construction.

## Validation and benchmarking

### Correctness

The converter records deterministic source episode identities. Tests and the
pilot compare both loaders using identical episode IDs and explicit frame/action
indexes, bypassing random selection. Required equality:

- main and wrist frames after the existing deterministic 256x256 resize;
- repeated, clamped, and out-of-range frame behavior;
- selected action and state values;
- caption and domain;
- final video/action/state shapes and dtypes;
- normalized video tensors within the existing tolerance.

Any mismatch fails the pilot and prevents full conversion.

### Performance

Benchmark the old and HDF5 loaders on the same 64 episodes and deterministic
sample-index stream with DataLoader worker counts 0, 2, 4, and 8. Use
`persistent_workers=true`, the production prefetch factor, batch size 8, and
enough warm-up/measurement batches to report stable median and p95 batch data
time. Record samples/s, bytes read, CPU utilization, and peak worker RSS.

Then run a 200-step model smoke on the target HPC3 training node and record
step time, DataLoader wait time, GPU utilization, and GPU power. Full
conversion requires:

- exact correctness;
- at least 1.5x sustained DataLoader samples/s at the selected production
  worker count;
- no greater than 25% increase in aggregate worker RSS;
- no regression in 200-step model throughput.

If neither compression mode passes, retain the original NumPy loader and stop;
the new backend remains optional and is not activated by existing configs.

## Error handling

- Converter errors include domain, episode, camera, source path, and shard.
- Missing or malformed source episodes fail explicitly; they are never replaced
  by a random episode.
- Reader rejects incomplete manifests, duplicate episode keys, missing shards,
  schema mismatch, wrong camera order, wrong dtype/shape, or episode lengths
  inconsistent across datasets.
- HDF5 read failures identify worker, shard, episode, and requested indexes.
- A failed shard conversion never replaces a previously valid shard.

## Rollout

1. Implement schema helpers, converter, independent reader, tests, and separate
   configuration.
2. Convert 64 episodes to uncompressed and LZF pilot datasets on HPC3.
3. Run correctness and DataLoader benchmarks.
4. Run the 200-step training smoke using the winning pilot format.
5. Present measurements before starting the full four-domain conversion.
6. Keep the original NumPy configuration as the supported fallback throughout.
