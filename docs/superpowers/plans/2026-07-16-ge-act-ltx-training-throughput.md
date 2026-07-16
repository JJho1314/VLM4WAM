# GE-Act LTX Training Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated MP4 decoding with a strict episode-level RGB cache and restart GE-Act LTX + SigLIP2 with batch 8/accumulation 2 on 8 H100s.

**Architecture:** A standalone predecoder mirrors each LIBERO camera MP4 as `[T,H,W,3]` uint8 NumPy data under a separate cache root. `CustomLeRobotDataset` maps each episode to that cache and fails on misses when strict mode is enabled. The training config and Slurm launcher preserve global batch 128 while increasing per-GPU work and controlling host thread oversubscription.

**Tech Stack:** Python 3.10, PyAV, NumPy mmap, PyTorch DataLoader, pytest, YAML, Slurm, DeepSpeed ZeRO-2.

## Global Constraints

- Preserve four future keyframes `[0, 3, 5, 8]`, two views, 256 SigLIP2 tokens per keyframe, and semantic injection in all 28 blocks.
- Keep SigLIP2, VAE, and T5 online; only RGB codec decoding is cached.
- Keep gradient checkpointing enabled.
- Keep global batch exactly 128.
- Save checkpoints only at 20,000, 25,000, and 30,000 optimizer steps.
- Never silently fall back to MP4 in the optimized training run.

---

### Task 1: Add strict indexed NumPy cache loading

**Files:**
- Modify: `tests/test_ge_act_source_completeness.py`
- Modify: `ge_act/data/lerobot_like_dataset.py`

**Interfaces:**
- Produces: `load_predecoded_rgb(path: str | Path, slices: Sequence[int]) -> np.ndarray`
- Produces: `CustomLeRobotDataset(..., predecoded_video_root=None, require_predecoded=False)`

- [ ] **Step 1: Write failing cache-index and strict-miss tests**

```python
def test_predecoded_rgb_preserves_order_repeats_and_clamps(tmp_path):
    frames = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
    path = tmp_path / "episode.npy"
    np.save(path, frames)
    actual = module.load_predecoded_rgb(path, [2, 0, 2, 99, -4])
    np.testing.assert_array_equal(actual, frames[[2, 0, 2, 3, 0]])

def test_predecoded_rgb_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="predecoded RGB cache"):
        module.load_predecoded_rgb(tmp_path / "missing.npy", [0])
```

- [ ] **Step 2: Run the tests and confirm they fail because the loader is absent**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_ge_act_source_completeness.py`

Expected: FAIL with `AttributeError: module ... has no attribute 'load_predecoded_rgb'`.

- [ ] **Step 3: Implement indexed cache loading and dataset path mapping**

```python
def load_predecoded_rgb(path, slices):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"predecoded RGB cache is missing: {path}")
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) == 0:
        raise ValueError(f"invalid predecoded RGB cache: {path}")
    indexes = np.clip(np.asarray(slices, dtype=np.int64), 0, len(frames) - 1)
    return np.asarray(frames[indexes])
```

Store a cache-path template in each dataset record. In `seek_mp4`, load camera arrays from that template when configured; if strict mode is false and no cache is configured, retain the existing PyAV behavior.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_ge_act_source_completeness.py`

Expected: all tests pass.

### Task 2: Add resumable atomic predecode and verification CLI

**Files:**
- Create: `ge_act/scripts/predecode_lerobot_videos.py`
- Create: `tests/test_ge_act_predecode_videos.py`

**Interfaces:**
- Produces: `cache_path_for_video(video_path: Path, data_root: Path, cache_root: Path) -> Path`
- Produces: `write_rgb_cache_atomic(cache_path: Path, frames: np.ndarray) -> None`
- Produces CLI: `python scripts/predecode_lerobot_videos.py --config <yaml> [--workers N] [--verify-only]`

- [ ] **Step 1: Write failing layout, atomic-write, and verification tests**

```python
def test_cache_path_mirrors_dataset_tree(tmp_path):
    source = tmp_path / "data/domain/videos/chunk-000/cam/episode_000001.mp4"
    expected = tmp_path / "cache/domain/videos/chunk-000/cam/episode_000001.npy"
    assert module.cache_path_for_video(source, tmp_path / "data", tmp_path / "cache") == expected

def test_atomic_cache_writer_round_trips_uint8(tmp_path):
    frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    destination = tmp_path / "episode.npy"
    module.write_rgb_cache_atomic(destination, frames)
    np.testing.assert_array_equal(np.load(destination), frames)
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run the tests and confirm module import fails**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_ge_act_predecode_videos.py`

Expected: FAIL because `scripts.predecode_lerobot_videos` does not exist.

- [ ] **Step 3: Implement discovery, PyAV decoding, atomic writes, process parallelism, and verify-only mode**

```python
def decode_rgb_video(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    return np.stack(frames).astype(np.uint8, copy=False)

def write_rgb_cache_atomic(cache_path, frames):
    validate_rgb_array(frames)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, frames, allow_pickle=False)
    os.replace(temporary, cache_path)
```

Read roots/domains/cameras/cache root from the training YAML, deduplicate source specifications, use `ProcessPoolExecutor`, skip valid existing files, and write `<cache_root>/manifest.json` with counts, bytes, failures, and timestamp. Verification exits nonzero for any missing or malformed expected cache.

- [ ] **Step 4: Run focused CLI tests**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_ge_act_predecode_videos.py`

Expected: all tests pass.

### Task 3: Update the training contract and HPC3 launcher

**Files:**
- Modify: `ge_act/configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py`
- Modify: `ge_act/scripts/train_ltx_siglip2.sh`
- Create: `ge_act/scripts/sbatch_train_ltx_siglip2_hpc3.sh`
- Modify: `tests/test_ge_act_siglip2_config.py`
- Modify: `tests/test_ge_act_source_completeness.py`

**Interfaces:**
- Consumes: `predecode_lerobot_videos.py --verify-only`
- Produces: an eight-GPU Slurm launcher requesting 96 CPUs with constrained host threading

- [ ] **Step 1: Write failing configuration and launcher tests**

```python
assert config["batch_size"] == 8
assert config["gradient_accumulation_steps"] == 2
assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
assert config["data"]["train"]["require_predecoded"] is True
assert config["data"]["train"]["predecoded_video_root"].endswith("LIBERO-fastwam-predecoded-rgb")
assert "#SBATCH --cpus-per-task=96" in launcher
assert "OMP_NUM_THREADS=1" in launcher
```

- [ ] **Step 2: Run the tests and confirm the old batch/cache contract fails**

Run: `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q tests/test_ge_act_siglip2_config.py tests/test_ge_act_source_completeness.py`

Expected: FAIL because batch is 2/8 and the strict cache/launcher are absent.

- [ ] **Step 3: Implement configuration, strict preflight, cache verification, and Slurm settings**

Set both train and validation data sections to:

```yaml
predecoded_video_root: /data/user/jhe724/junjie/datasets/LIBERO-fastwam-predecoded-rgb
require_predecoded: true
```

Set `batch_size: 8`, `gradient_accumulation_steps: 2`, and retain `gradient_checkpointing: true`. The shell launcher runs static preflight followed by `predecode_lerobot_videos.py --verify-only` before `torchrun`. The Slurm script requests 8 GPUs, 96 CPUs, 512 GiB RAM, and exports all BLAS/OpenMP thread counts as 1.

- [ ] **Step 4: Run all GE-Act semantic tests and shell syntax checks**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q \
  tests/test_ge_act_predecode_videos.py \
  tests/test_ge_act_source_completeness.py \
  tests/test_ge_act_siglip2_config.py \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_semantic_pipeline.py \
  tests/test_ge_act_ltx_semantic_guidance.py
bash -n ge_act/scripts/train_ltx_siglip2.sh ge_act/scripts/sbatch_train_ltx_siglip2_hpc3.sh
```

Expected: all tests pass and shell syntax exits zero.

### Task 4: Deploy, predecode, and restart with measured acceptance

**Files:**
- No new source files
- Remote cache: `/data/user/jhe724/junjie/datasets/LIBERO-fastwam-predecoded-rgb`
- Remote logs: `/data/user/jhe724/workspace/VLM4WAM_geact_semantic_d0ec565/logs/`

**Interfaces:**
- Consumes: committed `semantic-guidance` branch and the two launch scripts
- Produces: verified 30,000-step Slurm job and measured throughput comparison

- [ ] **Step 1: Stop the superseded job and record its final progress**

```bash
ssh HPC3_jhe724 'squeue -j 415891; scancel 415891; sacct -j 415891 --format=JobID,State,Elapsed,ExitCode'
```

- [ ] **Step 2: Commit and push only the formal implementation and tests**

```bash
git add ge_act tests/test_ge_act_predecode_videos.py tests/test_ge_act_source_completeness.py tests/test_ge_act_siglip2_config.py docs/superpowers
git commit -m "perf(ge-act): predecode LIBERO RGB and increase local batch"
git push origin semantic-guidance
```

- [ ] **Step 3: Pull the branch remotely and submit CPU predecode**

```bash
ssh HPC3_jhe724 'cd /data/user/jhe724/workspace/VLM4WAM_geact_semantic_d0ec565 && git pull --ff-only origin semantic-guidance'
ssh HPC3_jhe724 'sbatch --partition=acd_u --cpus-per-task=96 --mem=256G --time=04:00:00 --wrap=".../python ge_act/scripts/predecode_lerobot_videos.py --config ge_act/configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml --workers 32"'
```

- [ ] **Step 4: Verify the full cache and submit training**

```bash
ssh HPC3_jhe724 '/data/user/jhe724/.conda/envs/genie_envisioner/bin/python .../predecode_lerobot_videos.py --config .../video_model_libero_fastwam_siglip2.yaml --verify-only'
ssh HPC3_jhe724 'cd /data/user/jhe724/workspace/VLM4WAM_geact_semantic_d0ec565 && sbatch ge_act/scripts/sbatch_train_ltx_siglip2_hpc3.sh'
```

- [ ] **Step 5: Confirm runtime acceptance and report measured speedup**

Require log evidence for `train_batch_size: 128`, `gradient_accumulation_steps: 2`, no cache miss, no OOM/NaN, at least five completed optimizer steps, eight active GPUs, and compare seconds/step and samples/second with the 4.84 s/step baseline.

If batch 8 OOMs, change only `batch_size: 4` and `gradient_accumulation_steps: 4`, rerun the contract tests, recommit, and relaunch.
