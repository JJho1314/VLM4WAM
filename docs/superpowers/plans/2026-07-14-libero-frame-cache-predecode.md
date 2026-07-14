# LIBERO Frame-Cache Predecode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predecode all four LIBERO datasets into a verified mmap frame cache and restart the eight-GPU planner training without live MP4 decoding.

**Architecture:** Extend the existing FastWAM precompute utility with a backend-neutral all-frame decoder that selects TorchCodec when available and sequential PyAV otherwise. Keep the existing runtime cache interface unchanged, validate the cached tensors end to end, then enable the cache with `FASTWAM_FRAME_CACHE_DIR` in the existing pod launcher.

**Tech Stack:** Python 3.10, PyTorch, PyAV, NumPy mmap, torchvision resize, pytest, FastWAM/LeRobot, SSH, rsync, Accelerate, DeepSpeed ZeRO-2.

## Global Constraints

- Do not install a new decoder dependency on the remote pod; PyAV is already present and TorchCodec is absent.
- Cache path is `/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224`.
- Cache payload is `uint8 [N, 3, 224, 224]` plus an `.fps.json` sidecar per episode/camera MP4.
- Precompute is resumable and uses atomic rename; a partial file must never look complete.
- Existing live-decode fallback remains unchanged at runtime.
- Do not commit or overwrite unrelated dirty-worktree changes.
- Restart with the same eight-GPU, batch-per-GPU 16, global-batch 128, ZeRO-2, BF16, 30,000-step configuration.

---

### Task 1: Add an auto-selected sequential PyAV predecoder

**Files:**
- Create: `tests/test_fastwam_frame_precompute.py`
- Modify: `third_party/FastWAM/scripts/precompute_frames.py`

**Interfaces:**
- Produces: `resolve_decoder_backend(requested: str) -> str`
- Produces: `decode_all_frames_pyav(video_path: Path) -> tuple[torch.Tensor, float]`
- Produces: `decode_all_frames(video_path: Path, backend: str) -> tuple[torch.Tensor, float]`
- Updates: `decode_and_resize(..., decoder_backend: str) -> tuple[np.ndarray, float]`

- [ ] **Step 1: Write failing backend-selection and PyAV RGB tests**

Create tests that load `precompute_frames.py` by path, replace the imported `av` module with a fake container exposing two known `rgb24` arrays, and assert:

```python
def test_resolve_decoder_backend_auto_falls_back_to_pyav(monkeypatch):
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    assert module.resolve_decoder_backend("auto") == "pyav"


def test_decode_all_frames_pyav_returns_uint8_nchw(monkeypatch, tmp_path):
    frames, fps = module.decode_all_frames_pyav(tmp_path / "episode.mp4")
    assert frames.dtype == torch.uint8
    assert frames.shape == (2, 3, 2, 3)
    assert fps == 20.0
    assert frames[0, :, 0, 0].tolist() == [1, 2, 3]
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run `pytest -q tests/test_fastwam_frame_precompute.py`.

Expected: failures because `resolve_decoder_backend` and `decode_all_frames_pyav` do not exist.

- [ ] **Step 3: Implement backend selection and sequential PyAV decoding**

Add:

```python
import importlib.util


def resolve_decoder_backend(requested: str) -> str:
    if requested not in {"auto", "torchcodec", "pyav"}:
        raise ValueError(f"unsupported decoder backend: {requested}")
    if requested == "auto":
        return "torchcodec" if importlib.util.find_spec("torchcodec") else "pyav"
    if requested == "torchcodec" and importlib.util.find_spec("torchcodec") is None:
        raise RuntimeError("--decoder-backend torchcodec requested but torchcodec is unavailable")
    return requested


def decode_all_frames_pyav(video_path: Path) -> tuple[torch.Tensor, float]:
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        if rate is None or float(rate) <= 0:
            raise ValueError(f"video has no valid average fps: {video_path}")
        decoded = [
            torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous()
            for frame in container.decode(stream)
        ]
    if not decoded:
        raise ValueError(f"video contains no decoded RGB frames: {video_path}")
    return torch.stack(decoded).to(torch.uint8), float(rate)
```

Keep the TorchCodec path using the existing timestamp-index logic, normalize both backends to `uint8 NCHW`, and add `--decoder-backend {auto,torchcodec,pyav}` with default `auto`.

- [ ] **Step 4: Run focused tests**

Run `pytest -q tests/test_fastwam_frame_precompute.py`.

Expected: all tests pass.

- [ ] **Step 5: Commit the decoder change only**

```bash
git add tests/test_fastwam_frame_precompute.py third_party/FastWAM/scripts/precompute_frames.py
git commit -m "feat: add pyav frame-cache predecoder"
```

### Task 2: Make end-to-end cache verification accept remote dataset overrides

**Files:**
- Modify: `tests/test_fastwam_frame_precompute.py`
- Modify: `third_party/FastWAM/scripts/verify_frame_cache.py`

**Interfaces:**
- Produces: `apply_dataset_overrides(node, dataset_dirs: list[str] | None, norm_stats: str | None) -> None`
- Updates CLI with repeatable dataset paths and explicit normalization-stat path.

- [ ] **Step 1: Write a failing config-override test**

```python
def test_apply_dataset_overrides_updates_hydra_node(module, tmp_path):
    node = OmegaConf.create({"dataset_dirs": ["old"], "pretrained_norm_stats": None})
    module.apply_dataset_overrides(node, [str(tmp_path / "suite")], str(tmp_path / "stats.json"))
    assert list(node.dataset_dirs) == [str(tmp_path / "suite")]
    assert node.pretrained_norm_stats == str(tmp_path / "stats.json")
```

- [ ] **Step 2: Run the focused test and verify failure**

Run `pytest -q tests/test_fastwam_frame_precompute.py -k dataset_overrides`.

Expected: failure because `apply_dataset_overrides` does not exist.

- [ ] **Step 3: Implement explicit verifier overrides**

Add `--dataset-dirs`, `--pretrained-norm-stats`, and this helper:

```python
def apply_dataset_overrides(node, dataset_dirs, norm_stats) -> None:
    if dataset_dirs:
        node.dataset_dirs = [str(Path(path).expanduser().resolve()) for path in dataset_dirs]
    if norm_stats:
        node.pretrained_norm_stats = str(Path(norm_stats).expanduser().resolve())
```

Call the helper before `instantiate(node)` so the checked dataset is exactly the remote suite supplied on the command line.

- [ ] **Step 4: Run all cache tests**

Run `pytest -q tests/test_fastwam_frame_precompute.py`.

Expected: all tests pass.

- [ ] **Step 5: Commit the verifier change only**

```bash
git add tests/test_fastwam_frame_precompute.py third_party/FastWAM/scripts/verify_frame_cache.py
git commit -m "test: support remote frame-cache verification"
```

### Task 3: Verify locally and deploy the focused files

**Files:**
- Verify: `third_party/FastWAM/scripts/precompute_frames.py`
- Verify: `third_party/FastWAM/scripts/verify_frame_cache.py`
- Verify: `tests/test_fastwam_frame_precompute.py`

**Interfaces:**
- Consumes the Task 1 decoder and Task 2 verifier CLI.
- Produces identical focused files in `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713`.

- [ ] **Step 1: Run focused and related local tests**

Run `pytest -q tests/test_fastwam_frame_precompute.py tests/test_fastwam_sample_timing.py`.

Expected: all tests pass.

- [ ] **Step 2: Run syntax and whitespace checks**

Run:

```bash
python -m py_compile third_party/FastWAM/scripts/precompute_frames.py third_party/FastWAM/scripts/verify_frame_cache.py
git diff --check
```

Expected: exit code 0 for both commands.

- [ ] **Step 3: Sync only the two runtime utilities**

```bash
rsync -av third_party/FastWAM/scripts/precompute_frames.py third_party/FastWAM/scripts/verify_frame_cache.py root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/third_party/FastWAM/scripts/
```

Expected: rsync reports the two files transferred without touching training code or datasets.

- [ ] **Step 4: Run remote CLI import checks**

Run the remote precompute utility with `--help` under `/opt/conda/envs/vlm4wam/bin/python`.

Expected: exit code 0.

### Task 4: Stop live-decode training and build the complete cache

**Files:**
- Create remotely: `/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224/**`
- Create remotely: `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/logs/precompute_libero_frame_cache_224.log`

**Interfaces:**
- Consumes all four remote LIBERO dataset roots.
- Produces 3,424 `.npy` files and 3,424 `.fps.json` sidecars.

- [ ] **Step 1: Record final old-run metrics and terminate launcher PID 3134127**

Run `ssh -p 30282 root@182.242.159.145 'kill -TERM 3134127'`.

Expected: the launcher and eight rank processes exit; all eight H100s return below 500 MiB used.

- [ ] **Step 2: Launch resumable 32-process predecode**

Run the remote utility with `--decoder-backend auto --target 224 --interpolation bilinear --resize-space float --num-workers 32`, the four explicit dataset directories, and cache directory `/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224`. Launch with `nohup`, save the PID, and redirect stdout/stderr to `logs/precompute_libero_frame_cache_224.log`.

Expected startup log:

```text
3424 episode/camera videos across 4 dataset(s)
decoder=pyav
target=224x224 uint8
```

- [ ] **Step 3: Monitor until completion**

Check PID, log counters, CPU use, cache size, and free NAS space every few minutes. Expected final summary:

```text
written + skipped = 3424
missing=0 error=0
DONE
```

- [ ] **Step 4: Validate complete cache structure**

Run a remote Python check that opens every sidecar and mmap array and asserts:

```python
assert array.dtype == np.uint8
assert array.ndim == 4
assert array.shape[1:] == (3, 224, 224)
assert array.shape[0] == metadata["num_frames"]
assert metadata["fps"] == 20.0
```

Expected: 3,424 valid arrays, zero structural failures.

- [ ] **Step 5: Compare live and cached tensors for each suite**

Run `verify_frame_cache.py` four times, each with the Cosmos data config, one explicit suite directory, the existing normalization stats, the cache directory, and two samples.

Expected for every suite: exact shape match, finite reported error statistics, and final `PASS` below the configured maximum absolute-difference tolerance.

### Task 5: Restart cached training and measure throughput

**Files:**
- Create remotely: a new timestamped output directory and log under `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713`.

**Interfaces:**
- Consumes the verified frame cache through `FASTWAM_FRAME_CACHE_DIR`.
- Produces the same four-head planner training artifacts at steps 20,000, 25,000, and 30,000.

- [ ] **Step 1: Preflight GPUs, cache environment, code hashes, and free space**

Expected: eight GPUs below 500 MiB, cache directory readable, local/remote focused-file hashes equal, and no stale planner ranks.

- [ ] **Step 2: Launch the existing pod training wrapper with cache enabled**

Use:

```bash
RUN=qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_$(date +%Y%m%dT%H%M%S)
FASTWAM_FRAME_CACHE_DIR=/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224 \
RUN_KIND=formal NUM_GPUS=8 BATCH_SIZE=16 GRAD_ACCUM=1 \
MAX_STEPS=30000 SAVE_STEPS=5000 SAVE_START_STEP=20000 \
OUTPUT_DIR=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/$RUN \
bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
```

Launch under `nohup`, capture the launcher PID, and write a timestamped log.

- [ ] **Step 3: Verify the first 20 optimizer steps**

Expected: eight ranks alive, each GPU allocated about 69–70 GiB, finite losses at steps 10 and 20, and zero occurrences of `Traceback`, OOM, NaN, or `falling back to real mp4 decode`.

- [ ] **Step 4: Compare steady-state throughput**

Compute average seconds per step over steps 20–100 and compare to the old live-decode baseline of about 4.0 seconds per step. Report the raw old/new values and speedup ratio; do not claim improvement if the measured ratio is not above 1.0.

- [ ] **Step 5: Preserve repository scope**

Do not commit logs, datasets, cache payloads, outputs, or unrelated dirty-worktree files.
