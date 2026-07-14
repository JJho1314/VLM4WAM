# LIBERO Frame-Cache Predecode Design

## Goal

Remove repeated CPU MP4 decoding from the eight-GPU Qwen3-VL planner training path while preserving the current FastWAM video tensor contract and random sample coverage.

## Scope

- Predecode all RGB frames for both cameras in the four LIBERO LeRobot datasets used by the current run.
- Store resized per-camera frames as memory-mappable `uint8` NumPy arrays.
- Select cached frames through the existing FastWAM `decode_video_frames` interface.
- Validate cached tensors against live MP4 decoding before restarting training.
- Restart the same 30,000-step planner configuration with the cache enabled.

Sparse endpoint-only caching is excluded. A training sample may start at any dataset frame, so a sparse cache tied to the current K=1 geometry would either miss valid starts or require a more invasive sampling change.

## Cache Format and Location

Each episode/camera MP4 maps to:

- one `.npy` file with shape `[num_frames, 3, 224, 224]`, dtype `uint8`;
- one `.fps.json` sidecar containing FPS, frame count, and target size.

The cache is stored under `/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224`. The estimated payload is about 78 GiB for 555,426 camera frames. The NAS currently has about 303 TiB free.

Files are written to temporary paths and atomically renamed. Existing complete file/sidecar pairs are skipped, making the job safe to resume.

## Decoder Backend

The precompute tool gains an `auto` decoder mode:

1. use TorchCodec when available;
2. otherwise use a sequential PyAV decoder.

The current remote environment has PyAV but no TorchCodec, so this run will use PyAV. The PyAV implementation decodes each video once in presentation order, converts frames to RGB channel-first tensors, then applies the same float-space bilinear antialiased resize used by the FastWAM processor.

## Runtime Data Flow

Training is launched with:

```text
FASTWAM_FRAME_CACHE_DIR=/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224
```

For each requested timestamp, the existing runtime cache path computes `round(timestamp * fps)`, reads the corresponding frame from an mmap-backed `.npy`, and returns float32 `[0, 1]` tensors. Missing or corrupt cache entries retain the current warning-and-live-decode fallback.

The downstream path is unchanged: two cameras are processed, sampled to nine effective 5 FPS frames, concatenated horizontally, normalized, and the planner wrapper selects current frame 0 and future frame 8.

## Execution

1. Add and test the PyAV all-frame decoder and backend selection.
2. Sync only the relevant files to the existing remote code directory.
3. Stop the current run, which has no reusable optimizer checkpoint.
4. Predecode with 32 CPU processes; retain logs and the resumable cache.
5. Run structural checks over every cache entry.
6. Run end-to-end live-versus-cache comparisons on samples from all four suites.
7. Restart the same eight-GPU, global-batch-128, 30,000-step training job with the cache environment variable enabled.
8. Compare steady-state seconds per step against the previous live-decode run and check for OOM, NaN, traceback, or cache-fallback warnings.

## Validation Criteria

- Unit tests cover PyAV RGB ordering, frame count/FPS metadata, backend selection, and resumable writes.
- Every expected episode/camera video has both a readable `.npy` and `.fps.json` sidecar.
- Cached arrays have dtype `uint8`, shape `[N, 3, 224, 224]`, and metadata frame counts matching `N`.
- End-to-end FastWAM video shapes match exactly between live and cached paths.
- Numeric differences stay within the existing cache verifier tolerance, with mean error and large-error fraction reported.
- Restarted training reaches at least 20 optimizer steps with finite losses, eight active ranks, and zero live-decode fallback warnings.

## Failure Handling

- Decode failures are recorded per video without leaving a completed-looking cache entry.
- The precompute command is rerun after fixing any failures; completed videos are skipped.
- Training is not restarted if cache coverage or numeric verification fails.
- Runtime still falls back to live MP4 decoding for an isolated missing/corrupt entry, but any such warning during the post-restart smoke window is treated as a deployment failure.
