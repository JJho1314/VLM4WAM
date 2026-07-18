# OLA Predecoded VLM Input Design

## Goal

Run the dual-camera K4 planner exclusively from episode-level predecoded RGB
arrays on OLA. Missing or corrupt cache files must stop the run; the dataset
must never fall back to PyAV/MP4 decoding.

## Design

- Store both LIBERO camera streams under
  `/data/shared/datasets/libero_fastwam-predecoded-rgb`, preserving each MP4's
  path relative to `/data/shared/datasets/libero_fastwam` and replacing `.mp4`
  with `.npy`.
- Set `predecoded_video_root` to that directory and
  `require_predecoded: true` in the OLA planner config.
- Before every launch, run the repository's cache verifier over all 3,424
  camera videos, then run the existing dual-camera K4 data preflight.
- Keep DataLoader workers persistent so each epoch/window cycle does not
  repeatedly tear down and respawn 64 workers across eight ranks.

## Validation

The configuration contract and launcher verification are unit-tested. On OLA,
cache generation must finish with zero failures, `--verify-only` must pass for
all 3,424 arrays, and a one-step smoke run must pass before the 30k run starts.

