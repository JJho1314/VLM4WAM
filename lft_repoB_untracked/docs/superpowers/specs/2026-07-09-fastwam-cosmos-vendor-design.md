# FastWAM Cosmos Vendor Design

## Goal

Vendor the Cosmos-backbone FastWAM code from
`/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean` into this repository
without mixing it into the existing semantic planner and Cosmos Predict tree.

## Scope

Include the source repository's current working tree code and configs for the
Cosmos-backbone FastWAM line:

- Python package under `src/fastwam`
- Hydra configs under `configs`
- training and evaluation scripts under `scripts` and `experiments`
- lightweight vendored RoboTwin code under `third_party/RoboTwin`
- package metadata and human documentation such as README, README_zh, LICENSE,
  and `pyproject.toml`

Exclude local and generated artifacts:

- `.git`, `.codex`, `.claude`, editor metadata
- `runs`, `checkpoints`, `data`, `outputs`, `evaluate_results`
- logs, Python caches, egg-info, model weights, videos, arrays, archives, and
  other large ML artifacts
- untracked paper PDFs from the source worktree

## Layout

Place the imported project at `third_party/FastWAM`. This keeps FastWAM's
package layout intact while avoiding collisions with this repository's existing
`scripts`, `tests`, `README.md`, and `third_party/cosmos-predict2.5` files.

Add `third_party/FastWAM/README.VLM4WAM.md` to explain how this vendored copy is
expected to run from inside VLM4WAM.

## Cosmos Backbone Wiring

The FastWAM-Cosmos launcher should default to this repository's checked-in
Cosmos tree:

`third_party/cosmos-predict2.5`

The launcher can still be overridden with `REPO`, `COSMOS_REPO`, `COSMOS_PY`,
`COSMOS_WEIGHTS`, `CKPT`, and `TEXT_CACHE`, matching the source script's current
style.

## Verification

After copying and patching, verify:

- the target directory exists and contains the expected package/config/script
  files
- ignored runtime directories were not copied
- `git status` only shows the intended vendored files and docs
- lightweight syntax checks pass for the patched launcher-facing Python package
  when practical without importing heavyweight ML dependencies
