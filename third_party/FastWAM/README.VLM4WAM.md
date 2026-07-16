# FastWAM Vendor Notes for VLM4WAM

This directory vendors the Cosmos-backbone FastWAM working tree from:

```text
/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean
```

The imported copy keeps FastWAM self-contained under `third_party/FastWAM` so
it does not collide with VLM4WAM's root `scripts`, `tests`, README, or existing
Cosmos Predict tree.

## Included

- `src/fastwam`
- `configs`
- `scripts`
- `experiments`
- `third_party/RoboTwin`
- `pyproject.toml`, README files, LICENSE, and package metadata

The copy includes the source worktree state at import time, including the
Cosmos-backbone FastWAM edits and local evaluation helper scripts.

## Excluded

Generated or local artifacts were intentionally excluded:

- `.git`, agent/editor metadata
- `runs`, `checkpoints`, `data`, `outputs`, `evaluate_results`
- logs, Python caches, egg-info
- model weights, videos, arrays, archives, and paper PDFs

## Cosmos Backbone

`scripts/run_cosmos_libero_posttrain_local.sh` defaults to the Cosmos checkout in
this repository:

```text
third_party/cosmos-predict2.5
```

The launcher still honors environment overrides:

```bash
REPO=/path/to/FastWAM \
COSMOS_REPO=/path/to/cosmos-predict2.5 \
COSMOS_PY=/path/to/python \
bash scripts/run_cosmos_libero_posttrain_local.sh
```

From the VLM4WAM root, the default local launch path is:

```bash
cd third_party/FastWAM
bash scripts/run_cosmos_libero_posttrain_local.sh
```

The script exports:

```bash
PYTHONPATH="$REPO/src:$COSMOS_REPO:${PYTHONPATH:-}"
```

so FastWAM imports come from this vendored copy and Cosmos imports come from the
VLM4WAM `third_party/cosmos-predict2.5` backbone by default.

The main LIBERO evaluation entrypoints were also adjusted to infer the same
defaults:

```bash
cd third_party/FastWAM
python experiments/libero/cosmos_eval_libero.py --help
python experiments/libero/cosmos_eval_libero_plus.py --help
bash experiments/libero/run_cosmos_eval_plus_par.sh
```

Machine-specific automation scripts under `scripts/` may still contain defaults
for historical run directories or virtualenvs. Override `REPO`, `COSMOS_REPO`,
`VENV`, `PY`, `RUN_DIR`, `OUT`, and related environment variables when reusing
those scripts.
