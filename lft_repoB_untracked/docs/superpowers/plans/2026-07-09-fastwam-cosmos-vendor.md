# FastWAM Cosmos Vendor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Cosmos-backbone FastWAM working tree as a clean vendored project under `third_party/FastWAM`.

**Architecture:** Keep FastWAM as a self-contained third-party project and point its launcher defaults at VLM4WAM's existing `third_party/cosmos-predict2.5` checkout. Do not merge FastWAM root-level directories into VLM4WAM root directories.

**Tech Stack:** Python packaging with setuptools, Hydra configs, Accelerate/FSDP launch scripts, Cosmos Predict 2.5 as the video backbone.

## Global Constraints

- Destination is `third_party/FastWAM`.
- Source is `/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean`.
- Include source working-tree code, including current modified and untracked Cosmos evaluation helper scripts.
- Exclude `.git`, `runs`, `checkpoints`, `data`, logs, caches, egg-info, large ML artifacts, archives, videos, arrays, and paper PDFs.
- Preserve the existing VLM4WAM root project layout.
- Use `third_party/cosmos-predict2.5` as the default Cosmos backbone path.

---

## File Structure

- Create: `third_party/FastWAM/`
  - Vendored FastWAM-Cosmos project copied from the source working tree.
- Create: `third_party/FastWAM/README.VLM4WAM.md`
  - Notes for running the vendored copy inside VLM4WAM.
- Modify: `third_party/FastWAM/scripts/run_cosmos_libero_posttrain_local.sh`
  - Default `REPO` and `COSMOS_REPO` to the vendored FastWAM directory and VLM4WAM Cosmos directory.
- Create: `docs/superpowers/specs/2026-07-09-fastwam-cosmos-vendor-design.md`
  - Design record.
- Create: `docs/superpowers/plans/2026-07-09-fastwam-cosmos-vendor.md`
  - This implementation plan.

### Task 1: Copy FastWAM-Cosmos Code

**Files:**
- Create: `third_party/FastWAM/`

**Interfaces:**
- Consumes: source working tree at `/data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean`
- Produces: vendored project at `third_party/FastWAM`

- [ ] **Step 1: Confirm destination state**

Run: `test ! -e third_party/FastWAM || find third_party/FastWAM -maxdepth 1 -print`
Expected: either no output from `test`, or a short listing that can be reviewed before copying.

- [ ] **Step 2: Copy with artifact excludes**

Run: `rsync -a --delete --exclude='.git/' --exclude='.codex/' --exclude='.claude/' --exclude='.vscode/' --exclude='.idea/' --exclude='runs/' --exclude='checkpoints/' --exclude='data/' --exclude='outputs/' --exclude='evaluate_results/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.pyo' --exclude='*.log' --exclude='*.egg-info/' --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.safetensors' --exclude='*.distcp' --exclude='*.onnx' --exclude='*.engine' --exclude='*.mp4' --exclude='*.avi' --exclude='*.mov' --exclude='*.mkv' --exclude='*.npy' --exclude='*.npz' --exclude='*.h5' --exclude='*.hdf5' --exclude='*.parquet' --exclude='*.tar' --exclude='*.tar.gz' --exclude='*.zip' --exclude='*.pdf' /data/LFT-W02_data/junjie/VLA_WM/FastWAM_official_clean/ third_party/FastWAM/`
Expected: command exits 0 and `third_party/FastWAM/src/fastwam/models/cosmos/runtime.py` exists.

### Task 2: Patch VLM4WAM Defaults

**Files:**
- Modify: `third_party/FastWAM/scripts/run_cosmos_libero_posttrain_local.sh`
- Create: `third_party/FastWAM/README.VLM4WAM.md`

**Interfaces:**
- Consumes: VLM4WAM root path inferred from script location
- Produces: launcher defaults that use `third_party/cosmos-predict2.5`

- [ ] **Step 1: Patch launcher defaults**

Change the launcher to infer:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VLM4WAM_ROOT="$(cd "${FASTWAM_ROOT}/../.." && pwd -P)"

REPO="${REPO:-${FASTWAM_ROOT}}"
COSMOS_REPO="${COSMOS_REPO:-${VLM4WAM_ROOT}/third_party/cosmos-predict2.5}"
```

Expected: existing environment overrides still work.

- [ ] **Step 2: Add local README**

Create `third_party/FastWAM/README.VLM4WAM.md` with the source path, included
scope, excluded artifacts, and launcher usage.

Expected: the file explains `PYTHONPATH="$REPO/src:$COSMOS_REPO"` and the main
training command.

### Task 3: Verify Import Shape and Git State

**Files:**
- Read: `third_party/FastWAM/`

**Interfaces:**
- Consumes: copied and patched FastWAM vendor tree
- Produces: verification output for final report

- [ ] **Step 1: Check expected files exist**

Run: `test -f third_party/FastWAM/src/fastwam/models/cosmos/runtime.py`
Expected: exit 0.

Run: `test -f third_party/FastWAM/configs/model/fastwam_cosmos.yaml`
Expected: exit 0.

Run: `test -f third_party/FastWAM/scripts/run_cosmos_libero_posttrain_local.sh`
Expected: exit 0.

- [ ] **Step 2: Check artifact directories were excluded**

Run: `test ! -e third_party/FastWAM/runs && test ! -e third_party/FastWAM/checkpoints && test ! -e third_party/FastWAM/data`
Expected: exit 0.

- [ ] **Step 3: Run lightweight syntax check**

Run: `python -m py_compile third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py third_party/FastWAM/src/fastwam/models/cosmos/runtime.py third_party/FastWAM/src/fastwam/trainer.py`
Expected: exit 0, without importing heavyweight ML packages.

- [ ] **Step 4: Inspect git status**

Run: `git status -sb`
Expected: docs and `third_party/FastWAM/` show as new or modified; no `runs`, `checkpoints`, or `data` files show up.
