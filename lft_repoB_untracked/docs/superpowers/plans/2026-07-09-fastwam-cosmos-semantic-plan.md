# FastWAM Cosmos Semantic Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional semantic-plan conditioning to vendored FastWAM-Cosmos by reusing the existing Cosmos semantic-plan adapter and video-block cross-attention path.

**Architecture:** Semantic tokens are projected by the existing `SemanticPlanContextAdapter` and injected only into the Cosmos video DiT. FastWAM action predictions receive semantic guidance indirectly through existing `mot`, `cross_attn`, and `agra` couplings.

**Tech Stack:** Python, PyTorch, Hydra configs, FastWAM vendored package, Cosmos Predict 2.5 semantic-plan utilities.

## Global Constraints

- Default behavior must remain unchanged unless `semantic_plan_context: true` is configured and a batch provides `semantic_plan`.
- Do not add direct semantic-token attention to action expert or GR00T action head.
- Reuse `third_party/cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/semantic_plan_conditioning.py`.
- Keep FastWAM source under `third_party/FastWAM`; do not merge into repository root.
- Restore `third_party/FastWAM/configs/data/*.yaml`, which task configs already reference.

---

## File Structure

- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/video_expert.py`
  - Build semantic-capable Cosmos net and prepare semantic context for manual block loops.
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py`
  - Read semantic plan fields from batches and route them through coupling calls.
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/couplings/mot.py`
  - Pass semantic context to video blocks in the MoT block loop.
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/couplings/cross_attn.py`
  - Pass semantic plan tensors to the standalone video forward path.
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/couplings/agra.py`
  - Pass semantic plan tensors to both video-loss and foresight passes.
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/runtime.py`
  - Add Hydra factory parameters for semantic-plan conditioning.
- Modify: `third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py`
  - Add optional semantic-plan manifest/file loading.
- Modify: `third_party/FastWAM/configs/model/fastwam_cosmos.yaml`
  - Add disabled-by-default semantic-plan config block.
- Create/restore: `third_party/FastWAM/configs/data/*.yaml`
  - Restore data configs from the source FastWAM tree.
- Modify: `.gitignore`
  - Add precise exception for `third_party/FastWAM/configs/data/**`.
- Create: `tests/test_fastwam_cosmos_semantic_plan.py`
  - Unit tests for config restoration, dataset manifest loading, and model semantic context plumbing.

### Task 1: Tests

**Files:**
- Create: `tests/test_fastwam_cosmos_semantic_plan.py`

**Interfaces:**
- Produces failing tests that define the desired behavior before implementation.

- [ ] **Step 1: Add tests for missing behavior**

Tests must cover:

- FastWAM data config file exists.
- `RobotVideoDataset` can load semantic-plan records from a manifest and emit `semantic_plan` and `semantic_plan_times`.
- `FastWAMCosmos.training_loss()` passes semantic tensors into the coupling.
- `CosmosVideoExpert.prepare()` returns semantic context when semantic conditioning is enabled.

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_fastwam_cosmos_semantic_plan.py -q`
Expected: failures from missing config and missing semantic-plan hooks.

### Task 2: Model Plumbing

**Files:**
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/video_expert.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py`
- Modify: coupling files under `third_party/FastWAM/src/fastwam/models/cosmos/couplings/`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/runtime.py`
- Modify: `third_party/FastWAM/configs/model/fastwam_cosmos.yaml`

**Interfaces:**
- Consumes batch keys `semantic_plan` and `semantic_plan_times`.
- Produces video stream state keys `semantic_plan_crossattn` and `semantic_plan_rope`.

- [ ] **Step 1: Add semantic config parameters**

Add optional factory/model args matching the existing Cosmos semantic WM names.

- [ ] **Step 2: Build semantic-capable Cosmos net**

When `semantic_plan_context` is true, set the corresponding LazyConfig fields
before instantiating MiniTrainDIT.

- [ ] **Step 3: Route semantic tensors through training and couplings**

Move semantic batch tensors to model device/dtype and pass them to video paths.

- [ ] **Step 4: Pass semantic context into manual video blocks**

In MoT, call video blocks with semantic cross-attention context after joint
self-attention or through a helper equivalent to the original block section.

### Task 3: Dataset and Config Restoration

**Files:**
- Modify: `third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py`
- Create/restore: `third_party/FastWAM/configs/data/*.yaml`
- Modify: `.gitignore`

**Interfaces:**
- Dataset emits `semantic_plan`, `semantic_plan_times`, and `semantic_plan_meta`
  only when semantic-plan config is enabled.

- [ ] **Step 1: Restore data configs**

Copy `configs/data/*.yaml` from the source FastWAM tree.

- [ ] **Step 2: Add dataset semantic-plan loader**

Implement manifest parsing and file loading compatible with existing Cosmos
semantic-plan payloads.

- [ ] **Step 3: Add gitignore exception**

Ensure restored config files are not ignored by the repository-level `data/`
rule.

### Task 4: Verification

**Files:**
- Read/verify modified files.

**Interfaces:**
- Produces command outputs for final report.

- [ ] **Step 1: Run targeted tests**

Run: `pytest tests/test_fastwam_cosmos_semantic_plan.py -q`
Expected: pass.

- [ ] **Step 2: Run existing semantic-plan tests**

Run: `pytest tests/test_cosmos_semantic_plan_stage2.py -q`
Expected: pass.

- [ ] **Step 3: Run syntax checks**

Run: `python -m py_compile third_party/FastWAM/src/fastwam/models/cosmos/video_expert.py third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py third_party/FastWAM/src/fastwam/models/cosmos/couplings/mot.py third_party/FastWAM/src/fastwam/models/cosmos/couplings/cross_attn.py third_party/FastWAM/src/fastwam/models/cosmos/couplings/agra.py third_party/FastWAM/src/fastwam/models/cosmos/runtime.py third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py`
Expected: exit 0.
