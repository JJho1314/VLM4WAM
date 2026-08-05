# WorldArena Baton Planner Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Execute inline in this worktree; the user did not request sub-agent delegation.

**Goal:** Make the WorldArena Qwen3.5-2B Stage-1 planner use an assistant-side Baton blueprint with explicit time coordinates, exhaustive training-window coverage, backward-compatible checkpoint contracts, and deterministic language-grounding validation.

**Architecture:** Preserve the current Qwen3.5, SigLIP2 teacher, query tower, Sem-MLP, and raw MSE topology. Repair only the sequence/data contract and experiment-control layer: render a full system/user/assistant conversation, carry temporal provenance through batches, enumerate WorldArena windows explicitly, version those behaviors in checkpoint metadata, and evaluate every candidate against correct text, task-distinct shuffled text, and persistence.

**Tech Stack:** Python 3.10+, PyTorch, Transformers Qwen/SigLIP processors, HDF5, Accelerate/DDP, pytest.

---

## Compatibility invariants

- Do not change planner parameter topology or loss formulation.
- Do not restore positive/negative multi-row training.
- Format-4 checkpoints must load with byte-identical legacy row rendering.
- LIBERO keeps verbatim instructions and its existing sampling policy.
- The tracked production WorldArena recipe remains global batch 128.

### Task 1: Add versioned Baton sequence rendering

**Files:**
- Modify: `qwen35_baton/sequence.py`
- Test: `tests/test_qwen35_baton_sequence.py`
- Test: `tests/test_qwen35_baton_data.py`

**Step 1: Write failing tests**

Add tests that require:

```python
conversation = build_baton_conversation(
    instruction="pick up the red cube",
    source_indices=(12, 39, 66, 93, 120),
)
assert [row["role"] for row in conversation] == ["system", "user", "assistant"]
assert conversation[1]["content"][0]["type"] == "image"
assert conversation[2]["content"].count(PLAN_PAD) == 256 * 4
assert "Current frame: 12/120" in conversation[1]["content"][1]["text"]
assert build_plan_text("pick up the red cube") == LEGACY_EXPECTED_TEXT
```

Also test exact stripping of the shared WorldArena boilerplate, refusal of a blank remainder, and no stripping for LIBERO.

**Step 2: Run the tests and observe RED**

Run: `pytest -q tests/test_qwen35_baton_sequence.py tests/test_qwen35_baton_data.py`

Expected: failure because `build_baton_conversation`, template-kind dispatch, and instruction rendering do not exist.

**Step 3: Implement the smallest versioned sequence API**

Keep `build_plan_text()` unchanged for legacy checkpoints. Add constants and pure helpers:

```python
LEGACY_TEMPLATE_KIND = "legacy_user_plan_v1"
BATON_TEMPLATE_KIND = "baton_assistant_time_v2"
VERBATIM_INSTRUCTION_KIND = "verbatim_v1"
STRIP_WORLD_ARENA_INSTRUCTION_KIND = "strip_worldarena_boilerplate_v1"

def build_baton_conversation(
    instruction: str,
    source_indices: tuple[int, int, int, int, int],
) -> list[dict[str, object]]:
    current, *future = validate_source_indices(source_indices)
    return [
        {"role": "system", "content": BATON_SYSTEM_TEXT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": build_baton_user_text(instruction, current, future),
                },
            ],
        },
        {"role": "assistant", "content": build_plan_scaffold()},
    ]
```

The assistant payload contains only `<PLAN_START>`, four ordered frame blocks, and `<PLAN_END>`. Validate integral, strictly increasing indices in `[0, 120]`; render normalized current time to six decimals; use `add_generation_prompt=False` for this template.

**Step 4: Run targeted tests and observe GREEN**

Run: `pytest -q tests/test_qwen35_baton_sequence.py tests/test_qwen35_baton_data.py`

**Step 5: Commit**

```bash
git add qwen35_baton/sequence.py tests/test_qwen35_baton_sequence.py tests/test_qwen35_baton_data.py
git commit -m "feat: add assistant-side Baton planner template"
```

### Task 2: Carry temporal and rendered-instruction provenance through batches

**Files:**
- Modify: `qwen35_baton/data.py`
- Modify: `tests/test_qwen35_baton_data.py`
- Modify: `tests/test_qwen35_baton_cpu_path.py`

**Step 1: Write failing tests**

Require `BatonPlannerBatch` to expose original instructions, rendered instructions, and source indices for every Qwen row. Require `BatonLiberoDataset` to emit five relative source indices. Assert the new collator produces exactly one image and 1,024 assistant-side pad positions per row, rejects nonmonotonic indices before processor invocation, and preserves the existing single positive row per camera.

**Step 2: Run the tests and observe RED**

Run: `pytest -q tests/test_qwen35_baton_data.py tests/test_qwen35_baton_cpu_path.py`

Expected: new metadata fields and template dispatch are absent.

**Step 3: Implement batch and collator changes**

Extend the immutable batch contract without moving string metadata to GPU:

```python
@dataclass(frozen=True)
class BatonPlannerBatch:
    qwen_inputs: Mapping[str, torch.Tensor]
    plan_positions: torch.Tensor
    current_images: torch.Tensor
    future_images: torch.Tensor | None
    instructions: tuple[str, ...]
    rendered_instructions: tuple[str, ...]
    source_indices: tuple[tuple[int, int, int, int, int], ...]
    row_labels: tuple[tuple[int, str], ...]
    camera_names: tuple[str, ...] = ("main", "wrist")
    future_pixel_values: torch.Tensor | None = None
```

Add `input_template_kind` and `instruction_rendering_kind` constructor arguments to `BatonPlannerCollator`. Dispatch legacy checkpoints to the unchanged user-only rendering path. Dispatch the new kind to `build_baton_conversation()` and `add_generation_prompt=False`. Derive LIBERO indices from the selected current frame and `geometry.future_indices`.

**Step 4: Run targeted tests and observe GREEN**

Run: `pytest -q tests/test_qwen35_baton_data.py tests/test_qwen35_baton_cpu_path.py`

**Step 5: Commit**

```bash
git add qwen35_baton/data.py tests/test_qwen35_baton_data.py tests/test_qwen35_baton_cpu_path.py
git commit -m "feat: propagate Baton temporal batch metadata"
```

### Task 3: Add exhaustive WorldArena window indexing

**Files:**
- Modify: `qwen35_baton/worldarena_data.py`
- Modify: `tests/test_qwen35_baton_worldarena_data.py`

**Step 1: Write failing tests**

Build a two-record fixture and require:

```python
dataset = WorldArenaHDF5Dataset(
    manifest,
    seed=42,
    split="train",
    sampling_kind="all_windows_v1",
)
assert len(dataset) == 2 * 117
assert dataset[0]["source_indices"][0] == 0
assert dataset[116]["source_indices"][0] == 116
assert dataset[117]["episode_id"] != dataset[0]["episode_id"]
```

Prove `set_epoch()` does not change an all-window sample and still changes a legacy `episode_random_v1` training sample deterministically. Validation remains one fixed sample per episode.

**Step 2: Run the tests and observe RED**

Run: `pytest -q tests/test_qwen35_baton_worldarena_data.py -k 'sampling or all_window or epoch'`

Expected: constructor rejects `sampling_kind` and length is record count.

**Step 3: Implement the indexed sampling policy**

Add explicit constants and validate them:

```python
EPISODE_RANDOM_SAMPLING_KIND = "episode_random_v1"
ALL_WINDOWS_SAMPLING_KIND = "all_windows_v1"

if self.sampling_kind == ALL_WINDOWS_SAMPLING_KIND and self.split == "train":
    record_index, current_index = divmod(index, 117)
else:
    record_index, current_index = index, self._legacy_current_index(index)
```

Keep `future_frame_indices(current_index)` as the only future-index formula.

**Step 4: Run targeted tests and observe GREEN**

Run: `pytest -q tests/test_qwen35_baton_worldarena_data.py`

**Step 5: Commit**

```bash
git add qwen35_baton/worldarena_data.py tests/test_qwen35_baton_worldarena_data.py
git commit -m "feat: enumerate all WorldArena training windows"
```

### Task 4: Version checkpoint behavior contracts as format 5

**Files:**
- Modify: `qwen35_baton/config.py`
- Modify: `qwen35_baton/provider.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Modify: `tests/test_qwen35_baton_config.py`
- Modify: `tests/test_qwen35_baton_checkpoint.py`
- Modify: `tests/test_qwen35_baton_provider.py`

**Step 1: Write failing tests**

Require a format-5 round trip with:

```python
assert restored.input_template_kind == "baton_assistant_time_v2"
assert restored.worldarena_sampling_kind == "all_windows_v1"
assert restored.instruction_rendering_kind == "strip_worldarena_boilerplate_v1"
```

Load a format-4 fixture missing those fields and assert it migrates to the three legacy values. Require provider/evaluation collation to select behavior from metadata, not from an implicit current default.

**Step 2: Run the tests and observe RED**

Run: `pytest -q tests/test_qwen35_baton_config.py tests/test_qwen35_baton_checkpoint.py tests/test_qwen35_baton_provider.py`

Expected: format remains 4 and fields are unknown.

**Step 3: Implement strict migration and dispatch**

Increment `FORMAT_VERSION` to 5. Add three required string fields to serialization and equality checks. In only the format-4 migration, inject:

```python
payload.setdefault("input_template_kind", "legacy_user_plan_v1")
payload.setdefault("worldarena_sampling_kind", "episode_random_v1")
payload.setdefault("instruction_rendering_kind", "verbatim_v1")
```

New WorldArena artifacts use the approved v2/all-window/stripped values; new LIBERO artifacts use the v2 template with its existing sampling and verbatim instruction rendering. Derive `input_template_hash` from the actual versioned template contract.

**Step 4: Run targeted tests and observe GREEN**

Run: `pytest -q tests/test_qwen35_baton_config.py tests/test_qwen35_baton_checkpoint.py tests/test_qwen35_baton_provider.py`

**Step 5: Commit**

```bash
git add qwen35_baton/config.py qwen35_baton/provider.py qwen35_baton/cli/train_semantic_planner.py tests/test_qwen35_baton_config.py tests/test_qwen35_baton_checkpoint.py tests/test_qwen35_baton_provider.py
git commit -m "feat: version Baton input and sampling contracts"
```

### Task 5: Implement deterministic grounding validation

**Files:**
- Create: `qwen35_baton/validation.py`
- Create: `tests/test_qwen35_baton_validation.py`

**Step 1: Write failing tests**

Test task-distinct shuffling on repeated task labels, deterministic permutation under a seed, pointwise aggregate/per-horizon MSE and cosine, sample-win counting, norm and spatial-standard-deviation ratios, future-delta cosine, gate pass/fail reasons, nonfinite rejection, and atomic JSON replacement.

Use exact gate boundaries:

```python
decision = evaluate_grounding_gates(metrics)
assert decision.required_examples == 44
assert decision.minimum_correct_win_fraction == 0.60
assert decision.minimum_shuffle_mse_improvement == 0.05
assert decision.minimum_persistence_mse_improvement == 0.25
assert decision.accepted_norm_ratio == (0.85, 1.15)
```

**Step 2: Run the tests and observe RED**

Run: `pytest -q tests/test_qwen35_baton_validation.py`

Expected: module does not exist.

**Step 3: Implement pure validation primitives**

Keep model execution separate from metric math. Expose typed functions for task-distinct pairing, batch accumulation, final metric calculation, gate evaluation, and atomic artifact publication. Include per-sample task, episode, instruction, shuffled instruction, and source-index provenance in the artifact. Never train on shuffled rows.

**Step 4: Run targeted tests and observe GREEN**

Run: `pytest -q tests/test_qwen35_baton_validation.py`

**Step 5: Commit**

```bash
git add qwen35_baton/validation.py tests/test_qwen35_baton_validation.py
git commit -m "feat: add deterministic planner grounding gates"
```

### Task 6: Integrate the 5,000-step WorldArena schedule and validation cadence

**Files:**
- Modify: `qwen35_baton/cli/train_semantic_planner.py`
- Modify: `qwen35_baton/configs/worldarena_stage1.json`
- Modify: `qwen35_baton/cli/preflight.py`
- Modify: `tests/test_qwen35_baton_training.py`
- Modify: `tests/test_qwen35_baton_config.py`
- Modify: `tests/test_qwen35_baton_end_to_end.py`

**Step 1: Write failing tests**

Require WorldArena-only schedule values `max_steps=5000`, validation every 500, and save steps `(20, 500, 1000, 2000, 3000, 4000, 5000)`. Require the run to publish a validation artifact beside every evaluated checkpoint and to stop at 5,000 even if no gate passes. Keep LIBERO's current 30,000-step validation rules unchanged.

**Step 2: Run the tests and observe RED**

Run: `pytest -q tests/test_qwen35_baton_training.py tests/test_qwen35_baton_config.py tests/test_qwen35_baton_end_to_end.py`

Expected: production validation currently hard-codes 30,000/5,000 cadence and has no grounding evaluation schedule.

**Step 3: Implement dataset-scoped schedule configuration**

Add strict config fields such as `validation_every` and `evaluated_save_steps`. Validate the exact approved WorldArena schedule and preserve global batch 128. Construct a deterministic validation dataset/loader with one fixed row per validation episode. At each scheduled step, run correct, task-distinct shuffled, and persistence evaluation with gradients disabled, publish metrics atomically, and record gate eligibility in checkpoint metadata or its adjacent manifest.

Avoid holding validation HDF5 handles across workers or retaining prediction tensors after a pass. Reuse the existing worker-recycle mechanism and force `num_workers=0` for the small 44-example validation loader to remove FD/IPC leak risk.

**Step 4: Run targeted tests and observe GREEN**

Run: `pytest -q tests/test_qwen35_baton_training.py tests/test_qwen35_baton_config.py tests/test_qwen35_baton_end_to_end.py`

**Step 5: Commit**

```bash
git add qwen35_baton/cli/train_semantic_planner.py qwen35_baton/cli/preflight.py qwen35_baton/configs/worldarena_stage1.json tests/test_qwen35_baton_training.py tests/test_qwen35_baton_config.py tests/test_qwen35_baton_end_to_end.py
git commit -m "feat: add WorldArena grounding validation schedule"
```

### Task 7: Full regression, compatibility smoke, and remote-ready artifact

**Files:**
- Modify if required: `README.md`
- Modify if required: `qwen35_baton/README.md`

**Step 1: Run focused full suite**

Run:

```bash
pytest -q tests/test_qwen35_baton_sequence.py \
  tests/test_qwen35_baton_data.py \
  tests/test_qwen35_baton_worldarena_data.py \
  tests/test_qwen35_baton_config.py \
  tests/test_qwen35_baton_checkpoint.py \
  tests/test_qwen35_baton_provider.py \
  tests/test_qwen35_baton_validation.py \
  tests/test_qwen35_baton_training.py \
  tests/test_qwen35_baton_end_to_end.py
```

Expected: all pass with no skipped compatibility assertions.

**Step 2: Run static and repository hygiene checks**

Run:

```bash
python -m compileall -q qwen35_baton
git diff --check
git status --short
```

Expected: compilation and whitespace checks pass; only intentional tracked changes and the pre-existing untracked `runtime/` directory remain.

**Step 3: Run local tiny train/save/resume smoke**

Use the existing tiny artifact fixtures to take an optimizer step, save format 5, reload it, and assert the template/sampling/rendering contracts remain exact. Do not use the old step-15000 format-4 checkpoint as a new-format resume source.

**Step 4: Document the new run contract**

Record the 5,000-step cadence, all-window example count, validation gates, and the fact that GE-Act Stage 2/3 must wait for an eligible checkpoint.

**Step 5: Commit final integration**

```bash
git add README.md qwen35_baton/README.md
git commit -m "docs: document repaired WorldArena planner run"
```

**Step 6: Prepare remote smoke deployment**

Sync only tracked source/config/test files to a new revision-named directory on the selected GPU server. Run preflight and a bounded tiny smoke before any multi-hour launch. Preserve the old `c0812aa` directory and step-15000 checkpoint for reproducibility.
