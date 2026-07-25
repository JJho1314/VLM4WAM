# Qwen3.5 Plan-X TA-Tok Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Qwen3.5-VL Plan-X-style semantic planner that learns a 65,536-entry text-aligned SigLIP2 tokenizer, autoregressively predicts four 16x16 future semantic grids for each LIBERO camera, and optionally conditions GE-Act with the planner's pre-head hidden states.

**Architecture:** A new `qwen35_planx` package owns the LIBERO manifests, project-specific TA-Tok, visual-vocabulary expansion, causal planner, constrained decoding, caches, evaluation, and provider. Main and wrist cameras are separate `B*2` examples with shared weights and independent causal state. GE-Act selects the provider only when `semantic_plan.planner_backend: qwen35_planx`; its existing SigLIP2 path and all code under `qwen3_vl_semantic_planner` remain unchanged.

**Tech Stack:** Python 3.10+, PyTorch, Transformers `Qwen3_5ForConditionalGeneration`, SigLIP2-Large/Patch16/256, NumPy memmaps, PyAV/predecoded RGB caches, Accelerate, DeepSpeed ZeRO-2, pytest, GE-Act LTX.

## Global Constraints

- Do not modify any file under `qwen3_vl_semantic_planner`.
- Keep the existing Qwen3.5 K-means/query implementation and checkpoints as an untouched baseline.
- Add no learnable planner query tokens; every predicted visual token is a native causal language-model token.
- Keep camera order exactly `("main", "wrist")`, mapped to `("observation.images.image", "observation.images.wrist_image")`.
- Use future frame offsets `(1, 4, 6, 9)` from current index `c`, equivalent to GE-Act future indices `(0, 3, 5, 8)`.
- Use four keyframes, a 16x16 raster grid, 256 codes per keyframe, and 1,024 codes per camera.
- Use SigLIP2-Large/Patch16/256 and the penultimate vision hidden state.
- Use exactly 65,536 visual codes and store code IDs as `uint16`.
- Initialize the expanded Qwen visual token rows from the trained TA-Tok codebook; never modify the base model or tokenizer directory.
- Reject cache/checkpoint metadata mismatches instead of silently converting camera order, tokenizer hashes, grids, or keyframe offsets.
- GE-Act keeps same-camera semantic cross-attention, semantic dropout, explicit `(t,y,x)` coordinates, camera embeddings, and independent query/key 3D RoPE.
- Joint training uses `total_loss = ge_act_loss + 0.1 * ta_token_cross_entropy`.
- Generated semantic hidden states are the causal Qwen states whose logits predict each visual code, before the tied output head.

---

## File Structure

Create one focused package rather than copying the experimental
`semantic_localization/sg_improve` scripts:

```text
qwen35_planx/
  __init__.py                 Public constants and artifact types.
  config.py                   Frozen dataclasses and metadata validation.
  hashing.py                  Deterministic file/tree/config hashes.
  libero_data.py              Trajectory discovery, split, predecoded RGB access.
  anchors.py                  Qwen vocabulary filtering and deterministic anchor selection.
  ta_tok.py                   Text-aligned quantizer and three-block reconstruction decoder.
  ta_tok_trainer.py           Distributed TA-Tok optimization and validation.
  plan_cache.py               Ground-truth/generated code cache readers and writers.
  vocabulary.py               Experiment-local Qwen vocabulary expansion and initialization.
  sequence.py                 Prompt/response layout, loss mask, and dual-camera collator.
  planner.py                  Qwen3.5 causal planner forward/checkpoint contract.
  decoding.py                 Exact constrained autoregressive decoding.
  provider.py                 GE-Act-facing semantic provider.
  evaluation.py               Metrics and per-camera/per-keyframe aggregation.
  cli/
    build_libero_manifests.py
    train_ta_tok.py
    build_plan_cache.py
    build_generated_plan_cache.py
    train_semantic_planner.py
    evaluate_semantic_planner.py
    visualize_semantic_planner.py
    preflight.py
  scripts/
    train_ta_tok_ola.sh
    build_plan_cache_ola.sh
    build_generated_plan_cache_ola.sh
    train_semantic_planner_ola.sh
    preflight_ola.sh

ge_act/
  configs/ltx_model/libero/action_model_libero_qwen35_planx.yaml
  models/ltx_models/semantic_conditioning.py
  runner/ge_trainer.py
  scripts/train_ltx_qwen35_planx.sh

tests/
  test_qwen35_planx_config.py
  test_qwen35_planx_libero_data.py
  test_qwen35_planx_ta_tok.py
  test_qwen35_planx_plan_cache.py
  test_qwen35_planx_vocabulary.py
  test_qwen35_planx_sequence.py
  test_qwen35_planx_planner.py
  test_qwen35_planx_decoding.py
  test_qwen35_planx_provider.py
  test_ge_act_qwen35_planx.py
  test_qwen3vl2b_legacy_unchanged.py
```

The implementation is delivered in three executable stages:

1. TA-Tok and trajectory-safe LIBERO data.
2. Qwen3.5 causal planner and constrained decoding.
3. GE-Act provider, staged conditioning, and joint loss.

These stages remain in one plan because each downstream artifact is
cryptographically bound to the immediately preceding stage's camera geometry,
anchor mapping, tokenizer state, and checkpoint hash. Each stage nevertheless
ends in its own preflight and may be reviewed or trained independently.

---

### Task 1: Freeze the Qwen3.5 Plan-X contracts

**Files:**
- Create: `qwen35_planx/__init__.py`
- Create: `qwen35_planx/config.py`
- Create: `qwen35_planx/hashing.py`
- Test: `tests/test_qwen35_planx_config.py`
- Test: `tests/test_qwen3vl2b_legacy_unchanged.py`

**Interfaces:**
- Produces: `PlanGeometry`, `TATokMetadata`, `PlannerMetadata`, `sha256_file()`, and `sha256_json()`.
- Produces: canonical camera names/keys, keyframe offsets, grid shape, visual vocabulary size, and metadata validators used by every later task.

- [ ] **Step 1: Write the contract tests**

```python
from qwen35_planx.config import (
    CAMERA_KEYS,
    CAMERA_NAMES,
    PlanGeometry,
    PlannerMetadata,
)


def test_fixed_plan_geometry():
    geometry = PlanGeometry()
    assert CAMERA_NAMES == ("main", "wrist")
    assert CAMERA_KEYS == (
        "observation.images.image",
        "observation.images.wrist_image",
    )
    assert geometry.future_frame_offsets == (1, 4, 6, 9)
    assert geometry.ge_act_future_indices == (0, 3, 5, 8)
    assert geometry.tokens_per_frame == 256
    assert geometry.tokens_per_camera == 1024


def test_planner_metadata_rejects_camera_reordering():
    payload = PlannerMetadata.example().to_dict()
    payload["camera_names"] = ["wrist", "main"]
    with pytest.raises(ValueError, match="camera_names"):
        PlannerMetadata.from_dict(payload)
```

The legacy regression test records SHA-256 hashes for every tracked file under
`qwen3_vl_semantic_planner` before implementation and compares them after all
tasks. Store the expected path/hash mapping in the test module, generated from
`git ls-files qwen3_vl_semantic_planner`.

- [ ] **Step 2: Run the tests and verify the package is missing**

Run:

```bash
pytest -q tests/test_qwen35_planx_config.py tests/test_qwen3vl2b_legacy_unchanged.py
```

Expected: collection fails because `qwen35_planx` does not exist.

- [ ] **Step 3: Implement immutable geometry and strict metadata**

Use frozen dataclasses. `PlanGeometry.__post_init__()` must derive and validate
all dependent dimensions:

```python
@dataclass(frozen=True)
class PlanGeometry:
    num_keyframes: int = 4
    grid_size: int = 16
    future_frame_offsets: tuple[int, ...] = (1, 4, 6, 9)
    ge_act_future_indices: tuple[int, ...] = (0, 3, 5, 8)
    visual_vocab_size: int = 65536

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def tokens_per_camera(self) -> int:
        return self.num_keyframes * self.tokens_per_frame
```

`TATokMetadata` records the SigLIP2 identifier, selected layer `-2`, image
normalization, Qwen base identifier, anchor IDs, anchor hash, grid, codebook
size, and checkpoint state hash. `PlannerMetadata` additionally records
expanded token IDs, response length, causal hidden-state alignment, camera
layout, tokenizer hash, and TA-Tok hash. `from_dict()` rejects absent or
incompatible required fields.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_config.py tests/test_qwen3vl2b_legacy_unchanged.py
```

Expected: both files pass.

- [ ] **Step 5: Commit the contracts**

```bash
git add qwen35_planx/__init__.py qwen35_planx/config.py qwen35_planx/hashing.py \
  tests/test_qwen35_planx_config.py tests/test_qwen3vl2b_legacy_unchanged.py
git commit -m "feat(planx): define Qwen3.5 planner contracts"
```

---

### Task 2: Build trajectory-safe LIBERO manifests over predecoded RGB

**Files:**
- Create: `qwen35_planx/libero_data.py`
- Create: `qwen35_planx/cli/build_libero_manifests.py`
- Test: `tests/test_qwen35_planx_libero_data.py`

**Interfaces:**
- Consumes: `PlanGeometry`.
- Produces: `TrajectoryRecord`, `FrameRecord`, `PlannerWindowRecord`, `trajectory_split()`, `discover_trajectories()`, `iter_all_camera_frames()`, `iter_planner_windows()`, and `load_predecoded_frames()`.

- [ ] **Step 1: Write split, ordering, and window tests**

```python
def test_trajectory_split_never_splits_an_episode():
    records = make_fake_records(episodes=30)
    assigned = {
        record.trajectory_id: trajectory_split(record.trajectory_id, seed=0)
        for record in records
    }
    assert set(assigned.values()) == {"train", "val"}
    assert len(assigned) == 30


def test_planner_window_uses_exact_future_offsets():
    record = fake_trajectory(num_frames=20)
    window = next(iter_planner_windows(record, stride=10, max_windows=16))
    assert window.current_index == 0
    assert window.future_indices == (1, 4, 6, 9)
    assert window.camera_cache_paths == (
        record.camera_cache_paths["main"],
        record.camera_cache_paths["wrist"],
    )
```

Also test that an invalid cache dtype, missing camera, unequal main/wrist frame
count, and a window extending past the trajectory raise explicit exceptions.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_libero_data.py
```

Expected: import failure for `qwen35_planx.libero_data`.

- [ ] **Step 3: Implement discovery and deterministic splitting**

Read each suite's `meta/episodes.jsonl`, resolve both episode-level cache paths
under `LIBERO-fastwam-predecoded-rgb`, and validate each cache as
`[T,H,W,3] uint8`. Assign the whole `suite:episode_index` trajectory with:

```python
bucket = int.from_bytes(
    hashlib.sha256(f"{seed}:{trajectory_id}".encode()).digest()[:8], "big"
)
split = "val" if bucket % 20 == 0 else "train"
```

`iter_all_camera_frames()` emits both cameras for every frame. Planner windows
use the current index plus `(1,4,6,9)`, a configurable stride defaulting to 10,
and at most 16 windows per trajectory to preserve baseline comparability.

- [ ] **Step 4: Add the manifest CLI**

The CLI accepts repeated `--dataset-root/--domain` pairs, one
`--predecoded-root`, `--output-dir`, `--split-seed`, `--window-stride`, and
`--max-windows-per-trajectory`. It writes sorted `trajectories.jsonl`,
`ta_frames_train.jsonl`, `ta_frames_val.jsonl`, `planner_train.jsonl`,
`planner_val.jsonl`, and a hash-bearing `manifest.json`.

- [ ] **Step 5: Run the data tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_libero_data.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the data layer**

```bash
git add qwen35_planx/libero_data.py qwen35_planx/cli/build_libero_manifests.py \
  tests/test_qwen35_planx_libero_data.py
git commit -m "feat(planx): add trajectory-safe LIBERO manifests"
```

---

### Task 3: Implement deterministic Qwen embedding anchors

**Files:**
- Create: `qwen35_planx/anchors.py`
- Test: `tests/test_qwen35_planx_ta_tok.py`

**Interfaces:**
- Consumes: a Qwen tokenizer and base input embedding matrix.
- Produces: `select_anchor_token_ids()` and `build_frozen_anchor_matrix()`.

- [ ] **Step 1: Write deterministic selection tests**

```python
def test_anchor_selection_is_deterministic_and_excludes_controls(fake_tokenizer):
    first = select_anchor_token_ids(fake_tokenizer, count=8, seed=17)
    second = select_anchor_token_ids(fake_tokenizer, count=8, seed=17)
    assert first == second
    assert not set(first).intersection(fake_tokenizer.all_special_ids)
    assert all("<|" not in fake_tokenizer.convert_ids_to_tokens(i) for i in first)


def test_anchor_matrix_is_frozen():
    matrix = build_frozen_anchor_matrix(torch.arange(80).reshape(20, 4), [2, 5, 7])
    assert matrix.shape == (3, 4)
    assert matrix.requires_grad is False
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
pytest -q tests/test_qwen35_planx_ta_tok.py -k anchor
```

Expected: functions are missing.

- [ ] **Step 3: Implement the selection rule**

Candidates are base-vocabulary IDs that are not special or added control
tokens, do not decode to an empty/whitespace-only string, and whose token text
does not contain `<|`. Use `numpy.random.Generator(PCG64(seed)).permutation`
over sorted candidate IDs and take exactly 65,536 IDs. Sort only the final
selected IDs so the code ID to anchor ID mapping is stable. Fail if fewer than
65,536 candidates survive. Store the IDs and SHA-256 of their FP32 embedding
rows in TA-Tok metadata.

- [ ] **Step 4: Run the anchor tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_ta_tok.py -k anchor
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit anchor selection**

```bash
git add qwen35_planx/anchors.py tests/test_qwen35_planx_ta_tok.py
git commit -m "feat(planx): derive deterministic Qwen visual anchors"
```

---

### Task 4: Implement the project-specific TA-Tok

**Files:**
- Create: `qwen35_planx/ta_tok.py`
- Extend: `tests/test_qwen35_planx_ta_tok.py`

**Interfaces:**
- Consumes: trainable/frozen SigLIP2 vision modules and `[65536,Dq]` frozen Qwen anchors.
- Produces: `TextAlignedTokenizer`, `TATokOutput`, `nearest_code_indices()`, `encode_codes()`, and `decode_codes()`.

- [ ] **Step 1: Write quantization and loss tests**

```python
def test_ta_tok_shapes_and_stop_gradient(fake_siglip_pair, anchors):
    model = TextAlignedTokenizer.from_modules(
        student=fake_siglip_pair.student,
        teacher=fake_siglip_pair.teacher,
        frozen_anchors=anchors,
        feature_dim=8,
        qwen_dim=12,
        decoder_depth=3,
    )
    output = model(torch.rand(2, 3, 256, 256))
    assert output.codes.shape == (2, 256)
    assert output.quantized.shape == (2, 256, 12)
    assert output.reconstruction.shape == (2, 256, 8)
    assert output.losses["reconstruction"].ndim == 0
    assert output.losses["commitment"].ndim == 0
    assert output.losses["codebook"].ndim == 0
    output.loss.backward()
    assert model.student_projection.weight.grad is not None
    assert model.codebook_projection.weight.grad is not None
    assert model.frozen_anchors.grad is None
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
```

Add tests for chunked nearest-neighbor equivalence to a dense cosine matrix,
256-token enforcement, penultimate-layer selection, code usage/perplexity, and
round-trip `decode_codes(encode_codes(images))`.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_ta_tok.py -k 'not anchor'
```

Expected: `TextAlignedTokenizer` is missing.

- [ ] **Step 3: Implement the text-aligned codebook**

Keep anchor rows as a persistent frozen buffer. Define:

```python
codebook = F.normalize(codebook_projection(frozen_anchors.float()), dim=-1)
student_z = F.normalize(student_projection(student_features.float()), dim=-1)
codes = nearest_code_indices(student_z, codebook, codebook_chunk_size=2048)
selected = F.embedding(codes, codebook)
quantized = student_z + (selected - student_z).detach()
```

`nearest_code_indices()` scans codebook chunks and tracks the best cosine and
index without allocating `[B*256,65536]`.

- [ ] **Step 4: Implement the student, teacher, and decoder**

Load two independent copies of `siglip2-large-patch16-256`; initialize the
student from the teacher state, freeze/eval the teacher, and leave every
student parameter trainable. Normalize RGB from `[0,1]` with mean/std `0.5`,
resize to 256x256, request hidden states, and use `hidden_states[-2]`.

The reconstruction path projects `Dq -> 1024`, applies three
`nn.TransformerEncoderLayer` blocks with batch-first attention, then projects
to 1,024 frozen-teacher feature dimensions.

Use:

```python
reconstruction_loss = 1 - F.cosine_similarity(pred, teacher.detach(), dim=-1).mean()
commitment_loss = 0.25 * F.mse_loss(student_z, selected.detach())
codebook_loss = F.mse_loss(selected, student_z.detach())
loss = reconstruction_loss + commitment_loss + codebook_loss
```

Return code usage, perplexity, dead-code ratio, and reconstruction cosine in
`TATokOutput.metrics`.

- [ ] **Step 5: Run the TA-Tok tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_ta_tok.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit TA-Tok**

```bash
git add qwen35_planx/ta_tok.py tests/test_qwen35_planx_ta_tok.py
git commit -m "feat(planx): implement text-aligned SigLIP2 tokenizer"
```

---

### Task 5: Add TA-Tok training, validation, and artifact checks

**Files:**
- Create: `qwen35_planx/ta_tok_trainer.py`
- Create: `qwen35_planx/cli/train_ta_tok.py`
- Create: `qwen35_planx/scripts/train_ta_tok_ola.sh`
- Create: `qwen35_planx/cli/preflight.py`
- Test: `tests/test_qwen35_planx_ta_tok_trainer.py`

**Interfaces:**
- Consumes: frame manifests, base Qwen embeddings, and SigLIP2 weights.
- Produces: a self-contained TA-Tok checkpoint with `ta_tok.safetensors`, `metadata.json`, `anchor_ids.npy`, `metrics.json`, and `latest_checkpoint.txt`.

- [ ] **Step 1: Write trainer/checkpoint tests**

Test that both cameras occur in every validation report, validation trajectories
never occur in training, checkpoint hashes change if weights change, code
collapse fails preflight, and resume rejects a different anchor mapping.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_ta_tok_trainer.py
```

Expected: trainer module is missing.

- [ ] **Step 3: Implement distributed training**

Use Accelerate with bf16, deterministic samplers, fused AdamW when available,
gradient clipping at 1.0, warmup plus cosine decay, and one optimizer step per
configured accumulation interval. All frame records from both cameras are
eligible. Log total and per-camera reconstruction cosine, commitment loss,
codebook loss, coverage, perplexity, and dead-code ratio.

Save only on rank zero and write artifacts to a temporary sibling directory
before `os.replace()` makes the checkpoint visible. Do not serialize the
frozen teacher weights; record their source path and hash.

- [ ] **Step 4: Implement strict preflight**

The `ta-tok` preflight loads the checkpoint on CPU and rejects:

- codebook size other than 65,536;
- anchor count/hash mismatch;
- teacher other than project SigLIP2-Large/Patch16/256;
- selected layer other than `-2`;
- output other than `[B,256]`;
- non-finite losses;
- validation coverage below a configurable threshold;
- dead-code ratio above a configurable threshold.

- [ ] **Step 5: Add the OLA launcher**

The launcher accepts all paths through environment variables, defaults to eight
GPUs, uses the predecoded frame manifest, writes only under
`runs/qwen35_planx_ta_tok`, and runs preflight before distributed training.

- [ ] **Step 6: Run tests and shell validation**

Run:

```bash
pytest -q tests/test_qwen35_planx_ta_tok_trainer.py
bash -n qwen35_planx/scripts/train_ta_tok_ola.sh
```

Expected: tests pass and shell syntax is valid.

- [ ] **Step 7: Commit the TA-Tok training stage**

```bash
git add qwen35_planx/ta_tok_trainer.py qwen35_planx/cli/train_ta_tok.py \
  qwen35_planx/cli/preflight.py qwen35_planx/scripts/train_ta_tok_ola.sh \
  tests/test_qwen35_planx_ta_tok_trainer.py
git commit -m "feat(planx): add TA-Tok distributed training"
```

---

### Task 6: Build the dual-camera offline planner cache

**Files:**
- Create: `qwen35_planx/plan_cache.py`
- Create: `qwen35_planx/cli/build_plan_cache.py`
- Create: `qwen35_planx/scripts/build_plan_cache_ola.sh`
- Test: `tests/test_qwen35_planx_plan_cache.py`

**Interfaces:**
- Consumes: `planner_{train,val}.jsonl` and a frozen TA-Tok checkpoint.
- Produces: `PlanCache`, sharded current RGB `[N,2,256,256,3] uint8`, codes `[N,2,4,256] uint16`, instructions, trajectory/window metadata, and cache manifest hashes.

- [ ] **Step 1: Write cache tests**

```python
def test_plan_cache_preserves_camera_and_keyframe_order(tmp_path, fake_ta_tok):
    build_tiny_plan_cache(tmp_path, fake_ta_tok)
    sample = PlanCache(tmp_path, split="train")[0]
    assert sample.current_rgb.shape == (2, 256, 256, 3)
    assert sample.future_codes.shape == (2, 4, 256)
    assert sample.camera_names == ("main", "wrist")
    assert sample.future_frame_indices == (1, 4, 6, 9)
    assert sample.future_codes.dtype == torch.int64
```

Also test `uint16` storage, no train/validation trajectory overlap, tokenizer
hash rejection, camera swap rejection, interrupted-shard recovery, and
deterministic sample ordering.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_plan_cache.py
```

Expected: cache module is missing.

- [ ] **Step 3: Implement atomic sharded cache creation**

For every fixed planner window, load two current frames and eight future
camera/keyframe frames from predecoded memmaps. Encode future frames in a
configurable GPU microbatch, restore `[2,4,256]`, and store code IDs using
`numpy.lib.format.open_memmap`. Store instructions as UTF-8 JSONL and numeric
metadata as structured NumPy arrays. Finalize a shard only after its hashes and
record count are written.

- [ ] **Step 4: Implement strict loading**

`PlanCache` validates the cache manifest against `PlanGeometry`, TA-Tok
checkpoint hash, anchor mapping hash, camera names, image normalization,
trajectory split seed, and shard hashes before exposing samples.

- [ ] **Step 5: Run tests and launcher syntax checks**

Run:

```bash
pytest -q tests/test_qwen35_planx_plan_cache.py
bash -n qwen35_planx/scripts/build_plan_cache_ola.sh
```

Expected: all checks pass.

- [ ] **Step 6: Commit the planner cache**

```bash
git add qwen35_planx/plan_cache.py qwen35_planx/cli/build_plan_cache.py \
  qwen35_planx/scripts/build_plan_cache_ola.sh tests/test_qwen35_planx_plan_cache.py
git commit -m "feat(planx): cache dual-camera TA-token plans"
```

---

### Task 7: Expand Qwen3.5 with the TA visual vocabulary

**Files:**
- Create: `qwen35_planx/vocabulary.py`
- Test: `tests/test_qwen35_planx_vocabulary.py`

**Interfaces:**
- Consumes: an experiment-local processor/model and trained `[65536,Dq]` TA-Tok codebook.
- Produces: `VisualTokenLayout`, `install_visual_vocabulary()`, and `validate_visual_vocabulary()`.

- [ ] **Step 1: Write vocabulary tests**

```python
def test_visual_rows_are_contiguous_and_initialized(fake_processor, fake_qwen, codebook):
    layout = install_visual_vocabulary(fake_processor, fake_qwen, codebook)
    assert layout.visual_token_ids == tuple(
        range(layout.visual_token_start, layout.visual_token_start + 65536)
    )
    actual = fake_qwen.get_input_embeddings().weight[layout.visual_token_start:layout.visual_token_end]
    torch.testing.assert_close(actual.float(), codebook.float())
    assert fake_qwen.get_input_embeddings().weight.data_ptr() == (
        fake_qwen.get_output_embeddings().weight.data_ptr()
    )
```

Also test that base rows are bit-identical, reinstall is rejected, a wrong
codebook width fails, and saving never writes into the base model directory.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_vocabulary.py
```

Expected: vocabulary module is missing.

- [ ] **Step 3: Implement experiment-local expansion**

Add regular tokens `<|ta_00000|>` through `<|ta_65535|>` in code order. Add
special plan/frame boundary and camera tokens. Call
`resize_token_embeddings(len(tokenizer), mean_resizing=False)`, preserve all
base rows, copy the trained codebook into the tied input/output TA rows, and
initialize structural rows from the mean of non-special base text rows.

Record every new ID, the base vocabulary size/hash, tied-embedding status, and
the codebook hash in `VisualTokenLayout`.

- [ ] **Step 4: Run the vocabulary tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_vocabulary.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit vocabulary expansion**

```bash
git add qwen35_planx/vocabulary.py tests/test_qwen35_planx_vocabulary.py
git commit -m "feat(planx): expand Qwen3.5 visual vocabulary"
```

---

### Task 8: Build exact causal sequences and loss masks

**Files:**
- Create: `qwen35_planx/sequence.py`
- Test: `tests/test_qwen35_planx_sequence.py`

**Interfaces:**
- Consumes: `VisualTokenLayout`, current RGB, instructions, camera IDs, and `[4,256]` code targets.
- Produces: `build_response_ids()`, `response_schedule()`, `DualCameraPlanCollator`, `code_label_positions`, and `code_prediction_positions`.

- [ ] **Step 1: Write format and masking tests**

```python
def test_response_has_exact_planx_structure(layout):
    codes = torch.arange(1024).reshape(4, 256) % 65536
    response = build_response_ids(codes, layout)
    assert len(response) == 1034
    assert response[0] == layout.plan_start_id
    assert response[-1] == layout.plan_end_id
    assert len(code_label_positions(response, layout)) == 1024


def test_collator_masks_prompt_and_keeps_only_assistant_plan(layout, fake_processor):
    batch = DualCameraPlanCollator(fake_processor, layout)([fake_dual_camera_sample()])
    assert batch["input_ids"].shape[0] == 2
    assert torch.all(batch["labels"][batch["prompt_mask"]] == -100)
    assert torch.all(batch["labels"][batch["response_mask"]] != -100)
    assert batch["camera_index"].tolist() == [0, 1]
```

Also assert main and wrist examples have separate sequences, no camera token
appears in the other stream, raster code order is unchanged, padding is masked,
and each code prediction position is exactly one token before its label
position under causal LM shifting.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_sequence.py
```

Expected: sequence module is missing.

- [ ] **Step 3: Implement the response schedule**

Emit:

```text
plan_start,
frame_start, 256 visual IDs, frame_end,
frame_start, 256 visual IDs, frame_end,
frame_start, 256 visual IDs, frame_end,
frame_start, 256 visual IDs, frame_end,
plan_end
```

The schedule marks each position as `forced_boundary` or `visual_code` and
records `(keyframe,y,x)` for code positions.

- [ ] **Step 4: Implement the dual-camera collator**

Create two independent Qwen conversations per dataset sample:

```text
<current image>
Instruction: ...
Camera: <main_camera> or <wrist_camera>
Predict four future semantic keyframes at offsets 1, 4, 6, and 9.
```

Use the native Qwen processor for the prompt/image, append response token IDs,
extend every sequence-shaped processor field, pad, and set labels to `-100`
outside the assistant response. Return `sample_index`, `camera_index`,
`code_label_positions`, and `code_prediction_positions`.

- [ ] **Step 5: Run the sequence tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_sequence.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit sequence construction**

```bash
git add qwen35_planx/sequence.py tests/test_qwen35_planx_sequence.py
git commit -m "feat(planx): build causal dual-camera plan sequences"
```

---

### Task 9: Implement the Qwen3.5 causal planner and trainer

**Files:**
- Create: `qwen35_planx/planner.py`
- Create: `qwen35_planx/cli/train_semantic_planner.py`
- Create: `qwen35_planx/scripts/train_semantic_planner_ola.sh`
- Test: `tests/test_qwen35_planx_planner.py`

**Interfaces:**
- Consumes: expanded Qwen3.5, `DualCameraPlanCollator` batches, and `PlanCache`.
- Produces: `Qwen35PlanXPlanner`, `PlannerOutput`, `save_planner_checkpoint()`, and `load_planner_checkpoint()`.

- [ ] **Step 1: Write causal and hidden-state tests**

```python
def test_planner_has_no_query_module(planner):
    names = dict(planner.named_modules())
    assert not any("query" in name and "q_proj" not in name for name in names)


def test_future_targets_do_not_change_earlier_predictions(tiny_planner, batch):
    changed = clone_batch_with_codes_changed_after(batch, code_position=400)
    first = tiny_planner(batch, return_code_hidden_states=True)
    second = tiny_planner(changed, return_code_hidden_states=True)
    torch.testing.assert_close(
        first.code_logits[:, :400],
        second.code_logits[:, :400],
    )


def test_pre_head_hidden_layout(planner, batch):
    output = planner(batch, return_code_hidden_states=True)
    assert output.code_hidden_states.shape == (1, 2, 4, 256, planner.hidden_size)
```

Also test loss masking, structural-token CE, finite one-batch overfit, full
vision/language/new-vocabulary trainability, and strict checkpoint metadata.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_planner.py
```

Expected: planner module is missing.

- [ ] **Step 3: Implement the planner wrapper**

Call `qwen.model(...)` with native causal attention and do not create a
connector, query bank, or parallel code classifier. At the 1,024 code
prediction positions, multiply hidden states only by the tied contiguous TA
rows and compute cross-entropy over the 65,536 IDs allowed by constrained
decoding. At the ten structural prediction positions, multiply by the complete
tied output matrix and compute ordinary full-vocabulary cross-entropy. This is
the exact constrained likelihood while avoiding full 313k-vocabulary logits
at every visual position.

Gather the last hidden states from the same causal positions whose logits
predict the visual labels and reshape by `sample_index/camera_index` to
`[B,2,4,256,Dq]`.

Return:

```python
@dataclass
class PlannerOutput:
    loss: torch.Tensor
    token_cross_entropy: torch.Tensor
    code_logits: torch.Tensor
    code_hidden_states: torch.Tensor | None
```

- [ ] **Step 4: Implement distributed training**

Use Accelerate/ZeRO-2, bf16, trainable Qwen vision and language towers, tied
visual rows, gradient clipping at 1.0, warmup ratio 0.03, cosine decay, and
drop-last distributed sampling. The production launcher defaults to eight
GPUs, per-device batch 2, accumulation 16, global batch 256, 30,000 optimizer
steps, and checkpoints every 5,000 steps. These defaults remain overridable
for validated larger per-device batches.

Log total/code/structure CE, code accuracy by camera/keyframe, learning rate,
tokens/s, GPU memory, and step time.

- [ ] **Step 5: Save self-contained checkpoints**

Save the experiment-local processor, expanded Qwen model, visual token layout,
planner metadata, TA-Tok metadata/hash, optimizer/scheduler state, and metrics.
Loading validates all hashes and geometry before constructing the model.

- [ ] **Step 6: Run tests and shell validation**

Run:

```bash
pytest -q tests/test_qwen35_planx_planner.py
bash -n qwen35_planx/scripts/train_semantic_planner_ola.sh
```

Expected: tests pass and launcher syntax is valid.

- [ ] **Step 7: Commit the causal planner**

```bash
git add qwen35_planx/planner.py qwen35_planx/cli/train_semantic_planner.py \
  qwen35_planx/scripts/train_semantic_planner_ola.sh \
  tests/test_qwen35_planx_planner.py
git commit -m "feat(planx): train Qwen3.5 causal semantic planner"
```

---

### Task 10: Add constrained decoding, metrics, and visualizations

**Files:**
- Create: `qwen35_planx/decoding.py`
- Create: `qwen35_planx/evaluation.py`
- Create: `qwen35_planx/cli/build_generated_plan_cache.py`
- Create: `qwen35_planx/cli/evaluate_semantic_planner.py`
- Create: `qwen35_planx/cli/visualize_semantic_planner.py`
- Create: `qwen35_planx/scripts/build_generated_plan_cache_ola.sh`
- Test: `tests/test_qwen35_planx_decoding.py`

**Interfaces:**
- Consumes: a planner checkpoint and current RGB/instruction samples.
- Produces: `DecodedPlan` containing `code_ids [B,2,4,256]`, `hidden_states [B,2,4,256,Dq]`, confidences, validity, and latency.

- [ ] **Step 1: Write decoder tests**

Test that boundary steps are forced, visual steps can select only the 65,536 TA
rows, each camera uses a separate cache element, output has exactly 1,024 codes
per camera, malformed output raises `InvalidPlanGeneration`, and hidden states
align with the logits that selected each returned code.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_decoding.py
```

Expected: decoding module is missing.

- [ ] **Step 3: Implement KV-cached constrained generation**

Decode the two cameras as `B*2`. At forced boundaries, append the deterministic
token ID without sampling. At visual positions, slice logits to
`[visual_token_start:visual_token_end]`, apply temperature/top-k only inside
that slice, select the local code ID, and append its global token ID. Reuse the
Qwen cache at every step and record the pre-head hidden state and selected
probability.

- [ ] **Step 4: Implement evaluation**

Report token CE/accuracy, suite/camera/keyframe/spatial accuracy, decoded
teacher-feature cosine through `TA-Tok.decode_codes()`, temporal consistency,
validity rate, latency, and tokens/s. Save machine-readable JSON and CSV.

- [ ] **Step 5: Build the compact generated-plan cache**

Run constrained decoding once for each planner window and store only generated
`[2,4,256] uint16` code IDs, confidence summaries, validity, and lookup keys
`(domain, episode_index, current_index)`. Reuse the strict ground-truth
`PlanCache` manifest contract with an added planner checkpoint hash and
`target_source: generated`. The cache builder resumes only complete shards and
never stores Qwen hidden states.

- [ ] **Step 6: Implement split-camera visualization**

For each selected sample, write separate main/wrist 224x224 panels containing
current RGB, four future RGB frames, target semantic PCA maps, predicted
semantic PCA maps, and causal attention heatmaps. Do not concatenate cameras
before model input. For attention visualization only, switch selected
Qwen3.5 full-attention layers to the eager attention implementation, request
attention tensors, gather attention from visual prediction positions to
current-image token positions, average heads/layers, restore the Qwen image
patch grid, and resize the map to 224x224. Training and ordinary evaluation
remain on SDPA/Flash Attention and do not materialize attention matrices.

- [ ] **Step 7: Run decoder tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_decoding.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit inference and evaluation**

```bash
git add qwen35_planx/decoding.py qwen35_planx/evaluation.py \
  qwen35_planx/cli/build_generated_plan_cache.py \
  qwen35_planx/cli/evaluate_semantic_planner.py \
  qwen35_planx/cli/visualize_semantic_planner.py \
  qwen35_planx/scripts/build_generated_plan_cache_ola.sh \
  tests/test_qwen35_planx_decoding.py
git commit -m "feat(planx): decode and evaluate causal semantic plans"
```

---

### Task 11: Add the optional GE-Act Plan-X provider

**Files:**
- Create: `qwen35_planx/provider.py`
- Modify: `ge_act/models/ltx_models/semantic_conditioning.py`
- Modify: `ge_act/runner/ge_trainer.py`
- Test: `tests/test_qwen35_planx_provider.py`
- Test: `tests/test_ge_act_qwen35_planx.py`

**Interfaces:**
- Consumes: a strict planner checkpoint, current `[B,2,3,H,W]` RGB, instructions, and optional ground-truth/generated code IDs.
- Produces: `SemanticPlanBatch` with `hidden_states`, `code_ids`, `confidences`, `token_cross_entropy`, and `validity`.
- Produces: `PlanXSemanticProvider.predict()`, `.teacher_force()`, and `.embed_codes()`, each returning `SemanticPlanBatch` whose hidden-state layout is `[B,2,4,256,Dq]`.
- Produces: `build_semantic_condition_source()` in GE-Act with default backend `siglip2_gt`.

- [ ] **Step 1: Write provider and backend-gating tests**

```python
def test_provider_returns_two_independent_camera_plans(fake_planner_checkpoint):
    provider = PlanXSemanticProvider.from_pretrained(fake_planner_checkpoint)
    result = provider.predict(torch.rand(2, 2, 3, 224, 224), ["a", "b"])
    assert result.hidden_states.shape == (2, 2, 4, 256, provider.hidden_size)
    assert result.code_ids.shape == (2, 2, 4, 256)


def test_default_ge_act_backend_does_not_import_qwen35(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", fail_on_qwen35)
    source = build_semantic_condition_source({"enabled": True, "planner_backend": "siglip2_gt"})
    assert isinstance(source, OnlineSiglip2SemanticEncoder)
```

Also test checkpoint rejection, camera order, frozen/trainable provider modes,
same-camera output sensitivity, and unchanged GE-Act behavior when semantics
are disabled.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_provider.py tests/test_ge_act_qwen35_planx.py
```

Expected: provider and backend factory are missing.

- [ ] **Step 3: Implement provider modes**

- `embed_codes(codes)`: return tied visual token embeddings for ground-truth
  TA codes.
- `teacher_force(current_rgb, instructions, codes)`: run one causal Qwen
  forward and return prediction-aligned pre-head hidden states plus CE.
- `predict(current_rgb, instructions)`: run exact constrained decoding and
  return generated codes/hidden states/confidences.

All three methods validate `[B,2,4,256]` and camera order.

- [ ] **Step 4: Add explicit GE-Act backend construction**

If `semantic_plan.planner_backend` is absent or `siglip2_gt`, preserve the
current `OnlineSiglip2SemanticEncoder` path exactly. Only
`planner_backend: qwen35_planx` imports and constructs the new provider.
Configure `semantic_plan_in_dim` from the checkpoint's Qwen hidden width and
reject a mismatched LTX adapter.

- [ ] **Step 5: Preserve same-camera attention and 3D RoPE**

Do not change `LTXVideoSemanticAttentionProcessor2_0`. Route provider hidden
states through the existing `SemanticContextAdapter`, keyframe times, camera
embeddings, query RoPE, key RoPE, and semantic condition mask. Add assertions
that the provider batch/view layout equals the flattened LTX camera layout.

- [ ] **Step 6: Run provider and existing semantic tests**

Run:

```bash
pytest -q tests/test_qwen35_planx_provider.py tests/test_ge_act_qwen35_planx.py \
  tests/test_ge_act_semantic_conditioning.py tests/test_ge_act_siglip2_config.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit provider integration**

```bash
git add qwen35_planx/provider.py ge_act/models/ltx_models/semantic_conditioning.py \
  ge_act/runner/ge_trainer.py tests/test_qwen35_planx_provider.py \
  tests/test_ge_act_qwen35_planx.py
git commit -m "feat(ge-act): add optional Qwen3.5 Plan-X provider"
```

---

### Task 12: Implement staged and joint GE-Act training

**Files:**
- Modify: `ge_act/data/lerobot_like_dataset.py`
- Modify: `ge_act/runner/ge_trainer.py`
- Create: `ge_act/configs/ltx_model/libero/action_model_libero_qwen35_planx.yaml`
- Create: `ge_act/scripts/train_ltx_qwen35_planx.sh`
- Extend: `tests/test_ge_act_qwen35_planx.py`

**Interfaces:**
- Consumes: `semantic_plan.mode` in `{"gt_embedding", "teacher_forced_hidden", "generated_hidden", "joint"}`.
- Produces: optional dataset plan lookup keys and `compute_joint_planx_loss(ge_act_loss, planner_ce, weight=0.1)`.

- [ ] **Step 1: Write staged-mode and joint-loss tests**

```python
def test_joint_loss_has_exact_weight():
    ge_loss = torch.tensor(2.0)
    ta_ce = torch.tensor(3.0)
    assert compute_joint_planx_loss(ge_loss, ta_ce, weight=0.1).item() == pytest.approx(2.3)


def test_dataset_metadata_is_opt_in(fake_dataset):
    legacy = fake_dataset(return_plan_metadata=False)[0]
    planx = fake_dataset(return_plan_metadata=True)[0]
    assert "plan_lookup" not in legacy
    assert set(planx["plan_lookup"]) == {"domain", "episode_index", "current_index"}
```

Also test optimizer membership for frozen and joint planner modes, semantic
dropout shared across views, provider gradient flow in joint mode, action
expert still enabled, and exact stage-mode dispatch.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_ge_act_qwen35_planx.py -k 'joint or metadata or mode'
```

Expected: staged mode support is missing.

- [ ] **Step 3: Add optional dataset lookup metadata**

When `return_plan_metadata` is true, return domain, episode index, and the
current frame index selected by `get_frame_indexes()`. Keep the default false
so all existing dataloader outputs are unchanged.

- [ ] **Step 4: Implement the four semantic modes**

- `gt_embedding`: look up/encode target TA codes and call `embed_codes()`.
- `teacher_forced_hidden`: run the frozen planner once with ground-truth codes.
- `generated_hidden`: load cached generated codes, then run one frozen
  teacher-forced Qwen pass over those generated codes to reconstruct their
  causal hidden states without a 1,034-step loop per GE batch.
- `joint`: run trainable Qwen with ground-truth codes, condition GE-Act on its
  causal prediction hidden states, and retain planner CE.

Generated code caches use `uint16` and the same strict plan-cache metadata;
they never store `[Dq]` hidden tensors.

- [ ] **Step 5: Extend optimizer/Accelerate handling**

Frozen modes exclude planner parameters. Joint mode adds a named planner
parameter group with `semantic_plan.planner_lr`, includes the provider model in
`accelerator.prepare()` and `accelerator.accumulate()`, clips both trainable
models, and saves/restores both states.

Compute:

```python
ge_act_loss = loss_video + action_loss_scale * loss_action
loss = ge_act_loss if planner_ce is None else ge_act_loss + 0.1 * planner_ce
```

- [ ] **Step 6: Add the full action-training config**

Start from the current LIBERO FastWAM action config. Keep `action_expert: true`,
both cameras, predecoded RGB, the existing LTX/action learning rates, 3D RoPE,
and semantic cross-attention blocks. Add only the explicit Qwen3.5 backend,
checkpoint/cache paths, mode, planner LR, hidden width, and joint CE weight.

- [ ] **Step 7: Run the complete focused suite**

Run:

```bash
pytest -q tests/test_ge_act_qwen35_planx.py tests/test_qwen35_planx_provider.py \
  tests/test_ge_act_semantic_conditioning.py tests/test_ge_act_source_completeness.py \
  tests/test_ge_act_siglip2_config.py tests/test_qwen3vl2b_legacy_unchanged.py
bash -n ge_act/scripts/train_ltx_qwen35_planx.sh
```

Expected: all tests and shell checks pass.

- [ ] **Step 8: Commit staged and joint training**

```bash
git add ge_act/data/lerobot_like_dataset.py ge_act/runner/ge_trainer.py \
  ge_act/configs/ltx_model/libero/action_model_libero_qwen35_planx.yaml \
  ge_act/scripts/train_ltx_qwen35_planx.sh tests/test_ge_act_qwen35_planx.py
git commit -m "feat(ge-act): train with Qwen3.5 Plan-X semantics"
```

---

### Task 13: Run end-to-end preflight and one-batch overfit

**Files:**
- Extend: `qwen35_planx/cli/preflight.py`
- Create: `qwen35_planx/scripts/preflight_ola.sh`
- Create: `tests/test_qwen35_planx_end_to_end.py`

**Interfaces:**
- Consumes: all three stage artifacts.
- Produces: a JSON preflight report and a launch/no-launch exit status.

- [ ] **Step 1: Write the end-to-end fake-artifact test**

The test constructs a tiny two-camera cache, a reduced-size test TA-Tok
configuration, tiny causal Qwen, and tiny LTX adapter. Production metadata
validation is tested separately with the required 65,536-code geometry. The
test verifies cache-to-collator-to-planner-to-provider-to-GE shapes and one
optimizer update with finite gradients.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q tests/test_qwen35_planx_end_to_end.py
```

Expected: end-to-end preflight entry point is missing.

- [ ] **Step 3: Implement production preflight**

Check:

- all base/checkpoint/cache paths exist and are outside legacy output trees;
- tracked legacy Qwen3-VL-2B hashes are unchanged;
- manifests have no trajectory leakage;
- TA-Tok passes shape/hash/collapse checks;
- planner vocabulary and tied rows match TA-Tok;
- a real cache batch produces finite CE and `[B,2,4,256,Dq]`;
- constrained generation is structurally valid;
- GE-Act accepts provider hidden states with same-camera layout;
- action output remains present;
- one-batch planner overfit decreases CE;
- one joint GE/planner optimizer step has finite gradients.

Write exact versions, git SHA, artifact hashes, memory, timings, and every
check result to JSON. Exit nonzero on any failed check.

- [ ] **Step 4: Run all new tests and legacy regressions**

Run:

```bash
pytest -q tests/test_qwen35_planx_*.py tests/test_ge_act_qwen35_planx.py \
  tests/test_ge_act_semantic_conditioning.py tests/test_ge_act_siglip2_config.py \
  tests/test_qwen3vl2b_legacy_unchanged.py
bash -n qwen35_planx/scripts/*.sh ge_act/scripts/train_ltx_qwen35_planx.sh
git diff --check
```

Expected: all tests pass, all launchers parse, and the diff is clean.

- [ ] **Step 5: Commit end-to-end preflight**

```bash
git add qwen35_planx/cli/preflight.py qwen35_planx/scripts/preflight_ola.sh \
  tests/test_qwen35_planx_end_to_end.py
git commit -m "test(planx): add end-to-end training preflight"
```

---

## Production Execution Order

After the implementation passes locally:

1. Build trajectory/frame/window manifests from the existing predecoded LIBERO
   RGB cache.
2. Train TA-Tok on all train-split frames from both cameras.
3. Reject collapsed TA-Tok checkpoints and select the best validation
   reconstruction/coverage checkpoint.
4. Build ground-truth dual-camera planner caches with that frozen tokenizer.
5. Run planner one-batch overfit and one-GPU constrained-generation smoke.
6. Train the eight-GPU Qwen3.5 causal planner.
7. Evaluate token, semantic, temporal, validity, and latency metrics.
8. Train GE-Act first with ground-truth embeddings, then generated-plan hidden
   states, then joint Qwen/GE-Act fine-tuning.
9. Evaluate no-semantic, K-means/query, Plan-X, and ground-truth upper-bound
   variants on the same LIBERO split.

Do not start a later stage from an artifact that fails its preflight.
