# Qwen3.5 Video-Hindsight Grounded Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a released-384-TA-Tok Qwen3.5 planner that predicts four spatial future plans per LIBERO camera, learns text-grounded relevance from full-video hindsight, and jointly conditions GE-Act video and action experts.

**Architecture:** A frozen released TA-Tok supplies 27x27 visual code targets. An offline HDF5 target builder combines SigLIP2 gradient relevance, DINOv3 correspondence, action/gripper phases, and counterfactual text into soft source/target/action maps. Qwen3.5 predicts future codes, post-code semantic hidden states, and SigLIP2-aligned phrase anchors without an inference-time teacher; a 64-coverage-plus-32-top-token compressor supplies same-camera features and bounded relevance bias to the existing zero-gated LTX semantic branch.

**Tech Stack:** Python 3.10+, PyTorch 2.7+, Transformers 5.x with `Qwen3_5ForConditionalGeneration`, SigLIP2-SO400M/Patch14/384, DINOv3 ViT-H+/16, HDF5, NumPy memmaps, safetensors, Accelerate, DeepSpeed ZeRO-2, GE-Act LTX, pytest.

## Global Constraints

- Execute in a new worktree created with `superpowers:using-git-worktrees`; do not modify the dirty `.worktrees/qwen35-planx-implementation` tree.
- Base the implementation branch on `ge-act-dual-camera-planner` so the validated HDF5 loader, dual-camera planner integration, and GE-Act fixes are present.
- Cherry-pick only `066b476` and `fcdfae3` from `qwen35-planx-implementation`, then cherry-pick design commit `0b44f72`; do not cherry-pick `c629b55`, `de98cbe`, or the untracked TA-Tok trainer files.
- Preserve all behavior under `qwen3_vl_semantic_planner`; the new path lives under `qwen35_planx`.
- Use released TA-Tok geometry exactly: image size 384, grid 27x27, 729 codes per frame, four future frames, codebook size 65,536, code width 1,536.
- Keep Qwen3.5 hidden width 2,048 and SigLIP2 text projection width 1,152.
- Do not add a 1,536-to-2,048 projection for Qwen visual-vocabulary initialization.
- Main and wrist are separate effective `B*2` Qwen examples with shared weights and independent causal state.
- Use the GE-Act HDF5 image/action/state data; no MP4 decoding is allowed in cache building or training.
- Full video, DINOv3, future RGB, actions, and gripper phase are teacher-only and must be absent from inference dependencies.
- Planner stage: 30,000 optimizer steps, effective global batch 256, 1,000 warmup steps, cosine decay, checkpoint every 5,000 steps.
- Joint stage: 30,000 optimizer steps; LTX LR `2e-5`, action expert LR `1e-4`, semantic heads/adapters LR `5e-5`, top-eight Qwen layers LR `1e-6`, Qwen vision LR `5e-7`.
- Keep released TA-Tok, SigLIP2, and DINOv3 frozen in every stage.
- Use `apply_patch` for source edits and commit only files belonging to the current task.
- Design reference: `docs/superpowers/specs/2026-07-26-qwen35-video-hindsight-grounded-planner-design.md`.

## Execution Base

At execution time, after invoking `superpowers:using-git-worktrees`, create the
clean integration branch:

```bash
git worktree add .worktrees/qwen35-video-hindsight-grounding \
  -b qwen35-video-hindsight-grounding ge-act-dual-camera-planner
git -C .worktrees/qwen35-video-hindsight-grounding cherry-pick 066b476 fcdfae3
git -C .worktrees/qwen35-video-hindsight-grounding cherry-pick 0b44f72
```

Run all commands below from
`.worktrees/qwen35-video-hindsight-grounding`. Before Task 1, verify:

```bash
git status --short
pytest -q tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_qwen35_planx_config.py \
  tests/test_qwen35_planx_libero_data.py
```

Expected: a clean worktree and all selected baseline tests pass.

## File Structure

Create or revise the following focused units:

```text
qwen35_planx/
  config.py                    Immutable 384/K4/cache/checkpoint contracts.
  hashing.py                   Existing canonical SHA-256 helpers.
  instruction.py               LIBERO action/source/target parsing and negatives.
  libero_data.py               Existing trajectory-safe NPY path, kept as legacy.
  hindsight_data.py            Fixed HDF5 window records and full-trajectory reads.
  hindsight_schema.py          Atomic shard and finalized memmap cache format.
  official_ta_tok.py           Safe released-checkpoint loader and frozen adapter.
  siglip_relevance.py          Frozen phrase embeddings and gradient relevance.
  temporal_grounding.py        DINO tracking, phase priors, and soft-map fusion.
  hindsight_builder.py         Per-window teacher orchestration.
  vocabulary.py                Experiment-local Qwen visual vocabulary.
  sequence.py                  Camera-specific prompt/response and causal indices.
  planner_dataset.py           Hindsight-cache dataset and Qwen collator.
  losses.py                    Chunked visual CE and five planner objectives.
  planner.py                   Qwen backbone, pre/post states, prediction heads.
  decoding.py                  Constrained KV-cache autoregressive generation.
  compression.py               8x8 coverage plus exact top-32 compression.
  provider.py                  Frozen/trainable planner interface for GE-Act.
  evaluation.py                Offline metrics and ablation records.
  cli/
    build_hindsight_cache.py
    train_semantic_planner.py
    evaluate_semantic_planner.py
    visualize_semantic_planner.py
    preflight.py
  scripts/
    build_hindsight_cache_ola.sh
    train_semantic_planner_ola.sh

ge_act/
  data/libero_hindsight_hdf5_dataset.py
  models/ltx_models/semantic_conditioning.py
  models/ltx_models/transformer_ltx_multiview.py
  models/ltx_models/vlm_semantic_planner.py
  runner/ge_trainer.py
  configs/ltx_model/libero/action_model_libero_qwen35_grounded_hdf5.yaml
  scripts/train_ltx_qwen35_grounded.sh
  scripts/sbatch_train_ltx_qwen35_grounded_hpc3.sh

tests/
  test_qwen35_grounded_config.py
  test_qwen35_instruction.py
  test_qwen35_official_ta_tok.py
  test_qwen35_hindsight_schema.py
  test_qwen35_siglip_relevance.py
  test_qwen35_temporal_grounding.py
  test_qwen35_hindsight_builder.py
  test_qwen35_vocabulary.py
  test_qwen35_sequence.py
  test_qwen35_planner_dataset.py
  test_qwen35_grounded_losses.py
  test_qwen35_grounded_planner.py
  test_qwen35_decoding.py
  test_qwen35_compression.py
  test_qwen35_provider.py
  test_ge_act_qwen35_grounded.py
  test_qwen35_grounded_evaluation.py
```

---

### Task 1: Replace the Superseded Contracts and Add Structured LIBERO Text

**Files:**
- Modify: `qwen35_planx/config.py`
- Create: `qwen35_planx/instruction.py`
- Modify: `qwen35_planx/__init__.py`
- Create: `tests/test_qwen35_grounded_config.py`
- Create: `tests/test_qwen35_instruction.py`
- Modify: `tests/test_qwen35_planx_config.py`

**Interfaces:**
- Consumes: existing `sha256_json()` from `qwen35_planx.hashing`.
- Produces: `PlanGeometry`, `ReleasedTATokMetadata`,
  `HindsightCacheMetadata`, `GroundedPlannerMetadata`,
  `InstructionFields`, `InstructionVocabulary`,
  `parse_libero_instruction()`, `format_grounded_prompt()`, and
  `build_counterfactuals()`.

- [ ] **Step 1: Write failing 384-geometry and metadata tests**

Add tests that lock the new immutable geometry:

```python
def test_released_geometry_is_384_k4_27_square() -> None:
    from qwen35_planx.config import PlanGeometry

    geometry = PlanGeometry()
    assert geometry.image_size == 384
    assert geometry.grid_size == 27
    assert geometry.tokens_per_frame == 729
    assert geometry.tokens_per_camera == 2916
    assert geometry.tokens_per_sample == 5832
    assert geometry.visual_vocab_size == 65_536
    assert geometry.ta_code_dim == 1536
    assert geometry.qwen_hidden_dim == 2048
    assert geometry.text_align_dim == 1152
    assert geometry.ge_act_future_indices == (0, 3, 5, 8)


def test_released_ta_metadata_has_no_qwen_anchor_fields() -> None:
    from qwen35_planx.config import ReleasedTATokMetadata

    metadata = ReleasedTATokMetadata.example()
    payload = metadata.to_dict()
    assert payload["teacher"] == "google/siglip2-so400m-patch14-384"
    assert payload["checkpoint_hash"]
    assert "anchor_token_ids" not in payload
    assert "anchor_embedding_hash" not in payload
```

Add rejection tests for image size 256, grid 16, missing cache hashes, camera
reordering, and hidden-alignment values other than
`"pre_predicts_code_post_conditions_semantics"`.

- [ ] **Step 2: Run the contract tests and verify the old design fails**

Run:

```bash
pytest -q tests/test_qwen35_grounded_config.py tests/test_qwen35_planx_config.py
```

Expected: failures showing the existing 256/16x16 contracts and anchor fields
do not satisfy the released-checkpoint contract.

- [ ] **Step 3: Replace the config contracts**

Set exact constants:

```python
_IMAGE_SIZE = 384
_GRID_SIZE = 27
_NUM_KEYFRAMES = 4
_VISUAL_VOCAB_SIZE = 65_536
_TA_CODE_DIM = 1536
_QWEN_HIDDEN_DIM = 2048
_TEXT_ALIGN_DIM = 1152
_GE_ACT_FUTURE_INDICES = (0, 3, 5, 8)
_GROUNDING_ROLES = ("source", "target", "action")
```

Every three-role tensor, confidence tuple, phrase embedding table, loss, and
visualization uses `_GROUNDING_ROLES` in exactly this order, independent of
the textual marker order in the formatted prompt.

`ReleasedTATokMetadata` must validate:

```python
tokenizer_type == "released_ta_tok"
teacher == "google/siglip2-so400m-patch14-384"
image_size == 384
grid_size == 27
bottleneck_token_num == 729
codebook_size == 65_536
codebook_dim == 1536
selected_layer == -2
pool_scale == 1
```

`HindsightCacheMetadata` must include hashes for the HDF5 manifest, window
manifest, instruction parser, TA-Tok, SigLIP2, DINOv3, and preprocessing.
`GroundedPlannerMetadata` must record visual-token ID bounds, structure-token
IDs, loss weights, phrase roles, `h_pre`/`h_post` alignment, and the exact
hindsight-cache hash. It rejects any `phrase_roles` value other than
`("source", "target", "action")`.

- [ ] **Step 4: Write failing instruction parsing and counterfactual tests**

Use actual LIBERO-style examples:

```python
def test_parse_pick_place_instruction() -> None:
    from qwen35_planx.instruction import parse_libero_instruction

    fields = parse_libero_instruction(
        "pick up the black bowl between the plate and the ramekin"
    )
    assert fields.action == "pick up and place"
    assert fields.source == "the black bowl"
    assert fields.target == "between the plate and the ramekin"
    assert fields.confidences == (1.0, 1.0, 1.0)


def test_counterfactual_changes_exactly_one_field() -> None:
    from qwen35_planx.instruction import (
        InstructionVocabulary,
        build_counterfactuals,
        parse_libero_instruction,
    )

    fields = parse_libero_instruction("put the black bowl on the plate")
    vocab = InstructionVocabulary(
        actions=("put", "open"),
        sources=("the black bowl", "the white bowl"),
        targets=("on the plate", "in the basket"),
    )
    negatives = build_counterfactuals(fields, vocab, max_per_field=1)
    assert [item.changed_field for item in negatives] == [
        "action",
        "source",
        "target",
    ]
    assert all(item.fields != fields for item in negatives)
```

Also test drawer, stove, basket, spatial-relation, missing-field, idempotent
formatting, and deterministic vocabulary ordering.

- [ ] **Step 5: Implement structured parsing and prompting**

Define immutable records:

```python
@dataclass(frozen=True)
class InstructionFields:
    original: str
    action: str
    source: str
    target: str
    confidences: tuple[float, float, float]


@dataclass(frozen=True)
class CounterfactualInstruction:
    changed_field: str
    fields: InstructionFields
```

Use ordered LIBERO template regexes. Normalize whitespace and punctuation but
retain the original instruction. `format_grounded_prompt()` must emit:

```text
<ACT>{action}</ACT>
<SRC>{source}</SRC>
<TGT>{target}</TGT>
Instruction: {original}
<SRC_QUERY><TGT_QUERY><ACT_QUERY>
Predict four future semantic frames.
```

When a template cannot identify one field, preserve the original instruction,
use an empty field string, and set only that field's confidence to zero.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
pytest -q tests/test_qwen35_grounded_config.py \
  tests/test_qwen35_instruction.py \
  tests/test_qwen35_planx_config.py \
  tests/test_qwen3vl2b_legacy_unchanged.py
```

Expected: all pass.

Commit:

```bash
git add qwen35_planx/config.py qwen35_planx/instruction.py \
  qwen35_planx/__init__.py tests/test_qwen35_grounded_config.py \
  tests/test_qwen35_instruction.py tests/test_qwen35_planx_config.py
git commit -m "feat(planx): define released grounded planner contracts"
```

---

### Task 2: Load the Released TA-Tok Safely and Export Its Codebook

**Files:**
- Create: `qwen35_planx/official_ta_tok.py`
- Create: `tests/test_qwen35_official_ta_tok.py`
- Modify: `qwen35_planx/cli/preflight.py`
- Delete if present in the execution tree:
  `qwen35_planx/anchors.py`,
  `qwen35_planx/ta_tok_trainer.py`,
  `qwen35_planx/cli/train_ta_tok.py`,
  `tests/test_qwen35_planx_ta_tok_trainer.py`

**Interfaces:**
- Consumes: `ReleasedTATokMetadata`, a released `ta_tok.pth`, and local
  SigLIP2-SO400M weights.
- Produces: `ReleasedTATok.from_checkpoint()`,
  `ReleasedTATok.encode_codes()`, `ReleasedTATok.lookup_codes()`,
  `ReleasedTATok.decode_features()`, and `export_codebook_safetensors()`.

- [ ] **Step 1: Write failing checkpoint-contract tests**

Use a small synthetic checkpoint with the released key layout:

```python
def test_released_adapter_exposes_codes_and_codebook(fake_released_checkpoint) -> None:
    from qwen35_planx.official_ta_tok import ReleasedTATok

    tokenizer = ReleasedTATok.from_checkpoint(
        fake_released_checkpoint.path,
        encoder_factory=fake_released_checkpoint.encoder_factory,
        weights_only=True,
    )
    images = torch.zeros(2, 3, 384, 384)
    output = tokenizer.encode_codes(images)
    assert output.codes.shape == (2, 729)
    assert output.codes.dtype == torch.long
    assert tokenizer.codebook.shape == (65_536, 1536)
    assert not any(parameter.requires_grad for parameter in tokenizer.parameters())
```

Test that 256 input metadata, 16x16 output, wrong codebook shape, missing keys,
unsafe `weights_only=False`, and non-finite codebook values fail before GPU
allocation.

- [ ] **Step 2: Run the tests and verify the adapter is absent**

Run:

```bash
pytest -q tests/test_qwen35_official_ta_tok.py
```

Expected: import failure for `qwen35_planx.official_ta_tok`.

- [ ] **Step 3: Implement the exact released architecture adapter**

Reproduce only the inference modules required by the released checkpoint:

```python
class ReleasedTATok(nn.Module):
    IMAGE_SIZE = 384
    TOKENS = 729
    CODEBOOK_SIZE = 65_536
    CODE_DIM = 1536

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        siglip_model_path: Path | str | None = None,
        encoder_factory: Callable[[], nn.Module] | None = None,
        weights_only: bool = True,
    ) -> "ReleasedTATok":
        if not weights_only:
            raise ValueError("released TA-Tok loading requires weights_only=True")
```

Use `torch.serialization.safe_globals([EasyDict])` and
`torch.load(checkpoint_path, weights_only=True, map_location="cpu")`. Validate checkpoint
arguments before constructing the 2.3-GB model. Match the official modules:
SigLIP2 vision encoder, `encode_task_layer`, bottleneck input/output linear
layers, `SimVectorQuantizer`, and three-layer SigLIP2 feature decoder. Load
with `strict=True`, set deterministic VQ evaluation, call `eval()`, and disable
all gradients.

`encode_codes()` must resize RGB `[0,1]` to 384, normalize by 0.5/0.5, select
vision hidden layer `-2`, and return `bottleneck_rep`. `codebook` must be:

```python
F.normalize(
    embedding_proj(embedding.weight).float(),
    dim=-1,
) if l2_normalized else embedding_proj(embedding.weight).float()
```

- [ ] **Step 4: Add safe codebook export**

`export_codebook_safetensors()` writes only:

```text
ta_codebook.safetensors:
  codebook [65536,1536] float32
ta_codebook.json:
  released checkpoint SHA-256
  state SHA-256
  geometry and teacher metadata
```

Use a temporary sibling path and `os.replace()` for both files. A planner
checkpoint may depend on this 400-MB-class artifact, but no model weight is
added to Git.

- [ ] **Step 5: Update preflight, run real-checkpoint smoke, and commit**

Add CLI checks for checkpoint existence, safe load, exact argument values,
local SigLIP2 weights, and free output space. Run unit tests, then one local
real-checkpoint smoke:

```bash
pytest -q tests/test_qwen35_official_ta_tok.py \
  tests/test_qwen35_grounded_config.py
python -m qwen35_planx.cli.preflight released-ta \
  --ta-checkpoint /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/Tar-TA-Tok/ta_tok.pth \
  --siglip-model third_party/siglip2-so400m-patch14-384
```

Expected: tests pass; preflight reports `(729, 65536, 1536)` and a stable
checkpoint hash.

Commit:

```bash
git add qwen35_planx/official_ta_tok.py qwen35_planx/cli/preflight.py \
  tests/test_qwen35_official_ta_tok.py
git add -u qwen35_planx tests
git commit -m "feat(planx): load released 384 TA-Tok safely"
```

---

### Task 3: Define Exact HDF5 Planner Windows and the Hindsight Cache

**Files:**
- Create: `qwen35_planx/hindsight_data.py`
- Create: `qwen35_planx/hindsight_schema.py`
- Create: `tests/test_qwen35_hindsight_schema.py`
- Modify: `qwen35_planx/cli/build_libero_manifests.py`
- Modify: `tests/test_qwen35_planx_libero_data.py`

**Interfaces:**
- Consumes: `ge_act.data.libero_fastwam_hdf5_schema.load_manifest()`,
  `EpisodeRecord`, `PlanGeometry`, and the existing trajectory split.
- Produces: `HindsightWindowRecord`, `HDF5Trajectory`,
  `build_fixed_windows()`, `read_full_trajectory()`,
  `HindsightShardWriter`, `finalize_hindsight_cache()`, and
  `HindsightCache`.

- [ ] **Step 1: Write failing exact-window tests**

Lock GE-Act sampling semantics:

```python
def test_fixed_window_matches_ge_act_keyframes(fake_hdf5_manifest: Path) -> None:
    from qwen35_planx.hindsight_data import build_fixed_windows

    windows = build_fixed_windows(
        fake_hdf5_manifest,
        split_seed=42,
        window_stride=36,
        sample_n_frames=500,
    )
    window = windows[0]
    assert len(window.frame_indices) == 13
    assert len(window.action_indices) == 40
    assert window.current_index == window.frame_indices[3]
    assert window.future_indices == tuple(
        window.frame_indices[4 + index] for index in (0, 3, 5, 8)
    )
    assert window.camera_names == ("main", "wrist")
```

Use the HDF5 loader's `action_chunk=36`, `chunk=9`, `n_previous=4`, and video
temporal stride 4. Memory indices use deterministic uniform selection; cache
and joint training must consume the same explicit indices.

- [ ] **Step 2: Write failing shard round-trip and hash tests**

Create two samples and assert:

```python
with HindsightCache.open(cache_dir) as cache:
    sample = cache[1]
    assert sample.codes.shape == (2, 4, 729)
    assert sample.relevance.shape == (2, 4, 3, 729)
    assert sample.confidence.shape == (2, 4, 3)
    assert sample.flow.shape == (2, 3, 729, 3)
    assert sample.phrase_embeddings.shape == (3, 1152)
    assert torch.allclose(sample.relevance.sum(-1), torch.ones(2, 4, 3))
```

Test partial shard rejection, duplicate sample IDs, wrong split hash, wrong
teacher hash, out-of-range codes, non-finite flow, and camera reordering.

- [ ] **Step 3: Implement HDF5 records and full-trajectory reads**

Define:

```python
@dataclass(frozen=True)
class HindsightWindowRecord:
    sample_id: str
    episode_key: str
    split: str
    caption: str
    current_index: int
    future_indices: tuple[int, int, int, int]
    frame_indices: Sequence[int]
    action_indices: Sequence[int]
    camera_names: tuple[str, str] = ("main", "wrist")


@dataclass(frozen=True)
class HDF5Trajectory:
    rgb: np.ndarray
    actions: np.ndarray
    states: np.ndarray
```

`rgb` is `[2,T,256,256,3] uint8`; actions are `[T,7] float32`; states are
`[T,8] float32`. Validate the HDF5 group with the existing schema before
reading.

- [ ] **Step 4: Implement atomic trajectory shards and finalized memmaps**

Each atomic `.npz` trajectory shard contains its ordered window records and:

```text
codes              uint16 [N,2,4,729]
relevance_q        uint8  [N,2,4,3,729]
relevance_scale    float16[N,2,4,3]
confidence         float16[N,2,4,3]
flow               float16[N,2,3,729,3]
phrase_embeddings  float16[N,3,1152]
```

`finalize_hindsight_cache()` validates every shard and writes immutable
memmaps plus `index.jsonl` and `manifest.json`. Dequantization divides by each
map scale, clamps nonnegative, and renormalizes over 729 positions.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest -q tests/test_qwen35_hindsight_schema.py \
  tests/test_qwen35_planx_libero_data.py \
  tests/test_libero_fastwam_hdf5.py
```

Expected: all pass.

Commit:

```bash
git add qwen35_planx/hindsight_data.py qwen35_planx/hindsight_schema.py \
  qwen35_planx/cli/build_libero_manifests.py \
  tests/test_qwen35_hindsight_schema.py tests/test_qwen35_planx_libero_data.py
git commit -m "feat(planx): define HDF5 hindsight cache"
```

---

### Task 4: Build SigLIP2 Phrase Relevance and DINO/Action Temporal Grounding

**Files:**
- Create: `qwen35_planx/siglip_relevance.py`
- Create: `qwen35_planx/temporal_grounding.py`
- Create: `tests/test_qwen35_siglip_relevance.py`
- Create: `tests/test_qwen35_temporal_grounding.py`

**Interfaces:**
- Consumes: `InstructionFields`, frozen SigLIP2, frozen DINOv3, complete RGB,
  actions, and states.
- Produces: `PhraseRelevance`, `DinoTracks`, `ActionPhases`,
  `SiglipRelevanceTeacher`, `DinoTemporalTeacher`,
  `detect_action_phases()`, and `fuse_hindsight_maps()`.

- [ ] **Step 1: Write failing phrase-embedding and Grad-relevance tests**

Use a deterministic fake SigLIP2:

```python
def test_phrase_teacher_returns_normalized_dense_maps(fake_siglip) -> None:
    from qwen35_planx.siglip_relevance import SiglipRelevanceTeacher

    teacher = SiglipRelevanceTeacher.from_components(
        model=fake_siglip.model,
        processor=fake_siglip.processor,
    )
    output = teacher.encode(
        torch.zeros(2, 3, 384, 384),
        phrases=("pick up", "black bowl", "on the plate"),
    )
    assert output.phrase_embeddings.shape == (3, 1152)
    assert output.maps.shape == (2, 3, 27, 27)
    torch.testing.assert_close(
        output.maps.flatten(-2).sum(-1),
        torch.ones(2, 3),
    )
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())
```

Assert gradients exist only on captured spatial activations, not on teacher
parameters. Test empty phrases return zero confidence and a zero map.

- [ ] **Step 2: Implement frozen SigLIP2 relevance**

`SiglipRelevanceTeacher.encode()` must:

1. preprocess RGB at 384 with the local processor;
2. encode the three phrases to normalized `[3,1152]`;
3. retain gradients on the final 729 spatial activations;
4. compute global image/phrase similarity;
5. obtain positive gradient-times-activation relevance;
6. blend it with captured pooling attention;
7. normalize each 27x27 map and compute a confidence from map peak,
   counterfactual margin, and finite-value checks.

Teacher parameters stay frozen. Use `torch.enable_grad()` only inside the
relevance call and clear captured activations after every phrase.

- [ ] **Step 3: Write failing DINO tracking and action-phase tests**

Test a synthetic moving 2x2 feature block:

```python
def test_cycle_consistent_tracking_recovers_translation() -> None:
    from qwen35_planx.temporal_grounding import track_keyframes

    features = make_translated_features(frames=4, grid=27, dx=1, dy=0)
    tracks = track_keyframes(features, search_radius=4)
    assert tracks.flow.shape == (3, 729, 3)
    assert tracks.flow[:, :, 0].median().item() == 1
    assert tracks.flow[:, :, 1].abs().max().item() == 0
    assert tracks.flow[:, :, 2].min().item() > 0.9
```

Test gripper close, transport, release, missing-gripper fallback, arm-only
motion rejection, and main/wrist independent flow.

- [ ] **Step 4: Implement DINO correspondence and phase priors**

Load `DINOv3ViTModel` frozen, resize its patch grid to 27x27, normalize
features, and search cosine-nearest neighbors within radius four. Keep a match
only when forward/backward correspondence error is at most one grid cell.

`detect_action_phases(actions, states)` uses the seventh action channel for
gripper command. A closure/opening transition must persist for three steps.
Return normalized priors:

```python
@dataclass(frozen=True)
class ActionPhases:
    source: Tensor
    transport: Tensor
    target: Tensor
    confidence: float
```

If no stable transition exists, return broad visual-only priors and confidence
zero.

- [ ] **Step 5: Implement confidence-weighted hindsight fusion**

Use the fixed evidence exponents:

```python
EVIDENCE_EXPONENTS = {
    "source": (0.45, 0.30, 0.15, 0.10),
    "target": (0.45, 0.15, 0.25, 0.15),
    "action": (0.35, 0.25, 0.20, 0.20),
}
```

Combine text, track, correspondence-corrected change, and phase evidence with
a log-domain weighted geometric mean. Clamp each input to `1e-6`, normalize
the final 729 values, and set confidence to zero when fewer than two evidence
sources are valid.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
pytest -q tests/test_qwen35_siglip_relevance.py \
  tests/test_qwen35_temporal_grounding.py
```

Expected: all pass.

Commit:

```bash
git add qwen35_planx/siglip_relevance.py \
  qwen35_planx/temporal_grounding.py \
  tests/test_qwen35_siglip_relevance.py \
  tests/test_qwen35_temporal_grounding.py
git commit -m "feat(planx): derive video-hindsight grounding"
```

---

### Task 5: Orchestrate and Materialize the Hindsight Teacher Cache

**Files:**
- Create: `qwen35_planx/hindsight_builder.py`
- Create: `qwen35_planx/cli/build_hindsight_cache.py`
- Create: `qwen35_planx/scripts/build_hindsight_cache_ola.sh`
- Create: `tests/test_qwen35_hindsight_builder.py`
- Modify: `qwen35_planx/cli/preflight.py`

**Interfaces:**
- Consumes: Tasks 1-4 interfaces and HDF5 window manifest.
- Produces: `HindsightTargetBuilder.build_window()`,
  `build_counterfactual_vocabulary()`, shard build/finalize CLI, and
  teacher/cache audit metrics. Finalization also writes
  `phrase_vocabulary.json` and `phrase_embeddings.safetensors`, a train-split
  lookup table used to construct deterministic counterfactual negatives
  without running SigLIP2 during planner training.

- [ ] **Step 1: Write a failing full-window orchestration test**

Inject fake teachers and assert exact call/data flow:

```python
def test_builder_uses_complete_video_but_stores_only_k4(fake_builder_inputs) -> None:
    from qwen35_planx.hindsight_builder import HindsightTargetBuilder

    builder = HindsightTargetBuilder.from_components(**fake_builder_inputs.components)
    target = builder.build_window(
        fake_builder_inputs.trajectory,
        fake_builder_inputs.window,
    )
    assert fake_builder_inputs.dino.frames_seen == fake_builder_inputs.trajectory.rgb.shape[1]
    assert target.codes.shape == (2, 4, 729)
    assert target.relevance.shape == (2, 4, 3, 729)
    assert target.flow.shape == (2, 3, 729, 3)
    assert target.phrase_embeddings.shape == (3, 1152)
    assert target.teacher_only_fields == ()
```

Also test deterministic output, low-confidence field masking, illegal future
indices, and counterfactual vocabulary derived from train split only.

- [ ] **Step 2: Implement builder orchestration**

For one window:

1. parse and structure the caption;
2. encode all frames with DINO in bounded microbatches;
3. compute SigLIP relevance at interaction anchors and four target keyframes;
4. propagate phrase seeds through DINO tracks;
5. detect action phases from complete actions/states;
6. compute initial/final change and fused maps;
7. encode four TA-Tok keyframes per camera;
8. return only cache schema fields.

The builder must never include future RGB, actions, state, DINO features, or
SigLIP model outputs other than phrase embeddings in the finalized cache.

- [ ] **Step 3: Implement sharded CLI and audit metrics**

The CLI supports:

```text
build --hdf5-manifest "$HDF5_MANIFEST" --window-manifest "$WINDOW_MANIFEST" --ta-checkpoint "$TA_TOK_CHECKPOINT" --siglip-model "$SIGLIP2_MODEL_DIR" --dinov3-model "$DINOV3_MODEL_DIR" --output "$SHARD_ROOT" --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS"
finalize --window-manifest "$WINDOW_MANIFEST" --shard-root "$SHARD_ROOT" --output "$OUTPUT_DIR"
audit --cache "$OUTPUT_DIR" --samples 128 --output "$OUTPUT_DIR/metrics.json"
```

Audit reports per camera/phrase:

- valid-confidence ratio;
- effective support;
- map entropy;
- temporal cycle confidence;
- TA code usage;
- counterfactual margin;
- discarded/non-finite trajectory IDs.

- [ ] **Step 4: Add OLA launcher and preflight**

The launcher requires explicit:

```bash
HDF5_MANIFEST
WINDOW_MANIFEST
TA_TOK_CHECKPOINT
SIGLIP2_MODEL_DIR
DINOV3_MODEL_DIR
OUTPUT_DIR
NUM_GPUS
```

It uses one trajectory shard per worker assignment, never writes into the
HDF5 dataset, and runs `preflight hindsight-cache` before `torchrun`.

- [ ] **Step 5: Run tests, a one-trajectory smoke, and commit**

Run:

```bash
pytest -q tests/test_qwen35_hindsight_builder.py \
  tests/test_qwen35_hindsight_schema.py \
  tests/test_qwen35_siglip_relevance.py \
  tests/test_qwen35_temporal_grounding.py
bash -n qwen35_planx/scripts/build_hindsight_cache_ola.sh
```

Then run `build`, `finalize`, and `audit` on one HDF5 episode with fake or local
teacher weights. Expected: one finalized cache opens read-only and passes hash
validation.

Commit:

```bash
git add qwen35_planx/hindsight_builder.py \
  qwen35_planx/cli/build_hindsight_cache.py \
  qwen35_planx/cli/preflight.py \
  qwen35_planx/scripts/build_hindsight_cache_ola.sh \
  tests/test_qwen35_hindsight_builder.py
git commit -m "feat(planx): build grounded hindsight cache"
```

---

### Task 6: Expand Qwen Locally and Build the Exact Causal Sequence

**Files:**
- Create: `qwen35_planx/vocabulary.py`
- Create: `qwen35_planx/sequence.py`
- Create: `qwen35_planx/planner_dataset.py`
- Create: `tests/test_qwen35_vocabulary.py`
- Create: `tests/test_qwen35_sequence.py`
- Create: `tests/test_qwen35_planner_dataset.py`

**Interfaces:**
- Consumes: `GroundedPlannerMetadata`, `InstructionFields`,
  `HindsightCache`, HDF5 current images, and a Qwen3.5 processor/model.
- Produces: `VisualVocabularyLayout`, `install_visual_vocabulary()`,
  `CausalPlanSequence`, `build_plan_sequence()`,
  `HindsightPlannerDataset`, `CachedPlannerTargets`,
  `GroundedPlannerBatch`, and `GroundedPlannerCollator`.

- [ ] **Step 1: Write failing vocabulary tests**

```python
def test_visual_rows_use_mean_qwen_initialization(fake_qwen, fake_tokenizer) -> None:
    from qwen35_planx.vocabulary import install_visual_vocabulary

    old = fake_qwen.get_input_embeddings().weight.detach().clone()
    layout = install_visual_vocabulary(fake_tokenizer, fake_qwen)
    new_rows = fake_qwen.get_input_embeddings().weight[
        layout.visual_start_id:layout.visual_end_id
    ]
    torch.testing.assert_close(
        new_rows,
        old.mean(0, keepdim=True).expand_as(new_rows),
    )
    assert new_rows.shape == (65_536, 2048)
    assert not hasattr(layout, "codebook_projection")
```

Test contiguous IDs, tied input/output rows, no base-row changes, structural
token uniqueness, and experiment-local save paths.

- [ ] **Step 2: Implement visual vocabulary installation**

Add strings `<|ta_00000|>` through `<|ta_65535|>` and fixed structure/camera
tokens, including the three single-token `<SRC_QUERY>`, `<TGT_QUERY>`, and
`<ACT_QUERY>` role queries. Resize once, initialize all new rows to the
original row mean, preserve base rows byte-for-byte, and return an immutable
layout with hashes. Reject calling the installer twice or saving into the base
model directory.

- [ ] **Step 3: Write failing causal-index tests**

```python
def test_sequence_exposes_pre_and_post_positions(layout) -> None:
    from qwen35_planx.sequence import build_plan_sequence

    codes = torch.arange(4 * 729).remainder(65_536).reshape(4, 729)
    sequence = build_plan_sequence(
        camera="main",
        prompt="<ACT>pick</ACT><SRC>bowl</SRC><TGT>plate</TGT>",
        codes=codes,
        layout=layout,
    )
    assert sequence.code_targets.shape == (2916,)
    assert sequence.pre_positions.shape == (2916,)
    assert sequence.post_positions.shape == (2916,)
    assert sequence.field_positions.shape == (3,)
    assert sequence.field_mask.tolist() == [True, True, True]
    assert torch.equal(sequence.post_positions, sequence.code_positions)
    for frame_index in range(4):
        start = frame_index * 729
        end = start + 729
        assert sequence.pre_positions[start] == sequence.frame_start_positions[frame_index]
        assert torch.equal(
            sequence.pre_positions[start + 1:end],
            sequence.code_positions[start:end - 1],
        )
```

At every frame boundary, causal correctness requires the first code's
`h_pre` to be the matching `<FRAME_n>` state; it must never reuse the final
code state from the preceding frame. Subsequent codes within that frame use
the immediately preceding code state. Also assert raster order is stable,
prompt/structure positions are excluded from code loss, and main/wrist
examples never share a sequence.

- [ ] **Step 4: Implement target contract, sequence, and collator**

`GroundedPlannerCollator` flattens each batch into `B*2` examples. It sends one
camera image and one structured prompt to Qwen per example, appends the exact
plan sequence, pads with an explicit attention mask, and returns a
`GroundedPlannerBatch` with:

```python
@dataclass(frozen=True)
class CachedPlannerTargets:
    codes: LongTensor                    # [B,2,4,729]
    relevance: FloatTensor               # [B,2,4,3,729]
    relevance_confidence: FloatTensor    # [B,2,4,3]
    flow: FloatTensor                    # [B,2,3,729,3]
    phrase_embeddings: FloatTensor       # [B,3,1152]


@dataclass(frozen=True)
class GroundedPlannerBatch:
    qwen_inputs: Mapping[str, Tensor]
    code_targets: LongTensor             # [B*2,2916]
    pre_positions: LongTensor            # [B*2,2916]
    post_positions: LongTensor           # [B*2,2916]
    field_positions: LongTensor           # [B*2,3], SRC/TGT/ACT query states
    field_mask: BoolTensor                # [B*2,3]
    relevance_targets: FloatTensor       # [B*2,4,3,729]
    relevance_confidence: FloatTensor    # [B*2,4,3]
    flow_targets: FloatTensor            # [B*2,3,729,3]
    phrase_embeddings: FloatTensor       # [B*2,3,1152]
    counterfactual_embeddings: FloatTensor  # [B*2,3,N_neg,1152]
    counterfactual_mask: BoolTensor       # [B*2,3,N_neg]
```

The dataset reads current RGB from the exact HDF5 window record and all target
arrays from `HindsightCache`. The collator parses each instruction, selects
same-suite one-field replacements from `phrase_vocabulary.json`, and looks up
their frozen embeddings from `phrase_embeddings.safetensors`. It exposes both
`__call__(samples)` and
`build_teacher_forced(current_images, instructions, targets)`; both routes
must construct byte-identical sequences and negative embeddings for identical
inputs. `field_positions` always addresses the single-token
`<SRC_QUERY>`, `<TGT_QUERY>`, and `<ACT_QUERY>` markers after the current
image, structured fields, and complete original instruction have been
causally consumed, returned in canonical source/target/action order.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest -q tests/test_qwen35_vocabulary.py \
  tests/test_qwen35_sequence.py \
  tests/test_qwen35_planner_dataset.py
```

Expected: all pass.

Commit:

```bash
git add qwen35_planx/vocabulary.py qwen35_planx/sequence.py \
  qwen35_planx/planner_dataset.py tests/test_qwen35_vocabulary.py \
  tests/test_qwen35_sequence.py tests/test_qwen35_planner_dataset.py
git commit -m "feat(planx): build grounded causal planner batches"
```

---

### Task 7: Implement Chunked Code Prediction, Grounding Heads, and Losses

**Files:**
- Create: `qwen35_planx/losses.py`
- Create: `qwen35_planx/planner.py`
- Create: `tests/test_qwen35_grounded_losses.py`
- Create: `tests/test_qwen35_grounded_planner.py`

**Interfaces:**
- Consumes: Task 6 batch, Qwen3.5 backbone, frozen `[65536,1536]` codebook.
- Produces: `GroundedPlannerOutput`, `GroundedQwen35Planner.forward()`,
  `chunked_visual_cross_entropy()`, `dense_feature_loss()`,
  `grounding_loss()`, `counterfactual_loss()`, and `temporal_loss()`.

- [ ] **Step 1: Write failing chunked-CE equivalence and memory-contract tests**

```python
def test_chunked_visual_ce_matches_dense_reference() -> None:
    from qwen35_planx.losses import chunked_visual_cross_entropy

    hidden = torch.randn(2, 17, 8, requires_grad=True)
    weight = torch.randn(31, 8, requires_grad=True)
    targets = torch.randint(0, 31, (2, 17))
    actual = chunked_visual_cross_entropy(hidden, weight, targets, chunk_size=5)
    expected = F.cross_entropy(
        hidden.reshape(-1, 8) @ weight.T,
        targets.reshape(-1),
    )
    torch.testing.assert_close(actual, expected)
```

Add a fake output-head object that raises if the full Qwen LM head is called.
The planner must gather code positions and compute only visual-vocabulary
logits in chunks of at most 64 positions.

- [ ] **Step 2: Write failing planner shape and state-alignment tests**

```python
def test_planner_uses_pre_for_code_and_post_for_semantics(fake_backbone, batch) -> None:
    from qwen35_planx.planner import GroundedQwen35Planner

    planner = GroundedQwen35Planner.from_components(
        backbone=fake_backbone,
        visual_embedding_weight=fake_backbone.embedding.weight,
        codebook=torch.randn(65_536, 1536),
        hidden_dim=2048,
        text_dim=1152,
    )
    output = planner(batch)
    assert output.codes.shape == (batch.size, 4, 729)
    assert output.code_embeddings.shape == (batch.size, 4, 729, 1536)
    assert output.post_hidden.shape == (batch.size, 4, 729, 2048)
    assert output.predicted_phrase_embeddings.shape == (batch.size, 3, 1152)
    assert output.visual_regression.shape == (batch.size, 2916, 1536)
    assert output.semantic_features.shape == (batch.size, 4, 729, 1152)
    assert output.relevance_logits.shape == (batch.size, 4, 3, 729)
    assert output.fusion_gate.shape == (batch.size, 4, 729, 1)
    assert output.times.shape == (4,)
    assert output.debug_pre_positions == batch.pre_positions
    assert output.debug_post_positions == batch.post_positions
```

Test the final code token, camera flattening, confidence-zero masks, finite
outputs, and frozen codebook gradients.

- [ ] **Step 3: Implement planner heads without full-vocabulary logits**

Call the Qwen language backbone with `output_hidden_states=False` and
`return_dict=True`, then gather:

```python
h_pre = gather_positions(last_hidden_state, batch.pre_positions)
h_post = gather_positions(last_hidden_state, batch.post_positions)
h_fields = gather_positions(last_hidden_state, batch.field_positions)
```

Define:

```python
self.visual_regression = nn.Linear(2048, 1536)
self.semantic_projection = nn.Linear(2048, 1152)
self.phrase_projection = nn.Linear(2048, 1152)
self.grounding_query = nn.Linear(2048, 1152, bias=False)
self.fusion_gate = nn.Sequential(
    nn.Linear(2048 + 3, 256),
    nn.SiLU(),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)
```

Normalize `phrase_projection(h_fields)` and supervise it against the cached
SigLIP2 phrase embeddings using `field_mask`. Grounding logits are dot
products between normalized `grounding_query(h_post)` and the predicted
phrase embeddings, reshaped to
`[B,4,3,729]`. `GroundedPlannerOutput` stores `codes`, frozen-codebook
`code_embeddings`, `post_hidden`, `semantic_features`, normalized
`predicted_phrase_embeddings`, `relevance`, `fusion_gate`, the four normalized
target times, and the scalar losses. Its `unflatten_cameras(batch_size)`
method converts every per-camera tensor by splitting the leading `B*2` axis
into leading axes `[B,2]`, while leaving scalar losses unchanged.
Use visual token embedding rows only for chunked CE.

- [ ] **Step 4: Implement the five objectives exactly**

```python
total = (
    code_loss
    + 0.5 * dense_feature_loss
    + 0.5 * grounding_loss
    + 0.2 * counterfactual_loss
    + 0.1 * temporal_loss
)
```

`dense_feature_loss` averages normalized codebook regression,
confidence-weighted semantic cosine, and field-mask-weighted cosine regression
from predicted phrase anchors to cached SigLIP2 phrase embeddings.
`grounding_loss` uses JS divergence and a
`0.01` hinge penalty when `exp(entropy)/729` is outside `[0.01,0.40]`.
`counterfactual_loss` uses the three cached phrase embeddings as positives and
the collator-provided deterministic same-suite replacement embeddings as
negatives, applying `counterfactual_mask` before the margin reduction.
`temporal_loss` warps maps with cached `(dx,dy,confidence)` and ignores invalid
destinations.

- [ ] **Step 5: Run gradient and regression tests, then commit**

Run:

```bash
pytest -q tests/test_qwen35_grounded_losses.py \
  tests/test_qwen35_grounded_planner.py \
  tests/test_qwen3vl2b_legacy_unchanged.py
```

Expected: all pass; gradients reach Qwen hidden states, visual rows, and new
heads but not the codebook.

Commit:

```bash
git add qwen35_planx/losses.py qwen35_planx/planner.py \
  tests/test_qwen35_grounded_losses.py \
  tests/test_qwen35_grounded_planner.py
git commit -m "feat(planx): train grounded Qwen3.5 future plans"
```

---

### Task 8: Add Distributed Planner Training and Self-Contained Checkpoints

**Files:**
- Create: `qwen35_planx/cli/train_semantic_planner.py`
- Create: `qwen35_planx/scripts/train_semantic_planner_ola.sh`
- Modify: `qwen35_planx/cli/preflight.py`
- Extend: `tests/test_qwen35_grounded_planner.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 6, and 7 artifacts.
- Produces: distributed trainer, optimizer groups, resume logic, and a
  self-contained planner export.

- [ ] **Step 1: Write failing optimizer-group and checkpoint tests**

Assert exact groups:

```python
assert group_lrs == {
    "qwen_language": 1e-5,
    "qwen_vision": 5e-6,
    "visual_vocab_and_prediction_head": 1e-4,
    "semantic_phrase_grounding_fusion_heads": 1e-4,
}
```

The second head group contains the semantic projection, phrase-anchor
projection, grounding query, and fusion gate from Task 7. The downstream
GE-Act-width visual/hidden adapters do not exist until Task 9 and are trained
only in the joint stage.

Checkpoint test must require:

```text
planner.safetensors
ta_codebook.safetensors
processor/
tokenizer/
planner_meta.json
optimizer.pt
scheduler.pt
trainer_state.json
```

Validate save/resume restores optimizer step, scheduler step, visual-token
range, cache hash, and codebook hash.

- [ ] **Step 2: Implement optimizer groups and Accelerate loop**

Use effective global batch:

```python
global_batch = per_device_batch * accelerator.num_processes * grad_accum
if global_batch != 256:
    raise ValueError(f"effective global batch must be 256, got {global_batch}")
```

Train 30,000 optimizer steps with bf16, TF32, gradient clipping 1.0, warmup
1,000, cosine decay, and selective Qwen activation checkpointing. Log every 20
steps and validate/save every 5,000.

- [ ] **Step 3: Implement strict checkpoint export and resume**

Save the expanded experiment-local tokenizer/model, new heads, released
codebook export, metadata hashes, optimizer, scheduler, scaler, RNG state, and
current step atomically. Never write into the base Qwen or released TA-Tok
directories.

- [ ] **Step 4: Add launcher and preflight**

Launcher defaults:

```bash
NUM_GPUS=8
GLOBAL_BATCH=256
MAX_STEPS=30000
WARMUP_STEPS=1000
SAVE_EVERY=5000
QWEN_LANGUAGE_LR=1e-5
QWEN_VISION_LR=5e-6
HEAD_LR=1e-4
ADAPTER_LR=1e-4
```

Preflight loads one real HDF5/cache sample, validates all hashes, performs one
CPU shape-only batch, estimates per-GPU batch candidates, and refuses an
incomplete checkpoint resume.

- [ ] **Step 5: Run unit and one-step distributed smoke, then commit**

Run:

```bash
pytest -q tests/test_qwen35_grounded_planner.py \
  tests/test_qwen35_planner_dataset.py \
  tests/test_qwen35_official_ta_tok.py
bash -n qwen35_planx/scripts/train_semantic_planner_ola.sh
torchrun --standalone --nproc_per_node=1 \
  -m qwen35_planx.cli.train_semantic_planner \
  --config tests/fixtures/qwen35_grounded_tiny.json \
  --max-steps 1
```

Expected: one optimizer step, one validation pass, and a reloadable checkpoint.

Commit:

```bash
git add qwen35_planx/cli/train_semantic_planner.py \
  qwen35_planx/scripts/train_semantic_planner_ola.sh \
  qwen35_planx/cli/preflight.py tests/test_qwen35_grounded_planner.py \
  tests/fixtures/qwen35_grounded_tiny.json
git commit -m "feat(planx): train and export grounded planner"
```

---

### Task 9: Add Constrained Decoding and the Planner Provider

**Files:**
- Create: `qwen35_planx/decoding.py`
- Create: `qwen35_planx/provider.py`
- Create: `tests/test_qwen35_decoding.py`
- Create: `tests/test_qwen35_provider.py`

**Interfaces:**
- Consumes: trained `GroundedQwen35Planner`, vocabulary layout, current
  images/instructions, and optional cached teacher targets.
- Produces: `GeneratedGroundedPlan`,
  `generate_grounded_plan()`, `unflatten_generated_plan()`, and
  `Qwen35GroundedPlanProvider` with frozen inference, trainable
  teacher-forcing, and shared fusion modes.

- [ ] **Step 1: Write failing constrained-generation tests**

```python
def test_generation_emits_exact_k4_grid_and_records_post_states(fake_planner) -> None:
    from qwen35_planx.decoding import generate_grounded_plan

    plan = generate_grounded_plan(
        fake_planner,
        current_images=torch.zeros(2, 3, 384, 384),
        instructions=("pick", "place"),
        camera_names=("main", "wrist"),
    )
    assert plan.codes.shape == (2, 4, 729)
    assert plan.post_hidden.shape == (2, 4, 729, 2048)
    assert plan.predicted_phrase_embeddings.shape == (2, 3, 1152)
    assert plan.semantic_features.shape == (2, 4, 729, 1152)
    assert plan.relevance.shape == (2, 4, 3, 729)
    assert fake_planner.full_lm_head_calls == 0
```

Test illegal IDs, malformed structure, final-code hidden capture, EOS before
the required length, deterministic greedy mode, independent camera KV caches,
and successful generation after blocking all SigLIP2 imports.

- [ ] **Step 2: Implement KV-cache visual-only decoding**

At each code position:

1. run the Qwen backbone for the last token with `past_key_values`;
2. compute logits only against the 65,536 visual embedding rows;
3. select greedy/top-k code according to config;
4. append that visual token;
5. record the post-code state when it is consumed on the next step;
6. force deterministic frame/plan delimiters without scoring the full vocab.

Process main and wrist as two batch rows. Feed the final visual token once more
to obtain its `h_post` before forcing `</FRAME_4>`. The prompt pass also
gathers the three structured field states and runs `phrase_projection`; these
predicted anchors, not SigLIP2, drive all generated relevance maps.

- [ ] **Step 3: Implement frozen and teacher-forced provider modes**

Define:

```python
@dataclass(frozen=True)
class GeneratedGroundedPlan:
    codes: Tensor
    code_embeddings: Tensor
    post_hidden: Tensor
    predicted_phrase_embeddings: Tensor
    semantic_features: Tensor
    relevance: Tensor
    fusion_gate: Tensor
    times: Tensor


class Qwen35GroundedPlanProvider(nn.Module):
    def generate(self, current_images: Tensor, instructions: Sequence[str]) -> GeneratedGroundedPlan:
        batch_size = current_images.shape[0]
        flat_images = current_images.reshape(batch_size * 2, 3, *current_images.shape[-2:])
        camera_names = tuple(
            camera
            for _ in range(batch_size)
            for camera in ("main", "wrist")
        )
        flat_instructions = tuple(
            instruction
            for instruction in instructions
            for _ in range(2)
        )
        flat_plan = generate_grounded_plan(
            self.planner,
            current_images=flat_images,
            instructions=flat_instructions,
            camera_names=camera_names,
            layout=self.layout,
        )
        return unflatten_generated_plan(flat_plan, batch_size)

    def teacher_force(
        self,
        current_images: Tensor,
        instructions: Sequence[str],
        targets: CachedPlannerTargets,
    ) -> GroundedPlannerOutput:
        batch = self.collator.build_teacher_forced(
            current_images=current_images,
            instructions=instructions,
            targets=targets,
        )
        return self.planner(batch).unflatten_cameras(current_images.shape[0])

    def fuse(
        self,
        plan: GeneratedGroundedPlan | GroundedPlannerOutput,
        *,
        qwen_gradient_scale: float = 1.0,
    ) -> Tensor:
        code_embeddings = scale_gradient(plan.code_embeddings, qwen_gradient_scale)
        post_hidden = scale_gradient(plan.post_hidden, qwen_gradient_scale)
        fusion_gate = scale_gradient(plan.fusion_gate, qwen_gradient_scale)
        position_features = self.position_encoder(
            grid_size=(27, 27),
            times=plan.times,
            batch_shape=plan.codes.shape[:3],
        )
        return self.output_norm(
            self.visual_adapter(code_embeddings)
            + fusion_gate * self.hidden_adapter(post_hidden)
            + position_features
        )
```

`generate()` accepts `[B,2,3,H,W]` and returns explicit shapes:
`codes [B,2,4,729]`, `code_embeddings [B,2,4,729,1536]`,
`post_hidden [B,2,4,729,2048]`,
`predicted_phrase_embeddings [B,2,3,1152]`,
`semantic_features [B,2,4,729,1152]`,
`relevance [B,2,4,3,729]`, and `fusion_gate [B,2,4,729,1]`.
`teacher_force()` is differentiable and used only in joint training with the
complete cached `CachedPlannerTargets`. `fuse()` is the only conversion to the
GE-Act condition width: `visual_adapter: 1536 -> D_condition`,
`hidden_adapter: 2048 -> D_condition`, learned spatial/time positions, a
per-token gate, and final normalization. It must
produce the same shape and numerical result for equivalent generated and
teacher-forced plans. `scale_gradient(x, scale)` is exactly
`x.detach() + scale * (x - x.detach())`; it validates `0 <= scale <= 1`,
leaves the forward value unchanged, and scales only gradients crossing the
Qwen-to-provider boundary.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest -q tests/test_qwen35_decoding.py tests/test_qwen35_provider.py
```

Expected: all pass.

Commit:

```bash
git add qwen35_planx/decoding.py qwen35_planx/provider.py \
  tests/test_qwen35_decoding.py tests/test_qwen35_provider.py
git commit -m "feat(planx): decode grounded future plans"
```

---

### Task 10: Compress 27x27 Plans Without Losing Task Regions

**Files:**
- Create: `qwen35_planx/compression.py`
- Create: `tests/test_qwen35_compression.py`

**Interfaces:**
- Consumes: `[B,2,4,729,D]` fused tokens and `[B,2,4,3,729]` relevance.
- Produces: `CompressedSemanticPlan` and
  `compress_grounded_plan()` returning 64 coverage plus up to 32 exact tokens
  per frame, with one scalar relevance value and one source-grid coordinate
  for every emitted token.

- [ ] **Step 1: Write failing pooling and top-token tests**

```python
def test_compression_preserves_coverage_and_small_peak() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.arange(729.0).reshape(1, 1, 1, 729, 1)
    relevance = torch.zeros(1, 1, 1, 3, 729)
    relevance.select(-1, 17).fill_(1.0)
    plan = compress_grounded_plan(features, relevance, top_k=32)
    assert plan.tokens.shape == (1, 1, 1, 96, 1)
    assert plan.positions.shape == (1, 1, 1, 96, 2)
    assert plan.mask.shape == (1, 1, 1, 96)
    assert plan.relevance.shape == (1, 1, 1, 96)
    assert plan.source_indices.select(-1, 64).item() == 17
    assert plan.tokens[0, 0, 0, 64, 0].item() == 17
```

Test fractional 27-to-8 area overlap, duplicate top selections across three
roles, fewer than 32 valid peaks, coordinate bounds, deterministic ties, and
gradient flow through coverage and selected features.

- [ ] **Step 2: Implement relevance-weighted 8x8 area pooling**

Precompute a fixed overlap matrix `P [64,729]` whose rows sum to one. Weight
each source patch by:

```python
weight = overlap * (1.0 + relevance.max(dim=-2).values)
weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-6)
```

Pool features, original normalized coordinates, and maximum-role scalar
relevance with the same weights. Preserve the entire scene even when
relevance is zero.

- [ ] **Step 3: Implement de-duplicated exact top-32**

Rank positions by the maximum of source/target/action relevance; break ties by
raster index. Gather each position once, retain original coordinates, pad
unused slots with mask false, retain the exact maximum-role relevance for each
selected key, and concatenate after the 64 coverage tokens.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest -q tests/test_qwen35_compression.py
```

Expected: all pass.

Commit:

```bash
git add qwen35_planx/compression.py tests/test_qwen35_compression.py
git commit -m "feat(planx): compress grounded spatial plans"
```

---

### Task 11: Add Relevance Bias to the Existing LTX Semantic Attention

**Files:**
- Modify: `ge_act/models/ltx_models/semantic_conditioning.py:66`
- Modify: `ge_act/models/ltx_models/transformer_ltx_multiview.py:133`
- Create: `tests/test_ge_act_qwen35_grounded.py`
- Modify: `tests/test_ge_act_ltx_semantic_guidance.py`

**Interfaces:**
- Consumes: `CompressedSemanticPlan` tokens, positions, mask, relevance, and
  existing semantic plan times.
- Produces: adapted semantic hidden states, explicit positions, key mask, and
  bounded per-key logit bias used in every selected LTX block.

- [ ] **Step 1: Write failing zero-gate and bias tests**

```python
def test_zero_bias_gate_matches_existing_semantic_attention() -> None:
    baseline = make_semantic_attention()
    grounded = copy.deepcopy(baseline)
    grounded.set_grounding_bias_enabled(True)
    grounded.raw_semantic_bias_gate.data.zero_()
    actual = grounded(hidden_states=Q, encoder_hidden_states=K, relevance=R)
    expected = baseline(hidden_states=Q, encoder_hidden_states=K)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bias_is_bounded_and_prefers_relevant_key() -> None:
    module = make_grounded_semantic_attention()
    module.raw_semantic_bias_gate.data.fill_(10)
    weights = module.debug_attention_weights(relevance=torch.tensor([[1.0, 0.01]]))
    assert module.semantic_bias_gate.item() <= 2.0
    assert weights.select(-1, 0).mean() > weights.select(-1, 1).mean()
```

Test `[B,V,K,96,D]` shape, masked padding, same-camera isolation, and old
`[B,V,K,P,D]` call compatibility when relevance is absent.

- [ ] **Step 2: Extend the semantic adapter contract**

Add optional arguments:

```python
def forward(
    semantic_tokens: Tensor,
    semantic_plan_times: Tensor,
    latent_height: int,
    latent_width: int,
    latent_num_frames: int = 6,
    *,
    semantic_positions_xy: Tensor | None = None,
    semantic_token_mask: Tensor | None = None,
    semantic_relevance: Tensor | None = None,
) -> SemanticContext:
```

`SemanticContext` contains hidden states, `(t,y,x)` positions, boolean key
mask, and relevance. Preserve the old square-grid path exactly when explicit
positions are not supplied.

- [ ] **Step 3: Add bounded relevance bias to attention**

Parameterize:

```python
self.raw_semantic_bias_gate = nn.Parameter(torch.zeros(()))
bias_gate = 2.0 * torch.tanh(self.raw_semantic_bias_gate)
log_bias = bias_gate * torch.log(relevance.clamp_min(1e-6))
```

Combine `log_bias` with the padding mask before
`scaled_dot_product_attention`. Keep the existing zero-initialized semantic
residual modulation and same-camera batch layout.

- [ ] **Step 4: Thread explicit plan fields through LTX forward**

Add optional transformer inputs:

```text
semantic_plan_positions
semantic_plan_mask
semantic_plan_relevance
```

Forward them to every configured semantic block and include them in gradient
checkpoint closures. Old GE-Act configs must produce bitwise-equivalent
behavior.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest -q tests/test_ge_act_qwen35_grounded.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_semantic_pipeline.py \
  tests/test_ge_act_semantic_training_contract.py
```

Expected: all pass.

Commit:

```bash
git add ge_act/models/ltx_models/semantic_conditioning.py \
  ge_act/models/ltx_models/transformer_ltx_multiview.py \
  tests/test_ge_act_qwen35_grounded.py \
  tests/test_ge_act_ltx_semantic_guidance.py
git commit -m "feat(ge-act): bias LTX attention with grounded plans"
```

---

### Task 12: Integrate Teacher-Forced Qwen and Full Video/Action Joint Training

**Files:**
- Create: `ge_act/data/libero_hindsight_hdf5_dataset.py`
- Modify: `ge_act/models/ltx_models/vlm_semantic_planner.py`
- Modify: `ge_act/runner/ge_trainer.py:179`
- Create: `ge_act/configs/ltx_model/libero/action_model_libero_qwen35_grounded_hdf5.yaml`
- Create: `ge_act/scripts/train_ltx_qwen35_grounded.sh`
- Create: `ge_act/scripts/sbatch_train_ltx_qwen35_grounded_hpc3.sh`
- Extend: `tests/test_ge_act_qwen35_grounded.py`
- Modify: `tests/test_ge_act_vlm_semantic_planner.py`

**Interfaces:**
- Consumes: finalized hindsight cache, Task 9 provider, Task 10 compressor,
  GE-Act HDF5 base dataset, LTX video expert, and action expert.
- Produces: cache-aligned GE dataset, trainable Qwen provider mode, five
  optimizer groups, joint auxiliary loss, and 30k launchers.

- [ ] **Step 1: Write failing cache-aligned dataset tests**

```python
def test_joint_dataset_uses_exact_cached_window(fake_hdf5, fake_cache) -> None:
    from ge_act.data.libero_hindsight_hdf5_dataset import (
        LiberoHindsightHDF5Dataset,
    )

    dataset = LiberoHindsightHDF5Dataset(
        manifest_path=fake_hdf5.manifest,
        hindsight_cache=fake_cache.path,
        stat_file=fake_hdf5.stats,
        train_dataset=True,
    )
    sample = dataset[0]
    assert sample["episode_key"] == fake_cache.index[0]["episode_key"]
    assert sample["current_index"] == fake_cache.index[0]["current_index"]
    assert sample["target_codes"].shape == (2, 4, 729)
    assert sample["video"].shape[1:3] == (2, 13)
```

The wrapper calls the validated base dataset's `read_by_indexes()` and adds
cache targets without changing normalized video/action/state semantics.

- [ ] **Step 2: Write failing trainability and optimizer-group tests**

Assert exact groups:

```python
assert group_lrs == {
    "ltx_video": 2e-5,
    "action_expert": 1e-4,
    "semantic_adapter": 5e-5,
    "qwen_top8": 1e-6,
    "qwen_vision": 5e-7,
}
```

Assert TA-Tok/SigLIP2/DINO are absent, lower Qwen layers are frozen, top eight
layers and Qwen vision are trainable, and the planner auxiliary loss reaches
Qwen with coefficient 0.25.

Add a scalar gradient-routing regression: with `x=2`, GE loss `3*y`, and
planner loss `5*x`, where `y=scale_gradient(x, 0.1)`, the combined gradient on
`x` must equal `0.3 + 5.0`; the forward value of `y` must remain exactly `2`.

- [ ] **Step 3: Implement cache-aligned dataset and provider selection**

Add `semantic_plan.source: qwen35_grounded`. Metadata validation must happen
before Qwen allocation. In training:

```python
planner_output = provider.teacher_force(
    current_images,
    captions,
    targets=batch["planner_targets"],
)
compressed = compress_grounded_plan(
    provider.fuse(planner_output, qwen_gradient_scale=0.1),
    scale_gradient(planner_output.relevance, 0.1),
)
```

In validation/inference, call `provider.generate()` with no cache targets.

- [ ] **Step 4: Implement joint gradient routing**

Use teacher forcing in training so one packed Qwen forward provides all 2,916
states. Add:

```python
joint_loss = (
    loss_video
    + action_loss_scale * loss_action
    + 0.25 * planner_output.loss
)
```

Apply the explicit `scale_gradient()` identity only on the fused features and
relevance sent into GE-Act. This multiplies GE-Act-originating gradients
entering Qwen outputs by 0.1 while leaving forward values, provider-adapter
gradients, and planner auxiliary gradients unchanged. Do not attach a hook to
the shared `planner_output`, because that would also scale the auxiliary loss.
Freeze all but top eight language layers; keep Qwen vision at its separate LR.

- [ ] **Step 5: Split LTX, action, semantic, and Qwen optimizer groups**

Extend `build_optimizer_parameter_groups()` to accept the provider and exact
LRs. Classify action parameters by the existing `action_` naming contract,
semantic LTX parameters by `_is_semantic_parameter()`, and Qwen parameters by
explicit module ownership rather than name substring.

Prepare diffusion model, provider, optimizer, dataloader, and scheduler
together with Accelerate/DeepSpeed. Save and resume both model states.

- [ ] **Step 6: Add exact joint config and launchers**

The YAML must set:

```yaml
return_action: true
return_video: true
train_mode: all
train_steps: 30000
steps_to_save: 5000
mixed_precision: bf16
allow_tf32: true
lr: 2.0e-5
action_lr: 1.0e-4
semantic_lr: 5.0e-5
qwen_top_lr: 1.0e-6
qwen_vision_lr: 5.0e-7
planner_aux_weight: 0.25
qwen_ge_gradient_scale: 0.1
semantic_plan:
  enabled: true
  source: qwen35_grounded
  num_keyframes: 4
  tokens_per_frame: 96
  dropout: 0.15
```

Use HDF5 input, action expert enabled, all 28 semantic blocks, gradient
checkpointing, eight GPUs, and an effective global batch selected by preflight
without changing loss semantics.

- [ ] **Step 7: Run tests and one joint step, then commit**

Run:

```bash
pytest -q tests/test_ge_act_qwen35_grounded.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_libero_fastwam_hdf5.py \
  tests/test_ge_act_semantic_training_contract.py
bash -n ge_act/scripts/train_ltx_qwen35_grounded.sh
bash -n ge_act/scripts/sbatch_train_ltx_qwen35_grounded_hpc3.sh
```

Run a one-GPU, one-step tiny joint smoke and assert nonzero gradients in LTX,
action expert, semantic adapter, Qwen top layer, and Qwen vision; assert zero
gradients in lower Qwen layers and codebook.

Commit:

```bash
git add ge_act/data/libero_hindsight_hdf5_dataset.py \
  ge_act/models/ltx_models/vlm_semantic_planner.py \
  ge_act/runner/ge_trainer.py \
  ge_act/configs/ltx_model/libero/action_model_libero_qwen35_grounded_hdf5.yaml \
  ge_act/scripts/train_ltx_qwen35_grounded.sh \
  ge_act/scripts/sbatch_train_ltx_qwen35_grounded_hpc3.sh \
  tests/test_ge_act_qwen35_grounded.py \
  tests/test_ge_act_vlm_semantic_planner.py
git commit -m "feat(ge-act): jointly train grounded Qwen plans"
```

---

### Task 13: Add Offline Evaluation, Counterfactual Visualization, and Ablations

**Files:**
- Create: `qwen35_planx/evaluation.py`
- Create: `qwen35_planx/cli/evaluate_semantic_planner.py`
- Create: `qwen35_planx/cli/visualize_semantic_planner.py`
- Create: `tests/test_qwen35_grounded_evaluation.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: planner checkpoint, hindsight validation cache, HDF5 RGB, and
  generated plans.
- Produces: `PlannerMetrics`, JSON/CSV metrics, per-camera visualizations, and
  four fixed GE-Act ablation descriptors.

- [ ] **Step 1: Write failing metric tests**

```python
def test_metrics_separate_correct_and_counterfactual_text() -> None:
    from qwen35_planx.evaluation import evaluate_grounded_batch

    metrics = evaluate_grounded_batch(make_perfect_grounded_batch())
    assert metrics.correct_retrieval_accuracy == 1.0
    assert metrics.counterfactual_margin > 0
    assert metrics.temporal_consistency == 1.0
    assert metrics.main.code_top5 == 1.0
    assert metrics.wrist.code_top5 == 1.0
```

Test JS divergence, effective support, background concentration, code
perplexity, visual embedding cosine, semantic cosine, camera separation, and
JSON round trip.

- [ ] **Step 2: Implement offline evaluation**

Report:

```text
code CE/top1/top5/perplexity
codebook embedding cosine
semantic cosine
correct-vs-hard-negative retrieval
source/target/action JS
counterfactual heatmap change
DINO-warp temporal consistency
entropy/effective support/background concentration
main/wrist split metrics
```

Label hindsight-map metrics as weak-teacher agreement, never segmentation
accuracy.

- [ ] **Step 3: Implement the required visualization**

For one current image and four future keyframes, render separate main and
wrist figures containing:

```text
current RGB
future RGB target
decoded TA target
decoded TA prediction
source target/pred heatmaps
target target/pred heatmaps
action target/pred heatmaps
correct and three counterfactual text scores
8x8 coverage grid
top-32 exact locations
```

Use one shared colormap range for each target/pred pair and write a JSON sidecar
with sample ID, instruction fields, checkpoint hashes, and metrics.

- [ ] **Step 4: Encode the four ablation contracts**

Write machine-readable ablation entries:

```text
baseline_ge_act
ta_codes_only
ta_codes_plus_qwen_hidden
full_grounded_attention_bias
```

Each entry records the same GE-Act initialization, HDF5 manifest, split,
training steps, and seeds; only semantic condition switches may differ.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest -q tests/test_qwen35_grounded_evaluation.py \
  tests/test_qwen35_decoding.py \
  tests/test_qwen35_compression.py
```

Expected: all pass.

Commit:

```bash
git add qwen35_planx/evaluation.py \
  qwen35_planx/cli/evaluate_semantic_planner.py \
  qwen35_planx/cli/visualize_semantic_planner.py \
  tests/test_qwen35_grounded_evaluation.py README.md
git commit -m "feat(planx): evaluate grounded semantic plans"
```

---

### Task 14: Run Full Contract Verification and Prepare the Training Handoff

**Files:**
- Modify: `qwen35_planx/cli/preflight.py`
- Create: `tests/test_qwen35_grounded_integration.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: every artifact and launcher from Tasks 1-13.
- Produces: a single strict end-to-end preflight report and evidence that
  legacy paths remain unchanged.

- [ ] **Step 1: Write the integration test**

The tiny integration test must:

1. build one HDF5 window;
2. build/finalize one hindsight shard with fake teachers;
3. install a small fake visual vocabulary;
4. run one planner forward/backward;
5. run constrained generation for two frames in reduced test geometry;
6. compress the plan;
7. run one LTX semantic forward/backward with an action output;
8. verify teacher-only modules are absent from inference.

Keep production geometry tests separate and immutable; reduced geometry is
available only through injected test doubles.

- [ ] **Step 2: Implement the end-to-end preflight command**

`python -m qwen35_planx.cli.preflight all --config CONFIG` must check:

- model/checkpoint/cache paths and free disk;
- all hashes and exact dimensions;
- HDF5/cache split identity;
- Qwen visual-token range and tied rows;
- complete planner/joint checkpoints;
- one real data sample;
- one planner shape-only forward;
- one compressed plan;
- GE-Act configuration and all 28 semantic blocks;
- effective global batches and optimizer groups;
- absence of teacher dependencies from inference provider imports.

It writes `preflight.json` and exits nonzero on any mismatch.

- [ ] **Step 3: Run the complete focused suite**

Run:

```bash
pytest -q \
  tests/test_qwen35_grounded_config.py \
  tests/test_qwen35_instruction.py \
  tests/test_qwen35_official_ta_tok.py \
  tests/test_qwen35_hindsight_schema.py \
  tests/test_qwen35_siglip_relevance.py \
  tests/test_qwen35_temporal_grounding.py \
  tests/test_qwen35_hindsight_builder.py \
  tests/test_qwen35_vocabulary.py \
  tests/test_qwen35_sequence.py \
  tests/test_qwen35_planner_dataset.py \
  tests/test_qwen35_grounded_losses.py \
  tests/test_qwen35_grounded_planner.py \
  tests/test_qwen35_decoding.py \
  tests/test_qwen35_compression.py \
  tests/test_qwen35_provider.py \
  tests/test_ge_act_qwen35_grounded.py \
  tests/test_qwen35_grounded_evaluation.py \
  tests/test_qwen35_grounded_integration.py
```

Expected: all pass.

- [ ] **Step 4: Run legacy and source-completeness regression**

Run:

```bash
pytest -q \
  tests/test_qwen3vl2b_legacy_unchanged.py \
  tests/test_ge_act_source_completeness.py \
  tests/test_ge_act_semantic_pipeline.py \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_libero_fastwam_hdf5.py
```

Expected: all pass without selecting the new backend.

- [ ] **Step 5: Run real artifact smokes**

Run:

```bash
python -m qwen35_planx.cli.preflight released-ta \
  --ta-checkpoint "$TA_TOK_CHECKPOINT" \
  --siglip-model "$SIGLIP2_MODEL_DIR"
python -m qwen35_planx.cli.preflight hindsight-cache \
  --hdf5-manifest "$HDF5_MANIFEST" \
  --cache "$HINDSIGHT_CACHE"
python -m qwen35_planx.cli.preflight planner \
  --config "$PLANNER_CONFIG"
python -m qwen35_planx.cli.preflight joint \
  --config ge_act/configs/ltx_model/libero/action_model_libero_qwen35_grounded_hdf5.yaml
```

Expected: four successful JSON reports with matching hashes and no teacher-only
inference dependency.

- [ ] **Step 6: Update docs, ignore runtime artifacts, and commit**

Document:

- cache build/finalize/audit commands;
- planner train/resume/evaluate commands;
- joint GE-Act train/resume/evaluate commands;
- output directory contracts;
- checkpoint and hash compatibility;
- counterfactual visualization command;
- the fact that no released/model/cache weights are committed.

Ignore only new runtime cache, checkpoint, log, and visualization directories;
do not ignore source, tests, YAML, or JSON fixtures.

Commit:

```bash
git add qwen35_planx/cli/preflight.py \
  tests/test_qwen35_grounded_integration.py README.md .gitignore
git commit -m "test(planx): verify grounded planner pipeline"
```

## Final Execution Gate

Before launching the full cache build or any 30k training:

```bash
git status --short
git log --oneline --decorate -15
python -m qwen35_planx.cli.preflight all --config "$FULL_CONFIG"
```

Expected:

- worktree clean;
- fourteen task commits present;
- all focused and legacy tests pass;
- preflight JSON reports released TA-Tok `(384,27,729,65536,1536)`;
- cache and HDF5 splits/hashes match;
- planner global batch is 256;
- joint optimizer groups use the approved five learning rates;
- inference provider imports without DINOv3, SigLIP2 teacher, actions, state,
  or future RGB.

Only after this gate should the OLA hindsight cache, Qwen3.5 30k planner run,
and HPC3 30k joint GE-Act run be submitted.
