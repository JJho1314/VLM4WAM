# Qwen3.5 Baton Continuous Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Baton-style Qwen3.5-4B planner that predicts two independent cameras × four future keyframes × 256 continuous SigLIP2 patch features, then train GE-Act first with teacher features and then with frozen planner predictions.

**Architecture:** A new `qwen35_baton` package owns the continuous planner, online frozen SigLIP2 teacher, block-causal Query Tower, losses, checkpointing, and inference provider. The existing `qwen35_planx` and Qwen3-VL-2B packages remain untouched. GE-Act receives full `[B,2,4,256,1024]` grids through its existing same-camera semantic cross-attention, with explicit patch-center coordinates and no relevance bias or token compression.

**Tech Stack:** Python 3.11, PyTorch 2.x, Transformers Qwen3.5-4B and SigLIP2, Accelerate/DeepSpeed ZeRO-2, HDF5, Diffusers LTX, pytest, YAML.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-27-qwen35-baton-continuous-planner-design.md`.
- The Qwen backbone is the dense Qwen3.5-4B vision-language model.
- The teacher is frozen `SigLIP2-large-patch16-256`, with `256 x 256` inputs, penultimate vision-layer patch output, a `16 x 16` grid, and feature width `1024`.
- Camera order is exactly `("main", "wrist")`, flattened sample-major as `sample 0 main`, `sample 0 wrist`, `sample 1 main`, `sample 1 wrist`.
- Future keyframe indices are exactly `(0, 3, 5, 8)`.
- Planner output is exactly `[B,2,4,256,1024]`.
- Query Tower geometry is four pre-normalized blocks, width `1024`, `16` heads, FFN width `4096`, dropout `0.1`.
- Query self-attention and query-to-Qwen cross-attention are block-causal by future frame.
- The Sem-MLP is exactly `Linear(1024,2048) -> GELU -> Linear(2048,1024)` with no output normalization.
- Stage-1 loss weights are MSE `1.0`, cosine `0.5`, delta `0.5`, and instruction counterfactual `0.2`, with counterfactual margin `0.1`.
- Stage 1 trains the Query Tower, Sem-MLP, plan-token adapter, Qwen vision tower, and top eight Qwen language layers; the base token embedding and all other Qwen parameters remain frozen.
- Stage-1 learning rates are `5e-5` for planner modules, `1e-6` for Qwen top-eight language layers, and `5e-7` for the Qwen vision tower.
- Stage 1 runs `30,000` optimizer steps, Stage 2 runs `20,000`, and Stage 3 runs `30,000`; all save every `5,000` optimizer steps.
- Effective global batch is exactly `128` in all stages.
- Stage 2 and Stage 3 train LTX video at `2e-5`, action expert at `1e-4`, and Semantic Adapter at `5e-5`.
- Stage 2 uses online ground-truth future SigLIP2 grids; Stage 3 uses frozen planner predictions.
- Full Qwen/GE-Act joint updates, codebooks, TA-Tok, autoregressive visual decoding, DINO, DA3, depth, relevance bias, and token compression are excluded.
- Existing Qwen3-VL-2B behavior, existing `qwen35_planx` behavior, and GE-Act forward behavior without Baton conditions must remain bitwise unchanged in regression tests.
- Full feature targets are not cached to disk.

---

### Task 1: Create the Continuous Planner Contract and Package Boundary

**Files:**
- Create: `qwen35_baton/__init__.py`
- Create: `qwen35_baton/config.py`
- Create: `qwen35_baton/hashing.py`
- Create: `tests/test_qwen35_baton_config.py`
- Verify unchanged: `qwen35_planx/`
- Verify unchanged: `qwen3_vl_semantic_planner/`

**Interfaces:**
- Consumes: No new project interfaces.
- Produces: `BatonGeometry`, `BatonLossWeights`, `BatonCheckpointMetadata`, `sha256_file()`, and `sha256_json()`.

- [ ] **Step 1: Write the failing package-boundary and geometry tests**

```python
def test_baton_geometry_is_the_approved_256px_contract() -> None:
    geometry = BatonGeometry()
    assert geometry.camera_names == ("main", "wrist")
    assert geometry.future_indices == (0, 3, 5, 8)
    assert geometry.image_size == 256
    assert geometry.grid_size == 16
    assert geometry.tokens_per_frame == 256
    assert geometry.tokens_per_camera == 1024
    assert geometry.output_shape(3) == (3, 2, 4, 256, 1024)


def test_continuous_metadata_rejects_discrete_checkpoints() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload["architecture_kind"] = "qwen35_planx_grounded"
    with pytest.raises(ValueError, match="qwen35_baton_continuous"):
        BatonCheckpointMetadata.from_dict(payload)


def test_baton_package_does_not_import_legacy_planners() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "qwen35_baton").rglob("*.py"))
    )
    assert "qwen35_planx" not in source
    assert "qwen3_vl_semantic_planner" not in source
```

- [ ] **Step 2: Run the tests and confirm the package does not exist**

Run:

```bash
pytest -q tests/test_qwen35_baton_config.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'qwen35_baton'`.

- [ ] **Step 3: Implement immutable geometry, loss, and checkpoint metadata**

Use these public shapes and fixed fields:

```python
@dataclass(frozen=True)
class BatonGeometry:
    camera_names: tuple[str, ...] = ("main", "wrist")
    future_indices: tuple[int, ...] = (0, 3, 5, 8)
    image_size: int = 256
    patch_size: int = 16
    grid_size: int = 16
    feature_dim: int = 1024
    query_dim: int = 1024
    query_layers: int = 4
    query_heads: int = 16
    query_ffn_dim: int = 4096
    query_dropout: float = 0.1

    @property
    def tokens_per_frame(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def tokens_per_camera(self) -> int:
        return len(self.future_indices) * self.tokens_per_frame

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        return (
            batch_size,
            len(self.camera_names),
            len(self.future_indices),
            self.tokens_per_frame,
            self.feature_dim,
        )


@dataclass(frozen=True)
class BatonLossWeights:
    mse: float = 1.0
    cosine: float = 0.5
    delta: float = 0.5
    instruction_counterfactual: float = 0.2
    counterfactual_margin: float = 0.1
```

`BatonCheckpointMetadata` must serialize and validate the following exact contract: format version, architecture kind `qwen35_baton_continuous`, Qwen/tokenizer/processor/template hashes, added tokens and IDs, camera order, SigLIP2 artifact and preprocessing hashes, teacher layer `-2`, target shape `[2,4,256,1024]`, future indices, Query Tower dimensions and mask version, trainable Qwen layer indices, loss weights, HDF5 manifest hash, optimizer topology hash, scheduler topology hash, global step, distributed cursor, and RNG-state hash.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/test_qwen35_baton_config.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add qwen35_baton tests/test_qwen35_baton_config.py
git commit -m "feat(qwen35-baton): define continuous planner contract"
```

---

### Task 2: Build HDF5 Samples and Independent-Camera Qwen Sequences

**Files:**
- Create: `qwen35_baton/data.py`
- Create: `qwen35_baton/sequence.py`
- Create: `tests/test_qwen35_baton_data.py`
- Reuse unchanged: `ge_act/data/libero_fastwam_hdf5_dataset.py`
- Reuse unchanged: `ge_act/data/libero_fastwam_hdf5_schema.py`

**Interfaces:**
- Consumes: `BatonGeometry`.
- Produces: `BatonLiberoDataset`, `BatonPlannerBatch`, `BatonPlannerCollator`, `build_plan_text()`, and `find_plan_positions()`.

- [ ] **Step 1: Write failing tests for RGB selection and row ordering**

Use a fake base dataset returning normalized video in `[C,V,T,H,W]` and records with `domain` fields:

```python
def test_dataset_selects_current_and_four_future_frames() -> None:
    sample = dataset[0]
    assert sample["current_images"].shape == (2, 3, 256, 256)
    assert sample["future_images"].shape == (2, 4, 3, 256, 256)
    torch.testing.assert_close(
        sample["future_images"],
        expected_video[:, [4, 7, 9, 12]],
    )
    assert sample["suite"] == "libero_object"


def test_collator_orders_positive_then_negative_sample_major_rows() -> None:
    batch = collator([sample0, sample1])
    assert batch.row_labels == (
        ("positive", 0, "main"),
        ("positive", 0, "wrist"),
        ("positive", 1, "main"),
        ("positive", 1, "wrist"),
        ("negative", 0, "main"),
        ("negative", 0, "wrist"),
        ("negative", 1, "main"),
        ("negative", 1, "wrist"),
    )
    assert batch.plan_positions.shape == (8, 1024)
    assert torch.all(
        batch.qwen_inputs["input_ids"].gather(1, batch.plan_positions)
        == plan_pad_token_id
    )
```

Also assert that changing wrist pixels never changes the processor fields belonging to the main row and that every negative instruction differs from its positive instruction while retaining the same suite.

- [ ] **Step 2: Run the focused tests and confirm missing interfaces**

Run:

```bash
pytest -q tests/test_qwen35_baton_data.py
```

Expected: import fails for `qwen35_baton.data`.

- [ ] **Step 3: Implement the fixed plan template**

`build_plan_text(instruction)` must emit one start token, four frame tokens, exactly `256` `<PLAN_PAD>` tokens per frame, and one end token:

```python
PLAN_START = "<PLAN_START>"
FRAME_TOKENS = tuple(f"<FRAME_{index}>" for index in range(4))
PLAN_PAD = "<PLAN_PAD>"
PLAN_END = "<PLAN_END>"
ADDED_TOKENS = (PLAN_START, *FRAME_TOKENS, PLAN_PAD, PLAN_END)


def build_plan_text(instruction: str) -> str:
    blocks = [
        f"{FRAME_TOKENS[index]} " + " ".join([PLAN_PAD] * 256)
        for index in range(4)
    ]
    return (
        f"Instruction: {instruction}\n"
        f"{PLAN_START}\n"
        + "\n".join(blocks)
        + f"\n{PLAN_END}"
    )
```

`find_plan_positions(input_ids, plan_pad_token_id)` must return exactly `1024` positions per row or fail closed.

- [ ] **Step 4: Implement the dataset and collator**

`BatonLiberoDataset` wraps `LiberoFastWAMHDF5Dataset` without changing it. It converts the normalized `[C,V,T,H,W]` result back to uint8 RGB, selects current position `n_previous - 1`, and selects future positions `n_previous + (0,3,5,8)`.

Use this batch contract:

```python
@dataclass(frozen=True)
class BatonPlannerBatch:
    qwen_inputs: Mapping[str, torch.Tensor]
    plan_positions: torch.Tensor
    current_images: torch.Tensor
    future_images: torch.Tensor
    instructions: tuple[str, ...]
    negative_instructions: tuple[str, ...]
    row_labels: tuple[tuple[str, int, str], ...]

    @property
    def batch_size(self) -> int:
        return len(self.instructions)

    @property
    def positive_rows(self) -> slice:
        return slice(0, self.batch_size * 2)

    @property
    def negative_rows(self) -> slice:
        return slice(self.batch_size * 2, self.batch_size * 4)
```

Build a sorted suite-to-caption table from the manifest. Select the negative caption by hashing `(seed, episode_key, positive_caption)` and indexing candidates after removing the positive caption. Fail during dataset construction if any suite has fewer than two distinct instructions. Process positive rows first, then negative rows; inside each condition use sample-major main/wrist order.

- [ ] **Step 5: Run data tests**

Run:

```bash
pytest -q tests/test_qwen35_baton_data.py tests/test_libero_fastwam_hdf5.py
```

Expected: all tests pass, including the unchanged base HDF5 tests.

- [ ] **Step 6: Commit the data contract**

```bash
git add qwen35_baton/data.py qwen35_baton/sequence.py tests/test_qwen35_baton_data.py
git commit -m "feat(qwen35-baton): build independent camera plan batches"
```

---

### Task 3: Add the Online Frozen SigLIP2 Teacher

**Files:**
- Create: `qwen35_baton/teacher.py`
- Create: `tests/test_qwen35_baton_teacher.py`
- Reuse reference only: `ge_act/models/ltx_models/semantic_conditioning.py`
- Use local artifact: `third_party/siglip2-large-patch16-256/`

**Interfaces:**
- Consumes: uint8 or normalized RGB current/future tensors from Task 2.
- Produces: `FrozenSiglip2Teacher.encode_current()` returning `[B,2,256,1024]` and `encode_future()` returning `[B,2,4,256,1024]`.

- [ ] **Step 1: Write failing extraction and gradient tests**

```python
def test_teacher_extracts_penultimate_patch_grid() -> None:
    teacher = FrozenSiglip2Teacher.from_components(
        processor=fake_processor,
        vision_model=fake_vision_model,
        frame_microbatch_size=3,
    )
    features = teacher.encode_future(torch.randint(256, (2, 2, 4, 3, 256, 256), dtype=torch.uint8))
    assert features.shape == (2, 2, 4, 256, 1024)
    assert fake_vision_model.requested_hidden_states is True
    assert fake_vision_model.training is False


def test_teacher_targets_are_detached() -> None:
    future = teacher.encode_future(images)
    current = teacher.encode_current(current_images)
    assert future.requires_grad is False
    assert current.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in teacher.model.parameters())
```

Add a processor-consistency test comparing teacher pixel values against a direct call to the persisted `AutoImageProcessor` on the same two fixture images.

- [ ] **Step 2: Run the tests and confirm the teacher is missing**

Run:

```bash
pytest -q tests/test_qwen35_baton_teacher.py
```

Expected: import fails for `qwen35_baton.teacher`.

- [ ] **Step 3: Implement the frozen vision-only teacher**

Load `AutoProcessor` and `AutoModel`, retain only `vision_model`, set evaluation mode, call `requires_grad_(False)`, and wrap both public encode methods in `torch.no_grad()`. Request hidden states and extract `hidden_states[-2]`; discard any leading non-patch token only when the returned length is `257`, and otherwise require exactly `256`.

The internal frame method has this exact signature:

```python
def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
    """Encode [N,3,H,W] RGB into [N,256,1024] detached patch features."""
```

Process at most `frame_microbatch_size` frames per vision forward, concatenate in original order, and reject nonfinite outputs.

- [ ] **Step 4: Run teacher and existing SigLIP tests**

Run:

```bash
pytest -q \
  tests/test_qwen35_baton_teacher.py \
  tests/test_ge_act_ltx_semantic_guidance.py::test_online_encoder_discards_the_unused_siglip_text_tower
```

Expected: all tests pass.

- [ ] **Step 5: Commit the teacher**

```bash
git add qwen35_baton/teacher.py tests/test_qwen35_baton_teacher.py
git commit -m "feat(qwen35-baton): add online frozen SigLIP2 teacher"
```

---

### Task 4: Implement the Block-Causal Spatiotemporal Query Tower

**Files:**
- Create: `qwen35_baton/query_tower.py`
- Create: `tests/test_qwen35_baton_query_tower.py`

**Interfaces:**
- Consumes: gathered Qwen plan states `[B*2,4,256,D_qwen]` and camera IDs `[B*2]`.
- Produces: query states `[B*2,4,256,1024]` and, only when requested for visualization, head-mean cross-attention maps `[layers,B*2,1024,1024]`.

- [ ] **Step 1: Write failing mask, position, isolation, and shape tests**

```python
def test_block_causal_mask_allows_same_and_past_frames_only() -> None:
    allowed = build_block_causal_allowed_mask(num_frames=4, tokens_per_frame=3)
    assert allowed.shape == (12, 12)
    assert allowed[4, :6].all()
    assert not allowed[4, 6:].any()
    assert allowed[0, :3].all()
    assert not allowed[0, 3:].any()


def test_query_tower_keeps_one_query_per_target_patch() -> None:
    tower = SpatiotemporalQueryTower._from_test_config(
        qwen_dim=32,
        query_dim=16,
        num_frames=4,
        tokens_per_frame=4,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
    )
    output = tower(torch.randn(4, 4, 4, 32), camera_ids=torch.tensor([0, 1, 0, 1]))
    assert output.hidden_states.shape == (4, 4, 4, 16)


def test_camera_rows_are_isolated() -> None:
    changed = qwen_states.clone()
    changed[1].mul_(1000)
    actual = tower(qwen_states, camera_ids)
    mutated = tower(changed, camera_ids)
    torch.testing.assert_close(actual.hidden_states[0], mutated.hidden_states[0])
    assert not torch.allclose(actual.hidden_states[1], mutated.hidden_states[1])
```

Add a gradient-based mask test: backpropagate from frame 1 output, assert nonzero gradients for context frames 0 and 1, and exactly zero gradients for context frames 2 and 3.

- [ ] **Step 2: Run the tests and confirm the tower is absent**

Run:

```bash
pytest -q tests/test_qwen35_baton_query_tower.py
```

Expected: import fails for `qwen35_baton.query_tower`.

- [ ] **Step 3: Implement 3D RoPE and block-causal attention**

Use per-head width `64` and split rotary dimensions `(16,24,24)` across `(t,y,x)`. Patch positions are row-major `(frame, y, x)`. `BlockCausalAttention` must use explicit Q/K/V projections and `torch.nn.functional.scaled_dot_product_attention`; convert the public allowed mask to an additive mask internally so tests never depend on backend boolean-mask conventions.

Use this block order:

```python
x = x + self.self_attention(self.self_norm(x), self_positions, self_allowed)
x = x + self.cross_attention(
    self.cross_norm(x),
    context,
    query_positions,
    context_positions,
    cross_allowed,
)
x = x + self.feed_forward(self.ffn_norm(x))
```

Return:

```python
@dataclass(frozen=True)
class QueryTowerOutput:
    hidden_states: torch.Tensor
    cross_attention_maps: tuple[torch.Tensor, ...] | None
```

The production constructor always uses the approved `BatonGeometry`. `_from_test_config()` is the only path that accepts reduced dimensions and is rejected by checkpoint serialization. The initial query is the sum of learned `[4,256,1024]` queries, frame embeddings, two-dimensional spatial embeddings, and the selected camera-view embedding. Project Qwen context from `D_qwen` to `1024` once before the four blocks.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/test_qwen35_baton_query_tower.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the Query Tower**

```bash
git add qwen35_baton/query_tower.py tests/test_qwen35_baton_query_tower.py
git commit -m "feat(qwen35-baton): add block-causal query tower"
```

---

### Task 5: Connect Qwen3.5 Hidden States to Continuous SigLIP Predictions

**Files:**
- Create: `qwen35_baton/model.py`
- Create: `qwen35_baton/ownership.py`
- Create: `tests/test_qwen35_baton_model.py`
- Reuse unchanged: `qwen35_planx/planner.py` only as behavioral reference for `gather_positions`.

**Interfaces:**
- Consumes: `BatonPlannerBatch` and `SpatiotemporalQueryTower`.
- Produces: `BatonPlannerOutput`, `PlanTokenEmbeddingAdapter`, `BatonQwen35Planner.forward_rows()`, and `configure_stage1_trainable_modules()`.

- [ ] **Step 1: Write failing model shape and token-adapter tests**

```python
def test_model_splits_positive_and_negative_predictions() -> None:
    output = planner(batch)
    assert output.positive.shape == (2, 2, 4, 256, 1024)
    assert output.negative.shape == (2, 2, 4, 256, 1024)
    assert output.flat.shape == (8, 4, 256, 1024)


def test_plan_token_adapter_changes_only_added_rows() -> None:
    embeddings = adapter(input_ids)
    base = frozen_base_embedding(input_ids)
    ordinary = ~torch.isin(input_ids, added_token_ids)
    torch.testing.assert_close(embeddings[ordinary], base[ordinary])
    embeddings.sum().backward()
    assert adapter.plan_embeddings.weight.grad is not None
    assert frozen_base_embedding.weight.grad is None
```

Add a test that permuting wrist rows changes only wrist predictions, and a test that a batch with fewer or more than `1024` `<PLAN_PAD>` positions fails before invoking Qwen.

- [ ] **Step 2: Run the tests and confirm missing model interfaces**

Run:

```bash
pytest -q tests/test_qwen35_baton_model.py
```

Expected: import fails for `qwen35_baton.model`.

- [ ] **Step 3: Implement the embedding overlay and Qwen hidden gather**

`PlanTokenEmbeddingAdapter` wraps the frozen resized Qwen input embedding. It owns a trainable `[7,D_qwen]` table for `<PLAN_START>`, four frame tokens, `<PLAN_PAD>`, and `<PLAN_END>`, replacing only matching token positions during embedding lookup.

Use this output contract:

```python
@dataclass(frozen=True)
class BatonPlannerOutput:
    flat: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor | None
    cross_attention_maps: tuple[torch.Tensor, ...] | None
```

`BatonQwen35Planner.forward_rows(qwen_inputs, plan_positions, camera_ids)` performs one Qwen causal forward, gathers `[rows,1024,D_qwen]`, reshapes to `[rows,4,256,D_qwen]`, calls the Query Tower, then applies:

```python
self.sem_mlp = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.GELU(),
    nn.Linear(2048, 1024),
)
```

`forward(batch)` calls `forward_rows` once for the already-concatenated positive and negative rows and restores `[B,2,4,256,1024]` using the slices in `BatonPlannerBatch`.

- [ ] **Step 4: Implement explicit parameter ownership**

`configure_stage1_trainable_modules()` must first freeze the full planner, then enable exactly:

```python
Stage1Ownership(
    planner_modules=(planner.query_tower, planner.sem_mlp, planner.plan_token_adapter),
    qwen_top_layers=tuple(language_layers[-8:]),
    qwen_vision_modules=(resolved_vision_module,),
)
```

Resolve language and vision modules by explicit supported attribute paths, require exactly one resolution, reject overlaps, and prove that the union of owned parameter IDs equals the set of trainable parameter IDs.

- [ ] **Step 5: Run model tests**

Run:

```bash
pytest -q tests/test_qwen35_baton_model.py tests/test_qwen3vl2b_legacy_unchanged.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Qwen integration**

```bash
git add qwen35_baton/model.py qwen35_baton/ownership.py tests/test_qwen35_baton_model.py
git commit -m "feat(qwen35-baton): predict continuous SigLIP grids"
```

---

### Task 6: Implement the Four Stage-1 Objectives

**Files:**
- Create: `qwen35_baton/losses.py`
- Create: `tests/test_qwen35_baton_losses.py`

**Interfaces:**
- Consumes: positive/negative predictions plus current/future teacher features.
- Produces: `BatonPlannerLoss`, `changed_patch_weights()`, and `compute_baton_planner_loss()`.

- [ ] **Step 1: Write failing numerical tests**

```python
def test_changed_patch_weights_are_per_frame_and_bounded() -> None:
    weight = changed_patch_weights(future_teacher, current_teacher)
    assert weight.shape == (2, 2, 4, 256)
    assert float(weight.min()) >= 1.0
    assert float(weight.max()) <= 3.0


def test_perfect_prediction_has_zero_primary_losses() -> None:
    loss = compute_baton_planner_loss(
        positive=future_teacher,
        negative=wrong_prediction,
        future_teacher=future_teacher,
        current_teacher=current_teacher,
    )
    torch.testing.assert_close(loss.mse, torch.zeros_like(loss.mse))
    torch.testing.assert_close(loss.cosine, torch.zeros_like(loss.cosine), atol=1e-6, rtol=0)
    torch.testing.assert_close(loss.delta, torch.zeros_like(loss.delta))


def test_counterfactual_hinge_requires_correct_instruction_to_rank_better() -> None:
    good = future_teacher.clone()
    bad = -future_teacher
    ranked = compute_baton_planner_loss(good, bad, future_teacher, current_teacher)
    reversed_rank = compute_baton_planner_loss(bad, good, future_teacher, current_teacher)
    assert ranked.instruction_counterfactual < reversed_rank.instruction_counterfactual
```

Also test that static copying incurs nonzero delta loss when the future changes and that total loss equals the exact weighted sum.

- [ ] **Step 2: Run tests and confirm missing loss module**

Run:

```bash
pytest -q tests/test_qwen35_baton_losses.py
```

Expected: import fails for `qwen35_baton.losses`.

- [ ] **Step 3: Implement the exact loss definitions**

```python
change = torch.linalg.vector_norm(
    future_teacher - current_teacher[:, :, None],
    dim=-1,
)
mean_change = change.mean(dim=-1, keepdim=True)
patch_weight = 1.0 + torch.clamp(
    change / (mean_change + 1e-6),
    min=0.0,
    max=2.0,
)
```

Compute feature MSE per patch before weighting and normalize by the sum of patch weights. Compute cosine distance over the feature dimension. Compute delta as mean squared distance between predicted and teacher deltas. Compute the counterfactual hinge from per-example cosine distances:

```python
@dataclass(frozen=True)
class BatonPlannerLoss:
    mse: torch.Tensor
    cosine: torch.Tensor
    delta: torch.Tensor
    instruction_counterfactual: torch.Tensor
    total: torch.Tensor


def compute_baton_planner_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    future_teacher: torch.Tensor,
    current_teacher: torch.Tensor,
    loss_weights: BatonLossWeights = BatonLossWeights(),
) -> BatonPlannerLoss:
    patch_weight = changed_patch_weights(future_teacher, current_teacher)
    patch_mse = (positive - future_teacher).square().mean(dim=-1)
    mse_per_sample = (
        (patch_mse * patch_weight).flatten(1).sum(dim=1)
        / patch_weight.flatten(1).sum(dim=1).clamp_min(1e-12)
    )
    cosine_per_sample = (
        1.0 - F.cosine_similarity(positive, future_teacher, dim=-1)
    ).flatten(1).mean(dim=1)
    predicted_delta = positive - current_teacher[:, :, None]
    teacher_delta = future_teacher - current_teacher[:, :, None]
    delta_per_sample = (
        predicted_delta - teacher_delta
    ).square().flatten(1).mean(dim=1)
    correct_distance = (
        1.0 - F.cosine_similarity(positive, future_teacher, dim=-1)
    ).flatten(1).mean(dim=1)
    wrong_distance = (
        1.0 - F.cosine_similarity(negative, future_teacher, dim=-1)
    ).flatten(1).mean(dim=1)
    instruction_cf_per_sample = torch.relu(
        loss_weights.counterfactual_margin
        + correct_distance
        - wrong_distance
    )
    mse = mse_per_sample.mean()
    cosine = cosine_per_sample.mean()
    delta = delta_per_sample.mean()
    instruction_cf = instruction_cf_per_sample.mean()
    total = (
        loss_weights.mse * mse
        + loss_weights.cosine * cosine
        + loss_weights.delta * delta
        + loss_weights.instruction_counterfactual * instruction_cf
    )
    return BatonPlannerLoss(
        mse=mse,
        cosine=cosine,
        delta=delta,
        instruction_counterfactual=instruction_cf,
        total=total,
    )
```

All reductions first average feature/patch/frame/camera dimensions per sample and finally average samples so global-batch normalization is independent of gradient accumulation.

- [ ] **Step 4: Run loss tests**

Run:

```bash
pytest -q tests/test_qwen35_baton_losses.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the objectives**

```bash
git add qwen35_baton/losses.py tests/test_qwen35_baton_losses.py
git commit -m "feat(qwen35-baton): add continuous planner objectives"
```

---

### Task 7: Add Stage-1 Training, Preflight, Checkpointing, and Exact Resume

**Files:**
- Create: `qwen35_baton/checkpoint.py`
- Create: `qwen35_baton/cli/__init__.py`
- Create: `qwen35_baton/cli/preflight.py`
- Create: `qwen35_baton/cli/train_semantic_planner.py`
- Create: `qwen35_baton/configs/libero_stage1.json`
- Create: `qwen35_baton/scripts/train_semantic_planner.sh`
- Create: `qwen35_baton/scripts/sbatch_train_semantic_planner_hpc3.sh`
- Create: `ge_act/requirements-qwen35-baton.txt`
- Create: `tests/test_qwen35_baton_training.py`
- Create: `tests/test_qwen35_baton_checkpoint.py`
- Reference: `qwen35_planx/cli/train_semantic_planner.py`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces: `save_baton_checkpoint()`, `load_baton_checkpoint()`, `BatonResumeState`, a resumable Stage-1 launcher, and checkpoint directories `step_005000` through `step_030000`.

- [ ] **Step 1: Write failing optimizer and training-step tests**

```python
def test_stage1_optimizer_groups_are_exact_and_exhaustive() -> None:
    groups = build_stage1_optimizer_groups(planner, ownership, config)
    assert {group["name"]: group["lr"] for group in groups} == {
        "planner": 5e-5,
        "qwen_top8": 1e-6,
        "qwen_vision": 5e-7,
    }
    grouped_ids = [id(p) for group in groups for p in group["params"]]
    trainable_ids = [id(p) for p in planner.parameters() if p.requires_grad]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)


def test_one_tiny_stage1_step_updates_only_owned_parameters(tmp_path: Path) -> None:
    before = clone_parameters(planner)
    result = run_training(tiny_config(tmp_path, max_steps=1), artifacts=tiny_artifacts)
    assert result.global_step == 1
    assert_owned_parameters_changed(before, planner, ownership)
    assert_frozen_parameters_unchanged(before, planner)
    assert all(parameter.grad is None for parameter in teacher.model.parameters())
```

Add tests for global batch validation (`per_gpu * world_size * accumulation == 128`), save cadence, cosine warmup scheduler, and nonfinite-loss failure.

- [ ] **Step 2: Write failing checkpoint compatibility and resume tests**

```python
def test_continuous_loader_rejects_ta_tok_metadata(tmp_path: Path) -> None:
    write_metadata(tmp_path, {"architecture_kind": "qwen35_planx_grounded"})
    with pytest.raises(ValueError, match="qwen35_baton_continuous"):
        load_baton_checkpoint(
            tmp_path,
            planner=tiny_planner,
            optimizer=None,
            scheduler=None,
            expected_contract=BatonCheckpointMetadata.example(),
        )


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    full = run_tiny_training(tmp_path / "full", stop_at=4)
    first = run_tiny_training(tmp_path / "resume", stop_at=2)
    resumed = run_tiny_training(tmp_path / "resume", resume=first.checkpoint, stop_at=4)
    assert_state_dict_equal(full.model_state, resumed.model_state)
    assert_nested_equal(full.optimizer_state, resumed.optimizer_state)
    assert full.cursor == resumed.cursor
```

Interrupt after a non-epoch-boundary optimizer step with gradient accumulation greater than one. Include Python, NumPy, CPU Torch, every CUDA RNG stream, sampler epoch, consumed microbatch, optimizer, scheduler, scaler, and distributed rank in the saved state.

- [ ] **Step 3: Run tests and confirm missing CLI/checkpoint code**

Run:

```bash
pytest -q tests/test_qwen35_baton_training.py tests/test_qwen35_baton_checkpoint.py
```

Expected: imports fail for the new modules.

- [ ] **Step 4: Implement production artifact loading and preflight**

Use these checkpoint interfaces:

```python
@dataclass(frozen=True)
class BatonTrainingCursor:
    global_step: int
    epoch: int
    consumed_microbatches: int
    microbatches_per_epoch: int
    sampler_seed: int


@dataclass(frozen=True)
class BatonResumeState:
    metadata: BatonCheckpointMetadata
    cursor: BatonTrainingCursor
    rank_rng_state: Mapping[str, Any]


def save_baton_checkpoint(
    checkpoint_dir: Path,
    *,
    planner: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metadata: BatonCheckpointMetadata,
    cursor: BatonTrainingCursor,
    rank_rng_state: Mapping[str, Any],
) -> None:
    """Atomically save model, optimizer, scheduler, cursor, metadata, and rank RNG."""


def load_baton_checkpoint(
    checkpoint_dir: Path,
    *,
    planner: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    expected_contract: BatonCheckpointMetadata,
) -> BatonResumeState:
    """Validate metadata before loading any tensor state and restore exact resume state."""
```

The implementations write into a sibling temporary directory, fsync files, and atomically rename only after every rank has published its RNG state. The loader validates metadata, file hashes, topology, and cursor consistency before mutating planner/optimizer/scheduler objects.

Load local Qwen3.5-4B with its persisted processor/tokenizer and local SigLIP2 with `local_files_only=True`. Before GPU allocation, validate:

- all model and processor paths exist;
- the Qwen config is dense Qwen3.5-4B;
- all seven added tokens map to unique IDs;
- SigLIP2 reports image size `256`, patch size `16`, and hidden width `1024`;
- the HDF5 manifest hash matches runtime metadata;
- world size, per-device batch, and accumulation multiply to `128`;
- no output directory is an ancestor of the model or dataset paths.

`ge_act/requirements-qwen35-baton.txt` contains `-r requirements.txt` and no unpinned overrides; the pinned base already supplies PyTorch `2.7.1`, Transformers `5.14.1`, Accelerate `1.14.0`, DeepSpeed `0.16.9`, Diffusers `0.35.2`, safetensors, h5py, and pytest.

The preflight command is:

```bash
python -m qwen35_baton.cli.preflight \
  --config qwen35_baton/configs/libero_stage1.json \
  --world-size 8
```

- [ ] **Step 5: Implement the Stage-1 training loop**

For every batch:

```python
planner_output = planner(batch)
with torch.no_grad():
    future_teacher = teacher.encode_future(batch.future_images)
    current_teacher = teacher.encode_current(batch.current_images)
losses = compute_baton_planner_loss(
    planner_output.positive,
    planner_output.negative,
    future_teacher,
    current_teacher,
)
with accelerator.accumulate(planner):
    accelerator.backward(losses.total)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

Configure `Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)` so loss normalization occurs once inside `accelerator.accumulate`; do not divide the loss manually. Log total loss plus all four components, per-camera/per-keyframe MSE and cosine, counterfactual ranking accuracy, throughput, data time, Qwen time, teacher time, Query Tower time, and backward time. Save model tensors in safetensors plus metadata JSON, optimizer/scheduler states, distributed cursor, and per-rank RNG state every `5,000` optimizer steps.

- [ ] **Step 6: Implement fixed launchers**

The shell launcher must derive accumulation from `GLOBAL_BATCH=128`, `NUM_GPUS`, and `PER_DEVICE_BATCH`; reject non-divisible values; run preflight first; set offline Hugging Face flags; and launch with `torch.distributed.run`. The HPC3 launcher requests eight GPUs and delegates to the shell launcher without duplicating training arguments.

- [ ] **Step 7: Run training and checkpoint tests**

Run:

```bash
pytest -q \
  tests/test_qwen35_baton_training.py \
  tests/test_qwen35_baton_checkpoint.py \
  tests/test_qwen35_baton_config.py \
  tests/test_qwen35_baton_data.py \
  tests/test_qwen35_baton_teacher.py \
  tests/test_qwen35_baton_query_tower.py \
  tests/test_qwen35_baton_model.py \
  tests/test_qwen35_baton_losses.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Stage 1**

```bash
git add qwen35_baton ge_act/requirements-qwen35-baton.txt \
  tests/test_qwen35_baton_training.py tests/test_qwen35_baton_checkpoint.py
git commit -m "feat(qwen35-baton): train and resume continuous planner"
```

---

### Task 8: Add the Frozen Planner Provider and Attention Visualization

**Files:**
- Create: `qwen35_baton/provider.py`
- Create: `qwen35_baton/visualization.py`
- Create: `qwen35_baton/cli/visualize_attention.py`
- Create: `tests/test_qwen35_baton_provider.py`
- Create: `tests/test_qwen35_baton_visualization.py`

**Interfaces:**
- Consumes: a validated Stage-1 checkpoint and current `[B,2,3,H,W]` RGB.
- Produces: `BatonSemanticPlan` containing full grids, fixed future indices, normalized patch centers, optional Query Tower cross-attention maps, and correct-versus-counterfactual instruction sensitivity.

- [ ] **Step 1: Write failing provider contract tests**

```python
def test_provider_returns_full_independent_camera_grids() -> None:
    plan = provider.predict(current_images, instructions, return_attention=True)
    assert plan.tokens.shape == (2, 2, 4, 256, 1024)
    assert plan.future_indices == (0, 3, 5, 8)
    assert plan.positions_xy.shape == (2, 2, 4, 256, 2)
    torch.testing.assert_close(plan.positions_xy[0, 0, 0, 0], torch.tensor([1 / 32, 1 / 32]))
    torch.testing.assert_close(plan.positions_xy[0, 0, 0, -1], torch.tensor([31 / 32, 31 / 32]))
    assert plan.relevance is None
```

Add a checkpoint test that changes one tokenizer hash and asserts provider loading fails before model inference.

- [ ] **Step 2: Write a failing visualization smoke test**

```python
def test_attention_visualization_writes_one_panel_per_camera(tmp_path: Path) -> None:
    paths = render_attention_panels(sample, plan, output_dir=tmp_path)
    assert [path.name for path in paths] == ["sample_000_main.png", "sample_000_wrist.png"]
    assert all(path.stat().st_size > 0 for path in paths)
```

The panel must include current RGB, instruction, four future-keyframe attention overlays, camera name, layer/head aggregation rule, and no query-token bounding-box panel.

- [ ] **Step 3: Run the tests and confirm missing provider**

Run:

```bash
pytest -q tests/test_qwen35_baton_provider.py tests/test_qwen35_baton_visualization.py
```

Expected: imports fail for provider and visualization modules.

- [ ] **Step 4: Implement frozen inference**

```python
@dataclass(frozen=True)
class BatonSemanticPlan:
    tokens: torch.Tensor
    future_indices: tuple[int, ...]
    positions_xy: torch.Tensor
    cross_attention_maps: tuple[torch.Tensor, ...] | None
    instruction_sensitivity: torch.Tensor | None
    relevance: None = None
```

`FrozenBatonPlanner` is an `nn.Module`. `FrozenBatonPlanner.from_checkpoint()` validates the complete metadata contract, sets every module to eval/frozen, and constructs positive-only Qwen rows with the same processor/template as training. `predict()` runs under `torch.no_grad()` and rejects nonfinite or incorrectly shaped output. When visualization requests instruction sensitivity, make one batched positive/counterfactual forward and compute `1 - cosine(positive, counterfactual)` per `(camera,keyframe,patch)` without updating weights.

- [ ] **Step 5: Implement attention rendering**

Use correct-versus-counterfactual instruction sensitivity as the spatial heatmap: normalize each future frame only for display, reshape to `16 x 16`, bilinearly resize to the RGB size, and alpha-blend with a fixed colormap. Label it `instruction-conditioned query sensitivity`, not raw image-token attention. Separately summarize Query Tower cross-attention by averaging heads/layers and summing keys within each of the four Qwen plan blocks; write those unnormalized frame-to-frame attention values and the raw sensitivity arrays to compressed `.npz` files. This prevents the plan-pad attention from being misrepresented as direct image-pixel attention.

- [ ] **Step 6: Run provider and visualization tests**

Run:

```bash
pytest -q tests/test_qwen35_baton_provider.py tests/test_qwen35_baton_visualization.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit provider and visualization**

```bash
git add qwen35_baton/provider.py qwen35_baton/visualization.py \
  qwen35_baton/cli/visualize_attention.py \
  tests/test_qwen35_baton_provider.py tests/test_qwen35_baton_visualization.py
git commit -m "feat(qwen35-baton): add frozen provider and attention maps"
```

---

### Task 9: Inject Full Baton Grids into GE-Act with Exact Coordinates

**Files:**
- Create: `ge_act/models/ltx_models/baton_semantic_planner.py`
- Modify: `ge_act/models/ltx_models/semantic_conditioning.py`
- Modify: `ge_act/models/ltx_models/transformer_ltx_multiview.py`
- Create: `tests/test_ge_act_baton_semantic_guidance.py`
- Preserve: `tests/test_ge_act_ltx_semantic_guidance.py`

**Interfaces:**
- Consumes: `BatonSemanticPlan` or online teacher features `[B,2,4,256,1024]`.
- Produces: `build_patch_center_positions()`, `build_baton_semantic_context()`, and same-camera `SemanticContext` with `[B*2,1024,LTX_hidden]` and exact `(t,y,x)` positions.

- [ ] **Step 1: Write failing exact-coordinate and full-grid tests**

```python
def test_baton_patch_centers_are_exact_sixteenth_grid_centers() -> None:
    xy = build_patch_center_positions(
        batch_size=1,
        num_views=2,
        num_keyframes=4,
        grid_size=16,
    )
    assert xy.shape == (1, 2, 4, 256, 2)
    torch.testing.assert_close(xy[0, 0, 0, 0], torch.tensor([1 / 32, 1 / 32]))
    torch.testing.assert_close(xy[0, 0, 0, -1], torch.tensor([31 / 32, 31 / 32]))


def test_baton_semantic_context_keeps_all_1024_tokens_per_camera() -> None:
    context = build_baton_semantic_context(
        model,
        plan,
        n_previous=4,
        num_future_frames=9,
        latent_shape=(6, 32, 32),
    )
    assert context.hidden_states.shape == (2, 1024, model.inner_dim)
    assert context.positions.shape == (2, 1024, 3)
    assert context.relevance is None
```

Add a same-camera test mutating wrist context and proving main latent output is unchanged, and a block-order hook test proving text cross-attention executes before semantic cross-attention.

- [ ] **Step 2: Write a failing legacy zero-condition regression**

Clone a one-block LTX model before and after Baton support, load identical weights, omit semantic inputs, and require exact equality with `rtol=0, atol=0`.

- [ ] **Step 3: Run tests and confirm Baton GE-Act path is absent**

Run:

```bash
pytest -q tests/test_ge_act_baton_semantic_guidance.py
```

Expected: import fails for `baton_semantic_planner`.

- [ ] **Step 4: Implement exact positions without changing legacy defaults**

Add:

```python
def build_patch_center_positions(
    batch_size: int,
    num_views: int,
    num_keyframes: int,
    grid_size: int = 16,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    centers = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / grid_size
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    xy = torch.stack((x, y), dim=-1).reshape(1, 1, 1, grid_size * grid_size, 2)
    return xy.expand(batch_size, num_views, num_keyframes, -1, -1).contiguous()
```

Always pass these explicit positions for Baton Stage 2 and Stage 3. Do not change the default coordinate behavior used by existing `gt_siglip2`, `vlm_planner`, or `qwen35_grounded` sources.

- [ ] **Step 5: Implement the GE-Act Baton adapter**

`build_baton_semantic_context(model, plan, *, n_previous, num_future_frames, latent_shape)` validates the full-grid contract, derives full-clip times from `plan.future_indices`, and calls `model.semantic_adapter` with explicit positions. `FrozenDualCameraBatonPlanner` wraps `qwen35_baton.provider.FrozenBatonPlanner`. Its `predict()` returns all four keyframes, full `256` tokens, fixed future indices, exact normalized patch centers, and no relevance tensor. Keep semantic residual modulation zero-initialized. In selected blocks the existing execution order remains self-attention, text cross-attention, Baton semantic cross-attention, FFN.

- [ ] **Step 6: Run Baton and legacy semantic tests**

Run:

```bash
pytest -q \
  tests/test_ge_act_baton_semantic_guidance.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_qwen35_grounded.py \
  tests/test_ge_act_vlm_semantic_planner.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the GE-Act semantic interface**

```bash
git add ge_act/models/ltx_models/baton_semantic_planner.py \
  ge_act/models/ltx_models/semantic_conditioning.py \
  ge_act/models/ltx_models/transformer_ltx_multiview.py \
  tests/test_ge_act_baton_semantic_guidance.py
git commit -m "feat(ge-act): inject full Baton semantic grids"
```

---

### Task 10: Add Stage-2 Teacher Training and Stage-3 Predicted Training

**Files:**
- Modify: `ge_act/runner/ge_trainer.py`
- Modify: `ge_act/runner/ge_inferencer.py`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py`
- Create: `ge_act/configs/ltx_model/libero/action_model_libero_baton_stage2_hdf5.yaml`
- Create: `ge_act/configs/ltx_model/libero/action_model_libero_baton_stage3_hdf5.yaml`
- Create: `ge_act/scripts/train_ltx_baton_stage2.sh`
- Create: `ge_act/scripts/train_ltx_baton_stage3.sh`
- Create: `ge_act/scripts/sbatch_train_ltx_baton_stage2_hpc3.sh`
- Create: `ge_act/scripts/sbatch_train_ltx_baton_stage3_hpc3.sh`
- Create: `tests/test_ge_act_baton_training_contract.py`

**Interfaces:**
- Consumes: online teacher from Task 3, frozen planner provider from Task 8, and semantic GE-Act path from Task 9.
- Produces: `BatonConditioningComponents`, `prepare_baton_conditioning()`, and two independent, resumable GE-Act training recipes.

- [ ] **Step 1: Write failing source-mode tests**

```python
def test_stage2_uses_online_teacher_and_never_loads_planner() -> None:
    components = prepare_baton_conditioning(
        stage2_config,
        dataset,
        device="cpu",
        dtype=torch.float32,
    )
    assert components.source == "qwen35_baton_teacher"
    assert components.teacher is not None
    assert components.planner is None
    assert all(not p.requires_grad for p in components.teacher.model.parameters())


def test_stage3_uses_frozen_planner_and_never_loads_teacher() -> None:
    components = prepare_baton_conditioning(
        stage3_config,
        dataset,
        device="cpu",
        dtype=torch.float32,
    )
    assert components.source == "qwen35_baton_prediction"
    assert components.teacher is None
    assert components.planner is not None
    assert all(not p.requires_grad for p in components.planner.parameters())
```

Add a trainer-forward test for each source that asserts `semantic_plan_positions` is passed, `semantic_plan_relevance is None`, and semantic tokens remain `[B,2,4,256,1024]`.

Use this source-ownership record:

```python
@dataclass(frozen=True)
class BatonConditioningComponents:
    source: str
    teacher: FrozenSiglip2Teacher | None
    planner: FrozenDualCameraBatonPlanner | None
```

`prepare_baton_conditioning(config, dataset, device, dtype)` accepts only `qwen35_baton_teacher` and `qwen35_baton_prediction`, constructs exactly one frozen component, and rejects any hindsight cache or planner auxiliary-loss fields in Baton configs.

- [ ] **Step 2: Write failing recipe tests**

```python
def test_stage2_recipe_matches_approved_schedule() -> None:
    config = load_yaml(STAGE2_CONFIG)
    assert config["train_steps"] == 20_000
    assert config["steps_to_save"] == 5_000
    assert config["lr"] == 2e-5
    assert config["action_lr"] == 1e-4
    assert config["semantic_lr"] == 5e-5
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
    assert config["semantic_plan"]["source"] == "qwen35_baton_teacher"


def test_stage3_recipe_matches_approved_schedule() -> None:
    config = load_yaml(STAGE3_CONFIG)
    assert config["train_steps"] == 30_000
    assert config["steps_to_save"] == 5_000
    assert config["semantic_plan"]["source"] == "qwen35_baton_prediction"
    assert config["semantic_plan"]["tokens_per_frame"] == 256
```

- [ ] **Step 3: Run tests and confirm source modes are unknown**

Run:

```bash
pytest -q tests/test_ge_act_baton_training_contract.py
```

Expected: tests fail because the new semantic sources/configs do not exist.

- [ ] **Step 4: Implement Stage-2 conditioning**

For `source: qwen35_baton_teacher`, select future RGB indices `(0,3,5,8)`, run the frozen teacher online, construct semantic times with `build_semantic_plan_times`, construct exact patch-center positions, and pass no mask/relevance. The teacher is not registered in the optimizer or DeepSpeed model.

- [ ] **Step 5: Implement Stage-3 conditioning**

For `source: qwen35_baton_prediction`, extract the last current observation at `n_previous - 1`, call the frozen provider with captions, move detached predictions to the LTX dtype/device, validate provider future indices against config, construct full-clip semantic times with `build_semantic_plan_times`, and pass provider positions. Do not compute any planner auxiliary loss and do not register planner parameters in the optimizer.

- [ ] **Step 6: Keep GE-Act optimizer ownership exact**

For both stages, require exactly three nonempty, nonoverlapping groups:

```python
{
    "ltx_video": 2e-5,
    "action_expert": 1e-4,
    "semantic_adapter": 5e-5,
}
```

The union of these groups must equal every trainable GE-Act parameter. Qwen and SigLIP parameters must not appear in any group.

- [ ] **Step 7: Implement preflight and launchers**

Both configs use `LiberoFastWAMHDF5Dataset`, the same predecoded manifest contract, two views, four keyframes, `semantic_plan_in_dim: 1024`, all `28` semantic cross-attention blocks, dropout `0.15`, full video and action training, and global batch `128`. Stage 3 additionally validates the planner checkpoint before allocating LTX. Each launcher runs HDF5 and semantic preflight before `torch.distributed.run`.

Update validation/inference so Stage 2 reports teacher-conditioned video/action metrics, Stage 3 reports predicted-conditioned video/action metrics, and either config can run a `semantic_disabled` validation pass by setting the condition mask to zero without unloading weights. LIBERO and LIBERO-Plus evaluators must accept `qwen35_baton_prediction`; success accounting remains the existing official evaluator logic.

- [ ] **Step 8: Run training-contract tests**

Run:

```bash
pytest -q \
  tests/test_ge_act_baton_training_contract.py \
  tests/test_ge_act_baton_semantic_guidance.py \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_siglip2_config.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Stage 2 and Stage 3**

```bash
git add ge_act/runner ge_act/scripts/preflight_ltx_siglip2.py \
  ge_act/configs/ltx_model/libero/action_model_libero_baton_stage2_hdf5.yaml \
  ge_act/configs/ltx_model/libero/action_model_libero_baton_stage3_hdf5.yaml \
  ge_act/scripts/train_ltx_baton_stage2.sh \
  ge_act/scripts/train_ltx_baton_stage3.sh \
  ge_act/scripts/sbatch_train_ltx_baton_stage2_hpc3.sh \
  ge_act/scripts/sbatch_train_ltx_baton_stage3_hpc3.sh \
  tests/test_ge_act_baton_training_contract.py
git commit -m "feat(ge-act): add Baton teacher and predicted curricula"
```

---

### Task 11: Add End-to-End Smoke, Distributed Resume, and Legacy Gates

**Files:**
- Create: `qwen35_baton/cli/smoke_pipeline.py`
- Create: `qwen35_baton/scripts/smoke_two_rank.sh`
- Create: `tests/test_qwen35_baton_end_to_end.py`
- Modify: `tests/test_ge_act_source_completeness.py`
- Verify unchanged: all existing Qwen3-VL-2B, Qwen3.5 Plan-X, HDF5, and GE-Act tests.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: one command that validates Stage 1, Stage 2, Stage 3, two-rank gradients, and exact resume with tiny components.

- [ ] **Step 1: Write failing three-stage smoke tests**

```python
def test_tiny_pipeline_runs_all_three_stages(tmp_path: Path) -> None:
    result = run_tiny_pipeline(tmp_path)
    assert result.stage1.optimizer_steps == 1
    assert result.stage2.optimizer_steps == 1
    assert result.stage3.optimizer_steps == 1
    assert result.stage1.plan_shape == (1, 2, 4, 256, 1024)
    assert result.stage2.condition_source == "teacher"
    assert result.stage3.condition_source == "prediction"
```

Use tiny Qwen/SigLIP/LTX modules with production tensor contracts; do not download or claim validation of 4B weights in this unit test.

- [ ] **Step 2: Add a two-rank smoke entrypoint**

The shell command must be:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m qwen35_baton.cli.smoke_pipeline \
  --output-dir /tmp/qwen35_baton_two_rank \
  --verify-exact-resume
```

The command must assert identical parameter hashes across ranks after one optimizer step and identical final state between uninterrupted and interrupted/resumed two-step runs.

- [ ] **Step 3: Run the focused smoke tests**

Run:

```bash
pytest -q tests/test_qwen35_baton_end_to_end.py
bash qwen35_baton/scripts/smoke_two_rank.sh
```

Expected: both commands exit zero.

- [ ] **Step 4: Run source-boundary and legacy regression suites**

Run:

```bash
pytest -q \
  tests/test_qwen3vl2b_legacy_unchanged.py \
  tests/test_qwen35_grounded_config.py \
  tests/test_qwen35_sequence.py \
  tests/test_qwen35_grounded_planner.py \
  tests/test_qwen35_provider.py \
  tests/test_ge_act_source_completeness.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_qwen35_grounded.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_libero_fastwam_hdf5.py
```

- [ ] **Step 5: Run the complete repository test suite**

Run:

```bash
pytest -q
git diff --check
```

Expected: zero failed tests and no whitespace errors.

- [ ] **Step 6: Run bounded production preflight without claiming live-model success**

Run:

```bash
python -m qwen35_baton.cli.preflight \
  --config qwen35_baton/configs/libero_stage1.json \
  --world-size 8
python ge_act/scripts/preflight_ltx_siglip2.py \
  --config ge_act/configs/ltx_model/libero/action_model_libero_baton_stage2_hdf5.yaml \
  --world-size 8
python ge_act/scripts/preflight_ltx_siglip2.py \
  --config ge_act/configs/ltx_model/libero/action_model_libero_baton_stage3_hdf5.yaml \
  --world-size 8
```

Expected: all local artifacts and contracts validate. A live 4B forward or eight-GPU run is reported only after it actually executes.

- [ ] **Step 7: Commit the final verification gates**

```bash
git add qwen35_baton/cli/smoke_pipeline.py \
  qwen35_baton/scripts/smoke_two_rank.sh \
  tests/test_qwen35_baton_end_to_end.py \
  tests/test_ge_act_source_completeness.py
git commit -m "test: verify Baton planner curriculum end to end"
```

---

## Final Acceptance Checklist

- [ ] Stage-1 planner output is exactly `[B,2,4,256,1024]`.
- [ ] Every target patch has one query and exact `(t,y,x)` identity.
- [ ] Main and wrist rows remain isolated through Qwen, Query Tower, provider, and LTX attention.
- [ ] Query self-attention and query-to-Qwen attention are block-causal.
- [ ] Teacher features are online, penultimate-layer, detached, and never cached.
- [ ] MSE, cosine, delta, and same-suite instruction counterfactual objectives match the approved equations and weights.
- [ ] Only Query Tower, Sem-MLP, plan-token adapter, Qwen vision, and top-eight language layers train in Stage 1.
- [ ] Stage 2 uses online teacher features and Stage 3 uses frozen predicted features.
- [ ] GE-Act consumes all `1024` tokens per camera with exact patch-center coordinates and no relevance bias.
- [ ] Global batch, learning rates, step counts, and `5,000`-step save cadence match the approved design.
- [ ] Continuous checkpoints reject legacy TA-Tok metadata and validate all provenance fields.
- [ ] Exact resume passes after a mid-epoch, accumulated-gradient interruption.
- [ ] Existing Qwen3-VL-2B, `qwen35_planx`, HDF5, and GE-Act no-condition regressions pass.
- [ ] Live 4B/eight-GPU success is not claimed from tiny or static tests.
