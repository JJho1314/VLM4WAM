# LIBERO `[TGT]` Text Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one versioned, fail-closed `libero_tgt_v1` instruction transform and use the same marked instruction in standalone planner training/inference and joint Qwen/T5 training/inference.

**Architecture:** A dependency-free module owns target extraction, marker validation, batch preprocessing, and the preprocessing-version constant. Planner builders and GE-Act consume that module instead of owning rewrite logic; checkpoint metadata selects the prompt contract so legacy checkpoints can still load only when a legacy configuration is used.

**Tech Stack:** Python 3.10+, regular expressions, PyTorch, Qwen3-VL processor chat templates, GE-Act/LTX, YAML/JSON metadata, pytest.

## Global Constraints

- Implement against the canonical HPC3 tree at `/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af`; preserve its unrelated dirty changes and never replace it with the older local `semantic-guidance` tree.
- The preprocessing contract name is exactly `libero_tgt_v1`; the literal marker is exactly `[TGT]`.
- Mark the first direct object of the first recognized manipulation verb; `turn on the stove and put the moka pot on it` must mark `stove`.
- Produce exactly one marker, be idempotent for an already singly marked instruction, and raise on empty input, multiple markers, or no recognized target.
- Do not mutate LIBERO metadata, write rewritten caption caches, add tokenizer vocabulary entries, or resize tokenizer embeddings for `[TGT]`.
- Do not add TAViD loss, target masks, target-token indices, InstructSAM features, model-geometry changes, optimizer changes, or loss-weight changes.
- Use the same marked list for the Qwen planner and GE-Act T5 text condition in each joint microbatch.
- New target-aware configurations must require matching checkpoint metadata; existing no-marker configurations and checkpoints remain available only through the legacy `None` contract.

---

## File Map

- Create `qwen3_vl_semantic_planner/libero_target_text.py`: pure `libero_tgt_v1` transformation and validation.
- Create `qwen3_vl_semantic_planner/audit_libero_target_text.py`: read-only JSONL audit CLI.
- Create `tests/test_libero_target_text.py`: unit and four-suite fixture coverage.
- Create `tests/fixtures/libero_task_texts.json`: the 40 canonical LIBERO task strings grouped by suite.
- Modify `qwen3_vl_semantic_planner/ge_act_dual_camera.py`: apply the selected preprocessing contract inside the shared dual-camera conversation builder.
- Modify `qwen3_vl_semantic_planner/train_semantic_planner.py`: CLI selection, collator wiring, export metadata, and metadata validation.
- Modify `ge_act/models/ltx_models/vlm_semantic_planner.py`: load the prompt contract from metadata and reject incompatible target-aware loads.
- Modify `ge_act/runner/ge_trainer.py`: preprocess once for planner, T5 cache prewarm, and T5 batch lookup; propagate metadata to joint exports.
- Modify `ge_act/experiments/eval_libero_joint.py`: build one marked rollout prompt and pass it to both semantic planning and base GE-Act inference.
- Modify `ge_act/experiments/joint_libero_eval_contract.py`: require target-aware metadata for the target-aware evaluator.
- Modify `tests/test_ge_act_dual_camera_planner.py`: conversation and standalone metadata tests.
- Modify `tests/test_ge_act_vlm_semantic_planner.py`: provider prompt-contract and compatibility tests.
- Modify `tests/test_joint_vlm_geact_training.py`: joint prompt sharing, cache, and legacy-checkpoint rejection tests.
- Modify `tests/test_joint_vlm_geact_libero_eval.py`: inference prompt sharing and joint-checkpoint metadata tests.
- Modify `qwen3_vl_semantic_planner/README.md`: exact audit and training flags.

---

### Task 1: Shared `libero_tgt_v1` Preprocessor

**Files:**
- Create: `qwen3_vl_semantic_planner/libero_target_text.py`
- Create: `tests/test_libero_target_text.py`

**Interfaces:**
- Produces: `TGT_MARKER: str`, `LIBERO_TGT_PREPROCESSING: str`, `InstructionPreprocessingError`, `mark_libero_target(instruction: str) -> str`, `validate_instruction_preprocessing(preprocessing: str | None) -> str | None`, and `preprocess_libero_instructions(instructions: Sequence[str], *, preprocessing: str | None) -> list[str]`.
- Consumes: no project model, dataset, PyTorch, or Transformers objects.

- [ ] **Step 1: Write failing transformation and validation tests**

```python
# tests/test_libero_target_text.py
from __future__ import annotations

import pytest

from qwen3_vl_semantic_planner.libero_target_text import (
    InstructionPreprocessingError,
    LIBERO_TGT_PREPROCESSING,
    mark_libero_target,
    preprocess_libero_instructions,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "pick up the black bowl between the plate and the ramekin",
            "pick up the [TGT] black bowl between the plate and the ramekin",
        ),
        (
            "open the middle drawer of the cabinet",
            "open the [TGT] middle drawer of the cabinet",
        ),
        (
            "turn on the stove and put the moka pot on it",
            "turn on the [TGT] stove and put the moka pot on it",
        ),
        ("put the bowl on the plate", "put the [TGT] bowl on the plate"),
        (
            "pick up the alphabet soup and place it in the basket",
            "pick up the [TGT] alphabet soup and place it in the basket",
        ),
    ],
)
def test_mark_libero_target_uses_first_direct_object(
    raw: str, expected: str
) -> None:
    assert mark_libero_target(raw) == expected


def test_mark_libero_target_is_idempotent() -> None:
    marked = "open the [TGT] top drawer and put the bowl inside"
    assert mark_libero_target(marked) == marked


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "non-empty"),
        ("wait for the robot", "no target object"),
        ("pick up the [TGT] bowl and place the [TGT] bowl", "exactly one"),
    ],
)
def test_mark_libero_target_rejects_invalid_text(
    value: str, message: str
) -> None:
    with pytest.raises(InstructionPreprocessingError, match=message):
        mark_libero_target(value)


def test_mark_libero_target_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        mark_libero_target(7)  # type: ignore[arg-type]


def test_batch_preprocessing_selects_target_or_legacy_contract() -> None:
    raw = ["put the bowl on the plate"]
    assert preprocess_libero_instructions(
        raw,
        preprocessing=LIBERO_TGT_PREPROCESSING,
    ) == ["put the [TGT] bowl on the plate"]
    assert preprocess_libero_instructions(
        raw,
        preprocessing=None,
    ) == raw
    with pytest.raises(ValueError, match="unsupported instruction preprocessing"):
        preprocess_libero_instructions(raw, preprocessing="libero_tgt_v2")
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run:

```bash
cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af
pytest -q tests/test_libero_target_text.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'qwen3_vl_semantic_planner.libero_target_text'`.

- [ ] **Step 3: Implement the dependency-free preprocessor**

```python
# qwen3_vl_semantic_planner/libero_target_text.py
"""Versioned, fail-closed target marking for LIBERO instructions."""

from __future__ import annotations

import re
from collections.abc import Sequence


TGT_MARKER = "[TGT]"
LIBERO_TGT_PREPROCESSING = "libero_tgt_v1"

_VERB_PATTERN = re.compile(
    r"\b("
    r"turn on|turn off|switch on|switch off|pick up|"
    r"put|place|open|close|push|pull|pick|grab|grasp|take|get|"
    r"move|bring|shift|lift|remove|slide|pour|hang|insert|stack|"
    r"cover|uncover|rotate|press|wipe|fold|unfold|transfer|"
    r"align|arrange|position|release|drop|scoop|roll|connect|"
    r"attach|detach|hold|carry|collect"
    r")\s+",
    flags=re.IGNORECASE,
)
_STOP_PATTERN = re.compile(
    r"(?="
    r"\s+(?:in|into|inside|on|onto|to|from|out of|off|over|under|"
    r"closer|away|toward|towards|near|next to|with|using|through|"
    r"back|around|behind|beside)\b"
    r"|\s*,"
    r"|\s+and\s+(?:then\s+)?(?:put|place|move|pick|pick up|take|"
    r"open|close|flip|turn|turn on|turn off|slide|hang|pour|"
    r"remove|push|pull)\b"
    r"|\s+then\s+"
    r"|[.]"
    r"|$"
    r")",
    flags=re.IGNORECASE,
)
_ARTICLE_PATTERN = re.compile(
    r"^(?P<article>(?:the|a|an|this|that|these|those|another|one)\s+)"
    r"(?P<rest>.+)$",
    flags=re.IGNORECASE,
)
_BAD_OBJECTS = frozenset({"it", "them", "this", "that", "there"})


class InstructionPreprocessingError(ValueError):
    """A LIBERO instruction cannot satisfy the selected prompt contract."""


def validate_instruction_preprocessing(
    preprocessing: str | None,
) -> str | None:
    if preprocessing not in (None, LIBERO_TGT_PREPROCESSING):
        raise ValueError(
            "unsupported instruction preprocessing "
            f"{preprocessing!r}; expected None or "
            f"{LIBERO_TGT_PREPROCESSING!r}"
        )
    return preprocessing


def _insert_marker(phrase: str) -> str:
    article = _ARTICLE_PATTERN.match(phrase)
    if article is None:
        return f"{TGT_MARKER} {phrase}"
    return (
        f"{article.group('article')}{TGT_MARKER} "
        f"{article.group('rest')}"
    )


def mark_libero_target(instruction: str) -> str:
    if not isinstance(instruction, str):
        raise TypeError(
            "LIBERO instruction must be a string, "
            f"got {type(instruction).__name__}"
        )
    text = instruction.strip()
    if not text:
        raise InstructionPreprocessingError(
            "LIBERO instruction must be non-empty"
        )
    marker_count = text.count(TGT_MARKER)
    if marker_count == 1:
        return text
    if marker_count > 1:
        raise InstructionPreprocessingError(
            "LIBERO instruction must contain exactly one [TGT] marker: "
            f"{instruction!r}"
        )

    for verb_match in _VERB_PATTERN.finditer(text):
        object_start = verb_match.end()
        stop_match = _STOP_PATTERN.search(text, object_start)
        object_end = stop_match.start() if stop_match is not None else len(text)
        phrase = text[object_start:object_end].strip(" ,.;")
        if not phrase or phrase.lower() in _BAD_OBJECTS:
            continue
        marked = text[:object_start] + _insert_marker(phrase) + text[object_end:]
        if marked.count(TGT_MARKER) != 1:
            raise InstructionPreprocessingError(
                "target rewrite did not produce exactly one [TGT] marker: "
                f"{instruction!r}"
            )
        return marked

    raise InstructionPreprocessingError(
        f"LIBERO instruction has no target object: {instruction!r}"
    )


def preprocess_libero_instructions(
    instructions: Sequence[str],
    *,
    preprocessing: str | None,
) -> list[str]:
    validate_instruction_preprocessing(preprocessing)
    if isinstance(instructions, (str, bytes)) or not isinstance(
        instructions, Sequence
    ):
        raise TypeError("instructions must be a sequence of strings")
    values = list(instructions)
    if preprocessing == LIBERO_TGT_PREPROCESSING:
        return [mark_libero_target(value) for value in values]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                "LIBERO instruction must be a string, "
                f"got {type(value).__name__}"
            )
        if not value.strip():
            raise InstructionPreprocessingError(
                "LIBERO instruction must be non-empty"
            )
        normalized.append(value)
    return normalized
```

- [ ] **Step 4: Run the preprocessor tests**

Run:

```bash
pytest -q tests/test_libero_target_text.py
```

Expected: `11 passed`, with no warnings from model libraries because the module has no model dependencies.

- [ ] **Step 5: Commit the shared contract**

```bash
git add qwen3_vl_semantic_planner/libero_target_text.py \
  tests/test_libero_target_text.py
git commit -m "feat: add LIBERO target text preprocessing"
```

---

### Task 2: Standalone Dual-Camera Planner Template and Export Metadata

**Files:**
- Modify: `qwen3_vl_semantic_planner/ge_act_dual_camera.py:1-175`
- Modify: `qwen3_vl_semantic_planner/train_semantic_planner.py:55-320,972-1085,3690-3745,4110-4135`
- Modify: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Consumes: `LIBERO_TGT_PREPROCESSING`, `preprocess_libero_instructions`, and `validate_instruction_preprocessing` from Task 1.
- Produces: `build_dual_camera_planner_inputs(..., *, instruction_preprocessing: str | None = LIBERO_TGT_PREPROCESSING)`, `DualCameraPlannerCollator.instruction_preprocessing`, CLI flag `--instruction-preprocessing`, and export field `instruction_preprocessing`.

- [ ] **Step 1: Add failing conversation and export-contract tests**

Add these tests to `tests/test_ge_act_dual_camera_planner.py` using the file's existing recording processor and metadata helpers:

```python
from qwen3_vl_semantic_planner.libero_target_text import (
    LIBERO_TGT_PREPROCESSING,
)


def test_dual_camera_builder_marks_instruction_in_shared_user_turn() -> None:
    processor = RecordingProcessor()
    build_dual_camera_planner_inputs(
        processor,
        [(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)))],
        ["turn on the stove and put the moka pot on it"],
        ["<|sem_plan_0|>"],
        instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
    )

    assert len(processor.rendered_conversations) == 1
    user_text = processor.rendered_conversations[0]
    assert (
        "Instruction: turn on the [TGT] stove and put the moka pot on it"
        in user_text
    )
    assert user_text.count("[TGT]") == 1


def test_dual_camera_builder_preserves_legacy_no_marker_contract() -> None:
    processor = RecordingProcessor()
    build_dual_camera_planner_inputs(
        processor,
        [(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)))],
        ["turn on the stove"],
        ["<|sem_plan_0|>"],
        instruction_preprocessing=None,
    )
    user_text = processor.rendered_conversations[0]
    assert "Instruction: turn on the stove" in user_text
    assert "[TGT]" not in user_text


def test_dual_camera_k4_metadata_records_target_text_contract() -> None:
    metadata = planner.build_dual_camera_export_metadata(
        future_keyframe_offsets=(2, 4, 6, 8),
        num_keyframes=4,
        target_tokens_per_keyframe=256,
        planner_token_count=384,
        instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
    )
    assert (
        metadata["instruction_preprocessing"]
        == LIBERO_TGT_PREPROCESSING
    )
    planner.validate_dual_camera_export_metadata(
        metadata,
        expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
    )


def test_dual_camera_metadata_rejects_missing_target_text_contract() -> None:
    with pytest.raises(ValueError, match="instruction_preprocessing"):
        planner.validate_dual_camera_export_metadata(
            valid_dual_camera_metadata(),
            expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
        )
```

- [ ] **Step 2: Run the focused failures**

Run:

```bash
pytest -q tests/test_ge_act_dual_camera_planner.py \
  -k "marks_instruction or legacy_no_marker or target_text_contract"
```

Expected: failures report the missing `instruction_preprocessing` keyword or missing metadata field.

- [ ] **Step 3: Wire preprocessing into the shared builder and collator**

Apply these exact hunks in
`qwen3_vl_semantic_planner/ge_act_dual_camera.py`:

```diff
+try:
+    from .libero_target_text import (
+        LIBERO_TGT_PREPROCESSING,
+        preprocess_libero_instructions,
+    )
+except ImportError:
+    from libero_target_text import (  # type: ignore[no-redef]
+        LIBERO_TGT_PREPROCESSING,
+        preprocess_libero_instructions,
+    )
@@
 def build_dual_camera_planner_inputs(
     processor: Any,
     image_pairs: list[tuple[Any, Any]],
     instructions: list[Any],
     plan_sequence: str | list[str],
+    *,
+    instruction_preprocessing: str | None = LIBERO_TGT_PREPROCESSING,
 ) -> Any:
@@
+    marked_instructions = preprocess_libero_instructions(
+        instructions,
+        preprocessing=instruction_preprocessing,
+    )
     plan_text = plan_sequence if isinstance(plan_sequence, str) else "".join(plan_sequence)
@@
-    for image_pair, instruction in zip(image_pairs, instructions, strict=True):
+    for image_pair, instruction in zip(
+        image_pairs, marked_instructions, strict=True
+    ):
@@
 @dataclass
 class DualCameraPlannerCollator:
     processor: Any
     plan_sequence: list[str]
+    instruction_preprocessing: str | None = LIBERO_TGT_PREPROCESSING
@@
         inputs = build_dual_camera_planner_inputs(
             self.processor,
             [item["images"] for item in batch],
             [item["prompt"] for item in batch],
             self.plan_sequence,
+            instruction_preprocessing=self.instruction_preprocessing,
         )
```

No dataset rewrite is added: `GEActDualCameraPlannerDataset.__getitem__`
continues returning raw `str(sample["caption"])`.

- [ ] **Step 4: Add CLI selection and metadata propagation**

In `train_semantic_planner.py`, import the constant and validator, add:

```python
parser.add_argument(
    "--instruction-preprocessing",
    choices=[LIBERO_TGT_PREPROCESSING],
    default=None,
    help=(
        "versioned instruction contract; use libero_tgt_v1 for new "
        "target-aware LIBERO planner training"
    ),
)
```

Apply these exact metadata hunks:

```diff
 def build_dual_camera_export_metadata(
     *,
     future_keyframe_offsets: Sequence[int],
     num_keyframes: int,
     target_tokens_per_keyframe: int,
     planner_token_count: int,
+    instruction_preprocessing: str | None = None,
 ) -> dict[str, Any]:
+    """Build the camera/export contract from the trained planner geometry."""
+    validate_instruction_preprocessing(instruction_preprocessing)
-    """Build the camera/export contract from the trained planner geometry."""
@@
     if keyframes > 1:
         metadata.update(
             {
                 "num_keyframes": keyframes,
                 "target_tokens_per_keyframe": tokens_per_keyframe,
                 "planner_token_count": plan_tokens,
             }
         )
+    if instruction_preprocessing is not None:
+        metadata["instruction_preprocessing"] = instruction_preprocessing
     return metadata
@@
 def validate_dual_camera_export_metadata(
     metadata: dict[str, Any],
+    *,
+    expected_instruction_preprocessing: str | None = None,
 ) -> dict[str, Any]:
+    """Reject exports that cannot preserve separate main/wrist view context."""
+    validate_instruction_preprocessing(expected_instruction_preprocessing)
-    """Reject exports that cannot preserve separate main/wrist view context."""
@@
     if legacy_input_frame not in (None, "separate_camera_images"):
         raise ValueError(
             "incompatible dual-camera metadata field planner_input_frame: "
             "expected 'separate_camera_images', "
             f"got {legacy_input_frame!r}"
         )
+    actual_preprocessing = metadata.get("instruction_preprocessing")
+    validate_instruction_preprocessing(actual_preprocessing)
+    if (
+        expected_instruction_preprocessing is not None
+        and actual_preprocessing != expected_instruction_preprocessing
+    ):
+        raise ValueError(
+            "incompatible dual-camera metadata field "
+            "instruction_preprocessing: expected "
+            f"{expected_instruction_preprocessing!r}, "
+            f"got {actual_preprocessing!r}"
+        )
     return metadata
```

Pass the CLI value to the collator and export:

```python
collator = DualCameraPlannerCollator(
    processor=processor,
    plan_sequence=plan_sequence,
    instruction_preprocessing=args.instruction_preprocessing,
)
```

```python
build_dual_camera_export_metadata(
    future_keyframe_offsets=offsets,
    num_keyframes=module.num_keyframes,
    target_tokens_per_keyframe=module.target_len // module.num_keyframes,
    planner_token_count=module.latent_len,
    instruction_preprocessing=getattr(
        args, "instruction_preprocessing", None
    ),
)
```

Replace the save-time validation call with:

```python
validate_dual_camera_export_metadata(
    meta,
    expected_instruction_preprocessing=getattr(
        args, "instruction_preprocessing", None
    ),
)
```

- [ ] **Step 5: Run standalone planner contract tests**

Run:

```bash
pytest -q tests/test_ge_act_dual_camera_planner.py \
  -k "dual_camera and (builder or metadata or checkpoint)"
```

Expected: all selected tests pass; legacy metadata remains accepted when no expected preprocessing is supplied.

- [ ] **Step 6: Commit standalone template integration**

```bash
git add qwen3_vl_semantic_planner/ge_act_dual_camera.py \
  qwen3_vl_semantic_planner/train_semantic_planner.py \
  tests/test_ge_act_dual_camera_planner.py
git commit -m "feat: use target-aware dual-camera planner prompts"
```

---

### Task 3: Metadata-Selected Standalone Planner Inference

**Files:**
- Modify: `ge_act/models/ltx_models/vlm_semantic_planner.py:1-390`
- Modify: `tests/test_ge_act_vlm_semantic_planner.py`

**Interfaces:**
- Consumes: `instruction_preprocessing` in `planner_meta.json` and Task 1 batch preprocessor.
- Produces: `FrozenDualCameraVLMPlanner.from_checkpoint(..., expected_instruction_preprocessing: str | None = None)` and `FrozenDualCameraVLMPlanner.instruction_preprocessing`.

- [ ] **Step 1: Add failing target-aware and legacy loader tests**

In `tests/test_ge_act_vlm_semantic_planner.py`, import
`LIBERO_TGT_PREPROCESSING` and add:

```python
def test_frozen_provider_passes_metadata_selected_prompt_contract() -> None:
    metadata = valid_k4_metadata()
    metadata["instruction_preprocessing"] = LIBERO_TGT_PREPROCESSING
    received: dict[str, str | None] = {}

    def recording_builder(
        processor: Any,
        image_pairs: list[tuple[Image.Image, Image.Image]],
        instructions: list[str],
        plan_tokens: list[str],
        *,
        instruction_preprocessing: str | None,
    ) -> dict[str, Any]:
        received["instruction_preprocessing"] = instruction_preprocessing
        return flexible_fake_input_builder(
            processor,
            image_pairs,
            instructions,
            plan_tokens,
            instruction_preprocessing=instruction_preprocessing,
        )

    provider = FrozenDualCameraVLMPlanner.from_components(
        wrapper=FakeK4DualWrapper(),
        processor=FakeProcessor(),
        input_builder=recording_builder,
        input_mover=lambda value: value,
        device="cpu",
        plan_tokens=metadata["plan_token_strings"],
        metadata=metadata,
        expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
    )
    provider.prepare_inputs(
        torch.zeros(1, 2, 3, 8, 8),
        ["open the middle drawer of the cabinet"],
    )
    assert received == {
        "instruction_preprocessing": LIBERO_TGT_PREPROCESSING
    }


def test_target_aware_provider_rejects_legacy_checkpoint_metadata() -> None:
    metadata = valid_k4_metadata()
    with pytest.raises(ValueError, match="instruction_preprocessing"):
        validate_dual_camera_planner_metadata(
            metadata,
            expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
        )


def test_legacy_provider_keeps_unmarked_prompt() -> None:
    metadata = valid_k4_metadata()
    assert "instruction_preprocessing" not in metadata
    assert (
        validate_dual_camera_planner_metadata(metadata)
        is metadata
    )
```

- [ ] **Step 2: Run focused tests and confirm signature/contract failures**

Run:

```bash
pytest -q tests/test_ge_act_vlm_semantic_planner.py \
  -k "metadata_selected_target or target_aware_provider or legacy_provider"
```

Expected: failures mention missing `expected_instruction_preprocessing` support.

- [ ] **Step 3: Store and apply the checkpoint-selected contract**

In `vlm_semantic_planner.py`, import the validator and apply the exact
signature/return-site hunks:

```diff
+from qwen3_vl_semantic_planner.libero_target_text import (
+    validate_instruction_preprocessing,
+)
@@
 def validate_dual_camera_planner_metadata(
     metadata: dict[str, Any],
+    *,
+    expected_instruction_preprocessing: str | None = None,
 ) -> dict[str, Any]:
     """Validate the exact independently trained main/wrist export geometry."""
+    validate_instruction_preprocessing(expected_instruction_preprocessing)
@@
     if metadata.get("plan_token_strings") != expected_tokens:
         raise ValueError(
             "incompatible dual-camera planner metadata field plan_token_strings: "
             f"expected {planner_token_count} ordered planner tokens"
         )
+    actual_preprocessing = metadata.get("instruction_preprocessing")
+    validate_instruction_preprocessing(actual_preprocessing)
+    if (
+        expected_instruction_preprocessing is not None
+        and actual_preprocessing != expected_instruction_preprocessing
+    ):
+        raise ValueError(
+            "incompatible dual-camera planner metadata field "
+            "instruction_preprocessing: expected "
+            f"{expected_instruction_preprocessing!r}, "
+            f"got {actual_preprocessing!r}"
+        )
     return metadata
```

Add `instruction_preprocessing: str | None` to `__init__`, derive it from metadata in `from_components`, and pass it into the builder:

```python
return self.input_mover(
    self.input_builder(
        self.processor,
        image_pairs,
        [str(value) for value in instructions],
        self.plan_tokens,
        instruction_preprocessing=self.instruction_preprocessing,
    )
)
```

Add the expected contract to both constructors:

```python
@classmethod
def from_components(
    cls,
    *,
    wrapper: nn.Module,
    processor: Any,
    input_builder: Callable[..., dict[str, Any]],
    input_mover: Callable[[dict[str, Any]], dict[str, Any]],
    device: torch.device | str,
    plan_tokens: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    expected_instruction_preprocessing: str | None = None,
) -> "FrozenDualCameraVLMPlanner":
```

```python
@classmethod
def from_checkpoint(
    cls,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    expected_instruction_preprocessing: str | None = None,
) -> "FrozenDualCameraVLMPlanner":
```

Validate with the expected value before loading Qwen weights. A missing field must therefore fail before model construction for target-aware configs.

Update `fake_input_builder` and `flexible_fake_input_builder` in
`tests/test_ge_act_vlm_semantic_planner.py` with this keyword-only parameter,
and assert that only the two supported contracts reach a provider builder:

```python
    *,
    instruction_preprocessing: str | None = None,
) -> dict[str, Any]:
    assert instruction_preprocessing in (
        None,
        LIBERO_TGT_PREPROCESSING,
    )
```

- [ ] **Step 4: Run provider and full dual-camera tests**

Run:

```bash
pytest -q tests/test_ge_act_vlm_semantic_planner.py
```

Expected: all tests pass; fake input builders are updated to accept the new keyword-only argument.

- [ ] **Step 5: Commit inference compatibility handling**

```bash
git add ge_act/models/ltx_models/vlm_semantic_planner.py \
  tests/test_ge_act_vlm_semantic_planner.py
git commit -m "feat: validate planner prompt preprocessing metadata"
```

---

### Task 4: One Marked Caption List for Joint Qwen and T5 Training

**Files:**
- Modify: `ge_act/runner/ge_trainer.py:700-900,1054-1125,1390-1475,1500-1780,2080-2300`
- Modify: `tests/test_joint_vlm_geact_training.py`

**Interfaces:**
- Consumes: `semantic_plan.instruction_preprocessing` and `preprocess_libero_instructions`.
- Produces: `prepare_joint_instruction_prompts(captions: Sequence[str], *, preprocessing: str | None) -> list[str]`.

- [ ] **Step 1: Add failing prompt-sharing, cache, and metadata tests**

```python
def test_joint_instruction_prompts_are_target_aware_and_idempotent() -> None:
    symbols = _load_ge_trainer_symbols(
        "prepare_joint_instruction_prompts"
    )
    marked = symbols.prepare_joint_instruction_prompts(
        [
            "put the bowl on the plate",
            "open the [TGT] top drawer and put the bowl inside",
        ],
        preprocessing="libero_tgt_v1",
    )
    assert marked == [
        "put the [TGT] bowl on the plate",
        "open the [TGT] top drawer and put the bowl inside",
    ]


def test_target_aware_text_cache_is_keyed_by_marked_instruction() -> None:
    symbols = _load_ge_trainer_symbols(
        "prepare_joint_instruction_prompts",
        "get_cached_text_conditions",
    )
    cache: dict[str, dict[str, torch.Tensor]] = {}
    encoded_prompts: list[str] = []

    def encode_missing(prompts: list[str]) -> dict[str, torch.Tensor]:
        encoded_prompts.extend(prompts)
        return {
            "prompt_embeds": torch.zeros(len(prompts), 2, 3),
            "prompt_attention_mask": torch.ones(len(prompts), 2),
        }

    prompts = symbols.prepare_joint_instruction_prompts(
        ["turn on the stove"],
        preprocessing="libero_tgt_v1",
    )
    symbols.get_cached_text_conditions(
        prompts,
        cache=cache,
        encode_missing=encode_missing,
    )
    assert encoded_prompts == ["turn on the [TGT] stove"]
    assert list(cache) == ["turn on the [TGT] stove"]


def test_joint_loop_shares_one_marked_caption_variable() -> None:
    source = GE_TRAINER_PATH.read_text(encoding="utf-8")
    loop = source[source.index("for step, batch in enumerate(") :]
    assert "prepare_joint_instruction_prompts(" in loop
    assert "batch[\"caption\"]" in loop
    assert (
        "prepare_joint_planner_current_images(current_frames),\\n"
        "                            captions,"
    ) in loop
    assert (
        "text_conds = get_cached_text_conditions(\\n"
        "                        captions,"
    ) in loop
    assert "captions = batch['caption']" not in loop

```

In the existing
`test_joint_checkpoint_exports_both_models_metadata_and_training_state`, add
the contract to its concrete source JSON and config:

```diff
             {
                 "future_keyframe_offsets": [2, 4, 6, 8],
                 "num_keyframes": 4,
                 "target_tokens_per_keyframe": 256,
+                "instruction_preprocessing": "libero_tgt_v1",
             }
@@
-        semantic_plan={"planner_checkpoint": str(source_planner)},
+        semantic_plan={
+            "planner_checkpoint": str(source_planner),
+            "instruction_preprocessing": "libero_tgt_v1",
+        },
```

Add these assertions after reading `joint_meta`:

```python
assert joint_meta["instruction_preprocessing"] == "libero_tgt_v1"
exported_planner_meta = json.loads(
    (step_dir / "planner" / "planner_meta.json").read_text(
        encoding="utf-8"
    )
)
assert (
    exported_planner_meta["instruction_preprocessing"]
    == "libero_tgt_v1"
)
```

Because `_load_ge_trainer_symbols` executes selected AST nodes outside the
module import context, add the Task 1 function and `Sequence` to its explicit
test namespace:

```diff
+from collections.abc import Sequence
+from qwen3_vl_semantic_planner.libero_target_text import (
+    preprocess_libero_instructions,
+)
@@
     namespace = {
@@
+        "Sequence": Sequence,
+        "preprocess_libero_instructions": preprocess_libero_instructions,
     }
```

- [ ] **Step 2: Run focused joint-training failures**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py \
  -k "instruction_prompts or target_aware_text_cache or instruction_preprocessing"
```

Expected: the prompt helper is missing and joint metadata lacks the new field.

- [ ] **Step 3: Add one joint preprocessing boundary**

At module scope in `ge_trainer.py`:

```python
from qwen3_vl_semantic_planner.libero_target_text import (
    preprocess_libero_instructions,
)


def prepare_joint_instruction_prompts(
    captions: Sequence[str],
    *,
    preprocessing: str | None,
) -> list[str]:
    """Return the one prompt list shared by planner and frozen T5."""
    return preprocess_libero_instructions(
        captions,
        preprocessing=preprocessing,
    )
```

When loading the planner:

```python
self.semantic_planner = FrozenDualCameraVLMPlanner.from_checkpoint(
    semantic_config["planner_checkpoint"],
    device=device,
    dtype=dtype,
    expected_instruction_preprocessing=semantic_config.get(
        "instruction_preprocessing"
    ),
)
```

Before cache prewarm:

```python
captions = prepare_joint_instruction_prompts(
    tuple(caption_provider()),
    preprocessing=getattr(
        self.args, "semantic_plan", {}
    ).get("instruction_preprocessing"),
)
```

At the start of every training microbatch, before either consumer:

```python
captions = (
    prepare_joint_instruction_prompts(
        batch["caption"],
        preprocessing=self.args.semantic_plan.get(
            "instruction_preprocessing"
        ),
    )
    if joint_enabled
    else batch["caption"]
)
```

Use this exact `captions` variable in:

```python
planner_inputs = self.semantic_planner.prepare_inputs(
    prepare_joint_planner_current_images(current_frames),
    captions,
)
```

and:

```python
text_conds = get_cached_text_conditions(
    captions,
    cache=self.text_condition_cache,
    encode_missing=self._encode_missing_text_conditions,
)
```

Delete the later `captions = batch['caption']` assignment so it cannot silently restore raw text.

- [ ] **Step 4: Propagate the contract through joint checkpoint metadata**

After loading `source_planner_metadata`, require the configured value and add:

```python
instruction_preprocessing = args.semantic_plan.get(
    "instruction_preprocessing"
)
if (
    source_planner_metadata.get("instruction_preprocessing")
    != instruction_preprocessing
):
    raise ValueError(
        "joint source planner instruction preprocessing mismatch: "
        f"expected {instruction_preprocessing!r}, got "
        f"{source_planner_metadata.get('instruction_preprocessing')!r}"
    )
```

Add this exact field to `joint_meta.json`:

```python
"instruction_preprocessing": instruction_preprocessing,
```

The standalone planner export already copies the source metadata and therefore retains the same field.

- [ ] **Step 5: Run joint-training tests**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_training.py
```

Expected: all tests pass, including existing cache-offload and checkpoint tests.

- [ ] **Step 6: Commit joint training prompt sharing**

```bash
git add ge_act/runner/ge_trainer.py \
  tests/test_joint_vlm_geact_training.py
git commit -m "feat: share target-aware prompts in joint training"
```

---

### Task 5: Target-Aware Joint LIBERO Inference

**Files:**
- Modify: `ge_act/experiments/eval_libero_joint.py:1-120`
- Modify: `ge_act/experiments/joint_libero_eval_contract.py:1-150`
- Modify: `tests/test_joint_vlm_geact_libero_eval.py`

**Interfaces:**
- Consumes: `LIBERO_TGT_PREPROCESSING`, target-aware planner metadata, and target-aware joint metadata.
- Produces: `joint_libero_eval_contract.prepare_joint_inference_prompt(prompt: str) -> str`; both planner inference and base GE-Act/T5 inference receive its exact return value.

- [ ] **Step 1: Add failing inference prompt-sharing and metadata tests**

```python
def test_joint_inference_prompt_marks_first_target() -> None:
    assert prepare_joint_inference_prompt(
        "turn on the stove and put the moka pot on it"
    ) == "turn on the [TGT] stove and put the moka pot on it"


def test_semantic_condition_receives_marked_prompt() -> None:
    planner = RecordingPlanner()
    current = torch.zeros(2, 3, 8, 8)
    build_joint_semantic_condition(
        planner,
        current,
        prepare_joint_inference_prompt("pick up the black bowl"),
        device="cpu",
        dtype=torch.bfloat16,
    )
    assert planner.instructions == ["pick up the [TGT] black bowl"]


def test_joint_checkpoint_rejects_legacy_prompt_contract(
    tmp_path: Path,
) -> None:
    root = write_joint_export(tmp_path)
    joint_path = root / "joint_meta.json"
    joint = json.loads(joint_path.read_text())
    joint.pop("instruction_preprocessing", None)
    joint_path.write_text(json.dumps(joint))
    with pytest.raises(ValueError, match="instruction_preprocessing"):
        validate_joint_evaluation_checkpoint(root)
```

Extend the existing source-contract test with exact checks that the one
`marked_prompt` variable feeds both consumers:

```python
assert "marked_prompt = prepare_joint_inference_prompt(prompt)" in source
assert "current_images,\\n                marked_prompt," in source
assert "current_images,\\n                marked_prompt,\\n                excution_step=" in source
```

Import `prepare_joint_inference_prompt` from
`experiments.joint_libero_eval_contract` at the top of the test file.

- [ ] **Step 2: Run focused inference failures**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_libero_eval.py \
  -k "prompt or legacy_prompt_contract"
```

Expected: raw prompt assertions fail and the legacy checkpoint is still accepted.

- [ ] **Step 3: Mark once in the rollout evaluator**

Put the pure prompt helper in the already importable
`joint_libero_eval_contract.py`:

```python
from qwen3_vl_semantic_planner.libero_target_text import (
    LIBERO_TGT_PREPROCESSING,
    preprocess_libero_instructions,
)


def prepare_joint_inference_prompt(prompt: str) -> str:
    return preprocess_libero_instructions(
        [prompt],
        preprocessing=LIBERO_TGT_PREPROCESSING,
    )[0]
```

Import `prepare_joint_inference_prompt` into `eval_libero_joint.py`.

Load with the required contract:

```python
self.semantic_planner = FrozenDualCameraVLMPlanner.from_checkpoint(
    self.joint_checkpoint.planner_dir,
    device=self.device,
    dtype=self.weight_dtype,
    expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
)
```

At the top of `play`:

```python
marked_prompt = prepare_joint_inference_prompt(prompt)
```

Replace the two existing raw `prompt` arguments with `marked_prompt`:

```diff
         semantic_plan, semantic_plan_times, semantic_condition_mask = (
             build_joint_semantic_condition(
                 self.semantic_planner,
                 current_images,
-                prompt,
+                marked_prompt,
                 device=self.device,
                 dtype=self.weight_dtype,
             )
@@
             return super().play(
                 current_images,
-                prompt,
+                marked_prompt,
                 excution_step=excution_step,
                 state=state,
             )
```

- [ ] **Step 4: Require matching joint and planner metadata**

In `joint_libero_eval_contract.py`, add:

```python
from qwen3_vl_semantic_planner.libero_target_text import (
    LIBERO_TGT_PREPROCESSING,
)
```

Require it in `joint_meta.json`:

```python
"instruction_preprocessing": LIBERO_TGT_PREPROCESSING,
```

and validate planner metadata with:

```python
validate_dual_camera_planner_metadata(
    planner_meta,
    expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
)
```

Update the concrete `write_joint_export` helper so its default fixture remains
valid under the new evaluator:

```diff
             {
                 "global_step": global_step,
                 "num_camera_views": 2,
                 "num_keyframes": 4,
                 "tokens_per_keyframe": 256,
                 "future_keyframe_offsets": [2, 4, 6, 8],
+                "instruction_preprocessing": "libero_tgt_v1",
             }
@@
 def valid_k4_planner_metadata() -> dict[str, object]:
     return {
@@
+        "instruction_preprocessing": "libero_tgt_v1",
     }
```

- [ ] **Step 5: Run joint inference tests**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_libero_eval.py
```

Expected: all tests pass, with the recording planner and base evaluator seeing the identical marked prompt.

- [ ] **Step 6: Commit joint inference integration**

```bash
git add ge_act/experiments/eval_libero_joint.py \
  ge_act/experiments/joint_libero_eval_contract.py \
  tests/test_joint_vlm_geact_libero_eval.py
git commit -m "feat: use target-aware prompts in joint inference"
```

---

### Task 6: Four-Suite Caption Audit and Training Preflight

**Files:**
- Create: `qwen3_vl_semantic_planner/audit_libero_target_text.py`
- Create: `tests/fixtures/libero_task_texts.json`
- Modify: `tests/test_libero_target_text.py`
- Modify: `qwen3_vl_semantic_planner/README.md`

**Interfaces:**
- Consumes: JSONL rows with a non-empty `task` string and `mark_libero_target`.
- Produces: `audit_task_files(paths: Sequence[Path]) -> dict[str, object]` and a read-only CLI that exits nonzero on the first incompatible instruction.

- [ ] **Step 1: Add the canonical four-suite fixture**

Create `tests/fixtures/libero_task_texts.json` with four keys and the exact 10 strings currently present in each HPC3 `meta/tasks.jsonl`:

```json
{
  "libero_10": [
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the yellow and white mug in the microwave and close it",
    "put both moka pots on the stove",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "pick up the book and place it in the back compartment of the caddy"
  ],
  "libero_goal": [
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
    "push the plate to the front of the stove",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the cream cheese in the bowl",
    "put the wine bottle on the rack",
    "put the wine bottle on top of the cabinet",
    "turn on the stove"
  ],
  "libero_object": [
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket"
  ],
  "libero_spatial": [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate"
  ]
}
```

- [ ] **Step 2: Add failing fixture and CLI tests**

```python
import json
from pathlib import Path


def test_all_four_libero_suites_produce_exactly_one_marker() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "libero_task_texts.json"
    )
    suites = json.loads(fixture.read_text())
    assert {name: len(tasks) for name, tasks in suites.items()} == {
        "libero_10": 10,
        "libero_goal": 10,
        "libero_object": 10,
        "libero_spatial": 10,
    }
    marked = [
        mark_libero_target(task)
        for tasks in suites.values()
        for task in tasks
    ]
    assert len(marked) == 40
    assert all(value.count("[TGT]") == 1 for value in marked)


def test_audit_task_files_reports_each_suite(tmp_path: Path) -> None:
    from qwen3_vl_semantic_planner.audit_libero_target_text import (
        audit_task_files,
    )

    paths = []
    for suite in ("libero_10", "libero_goal"):
        path = tmp_path / suite / "meta" / "tasks.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"task_index": 0, "task": "put the bowl on the plate"})
            + "\n"
        )
        paths.append(path)
    report = audit_task_files(paths)
    assert report["total_tasks"] == 2
    assert report["total_marked"] == 2
    assert len(report["files"]) == 2
```

- [ ] **Step 3: Run the focused failures**

Run:

```bash
pytest -q tests/test_libero_target_text.py
```

Expected: the fixture test passes once the fixture exists; the CLI import fails until the audit module is created.

- [ ] **Step 4: Implement the read-only audit CLI**

```python
# qwen3_vl_semantic_planner/audit_libero_target_text.py
"""Audit LIBERO task JSONL files for the libero_tgt_v1 contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from qwen3_vl_semantic_planner.libero_target_text import (
    LIBERO_TGT_PREPROCESSING,
    mark_libero_target,
)


def audit_task_files(paths: Sequence[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one tasks.jsonl path is required")
    files: list[dict[str, object]] = []
    total = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing LIBERO task metadata: {path}")
        count = 0
        examples: list[dict[str, str]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task")
            if not isinstance(task, str):
                raise ValueError(
                    f"{path}:{line_number} needs a string task field"
                )
            marked = mark_libero_target(task)
            count += 1
            if len(examples) < 3:
                examples.append({"raw": task, "marked": marked})
        if count == 0:
            raise ValueError(f"no tasks found in {path}")
        files.append(
            {
                "path": str(path),
                "tasks": count,
                "marked": count,
                "examples": examples,
            }
        )
        total += count
    return {
        "instruction_preprocessing": LIBERO_TGT_PREPROCESSING,
        "total_tasks": total,
        "total_marked": total,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_files", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_task_files(args.task_files),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the unit test and real four-suite preflight**

Run:

```bash
pytest -q tests/test_libero_target_text.py
python -m qwen3_vl_semantic_planner.audit_libero_target_text \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_goal_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_object_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_spatial_no_noops_lerobot/meta/tasks.jsonl
```

Expected: pytest passes; the JSON report contains `"total_tasks": 40`, `"total_marked": 40`, and `"instruction_preprocessing": "libero_tgt_v1"`.

- [ ] **Step 6: Document the mandatory preflight and planner flag**

Add to `qwen3_vl_semantic_planner/README.md`:

````markdown
### LIBERO target-aware text

New dual-camera LIBERO planner runs must pass:

```bash
--instruction-preprocessing libero_tgt_v1
```

`[TGT]` is ordinary prompt text; do not add it as a tokenizer token. Before a
long run, audit all four `meta/tasks.jsonl` files with
`python -m qwen3_vl_semantic_planner.audit_libero_target_text ...`. A new
target-aware GE-Act config must also set
`semantic_plan.instruction_preprocessing: libero_tgt_v1`; this intentionally
rejects the old unmarked planner checkpoint.
````

- [ ] **Step 7: Commit audit coverage**

```bash
git add qwen3_vl_semantic_planner/audit_libero_target_text.py \
  qwen3_vl_semantic_planner/README.md \
  tests/fixtures/libero_task_texts.json \
  tests/test_libero_target_text.py
git commit -m "test: audit LIBERO target text coverage"
```

---

### Task 7: Full Contract Verification

**Files:**
- Verify only; do not create generated artifacts in the repository.

**Interfaces:**
- Consumes: all interfaces from Tasks 1-6.
- Produces: evidence that target-aware and legacy contracts behave as designed before planner retraining starts.

- [ ] **Step 1: Run formatting and placeholder checks**

Run:

```bash
git diff --check HEAD~6..HEAD
grep -R -nE "Instruction: \\{instruction\\}" \
  qwen3_vl_semantic_planner ge_act | sort
```

Expected: `git diff --check` prints nothing. Every production Qwen planner template occurrence routes through the shared preprocessing contract; no joint T5 use restores `batch["caption"]` after `captions` is marked.

- [ ] **Step 2: Run the five focused test files together**

Run:

```bash
pytest -q \
  tests/test_libero_target_text.py \
  tests/test_ge_act_dual_camera_planner.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_joint_vlm_geact_training.py \
  tests/test_joint_vlm_geact_libero_eval.py
```

Expected: all tests pass.

- [ ] **Step 3: Run the broader GE-Act semantic contract tests**

Run:

```bash
pytest -q \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_siglip2_config.py
```

Expected: all tests pass; no geometry, loss, or semantic-injection regressions.

- [ ] **Step 4: Prove legacy and target-aware checkpoint behavior without loading model weights**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

from ge_act.models.ltx_models.vlm_semantic_planner import (
    validate_dual_camera_planner_metadata,
)

checkpoint = Path(
    "/data/user/jhe724/junjie/vlm4wam_joint_assets/"
    "planner_step_030000"
)
metadata = json.loads(
    (checkpoint / "planner_meta.json").read_text(encoding="utf-8")
)
validate_dual_camera_planner_metadata(metadata)
try:
    validate_dual_camera_planner_metadata(
        metadata,
        expected_instruction_preprocessing="libero_tgt_v1",
    )
except ValueError as error:
    assert "instruction_preprocessing" in str(error)
    print("legacy accepted by legacy contract and rejected by target-aware contract")
else:
    raise AssertionError("legacy planner unexpectedly passed target-aware validation")
PY
```

Expected: `legacy accepted by legacy contract and rejected by target-aware contract`.

- [ ] **Step 5: Re-run the real dataset audit immediately before launch**

Run the four-path audit command from Task 6.

Expected: 40/40 tasks are marked successfully. Save the terminal JSON in the external training log, not in the Git repository.

- [ ] **Step 6: Review the final diff**

Run:

```bash
git status --short
git diff --stat HEAD~6..HEAD
git log --oneline -6
```

Expected: only the files in this plan are changed by these commits; pre-existing unrelated HPC3 changes remain untouched.

After this verification, retrain the dual-camera planner with
`--instruction-preprocessing libero_tgt_v1`. Do not start the frozen-planner
GE-Act-only run until that planner export contains
`"instruction_preprocessing": "libero_tgt_v1"`.
