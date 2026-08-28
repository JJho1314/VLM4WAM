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

