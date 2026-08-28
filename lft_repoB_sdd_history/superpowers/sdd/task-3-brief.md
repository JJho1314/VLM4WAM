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

