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

