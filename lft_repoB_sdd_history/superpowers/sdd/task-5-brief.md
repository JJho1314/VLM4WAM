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

