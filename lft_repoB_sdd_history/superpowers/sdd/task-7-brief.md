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
