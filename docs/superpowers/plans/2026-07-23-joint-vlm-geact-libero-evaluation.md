# Joint VLM–GE-Act LIBERO Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the step-40,000 joint dual-camera K4 VLM–GE-Act checkpoint on local A6000 GPU 1 without falling back to semantic-free LTX inference.

**Architecture:** Add a small, dependency-light checkpoint and semantic-conditioning contract module, then add a dedicated evaluator that normalizes the current two-view input once and wraps the existing pipeline with a fail-closed conditioning proxy. The original evaluator remains untouched. A local configuration and launcher run a one-episode smoke gate before the four standard 50-trial-per-task suites.

**Tech Stack:** Python 3.10, PyTorch 2.6, Transformers, Diffusers, Qwen3-VL, LTX-Video, LIBERO/robosuite, pytest, Bash, rsync.

## Global Constraints

- Evaluate exactly `/data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k4_50k/2026_07_21_14_48_26/step_40000`.
- Transfer only `joint_meta.json`, `ltx/`, and `planner/`; do not copy `training_state/`.
- Camera order is always `[main, wrist]`.
- Semantic tokens must be finite `[B, 2, 4, 256, 1024]`.
- Semantic times must be finite `[B*2, 4]` and equal to `[0.25, 0.5, 0.75, 1.0]` for both views.
- Evaluation must fail closed; it must never continue without planner conditioning.
- Preserve the existing FastWAM state/action statistics and gripper conversion.
- Run on local `cuda:1` using bfloat16.
- Do not modify or add the unrelated untracked AgiBot and `semantic_localization/` files.

## File Structure

- Create `ge_act/experiments/joint_libero_eval_contract.py`: pure checkpoint validation, one-pass image normalization, planner-output validation, and fail-closed pipeline proxy.
- Create `ge_act/experiments/eval_libero_joint.py`: load the joint export and frozen planner, produce online conditions before each base `play()` call, and expose a joint-only CLI.
- Create `ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml`: local model geometry and FastWAM evaluation settings.
- Create `ge_act/scripts/eval_joint_vlm_geact_a6000.sh`: preflight, smoke, and full-suite launcher.
- Create `tests/test_joint_vlm_geact_libero_eval.py`: checkpoint, tensor, integration-source, config, and launcher contracts.

---

### Task 1: Add the fail-closed joint checkpoint and semantic tensor contract

**Files:**
- Create: `ge_act/experiments/joint_libero_eval_contract.py`
- Create: `tests/test_joint_vlm_geact_libero_eval.py`

**Interfaces:**
- Consumes: `FrozenDualCameraVLMPlanner.predict(current_images, instructions)`.
- Produces: `JointEvaluationCheckpoint`, `validate_joint_evaluation_checkpoint(path, expected_step=40000)`, `normalize_joint_current_images(images)`, `build_joint_semantic_condition(planner, current_images, instruction, device, dtype)`, and `SemanticConditionedPipelineProxy`.

- [ ] **Step 1: Write failing checkpoint-contract tests**

Add fixtures that create a minimal valid export and assert the exact contract:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPO_ROOT / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from experiments.joint_libero_eval_contract import (
    SemanticConditionedPipelineProxy,
    build_joint_semantic_condition,
    normalize_joint_current_images,
    validate_joint_evaluation_checkpoint,
)

EVAL_LIBERO_PATH = GE_ACT_ROOT / "experiments" / "eval_libero.py"
JOINT_EVAL_PATH = GE_ACT_ROOT / "experiments" / "eval_libero_joint.py"
EVAL_CONFIG_PATH = (
    GE_ACT_ROOT
    / "configs"
    / "ltx_model"
    / "libero"
    / "action_model_libero_joint_step40000_eval.yaml"
)
EVAL_LAUNCHER_PATH = (
    GE_ACT_ROOT / "scripts" / "eval_joint_vlm_geact_a6000.sh"
)


def valid_k4_planner_metadata() -> dict[str, object]:
    return {
        "planner_input_layout": "separate_camera_images",
        "camera_names": ["main", "wrist"],
        "num_camera_views": 2,
        "camera_head_sharing": "shared_head_per_view_image_context",
        "semantic_output_layout": "batch_view_keyframe_token_feature",
        "semantic_teacher": "siglip2-large-patch16-256",
        "future_keyframe_offsets": [2, 4, 6, 8],
        "num_keyframes": 4,
        "sequence_length": 9,
        "grid_size": 16,
        "semantic_dim": 1024,
        "target_tokens": 1024,
        "target_tokens_per_keyframe": 256,
        "planner_token_count": 384,
        "video_target_type": "siglip2",
        "use_current_alignment": False,
        "plan_token_strings": [
            f"<|sem_plan_{index}|>" for index in range(384)
        ],
    }


def write_joint_export(tmp_path: Path, *, global_step: int) -> Path:
    root = tmp_path / "step_40000"
    ltx = root / "ltx"
    planner = root / "planner"
    ltx.mkdir(parents=True)
    (planner / "qwen3vl_lora_or_model").mkdir(parents=True)
    (planner / "processor").mkdir()
    (root / "joint_meta.json").write_text(json.dumps({
        "global_step": global_step,
        "num_camera_views": 2,
        "num_keyframes": 4,
        "tokens_per_keyframe": 256,
        "future_keyframe_offsets": [2, 4, 6, 8],
    }))
    (ltx / "config.json").write_text(json.dumps({
        "semantic_plan_context": True,
        "semantic_plan_in_dim": 1024,
        "semantic_plan_num_keyframes": 4,
        "semantic_plan_num_views": 2,
    }))
    (ltx / "diffusion_pytorch_model.safetensors").touch()
    (planner / "planner_meta.json").write_text(
        json.dumps(valid_k4_planner_metadata())
    )
    for name in ("plan_head.pt", "depth_head.pt", "plan_token_embedding.pt"):
        (planner / name).touch()
    return root


def test_joint_checkpoint_contract_accepts_exact_step40000(tmp_path: Path) -> None:
    root = write_joint_export(tmp_path, global_step=40_000)
    checkpoint = validate_joint_evaluation_checkpoint(root)
    assert checkpoint.root == root
    assert checkpoint.ltx_dir == root / "ltx"
    assert checkpoint.planner_dir == root / "planner"
    assert checkpoint.metadata["future_keyframe_offsets"] == [2, 4, 6, 8]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_step", 39_999),
        ("num_camera_views", 1),
        ("num_keyframes", 1),
        ("tokens_per_keyframe", 64),
        ("future_keyframe_offsets", [1, 3, 5, 7]),
    ],
)
def test_joint_checkpoint_contract_rejects_metadata_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = write_joint_export(tmp_path, global_step=40_000)
    metadata = json.loads((root / "joint_meta.json").read_text())
    metadata[field] = value
    (root / "joint_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        validate_joint_evaluation_checkpoint(root)


def test_joint_checkpoint_contract_requires_both_model_exports(tmp_path: Path) -> None:
    root = write_joint_export(tmp_path, global_step=40_000)
    (root / "planner" / "plan_head.pt").unlink()
    with pytest.raises(FileNotFoundError, match="plan_head.pt"):
        validate_joint_evaluation_checkpoint(root)
```

The fixture writes `ltx/config.json` with:

```json
{
  "semantic_plan_context": true,
  "semantic_plan_in_dim": 1024,
  "semantic_plan_num_keyframes": 4,
  "semantic_plan_num_views": 2
}
```

The helper above creates the Qwen/processor directories and all three exported head/embedding files without allocating real model weights.

- [ ] **Step 2: Run the checkpoint tests and confirm the module is absent**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_libero_eval.py -k checkpoint
```

Expected: collection fails with `ModuleNotFoundError: experiments.joint_libero_eval_contract`.

- [ ] **Step 3: Implement exact checkpoint validation**

Create:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.ltx_models.vlm_semantic_planner import (
    validate_dual_camera_planner_metadata,
)


@dataclass(frozen=True)
class JointEvaluationCheckpoint:
    root: Path
    ltx_dir: Path
    planner_dir: Path
    metadata: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


def _require_equal(metadata: dict[str, Any], field: str, expected: Any) -> None:
    actual = metadata.get(field)
    if actual != expected:
        raise ValueError(
            f"incompatible joint metadata field {field}: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_joint_evaluation_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_step: int = 40_000,
) -> JointEvaluationCheckpoint:
    root = Path(checkpoint_dir).expanduser().resolve()
    joint = _read_json(root / "joint_meta.json")
    for field, expected in {
        "global_step": expected_step,
        "num_camera_views": 2,
        "num_keyframes": 4,
        "tokens_per_keyframe": 256,
        "future_keyframe_offsets": [2, 4, 6, 8],
    }.items():
        _require_equal(joint, field, expected)

    ltx_dir = root / "ltx"
    planner_dir = root / "planner"
    ltx_config = _read_json(ltx_dir / "config.json")
    for field, expected in {
        "semantic_plan_context": True,
        "semantic_plan_in_dim": 1024,
        "semantic_plan_num_keyframes": 4,
        "semantic_plan_num_views": 2,
    }.items():
        _require_equal(ltx_config, field, expected)
    if not any(ltx_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"missing LTX safetensors export in {ltx_dir}")

    planner_meta = _read_json(planner_dir / "planner_meta.json")
    validate_dual_camera_planner_metadata(planner_meta)
    for name in ("plan_head.pt", "depth_head.pt", "plan_token_embedding.pt"):
        if not (planner_dir / name).is_file():
            raise FileNotFoundError(f"missing planner export file: {planner_dir / name}")
    for name in ("qwen3vl_lora_or_model", "processor"):
        if not (planner_dir / name).is_dir():
            raise FileNotFoundError(f"missing planner export directory: {planner_dir / name}")
    return JointEvaluationCheckpoint(root, ltx_dir, planner_dir, joint)
```

- [ ] **Step 4: Add failing semantic-conditioning tests**

Use a recording fake planner, distinct main/wrist pixels, and a recording base pipeline:

```python
class RecordingPlanner:
    def __init__(self, *, fill_value: float = 0.0) -> None:
        self.fill_value = fill_value
        self.current_images: torch.Tensor | None = None

    def predict(
        self,
        current_images: torch.Tensor,
        instructions: list[str],
    ) -> SimpleNamespace:
        self.current_images = current_images.clone()
        assert instructions == ["pick the bowl"] or instructions == ["pick"]
        return SimpleNamespace(
            semantic_tokens=torch.full(
                (1, 2, 4, 256, 1024),
                self.fill_value,
            ),
            times=torch.tensor(
                [[0.25, 0.5, 0.75, 1.0]] * 2,
            ),
        )


class RecordingPipeline:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def infer(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return kwargs


def test_joint_image_normalization_converts_uint8_exactly_once() -> None:
    raw = torch.stack((
        torch.zeros(3, 8, 8, dtype=torch.uint8),
        torch.full((3, 8, 8), 255, dtype=torch.uint8),
    ))
    normalized = normalize_joint_current_images(raw)
    assert normalized.dtype == torch.float32
    assert normalized[0].eq(-1).all()
    assert normalized[1].eq(1).all()


def test_semantic_condition_preserves_main_wrist_order() -> None:
    planner = RecordingPlanner()
    current = torch.stack((torch.full((3, 8, 8), -0.5),
                           torch.full((3, 8, 8), 0.5)))
    plan, times, mask = build_joint_semantic_condition(
        planner, current, "pick the bowl", device="cpu", dtype=torch.bfloat16
    )
    assert planner.current_images.shape == (1, 2, 3, 8, 8)
    torch.testing.assert_close(planner.current_images[0], current)
    assert plan.shape == (1, 2, 4, 256, 1024)
    assert times.shape == (2, 4)
    assert mask.tolist() == [1.0, 1.0]


def test_semantic_condition_rejects_nonfinite_output() -> None:
    planner = RecordingPlanner(fill_value=float("nan"))
    with pytest.raises(RuntimeError, match="finite"):
        build_joint_semantic_condition(
            planner, torch.zeros(2, 3, 8, 8), "pick",
            device="cpu", dtype=torch.bfloat16,
        )


def test_pipeline_proxy_requires_and_forwards_semantics() -> None:
    base = RecordingPipeline()
    pending = [{
        "semantic_plan": torch.zeros(1, 2, 4, 256, 1024),
        "semantic_plan_times": torch.zeros(2, 4),
        "semantic_condition_mask": torch.ones(2),
    }]
    proxy = SemanticConditionedPipelineProxy(base, pending.pop)
    proxy.infer(image=torch.zeros(2, 3, 4, 8, 8))
    assert set(base.kwargs) >= {
        "semantic_plan",
        "semantic_plan_times",
        "semantic_condition_mask",
    }
    with pytest.raises(RuntimeError, match="conditioning"):
        proxy.infer(image=torch.zeros(2, 3, 4, 8, 8))
```

Import `SimpleNamespace` from `types`; the fake returns the same public fields as `DualCameraSemanticPlan` without importing the real Qwen loader.

- [ ] **Step 5: Run the semantic tests and confirm the function is absent**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_libero_eval.py -k semantic_condition
```

Expected: fail because `build_joint_semantic_condition` is not defined.

- [ ] **Step 6: Implement normalization, semantic conditioning, and the proxy**

Append:

```python
def normalize_joint_current_images(images: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(images)
    if value.ndim != 4:
        raise ValueError(f"two-view images must be rank 4, got {tuple(value.shape)}")
    if value.shape[1] != 3 and value.shape[-1] == 3:
        value = value.permute(0, 3, 1, 2)
    if tuple(value.shape[:2]) != (2, 3):
        raise ValueError(
            f"images must be ordered [main,wrist] as [2,3,H,W], "
            f"got {tuple(value.shape)}"
        )
    if value.dtype == torch.uint8:
        value = value.float().div(127.5).sub(1.0)
    else:
        value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError("current images must be finite")
        if value.min() < -1.0001 or value.max() > 1.0001:
            raise ValueError("float current images must already be in [-1,1]")
    return value.contiguous()


def build_joint_semantic_condition(
    planner: Any,
    current_images: torch.Tensor,
    instruction: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(current_images.shape[:2]) != (2, 3) or current_images.ndim != 4:
        raise ValueError(
            f"current images must be ordered [main,wrist] as [2,3,H,W], "
            f"got {tuple(current_images.shape)}"
        )
    if not torch.isfinite(current_images).all():
        raise ValueError("current images must be finite")
    if current_images.min() < -1.0001 or current_images.max() > 1.0001:
        raise ValueError("current images must be normalized to [-1,1]")
    result = planner.predict(current_images.unsqueeze(0), [str(instruction)])
    plan = result.semantic_tokens
    times = result.times
    if tuple(plan.shape) != (1, 2, 4, 256, 1024):
        raise RuntimeError(f"semantic plan shape mismatch: {tuple(plan.shape)}")
    if tuple(times.shape) != (2, 4):
        raise RuntimeError(f"semantic time shape mismatch: {tuple(times.shape)}")
    if not torch.isfinite(plan).all() or not torch.isfinite(times).all():
        raise RuntimeError("semantic plan and times must be finite")
    expected_times = torch.tensor(
        [0.25, 0.5, 0.75, 1.0],
        device=times.device,
        dtype=times.dtype,
    ).repeat(2, 1)
    if not torch.allclose(times, expected_times, atol=1e-6, rtol=0):
        raise RuntimeError(f"semantic time values mismatch: {times}")
    target_device = torch.device(device)
    return (
        plan.to(device=target_device, dtype=dtype),
        times.to(device=target_device, dtype=torch.float32),
        torch.ones(2, device=target_device, dtype=dtype),
    )


class SemanticConditionedPipelineProxy:
    def __init__(self, pipeline: Any, consume_conditioning: Any) -> None:
        self._pipeline = pipeline
        self._consume_conditioning = consume_conditioning

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        try:
            conditioning = self._consume_conditioning()
        except IndexError as error:
            raise RuntimeError(
                "joint pipeline inference has no pending semantic conditioning"
            ) from error
        required = {
            "semantic_plan",
            "semantic_plan_times",
            "semantic_condition_mask",
        }
        if set(conditioning) != required:
            raise RuntimeError(
                f"joint pipeline conditioning keys mismatch: {set(conditioning)}"
            )
        overlap = required.intersection(kwargs)
        if overlap:
            raise RuntimeError(f"duplicate semantic conditioning keys: {overlap}")
        return self._pipeline.infer(*args, **kwargs, **conditioning)
```

- [ ] **Step 7: Run and commit Task 1**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_libero_eval.py
git diff --check
git add ge_act/experiments/joint_libero_eval_contract.py tests/test_joint_vlm_geact_libero_eval.py
git commit -m "feat(eval): validate joint VLM GE-Act checkpoints"
```

Expected: all new tests pass and the commit contains only the contract and its tests.

---

### Task 2: Integrate online planner conditioning into LIBERO evaluation

**Files:**
- Create: `ge_act/experiments/eval_libero_joint.py`
- Modify: `tests/test_joint_vlm_geact_libero_eval.py`

**Interfaces:**
- Consumes: Task 1's validator, normalizer, semantic builder, and pipeline proxy.
- Produces: `JointInferenceLibero`, whose overridden `play()` creates exactly one planner condition before delegating history/action processing to the unmodified base evaluator.

- [ ] **Step 1: Add a failing joint evaluator contract**

Add:

```python
def test_joint_evaluator_has_no_semantic_free_fallback() -> None:
    source = JOINT_EVAL_PATH.read_text(encoding="utf-8")
    assert "FrozenDualCameraVLMPlanner.from_checkpoint" in source
    assert "normalize_joint_current_images(" in source
    assert "build_joint_semantic_condition(" in source
    assert "SemanticConditionedPipelineProxy(" in source
    assert "return super().play(" in source
    assert "return {}" not in source


def test_original_evaluator_is_not_modified_for_joint_conditioning() -> None:
    source = EVAL_LIBERO_PATH.read_text(encoding="utf-8")
    assert "joint_libero_eval_contract" not in source
    assert "semantic_planner" not in source
```

- [ ] **Step 2: Run the integration contract and observe failure**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_libero_eval.py -k evaluator
```

Expected: fail because the dedicated joint evaluator does not exist.

- [ ] **Step 3: Implement the dedicated joint evaluator**

Create:

```python
from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from experiments.eval_libero import InferenceLibero
from experiments.joint_libero_eval_contract import (
    SemanticConditionedPipelineProxy,
    build_joint_semantic_condition,
    normalize_joint_current_images,
    validate_joint_evaluation_checkpoint,
)
from models.ltx_models.vlm_semantic_planner import (
    FrozenDualCameraVLMPlanner,
)


class JointInferenceLibero(InferenceLibero):
    def __init__(self, *, joint_checkpoint_dir: str, **kwargs: Any) -> None:
        self.joint_checkpoint = validate_joint_evaluation_checkpoint(
            joint_checkpoint_dir
        )
        self.semantic_planner: FrozenDualCameraVLMPlanner | None = None
        self._pending_conditioning: list[dict[str, torch.Tensor]] = []
        self._semantic_shape_logged = False
        super().__init__(
            model_path=str(self.joint_checkpoint.ltx_dir),
            **kwargs,
        )

    def prepare_models(self) -> None:
        super().prepare_models()
        self.semantic_planner = FrozenDualCameraVLMPlanner.from_checkpoint(
            self.joint_checkpoint.planner_dir,
            device=self.device,
            dtype=self.weight_dtype,
        )
        self.pipeline = SemanticConditionedPipelineProxy(
            self.pipeline,
            self._pending_conditioning.pop,
        )
        self.log_file.write(
            "joint_checkpoint="
            f"{self.joint_checkpoint.root} camera_order=main,wrist "
            "semantic_shape=[1,2,4,256,1024] offsets=[2,4,6,8]\n"
        )
        self.log_file.flush()

    def play(
        self,
        obs: torch.Tensor,
        prompt: str,
        excution_step: int = 1,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.semantic_planner is None:
            raise RuntimeError("joint semantic planner is not loaded")
        if self._pending_conditioning:
            raise RuntimeError("stale semantic conditioning was not consumed")
        current_images = normalize_joint_current_images(obs)
        semantic_plan, semantic_plan_times, semantic_condition_mask = (
            build_joint_semantic_condition(
                self.semantic_planner,
                current_images,
                prompt,
                device=self.device,
                dtype=self.weight_dtype,
            )
        )
        if not self._semantic_shape_logged:
            self.log_file.write(
                f"actual_semantic_shape={list(semantic_plan.shape)} "
                f"actual_semantic_times_shape={list(semantic_plan_times.shape)}\n"
            )
            self.log_file.flush()
            self._semantic_shape_logged = True
        self._pending_conditioning.append({
            "semantic_plan": semantic_plan,
            "semantic_plan_times": semantic_plan_times,
            "semantic_condition_mask": semantic_condition_mask,
        })
        try:
            actions = super().play(
                current_images,
                prompt,
                excution_step=excution_step,
                state=state,
            )
        except BaseException:
            self._pending_conditioning.clear()
            raise
        if self._pending_conditioning:
            self._pending_conditioning.clear()
            raise RuntimeError(
                "base evaluator returned without consuming semantic conditioning"
            )
        return actions
```

The one-pass normalizer prevents the existing base evaluator's torch-uint8 branch from applying its duplicate conversion: the base receives finite float images already in `[-1,1]`. The proxy consumes exactly one semantic condition per `play()` call and raises if inference is attempted without it.

Use this exact CLI wiring:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--joint_ckpt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--exec_step", type=int, default=8)
    parser.add_argument("--num_trails_per_task", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.task_suite_name)
    evaluator = JointInferenceLibero(
        joint_checkpoint_dir=args.joint_ckpt_dir,
        config_file=args.config_file,
        output_dir=output_dir,
        task_suite_name=args.task_suite_name,
        exec_step=args.exec_step,
        device=f"cuda:{args.device}",
        threshold=args.threshold,
    )
    evaluator.prepare_models()
    evaluator.infer(
        num_trails_per_task=args.num_trails_per_task,
        image_shape=evaluator.args.data["train"]["sample_size"],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify imports and contracts**

Run:

```bash
PYTHONPATH="$PWD/ge_act:/data/LFT-W02_data/junjie/VLA_RL/docker_libero/LIBERO" \
  /data/LFT-W02_data/.conda/envs/ge-act/bin/python \
  -m py_compile \
  ge_act/experiments/eval_libero.py \
  ge_act/experiments/eval_libero_joint.py \
  ge_act/experiments/joint_libero_eval_contract.py
pytest -q tests/test_joint_vlm_geact_libero_eval.py
```

Expected: compilation succeeds and every joint evaluator test passes.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git diff --check
git add \
  ge_act/experiments/eval_libero_joint.py \
  tests/test_joint_vlm_geact_libero_eval.py
git commit -m "feat(eval): condition LIBERO rollouts with joint planner"
```

---

### Task 3: Add the local A6000 configuration and guarded launcher

**Files:**
- Create: `ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml`
- Create: `ge_act/scripts/eval_joint_vlm_geact_a6000.sh`
- Modify: `tests/test_joint_vlm_geact_libero_eval.py`

**Interfaces:**
- Consumes: Task 2's `eval_libero_joint.py` CLI.
- Produces: `bash ge_act/scripts/eval_joint_vlm_geact_a6000.sh smoke|full`.

- [ ] **Step 1: Add failing config and launcher tests**

Add tests that parse the YAML and shell source:

```python
def test_a6000_eval_config_matches_joint_training_geometry() -> None:
    config = yaml.safe_load(EVAL_CONFIG_PATH.read_text())
    assert config["pretrained_model_name_or_path"] == (
        "/data/LFT-W02_data/junjie/weights/LTX-Video"
    )
    assert config["return_action"] is True
    assert config["add_state"] is True
    assert config["data"]["train"]["sample_size"] == [256, 256]
    assert config["data"]["train"]["n_previous"] == 4
    assert config["data"]["train"]["chunk"] == 9
    assert config["data"]["train"]["action_chunk"] == 36
    model = config["diffusion_model"]["config"]
    assert model["semantic_plan_context"] is True
    assert model["semantic_plan_num_views"] == 2
    assert model["semantic_plan_num_keyframes"] == 4
    assert model["semantic_plan_in_dim"] == 1024
    assert model["action_in_channels"] == 15
    assert model["action_out_channels"] == 15


def test_a6000_launcher_smoke_gates_full_evaluation() -> None:
    source = EVAL_LAUNCHER_PATH.read_text()
    assert "CUDA_VISIBLE_DEVICES=1" in source
    assert "SMOKE_MAX_TASKS=1" in source
    assert "--num_trails_per_task 1" in source
    assert "libero_spatial libero_object libero_goal libero_10" in source
    assert "--num_trails_per_task 50" in source
    assert "training_state" not in source
```

- [ ] **Step 2: Run tests and confirm both files are absent**

Run:

```bash
pytest -q tests/test_joint_vlm_geact_libero_eval.py -k 'a6000'
```

Expected: fail on missing YAML and shell launcher.

- [ ] **Step 3: Create the local evaluation YAML**

Create this complete inference-only YAML:

```yaml
model_name: ltx_train
output_dir: /data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000
pretrained_model_name_or_path: /data/LFT-W02_data/junjie/weights/LTX-Video
tokenizer_class_path: transformers
tokenizer_class: T5Tokenizer
textenc_class_path: transformers
textenc_class: T5EncoderModel
vae_class_path: models/ltx_models/autoencoder_kl_ltx.py
vae_class: AutoencoderKLLTXVideo
diffusion_model_class_path: models/ltx_models/transformer_ltx_multiview.py
diffusion_model_class: LTXVideoTransformer3DModel
diffusion_scheduler_class_path: diffusers
diffusion_scheduler_class: FlowMatchEulerDiscreteScheduler
pipeline_class_path: models/pipeline/custom_pipeline.py
pipeline_class: CustomPipeline
return_action: true
return_video: false
add_state: true
enable_slicing: false
enable_tiling: true
mixed_precision: bf16
pixel_wise_timestep: true
num_inference_step: 10
load_weights: true

diffusion_model:
  model_path: /data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000/ltx
  config:
    activation_fn: gelu-approximate
    attention_bias: true
    attention_head_dim: 64
    attention_out_bias: true
    caption_channels: 4096
    cross_attention_dim: 2048
    in_channels: 128
    norm_elementwise_affine: false
    norm_eps: 1.0e-6
    num_attention_heads: 32
    num_layers: 28
    out_channels: 128
    patch_size: 1
    patch_size_t: 1
    qk_norm: rms_norm_across_heads
    action_expert: true
    action_in_channels: 15
    action_out_channels: 15
    action_num_attention_heads: 16
    action_attention_head_dim: 32
    semantic_plan_context: true
    semantic_plan_in_dim: 1024
    semantic_plan_coordinate_dim: 256
    semantic_plan_num_keyframes: 4
    semantic_plan_num_views: 2
    semantic_plan_adaln_rank: 256
    semantic_plan_cross_attention_blocks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

data:
  train:
    sample_size: [256, 256]
    sample_n_frames: 500
    n_previous: 4
    chunk: 9
    action_chunk: 36
    stat_file: configs/ltx_model/libero/libero_fastwam_mix.json
  val:
    action_chunk: 36
    chunk: 9
    stat_file: configs/ltx_model/libero/libero_fastwam_mix.json
```

- [ ] **Step 4: Create the guarded launcher**

Implement:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-/data/LFT-W02_data/.conda/envs/ge-act/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-/data/LFT-W02_data/junjie/VLA_RL/docker_libero/LIBERO}
CHECKPOINT=${CHECKPOINT:-/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000}
CONFIG=${CONFIG:-$ROOT/ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml}
OUTPUT=${OUTPUT:-/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000}
MODE=${1:-smoke}

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT/ge_act:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}

COMMON=(
  "$PY" "$ROOT/ge_act/experiments/eval_libero_joint.py"
  --config_file "$CONFIG"
  --joint_ckpt_dir "$CHECKPOINT"
  --output_dir "$OUTPUT"
  --device 0
  --exec_step 8
  --threshold 20
)

if [[ "$MODE" == smoke ]]; then
  SMOKE_MAX_TASKS=1 "${COMMON[@]}" \
    --task_suite_name libero_goal \
    --num_trails_per_task 1
elif [[ "$MODE" == full ]]; then
  SMOKE_MAX_TASKS=1 "${COMMON[@]}" \
    --task_suite_name libero_goal \
    --num_trails_per_task 1
  for suite in libero_spatial libero_object libero_goal libero_10; do
    "${COMMON[@]}" \
      --task_suite_name "$suite" \
      --num_trails_per_task 50
  done
else
  echo "usage: $0 smoke|full" >&2
  exit 2
fi
```

The full mode deliberately reruns the smoke gate in the same environment before starting suite evaluation.

- [ ] **Step 5: Verify config, shell, and focused tests**

Run:

```bash
bash -n ge_act/scripts/eval_joint_vlm_geact_a6000.sh
pytest -q tests/test_joint_vlm_geact_libero_eval.py
git diff --check
```

Expected: shell syntax and all focused tests pass.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add \
  ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml \
  ge_act/scripts/eval_joint_vlm_geact_a6000.sh \
  tests/test_joint_vlm_geact_libero_eval.py
git commit -m "feat(eval): add guarded A6000 joint evaluation launcher"
```

---

### Task 4: Run repository verification and transfer the inference checkpoint

**Files:**
- No repository files modified.
- Generated locally, not committed: `/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000/`

**Interfaces:**
- Consumes: committed Tasks 1–3 and the completed HPC3 step-40,000 export.
- Produces: verified local LTX and planner inference exports.

- [ ] **Step 1: Run focused and neighboring regression tests**

Run:

```bash
pytest -q \
  tests/test_joint_vlm_geact_libero_eval.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_ge_act_semantic_pipeline.py \
  tests/test_ge_act_dual_camera_planner.py
bash -n ge_act/scripts/eval_joint_vlm_geact_a6000.sh
git diff --check
```

Expected: all tests pass, shell syntax is valid, and there are no whitespace errors.

- [ ] **Step 2: Verify the remote export and local free space**

Run:

```bash
REMOTE=/data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k4_50k/2026_07_21_14_48_26/step_40000
ssh hpc3 "test -f '$REMOTE/joint_meta.json' \
  && test -f '$REMOTE/ltx/config.json' \
  && test -f '$REMOTE/ltx/diffusion_pytorch_model.safetensors' \
  && test -f '$REMOTE/planner/planner_meta.json' \
  && test -f '$REMOTE/planner/plan_head.pt' \
  && test -d '$REMOTE/planner/qwen3vl_lora_or_model' \
  && du -sh '$REMOTE/ltx' '$REMOTE/planner'"
df -h /data/LFT-W02_data
```

Expected: remote LTX is about 5.0 GB, planner about 4.2 GB, and local free space exceeds 12 GB.

- [ ] **Step 3: Transfer only inference exports with resume support**

Run:

```bash
REMOTE=/data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k4_50k/2026_07_21_14_48_26/step_40000
LOCAL=/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000
mkdir -p "$LOCAL"
rsync -a --partial --append-verify --info=progress2 \
  hpc3:"$REMOTE/joint_meta.json" "$LOCAL/"
rsync -a --partial --append-verify --info=progress2 \
  hpc3:"$REMOTE/ltx/" "$LOCAL/ltx/"
rsync -a --partial --append-verify --info=progress2 \
  hpc3:"$REMOTE/planner/" "$LOCAL/planner/"
```

Expected: local checkpoint contains about 9.2 GB and has no `training_state/`.

- [ ] **Step 4: Compare file manifests and run checkpoint preflight**

Run:

```bash
REMOTE=/data/user/jhe724/junjie/outputs/joint_vlm_geact_action_k4_50k/2026_07_21_14_48_26/step_40000
LOCAL=/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000
ssh hpc3 "cd '$REMOTE' && find ltx planner -type f -printf '%P %s\n' | sort" \
  > /tmp/joint_step40000.remote.manifest
(cd "$LOCAL" && find ltx planner -type f -printf '%P %s\n' | sort) \
  > /tmp/joint_step40000.local.manifest
diff -u /tmp/joint_step40000.remote.manifest /tmp/joint_step40000.local.manifest
test ! -e "$LOCAL/training_state"
PYTHONPATH="$PWD/ge_act" /data/LFT-W02_data/.conda/envs/ge-act/bin/python - <<'PY'
from experiments.joint_libero_eval_contract import validate_joint_evaluation_checkpoint
ckpt = validate_joint_evaluation_checkpoint(
    "/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000"
)
print(ckpt.metadata)
PY
```

Expected: manifests match exactly, `training_state/` is absent, and preflight prints `global_step: 40000`.

---

### Task 5: Run A6000 smoke and launch the full LIBERO evaluation

**Files:**
- No repository files modified.
- Runtime output: `/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/`
- Runtime logs: `/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/{smoke,full}.log`

**Interfaces:**
- Consumes: verified code, environment, and local step-40,000 export.
- Produces: smoke evidence and four-suite LIBERO metrics.

- [ ] **Step 1: Confirm GPU 1 is free and the environment imports**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader
PYTHONPATH="$PWD/ge_act:/data/LFT-W02_data/junjie/VLA_RL/docker_libero/LIBERO" \
  /data/LFT-W02_data/.conda/envs/ge-act/bin/python - <<'PY'
import torch
import libero
import qwen_vl_utils
from experiments.eval_libero_joint import JointInferenceLibero
print(torch.__version__, JointInferenceLibero.__name__)
PY
```

Expected: physical GPU 1 is idle, LIBERO and Qwen utilities import, and the evaluator class prints.

- [ ] **Step 2: Run the one-task, one-trial smoke synchronously**

Run:

```bash
mkdir -p /data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000
set -o pipefail
bash ge_act/scripts/eval_joint_vlm_geact_a6000.sh smoke 2>&1 \
  | tee /data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/smoke.log
```

Expected: exit code 0; logs show step 40000, `camera_order=main,wrist`, semantic shape `[1,2,4,256,1024]`, finite action output, and one completed episode. Success or failure of the task itself is a metric, not a smoke failure.

- [ ] **Step 3: Inspect smoke artifacts before launching full evaluation**

Run:

```bash
OUTPUT=/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000
rg -n "joint_checkpoint=|semantic_shape=|Success:|episodes completed" \
  "$OUTPUT/smoke.log"
find "$OUTPUT" -type f \( -name '*.mp4' -o -name 'inference_*.txt' \) -printf '%p %s\n'
```

Expected: one rollout result and one non-empty rollout video are present, with no traceback or CUDA OOM.

- [ ] **Step 4: Launch the full four-suite evaluation detached**

Run:

```bash
OUTPUT=/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000
nohup bash ge_act/scripts/eval_joint_vlm_geact_a6000.sh full \
  > "$OUTPUT/full.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" | tee "$OUTPUT/full.pid"
sleep 2
ps -p "$PID" -o pid,etime,cmd
```

Expected: the process remains alive and starts with the smoke gate before `libero_spatial`.

- [ ] **Step 5: Record launch evidence without continuously monitoring**

Run once:

```bash
OUTPUT=/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000
PID=$(cat "$OUTPUT/full.pid")
ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
tail -40 "$OUTPUT/full.log"
```

Expected: the PID is active on visible GPU 0, which maps to physical A6000 GPU 1 through `CUDA_VISIBLE_DEVICES=1`. Report the PID, output directory, current suite, and first observed GPU memory usage to the user; do not enter a continuous polling loop.
