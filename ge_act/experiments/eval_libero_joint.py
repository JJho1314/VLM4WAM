"""LIBERO rollout evaluation for a joint dual-camera VLM–GE-Act export."""

from __future__ import annotations

import argparse
import os
import sys
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
    """Run GE-Act with a mandatory online semantic plan at every replan."""

    def __init__(
        self,
        *,
        joint_checkpoint_dir: str,
        **kwargs: Any,
    ) -> None:
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
            f"joint_checkpoint={self.joint_checkpoint.root} "
            "global_step=40000 camera_order=main,wrist "
            "semantic_shape=[1,2,4,256,1024] offsets=[2,4,6,8] "
            f"dtype={self.weight_dtype} device={self.device}\n"
        )
        self.log_file.flush()

    @torch.no_grad()
    def play(
        self,
        obs: Any,
        prompt: str,
        excution_step: int = 1,
        state: Any = None,
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

        self._pending_conditioning.append(
            {
                "semantic_plan": semantic_plan,
                "semantic_plan_times": semantic_plan_times,
                "semantic_condition_mask": semantic_condition_mask,
            }
        )
        try:
            return super().play(
                current_images,
                prompt,
                excution_step=excution_step,
                state=state,
            )
        finally:
            condition_was_not_consumed = bool(self._pending_conditioning)
            self._pending_conditioning.clear()
            if condition_was_not_consumed and sys.exc_info()[0] is None:
                raise RuntimeError(
                    "base evaluator returned without consuming semantic conditioning"
                )


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
