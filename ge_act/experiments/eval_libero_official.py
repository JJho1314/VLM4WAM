#!/usr/bin/env python
"""Official GE-Act checkpoint support (ge_act_libero_{goal,object,spatial,10}.safetensors).

Kept SEPARATE from experiments/eval_libero.py on purpose: the official checkpoints and our
fastwam-trained ones use mutually destructive conventions, and mixing them up produces
plausible-looking-but-garbage actions rather than an error.

                     official (this file)              fastwam (eval_libero.py)
  action de-norm     (a+1)/2 * (q99-q01+1e-6) + q01    a * std + mean
  gripper            already env-native [-1,1] ->       stored [0,1] (0=close,1=open) ->
                     PASS THROUGH untouched             g_env = 1 - 2*g
  state norm         (s-q01)/(q99-q01+1e-6) * 2 - 1    (s - mean) / (std + 1e-6)
  stat file          configs/ltx_model/libero/          .../libero_fastwam_mix.json
                     libero_all.json (keys "libero_*")

Verified against data/libero_dataset.py (the loader the official weights were trained with):
  L474-475  state  = (state  - q01)/(q99 - q01 + 1e-6);  state  = state  * 2 - 1
  L482-483  action = (action - q01)/(q99 - q01 + 1e-6);  action = action * 2 - 1
and configs/ltx_model/libero/libero_all.json, where libero_eef dim6 (gripper) has
q01=-1.0 / q99=+1.0 -- i.e. the minmax round-trip is the identity for the gripper, so the
official model already emits the env's own convention. Applying the fastwam 1-2g flip here
would map -1 -> +3 and destroy every grasp.
"""
import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments.eval_libero import InferenceLibero


class InferenceLiberoOfficial(InferenceLibero):
    """InferenceLibero with the official LIBERO q01/q99 normalization."""

    def denormalize_actions(self, actions_pred):
        """Model output -> LIBERO env action space (official checkpoints)."""
        d = self.basic_action_dim
        ### inverse of data/libero_dataset.py L482-483
        actions_pred = (actions_pred.float() + 1.0) / 2.0
        actions_pred = actions_pred * (self.act_max[:, :d] - self.act_min[:, :d] + 1e-6) \
            + self.act_min[:, :d]
        ### gripper: NO flip -- see module docstring.
        return actions_pred

    def normalize_state(self, state):
        """Raw env state -> model input space (official checkpoints)."""
        state = state.float() if torch.is_tensor(state) else torch.tensor(state).float()
        ### inverse-free forward of data/libero_dataset.py L474-475
        state = (state - self.states_min) / (self.states_max - self.states_min + 1e-6)
        return state * 2.0 - 1.0


if __name__ == "__main__":
    ### Standard (unperturbed) LIBERO with an official checkpoint -- the wiring check.
    ### The release reports ~0.96 here, so a number anywhere near that confirms the
    ### normalization/gripper plumbing; a collapse means this file is wired wrong, NOT
    ### that the released weights are weak.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_suite_name", type=str, default="libero_goal")
    parser.add_argument("--exec_step", type=int, default=8)
    parser.add_argument("--num_trails_per_task", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=20)
    args = parser.parse_args()

    infer = InferenceLiberoOfficial(
        config_file=args.config_file,
        output_dir=os.path.join(args.output_dir, args.task_suite_name),
        task_suite_name=args.task_suite_name, model_path=args.ckpt_path,
        exec_step=args.exec_step, device=f"cuda:{args.device}", threshold=args.threshold,
    )
    infer.prepare_models()
    infer.infer(num_trails_per_task=args.num_trails_per_task,
                image_shape=infer.args.data["train"]["sample_size"])
