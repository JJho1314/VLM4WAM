#!/usr/bin/env python
"""LIBERO-plus eval entrypoint for the RELEASED official GE-Act weights.

Separate entrypoint from experiments/eval_libero_plus.py so the official q01/q99 +
pass-through-gripper convention can never be applied to our fastwam checkpoints (or vice
versa) by accident -- see experiments/eval_libero_official.py for why that would silently
produce garbage instead of an error.

The official release ships ONE checkpoint PER SUITE, so run this once per suite:

  python experiments/eval_libero_plus_official.py \
    --config_file configs/ltx_model/libero/action_model_libero_official_eval.yaml \
    --ckpt_path <.../ge_act_libero_goal.safetensors> \
    --suites libero_goal --out_dir <.../official_goal> \
    --device 0 --shard 0 --num_shards 5

The config must point stat_file at configs/ltx_model/libero/libero_all.json (the official
single-domain stats, keys "libero_eef"/"libero_state_eef").
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments.eval_libero_plus import main
from experiments.eval_libero_official import InferenceLiberoOfficial


if __name__ == "__main__":
    main(inference_cls=InferenceLiberoOfficial)
