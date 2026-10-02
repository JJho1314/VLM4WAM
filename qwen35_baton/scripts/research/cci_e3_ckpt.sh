#!/bin/bash
# E3 on a Stage-2 checkpoint (runs on Ola_2208). Env: CKPT, OUT, VIEWS=main|both|control|planner.
# planner: adds planner* modes whose guidance comes from the frozen Qwen3.5 baton planner.
set -euo pipefail
cd /data/users/junjie/workspace/VLM4WAM_baton
D=/data/users/junjie/workspace/hpc3_jhe724
export HDF5_USE_FILE_LOCKING=FALSE NUM_SAMPLES=${NUM_SAMPLES:-210} NUM_SHARDS=${NUM_SHARDS:-2} BATON_RESEARCH_TRACE_SEMANTIC=1
if [ "${VIEWS:-both}" != control ]; then
  export BATON_RESEARCH_SEMANTIC_GATE_MODE=zero_out BATON_RESEARCH_SEMANTIC_ADDITIVE=1
fi
if [ "${VIEWS:-}" = main ]; then export BATON_RESEARCH_SEMANTIC_VIEWS=main; fi
if [ "${VIEWS:-}" = planner ]; then
  # Stage-2 teacher config; planner* modes swap in frozen planner predictions.
  export E3_PLANNER_CHECKPOINT=$D/outputs/qwen35_baton_strict_acd1_18_ddp_30k/step_020000
  export E3_PLANNER_QWEN_PATH=$D/junjie/weights/Qwen3.5-2B-baton-v1
  export E3_MODES=${E3_MODES:-teacher,semantic_disabled,planner,planner_wrist_masked,planner_main_masked}
fi
mkdir -p "$OUT/logs"
bash qwen35_baton/scripts/research/cci_e3_eval.sh
