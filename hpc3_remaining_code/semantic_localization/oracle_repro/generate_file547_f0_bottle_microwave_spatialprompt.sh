#!/usr/bin/env bash
#SBATCH --job-name=cosmos-file547-f0-spatial
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail

VLM4WAM_ROOT=/data/user/jhe724/workspace/VLM4WAM
COSMOS_ROOT="$VLM4WAM_ROOT/third_party/cosmos-predict2.5"
COSMOS_VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
WEIGHTS_DIR=/data/user/jhe724/workspace/weights
STAGE="$VLM4WAM_ROOT/eval_inputs_oracle_iter3000_file547_firstframe_bottle_microwave_spatialprompt_REPRO"
OUT="$VLM4WAM_ROOT/eval_results_oracle_iter3000_file547_firstframe_bottle_microwave_spatialprompt_REPRO"

for required in \
  "$STAGE/yc74616_s0.json" \
  "$STAGE/first_frames/yc74616_f0.png" \
  "$STAGE/plans/yc74616_s0_oracle.pt" \
  "$VLM4WAM_ROOT/outputs/cosmos_semantic_plan/converted_iter3000/model_ema_bf16.pt" \
  "$COSMOS_VENV/bin/python"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing required artifact: $required" >&2
    exit 2
  fi
done

ORIG="$STAGE" \
OUT="$OUT" \
COSMOS_ROOT="$COSMOS_ROOT" \
COSMOS_VENV="$COSMOS_VENV" \
WEIGHTS_DIR="$WEIGHTS_DIR" \
COSMOS_NUM_FRAMES=49 \
SEMANTIC_PLAN_NUM_KEYFRAMES=5 \
SEMANTIC_PLAN_SPATIAL_GRID=0 \
bash "$VLM4WAM_ROOT/semantic_localization/oracle_repro/reproduce_oracle_yc.sh"
