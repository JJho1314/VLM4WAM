#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM_baton/logs/%x-%A_%a.out
# E3 shard: SLURM_ARRAY_TASK_ID selects the shard. Env: CKPT, OUT, NUM_SAMPLES, NUM_SHARDS.
set -euo pipefail
ROOT=/data/user/jhe724/workspace/VLM4WAM_baton
set -a; . "$ROOT/stage2_env.sh"; set +a
export BATON_OUTPUT_DIR="$OUT/runner"
P=/data/user/jhe724/.conda/envs/qwen35/bin/python
export PYTHONPATH="$ROOT:$ROOT/ge_act:/data/user/jhe724/envs/overlay_peft_tf5"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID}"; mkdir -p "$TRITON_CACHE_DIR"
cd "$ROOT/ge_act"
CFG="${TMPDIR:-/tmp}/e3-${SLURM_JOB_ID}.yaml"
"$P" scripts/preflight_ltx_siglip2.py --config configs/ltx_model/libero/action_model_libero_baton_stage2_hdf5.yaml --materialize-output "$CFG"
"$P" "$ROOT/qwen35_baton/scripts/research/research_eval_stage2_modes.py" \
  --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
  --num-samples "${NUM_SAMPLES:-210}" --shard-index "$SLURM_ARRAY_TASK_ID" --num-shards "${NUM_SHARDS:-7}"
