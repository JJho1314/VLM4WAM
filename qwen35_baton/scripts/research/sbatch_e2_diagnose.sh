#!/usr/bin/env bash
#SBATCH --job-name=q35-e2-diag
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM_baton/logs/e2-diag-%j.out
set -euo pipefail
ROOT=/data/user/jhe724/workspace/VLM4WAM_baton
W=/data/user/jhe724/junjie/weights
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" && cp -r /data/user/jhe724/.triton/cache/. "$TRITON_CACHE_DIR"/ 2>/dev/null || true
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export PYTHONPATH="$ROOT:$ROOT/ge_act" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$ROOT"
if [[ -n "${RESEARCH_CKPT:-}" ]]; then
  CKPT_ARGS=(--research-checkpoint "$RESEARCH_CKPT")
else
  CKPT_ARGS=(--checkpoint /data/user/jhe724/outputs/qwen35_baton_strict_acd1_18_ddp_30k/step_020000)
fi
/data/user/jhe724/.conda/envs/qwen35/bin/python -m qwen35_baton.cli.diagnose_planner \
  "${CKPT_ARGS[@]}" \
  --qwen-path "$W/Qwen3.5-2B-baton-v1" \
  --siglip2-path "$W/siglip2-large-patch16-256" \
  --manifest /data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5/manifest.json \
  --stat-file "$ROOT/ge_act/configs/ltx_model/libero/libero_fastwam_mix.json" \
  --output "${OUTPUT_JSON:-/data/user/jhe724/outputs/research_q35/e2_planner_diagnostic_step020000.json}" \
  --num-episodes "${NUM_EPISODES:-400}"
