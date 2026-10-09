#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=10:00:00
#SBATCH --output=/data/users/junjie/workspace/VLM4WAM_baton/logs/%x-%j.out
# E4 planner variant. Env: VARIANT (name), CURRENT_MODE (none/context/residual), SPATIAL_WEIGHT,
# optional MAX_STEPS, NGPU, PER_DEVICE_BATCH, GRAD_ACCUM, NUM_WORKERS.
set -euo pipefail
ROOT=/data/users/junjie/workspace/VLM4WAM_baton
W=/data/users/junjie/workspace/hpc3_jhe724/junjie/weights
A=/data/users/junjie/workspace/hpc3_jhe724/junjie/vlm4wam_joint_assets
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" && cp -r /data/users/junjie/workspace/hpc3_jhe724/.triton/cache/. "$TRITON_CACHE_DIR"/ 2>/dev/null || true
export CC=/usr/bin/gcc CXX=/usr/bin/g++
# Plain DDP replicates fp32 weights + AdamW (~37 GB/GPU); 4 samples/GPU OOMs on 80 GB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT:$ROOT/ge_act" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
cd "$ROOT"
/data/users/junjie/workspace/hpc3_jhe724/.conda/envs/qwen35/bin/python -m accelerate.commands.launch \
  --num_processes "${NGPU:-8}" --num_machines 1 --main_process_port "$((20000 + SLURM_JOB_ID % 20000))" --mixed_precision bf16 --dynamo_backend no \
  -m qwen35_baton.cli.research_train \
  --init-checkpoint /data/users/junjie/workspace/hpc3_jhe724/outputs/qwen35_baton_strict_acd1_18_ddp_30k/step_020000 \
  --qwen-path "$W/Qwen3.5-2B-baton-v1" \
  --siglip2-path "$W/siglip2-large-patch16-256" \
  --manifest /data/users/junjie/workspace/hpc3_jhe724/junjie/datasets/LIBERO-fastwam-hdf5/manifest.json \
  --stat-file "$ROOT/ge_act/configs/ltx_model/libero/libero_fastwam_mix.json" \
  --output-dir "/data/users/junjie/workspace/hpc3_jhe724/outputs/research_q35/e4_${VARIANT}" \
  --spatial-weight "${SPATIAL_WEIGHT:-0}" \
  --da3-ckpt "$A/DA3-LARGE-1.1" --da3-code-root "$A/Depth-Anything-3" \
  --max-steps "${MAX_STEPS:-5000}" \
  --per-device-batch "${PER_DEVICE_BATCH:-2}" --grad-accum "${GRAD_ACCUM:-8}" \
  --num-workers "${NUM_WORKERS:-8}" --log-every "${LOG_EVERY:-20}" --current-mode "${CURRENT_MODE:-none}" \
  --worker-malloc-trim --gpu-teacher-preprocess ${EXTRA_ARGS:-}
