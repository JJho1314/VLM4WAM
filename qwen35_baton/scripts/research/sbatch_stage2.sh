#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM_baton/logs/%x-%j.out
# Stage 2 (teacher SigLIP2) on the qwen35 env plus a transformers-5 overlay
# (peft 0.21, diffusers 0.35.2, tensorboard 2.17.1). Env: NGPU, GLOBAL_BATCH,
# MAX_TRAIN_STEPS, BATON_OUTPUT_DIR (default from stage2_env.sh).
set -euo pipefail
ROOT=/data/user/jhe724/workspace/VLM4WAM_baton
set -a; . "$ROOT/stage2_env.sh"; set +a
if [[ -n "${OUTPUT_OVERRIDE:-}" ]]; then export BATON_OUTPUT_DIR="$OUTPUT_OVERRIDE"; fi
export CONDA_ENV=/data/user/jhe724/.conda/envs/qwen35
export PYTHON_BIN="$CONDA_ENV/bin/python" PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/data/user/jhe724/envs/overlay_peft_tf5
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" && cp -r /data/user/jhe724/.triton/cache/. "$TRITON_CACHE_DIR"/ 2>/dev/null || true
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NPROC_PER_NODE="${NGPU:-8}" GLOBAL_BATCH="${GLOBAL_BATCH:-128}" PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
cd "$ROOT/ge_act"
bash scripts/train_ltx_baton_stage2.sh configs/ltx_model/libero/action_model_libero_baton_stage2_hdf5.yaml
