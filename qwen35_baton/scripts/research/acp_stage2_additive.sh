#!/usr/bin/env bash
# ACP launcher for Baton Stage 2 + additive semantic injection (one 8-GPU Worker).
# Env: RUN_ID, MAX_TRAIN_STEPS, optional INIT_OVERRIDE. Output and log go to AFS.
set -euo pipefail
ROOT=/data/users/junjie/workspace/VLM4WAM_baton
D=/data/users/junjie/workspace/hpc3_jhe724
OUT=$D/outputs/research_q35/$RUN_ID
LOG_DIR=$D/outputs/research_q35/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/$RUN_ID.log") 2>&1
echo "[acp] RUN_ID=$RUN_ID host=$(hostname) steps=$MAX_TRAIN_STEPS init=${INIT_OVERRIDE:-GE base} $(date -u +%FT%TZ)"
nvidia-smi -L
which gcc g++ || true
export SLURM_JOB_ID=$(( 60000 + RANDOM % 1000 ))
export TMPDIR=/tmp/$RUN_ID; mkdir -p "$TMPDIR"
export NGPU=${NGPU:-8} GLOBAL_BATCH=128 PER_DEVICE_BATCH=4
export BATON_TEACHER_GPU_PREPROCESS=1
# BATON_CONTROL=1 reproduces the original Stage 2 mainline (no research semantic opt-ins).
if [ "${BATON_CONTROL:-0}" != 1 ]; then
  export BATON_RESEARCH_SEMANTIC_GATE_MODE=zero_out BATON_RESEARCH_SEMANTIC_REZERO_OUT=1 \
    BATON_RESEARCH_SEMANTIC_LR=5e-4 BATON_RESEARCH_SEMANTIC_ADDITIVE=1
fi
# AFS (quarkfs FUSE) intermittently fails HDF5 file locks (errno 9); reads are read-only.
export HDF5_USE_FILE_LOCKING=FALSE
export OUTPUT_OVERRIDE=$OUT
cd "$ROOT"
exec bash sbatch_stage2.sh
