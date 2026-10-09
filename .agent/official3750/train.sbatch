#!/usr/bin/env bash
#SBATCH --job-name=rf3750_text_ac32_53k
#SBATCH --partition=acd_u
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1920G
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/jhe724/outputs/robofollow_q35/official3750_text_53k_ac32_bs128/logs/slurm-%j.out
#SBATCH --error=/data/user/jhe724/outputs/robofollow_q35/official3750_text_53k_ac32_bs128/logs/slurm-%j.err
set -euo pipefail
cd /data/user/jhe724/workspace/VLM4WAM_robofollow_official3750
test "$(id -un)" = jhe724
test -f /data/user/jhe724/outputs/robofollow_q35/data_official3750/DATA_READY
sha256sum --status -c .agent/official3750/code.sha256
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/data/user/jhe724/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml research .agent/official3750/check_data.py
test "${SLURM_NNODES:?}" -eq 2
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | sed -n '1p')"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
exec srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 \
  /data/user/jhe724/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml train
