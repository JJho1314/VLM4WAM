#!/usr/bin/env bash
#SBATCH --job-name=qwen35-baton-stage1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_GPUS=8
exec "${SCRIPT_DIR}/train_semantic_planner.sh" "$@"
