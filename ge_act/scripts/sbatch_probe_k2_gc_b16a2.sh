#!/usr/bin/env bash
#SBATCH --job-name=probe_k2_gc_b16a2
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-probe-k2-gc-b16a2-%j.out

set -euo pipefail

GE_ACT_ROOT=${GE_ACT_ROOT:-/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act}
REPO_ROOT=$(dirname "$GE_ACT_ROOT")
PY=${PY:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/python}
TORCHRUN=${TORCHRUN:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/torchrun}
CONFIG=${CONFIG:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_action_k2_gc_b16a2_probe.yaml}
PROBE_OUTPUT_DIR=${PROBE_OUTPUT_DIR:-/data/user/jhe724/junjie/outputs/probe_joint_vlm_geact_action_k2_gc_b16a2}

export PYTHONPATH="$REPO_ROOT:$GE_ACT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ ! -x "$PY" ]]; then
    echo "HPC3 Python is not executable: $PY" >&2
    exit 2
fi
if [[ ! -x "$TORCHRUN" ]]; then
    echo "HPC3 torchrun is not executable: $TORCHRUN" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "K2 joint action config does not exist: $CONFIG" >&2
    exit 2
fi

MAIN_ARGS=(
    --config_file "$CONFIG"
    --max_train_steps 20
    --lr_warmup_steps_override 0
    --output_dir_override "$PROBE_OUTPUT_DIR"
)

cd "$GE_ACT_ROOT"

"$PY" scripts/predecode_lerobot_videos.py \
    --config "$CONFIG" \
    --verify-only

exec "$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py \
    "${MAIN_ARGS[@]}"
