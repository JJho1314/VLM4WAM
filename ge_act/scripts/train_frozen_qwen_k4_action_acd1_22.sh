#!/usr/bin/env bash
set -euo pipefail

GE_ACT_ROOT=${GE_ACT_ROOT:-/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/ge_act}
REPO_ROOT=$(dirname "$GE_ACT_ROOT")
PY=${PY:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/python}
TORCHRUN=${TORCHRUN:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/torchrun}
CONFIG=${CONFIG:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_frozen_qwen_k4_action_30k_hpc3.yaml}
RUN_KIND=${RUN_KIND:-formal}

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
    echo "Frozen-Qwen K4 config does not exist: $CONFIG" >&2
    exit 2
fi

MAIN_ARGS=(--config_file "$CONFIG")
if [[ "$RUN_KIND" == "smoke8" ]]; then
    SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR:-/data/user/jhe724/junjie/outputs/smoke_frozen_qwen_k4_action_${SLURM_JOB_ID:-manual}}
    MAIN_ARGS+=(
        --max_train_steps 2
        --lr_warmup_steps_override 0
        --output_dir_override "$SMOKE_OUTPUT_DIR"
    )
elif [[ "$RUN_KIND" != "formal" ]]; then
    echo "RUN_KIND must be formal or smoke8, got '$RUN_KIND'" >&2
    exit 2
fi

cd "$GE_ACT_ROOT"

"$PY" scripts/predecode_lerobot_videos.py \
    --config "$CONFIG" \
    --verify-only

"$PY" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
errors = []
semantic = config.get("semantic_plan", {})
model = config.get("diffusion_model", {}).get("config", {})
joint = config.get("joint_training", {})
if isinstance(joint, dict) and joint.get("enabled", False):
    errors.append("joint_training must be absent or disabled")
if semantic.get("source") != "vlm_planner":
    errors.append("semantic_plan.source must be vlm_planner")
if semantic.get("keyframe_indices") != [2, 4, 6, 8]:
    errors.append("semantic K4 offsets must be [2, 4, 6, 8]")
if model.get("semantic_plan_num_keyframes") != 4:
    errors.append("LTX must be configured for four semantic keyframes")
if model.get("semantic_plan_num_views") != 2:
    errors.append("LTX must preserve main/wrist semantic views")
if model.get("semantic_plan_cross_attention_blocks") != list(range(28)):
    errors.append("semantic cross-attention must cover all 28 LTX blocks")
if not model.get("action_expert", False):
    errors.append("GE-Act action expert must be enabled")
if not config.get("return_video", False) or not config.get("return_action", False):
    errors.append("both video and action objectives must be enabled")
if config.get("batch_size", 0) * config.get("gradient_accumulation_steps", 0) * 8 != 128:
    errors.append("global batch must be 128")

checkpoint = Path(semantic.get("planner_checkpoint", ""))
meta_path = checkpoint / "planner_meta.json"
if not meta_path.is_file():
    errors.append(f"missing planner metadata: {meta_path}")
else:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("num_keyframes") != 4:
        errors.append("planner checkpoint is not K4")
    if meta.get("future_keyframe_offsets") != [2, 4, 6, 8]:
        errors.append("planner checkpoint offsets are not [2, 4, 6, 8]")

for path in (
    Path(config["pretrained_model_name_or_path"]),
    Path(config["diffusion_model"]["model_path"]),
    checkpoint,
    Path(config["data"]["train"]["predecoded_video_root"]),
):
    if not path.exists():
        errors.append(f"missing required path: {path}")

if errors:
    print("Frozen-Qwen K4 preflight failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print("Frozen-Qwen K4 preflight passed")
PY

exec "$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py \
    "${MAIN_ARGS[@]}"
