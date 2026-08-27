#!/usr/bin/env bash
#SBATCH --job-name=cosmos-oracle-eval
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-oracle-eval-yc-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-oracle-eval-yc-%j.err
set -euo pipefail

VLM4WAM_ROOT=/data/user/jhe724/workspace/VLM4WAM
COSMOS_ROOT=$VLM4WAM_ROOT/third_party/cosmos-predict2.5
cd "$COSMOS_ROOT"
mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 2>/dev/null || true   # cuda/nccl dropped: use venv-bundled CUDA/cuDNN (avoids SUBLIBRARY_LOADING_FAILED)
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
# conda activate cosmos_fw 2>/dev/null || true
COSMOS_VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
if [[ -x "$COSMOS_VENV/bin/python" ]]; then PY="$COSMOS_VENV/bin/python"; else PY="$COSMOS_ROOT/.venv/bin/python"; fi
if [[ -d "$COSMOS_VENV" ]]; then
  export VIRTUAL_ENV="$COSMOS_VENV"; export PATH="$COSMOS_VENV/bin:$PATH"
  NV="$COSMOS_VENV/lib/python3.10/site-packages/nvidia"
  [[ -d "$NV" ]] && export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/cufft/lib:$NV/curand/lib:$NV/nccl/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$COSMOS_ROOT:${PYTHONPATH:-}"
export COSMOS_HF_LOCAL_DIRS=/data/user/jhe724/workspace/weights
export COSMOS_LOCAL_MODEL_DIR=/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B
export COSMOS_DISABLE_CUDNN_SDPA=1
export COSMOS_DISABLE_CUDNN_CONV=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- MUST match training so the checkpoint architecture matches ----
export COSMOS_NUM_FRAMES=49
export COSMOS_SAVE_FPS=10  # train/infer consistent: dataset is 10Hz
export VIDEO_HEIGHT=320
export VIDEO_WIDTH=576
export SEMANTIC_PLAN_DIM=1152
export SEMANTIC_PLAN_SPATIAL_GRID=0
export SEMANTIC_PLAN_NUM_KEYFRAMES=5
export SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES=5
export SEMANTIC_PLAN_ONLINE=1
export SEMANTIC_PLAN_ONLINE_NUM_KEYFRAMES=5
export SEMANTIC_PLAN_ONLINE_ENCODER_PATH=$VLM4WAM_ROOT/third_party/siglip2-so400m-patch14-384
export SEMANTIC_PLAN_EPISODE_SAMPLING=1
export SEMANTIC_PLAN_DROPOUT_PROB=0.15

DR=/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576
CKPT_DISTCP=/data/user/jhe724/workspace/VLM4WAM/outputs/cosmos_semantic_plan/cosmos_predict_v2p5/semantic_plan_video2world/2b_semplan_gt_online_native_k5_1fps_dp015_320x576_49f_epsample_3000/checkpoints/iter_000003000
OUT=$VLM4WAM_ROOT/eval_results_oracle_iter3000_yellowcarrot_$(date +%Y%m%d_%H%M%S)
PLAN_DIR=$OUT/plans; FRAME_DIR=$OUT/first_frames; GT_DIR=$OUT/gt_clips
mkdir -p "$OUT" "$PLAN_DIR" "$FRAME_DIR" "$GT_DIR"
ENCODER=$SEMANTIC_PLAN_ONLINE_ENCODER_PATH
GUIDANCE=${GUIDANCE:-7}
NUM_STEPS=${NUM_STEPS:-35}

# ---- 1) convert distcp -> consolidated EMA bf16 .pt (once) ----
CKPT_PT_DIR=$VLM4WAM_ROOT/outputs/cosmos_semantic_plan/converted_iter3000
CKPT_PT=$CKPT_PT_DIR/model_ema_bf16.pt
mkdir -p "$CKPT_PT_DIR"
if [[ ! -f "$CKPT_PT" ]]; then
  echo "=== converting distcp -> pt (--ema) ==="
  "$PY" scripts/convert_distcp_to_pt.py "$CKPT_DISTCP/model" "$CKPT_PT_DIR" --ema
fi
ls -lh "$CKPT_PT_DIR"
[[ -f "$CKPT_PT" ]] || { echo "ERROR: $CKPT_PT missing after convert" >&2; exit 2; }

# ---- 2) per-episode: extract start-frame PNG + GT 49f clip + export GT plan + sample.json ----
# entries: "<stem> <start>"  (window = [start, start+48], stride 1; start = motion-range start)
YCVID=/data/user/jhe724/workspace/VLM4WAM/yc74616_work/yc74616_49f_10fps_576x320.mp4
YCCAP=/data/user/jhe724/workspace/VLM4WAM/yc74616_work/yc74616_caption.txt
EPISODES=(
  "yc74616 0"
)

SAMPLES=()
for entry in "${EPISODES[@]}"; do
  set -- $entry; stem=$1; start=$2
  video="$YCVID"
  cap="$YCCAP"
  png="$FRAME_DIR/${stem}_f${start}.png"
  gtclip="$GT_DIR/${stem}_s${start}_gt.mp4"
  plan="$PLAN_DIR/${stem}_s${start}_oracle.pt"
  spec="$OUT/${stem}_s${start}.json"

  echo "=== [$stem start=$start] extract first frame + GT clip ==="
  "$PY" - "$video" "$start" "$png" "$gtclip" <<'PY'
import sys, decord, numpy as np, imageio.v2 as imageio
video, start, png, gtclip = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
vr = decord.VideoReader(video, ctx=decord.cpu(0))
frames = vr.get_batch([start + i for i in range(49)]).asnumpy()  # (49,H,W,3) uint8
imageio.imwrite(png, frames[0])
imageio.mimwrite(gtclip, list(frames), fps=10, quality=8)
print("first_frame", png, "gt_clip", gtclip, "shape", frames.shape)
PY

  echo "=== [$stem] export GT semantic plan ==="
  "$PY" scripts/export_semantic_plan_from_video.py \
    --video "$video" --start "$start" --num-frames 49 --frame-stride 1 \
    --num-keyframes 5 --grid-size 0 --encoder-path "$ENCODER" --output "$plan"

  "$PY" - "$spec" "$stem" "$start" "$png" "$cap" "$plan" "$GUIDANCE" "$NUM_STEPS" <<'PY'
import json, sys
spec, stem, start, png, cap, plan, guidance, steps = sys.argv[1:9]
json.dump({
  "name": f"{stem}_s{start}_oracle",
  "inference_type": "image2world",
  "input_path": png,
  "semantic_plan_path": plan,
  "prompt": open(cap).read().strip(),
  "resolution": "none",
  "num_output_frames": 49,
  "num_steps": int(steps),
  "guidance": int(guidance),
  "seed": 0,
}, open(spec, "w"), indent=2)
print("wrote", spec)
PY
  SAMPLES+=("$spec")
done

# ---- 3) run inference (tyro OmitArgPrefixes: no `setup.` prefix) ----
echo "=== running inference on ${#SAMPLES[@]} samples (guidance=$GUIDANCE steps=$NUM_STEPS) ==="
"$PY" examples/inference.py \
  -i "${SAMPLES[@]}" \
  --experiment predict2_video2world_training_2b_droid_semantic_plan_320x576_93f \
  --checkpoint-path "$CKPT_PT" \
  --config-file cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --output-dir "$OUT" \
  --disable-guardrails 2>&1 | tee "$OUT/inference.log"

# ---- 4) sanity: were semantic-plan weights actually loaded? ----
echo "=== missing/unexpected keys mentioning semantic_plan (should be none) ==="
grep -iE "missing_keys|unexpected_keys|semantic_plan" "$OUT/inference.log" | grep -iE "semantic_plan|missing|unexpected" | head -20 || true

echo "=== DONE. outputs in $OUT ==="
ls -lh "$OUT"/*.mp4 2>/dev/null
echo "GT reference clips in $GT_DIR ; generated videos above"
