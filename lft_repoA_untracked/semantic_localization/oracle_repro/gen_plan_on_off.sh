#!/usr/bin/env bash
# Oracle plan ON vs OFF on the iter3000 SG-WAM -- clean ablation, correct settings.
#
# WHY A SHELL SCRIPT AND NOT A DIRECT generate_vid2world() CALL
#   A first attempt hand-called pipe.generate_vid2world(...) and silently produced 93-frame clips
#   with wrong content, because several arguments that examples/inference.py fills in from the JSON
#   spec were missing or guessed: resolution must be "none" (use the model's native size, NOT an
#   explicit "320,576"), the long negative_prompt matters, num_latent_conditional_frames comes from
#   the spec, and num_video_frames alone did not pin the length. Driving inference.py with a JSON
#   spec -- the exact path that reproduced the original result -- removes all of that guesswork.
#
#   plan_on  : spec identical to the original oracle run (semantic_plan_path set)
#   plan_off : same spec with semantic_plan_path removed -> RGB + text only
#   Everything else (prompt, negative prompt, seed 0, 35 steps, guidance 7, 49 frames) is byte
#   identical between the two, so the difference is attributable to the plan alone.
set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM}
COSMOS_ROOT=${COSMOS_ROOT:-$VLM4WAM_ROOT/cosmos-predict2.5}
COSMOS_VENV=${COSMOS_VENV:-/data/LFT-W02_data/junjie/cosmos-predict2.5/.venv}
WEIGHTS_DIR=${WEIGHTS_DIR:-/data/LFT-W02_data/junjie/weights}
REPRO=${REPRO:-$VLM4WAM_ROOT/semantic_localization/oracle_repro}
CKPT_PT=${CKPT_PT:-$WEIGHTS_DIR/cosmos_semantic_plan_iter3000/model_ema_bf16.pt}
OUT=${OUT:-$REPRO/plan_on_off}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_semantic_plan_320x576_93f}
GPU=${CUDA_VISIBLE_DEVICES:-1}

cd "$COSMOS_ROOT"; mkdir -p "$OUT"
PY="$COSMOS_VENV/bin/python"
export VIRTUAL_ENV="$COSMOS_VENV"; export PATH="$COSMOS_VENV/bin:$PATH"
NV="$COSMOS_VENV/lib/python3.10/site-packages/nvidia"
[ -d "$NV" ] && export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/cufft/lib:$NV/curand/lib:$NV/nccl/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COSMOS_ROOT:${PYTHONPATH:-}"
export COSMOS_HF_LOCAL_DIRS="$WEIGHTS_DIR"
export COSMOS_LOCAL_MODEL_DIR="$WEIGHTS_DIR/Cosmos-Predict2.5-2B"
export SEMANTIC_PLAN_ONLINE_ENCODER_PATH=${SEMANTIC_PLAN_ONLINE_ENCODER_PATH:-$VLM4WAM_ROOT/third_party/siglip2-so400m-patch14-384}
# iter3000 was trained native k5 / 27x27 = 3645 plan tokens; this checkout defaults to 6 keyframes /
# grid 9 (486), which silently resamples the plan into a layout the checkpoint never saw.
# Clip length comes from the model config, NOT the spec: generate_vid2world overrides
# num_video_frames with tokenizer.get_pixel_num_frames(config.state_t). The original run set
# COSMOS_NUM_FRAMES=49 -> state_t=13 -> 49 frames; the repo default of 93 -> state_t=24 makes
# the model extrapolate to double length, far outside what iter3000 was trained on.
export COSMOS_NUM_FRAMES=${COSMOS_NUM_FRAMES:-49}
export SEMANTIC_PLAN_NUM_KEYFRAMES=${SEMANTIC_PLAN_NUM_KEYFRAMES:-5}
export SEMANTIC_PLAN_SPATIAL_GRID=${SEMANTIC_PLAN_SPATIAL_GRID:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

SRC_SPEC=${SRC_SPEC:-$REPRO/yc74616_s0.json}
[ -f "$SRC_SPEC" ] || { echo "ERROR: source spec not found: $SRC_SPEC" >&2; exit 2; }

# derive the two specs from the validated original: same everything, plan present vs absent
"$PY" - "$SRC_SPEC" "$OUT" "$REPRO" <<'PYEOF'
import json, sys, os
src, out, repro = sys.argv[1], sys.argv[2], sys.argv[3]
s = json.load(open(src))
s["input_path"] = os.path.join(repro, "yc74616_f0.png")
on = dict(s, name="plan_on", semantic_plan_path=os.path.join(repro, "yc74616_s0_oracle.pt"))
off = {k: v for k, v in s.items() if k != "semantic_plan_path"}
off["name"] = "plan_off"
json.dump(on, open(os.path.join(out, "spec_plan_on.json"), "w"), indent=2)
json.dump(off, open(os.path.join(out, "spec_plan_off.json"), "w"), indent=2)
print("plan_on :", json.dumps(on)[:200])
print("plan_off:", json.dumps(off)[:200])
PYEOF

for tag in plan_on plan_off; do
  echo "=== generating $tag ==="
  "$PY" examples/inference.py \
    -i "$OUT/spec_$tag.json" \
    --experiment "$EXPERIMENT" \
    --checkpoint-path "$CKPT_PT" \
    --config-file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --output-dir "$OUT" \
    --disable-guardrails 2>&1 | tail -3
done

echo "=== outputs ==="
ls -lh "$OUT"/*.mp4 2>/dev/null
"$PY" - "$OUT" <<'PYEOF'
import sys, glob, numpy as np, av
out = sys.argv[1]
def load(p):
    c = av.open(p); v = [np.asarray(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]; c.close()
    return np.stack(v).astype(np.float32)
try:
    on, off = load(f"{out}/plan_on.mp4"), load(f"{out}/plan_off.mp4")
except Exception as e:
    print("compare skipped:", e); raise SystemExit
n = min(len(on), len(off)); on, off = on[:n], off[:n]
d = np.abs(on - off).mean(); ref = np.abs(on[1:] - on[:-1]).mean()
lines = [f"frames: plan_on={len(on)} plan_off={len(off)}",
         f"|plan_on - plan_off| mean = {d:.3f}/255  PSNR = {10*np.log10(255**2/((on-off)**2).mean()):.1f} dB",
         f"(scale) within-clip frame-to-frame motion = {ref:.3f}/255   ratio = {d/ref:.2f}x"]
open(f"{out}/result.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF
echo "=== DONE -> $OUT ==="
