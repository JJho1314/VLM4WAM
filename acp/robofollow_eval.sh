#!/usr/bin/env bash
# ACP launcher for the parallel RoboFollow benchmark (one 8-GPU Worker).
# Env: RUN (training RUN_ID), STEP, TAG (output name), LEVELS, optional PLANNER_MODE (disabled|predicted), SLOTS, ROUNDS, MODES.
set -euo pipefail
O=/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_q35
W=/data/users/junjie/workspace/VLM4WAM_robofollow_q35
mkdir -p $O/eval $O/logs
LOG=$O/logs/eval_$TAG.log
exec > >(tee -a "$LOG") 2>&1
echo "[eval] RUN=$RUN STEP=$STEP TAG=$TAG LEVELS=$LEVELS host=$(hostname) caps=${NVIDIA_DRIVER_CAPABILITIES:-unset} $(date -u +%FT%TZ)"
ls /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/ 2>&1 || true
# Fail fast when the container exposes no Vulkan rendering device.
/data/users/junjie/workspace/robofollow/env/bin/python "$W/scripts/rf_render_check.py"
CK=$(ls $O/$RUN/*/step_$STEP/diffusion_pytorch_model.safetensors | head -1)
cd "$W"
HDF5_USE_FILE_LOCKING=FALSE /data/users/junjie/workspace/hpc3_jhe724/.conda/envs/qwen35/bin/python scripts/rf_bench_eval.py \
  --config $O/$RUN/run.yaml --checkpoint "$CK" --out $O/eval/$TAG --planner-mode ${PLANNER_MODE:-disabled} \
  --modes ${MODES:-correct,shuffle} --levels $LEVELS --rounds ${ROUNDS:-1} --max-steps 10 --slots-per-gpu ${SLOTS:-3}
