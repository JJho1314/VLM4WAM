#!/usr/bin/env bash
# E3 on a Slurm-less container (Qianhai CCI): runs NUM_SHARDS shards over the
# visible GPUs, one process per GPU, then merges. Env: CKPT, OUT, and optionally
# DATA_ROOT (mirror of the HPC3 /data/user/jhe724 tree), NUM_SAMPLES, NUM_SHARDS.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/data/users/junjie/workspace/hpc3_jhe724}
set -a; . "$ROOT/stage2_env.sh"; set +a
export BATON_OUTPUT_DIR="$OUT/runner"
P=$DATA_ROOT/.conda/envs/qwen35/bin/python
export PYTHONPATH="$ROOT:$ROOT/ge_act:$DATA_ROOT/envs/overlay_peft_tf5"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
NUM_SHARDS=${NUM_SHARDS:-7}
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p "$OUT/logs"
cd "$ROOT/ge_act"
CFG="$OUT/e3-config.yaml"
"$P" scripts/preflight_ltx_siglip2.py \
  --config configs/ltx_model/libero/action_model_libero_baton_stage2_hdf5.yaml \
  --materialize-output "$CFG"
for gpu in $(seq 0 $((NGPU - 1))); do
  (
    for shard in $(seq "$gpu" "$NGPU" $((NUM_SHARDS - 1))); do
      CUDA_VISIBLE_DEVICES=$gpu SLURM_JOB_ID=$((40000 + shard)) \
      TRITON_CACHE_DIR=/tmp/triton-e3-$shard \
        "$P" "$ROOT/qwen35_baton/scripts/research/research_eval_stage2_modes.py" \
        --config "$CFG" --checkpoint "$CKPT" --output-dir "$OUT" \
        --num-samples "${NUM_SAMPLES:-210}" --shard-index "$shard" --num-shards "$NUM_SHARDS" \
        > "$OUT/logs/shard$shard.log" 2>&1
    done
  ) &
done
wait
"$P" "$ROOT/qwen35_baton/scripts/research/research_eval_stage2_modes.py" --merge --output-dir "$OUT" \
  > "$OUT/logs/merge.log"
echo "E3 done: $OUT/summary.json"
