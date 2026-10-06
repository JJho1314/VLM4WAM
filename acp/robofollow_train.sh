#!/usr/bin/env bash
# ACP launcher for RoboFollow GE-Act training: one 8-GPU Worker, or NNODES Workers with total_replicas=NNODES.
# Env: RUN_ID, CONFIG (yaml under ge_act/), MAX_TRAIN_STEPS, optional NGPU, NNODES, EXTRA_YAML (python dict literal).
set -euo pipefail
W=/data/users/junjie/workspace/VLM4WAM_robofollow_q35
D=/data/users/junjie/workspace/hpc3_jhe724
P=$D/.conda/envs/qwen35/bin/python
OUT=$D/outputs/robofollow_q35/$RUN_ID
LOG_DIR=$D/outputs/robofollow_q35/logs
NGPU=${NGPU:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${RANK:-${OMNI_WORKER_INDEX:-0}}}
mkdir -p "$OUT" "$LOG_DIR"
if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" -gt 0 ]; then LOG=$LOG_DIR/$RUN_ID.node$NODE_RANK.log; else LOG=$LOG_DIR/$RUN_ID.log; fi
exec > >(tee -a "$LOG") 2>&1
echo "[acp] RUN_ID=$RUN_ID host=$(hostname) nnodes=$NNODES node_rank=$NODE_RANK steps=$MAX_TRAIN_STEPS config=$CONFIG commit=$(git -C $W rev-parse --short HEAD 2>/dev/null) $(date -u +%FT%TZ)"
env | grep -E '^(MASTER_ADDR|MASTER_PORT|WORLD_SIZE|RANK|OMNI_WORKER_INDEX|PET_)' | sort || true
nvidia-smi -L; nproc
if [ "$NODE_RANK" -eq 0 ]; then
"$P" - <<PY
import yaml,ast,os
c=yaml.safe_load(open(os.path.join('$W/ge_act','$CONFIG')))
c['output_dir']='$OUT'
c.update(ast.literal_eval(os.environ.get('EXTRA_YAML','{}')))
yaml.safe_dump(c,open('$OUT/run.yaml','w'),sort_keys=False)
print('[acp] effective overrides',os.environ.get('EXTRA_YAML','{}'))
PY
else
  for i in $(seq 1 120); do [ -f "$OUT/run.yaml" ] && break; sleep 5; done
fi
export PYTHONPATH="$W:$D/envs/overlay_peft_tf5" HDF5_USE_FILE_LOCKING=FALSE TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=/tmp/triton-$RUN_ID; mkdir -p "$TRITON_CACHE_DIR"
cd "$W/ge_act"
if [ "$NNODES" -gt 1 ]; then
  DIST=(--nnodes "$NNODES" --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29500}")
else
  DIST=(--standalone)
fi
# torchrun overwrites RANK/WORLD_SIZE for its children with per-process values.
unset RANK WORLD_SIZE
exec "$P" -m torch.distributed.run "${DIST[@]}" --nproc_per_node=$NGPU main.py \
  --runner_class_path runner/robofollow_trainer.py --runner_class RoboFollowTrainer \
  --config_file "$OUT/run.yaml" --max_train_steps "$MAX_TRAIN_STEPS"
