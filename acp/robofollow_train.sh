#!/usr/bin/env bash
# ACP launcher for RoboFollow GE-Act training (one 8-GPU Worker).
# Env: RUN_ID, CONFIG (yaml under ge_act/configs or absolute), MAX_TRAIN_STEPS, optional NGPU, EXTRA_YAML (python dict literal of top-level overrides).
set -euo pipefail
W=/data/users/junjie/workspace/VLM4WAM_robofollow_q35
D=/data/users/junjie/workspace/hpc3_jhe724
P=$D/.conda/envs/qwen35/bin/python
OUT=$D/outputs/robofollow_q35/$RUN_ID
LOG_DIR=$D/outputs/robofollow_q35/logs
mkdir -p "$OUT" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/$RUN_ID.log") 2>&1
NGPU=${NGPU:-8}
echo "[acp] RUN_ID=$RUN_ID host=$(hostname) steps=$MAX_TRAIN_STEPS config=$CONFIG commit=$(git -C $W rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L; nproc
# Freeze the effective config next to the outputs.
"$P" - <<PY
import yaml,ast,os
c=yaml.safe_load(open(os.path.join('$W/ge_act','$CONFIG')))
c['output_dir']='$OUT'
c.update(ast.literal_eval(os.environ.get('EXTRA_YAML','{}')))
yaml.safe_dump(c,open('$OUT/run.yaml','w'),sort_keys=False)
print('[acp] effective overrides',os.environ.get('EXTRA_YAML','{}'))
PY
export PYTHONPATH="$W:$D/envs/overlay_peft_tf5" HDF5_USE_FILE_LOCKING=FALSE TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=/tmp/triton-$RUN_ID; mkdir -p "$TRITON_CACHE_DIR"
cd "$W/ge_act"
exec "$P" -m torch.distributed.run --standalone --nproc_per_node=$NGPU main.py \
  --runner_class_path runner/robofollow_trainer.py --runner_class RoboFollowTrainer \
  --config_file "$OUT/run.yaml" --max_train_steps "$MAX_TRAIN_STEPS"
