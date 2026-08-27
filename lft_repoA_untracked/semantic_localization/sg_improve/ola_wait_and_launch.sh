#!/usr/bin/env bash
# Wait for the shared H100s to free up, then claim them for the 15k-step full fine-tune.
# Another user's eval_policy.py jobs are holding ~40GB on 7 of the 8 cards; rather than squeezing in
# next to them (which risks OOM-ing their run and ours) this polls until enough cards are actually
# free and then launches immediately, so the moment they finish the box is ours.
set -u
ROOT=/home/6fcb109c-77d2-48/workspace/VLM4WAM
PY=/home/6fcb109c-77d2-48/miniforge3/envs/qwen35/bin/python   # absolute: `~` does not expand under setsid/nohup
LOG=$ROOT/logs/wait_launch.log
NEED_GPUS=${NEED_GPUS:-8}
FREE_MB=${FREE_MB:-70000}      # a card counts as free above this
STABLE=${STABLE:-3}            # consecutive polls that must agree (ignore transient dips)
POLL=${POLL:-120}

echo "$(date) watcher start: need $NEED_GPUS gpus with >${FREE_MB}MiB free, $STABLE stable polls" >> "$LOG"
ok=0
while true; do
  n=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v t="$FREE_MB" '$1>t{c++} END{print c+0}')
  if [ "$n" -ge "$NEED_GPUS" ]; then
    ok=$((ok+1))
    echo "$(date) $n free gpus (stable $ok/$STABLE)" >> "$LOG"
  else
    [ "$ok" -gt 0 ] && echo "$(date) only $n free, reset" >> "$LOG"
    ok=0
  fi
  [ "$ok" -ge "$STABLE" ] && break
  sleep "$POLL"
done

echo "$(date) LAUNCHING ${NEED_GPUS}-gpu full fine-tune" >> "$LOG"
cd "$ROOT" || exit 2
setsid nohup env \
  MAX_STEPS=15000 SAVE_STEPS=5000 EVAL_STEPS=1000 FULL_FT=1 BATCH_SIZE=4 \
  LR=1e-5 HEAD_LR=1e-4 WARMUP_STEPS=300 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OUT_DIR=$ROOT/runs/qwen35_discrete_ola \
  "$PY" -m torch.distributed.run --nproc_per_node="$NEED_GPUS" --master_port=29537 \
  code/sg_improve/ola_train_codes.py > "$ROOT/logs/train.log" 2>&1 < /dev/null &
echo "$(date) launched pid $!" >> "$LOG"
