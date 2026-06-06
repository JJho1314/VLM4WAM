#!/usr/bin/env bash
# Local watcher: wait for rsync tmux to finish, verify remote counts, then start ACD1-3 training.

set -euo pipefail
cd /data/LFT-W02_data/junjie/cosmos-predict2.5

SYNC_SESSION=${SYNC_SESSION:-droid480p_sync}
TRAIN_SESSION=${TRAIN_SESSION:-droid480_train}
REMOTE_REPO=/data/user/jhe724/workspace/cosmos-predict2.5

echo "WATCH_START $(date)"
while tmux has-session -t "$SYNC_SESSION" 2>/dev/null; do
  echo "WAIT_SYNC $(date)"
  sleep 60
done
echo "SYNC_SESSION_DONE $(date)"

rsync -avR \
  ./scripts/run_droid480_sweep_train_remote.sh \
  ./scripts/sbatch_droid480_sweep_train.sh \
  HPC3_jhe724:"$REMOTE_REPO"/

ssh -o BatchMode=yes HPC3_jhe724 "
  set -e
  count_train=\$(find /data/user/jhe724/workspace/datasets/droid_success_left_train_480x864/videos -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
  count_test=\$(find /data/user/jhe724/workspace/datasets/droid_success_left_test_480x864/videos -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
  count_failure=\$(find /data/user/jhe724/workspace/datasets/droid_failure_left_all_clean_480x864/videos -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
  echo remote_counts train=\$count_train test=\$count_test failure=\$count_failure
  test \"\$count_train\" -eq 52231
  test \"\$count_test\" -eq 1000
  test \"\$count_failure\" -eq 13180
"

ssh -o BatchMode=yes HPC3_jhe724 "
  set -e
  cd $REMOTE_REPO
  chmod +x scripts/run_droid480_sweep_train_remote.sh scripts/sbatch_droid480_sweep_train.sh
  sbatch scripts/sbatch_droid480_sweep_train.sh
"

echo "SLURM_TRAIN_SUBMITTED $(date)"
