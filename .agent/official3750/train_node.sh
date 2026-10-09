#!/usr/bin/env bash
set -euo pipefail
test "${SLURM_NNODES:?}" -eq 2
test "${SLURM_NODEID:?}" = 0 || test "$SLURM_NODEID" = 1
exec /data/user/jhe724/.conda/envs/qwen35/bin/python -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=8 --node_rank="$SLURM_NODEID" \
  --master_addr="${MASTER_ADDR:?}" --master_port="${MASTER_PORT:?}" \
  main.py --runner_class_path runner/robofollow_trainer.py --runner_class RoboFollowTrainer \
  --config_file configs/ltx_model/robofollow/q35_text_p256_official3750_53k_hpc3.yaml
