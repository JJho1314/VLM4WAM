# Joint VLM + GE-Act Micro-Batch 2 Design

## Goal

Increase per-GPU utilization on the 8×H100 80GB node without changing the effective global batch size or the 30,000-step training schedule.

## Configuration

- Increase `batch_size` from 1 to 2 per GPU.
- Reduce `gradient_accumulation_steps` from 16 to 8.
- Keep 8 distributed workers, giving an effective global batch size of `2 × 8 × 8 = 128`.
- Preserve the existing optimizer, learning-rate schedule, predecoded RGB dataset, gradient checkpointing, ZeRO-2 configuration, and checkpoint steps 20,000, 25,000, and 30,000.

## Deployment Flow

1. Stop the current 30332 training process cleanly.
2. Synchronize the changed configuration to the clean deployment directory on 30332.
3. Run a short 8-GPU smoke test with the production micro-batch and accumulation settings.
4. Accept the change only if the smoke test completes without OOM, non-finite loss, or worker failure.
5. Restart the 30,000-step training from step 0 and verify the first optimizer steps in the persistent log.

## Success Criteria

- All eight GPUs participate in training.
- Peak memory remains below the 80GB device capacity with practical headroom.
- Training loss is finite.
- Sustained samples per second is no worse than the micro-batch 1 baseline.
- The effective global batch size remains 128.

## Rollback

If the smoke test fails or throughput regresses, restore `batch_size: 1` and `gradient_accumulation_steps: 16`, then restart from the original configuration.
