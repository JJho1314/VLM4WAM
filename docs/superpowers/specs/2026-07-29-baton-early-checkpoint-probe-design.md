# Baton Stage-1 Early Checkpoint Probe

## Goal

Verify the repaired DDP checkpoint path early in the ACD1-18 production run
instead of waiting until step 5,000.

## Design

- Save one additional checkpoint at optimizer step 20.
- Keep the existing periodic cadence at steps 5,000, 10,000, 15,000, 20,000,
  25,000, and 30,000.
- The step-20 artifact uses the exact same atomic save path, full AdamW state,
  scheduler state, distributed RNG state, cursor, metadata, hashes, and topology
  validation as every periodic checkpoint.
- After the step-20 directory is published, run a read-only resume preflight
  against it before treating the long run as healthy.
- Do not alter the model, dataset, loss, learning rate, BF16 mode, gradient
  checkpointing mode, per-device batch, gradient accumulation, or global batch.

## Failure Handling

If saving or loading step 20 fails, stop the run and preserve its log and
incomplete artifact for diagnosis. Do not continue until the checkpoint path is
fixed. If it succeeds, leave the training process running toward 30,000 steps.

## Verification

- Unit-test that the production checkpoint schedule is exactly step 20 plus the
  existing 5,000-step cadence.
- Run the checkpoint and training regression suites.
- On ACD1-18, confirm `step_000020` contains a complete manifest-valid checkpoint
  and that a fresh process can load it.
