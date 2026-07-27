# Qwen3.5 Baton checkpoint contract

Stage-1 writes `planner_topology.json` at the training output root before it
writes the first `step_*` checkpoint. This independently published file records
the ordered safetensors keys, shapes, dtypes, aliases, and their canonical
SHA-256. Every format-version-2 checkpoint metadata file binds to that hash.
Format-version-1 Baton checkpoints are intentionally incompatible with this
trust model and are rejected without mutating runtime state.

Keep `planner_topology.json` beside the checkpoint directories:

```text
output/
  planner_topology.json
  step_005000/
  step_010000/
```

Training resume and the frozen provider discover the file as
`CHECKPOINT.parent / "planner_topology.json"`. If a checkpoint directory is
relocated without its output root, copy the original trusted topology file and
pass its path explicitly through `expected_planner_topology` or
`--expected-planner-topology`. Do not regenerate the file from the relocated
checkpoint: that would make the checkpoint its own trust source.

The frozen provider's supported public inference dtypes are `torch.bfloat16`
and `torch.float32`; the visualization CLI exposes these as `bf16` and `fp32`.
