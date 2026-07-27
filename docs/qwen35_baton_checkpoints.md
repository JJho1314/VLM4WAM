# Qwen3.5 Baton checkpoint contract

Stage-1 writes `planner_topology.json` at the training output root before it
writes the first `step_*` checkpoint. This independently published file records
the ordered safetensors keys, shapes, dtypes, aliases, and their canonical
SHA-256. Every format-version-2 checkpoint metadata file binds to that hash.
Format-version-1 Baton checkpoints are intentionally incompatible with this
trust model and are rejected without mutating runtime state.

Topology publication requires a POSIX-like filesystem that supports atomic
hard links within one filesystem and directory `fsync`. The publisher creates
and fsyncs a private temporary file in the output directory, sets its mode to
`0644`, atomically hard-links it to `planner_topology.json` only when that name
is absent, and fsyncs the directory. The output directory must therefore allow
the training account to create, chmod, hard-link, unlink, and fsync entries;
its directory permissions must also allow the intended shared HPC accounts to
traverse and read the resulting `0644` topology file.

An unsupported filesystem or a chmod, hard-link, permission, or fsync failure
intentionally aborts publication. There is no unsafe `replace` fallback.
Private temporary files are removed where possible, and a preexisting anchor
is never overwritten. An identical preexisting anchor is accepted only when
its mode is already `0644`.

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
checkpoint or replace the original anchor: that would make the checkpoint its
own trust source.

The frozen provider's supported public inference dtypes are `torch.bfloat16`
and `torch.float32`; the visualization CLI exposes these as `bf16` and `fp32`.
