# SDD ledger — plan: docs/superpowers/plans/2026-07-27-qwen35-baton-continuous-planner.md

Merge base: a02caf7
Workspace: /data/LFT-W02_data/junjie/workspace/VLM4WAM/.worktrees/qwen35-video-hindsight-grounding

Baseline: collection blocked in `/opt/miniconda3` before tests by missing `pytest-xdist`, `h5py`, root import path, and incompatible Transformers/PEFT. Existing named environments split Qwen3.5 (`qwen35`) and GE-Act (`ge-act`) dependencies; Task 7 owns the unified pinned environment contract. Task-scoped tests must record the interpreter used.

Task 1: fix round 1/5 (4 addressed, 0 open — strict model/hash provenance; pre-norm contract; integer token IDs; duplicate cursor names; commit e2d5049..385faad)
Task 1: complete (commits a02caf7..385faad, review clean)
Task 2: fix round 1/5 (2 addressed, 0 open — auditable HDF5 regression; cross-suite negative coverage; commit 1634abb..75953ad)
Task 2: complete (commits 385faad..75953ad, review clean)
Task 3: complete (commits 75953ad..fc2161d, review clean)
Task 4: minor (deferred): add direct production-geometry constructor/introspection coverage.
Task 4: minor (deferred): add explicit parent-module state_dict rejection test.
Task 4: minor (deferred): consider chunked/head-wise opt-in attention tracing to reduce visualization peak memory.
Task 4: complete (commits fc2161d..65df046, review clean)
Task 5: fix round 1/5 (4 addressed, 0 open — logits/cache-free multimodal base forward; per-owner overlap rejection; production forward contract; Sem-MLP/state_dict coverage; commit f0a7abf..2f4f6a0)
Task 5: complete (commits 65df046..2f4f6a0, review clean)
Task 6: fix round 1/5 (2 addressed, 0 open — safe FP16/BF16 loss arithmetic; finite low-precision output/backward regression; commit c52fbf6..559b173)
Task 6: complete (commits 2f4f6a0..559b173, review clean)
Task 7: fix round 1/5 (10 addressed, 3 open — Qwen3.5-2B contract; BF16 loss integration; raw once-per-update scheduler; non-tail accumulation; deterministic worker resume; fail-closed RNG/cursor checks; canonical optimizer identity; effective launcher overrides; seed order; SigLIP2 hash/defaults; commit e06090f..879ef4e)
Task 7: fix round 2/5 (2 addressed, 2 open — optimizer/scheduler current-LR binding; skipped-update cursor/scheduler semantics; slowest-rank throughput; JSONL recovery hardening; commit 879ef4e..68e09e7)
Task 7: fix round 3/5 (2 addressed, 0 open — durable checksummed JSONL envelope/reconciliation; bounded consecutive skipped-update liveness guard; commit 68e09e7..87cd7fe)
Task 7: complete (commits 559b173..87cd7fe, review clean)
Task 8: fix round 1/5 (3 addressed, 3 open — BF16 provider autocast; SigLIP2 provenance; strict instruction validation; planner topology preflight incomplete; metadata version transition missing; unsupported FP16 accepted; commit 542caa8..85a3c6a)
Task 8: fix round 2/5 (1 addressed, 2 open — format-v2 schema transition; independent topology publication/save/load/provider trust chain; BF16/FP32-only public dtype contract; concurrent trust-anchor overwrite and CUDA ordinal remain; commit 85a3c6a..908b1ef)
Task 8: fix round 3/5 (2 addressed, 1 open — create-once concurrent topology anchor; early canonical CPU/CUDA ordinal validation; shared-read permission/docs remain; commit 908b1ef..2e2a53c)
Task 8: fix round 4/5 (1 addressed, 0 open — shared-readable 0644 topology anchor; POSIX hard-link/fsync/relocation/fail-closed documentation; commit 2e2a53c..a04770a)
Task 8: complete (commits 87cd7fe..a04770a, review clean)
Task 9: complete (commits a04770a..4fa0cce, review clean)
Task 10: fix round 1/5 (6 addressed, 4 open — Baton HDF5 dispatch; Stage2→Stage3 initialization; exact training-state resume; concrete launcher materialization; paired validation; protected preflight regression; commit 7840545..8454349)
Task 10: fix round 2/5 (4 addressed, 1 open — worker-invariant stateless HDF5 sampling; exact runtime/checkpoint provenance equality; Stage1/Stage2 semantic artifact binding; strict complete Stage3 LTX loading; commit 8454349..e6df762)
Task 10: fix round 3/5 (1 addressed, 1 open — same-byte safetensors hash/deserialization and immutable snapshot manifest; commit e6df762..d6c5027)
Task 10: fix round 4/5 (1 addressed, 0 open — single sealed Stage2 envelope load eliminates split-envelope TOCTOU; commit d6c5027..4780812)
Task 10: complete (commits 4fa0cce..4780812, independent review approved; required gate 128 passed, HDF5 protected/functional gate 80 passed)
