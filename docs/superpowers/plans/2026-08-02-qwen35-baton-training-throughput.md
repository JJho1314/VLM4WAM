# Qwen3.5 Baton Training Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase eight-GPU WorldArena Stage-1 throughput without changing the Baton loss, four-frame 1,024-token output geometry, global batch 128, sample order, or online SigLIP2 target semantics.

**Architecture:** First separate asynchronous telemetry and debug validation from the production hot path. Then preprocess exact SigLIP2 pixels in workers and overlap pinned host-to-device transfer. A fixed-global-batch benchmark selects the largest stable DDP microbatch; optional bf16 ZeRO-2 is added only as a memory-enabling strategy with explicit checkpoint compatibility.

**Tech Stack:** Python 3.12, PyTorch, Accelerate, Transformers Qwen3.5/SigLIP2, DeepSpeed ZeRO-2, pytest, HDF5, Slurm/torchrun.

## Global Constraints

- Global batch size is exactly 128 on eight GPUs.
- Four future frames and 256 SigLIP2 tokens per frame remain unchanged.
- Qwen3.5 and Baton query tower remain jointly trainable.
- Mixed precision remains bf16 and gradient checkpointing remains disabled in the preferred fast path.
- SigLIP2 targets remain online; no persistent feature cache is created.
- Existing DDP checkpoints remain resumable and sample cursor semantics remain deterministic.
- No long training run starts before a short throughput sweep succeeds.

---

### Task 1: Asynchronous metric accumulation and CUDA-event timing

**Files:**
- Create: `qwen35_baton/training_telemetry.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py:40-72,867-903,1318-1455`
- Test: `tests/test_qwen35_baton_telemetry.py`
- Test: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Produces: `Stage1MetricAccumulator.add_loss(...)`, `Stage1MetricAccumulator.add_scalar(...)`, and `Stage1MetricAccumulator.flush(accelerator, divisor)`.
- Produces: `CudaEventTimer.start(name)`, `CudaEventTimer.stop(name)`, and `CudaEventTimer.resolve()`; CPU tests use the same interface through injected event factories.
- Consumes: existing `_average_metrics`, durable metric names, `BatonPlannerLoss`, prediction and target tensors.

- [ ] **Step 1: Write failing tensor-accumulation tests**

```python
def test_metric_accumulator_keeps_values_on_device_until_flush():
    accumulator = Stage1MetricAccumulator()
    loss = torch.tensor(2.0)
    prediction = torch.zeros((1, 1, 4, 2, 3))
    target = torch.ones_like(prediction)
    accumulator.add_loss(loss, prediction, target, camera_names=("head",))
    assert all(isinstance(value, torch.Tensor) for value in accumulator.sums.values())
    metrics = accumulator.flush(_IdentityAccelerator(), divisor=1)
    assert metrics["loss/total"] == 2.0
    assert metrics["mse/head/frame_0"] == 1.0
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_telemetry.py::test_metric_accumulator_keeps_values_on_device_until_flush`

Expected: FAIL because `qwen35_baton.training_telemetry` does not exist.

- [ ] **Step 3: Implement the minimal GPU-tensor accumulator**

```python
class Stage1MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}

    def add_scalar(self, name: str, value: torch.Tensor | float, *, device=None) -> None:
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
        tensor = tensor.to(dtype=torch.float64)
        self.sums[name] = self.sums.get(name, torch.zeros_like(tensor)) + tensor
```

Implement `add_loss` with one vectorized `[camera, frame]` MSE reduction and `flush` with one packed tensor reduction and one CPU transfer.

- [ ] **Step 4: Add a failing CUDA-event timer test**

```python
def test_cuda_event_timer_resolves_without_synchronizing_on_start_or_stop():
    factory = _FakeEventFactory()
    timer = CudaEventTimer(enabled=True, event_factory=factory)
    timer.start("qwen")
    timer.stop("qwen")
    assert factory.synchronize_calls == 0
    assert timer.resolve()["qwen"] == 0.125
    assert factory.synchronize_calls == 1
```

- [ ] **Step 5: Run the timer test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_telemetry.py::test_cuda_event_timer_resolves_without_synchronizing_on_start_or_stop`

Expected: FAIL because `CudaEventTimer` is missing.

- [ ] **Step 6: Implement CUDA-event timing and integrate it into training**

Record events around planner, query tower, teacher, and backward. Register query-tower hooks once before the training loop and remove them in `finally`. Resolve all events once at a synchronized optimizer boundary. Remove per-microbatch `_synchronize_device` calls and replace `_loss_metrics` CPU scalarization with `Stage1MetricAccumulator`.

- [ ] **Step 7: Add a regression test that production training does not call the legacy synchronizer per microbatch**

```python
def test_stage1_synchronizes_telemetry_once_per_completed_update(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(training_module, "_synchronize_device", lambda device: calls.append(device))
    result = run_training(_tiny_config(tmp_path, gradient_accumulation_steps=4), artifacts=_artifacts(), stop_at_step=1)
    assert result.global_step == 1
    assert len(calls) <= 1
```

- [ ] **Step 8: Run Task 1 tests and existing training tests**

Run: `pytest -q tests/test_qwen35_baton_telemetry.py tests/test_qwen35_baton_training.py`

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add qwen35_baton/training_telemetry.py qwen35_baton/cli/train_semantic_planner.py tests/test_qwen35_baton_telemetry.py tests/test_qwen35_baton_training.py
git commit -m "perf(baton): remove per-microbatch synchronization"
```

### Task 2: Production row-validation fast path

**Files:**
- Modify: `qwen35_baton/model.py:185-260`
- Modify: `qwen35_baton/cli/train_semantic_planner.py:80-210,1344-1346`
- Modify: `qwen35_baton/configs/worldarena_stage1.json`
- Modify: `qwen35_baton/configs/libero_stage1.json`
- Test: `tests/test_qwen35_baton_model.py`
- Test: `tests/test_qwen35_baton_config.py`
- Test: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Produces: `BatonQwen35Planner.forward(..., validate_input_contents: bool = True)` and matching `forward_rows` keyword.
- Produces: `Stage1TrainingConfig.runtime_input_validation: bool`, defaulting to `False` for production configs.
- Consumes: CPU collator `find_plan_positions`, which remains the authoritative static content validator.

- [ ] **Step 1: Write a failing fast-path equivalence test**

```python
def test_fast_row_validation_matches_strict_forward():
    planner, batch = _planner_and_valid_batch()
    strict = planner(batch, validate_input_contents=True).positive
    fast = planner(batch, validate_input_contents=False).positive
    torch.testing.assert_close(fast, strict, rtol=0, atol=0)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_model.py::test_fast_row_validation_matches_strict_forward`

Expected: FAIL because `validate_input_contents` is not accepted.

- [ ] **Step 3: Implement the fast path**

Keep rank, shape, dtype, and field-type checks in both paths. Gate GPU value checks (`any`, `all`, `torch.equal`, per-row `nonzero`) behind `validate_input_contents`. The strict default preserves inference/provider behavior; Stage-1 explicitly passes its production setting.

- [ ] **Step 4: Add failing config validation tests**

```python
def test_worldarena_production_disables_redundant_gpu_input_validation():
    config = json.loads((REPO_ROOT / "qwen35_baton/configs/worldarena_stage1.json").read_text())
    assert config["runtime_input_validation"] is False

def test_runtime_input_validation_requires_boolean():
    with pytest.raises(ValueError, match="runtime_input_validation must be boolean"):
        Stage1TrainingConfig.from_mapping({**_valid_config(), "runtime_input_validation": 0})
```

- [ ] **Step 5: Run config tests and verify RED**

Run: `pytest -q tests/test_qwen35_baton_config.py -k runtime_input_validation`

Expected: FAIL because the field is absent.

- [ ] **Step 6: Add the config field and route it into planner forward**

Add `runtime_input_validation: bool = False` and validate with `type(value) is bool`. Update both production JSON files. Do not alter the immutable geometry or checkpoint tensor topology.

- [ ] **Step 7: Run Task 2 tests**

Run: `pytest -q tests/test_qwen35_baton_model.py tests/test_qwen35_baton_config.py tests/test_qwen35_baton_training.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add qwen35_baton/model.py qwen35_baton/cli/train_semantic_planner.py qwen35_baton/configs/worldarena_stage1.json qwen35_baton/configs/libero_stage1.json tests/test_qwen35_baton_model.py tests/test_qwen35_baton_config.py tests/test_qwen35_baton_training.py
git commit -m "perf(baton): move static row checks off the hot path"
```

### Task 3: Exact worker-side SigLIP2 preprocessing and device prefetch

**Files:**
- Create: `qwen35_baton/device_prefetch.py`
- Modify: `qwen35_baton/data.py:17-31,170-356`
- Modify: `qwen35_baton/teacher.py:100-188`
- Modify: `qwen35_baton/cli/train_semantic_planner.py:498-522,587-641,1218-1224,1310-1357`
- Test: `tests/test_qwen35_baton_teacher.py`
- Test: `tests/test_qwen35_baton_data.py`
- Test: `tests/test_qwen35_baton_training.py`
- Test: `tests/test_qwen35_baton_device_prefetch.py`

**Interfaces:**
- Produces: `preprocess_siglip2_future(processor, images, dtype) -> Tensor[B,C,4,3,H,W]`.
- Produces: `FrozenSiglip2Teacher.encode_pixel_values(pixel_values) -> Tensor[B,C,4,256,1024]`.
- Extends: `BatonPlannerBatch.future_pixel_values: Tensor | None` and `BatonPlannerBatch.pin_memory()`.
- Produces: `enable_device_prefetch(loader, device)` using a dedicated CUDA stream and non-blocking transfer.

- [ ] **Step 1: Write a failing teacher preprocessing-equivalence test**

```python
def test_preprocessed_future_matches_encode_future():
    teacher, _ = make_teacher()
    images = torch.randint(256, (2, 1, 4, 3, 256, 256), dtype=torch.uint8)
    pixel_values = preprocess_siglip2_future(teacher.processor, images, dtype=teacher.dtype)
    direct = teacher.encode_future(images)
    prepared = teacher.encode_pixel_values(pixel_values)
    torch.testing.assert_close(prepared, direct, rtol=0, atol=0)
```

- [ ] **Step 2: Run the teacher test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_teacher.py::test_preprocessed_future_matches_encode_future`

Expected: FAIL because preprocessing and `encode_pixel_values` APIs are absent.

- [ ] **Step 3: Implement batched exact preprocessing and feature encoding**

Flatten `[B,C,4,3,H,W]` once, call the released processor once, cast to the teacher dtype, restore geometry, and let `encode_pixel_values` flatten only for vision-model microbatching. Retain `encode_future` by composing preprocessing with `encode_pixel_values`.

- [ ] **Step 4: Write a failing collator test**

```python
def test_collator_emits_future_teacher_pixels_without_raw_future_transfer():
    collator = BatonPlannerCollator(_Processor(), camera_names=("head",), siglip_processor=_SiglipProcessor())
    batch = collator([_worldarena_sample()])
    assert batch.future_pixel_values.shape == (1, 1, 4, 3, 256, 256)
    assert batch.future_pixel_values.dtype == torch.bfloat16
```

- [ ] **Step 5: Run the collator test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_data.py::test_collator_emits_future_teacher_pixels_without_raw_future_transfer`

Expected: FAIL because the collator has no SigLIP processor argument.

- [ ] **Step 6: Extend the batch and collator**

Process all future frames in the worker collator. Preserve `future_images` only for compatibility when no SigLIP processor is configured; production supplies `future_pixel_values` and does not transfer raw future RGB to CUDA. Add a dataclass `pin_memory()` method that recursively pins tensor fields while leaving strings and labels unchanged.

- [ ] **Step 7: Write and run a failing batched-Qwen-processor equivalence test**

```python
def test_batched_qwen_rows_match_reference_rowwise_collation():
    samples = [_worldarena_sample(value=3), _worldarena_sample(value=7)]
    reference = BatonPlannerCollator(_BatchProcessor(), camera_names=("head",), batch_qwen_rows=False)(samples)
    processor = _BatchProcessor()
    batched = BatonPlannerCollator(processor, camera_names=("head",), batch_qwen_rows=True)(samples)
    assert processor.call_batch_sizes == [2]
    for key in reference.qwen_inputs:
        torch.testing.assert_close(batched.qwen_inputs[key], reference.qwen_inputs[key], rtol=0, atol=0)
    torch.testing.assert_close(batched.plan_positions, reference.plan_positions, rtol=0, atol=0)
```

Run: `pytest -q tests/test_qwen35_baton_data.py::test_batched_qwen_rows_match_reference_rowwise_collation`

Expected: FAIL because `batch_qwen_rows` is absent. Implement a single processor call over all sample-major camera rows while retaining the row-wise reference mode for equivalence diagnosis.

- [ ] **Step 8: Write a failing asynchronous-prefetch test**

```python
def test_prefetch_uses_nonblocking_transfer_and_records_stream():
    batch = _PinAwareBatch()
    loader = enable_device_prefetch(_SingleBatchLoader(batch), torch.device("cuda:0"), stream_factory=_FakeStreamFactory())
    assert next(iter(loader)).non_blocking is True
    assert batch.recorded_stream is not None
```

- [ ] **Step 9: Run the prefetch test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_device_prefetch.py::test_prefetch_uses_nonblocking_transfer_and_records_stream`

Expected: FAIL because the prefetch module does not exist.

- [ ] **Step 10: Implement VLAForge-style device prefetch**

Prepare the DataLoader with `device_placement=False`, enable `pin_memory=True` and `prefetch_factor=4`, transfer the next batch on a dedicated CUDA stream, wait from the current stream, and call `record_stream` recursively. The training loop consumes the already-device-resident batch and calls `teacher.encode_pixel_values`.

- [ ] **Step 11: Run Task 3 tests**

Run: `pytest -q tests/test_qwen35_baton_teacher.py tests/test_qwen35_baton_data.py tests/test_qwen35_baton_device_prefetch.py tests/test_qwen35_baton_training.py`

Expected: PASS.

- [ ] **Step 12: Commit Task 3**

```bash
git add qwen35_baton/device_prefetch.py qwen35_baton/data.py qwen35_baton/teacher.py qwen35_baton/cli/train_semantic_planner.py tests/test_qwen35_baton_teacher.py tests/test_qwen35_baton_data.py tests/test_qwen35_baton_device_prefetch.py tests/test_qwen35_baton_training.py
git commit -m "perf(baton): overlap exact online SigLIP preprocessing"
```

### Task 4: Fixed-global-batch throughput sweep

**Files:**
- Create: `qwen35_baton/cli/benchmark_stage1_throughput.py`
- Create: `qwen35_baton/scripts/benchmark_worldarena_batches.sh`
- Modify: `qwen35_baton/cli/train_semantic_planner.py:55-72,1435-1470`
- Test: `tests/test_qwen35_baton_benchmark.py`
- Test: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Produces: `BatchCandidate(per_device_batch, gradient_accumulation_steps)`.
- Produces: `run_batch_sweep(base_config, candidates, output_dir, warmup_steps, measured_steps, command_runner)`.
- Adds durable metrics `max_memory_allocated_gib`, `max_memory_reserved_gib`, and `step_time` at completed-update boundaries.

- [ ] **Step 1: Write a failing candidate-validation test**

```python
def test_worldarena_sweep_preserves_global_batch_128():
    candidates = default_worldarena_candidates(world_size=8)
    assert [(item.per_device_batch, item.gradient_accumulation_steps) for item in candidates] == [(4, 4), (8, 2), (16, 1)]
    assert all(item.per_device_batch * 8 * item.gradient_accumulation_steps == 128 for item in candidates)
```

- [ ] **Step 2: Run the candidate test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_benchmark.py::test_worldarena_sweep_preserves_global_batch_128`

Expected: FAIL because the benchmark module does not exist.

- [ ] **Step 3: Implement candidate generation and isolated trial configs**

Each trial writes a complete immutable config under `<output>/configs/b{batch}_a{accum}.json`, uses a distinct output root, and invokes the existing preflight and torchrun launcher. Trial output must not point at a production checkpoint directory.

- [ ] **Step 4: Write a failing OOM-and-summary test**

```python
def test_sweep_records_oom_and_selects_fastest_stable_candidate(tmp_path):
    runner = _FakeRunner({(4, 4): _ok(70.0, 52.0), (8, 2): _ok(91.0, 68.0), (16, 1): _oom()})
    result = run_batch_sweep(_base_config(), default_worldarena_candidates(8), tmp_path, 5, 20, runner)
    assert result.selected.per_device_batch == 8
    assert result.trials[-1].status == "oom"
    assert json.loads((tmp_path / "summary.json").read_text())["selected"]["per_device_batch"] == 8
```

- [ ] **Step 5: Run the summary test and verify RED**

Run: `pytest -q tests/test_qwen35_baton_benchmark.py::test_sweep_records_oom_and_selects_fastest_stable_candidate`

Expected: FAIL because sweep execution is missing.

- [ ] **Step 6: Implement subprocess isolation and result parsing**

Treat only CUDA out-of-memory exit text as `oom`; all other nonzero exits are `failed` and stop selection. Parse the final integrity-valid metrics record, require finite throughput/memory/step time, and select the highest throughput stable trial with at least 5 GiB unallocated device memory.

- [ ] **Step 7: Run Task 4 tests**

Run: `pytest -q tests/test_qwen35_baton_benchmark.py tests/test_qwen35_baton_training.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add qwen35_baton/cli/benchmark_stage1_throughput.py qwen35_baton/scripts/benchmark_worldarena_batches.sh qwen35_baton/cli/train_semantic_planner.py tests/test_qwen35_baton_benchmark.py tests/test_qwen35_baton_training.py
git commit -m "feat(baton): add fixed-global-batch throughput sweep"
```

### Task 5: Optional bf16 ZeRO-2 strategy with fail-closed resume

**Files:**
- Modify: `qwen35_baton/cli/train_semantic_planner.py:80-210,316-345,1162-1179`
- Modify: `qwen35_baton/checkpoint.py`
- Modify: `qwen35_baton/config.py`
- Modify: `qwen35_baton/configs/deepspeed_zero2.json`
- Test: `tests/test_qwen35_baton_training.py`
- Test: `tests/test_qwen35_baton_checkpoint.py`

**Interfaces:**
- Produces: `Stage1TrainingConfig.distributed_strategy`, restricted to `"ddp"` and `"zero2"`.
- Produces: `build_accelerator(config, world_size)`; DDP passes no plugin, ZeRO-2 passes `DeepSpeedPlugin(hf_ds_config=resolve_deepspeed_runtime_config(...))`.
- Extends checkpoint metadata with `distributed_strategy`; legacy metadata without the field migrates deterministically to `"ddp"`.

- [ ] **Step 1: Write failing strategy-selection tests**

```python
def test_ddp_builds_accelerator_without_deepspeed(monkeypatch):
    captured = _capture_accelerator(monkeypatch)
    build_accelerator(_config(distributed_strategy="ddp"), world_size=8)
    assert captured.kwargs.get("deepspeed_plugin") is None

def test_zero2_resolves_exact_global_batch(monkeypatch):
    captured = _capture_accelerator(monkeypatch)
    build_accelerator(_config(distributed_strategy="zero2", per_device_batch=8, gradient_accumulation_steps=2), world_size=8)
    payload = captured.kwargs["deepspeed_plugin"].hf_ds_config.config
    assert payload["zero_optimization"]["stage"] == 2
    assert payload["train_batch_size"] == 128
    assert "offload_optimizer" not in payload["zero_optimization"]
```

- [ ] **Step 2: Run strategy tests and verify RED**

Run: `pytest -q tests/test_qwen35_baton_training.py -k 'builds_accelerator or zero2_resolves'`

Expected: FAIL because `build_accelerator` and `distributed_strategy` are absent.

- [ ] **Step 3: Implement explicit strategy construction**

Keep DDP as the default. Construct `DeepSpeedPlugin` only for `zero2`, using the already resolved micro/global batch fields. Reject stage other than 2 and every CPU/NVMe offload key during preflight.

- [ ] **Step 4: Write failing checkpoint compatibility tests**

```python
def test_legacy_checkpoint_strategy_migrates_to_ddp(tmp_path):
    checkpoint = _write_checkpoint(tmp_path, omit_distributed_strategy=True)
    state = load_baton_checkpoint(checkpoint, expected_distributed_strategy="ddp", **_load_args())
    assert state.metadata.distributed_strategy == "ddp"

def test_checkpoint_rejects_distributed_strategy_mismatch(tmp_path):
    checkpoint = _write_checkpoint(tmp_path, distributed_strategy="ddp")
    with pytest.raises(ValueError, match="distributed strategy"):
        load_baton_checkpoint(checkpoint, expected_distributed_strategy="zero2", **_load_args())
```

- [ ] **Step 5: Run checkpoint tests and verify RED**

Run: `pytest -q tests/test_qwen35_baton_checkpoint.py -k distributed_strategy`

Expected: FAIL because checkpoint metadata has no strategy contract.

- [ ] **Step 6: Implement metadata migration and strategy checks**

Persist `distributed_strategy` in new checkpoints, interpret its absence as legacy DDP, and validate before optimizer state loading. For ZeRO-2, use DeepSpeed's distributed checkpoint save/load on every rank and publish the existing Baton manifest only after all shards are durable. Keep the current single-file DDP checkpoint path unchanged.

- [ ] **Step 7: Run Task 5 tests**

Run: `pytest -q tests/test_qwen35_baton_training.py tests/test_qwen35_baton_checkpoint.py tests/test_qwen35_baton_config.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```bash
git add qwen35_baton/cli/train_semantic_planner.py qwen35_baton/checkpoint.py qwen35_baton/config.py qwen35_baton/configs/deepspeed_zero2.json tests/test_qwen35_baton_training.py tests/test_qwen35_baton_checkpoint.py tests/test_qwen35_baton_config.py
git commit -m "feat(baton): add optional ZeRO-2 training strategy"
```

### Task 6: Full regression verification and eight-GPU selection

**Files:**
- Modify: `qwen35_baton/configs/worldarena_stage1.json`
- Create on execution host: isolated benchmark outputs under a non-checkpoint `benchmarks/` directory

**Interfaces:**
- Consumes: batch sweep CLI and current WorldArena HDF5/model artifacts.
- Produces: selected per-device batch/accumulation values and evidence JSON.

- [ ] **Step 1: Run the focused Stage-1 suite**

Run:

```bash
pytest -q \
  tests/test_qwen35_baton_telemetry.py \
  tests/test_qwen35_baton_model.py \
  tests/test_qwen35_baton_teacher.py \
  tests/test_qwen35_baton_data.py \
  tests/test_qwen35_baton_device_prefetch.py \
  tests/test_qwen35_baton_benchmark.py \
  tests/test_qwen35_baton_training.py \
  tests/test_qwen35_baton_checkpoint.py \
  tests/test_qwen35_baton_config.py \
  tests/test_qwen35_baton_worldarena_data.py
```

Expected: PASS with zero failures.

- [ ] **Step 2: Run repository diff and syntax checks**

Run:

```bash
git diff --check
python -m compileall -q qwen35_baton
```

Expected: both commands exit 0.

- [ ] **Step 3: Sync the exact commit to the authorized eight-GPU host**

Record `git rev-parse HEAD`, copy only tracked source/config changes, and verify the remote checkout reports the same commit or source-tree digest. Do not copy local `runtime/`, caches, test artifacts, or benchmark outputs into Git.

- [ ] **Step 4: Run DDP batch sweep**

Run on the allocated eight-GPU node:

```bash
bash qwen35_baton/scripts/benchmark_worldarena_batches.sh \
  --config qwen35_baton/configs/worldarena_stage1.json \
  --candidates 4:4,8:2,16:1 \
  --warmup-steps 5 \
  --measure-steps 20 \
  --output benchmarks/worldarena_ddp_$(date +%Y%m%d_%H%M%S)
```

Expected: `summary.json` contains a stable selected candidate and all candidates retain global batch 128.

- [ ] **Step 5: Run ZeRO-2 comparison only when it can unlock a larger microbatch**

If DDP batch 16 is stable, retain DDP and skip ZeRO-2. If DDP batch 16 OOMs, run the same batch 16 / accumulation 1 trial with `distributed_strategy=zero2`. Select ZeRO-2 only if it is stable and its throughput exceeds the best DDP result.

- [ ] **Step 6: Update production config with measured winner**

Set only `per_device_batch`, `gradient_accumulation_steps`, and `distributed_strategy` from the winning summary. Keep all optimization, LR, geometry, cadence, and data fields unchanged.

- [ ] **Step 7: Run a 20-update resume smoke test**

Resume from the latest compatible durable checkpoint into a new output directory, stop after 20 completed optimizer updates, verify finite loss, exact global batch 128, increasing cursor, and a clean checkpoint/metrics integrity check.

- [ ] **Step 8: Commit the measured production configuration**

```bash
git add qwen35_baton/configs/worldarena_stage1.json
git commit -m "perf(worldarena): select measured Baton batch configuration"
```

- [ ] **Step 9: Report evidence before requesting a long run**

Report baseline and selected throughput, step time, peak memory, strategy, per-device batch, accumulation, and the exact tested commit. Do not start the 30k run until the user authorizes it.
