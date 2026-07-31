# Baton DataLoader Worker Recycling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the eight-worker-per-rank Stage-1 throughput while preventing persistent DataLoader workers from accumulating CPU memory until the Slurm cgroup OOMs.

**Architecture:** Add an absolute-epoch recycling schedule to the Stage-1 configuration. Keep lifecycle handling in a focused module that unwraps Accelerate loaders, shuts down the underlying PyTorch persistent iterator, validates rank-local outcomes collectively, and writes separate lifecycle events. Invoke it only after a complete epoch and before the next iterator is created.

**Tech Stack:** Python 3.10+, PyTorch `DataLoader`, Hugging Face Accelerate, pytest, JSONL, DDP.

## Global Constraints

- Production uses `worker_restart_interval_epochs=100`.
- Recycling is active only with `num_workers > 0` and `persistent_workers=true`.
- All ranks recycle at the same absolute epoch boundary.
- No recycling occurs after a partial final epoch.
- Existing `training_metrics.jsonl`, checkpoint, optimizer, scheduler, scaler, sampler, and cursor contracts remain unchanged.
- Unsupported PyTorch or Accelerate loader lifecycles fail closed with a descriptive error.
- Tests use real worker processes for worker lifetime behavior.

## File structure

- Create `qwen35_baton/worker_lifecycle.py`: schedule predicate, loader unwrapping, worker shutdown, distributed status validation, and lifecycle JSONL append.
- Create `tests/test_qwen35_baton_worker_lifecycle.py`: focused real-process and status-validation tests.
- Modify `qwen35_baton/cli/train_semantic_planner.py`: configuration field, validation, distributed orchestration, and epoch-boundary integration.
- Modify `tests/test_qwen35_baton_training.py`: configuration and end-to-end tiny-training coverage.
- Modify `qwen35_baton/configs/libero_stage1.json`: production interval and persistent-worker defaults.

---

### Task 1: Configuration and deterministic restart schedule

**Files:**
- Create: `qwen35_baton/worker_lifecycle.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py:67-175`
- Test: `tests/test_qwen35_baton_worker_lifecycle.py`
- Test: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Produces: `should_restart_workers(*, completed_epoch: int, interval_epochs: int | None) -> bool`
- Produces: `Stage1TrainingConfig.worker_restart_interval_epochs: int | None`
- Consumes: zero-based absolute cursor epochs.

- [ ] **Step 1: Write failing configuration and schedule tests**

```python
@pytest.mark.parametrize("value", [None, 1, 100])
def test_worker_restart_interval_accepts_disabled_or_positive_values(
    tmp_path: Path, value: int | None
) -> None:
    config = replace(
        _config(tmp_path),
        worker_restart_interval_epochs=value,
    )
    assert config.worker_restart_interval_epochs == value


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_worker_restart_interval_rejects_invalid_values(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(
        ValueError,
        match="worker_restart_interval_epochs",
    ):
        replace(
            _config(tmp_path),
            worker_restart_interval_epochs=value,
        )


@pytest.mark.parametrize(
    ("completed_epoch", "interval", "expected"),
    [
        (0, None, False),
        (98, 100, False),
        (99, 100, True),
        (100, 100, False),
        (199, 100, True),
    ],
)
def test_worker_restart_schedule_uses_absolute_completed_epochs(
    completed_epoch: int,
    interval: int | None,
    expected: bool,
) -> None:
    assert (
        should_restart_workers(
            completed_epoch=completed_epoch,
            interval_epochs=interval,
        )
        is expected
    )
```

- [ ] **Step 2: Run the tests and verify the missing API fails**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_worker_lifecycle.py \
  tests/test_qwen35_baton_training.py::test_worker_restart_interval_accepts_disabled_or_positive_values \
  tests/test_qwen35_baton_training.py::test_worker_restart_interval_rejects_invalid_values
```

Expected: collection or assertion failure because `should_restart_workers` and `worker_restart_interval_epochs` do not exist.

- [ ] **Step 3: Implement the minimal schedule and validation**

Add to `Stage1TrainingConfig`:

```python
worker_restart_interval_epochs: int | None = 100
```

Add validation:

```python
if (
    self.worker_restart_interval_epochs is not None
    and (
        type(self.worker_restart_interval_epochs) is not int
        or self.worker_restart_interval_epochs <= 0
    )
):
    raise ValueError(
        "worker_restart_interval_epochs must be None or a positive integer"
    )
```

Create the schedule function:

```python
def should_restart_workers(
    *,
    completed_epoch: int,
    interval_epochs: int | None,
) -> bool:
    if type(completed_epoch) is not int or completed_epoch < 0:
        raise ValueError("completed_epoch must be a non-negative integer")
    if interval_epochs is None:
        return False
    if type(interval_epochs) is not int or interval_epochs <= 0:
        raise ValueError("interval_epochs must be None or a positive integer")
    return (completed_epoch + 1) % interval_epochs == 0
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Commit the schedule contract**

```bash
git add \
  qwen35_baton/worker_lifecycle.py \
  qwen35_baton/cli/train_semantic_planner.py \
  tests/test_qwen35_baton_worker_lifecycle.py \
  tests/test_qwen35_baton_training.py
git commit -m "feat(baton): configure periodic worker recycling"
```

### Task 2: Real persistent-worker lifecycle adapter

**Files:**
- Modify: `qwen35_baton/worker_lifecycle.py`
- Test: `tests/test_qwen35_baton_worker_lifecycle.py`

**Interfaces:**
- Consumes: a PyTorch `DataLoader` or an Accelerate wrapper exposing `base_dataloader`.
- Produces: `recycle_persistent_dataloader_workers(batches: Any) -> bool`
- Returns: `True` only when an active persistent iterator was shut down.

- [ ] **Step 1: Write a real-process failing test**

```python
def _worker_pids(loader: torch.utils.data.DataLoader) -> tuple[int, ...]:
    iterator = loader._iterator
    assert iterator is not None
    return tuple(worker.pid for worker in iterator._workers)


def test_recycling_preserves_order_and_replaces_real_worker_pids(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        per_device_batch=1,
        num_workers=2,
        persistent_workers=True,
    )
    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(8)),
        collate_fn=None,
        config=config,
    )
    try:
        first_values = [batch[0].item() for batch in loader]
        first_workers = tuple(loader._iterator._workers)
        first_pids = _worker_pids(loader)

        second_values = [batch[0].item() for batch in loader]
        assert _worker_pids(loader) == first_pids
        assert second_values == first_values

        assert recycle_persistent_dataloader_workers(loader) is True
        assert loader._iterator is None
        assert all(not worker.is_alive() for worker in first_workers)

        third_values = [batch[0].item() for batch in loader]
        third_pids = _worker_pids(loader)
        assert set(first_pids).isdisjoint(third_pids)
        assert third_values == first_values
    finally:
        recycle_persistent_dataloader_workers(loader)


def test_recycling_without_an_active_iterator_is_a_noop(tmp_path: Path) -> None:
    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(4)),
        collate_fn=None,
        config=replace(
            _config(tmp_path),
            per_device_batch=1,
            num_workers=1,
            persistent_workers=True,
        ),
    )
    assert recycle_persistent_dataloader_workers(loader) is False


def test_recycling_rejects_an_unsupported_wrapper() -> None:
    class UnsupportedLoader:
        pass

    with pytest.raises(RuntimeError, match="UnsupportedLoader"):
        recycle_persistent_dataloader_workers(UnsupportedLoader())
```

- [ ] **Step 2: Run the real-process test and verify it fails**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_worker_lifecycle.py::test_recycling_preserves_order_and_replaces_real_worker_pids
```

Expected: failure because `recycle_persistent_dataloader_workers` is missing.

- [ ] **Step 3: Implement strict loader unwrapping and shutdown**

```python
def _underlying_torch_dataloader(batches: Any) -> torch.utils.data.DataLoader:
    current = batches
    seen: set[int] = set()
    while True:
        identity = id(current)
        if identity in seen:
            raise RuntimeError("DataLoader wrapper chain contains a cycle")
        seen.add(identity)
        nested = getattr(current, "base_dataloader", None)
        if nested is not None:
            current = nested
            continue
        if isinstance(current, torch.utils.data.DataLoader):
            return current
        raise RuntimeError(
            "cannot recycle workers for unsupported loader type "
            f"{type(current).__module__}.{type(current).__qualname__}"
        )


def recycle_persistent_dataloader_workers(batches: Any) -> bool:
    loader = _underlying_torch_dataloader(batches)
    if loader.num_workers == 0 or not loader.persistent_workers:
        return False
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return False
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if not callable(shutdown):
        raise RuntimeError(
            "active persistent DataLoader iterator does not expose "
            "worker shutdown"
        )
    shutdown()
    loader._iterator = None
    return True
```

- [ ] **Step 4: Add and run an Accelerate-wrapper characterization test**

```python
def test_accelerate_prepared_loader_recycles_the_underlying_workers(
    tmp_path: Path,
) -> None:
    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(8)),
        collate_fn=None,
        config=replace(
            _config(tmp_path),
            per_device_batch=1,
            num_workers=1,
            persistent_workers=True,
        ),
    )
    prepared = Accelerator(cpu=True).prepare_data_loader(loader)
    try:
        next(iter(prepared))
        old_pids = _worker_pids(loader)
        assert recycle_persistent_dataloader_workers(prepared) is True
        assert loader._iterator is None
        next(iter(prepared))
        assert set(old_pids).isdisjoint(_worker_pids(loader))
    finally:
        recycle_persistent_dataloader_workers(prepared)
```

Run:

```bash
PYTHONPATH=. pytest -q tests/test_qwen35_baton_worker_lifecycle.py
```

Expected: all lifecycle tests pass.

- [ ] **Step 5: Commit the lifecycle adapter**

```bash
git add qwen35_baton/worker_lifecycle.py tests/test_qwen35_baton_worker_lifecycle.py
git commit -m "fix(baton): recycle persistent data workers safely"
```

### Task 3: Distributed epoch-boundary orchestration and event logging

**Files:**
- Modify: `qwen35_baton/worker_lifecycle.py`
- Modify: `qwen35_baton/cli/train_semantic_planner.py:1004-1365`
- Modify: `tests/test_qwen35_baton_training.py`
- Test: `tests/test_qwen35_baton_worker_lifecycle.py`

**Interfaces:**
- Produces: `validate_recycle_statuses(statuses: Sequence[Mapping[str, Any]], *, world_size: int, completed_epoch: int) -> float`
- Produces: `append_worker_lifecycle_event(path: Path, *, completed_epoch: int, next_epoch: int, restart_count: int, interval_epochs: int, elapsed_seconds: float) -> None`
- Consumes: `recycle_persistent_dataloader_workers`, `should_restart_workers`, `accelerate.utils.gather_object`.

- [ ] **Step 1: Write failing literal-status and JSONL tests**

```python
def test_recycle_status_validation_returns_the_slowest_rank_duration() -> None:
    statuses = [
        {"rank": 0, "epoch": 99, "recycled": True, "error": None, "elapsed": 1.25},
        {"rank": 1, "epoch": 99, "recycled": True, "error": None, "elapsed": 1.75},
    ]
    assert validate_recycle_statuses(
        statuses,
        world_size=2,
        completed_epoch=99,
    ) == 1.75


@pytest.mark.parametrize(
    "statuses",
    [
        [{"rank": 0, "epoch": 99, "recycled": False, "error": None, "elapsed": 1.0}],
        [{"rank": 0, "epoch": 98, "recycled": True, "error": None, "elapsed": 1.0}],
        [{"rank": 0, "epoch": 99, "recycled": False, "error": "shutdown failed", "elapsed": 1.0}],
    ],
)
def test_recycle_status_validation_fails_closed(
    statuses: list[dict[str, object]],
) -> None:
    with pytest.raises(RuntimeError, match="worker recycl"):
        validate_recycle_statuses(
            statuses,
            world_size=1,
            completed_epoch=99,
        )


def test_worker_lifecycle_event_is_separate_from_training_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker_lifecycle.jsonl"
    append_worker_lifecycle_event(
        path,
        completed_epoch=99,
        next_epoch=100,
        restart_count=1,
        interval_epochs=100,
        elapsed_seconds=1.75,
    )
    assert json.loads(path.read_text()) == {
        "completed_epoch": 99,
        "elapsed_seconds": 1.75,
        "event": "dataloader_workers_restarted",
        "interval_epochs": 100,
        "next_epoch": 100,
        "restart_count": 1,
        "schema_version": 1,
    }
```

- [ ] **Step 2: Run the focused tests and verify the APIs are missing**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_worker_lifecycle.py::test_recycle_status_validation_returns_the_slowest_rank_duration \
  tests/test_qwen35_baton_worker_lifecycle.py::test_recycle_status_validation_fails_closed \
  tests/test_qwen35_baton_worker_lifecycle.py::test_worker_lifecycle_event_is_separate_from_training_metrics
```

Expected: import failure because the validation and event functions do not exist.

- [ ] **Step 3: Implement validation and durable event append**

```python
def validate_recycle_statuses(
    statuses: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    completed_epoch: int,
) -> float:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if len(statuses) != world_size:
        raise RuntimeError(
            "worker recycling did not publish exactly one status per rank"
        )
    ranks: list[int] = []
    elapsed_values: list[float] = []
    for status in statuses:
        rank = status.get("rank")
        elapsed = status.get("elapsed")
        if (
            type(rank) is not int
            or status.get("epoch") != completed_epoch
            or status.get("recycled") is not True
            or status.get("error") is not None
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise RuntimeError(f"worker recycling failed closed: {statuses!r}")
        ranks.append(rank)
        elapsed_values.append(float(elapsed))
    if sorted(ranks) != list(range(world_size)):
        raise RuntimeError(
            "worker recycling did not publish one unique status per rank"
        )
    return max(elapsed_values)


def append_worker_lifecycle_event(
    path: Path,
    *,
    completed_epoch: int,
    next_epoch: int,
    restart_count: int,
    interval_epochs: int,
    elapsed_seconds: float,
) -> None:
    record = {
        "schema_version": 1,
        "event": "dataloader_workers_restarted",
        "completed_epoch": completed_epoch,
        "next_epoch": next_epoch,
        "restart_count": restart_count,
        "interval_epochs": interval_epochs,
        "elapsed_seconds": elapsed_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
```

- [ ] **Step 4: Add distributed orchestration to the trainer**

Add `_recycle_stage1_workers_distributed` to
`qwen35_baton/cli/train_semantic_planner.py`:

```python
def _recycle_stage1_workers_distributed(
    *,
    accelerator: Any,
    batches: Any,
    completed_epoch: int,
) -> float:
    accelerator.wait_for_everyone()
    started = time.perf_counter()
    try:
        recycled = recycle_persistent_dataloader_workers(batches)
        error = None
    except Exception as exception:
        recycled = False
        error = f"{type(exception).__name__}: {exception}"
    local = {
        "rank": accelerator.process_index,
        "epoch": completed_epoch,
        "recycled": recycled,
        "error": error,
        "elapsed": time.perf_counter() - started,
    }
    if accelerator.num_processes == 1:
        statuses = [local]
    else:
        from accelerate.utils import gather_object
        statuses = gather_object([local])
    return validate_recycle_statuses(
        statuses,
        world_size=accelerator.num_processes,
        completed_epoch=completed_epoch,
    )
```

Add the logging helper so rank-zero I/O errors are collected before any rank
raises:

```python
def _append_worker_lifecycle_event_distributed(
    *,
    accelerator: Any,
    path: Path,
    completed_epoch: int,
    next_epoch: int,
    restart_count: int,
    interval_epochs: int,
    elapsed_seconds: float,
) -> None:
    error = None
    if accelerator.is_main_process:
        try:
            append_worker_lifecycle_event(
                path,
                completed_epoch=completed_epoch,
                next_epoch=next_epoch,
                restart_count=restart_count,
                interval_epochs=interval_epochs,
                elapsed_seconds=elapsed_seconds,
            )
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    local = {"rank": accelerator.process_index, "error": error}
    if accelerator.num_processes == 1:
        statuses = [local]
    else:
        from accelerate.utils import gather_object
        statuses = gather_object([local])
    ranks = sorted(
        status.get("rank")
        for status in statuses
        if isinstance(status, Mapping) and type(status.get("rank")) is int
    )
    errors = [
        status.get("error")
        for status in statuses
        if isinstance(status, Mapping) and status.get("error") is not None
    ]
    if ranks != list(range(accelerator.num_processes)) or errors:
        raise RuntimeError(
            "worker lifecycle event logging failed closed: "
            f"statuses={statuses!r}"
        )
    accelerator.wait_for_everyone()
```

In `run_training`, initialize:

```python
worker_lifecycle_path = Path(config.output_dir) / "worker_lifecycle.jsonl"
```

Replace the epoch-exhaustion branch with:

```python
        else:
            completed_epoch = epoch
            interval_epochs = config.worker_restart_interval_epochs
            if (
                cursor.global_step < target_step
                and config.num_workers > 0
                and config.persistent_workers
                and should_restart_workers(
                    completed_epoch=completed_epoch,
                    interval_epochs=interval_epochs,
                )
            ):
                assert interval_epochs is not None
                elapsed = _recycle_stage1_workers_distributed(
                    accelerator=accelerator,
                    batches=train_batches,
                    completed_epoch=completed_epoch,
                )
                _append_worker_lifecycle_event_distributed(
                    accelerator=accelerator,
                    path=worker_lifecycle_path,
                    completed_epoch=completed_epoch,
                    next_epoch=cursor.epoch,
                    restart_count=(completed_epoch + 1) // interval_epochs,
                    interval_epochs=interval_epochs,
                    elapsed_seconds=elapsed,
                )
            continue
```

- [ ] **Step 5: Write and run a tiny-training integration test**

Create a top-level picklable `_TinyBatchDataset` and `_identity_tiny_batch`
collator in `tests/test_qwen35_baton_training.py`. Run two one-microbatch
epochs with one persistent spawn worker and interval one:

```python
def test_training_recycles_workers_only_between_complete_epochs(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path, max_steps=2, save_every=2),
        per_device_batch=1,
        num_workers=1,
        persistent_workers=True,
        worker_restart_interval_epochs=1,
    )
    artifacts = _artifacts(config)
    artifacts.train_batches = build_stage1_dataloader(
        _TinyBatchDataset(count=1),
        collate_fn=_identity_tiny_batch,
        config=config,
    )

    try:
        result = run_training(config, artifacts=artifacts)
    finally:
        recycle_persistent_dataloader_workers(artifacts.train_batches)

    assert result.global_step == 2
    assert result.cursor.epoch == 2
    assert result.cursor.consumed_microbatches == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "worker_lifecycle.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["completed_epoch"] == 0
    assert records[0]["next_epoch"] == 1
```

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_worker_lifecycle.py \
  tests/test_qwen35_baton_training.py::test_training_recycles_workers_only_between_complete_epochs
```

Expected: all selected tests pass and no worker process remains after test
cleanup.

- [ ] **Step 6: Commit distributed integration**

```bash
git add \
  qwen35_baton/worker_lifecycle.py \
  qwen35_baton/cli/train_semantic_planner.py \
  tests/test_qwen35_baton_worker_lifecycle.py \
  tests/test_qwen35_baton_training.py
git commit -m "fix(baton): recycle workers at synchronized epoch boundaries"
```

### Task 4: Production recipe and regression verification

**Files:**
- Modify: `qwen35_baton/configs/libero_stage1.json`
- Modify: `tests/test_qwen35_baton_training.py`

**Interfaces:**
- Consumes: `Stage1TrainingConfig.worker_restart_interval_epochs`
- Produces: production recipe values `persistent_workers=true` and `worker_restart_interval_epochs=100`.

- [ ] **Step 1: Write the failing production-recipe assertion**

Extend `test_stage1_recipe_requirements_and_launchers_are_fixed`:

```python
assert config["num_workers"] == 8
assert config["persistent_workers"] is True
assert config["worker_restart_interval_epochs"] == 100
```

- [ ] **Step 2: Run the assertion and verify it fails**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_training.py::test_stage1_recipe_requirements_and_launchers_are_fixed
```

Expected: failure because the recipe lacks the two worker-lifecycle fields.

- [ ] **Step 3: Add the production recipe**

Add after `num_workers` in `qwen35_baton/configs/libero_stage1.json`:

```json
"persistent_workers": true,
"worker_restart_interval_epochs": 100,
```

- [ ] **Step 4: Run Baton regression tests**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_qwen35_baton_worker_lifecycle.py \
  tests/test_qwen35_baton_training.py \
  tests/test_qwen35_baton_data.py \
  tests/test_qwen35_baton_checkpoint.py
```

Expected: all tests pass with no leaked `pt_data_worker` processes.

- [ ] **Step 5: Run static checks**

Run:

```bash
git diff --check
python -m compileall -q qwen35_baton
git status --short
```

Expected: no whitespace errors, compilation succeeds, and only intended
tracked files plus the pre-existing untracked `runtime/` directory appear.

- [ ] **Step 6: Commit the production default**

```bash
git add qwen35_baton/configs/libero_stage1.json tests/test_qwen35_baton_training.py
git commit -m "config(baton): recycle workers every 100 epochs"
```

- [ ] **Step 7: Verify the final commit range**

Run:

```bash
git log --oneline --decorate -5
git diff 4a58318..HEAD --check
```

Expected: the worker-recycling design, plan, implementation, tests, and
production recipe are committed; the untracked runtime files are not included.
