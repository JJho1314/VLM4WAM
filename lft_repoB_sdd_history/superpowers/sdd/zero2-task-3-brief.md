### Task 3: Replace manual DDP in the planner trainer

**Files:**
- Modify: `tests/test_lingbot_zero2_runtime.py`
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`

**Interfaces:**
- Consumes: all Task 1 runtime functions and existing `build_optimizer`, `build_scheduler`, `save_checkpoint`, dataset, teacher, and model APIs.
- Produces: an Accelerate-prepared planner/optimizer/DataLoader and optimizer-step bookkeeping that is correct for ZeRO-2 and ordinary Accelerate runs.

- [ ] **Step 1: Add failing trainer integration assertions**

Append this test to `tests/test_lingbot_zero2_runtime.py`:

```python
TRAINER = ROOT / "scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py"


def test_trainer_uses_accelerate_runtime_without_manual_ddp():
    source = TRAINER.read_text(encoding="utf-8")
    assert "accelerator.prepare(wrapper, optim, loader)" in source
    assert "accelerator.backward(out[\"loss\"])" in source
    assert "checkpoint_module(accelerator, wrapper)" in source
    assert "DistributedDataParallel" not in source
    assert "DistributedSampler" not in source
    assert ".backward()" not in source
```

Add this parser assertion beside the existing gradient-checkpointing tests in
`tests/test_lingbot_dino_depth_contract.py`:

```python
def test_expected_global_batch_cli_defaults_to_unconstrained_and_accepts_128(monkeypatch):
    module = load_trainer_module()
    monkeypatch.setattr(sys, "argv", _fastwam_parser_argv())
    assert module.parse_args().expected_global_batch == 0

    monkeypatch.setattr(
        sys,
        "argv",
        _fastwam_parser_argv("--expected-global-batch", "128"),
    )
    assert module.parse_args().expected_global_batch == 128
```

Remove the obsolete `ddp_info` monkeypatch from
`test_main_preflights_fastwam_before_loading_qwen`; preflight must still fail
before Accelerate or Qwen initialization.

- [ ] **Step 2: Run the focused trainer tests and verify RED**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py::test_trainer_uses_accelerate_runtime_without_manual_ddp tests/test_lingbot_dino_depth_contract.py::test_expected_global_batch_cli_defaults_to_unconstrained_and_accepts_128 tests/test_lingbot_dino_depth_contract.py::test_main_preflights_fastwam_before_loading_qwen
```

Expected: the integration and parser tests fail against the manual-DDP trainer; the preflight-order test continues to pass.

- [ ] **Step 3: Replace imports, CLI, and distributed initialization**

In the trainer, remove imports of `torch.distributed`, `DistributedDataParallel`,
and `DistributedSampler`. Import these symbols after adding
`lingbot_dino_4b` to `sys.path`:

```python
from distributed_runtime import (  # noqa: E402
    accumulation_context,
    build_accelerator,
    checkpoint_module,
    is_deepspeed,
    is_optimizer_update,
    validate_runtime_contract,
)
```

Add this CLI argument after `--grad-accum` and remove
`--ddp-find-unused-parameters`:

```python
parser.add_argument("--expected-global-batch", type=int, default=0)
```

Delete `ddp_info`. Preserve `is_main(rank)` because the checkpoint helper's
existing unit-test API still accepts an explicit rank.

Change the checkpoint helper annotation and unwrap line from DDP-specific code
to:

```python
def save_checkpoint(
    output_dir: Path,
    step: int,
    wrapper: PlannerWrapper,
    processor: Any,
    args: argparse.Namespace,
    rank: int,
) -> None:
    if not is_main(rank):
        return
    module = wrapper
```

After argument validation and FastWAM preflight, replace manual DDP setup with:

```python
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
accelerator = build_accelerator(grad_accum=args.grad_accum, dtype=args.dtype)
runtime_contract = validate_runtime_contract(
    accelerator,
    per_device_batch_size=args.batch_size,
    grad_accum=args.grad_accum,
    expected_global_batch=args.expected_global_batch,
)
rank = int(accelerator.process_index)
world = int(accelerator.num_processes)
device = accelerator.device
if accelerator.is_main_process:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "distributed_type": runtime_contract.distributed_type,
                "world_size": runtime_contract.world_size,
                "batch_size_per_gpu": runtime_contract.per_device_batch_size,
                "gradient_accumulation_steps": runtime_contract.gradient_accumulation_steps,
                "global_batch_size": runtime_contract.global_batch_size,
                "zero_stage": runtime_contract.zero_stage,
                "dtype": args.dtype,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
            }
        ),
        flush=True,
    )
accelerator.wait_for_everyone()
random.seed(args.seed + rank)
torch.manual_seed(args.seed + rank)
```

- [ ] **Step 4: Prepare the model, optimizer, and DataLoader exactly once**

Remove the `DDP(...)` block. Build the DataLoader without a manual sampler:

```python
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=Collator(
        processor=processor,
        plan_sequence=plan_sequence,
    ),
    pin_memory=True,
)
optim = build_optimizer(wrapper, args)
scheduler = build_scheduler(optim, args)
wrapper, optim, loader = accelerator.prepare(wrapper, optim, loader)
```

Replace every training-time
`wrapper.module if isinstance(wrapper, DDP) else wrapper` with
`accelerator.unwrap_model(wrapper)`. Gate parameter logging, tqdm, and W&B on
`accelerator.is_main_process`. Remove the `sampler.set_epoch` branch from the
outer loop; retain `dataset.set_epoch(step)`.

- [ ] **Step 5: Replace the backward/update loop with a ZeRO-safe loop**

Initialize the loop with:

```python
step = 0
micro_step = 0
running_loss = 0.0
deepspeed_enabled = is_deepspeed(accelerator)
if not deepspeed_enabled:
    optim.zero_grad(set_to_none=True)
pbar = tqdm(
    total=args.max_steps,
    disable=not accelerator.is_local_main_process,
    desc="qwen3vl planner",
)
```

Replace the current direct backward and `accum` branch with:

```python
with accumulation_context(accelerator, wrapper):
    out = wrapper(**batch)
    accelerator.backward(out["loss"])
    if not deepspeed_enabled:
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                (parameter for parameter in wrapper.parameters() if parameter.requires_grad),
                1.0,
            )
        optim.step()
        optim.zero_grad(set_to_none=True)
running_loss += float(out["loss"].detach())
micro_step += 1
if is_optimizer_update(accelerator, micro_step, args.grad_accum):
    if scheduler is not None:
        scheduler.step()
    step += 1
    if accelerator.is_main_process:
        pbar.update(1)
        if step % args.log_steps == 0:
            average_loss = running_loss / max(args.log_steps * args.grad_accum, 1)
            log_entry = {
                "step": step,
                "loss": average_loss,
                "lr": scheduler.get_last_lr()[0] if scheduler is not None else args.lr,
            }
            log_entry.update(
                {key: float(value) for key, value in out.items() if key != "loss"}
            )
            print(json.dumps(log_entry), flush=True)
            if wandb_run is not None:
                wandb_run.log(log_entry, step=step)
    if step % args.log_steps == 0:
        running_loss = 0.0
    if step % args.save_steps == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                args.output_dir,
                step,
                checkpoint_module(accelerator, wrapper),
                processor,
                args,
                rank=0,
            )
        accelerator.wait_for_everyone()
    if step >= args.max_steps:
        break
```

At finalization, use the same synchronized unwrapped save, finish W&B only on
the main process, and call `accelerator.end_training()` instead of manually
destroying a process group.

- [ ] **Step 6: Verify the trainer refactor**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py tests/test_lingbot_dino_depth_contract.py
python -m py_compile scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py
```

Expected: both test files pass and Python compilation produces no output.

---

