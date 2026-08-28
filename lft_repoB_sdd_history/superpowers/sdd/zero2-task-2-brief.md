### Task 2: Generate matched Accelerate and DeepSpeed configurations

**Files:**
- Modify: `tests/test_lingbot_zero2_runtime.py`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py`

**Interfaces:**
- Produces: `make_zero2_configs(grad_accum: int, num_processes: int, output_dir: Path) -> tuple[Path, Path]`.
- Consumes later: generic launcher environment values `GRAD_ACCUM`, `NUM_GPUS`, and `OUTPUT_DIR`.

- [ ] **Step 1: Add failing config-generation tests**

Append to `tests/test_lingbot_zero2_runtime.py`:

```python
import json

import yaml


CONFIG_GENERATOR = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "make_zero2_config.py"
)


def load_config_generator():
    spec = importlib.util.spec_from_file_location("lingbot_zero2_config", CONFIG_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero2_config_matches_batch_accumulation_and_process_count(tmp_path):
    module = load_config_generator()
    accelerate_path, deepspeed_path = module.make_zero2_configs(
        grad_accum=2,
        num_processes=8,
        output_dir=tmp_path,
    )

    deepspeed = json.loads(deepspeed_path.read_text())
    accelerate = yaml.safe_load(accelerate_path.read_text())
    assert deepspeed["gradient_accumulation_steps"] == 2
    assert deepspeed["zero_optimization"]["stage"] == 2
    assert deepspeed["zero_optimization"]["overlap_comm"] is True
    assert deepspeed["zero_optimization"]["contiguous_gradients"] is True
    assert deepspeed["bf16"]["enabled"] is True
    assert deepspeed["fp16"]["enabled"] is False
    assert accelerate["num_processes"] == 8
    assert accelerate["distributed_type"] == "DEEPSPEED"
    assert Path(accelerate["deepspeed_config"]["deepspeed_config_file"]) == deepspeed_path


@pytest.mark.parametrize(("grad_accum", "num_processes"), [(0, 8), (2, 0)])
def test_zero2_config_rejects_nonpositive_runtime_values(tmp_path, grad_accum, num_processes):
    module = load_config_generator()
    with pytest.raises(ValueError, match="positive"):
        module.make_zero2_configs(
            grad_accum=grad_accum,
            num_processes=num_processes,
            output_dir=tmp_path,
        )
```

- [ ] **Step 2: Run only the new tests and verify RED**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py -k zero2_config
```

Expected: FAIL because `make_zero2_config.py` does not exist.

- [ ] **Step 3: Implement deterministic config generation and CLI output**

Create `make_zero2_config.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_zero2_configs(
    *,
    grad_accum: int,
    num_processes: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    if grad_accum <= 0 or num_processes <= 0:
        raise ValueError("gradient accumulation and process count must be positive")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"np{num_processes}_ga{grad_accum}_zero2"
    deepspeed_path = output_dir / f"deepspeed_{tag}.json"
    accelerate_path = output_dir / f"accelerate_{tag}.yaml"
    deepspeed_config = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": grad_accum,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 500_000_000,
            "reduce_scatter": True,
            "reduce_bucket_size": 500_000_000,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }
    deepspeed_path.write_text(
        json.dumps(deepspeed_config, indent=2) + "\n",
        encoding="utf-8",
    )
    accelerate_path.write_text(
        "\n".join(
            [
                "compute_environment: LOCAL_MACHINE",
                "debug: false",
                "deepspeed_config:",
                f'  deepspeed_config_file: "{deepspeed_path}"',
                "  deepspeed_multinode_launcher: standard",
                "  zero3_init_flag: false",
                "distributed_type: DEEPSPEED",
                "downcast_bf16: 'no'",
                "machine_rank: 0",
                "main_training_function: main",
                "mixed_precision: bf16",
                "num_machines: 1",
                f"num_processes: {num_processes}",
                "rdzv_backend: static",
                "same_network: true",
                "use_cpu: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return accelerate_path, deepspeed_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    accelerate_path, _ = make_zero2_configs(
        grad_accum=args.grad_accum,
        num_processes=args.num_processes,
        output_dir=args.output_dir,
    )
    print(accelerate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify config tests and CLI**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py -k zero2_config
python scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py --grad-accum 2 --num-processes 8 --output-dir /tmp/vlm4wam-zero2-config-test
```

Expected: tests pass; the CLI prints an Accelerate YAML path under `/tmp/vlm4wam-zero2-config-test`.

---

