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
