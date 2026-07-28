from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from accelerate import Accelerator
from safetensors.torch import save_file
import torch
import torch.distributed as dist
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GE_ACT_ROOT = REPOSITORY_ROOT / "ge_act"
for path in (REPOSITORY_ROOT, GE_ACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runner.ge_trainer import (  # noqa: E402
    TrainingCursor,
    save_baton_training_checkpoint,
)
from qwen35_baton.hashing import sha256_json  # noqa: E402


class _DistributedTinyDiffusion(nn.Linear):
    def __init__(self, *, fail_snapshot: bool) -> None:
        super().__init__(1, 1, bias=False)
        self.fail_snapshot = fail_snapshot

    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        safe_serialization: bool,
    ) -> None:
        if self.fail_snapshot:
            raise RuntimeError("injected rank-zero snapshot failure")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        save_file(
            {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
            },
            str(output / "diffusion_pytorch_model.safetensors"),
        )


def _training_provenance() -> dict[str, str | int]:
    sampling = {
        "algorithm": "libero_fastwam_hdf5_stateless_sha256",
        "version": 1,
        "seed": 42,
    }
    return {
        "hdf5_manifest_hash": "4" * 64,
        "siglip2_config_hash": "1" * 64,
        "siglip2_artifact_hash": "2" * 64,
        "teacher_preprocessing_hash": "3" * 64,
        "window_sampling_algorithm": sampling["algorithm"],
        "window_sampling_version": sampling["version"],
        "window_sampling_seed": sampling["seed"],
        "window_sampling_topology_hash": sha256_json(sampling),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure", choices=("destination", "snapshot"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args()

    accelerator = Accelerator(cpu=True)
    model = _DistributedTinyDiffusion(fail_snapshot=args.failure == "snapshot")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    prepared_model, optimizer, scheduler = accelerator.prepare(
        model,
        optimizer,
        scheduler,
    )
    raw_model = accelerator.unwrap_model(prepared_model)
    cursor = TrainingCursor(
        global_step=1,
        epoch=0,
        consumed_microbatches=1,
        microbatches_per_epoch=4,
        sampler_seed=42,
    )
    output_dir = Path(args.output_dir)
    if args.failure == "destination" and accelerator.is_main_process:
        (output_dir / "step_000001").mkdir(parents=True)
    accelerator.wait_for_everyone()

    caught: dict[str, str] | None = None
    try:
        save_baton_training_checkpoint(
            accelerator,
            output_dir,
            cursor=cursor,
            diffusion_model=raw_model,
            source="qwen35_baton_teacher",
            training_provenance=_training_provenance(),
        )
    except Exception as error:
        caught = {
            "type": type(error).__name__,
            "message": str(error),
        }
    gathered: list[dict[str, str] | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, caught)
    if accelerator.is_main_process:
        Path(args.result_path).write_text(
            json.dumps(gathered, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
