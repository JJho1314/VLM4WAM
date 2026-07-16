"""Batch LIBERO eval: load the model ONCE, then loop over a task-id range.

Same per-task protocol/outputs as eval_libero_single.py (reuses its functions),
but avoids the ~4-min model reload per task. Resumable: skips task ids whose
results json already exists.

Extra overrides:
  +EVALUATION.task_id_start=682 +EVALUATION.task_id_end=1101 +EVALUATION.task_id_stride=2
(`task_id_end` exclusive; stride/offset lets N GPUs split round-robin via task_id_start.)
"""
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from omegaconf import DictConfig

from experiments.libero.eval_libero_single import (
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_task,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from hydra.utils import instantiate
from libero.libero import benchmark


@contextmanager
def _libero_torch_load_compat():
    """torch>=2.6 defaults weights_only=True; LIBERO init-state pickles need False."""
    original_load = torch.load

    def _load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_load


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    partial_state = PartialState()
    partial_state.config = cfg

    _validate_visualize_future_video_cfg(cfg)

    tid_start = int(cfg.EVALUATION.get("task_id_start", 0))
    tid_end = int(cfg.EVALUATION.get("task_id_end", -1))
    tid_stride = int(cfg.EVALUATION.get("task_id_stride", 1))

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats = load_dataset_stats_from_json(str(_resolve_dataset_stats_path(cfg)))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
    )
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    out_dir = local_log_dir / cfg.EVALUATION.task_suite_name
    out_dir.mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()[cfg.EVALUATION.task_suite_name]()
    n_tasks = int(task_suite.n_tasks)
    if tid_end < 0:
        tid_end = n_tasks
    task_ids = list(range(tid_start, min(tid_end, n_tasks), tid_stride))
    logging.info("Batch eval: %d tasks (%d..%d stride %d)", len(task_ids), tid_start, tid_end, tid_stride)

    for k, tid in enumerate(task_ids):
        out_file = out_dir / f"gpu{cfg.gpu_id}_task{tid}_results.json"
        if out_file.exists():
            continue
        cfg.EVALUATION.task_id = tid  # used by run_single_task for video naming
        task = task_suite.get_task(tid)
        with _libero_torch_load_compat():
            initial_states = task_suite.get_task_init_states(tid)
        trial_start = int(cfg.EVALUATION.get("trial_start", 0))
        trial_stop = trial_start + int(cfg.EVALUATION.num_trials)
        initial_states = list(initial_states)
        while len(initial_states) < trial_stop:
            initial_states.extend(initial_states[: (trial_stop - len(initial_states))])
        initial_states = initial_states[trial_start:trial_stop]

        t0 = time.time()
        results = {
            "task_suite": cfg.EVALUATION.task_suite_name,
            "task_id": tid,
            "task_description": None,
            "successes": 0,
            "total_episodes": int(cfg.EVALUATION.num_trials),
            "gpu_id": int(cfg.gpu_id),
            "success_episodes": [],
            "failure_episodes": [],
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": 0,
        }
        try:
            task_results = run_single_task(
                task=task,
                initial_states=initial_states,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            results.update(task_results)
        except Exception as exc:  # keep the batch alive on isolated task failures
            logging.exception("task %d raised: %s", tid, exc)
            results["error"] = str(exc)
        results["duration"] = time.time() - t0
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        print(
            f"[{k + 1}/{len(task_ids)}] task {tid}: {results['successes']}/{cfg.EVALUATION.num_trials} "
            f"({results['duration']:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
