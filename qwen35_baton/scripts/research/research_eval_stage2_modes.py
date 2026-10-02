"""E3: paired multi-sample evaluation of a Baton Stage-2 LTX checkpoint.

For each validation window the same inputs and noise are run under several
semantic modes (teacher guidance on, off, wrist-only masked, main-only masked;
``planner*`` modes use frozen Qwen3.5 baton planner predictions instead),
recording action MSE per horizon and future-video MSE. Paired differences
against ``semantic_disabled`` test H1 (does oracle guidance help) and H8 (is
wrist guidance useful). Shardable across 1-GPU jobs; merge with --merge.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import default_collate

MODES = tuple(
    os.environ.get(
        "E3_MODES",
        "teacher,semantic_disabled,teacher_wrist_masked,teacher_main_masked",
    ).split(",")
)
HORIZONS = ((0, 1), (1, 9), (9, 25), (25, 36))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="materialized Stage-2 YAML")
    parser.add_argument("--checkpoint", help="Stage-2 step dir containing diffusion_model/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="only summarize shard files")
    return parser


def _semantic_kwargs(runner, condition, mode: str, n_view: int) -> dict:
    from runner.ge_trainer import apply_baton_validation_mode

    base_mode = "semantic_disabled" if mode == "semantic_disabled" else "teacher"
    tokens = condition.tokens
    if mode == "random_tokens":
        # Wiring probe: large random guidance must change the output if the
        # semantic path is live at inference.
        generator = torch.Generator(device=tokens.device).manual_seed(0)
        tokens = torch.randn(
            tokens.shape, generator=generator, device=tokens.device, dtype=torch.float32
        ).to(tokens.dtype) * (3 * tokens.float().std())
    selection = apply_baton_validation_mode(
        runner.baton_components,
        tokens=tokens,
        mode=base_mode,
        batch_size=1,
        n_view=n_view,
        device=runner.state.accelerator.device,
        dtype=runner.state.weight_dtype,
    )
    mask = selection.condition_mask.clone()
    # Rows are (b v) with views ordered main, wrist.
    if mode.endswith("_wrist_masked"):
        mask[1::n_view] = 0
    elif mode.endswith("_main_masked"):
        mask[0::n_view] = 0
    return {
        "semantic_plan": selection.tokens,
        "semantic_plan_times": condition.times,
        "semantic_plan_positions": condition.positions,
        "semantic_plan_mask": condition.mask,
        "semantic_plan_relevance": condition.relevance,
        "semantic_condition_mask": mask,
    }


def _semantic_trace(runner):
    """The trace list of the module that actually defines the loaded blocks.

    The trainer imports model classes by file path, so this can differ from
    ``models.ltx_models.transformer_ltx_multiview`` imported by name.
    """

    import sys

    for module in runner.diffusion_model.modules():
        if hasattr(module, "semantic_gate_mode"):
            return getattr(sys.modules[type(module).__module__], "_SEMANTIC_TRACE", None)
    return None


@torch.no_grad()
def _run_mode(runner, pipe, batch, condition, mode: str, seed: int) -> dict:
    args = runner.args
    accelerator = runner.state.accelerator
    n_prev = args.data["train"]["n_previous"]
    image = batch["video"][:, :, :, :n_prev].clone()
    _, _, n_view, _, height, width = image.shape
    image = rearrange(image, "b c v t h w -> (b v) c t h w")
    history = None
    if args.return_action and getattr(args, "add_state", False):
        history = batch["state"][:1]
        if history.shape[1] > 1:
            history = history[:, n_prev - 1 : n_prev, :]
        history = history.contiguous()
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    trace = _semantic_trace(runner)
    if trace is not None:
        trace.clear()
    preds = pipe.infer(
        image=image,
        prompt=[""] if os.environ.get("E3_EMPTY_PROMPT") == "1" else batch["caption"][:1],
        negative_prompt="",
        num_inference_steps=args.num_inference_step,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        guidance_scale=1.0,
        height=height,
        width=width,
        n_view=n_view,
        return_action=args.return_action,
        n_prev=n_prev,
        chunk=(args.data["train"]["chunk"] - 1) // runner.TEMPORAL_DOWN_RATIO + 1,
        return_video=args.return_video,
        noise_seed=seed,
        action_chunk=args.data["train"]["action_chunk"],
        history_action_state=history,
        pixel_wise_timestep=args.pixel_wise_timestep,
        num_frames=args.data["train"]["chunk"],
        postprocess_video=False,
        n_chunk=1,
        action_dim=args.diffusion_model["config"]["action_in_channels"] if args.return_action else None,
        frame_rate=runner.video_frame_rate,
        generator=generator,
        **_semantic_kwargs(runner, condition, mode, n_view),
    )[0]
    result: dict = {}
    trace = _semantic_trace(runner)
    if trace:
        result["semantic_ratio"] = float(np.mean(trace))
        trace.clear()
    gt_actions = batch["actions"][:, -args.data["train"]["action_chunk"] :].float()
    action_dim = gt_actions.shape[-1]
    squared = (preds["action"][:1, :, :action_dim].float().cpu() - gt_actions[:1]).square()
    for start, stop in HORIZONS:
        result[f"action_{start}_{stop}"] = float(squared[:, start:stop].mean())
    result["action_all"] = float(squared.mean())
    if args.return_video:
        predicted = rearrange(preds["video"].float().cpu(), "(b v) c t h w -> b c v t h w", b=1, v=n_view)
        target = batch["video"][:1, :, :, n_prev:].float().cpu()
        frames = min(predicted.shape[3], target.shape[3])
        per_view = (predicted[:, :, :, :frames] - target[:, :, :, :frames]).square().mean(dim=(0, 1, 3, 4, 5))
        result["video"] = float(per_view.mean())
        result["video_main"] = float(per_view[0])
        result["video_wrist"] = float(per_view[1])
        # Motion-region error: only pixels whose true future differs from the
        # last memory frame (arm and manipulated objects); full-frame MSE is
        # dominated by static background.
        last = batch["video"][:1, :, :, n_prev - 1 : n_prev].float().cpu()
        motion = (target[:, :, :, :frames] - last).abs().mean(dim=1, keepdim=True) > 0.1
        squared_error = (predicted[:, :, :, :frames] - target[:, :, :, :frames]).square()
        masked = (squared_error * motion).sum(dim=(0, 1, 3, 4, 5))
        pixels = motion.sum(dim=(0, 1, 3, 4, 5)).clamp_min(1) * squared_error.shape[1]
        motion_per_view = masked / pixels
        result["video_motion"] = float(motion_per_view.mean())
        result["video_motion_main"] = float(motion_per_view[0])
        result["video_motion_wrist"] = float(motion_per_view[1])
        result["motion_fraction"] = float(motion.float().mean())
    return result


def _evaluate(args: argparse.Namespace) -> None:
    import dataclasses

    from runner.ge_trainer import (
        Trainer,
        _normalized_video_to_uint8,
        build_baton_semantic_condition,
        compute_ltx_latent_frames,
    )

    # Trainer.__init__ broadcasts over the default process group, so a
    # standalone evaluator must provide a single-rank one.
    if not torch.distributed.is_initialized():
        job = int(os.environ.get("SLURM_JOB_ID", "0"))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(20000 + job % 20000))
        for name, value in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
            os.environ.setdefault(name, value)
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Trainer(args.config, output_dir=str(output_dir / f"runner_shard{args.shard_index}"))
    runner.args.load_weights = True
    runner.args.load_diffusion_model_weights = True
    runner.args.diffusion_model["model_path"] = str(Path(args.checkpoint) / "diffusion_model")
    runner.prepare_dataset()
    runner.prepare_val_dataset()
    runner.prepare_models()
    runner.diffusion_model.eval()
    accelerator = runner.state.accelerator
    pipe = runner.pipeline_class(
        runner.scheduler, runner.vae, runner.text_encoder, runner.tokenizer,
        runner._unwrapped_diffusion_model(accelerator),
    )
    n_prev = runner.args.data["train"]["n_previous"]
    raw_future = runner.args.data["train"]["chunk"]
    latent_frames = compute_ltx_latent_frames(
        raw_future, temporal_compression_ratio=runner.TEMPORAL_DOWN_RATIO, n_previous=n_prev
    )
    # Optional frozen Qwen3.5 baton planner: "planner*" modes replace the
    # SigLIP2 teacher tokens with its predictions from the current frames.
    planner = None
    if os.environ.get("E3_PLANNER_CHECKPOINT"):
        from qwen35_baton.provider import FrozenBatonPlanner

        qwen = os.environ["E3_PLANNER_QWEN_PATH"]
        planner = FrozenBatonPlanner.from_checkpoint(
            os.environ["E3_PLANNER_CHECKPOINT"],
            qwen_model_path=qwen,
            qwen_tokenizer_path=qwen,
            qwen_processor_path=qwen,
            siglip2_model_path=runner.args.semantic_plan["siglip2_model_path"],
            device=accelerator.device,
        )
    if any(mode.startswith("planner") for mode in MODES) and planner is None:
        raise ValueError("planner modes need E3_PLANNER_CHECKPOINT and E3_PLANNER_QWEN_PATH")
    dataset = runner.val_dataset
    indices = np.linspace(0, len(dataset) - 1, args.num_samples).round().astype(int).tolist()
    mine = indices[args.shard_index :: args.num_shards]
    out_path = output_dir / f"shard{args.shard_index:02d}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["index"] for line in out_path.read_text().splitlines() if line.strip()}
    started = time.time()
    with out_path.open("a", encoding="utf-8") as stream:
        for count, index in enumerate(mine):
            if index in done:
                continue
            batch = default_collate([dataset[index]])
            condition = build_baton_semantic_condition(
                runner.baton_components,
                runner.args.semantic_plan,
                batch["video"][:1],
                batch["caption"][:1],
                n_previous=n_prev,
                num_future_frames=raw_future,
                num_latent_frames=latent_frames,
                device=accelerator.device,
                dtype=runner.state.weight_dtype,
            )
            planner_condition = None
            if planner is not None:
                current = batch["video"][:1, :, :, n_prev - 1].permute(0, 2, 1, 3, 4)
                current = _normalized_video_to_uint8(current).contiguous()
                with torch.no_grad():
                    predicted = planner.predict(current, tuple(batch["caption"][:1])).tokens
                planner_condition = dataclasses.replace(
                    condition,
                    tokens=predicted.to(device=condition.tokens.device, dtype=condition.tokens.dtype),
                )
            record = {"index": int(index), "caption": batch["caption"][0]}
            for mode in MODES:
                mode_condition = planner_condition if mode.startswith("planner") else condition
                record[mode] = _run_mode(
                    runner, pipe, batch, mode_condition, mode, seed=1000 + int(index)
                )
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps({"done": count + 1, "of": len(mine), "elapsed_s": round(time.time() - started)}), flush=True)


def _merge(args: argparse.Namespace) -> None:
    records = []
    for path in sorted(glob.glob(os.path.join(args.output_dir, "shard*.jsonl"))):
        records += [json.loads(line) for line in open(path) if line.strip()]
    metrics = sorted({key for record in records for mode in MODES for key in record[mode]})
    summary = {"num_samples": len(records), "reference": "semantic_disabled", "modes": {}}
    for mode in MODES:
        summary["modes"][mode] = {}
        for metric in metrics:
            if any(metric not in r[mode] or metric not in r["semantic_disabled"] for r in records):
                if all(metric in r[mode] for r in records):
                    summary["modes"][mode][metric] = {
                        "mean": float(np.mean([r[mode][metric] for r in records]))
                    }
                continue
            values = np.array([r[mode][metric] for r in records])
            base = np.array([r["semantic_disabled"][metric] for r in records])
            diff = values - base
            summary["modes"][mode][metric] = {
                "mean": float(values.mean()),
                "paired_diff_vs_disabled": float(diff.mean()),
                "diff_se": float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else None,
                "win_rate": float((diff < 0).mean()),
            }
    path = Path(args.output_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parsed = _parser().parse_args()
    if parsed.merge:
        _merge(parsed)
    else:
        _evaluate(parsed)
