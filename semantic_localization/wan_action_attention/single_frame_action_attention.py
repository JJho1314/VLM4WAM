"""Export paired SG-WAM action-to-video attention for one LIBERO frame.

The two forwards use the same trained checkpoint, real action/state, noisy
latents, timestep, text embedding, and random tensors.  The only difference is
whether the semantic plan is supplied.  Heavy model imports are intentionally
kept inside runtime functions so the map helpers remain cheap to test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ORIGINAL_REPO = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM")
DATASET_ROOT = Path("/data/LFT-W02_data/junjie/data/LIBERO-fastwam")
CHECKPOINT_ROOT = Path(
    "/data/LFT-W02_data/junjie/weights/"
    "joint_vlm_geact_action_k4_50k/step_40000/ltx"
)
LTX_ROOT = Path("/data/LFT-W02_data/junjie/weights/LTX-Video")
SIGLIP_ROOT = ORIGINAL_REPO / "third_party/siglip2-large-patch16-256"
DEFAULT_RGB = (
    ORIGINAL_REPO
    / "outputs/libero_episode_000288_siglip2_da3_stride16_probe"
    / "main/frame_000080/rgb.png"
)
STATS_PATH = (
    ORIGINAL_REPO
    / "ge_act/configs/ltx_model/libero/libero_fastwam_mix.json"
)
TRAINING_CONFIG_PATH = (
    ORIGINAL_REPO
    / "ge_act/configs/ltx_model/libero/"
    "video_model_libero_joint_vlm_geact_action_k4_hpc3.yaml"
)

SUITE = "10"
DATASET_NAME = "libero_10_no_noops_lerobot"
EPISODE_INDEX = 288
FRAME_INDEX = 80
NUM_PREVIOUS = 4
NUM_FUTURE = 9
WINDOW_START = FRAME_INDEX - NUM_PREVIOUS + 1
WINDOW_END = FRAME_INDEX + NUM_FUTURE + 1
CURRENT_OFFSET = NUM_PREVIOUS - 1
KEYFRAME_INDICES = (2, 4, 6, 8)
ACTION_HORIZON = 32
RESOLUTION = 256
ACTION_NOISE_STRENGTH = 0.05
VIDEO_NOISE_STRENGTH = 0.15
SEED = 80


def normalize_q01_q99(
    values: np.ndarray,
    stats: dict[str, list[float]],
) -> np.ndarray:
    """Map per-dimension q01/q99 statistics to [-1, 1] and clip."""
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    return np.clip(
        2.0 * (values - low) / (high - low + 1e-6) - 1.0,
        -1.0,
        1.0,
    )


def normalize_mean_std(
    values: np.ndarray,
    stats: dict[str, list[float]],
) -> np.ndarray:
    """Apply the exact lerobot_like_dataset mean/std training convention."""
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (values - mean) / (std + 1e-6)


def pack_real_action_state(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    frame_index: int,
    horizon: int,
    action_stats: dict[str, list[float]],
    state_stats: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build training-format `[action7; state8]` and history tokens."""
    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"expected actions [T,7], got {actions.shape}")
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError(f"expected states [T,8], got {states.shape}")
    if not 0 <= frame_index < len(actions) or frame_index >= len(states):
        raise IndexError(
            f"frame {frame_index} outside actions/states "
            f"{len(actions)}/{len(states)}"
        )

    action_chunk = actions[frame_index : frame_index + horizon]
    state_chunk = states[frame_index : frame_index + horizon]
    if not len(action_chunk):
        raise ValueError("real action chunk is empty")
    if len(action_chunk) < horizon:
        action_chunk = np.concatenate(
            [
                action_chunk,
                np.repeat(action_chunk[-1:], horizon - len(action_chunk), axis=0),
            ],
            axis=0,
        )
    if len(state_chunk) < horizon:
        state_chunk = np.concatenate(
            [
                state_chunk,
                np.repeat(state_chunk[-1:], horizon - len(state_chunk), axis=0),
            ],
            axis=0,
        )

    normalized_actions = normalize_mean_std(action_chunk, action_stats)
    normalized_states = normalize_mean_std(state_chunk, state_stats)
    current_state = normalize_mean_std(states[frame_index], state_stats)
    packed_action = np.concatenate(
        [normalized_actions, normalized_states],
        axis=1,
    ).astype(np.float32)
    history = np.concatenate(
        [np.zeros(7, dtype=np.float32), current_state],
    ).astype(np.float32)
    if packed_action.shape != (horizon, 15) or history.shape != (15,):
        raise ValueError(
            f"unexpected packed action/state shapes "
            f"{packed_action.shape} and {history.shape}"
        )
    return packed_action, history


def resolve_action_model_config(
    checkpoint_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Restore action kwargs omitted by Diffusers' serialized config."""
    action_keys = (
        "action_in_channels",
        "action_out_channels",
        "action_num_attention_heads",
        "action_attention_head_dim",
    )
    try:
        trained = training_config["diffusion_model"]["config"]
    except KeyError as error:
        raise ValueError(
            "training config must contain diffusion_model.config"
        ) from error
    missing = [key for key in action_keys if key not in trained]
    if missing:
        raise ValueError(f"training config is missing action keys: {missing}")
    resolved = dict(checkpoint_config)
    resolved.update({key: int(trained[key]) for key in action_keys})
    return resolved


def normalize_map(values: np.ndarray) -> np.ndarray:
    """Min-max normalize one finite, non-constant attention map."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("attention map contains non-finite values")
    span = float(values.max() - values.min())
    if span <= 1e-12:
        raise ValueError("attention map is constant")
    return (values - float(values.min())) / span


def positive_gain(plan_off: np.ndarray, plan_on: np.ndarray) -> np.ndarray:
    """Return only spatial attention added by enabling the semantic plan."""
    if np.shape(plan_off) != np.shape(plan_on):
        raise ValueError(
            "plan_off and plan_on must have the same shape, got "
            f"{np.shape(plan_off)} and {np.shape(plan_on)}"
        )
    return np.clip(normalize_map(plan_on) - normalize_map(plan_off), 0.0, None)


def extract_frame_map(
    values: np.ndarray,
    *,
    view: int,
    time_index: int,
    temporal: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Select one `[H,W]` map from flattened `[views,T,H,W]` video tokens."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    per_view = temporal * height * width
    if per_view <= 0 or flat.size % per_view:
        raise ValueError(
            f"expected a multiple of T*H*W={per_view} values, got {flat.size}"
        )
    views = flat.size // per_view
    if not 0 <= view < views:
        raise IndexError(f"view {view} outside [0, {views})")
    if not 0 <= time_index < temporal:
        raise IndexError(
            f"time_index {time_index} outside [0, {temporal})"
        )
    return flat.reshape(views, temporal, height, width)[view, time_index]


def aggregate_layer_maps(layer_maps: dict[int, np.ndarray]) -> np.ndarray:
    """Validate and average captured per-layer spatial maps."""
    if not layer_maps:
        raise ValueError("at least one attention layer is required")
    shapes = {np.shape(attention) for attention in layer_maps.values()}
    if len(shapes) != 1:
        raise ValueError(f"all attention layers must have the same shape: {shapes}")
    validated = []
    for layer_index in sorted(layer_maps):
        attention = np.asarray(layer_maps[layer_index], dtype=np.float32)
        normalize_map(attention)
        validated.append(attention)
    return np.stack(validated).mean(axis=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: Any) -> str:
    if hasattr(values, "detach"):
        values = values.detach().to(device="cpu", dtype=None).float().numpy()
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-rgb", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--ltx-root", type=Path, default=LTX_ROOT)
    parser.add_argument("--siglip-root", type=Path, default=SIGLIP_ROOT)
    parser.add_argument("--stats-path", type=Path, default=STATS_PATH)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=TRAINING_CONFIG_PATH,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_instruction(dataset_root: Path) -> str:
    metadata = (
        dataset_root / DATASET_NAME / "meta/episodes.jsonl"
    )
    with metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if int(item["episode_index"]) == EPISODE_INDEX:
                tasks = item["tasks"]
                if len(tasks) != 1:
                    raise ValueError(
                        f"expected one episode task, found {len(tasks)}"
                    )
                return str(tasks[0])
    raise ValueError(f"episode {EPISODE_INDEX} missing from {metadata}")


def read_window(
    dataset_root: Path,
    source_rgb: Path,
) -> tuple[np.ndarray, dict[str, float]]:
    """Read dual-camera frames 77..89 and pin main frame 80 to source_rgb."""
    import av

    frames_by_view = []
    for camera in (
        "observation.images.image",
        "observation.images.wrist_image",
    ):
        path = (
            dataset_root
            / DATASET_NAME
            / f"videos/chunk-000/{camera}/episode_{EPISODE_INDEX:06d}.mp4"
        )
        container = av.open(str(path))
        frames = []
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index < WINDOW_START:
                continue
            if frame_index >= WINDOW_END:
                break
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            frames.append(
                np.asarray(
                    image.resize(
                        (RESOLUTION, RESOLUTION),
                        Image.Resampling.BILINEAR,
                    )
                )
            )
        container.close()
        if len(frames) != NUM_PREVIOUS + NUM_FUTURE:
            raise ValueError(
                f"{camera} yielded {len(frames)} frames for "
                f"[{WINDOW_START}, {WINDOW_END}), expected "
                f"{NUM_PREVIOUS + NUM_FUTURE}"
            )
        frames_by_view.append(np.stack(frames))

    clip = np.stack(frames_by_view)
    exact_rgb = np.asarray(
        Image.open(source_rgb)
        .convert("RGB")
        .resize((RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR)
    )
    decoded_rgb = clip[0, CURRENT_OFFSET]
    delta = np.abs(
        decoded_rgb.astype(np.float32) - exact_rgb.astype(np.float32)
    )
    clip[0, CURRENT_OFFSET] = exact_rgb
    return clip, {
        "decoded_source_mean_abs_error": float(delta.mean()),
        "decoded_source_max_abs_error": float(delta.max()),
    }


def read_real_action_state(
    dataset_root: Path,
    stats_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    parquet = (
        dataset_root
        / DATASET_NAME
        / f"data/chunk-000/episode_{EPISODE_INDEX:06d}.parquet"
    )
    frame_table = pd.read_parquet(parquet)
    actions = np.stack(frame_table["action"].values).astype(np.float32)
    states = np.stack(frame_table["observation.state"].values).astype(np.float32)
    if FRAME_INDEX >= len(actions) or FRAME_INDEX >= len(states):
        raise IndexError(
            f"frame {FRAME_INDEX} outside episode length "
            f"actions={len(actions)} states={len(states)}"
        )

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    action_stats = stats[f"{DATASET_NAME}_eef"]
    state_stats = stats[f"{DATASET_NAME}_state_eef"]
    return pack_real_action_state(
        actions,
        states,
        frame_index=FRAME_INDEX,
        horizon=ACTION_HORIZON,
        action_stats=action_stats,
        state_stats=state_stats,
    )


def install_action_attention_capture(model: Any) -> dict[str, Any]:
    """Hook action-query to video-token attention for every action block."""
    import torch

    capture: dict[str, Any] = {"layers": {}}
    for layer_index, block in enumerate(model.action_blocks):
        attention = block.attn2
        original_forward = attention.forward

        def wrapped_forward(
            hidden_states,
            encoder_hidden_states=None,
            *,
            _attention=attention,
            _layer_index=layer_index,
            _original_forward=original_forward,
            **kwargs,
        ):
            if encoder_hidden_states is not None:
                with torch.no_grad():
                    query = _attention.to_q(hidden_states)
                    key = _attention.to_k(encoder_hidden_states)
                    if _attention.norm_q is not None:
                        query = _attention.norm_q(query)
                    if _attention.norm_k is not None:
                        key = _attention.norm_k(key)
                    heads = _attention.heads
                    head_dim = query.shape[-1] // heads
                    query = (
                        query.unflatten(-1, (heads, head_dim))
                        .transpose(1, 2)
                        .float()
                    )
                    key = (
                        key.unflatten(-1, (heads, head_dim))
                        .transpose(1, 2)
                        .float()
                    )
                    probabilities = (
                        query @ key.transpose(-1, -2) / math.sqrt(head_dim)
                    ).softmax(dim=-1)
                    capture["layers"][_layer_index] = (
                        probabilities.mean(dim=1)
                        .mean(dim=1)[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
            return _original_forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )

        attention.forward = wrapped_forward
    return capture


def prepare_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Load trained components and prepare exactly paired model inputs."""
    import torch

    ge_act_root = Path(__file__).resolve().parents[2] / "ge_act"
    os.chdir(ge_act_root)
    sys.path.insert(0, str(ge_act_root))

    from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    from models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder
    from models.ltx_models.transformer_ltx_multiview import (
        LTXVideoTransformer3DModel,
    )
    from transformers import T5EncoderModel, T5Tokenizer
    from utils.data_utils import _normalize_latents, _pack_latents
    from utils.model_utils import load_diffusion_model, load_vae_models
    import yaml

    device = args.device
    dtype = torch.bfloat16
    print("loading trained SG-WAM components", flush=True)
    checkpoint_config = json.loads(
        (args.checkpoint_root / "config.json").read_text(encoding="utf-8")
    )
    training_config = yaml.safe_load(
        args.training_config.read_text(encoding="utf-8")
    )
    config = resolve_action_model_config(checkpoint_config, training_config)
    model = load_diffusion_model(
        LTXVideoTransformer3DModel,
        model_dir=str(args.checkpoint_root),
        load_weights=True,
        **config,
    )
    expected_action_dim = int(config["action_in_channels"])
    if model.action_proj_in.in_features != expected_action_dim:
        raise RuntimeError(
            "action checkpoint contract was not restored: "
            f"model in_features={model.action_proj_in.in_features}, "
            f"expected {expected_action_dim}"
        )
    if model.action_proj_out.out_features != int(config["action_out_channels"]):
        raise RuntimeError(
            "action checkpoint output contract was not restored: "
            f"model out_features={model.action_proj_out.out_features}, "
            f"expected {config['action_out_channels']}"
        )
    model = model.to(device, dtype).eval()
    vae = load_vae_models(
        AutoencoderKLLTXVideo,
        str(args.ltx_root / "vae"),
    ).to(device, dtype).eval()
    if isinstance(vae.latents_mean, list):
        vae.latents_mean = torch.tensor(vae.latents_mean)
    if isinstance(vae.latents_std, list):
        vae.latents_std = torch.tensor(vae.latents_std)
    tokenizer = T5Tokenizer.from_pretrained(str(args.ltx_root / "tokenizer"))
    text_encoder = T5EncoderModel.from_pretrained(
        str(args.ltx_root / "text_encoder")
    ).to(device, dtype).eval()
    semantic_encoder = OnlineSiglip2SemanticEncoder(
        str(args.siglip_root),
        device=device,
        dtype=dtype,
    )

    source_sha256 = sha256_file(args.source_rgb)
    instruction = read_instruction(args.dataset_root)
    clip, decode_metrics = read_window(args.dataset_root, args.source_rgb)
    action_np, state_np = read_real_action_state(
        args.dataset_root,
        args.stats_path,
    )

    with torch.no_grad():
        video = (
            torch.from_numpy(clip)
            .permute(0, 1, 4, 2, 3)
            .float()
            .to(device, dtype)
            / 255.0
            * 2.0
            - 1.0
        )
        memory = video[:, :NUM_PREVIOUS]
        future = video[:, NUM_PREVIOUS : NUM_PREVIOUS + NUM_FUTURE]
        memory_inputs = (
            memory.reshape(-1, 1, 3, RESOLUTION, RESOLUTION)
            .permute(0, 2, 1, 3, 4)
        )
        memory_latents = _normalize_latents(
            vae.encode(memory_inputs).latent_dist.sample().to(dtype),
            vae.latents_mean.to(device),
            vae.latents_std.to(device),
        )
        _, channels, _, height, width = memory_latents.shape
        memory_latents = (
            memory_latents.reshape(
                2,
                NUM_PREVIOUS,
                channels,
                1,
                height,
                width,
            )
            .permute(0, 2, 1, 4, 5, 3)
            .reshape(2, channels, NUM_PREVIOUS, height, width)
        )
        future_latents = _normalize_latents(
            vae.encode(future.permute(0, 2, 1, 3, 4))
            .latent_dist.sample()
            .to(dtype),
            vae.latents_mean.to(device),
            vae.latents_std.to(device),
        )
        latents = torch.cat([memory_latents, future_latents], dim=2)
        temporal = int(latents.shape[2])
        packed = _pack_latents(latents, 1, 1)

        generator = torch.Generator(device=device).manual_seed(args.seed)
        video_noise = torch.randn(
            packed.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noisy_latents = (
            (1.0 - VIDEO_NOISE_STRENGTH) * packed
            + VIDEO_NOISE_STRENGTH * video_noise
        )

        semantic_plan = semantic_encoder.encode(
            future[:, KEYFRAME_INDICES].unsqueeze(0)
        )
        text_inputs = tokenizer(
            [instruction],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        text_embedding = text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]

        real_action = torch.tensor(
            action_np,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        state = (
            torch.tensor(state_np, device=device, dtype=dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        action_noise = torch.randn(
            real_action.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noisy_action = (
            (1.0 - ACTION_NOISE_STRENGTH) * real_action
            + ACTION_NOISE_STRENGTH * action_noise
        )
        video_timestep = torch.full(
            (2,),
            500,
            device=device,
            dtype=torch.long,
        )
        action_timestep = torch.full(
            (1, ACTION_HORIZON),
            int(ACTION_NOISE_STRENGTH * 1000),
            device=device,
            dtype=torch.long,
        )

    return {
        "model": model,
        "clip": clip,
        "instruction": instruction,
        "source_sha256": source_sha256,
        "decode_metrics": decode_metrics,
        "noisy_latents": noisy_latents,
        "semantic_plan": semantic_plan,
        "text_embedding": text_embedding,
        "noisy_action": noisy_action,
        "state": state,
        "video_timestep": video_timestep,
        "action_timestep": action_timestep,
        "temporal": temporal,
        "height": int(height),
        "width": int(width),
        "pairing_hashes": {
            "noisy_latents": sha256_array(noisy_latents),
            "noisy_action": sha256_array(noisy_action),
            "state": sha256_array(state),
            "text_embedding": sha256_array(text_embedding),
            "video_timestep": sha256_array(video_timestep),
            "action_timestep": sha256_array(action_timestep),
        },
    }


def run_condition(runtime: dict[str, Any], use_plan: bool) -> dict[int, np.ndarray]:
    import torch

    from models.ltx_models.semantic_conditioning import build_semantic_plan_times

    model = runtime["model"]
    capture = runtime["capture"]
    capture["layers"] = {}
    semantic_plan = runtime["semantic_plan"] if use_plan else None
    semantic_times = None
    if use_plan:
        semantic_times = build_semantic_plan_times(
            1,
            2,
            NUM_PREVIOUS,
            NUM_FUTURE,
            runtime["temporal"],
            KEYFRAME_INDICES,
            device=runtime["noisy_latents"].device,
            dtype=torch.float32,
        )

    with torch.no_grad():
        model(
            hidden_states=runtime["noisy_latents"].clone(),
            encoder_hidden_states=runtime["text_embedding"].clone(),
            timestep=runtime["video_timestep"].clone(),
            n_view=2,
            action_states=runtime["noisy_action"].clone(),
            action_timestep=runtime["action_timestep"].clone(),
            return_action=True,
            return_video=True,
            history_action_state=runtime["state"].clone(),
            num_frames=runtime["temporal"],
            height=runtime["height"],
            width=runtime["width"],
            rope_interpolation_scale=[1.0 / (30.0 / 8.0), 32, 32],
            return_dict=False,
            semantic_plan=semantic_plan,
            semantic_plan_times=semantic_times,
        )

    if len(capture["layers"]) != len(model.action_blocks):
        raise RuntimeError(
            f"captured {len(capture['layers'])} action layers, "
            f"expected {len(model.action_blocks)}"
        )
    maps = {}
    for layer_index, values in capture["layers"].items():
        maps[int(layer_index)] = extract_frame_map(
            values,
            view=0,
            time_index=CURRENT_OFFSET,
            temporal=runtime["temporal"],
            height=runtime["height"],
            width=runtime["width"],
        )
    return maps


def display_map(attention: np.ndarray) -> np.ndarray:
    normalized = normalize_map(attention)
    low, high = np.percentile(normalized, [50.0, 99.5])
    if high - low <= 1e-12:
        return normalized
    return np.clip((normalized - low) / (high - low), 0.0, 1.0) ** 1.7


def render_overlay(
    rgb: np.ndarray,
    attention: np.ndarray,
    *,
    gain: bool = False,
) -> np.ndarray:
    import matplotlib

    if gain:
        maximum = float(attention.max())
        heat = (
            np.zeros_like(attention, dtype=np.float32)
            if maximum <= 1e-12
            else np.asarray(attention, dtype=np.float32) / maximum
        )
        alpha = 0.52
    else:
        heat = display_map(attention)
        alpha = 0.55
    heat_image = Image.fromarray(heat.astype(np.float32))
    heat_image = heat_image.resize(
        (rgb.shape[1], rgb.shape[0]),
        Image.Resampling.BILINEAR,
    )
    color = matplotlib.colormaps["turbo"](np.asarray(heat_image))[..., :3]
    base = rgb.astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * base + alpha * color, 0.0, 1.0)


def save_outputs(
    runtime: dict[str, Any],
    plan_off_layers: dict[int, np.ndarray],
    plan_on_layers: dict[int, np.ndarray],
    source_rgb: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.asarray(Image.open(source_rgb).convert("RGB"))
    plan_off = aggregate_layer_maps(plan_off_layers)
    plan_on = aggregate_layer_maps(plan_on_layers)
    gain = positive_gain(plan_off, plan_on)
    off_overlay = render_overlay(rgb, plan_off)
    on_overlay = render_overlay(rgb, plan_on)
    gain_overlay = render_overlay(rgb, gain, gain=True)
    output_dir = source_rgb.parent
    paths = {
        "plan_off": output_dir / "action_attn_plan_off.png",
        "plan_on": output_dir / "action_attn_plan_on.png",
        "gain": output_dir / "action_attn_sg_gain.png",
        "comparison": output_dir / "action_attn_comparison.png",
        "layers": output_dir / "action_attn_layers.png",
        "maps": output_dir / "action_attn_maps.npz",
        "metadata": output_dir / "action_attn_metadata.json",
    }
    Image.fromarray((off_overlay * 255.0).astype(np.uint8)).save(
        paths["plan_off"]
    )
    Image.fromarray((on_overlay * 255.0).astype(np.uint8)).save(
        paths["plan_on"]
    )
    Image.fromarray((gain_overlay * 255.0).astype(np.uint8)).save(paths["gain"])

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = [
        (rgb, "RGB"),
        (off_overlay, "Plan off"),
        (on_overlay, "SG-WAM plan on"),
        (gain_overlay, "SG added focus"),
    ]
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(paths["comparison"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    layer_indices = sorted(plan_off_layers)
    blocks_per_row = 4
    row_groups = math.ceil(len(layer_indices) / blocks_per_row)
    fig, axes = plt.subplots(
        row_groups,
        blocks_per_row * 2,
        figsize=(16, 3.7 * row_groups),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for position, layer_index in enumerate(layer_indices):
        row = position // blocks_per_row
        pair_column = (position % blocks_per_row) * 2
        axes[row, pair_column].imshow(
            render_overlay(rgb, plan_off_layers[layer_index])
        )
        axes[row, pair_column].set_title(
            f"Block {layer_index} · off",
            fontsize=8,
        )
        axes[row, pair_column + 1].imshow(
            render_overlay(rgb, plan_on_layers[layer_index])
        )
        axes[row, pair_column + 1].set_title(
            f"Block {layer_index} · on",
            fontsize=8,
        )
        axes[row, pair_column].axis("off")
        axes[row, pair_column + 1].axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(paths["layers"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    off_stack = np.stack([plan_off_layers[index] for index in layer_indices])
    on_stack = np.stack([plan_on_layers[index] for index in layer_indices])
    np.savez_compressed(
        paths["maps"],
        layer_indices=np.asarray(layer_indices, dtype=np.int64),
        plan_off_layers=off_stack,
        plan_on_layers=on_stack,
        plan_off_mean=plan_off,
        plan_on_mean=plan_on,
        positive_gain=gain,
    )

    source_after = sha256_file(source_rgb)
    if source_after != runtime["source_sha256"]:
        raise RuntimeError("source rgb.png changed during attention export")
    metadata = {
        "dataset": DATASET_NAME,
        "episode_index": EPISODE_INDEX,
        "frame_index": FRAME_INDEX,
        "instruction": runtime["instruction"],
        "checkpoint": str(args.checkpoint_root),
        "source_rgb": str(source_rgb),
        "source_rgb_sha256_before": runtime["source_sha256"],
        "source_rgb_sha256_after": source_after,
        "temporal_window": [WINDOW_START, WINDOW_END - 1],
        "memory_frames": [WINDOW_START, FRAME_INDEX],
        "future_frames": [FRAME_INDEX + 1, WINDOW_END - 1],
        "attention_frame_time_index": CURRENT_OFFSET,
        "keyframe_indices_in_future": list(KEYFRAME_INDICES),
        "real_action_frame": FRAME_INDEX,
        "action_horizon": ACTION_HORIZON,
        "action_token_format": "[normalized action7; normalized state8]",
        "history_token_format": "[zeros7; normalized current state8]",
        "normalization": "lerobot_like_dataset mean/std",
        "training_config": str(args.training_config),
        "action_noise_strength": ACTION_NOISE_STRENGTH,
        "video_noise_strength": VIDEO_NOISE_STRENGTH,
        "seed": args.seed,
        "hook": "action_blocks[*].attn2 action-query -> video-token",
        "captured_blocks": layer_indices,
        "map_shape": list(plan_off.shape),
        "pairing_hashes": runtime["pairing_hashes"],
        "paired_non_plan_inputs_identical": True,
        "decode_metrics": runtime["decode_metrics"],
        "visualization": {
            "colormap": "turbo",
            "absolute_percentiles": [50.0, 99.5],
            "absolute_gamma": 1.7,
            "absolute_overlay_alpha": 0.55,
            "gain": "positive(normalize(plan_on)-normalize(plan_off))",
            "gain_overlay_alpha": 0.52,
        },
    }
    metadata["output_sha256"] = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key != "metadata"
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    args = parse_args()
    if not args.source_rgb.is_file():
        raise FileNotFoundError(args.source_rgb)
    runtime = prepare_runtime(args)
    runtime["capture"] = install_action_attention_capture(runtime["model"])

    print("running paired condition: plan off", flush=True)
    plan_off_layers = run_condition(runtime, use_plan=False)
    print("running paired condition: semantic plan on", flush=True)
    plan_on_layers = run_condition(runtime, use_plan=True)
    paths = save_outputs(
        runtime,
        plan_off_layers,
        plan_on_layers,
        args.source_rgb,
        args,
    )
    print(
        f"captured {len(plan_off_layers)} layers at "
        f"{runtime['height']}x{runtime['width']}",
        flush=True,
    )
    for key, path in paths.items():
        print(f"saved {key}: {path}", flush=True)


if __name__ == "__main__":
    main()
