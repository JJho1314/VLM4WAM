from typing import Dict, Optional, Union
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import save_model, load_file, save_file


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def count_model_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def load_index_file(index_filename):
    checkpoint_folder = os.path.split(index_filename)[0]
    with open(index_filename) as f:
        index = json.loads(f.read())

    if "weight_map" in index:
        index = index["weight_map"]
    checkpoint_files = sorted(list(set(index.values())))
    checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
    state_dict = {}
    for checkpoint_file in checkpoint_files:
        state_dict.update(load_file(checkpoint_file))
    return state_dict


def resolve_checkpoint_files(pretrained_ckpt: Union[str, os.PathLike]) -> list[Path]:
    """Resolve a checkpoint file, a single-file directory, or a sharded directory."""

    checkpoint_path = Path(pretrained_ckpt)
    if checkpoint_path.is_file():
        return [checkpoint_path]
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    index_path = checkpoint_path / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", index)
        filenames = list(dict.fromkeys(weight_map.values()))
        files = [checkpoint_path / filename for filename in filenames]
    else:
        canonical_file = checkpoint_path / "diffusion_pytorch_model.safetensors"
        if canonical_file.is_file():
            files = [canonical_file]
        else:
            files = sorted(checkpoint_path.glob("*.safetensors"))
            if len(files) != 1:
                raise FileNotFoundError(
                    f"expected one safetensors file or a shard index in {checkpoint_path}, found {len(files)}"
                )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint index references missing shards: {missing}")
    return files

def _find_mismatched_keys(
    state_dict,
    model_state_dict,
    loaded_keys,
):
    mismatched_keys = []
    for checkpoint_key in loaded_keys:
        model_key = checkpoint_key

        if (
            model_key in model_state_dict
            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
        ):
            mismatched_keys.append(
                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
            )
            del state_dict[checkpoint_key]

    return mismatched_keys

def load_checkpoints(model, pretrained_ckpt, strict=False, ignore_mismatched_sizes=True):
    """
    Load safetensors model state dict file.
    """

    state_dict = {}
    for checkpoint_file in resolve_checkpoint_files(pretrained_ckpt):
        state_dict.update(load_file(str(checkpoint_file)))

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        if ignore_mismatched_sizes:
            model_state_dict = model.state_dict()
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                list(state_dict.keys()),
            )
        else:
            mismatched_keys = []
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(">>> mismatched_keys: %s" % mismatched_keys)
        print(">>> missing: %s" % missing)
        print(">>> unexpected: %s" % unexpected)
    print(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)



def load_condition_models(
    tokenizer_class,
    textenc_class,
    model_id: str = "a-r-r-o-w/LTX-Video-0.9.1-diffusers",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    load_weights: bool = True,
    **kwargs,
) -> Dict[str, nn.Module]:
    tokenizer = tokenizer_class.from_pretrained(
        model_id,
        subfolder="tokenizer",
        revision=revision,
        cache_dir=cache_dir
    )

    if load_weights:
        text_encoder = textenc_class.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=text_encoder_dtype,
            revision=revision,
            cache_dir=cache_dir
        )
    else:
        # logger.warning('You are not lodding the checkpoint of the text Embedder, please check the code!!!')
        config = textenc_class.config_class.from_pretrained(
            model_id,
            subfolder="text_encoder",
            revision=revision,
            cache_dir=cache_dir
        )
        text_encoder = textenc_class(config)  # 仅初始化模型，不加载权重

    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_cls,
    model_id,
    vae_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    vae = model_cls.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_model(model_cls, model_dir, load_weights=True, **kwargs):
    model = model_cls(**kwargs)
    print(model_dir)
    if load_weights:
        load_checkpoints(model, pretrained_ckpt=model_dir)
    return model


def load_vae_models(model_cls, model_dir, load_weights=True):
    with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as f:
        vae_kwargs = json.load(f)
    model = model_cls(**vae_kwargs)
    if load_weights:
        load_checkpoints(model, pretrained_ckpt=os.path.join(model_dir, 'diffusion_pytorch_model.safetensors'))
    else:
        print('You are not loading the weights of the vae model, please check your code.')
        pass
    return model


def forward_pass(
    model,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    num_frames: int,
    height: int,
    width: int,
    n_view: int = 1,
    frame_rate = 30,
    temporal_compression_ratio = 8,
    spatial_compression_ratio = 32,
    **kwargs,
) -> torch.Tensor:
    latent_frame_rate = frame_rate / temporal_compression_ratio
    rope_interpolation_scale = [1 / latent_frame_rate, spatial_compression_ratio, spatial_compression_ratio]
    
    denoised_latents = model(
        hidden_states=noisy_latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        encoder_attention_mask=prompt_attention_mask,
        num_frames=num_frames,
        height=height,
        width=width,
        n_view=n_view,
        rope_interpolation_scale=rope_interpolation_scale,
        return_dict=False,
        **kwargs,
    )[0]
    return {"latents": denoised_latents}
