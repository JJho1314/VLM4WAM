import os, random, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from datetime import datetime, timedelta
import argparse
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib

from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video

# ----------------------------------------------------
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

# ----------------------------------------------------
from utils.extra_utils import act_metric
from models.ltx_models.semantic_conditioning import (
    OnlineSiglip2SemanticEncoder,
    build_semantic_plan_times,
    select_future_keyframes,
)
from models.ltx_models.vlm_semantic_planner import FrozenDualCameraVLMPlanner
from models.ltx_models.joint_vlm_geact import (  # noqa: F401
    JointVLMGEActModel,
    build_joint_optimizer_parameter_groups,
)

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
logger = get_logger("wm_runner")
logger.setLevel(LOG_LEVEL)


def compute_effective_video_fps(data_config: Dict[str, Any], default_source_fps: float = 30.0) -> float:
    """Convert source control FPS to the sampled video FPS used by GE-Act."""

    source_fps = float(data_config.get("source_fps", default_source_fps))
    chunk = int(data_config["chunk"])
    action_chunk = int(data_config["action_chunk"])
    if chunk <= 0 or action_chunk % chunk != 0:
        raise ValueError("action_chunk must be an integer multiple of chunk")
    return source_fps / (action_chunk // chunk)


def _joint_training_enabled(args: argparse.Namespace) -> bool:
    config = getattr(args, "joint_training", {})
    return isinstance(config, dict) and bool(config.get("enabled", False))


def select_joint_planner_frames(
    video: torch.Tensor,
    n_previous: int,
    offsets: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the planner current frame and approved K future source frames.

    Planner offsets are relative to ``n_previous`` (the first future position),
    not to the zero-based current frame. Thus ``n_previous=4`` with offsets
    ``(2,4,6,8)`` selects source indices ``(6,8,10,12)``.
    """

    if video.ndim != 6 or video.shape[1] != 3 or video.shape[2] != 2:
        raise ValueError(f"video must be [B,3,2,T,H,W], got {tuple(video.shape)}")
    current_index = int(n_previous) - 1
    if current_index < 0 or current_index >= video.shape[3]:
        raise ValueError(
            f"n_previous={n_previous} selects invalid current index {current_index}"
        )
    resolved_offsets = tuple(int(offset) for offset in offsets)
    if (
        not resolved_offsets
        or any(offset <= 0 for offset in resolved_offsets)
        or any(
            left >= right
            for left, right in zip(resolved_offsets, resolved_offsets[1:])
        )
    ):
        raise ValueError(
            "joint planner offsets must be strictly increasing positive integers, "
            f"got {resolved_offsets}"
        )
    future_indices = tuple(int(n_previous) + offset for offset in resolved_offsets)
    if future_indices[-1] >= video.shape[3]:
        raise ValueError(
            f"joint planner source index {future_indices[-1]} exceeds T={video.shape[3]}"
        )
    current = video[:, :, :, current_index].permute(0, 2, 3, 4, 1).contiguous()
    future = torch.stack(
        [
            video[:, :, :, source_index].permute(0, 2, 3, 4, 1)
            for source_index in future_indices
        ],
        dim=2,
    ).contiguous()
    return current, future


def encode_joint_planner_targets(
    current: torch.Tensor,
    future: torch.Tensor,
    *,
    semantic_teacher: Any,
    depth_teacher: Any,
    target_encoder: Callable[..., dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Run both online teachers with autograd disabled."""

    with torch.no_grad():
        targets = target_encoder(
            current,
            future,
            appearance_encoder=semantic_teacher,
            depth_encoder=depth_teacher,
        )
    return {name: value.detach() for name, value in targets.items()}


def combine_joint_training_loss(
    loss_video: torch.Tensor,
    planner_losses: Dict[str, torch.Tensor],
    *,
    planner_loss_weight: float,
) -> torch.Tensor:
    planner_loss = planner_losses.get("loss")
    if not torch.is_tensor(loss_video) or not torch.is_tensor(planner_loss):
        raise TypeError("joint video and planner losses must be tensors")
    return loss_video + float(planner_loss_weight) * planner_loss


def _configure_qwen_gradient_checkpointing(
    model: torch.nn.Module,
    *,
    enabled: bool,
) -> None:
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    elif not enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()


def _gradient_norm(parameters) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    return torch.stack([gradient.to(gradients[0].device) for gradient in gradients]).norm(2)


def build_deepspeed_batch_config(
    deepspeed_config: Dict[str, Any],
    *,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> Dict[str, Any]:
    config = dict(deepspeed_config)
    config["train_batch_size"] = (
        per_device_batch_size * world_size * gradient_accumulation_steps
    )
    # Accelerate defaults this field to 1 when an explicit DeepSpeed config is
    # supplied, even if DeepSpeedPlugin receives a different constructor value.
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    return config


def compute_ltx_latent_frames(
    raw_future_frames: int,
    temporal_compression_ratio: int,
    n_previous: int,
) -> int:
    """LTX temporal VAE keeps the first frame and then compresses intervals."""

    if raw_future_frames < 1:
        raise ValueError("at least one future frame is required")
    return (raw_future_frames - 1) // temporal_compression_ratio + 1 + n_previous


def sample_semantic_condition_mask(
    batch_size: int,
    n_view: int,
    dropout_probability: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one semantic keep/drop decision per scene and share it across views."""

    if not 0 <= dropout_probability <= 1:
        raise ValueError("semantic dropout probability must be in [0, 1]")
    keep = torch.rand(batch_size, device=device, generator=generator) >= dropout_probability
    return keep.to(dtype=dtype).repeat_interleave(n_view)


def build_vlm_semantic_condition(
    provider,
    video: torch.Tensor,
    instructions,
    *,
    n_previous: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict from the last current observation without exposing future RGB."""

    if video.ndim != 6 or video.shape[1] != 3 or video.shape[2] != 2:
        raise ValueError(f"video must be [B,3,2,T,H,W], got {tuple(video.shape)}")
    current_index = int(n_previous) - 1
    if current_index < 0 or current_index >= video.shape[3]:
        raise ValueError(
            f"n_previous={n_previous} selects invalid current index {current_index}"
        )
    current_images = video[:, :, :, current_index].permute(0, 2, 1, 3, 4)
    plan = provider.predict(current_images.contiguous(), instructions)
    num_keyframes = int(provider.num_keyframes)
    tokens_per_keyframe = int(provider.target_tokens_per_keyframe)
    expected_tokens = (
        video.shape[0],
        2,
        num_keyframes,
        tokens_per_keyframe,
        1024,
    )
    expected_times = (video.shape[0] * 2, num_keyframes)
    if tuple(plan.semantic_tokens.shape) != expected_tokens:
        raise RuntimeError(
            "VLM semantic tokens must have shape "
            f"{expected_tokens}, got {tuple(plan.semantic_tokens.shape)}"
        )
    if tuple(plan.times.shape) != expected_times:
        raise RuntimeError(
            f"VLM semantic times must have shape {expected_times}, "
            f"got {tuple(plan.times.shape)}"
        )
    return plan.semantic_tokens, plan.times


def freeze_conditioning_modules(*components) -> None:
    """Explicitly freeze and eval all condition-only neural networks."""

    for component in components:
        if component is None:
            continue
        modules = [component, getattr(component, "model", component)]
        for module in modules:
            if isinstance(module, torch.nn.Module):
                module.requires_grad_(False)
                module.eval()


def _is_semantic_parameter(name: str) -> bool:
    return name.startswith("semantic_") or ".semantic_" in name


def build_optimizer_parameter_groups(
    model: torch.nn.Module,
    train_mode: str,
    base_lr: float,
    semantic_lr: float,
) -> List[Dict[str, Any]]:
    """Apply GE-Act train-mode filtering and split semantic parameters by LR."""

    base_parameters = []
    semantic_parameters = []
    for name, parameter in model.named_parameters():
        is_action = "action_" in name
        if train_mode == "action_only":
            trainable = is_action
        elif train_mode == "video_only":
            trainable = not is_action
        elif train_mode in ("all", "action_full"):
            trainable = True
        else:
            raise NotImplementedError(f"unknown train mode: {train_mode}")
        parameter.requires_grad_(trainable)
        if not trainable:
            continue
        if _is_semantic_parameter(name):
            semantic_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    groups = []
    if base_parameters:
        groups.append({"name": "base_ltx", "params": base_parameters, "lr": base_lr})
    if semantic_parameters:
        groups.append({"name": "semantic", "params": semantic_parameters, "lr": semantic_lr})
    return groups


def should_save_checkpoint(global_step: int, args: argparse.Namespace) -> bool:
    explicit_steps = getattr(args, "save_steps", None)
    if explicit_steps:
        return global_step in {int(step) for step in explicit_steps}
    return global_step > 0 and global_step % int(args.steps_to_save) == 0


def _export_joint_planner(
    planner: torch.nn.Module,
    processor: Any,
    output_dir: Path,
    *,
    source_checkpoint: str | Path,
    global_step: int,
) -> None:
    """Write a provider-compatible standalone planner export."""

    output_dir.mkdir(parents=True, exist_ok=True)
    planner.model.save_pretrained(output_dir / "qwen3vl_lora_or_model")
    processor.save_pretrained(output_dir / "processor")
    torch.save(planner.plan_head.state_dict(), output_dir / "plan_head.pt")
    torch.save(planner.depth_head.state_dict(), output_dir / "depth_head.pt")
    plan_embedding_injector = getattr(planner, "plan_embedding_injector", None)
    if bool(getattr(planner, "uses_pooled_head_query_embeddings", False)):
        plan_embedding = planner.pooled_lingbot_prefix_queries().detach().cpu()
    elif plan_embedding_injector is not None:
        plan_embedding = plan_embedding_injector.weight.detach().cpu()
    else:
        plan_ids = torch.as_tensor(planner.plan_token_ids)
        plan_embedding = (
            planner.model.get_input_embeddings().weight[plan_ids].detach().cpu()
        )
    torch.save(plan_embedding, output_dir / "plan_token_embedding.pt")

    source_meta_path = Path(source_checkpoint) / "planner_meta.json"
    try:
        metadata = json.loads(source_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot export joint planner without valid metadata: {source_meta_path}"
        ) from error
    metadata["step"] = int(global_step)
    metadata["joint_finetune_source"] = str(source_checkpoint)
    (output_dir / "planner_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_joint_checkpoint(
    *,
    accelerator: Accelerator,
    joint_model: torch.nn.Module,
    planner_provider: FrozenDualCameraVLMPlanner,
    step_dir: str | Path,
    args: argparse.Namespace,
    global_step: int,
) -> None:
    """Save model exports on main and exact distributed state on every rank."""

    step_dir = Path(step_dir)
    if accelerator.is_main_process:
        step_dir.mkdir(parents=True, exist_ok=True)
        model = accelerator.unwrap_model(joint_model)
        model.ltx.save_pretrained(
            step_dir / "ltx",
            safe_serialization=True,
        )
        planner_checkpoint = args.semantic_plan["planner_checkpoint"]
        _export_joint_planner(
            model.planner,
            planner_provider.processor,
            step_dir / "planner",
            source_checkpoint=planner_checkpoint,
            global_step=global_step,
        )
        joint_config = args.joint_training
        source_planner_metadata = json.loads(
            (Path(planner_checkpoint) / "planner_meta.json").read_text(
                encoding="utf-8"
            )
        )
        metadata = {
            "global_step": int(global_step),
            "source_planner_checkpoint": str(planner_checkpoint),
            "source_ltx_checkpoint": str(args.diffusion_model["model_path"]),
            "planner_loss_weight": float(joint_config["planner_loss_weight"]),
            "optimizer_group_lrs": {
                "base_ltx": float(args.lr),
                "semantic_ltx": float(args.semantic_lr),
                "qwen": float(joint_config["qwen_lr"]),
                "planner_heads": float(joint_config["planner_head_lr"]),
            },
            "future_keyframe_offsets": [
                int(value)
                for value in getattr(
                    planner_provider,
                    "future_keyframe_offsets",
                    source_planner_metadata["future_keyframe_offsets"],
                )
            ],
            "num_keyframes": int(
                getattr(
                    planner_provider,
                    "num_keyframes",
                    source_planner_metadata["num_keyframes"],
                )
            ),
            "tokens_per_keyframe": int(
                getattr(
                    planner_provider,
                    "target_tokens_per_keyframe",
                    source_planner_metadata["target_tokens_per_keyframe"],
                )
            ),
            "num_camera_views": 2,
            "global_batch_size": int(args.batch_size)
            * int(args.gradient_accumulation_steps)
            * int(accelerator.num_processes),
            "trainable_parameters": {
                "ltx": sum(
                    parameter.numel()
                    for parameter in model.ltx.parameters()
                    if parameter.requires_grad
                ),
                "planner": sum(
                    parameter.numel()
                    for parameter in model.planner.parameters()
                    if parameter.requires_grad
                ),
            },
        }
        (step_dir / "joint_meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    accelerator.save_state(step_dir / "training_state")
    accelerator.wait_for_everyone()


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None



class Trainer:

    def __init__(
        self,
        config_file,
        config_overrides=None,
        to_log=True,
        output_dir=None,
    ) -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        if config_overrides:
            cd.update(dict(config_overrides))
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.semantic_lr = float(getattr(args, "semantic_lr", args.lr))
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')
        self.state = State()

        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        self.semantic_encoder = None
        self.semantic_planner = None
        self.semantic_teacher = None
        self.depth_teacher = None
        self.joint_model = None
        self.joint_target_encoder = None
        self.video_frame_rate = compute_effective_video_fps(self.args.data["train"])

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if self.state.accelerator.is_main_process:

            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for k, v in args_dict.items():
                args_dict[k] = str(v)
            with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)
            
            if to_log:
                self.writer = SummaryWriter(log_dir=self.save_folder)
            else:
                self.writer = None

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

        init_logging(self.save_folder, rank=self.state.accelerator.process_index)


    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))  # 或 self.args.world_size
            grad_accum = self.args.gradient_accumulation_steps

            self.args.deepspeed = build_deepspeed_batch_config(
                self.args.deepspeed,
                per_device_batch_size=per_device_bs,
                world_size=world_size,
                gradient_accumulation_steps=grad_accum,
            )
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype


    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)
        

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir


    def prepare_dataset(self) -> None:

        logger.info(f"Training Dataset: {self.args.train_data_class}")
        train_dataset_class = import_custom_class(
            self.args.train_data_class, self.args.train_data_class_path
        )
        self.train_dataset = train_dataset_class(**self.args.data['train'])

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=getattr(self.args, "pin_memory", False),
            persistent_workers=self.args.dataloader_num_workers > 0,
            prefetch_factor=(
                getattr(self.args, "dataloader_prefetch_factor", 2)
                if self.args.dataloader_num_workers > 0
                else None
            ),
            multiprocessing_context=None,
        )
        logger.info(f">>>>>>>>>>>>>Total Train Eps: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")


        if 'val' in self.args.data:
            self.prepare_val_dataset()


    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        self.val_dataset = val_dataset_class(**self.args.data['val'])

        self.val_index = []
        for _ in range(self.args.batch_size):
            self.val_index.append(random.randint(0, len(self.val_dataset)-1))
        if self.state.accelerator.is_main_process:
            with open(os.path.join(self.save_folder, 'idx.txt'), "w") as file:
                file.write(", ".join(map(str, self.val_index)))

        subset = torch.utils.data.Subset(self.val_dataset, self.val_index)
        self.val_dataloader = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size, shuffle=getattr(self.args, "val_shuffle", False)
        )
        logger.info(f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")


    def prepare_models(self):

        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=True
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_encoder.requires_grad_(False)
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            self.vae.requires_grad_(False)
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        logger.info(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        logger.info(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        logger.info(f'Total parameters for transformer model:{total_params}')

        semantic_config = getattr(self.args, "semantic_plan", {})
        joint_enabled = _joint_training_enabled(self.args)
        if semantic_config.get("enabled", False):
            semantic_source = semantic_config.get("source", "gt_siglip2")
            if semantic_source == "gt_siglip2":
                if joint_enabled:
                    raise ValueError(
                        "joint training requires semantic_plan.source=vlm_planner"
                    )
                self.semantic_encoder = OnlineSiglip2SemanticEncoder(
                    semantic_config["model_name_or_path"],
                    device=device,
                    dtype=dtype,
                    frame_microbatch_size=int(semantic_config.get("frame_microbatch_size", 32)),
                    expected_tokens=int(semantic_config.get("tokens_per_frame", 256)),
                    expected_feature_dim=int(semantic_config.get("feature_dim", 1024)),
                )
            elif semantic_source == "vlm_planner":
                self.semantic_planner = FrozenDualCameraVLMPlanner.from_checkpoint(
                    semantic_config["planner_checkpoint"],
                    device=device,
                    dtype=dtype,
                )
                if joint_enabled:
                    if (
                        self.semantic_planner.num_keyframes != 4
                        or self.semantic_planner.future_keyframe_offsets
                        != (2, 4, 6, 8)
                        or self.semantic_planner.target_tokens_per_keyframe != 256
                    ):
                        raise ValueError(
                            "joint training requires dual-camera K4 planner geometry "
                            "with offsets (2,4,6,8) and 256 tokens/keyframe"
                        )
                    joint_config = self.args.joint_training
                    from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_target import (
                        Siglip2TargetEncoder,
                    )
                    from qwen3_vl_semantic_planner.dinov3_da3_2b.depth_anything3_target import (
                        DepthAnything3TargetEncoder,
                    )
                    from qwen3_vl_semantic_planner.train_qwen3vl4b_lingbot_dino_planner import (
                        encode_dual_camera_future_targets,
                    )

                    self.semantic_teacher = Siglip2TargetEncoder(
                        joint_config["siglip2_model_dir"],
                        input_size=int(joint_config.get("siglip2_input_size", 256)),
                        grid_size=16,
                        device=device,
                        dtype=dtype,
                    )
                    self.depth_teacher = DepthAnything3TargetEncoder(
                        joint_config["da3_ckpt_dir"],
                        process_res=int(joint_config.get("da3_process_res", 224)),
                        feature_slice="full",
                        align_strategy="wsa_multilayer",
                        teacher_layers=(11, 15, 19, 23),
                        layer_weights=(1.0, 1.2, 1.4, 1.6),
                        device=device,
                        dtype=dtype,
                        code_root=joint_config["da3_code_root"],
                    )
                    self.joint_target_encoder = encode_dual_camera_future_targets
                    self.semantic_planner.wrapper.requires_grad_(True)
                    self.semantic_planner.wrapper.train()
                    self.joint_model = JointVLMGEActModel(
                        self.semantic_planner.wrapper,
                        self.diffusion_model,
                        num_keyframes=4,
                        tokens_per_keyframe=256,
                    )
            else:
                raise ValueError(f"unknown semantic_plan.source: {semantic_source}")
        elif joint_enabled:
            raise ValueError("joint training requires semantic_plan.enabled=true")


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        joint_enabled = _joint_training_enabled(self.args)
        freeze_conditioning_modules(
            self.text_encoder,
            self.vae,
            self.semantic_encoder,
            self.semantic_teacher,
            self.depth_teacher,
            (
                None
                if joint_enabled
                else getattr(self.semantic_planner, "wrapper", None)
            ),
        )

        if joint_enabled:
            if self.joint_model is None or self.semantic_planner is None:
                raise RuntimeError("joint models must be prepared before trainable parameters")
            self.semantic_planner.wrapper.requires_grad_(True)
            self.semantic_planner.wrapper.train()
            _configure_qwen_gradient_checkpointing(
                self.semantic_planner.wrapper.model,
                enabled=bool(
                    self.args.joint_training.get(
                        "qwen_gradient_checkpointing",
                        True,
                    )
                ),
            )

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True


    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        if _joint_training_enabled(self.args):
            if self.joint_model is None:
                raise RuntimeError("joint model must be prepared before optimizer")
            if self.args.mixed_precision == "fp16":
                cast_training_params([self.joint_model], dtype=torch.float32)

            lr_scale = 1.0
            if self.args.scale_lr:
                lr_scale = (
                    self.args.gradient_accumulation_steps
                    * self.args.batch_size
                    * self.state.accelerator.num_processes
                )
            joint_config = self.args.joint_training
            params_to_optimize = build_joint_optimizer_parameter_groups(
                self.joint_model,
                ltx_lr=float(self.args.lr) * lr_scale,
                semantic_lr=float(self.args.semantic_lr) * lr_scale,
                qwen_lr=float(joint_config["qwen_lr"]) * lr_scale,
                planner_head_lr=float(joint_config["planner_head_lr"]) * lr_scale,
            )
            trainable_parameters = [
                parameter
                for group in params_to_optimize
                for parameter in group["params"]
            ]
            self.state.learning_rate = float(self.args.lr) * lr_scale
            self.state.num_trainable_parameters = sum(
                parameter.numel() for parameter in trainable_parameters
            )
            logger.info(
                "Joint optimizer groups: %s",
                {
                    group["name"]: {
                        "lr": group["lr"],
                        "params": sum(p.numel() for p in group["params"]),
                    }
                    for group in params_to_optimize
                },
            )
            self.optimizer = get_optimizer(
                params_to_optimize=params_to_optimize,
                optimizer_name=self.args.optimizer,
                learning_rate=self.args.lr,
                beta1=self.args.beta1,
                beta2=self.args.beta2,
                beta3=self.args.beta3,
                epsilon=self.args.epsilon,
                weight_decay=self.args.weight_decay,
                use_8bit=self.args.optimizer_8bit,
                use_torchao=self.args.optimizer_torchao,
            )
            num_update_steps_per_epoch = math.ceil(
                len(self.train_dataloader)
                / self.args.gradient_accumulation_steps
            )
            if self.state.train_steps is None:
                self.state.train_steps = (
                    self.state.train_epochs * num_update_steps_per_epoch
                )
                self.state.overwrote_max_train_steps = True
            self.lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=(
                    self.args.lr_warmup_steps
                    * self.state.accelerator.num_processes
                ),
                num_training_steps=(
                    self.state.train_steps
                    * self.state.accelerator.num_processes
                ),
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )
            return

        train_mode = self.args.train_mode

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        semantic_learning_rate = self.args.semantic_lr
        if self.args.scale_lr:
            lr_scale = (
                self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )
            self.state.learning_rate *= lr_scale
            semantic_learning_rate *= lr_scale

        params_to_optimize = build_optimizer_parameter_groups(
            self.diffusion_model,
            train_mode=train_mode,
            base_lr=self.state.learning_rate,
            semantic_lr=semantic_learning_rate,
        )
        diffusion_model_trainable_params = [
            parameter for group in params_to_optimize for parameter in group["params"]
        ]
        num_trainable_params = sum(parameter.numel() for parameter in diffusion_model_trainable_params)
        logger.info(f'Total trainable parameters: {num_trainable_params}')
        logger.info(
            "Optimizer groups: %s",
            {group["name"]: {"lr": group["lr"], "params": sum(p.numel() for p in group["params"])} for group in params_to_optimize},
        )
        self.state.num_trainable_parameters = sum(p.numel() for p in diffusion_model_trainable_params)

        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        

    def prepare_for_training(self):
        if _joint_training_enabled(self.args):
            self.joint_model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
                self.state.accelerator.prepare(
                    self.joint_model,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                )
            )
        else:
            self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
                self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )


    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)


    def train(self):
        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        
        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        # loss spikes
        anomalies = []
        joint_enabled = _joint_training_enabled(self.args)

        for epoch in range(first_epoch, self.state.train_epochs):
            if global_step >= self.state.train_steps:
                break

            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            if joint_enabled:
                self.joint_model.train()
            else:
                self.diffusion_model.train()

            running_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                joint_grad_metrics = {}
                accumulation_context = (
                    accelerator.accumulate(self.joint_model)
                    if joint_enabled
                    else accelerator.accumulate([self.diffusion_model])
                )
                with accumulation_context:
                    
                    video = batch['video']

                    # shape: {b, c, v, t, h, w}; ranging from -1 to 1
                    video = video.to(accelerator.device, dtype=weight_dtype).contiguous()
                    batch_size, c, n_view, _, h, w = video.shape
                    mem_size = self.args.data['train']['n_previous']
                    semantic_keyframes = None
                    planner_semantic_plan = None
                    planner_semantic_times = None
                    planner_inputs = None
                    planner_targets = None
                    joint_output = None
                    planner_metrics = None
                    if joint_enabled:
                        current_frames, future_frames = select_joint_planner_frames(
                            video,
                            n_previous=mem_size,
                            offsets=self.semantic_planner.future_keyframe_offsets,
                        )
                        planner_inputs = self.semantic_planner.prepare_inputs(
                            current_frames.permute(0, 1, 4, 2, 3).contiguous(),
                            batch['caption'],
                        )
                        planner_targets = encode_joint_planner_targets(
                            current_frames,
                            future_frames,
                            semantic_teacher=self.semantic_teacher,
                            depth_teacher=self.depth_teacher,
                            target_encoder=self.joint_target_encoder,
                        )
                    if self.semantic_encoder is not None:
                        semantic_config = self.args.semantic_plan
                        semantic_future = rearrange(
                            video[:, :, :, mem_size:],
                            'b c v t h w -> b v t c h w',
                        )
                        semantic_keyframes = select_future_keyframes(
                            semantic_future,
                            indices=tuple(semantic_config.get("keyframe_indices", (0, 3, 5, 8))),
                        ).contiguous()
                    elif self.semantic_planner is not None and not joint_enabled:
                        planner_semantic_plan, planner_semantic_times = (
                            build_vlm_semantic_condition(
                                self.semantic_planner,
                                video,
                                batch['caption'],
                                n_previous=mem_size,
                            )
                        )
                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')

                    # here we use color jitter to the video, with different views or different batches different jitter
                    if self.args.use_color_jitter:
                        video = apply_color_jitter_to_video(video)

                    mem = video[:,:,:mem_size]
                    future_video = video[:,:,mem_size:]

                    if self.args.return_action:
                        future_video = future_video[:,:,:1].repeat(1,1,self.args.data['train']['chunk'],1,1)

                    # get the shape params
                    _, _, raw_frames, raw_height, raw_width = future_video.shape

                    latent_frames = compute_ltx_latent_frames(
                        raw_frames,
                        temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                        n_previous=mem_size,
                    )
                    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
                    latent_width = raw_width // self.SPATIAL_DOWN_RATIO

                    semantic_plan = None
                    semantic_plan_times = None
                    semantic_condition_mask = None
                    if self.semantic_encoder is not None:
                        semantic_plan = self.semantic_encoder.encode(semantic_keyframes)
                        semantic_plan_times = build_semantic_plan_times(
                            batch_size=batch_size,
                            n_view=n_view,
                            n_previous=mem_size,
                            num_future_frames=raw_frames,
                            num_latent_frames=latent_frames,
                            indices=tuple(self.args.semantic_plan.get("keyframe_indices", (0, 3, 5, 8))),
                            device=accelerator.device,
                        )
                    elif self.semantic_planner is not None and not joint_enabled:
                        semantic_plan = planner_semantic_plan.to(
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )
                        semantic_plan_times = planner_semantic_times.to(
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                    elif joint_enabled:
                        semantic_plan_times = (
                            torch.tensor(
                                self.semantic_planner.future_keyframe_offsets,
                                device=accelerator.device,
                                dtype=torch.float32,
                            )
                            / float(self.semantic_planner.sequence_length - 1)
                        ).reshape(1, 4).expand(batch_size * n_view, -1).clone()
                    if semantic_plan is not None or joint_enabled:
                        semantic_condition_mask = sample_semantic_condition_mask(
                            batch_size=batch_size,
                            n_view=n_view,
                            dropout_probability=float(self.args.semantic_plan.get("dropout", 0.15)),
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )

                    dropout_factor = torch.rand(batch_size).to(accelerator.device, dtype=weight_dtype)
                    dropout_mask_prompt = dropout_factor < self.args.caption_dropout_p
                    dropout_mask_prompt = dropout_mask_prompt.unsqueeze(1).unsqueeze(2)

                    # In action training with noisy_video, the future clip is replaced by pure
                    # noise (ss=1) and carries no video loss, so its VAE encode (~32% of a step,
                    # here of a repeated dummy frame) is pure waste — skip it.
                    _skip_future_encode = (
                        getattr(self.args, "noisy_video", False)
                        and self.args.train_mode in ("action_only", "action_full")
                    )
                    mem_latents, future_video_latents = get_latents(
                        self.vae, mem, future_video, encode_future=not _skip_future_encode
                    )

                    mem_latents = rearrange(mem_latents, '(b v m) (h w) c -> (b v) c m h w', b=batch_size, m=mem_size, h=latent_height)
                    future_video_latents = rearrange(future_video_latents, '(b v) (f h w) c -> (b v) c f h w',b=batch_size,h=latent_height,w=latent_width)
                    latents = torch.cat((mem_latents, future_video_latents), dim=2)

                    video_attention_mask = None
                    latents = rearrange(latents, 'bv c f h w -> bv (f h w) c')

                    captions = batch['caption']
                    text_conds = get_text_conditions(self.tokenizer,self.text_encoder,captions)
                    prompt_embeds = text_conds['prompt_embeds']
                    prompt_attention_mask = text_conds['prompt_attention_mask']
                    prompt_embeds = self.uncond_prompt_embeds.repeat(batch_size,1,1)*dropout_mask_prompt + \
                                    prompt_embeds*~dropout_mask_prompt

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    action_weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    # 0-1, 0 -> most noisy, 1 -> almost clean
                    action_indices = (action_weights * self.scheduler.config.num_train_timesteps).long()
                    action_sigmas = scheduler_sigmas[action_indices]
                    action_timesteps = (action_sigmas * 1000.0).long()

                    if self.args.return_action and self.args.noisy_video:
                        weights = torch.full_like(action_weights, 0.0).unsqueeze(1).repeat(1,n_view)
                    else:
                        weights = action_weights.unsqueeze(1).repeat(1,n_view)

                    weights = rearrange(weights, 'b v -> (b v)')
                    indices = (weights * self.scheduler.config.num_train_timesteps).long()
                    sigmas = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()

                    if self.args.return_action:
                        if getattr(self.args, "add_state", False):
                            # NOTE add states from the batch:
                            act_state = batch['state']
                            if act_state.shape[1] != 1:
                                act_state = act_state[:, mem_size-1:mem_size]
                            act_state = act_state.to(accelerator.device, dtype=weight_dtype).contiguous()
                        else:
                            act_state = None
                            

                        actions = batch['actions'][:, -self.args.data['train']['action_chunk']:].to(accelerator.device, dtype=weight_dtype).contiguous()   # shape b,t,c
                        action_dim = actions.shape[-1]

                        noise_actions = randn_tensor(actions.shape, device=accelerator.device, dtype=weight_dtype)

                        # here we get action_timesteps, shape (b,) originally, target shape (b, l) 
                        action_timesteps = action_timesteps.unsqueeze(-1).repeat(1, actions.shape[1])
                        action_ss= action_sigmas.reshape(-1, 1, 1).repeat(1, 1, actions.shape[-1])

                        noisy_actions = (1.0 - action_ss) * actions + action_ss * noise_actions

                        action_weights = compute_loss_weighting_for_sd3(
                            weighting_scheme=self.args.flow_weighting_scheme, sigmas=action_sigmas
                        ).reshape(-1, 1, 1).repeat(1, 1, actions.size(-1))
                    else:
                        actions = None
                        action_timesteps = None
                        noisy_actions = None
                        act_state = None

                    # shape:  bv, l, c and bv, l
                    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                        mem_latents, latent_frames, latent_height, latent_width, noise_to_condition_frames=self.args.noise_to_first_frame
                    )  # set initial frames noise to 0
                    if self.args.pixel_wise_timestep:
                        # shape: bv, thw
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        # shape: bv, t
                        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

                    # shape: bv,1,c
                    ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    if self.args.return_action and self.args.noisy_video:
                        ss = torch.full_like(ss, 1.0)

                    noisy_latents = (1.0 - ss) * latents + ss * noise

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss, shape bv,1,c
                    weights = compute_loss_weighting_for_sd3(
                        weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas
                    ).reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))

                    ltx_inputs = {
                        "timesteps": timesteps,
                        "noisy_latents": noisy_latents,
                        "prompt_embeds": prompt_embeds,
                        "prompt_attention_mask": prompt_attention_mask,
                        "num_frames": latent_frames,
                        "height": latent_height,
                        "width": latent_width,
                        "n_view": n_view,
                        "action_states": noisy_actions,
                        "action_timestep": action_timesteps,
                        "return_video": self.args.return_video or self.args.return_action,
                        "return_action": self.args.return_action,
                        "video_attention_mask": video_attention_mask,
                        "history_action_state": act_state,
                        "condition_mask": conditioning_mask,
                        "frame_rate": self.video_frame_rate,
                        "temporal_compression_ratio": self.TEMPORAL_DOWN_RATIO,
                        "spatial_compression_ratio": self.SPATIAL_DOWN_RATIO,
                        "semantic_plan_times": semantic_plan_times,
                        "semantic_condition_mask": semantic_condition_mask,
                    }
                    if joint_enabled:
                        joint_output = self.joint_model(
                            planner_inputs=planner_inputs,
                            semantic_labels=planner_targets["semantic_plan_labels"],
                            depth_labels=planner_targets["depth_plan_labels"],
                            ltx_inputs=ltx_inputs,
                        )
                        pred_all = joint_output.ltx_predictions
                        planner_metrics = joint_output.planner_losses
                    else:
                        pred_all = forward_pass(
                            model=self.diffusion_model,
                            semantic_plan=semantic_plan,
                            **ltx_inputs,
                        )["latents"]

                    if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                        pred = pred_all['video']
                        target = noise - latents
                        loss_video = weights.float() * (pred.float() - target.float()).pow(2)
                        loss_video = loss_video * (1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss_video.size(-1)))
                        # Average loss across channel dimension
                        loss_video = loss_video.mean(list(range(1, loss_video.ndim)))
                        # Average loss across batch dimension
                        loss_video = loss_video.mean()
                    else:
                        loss_video = 0.

                    if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                        target_action = noise_actions - actions
                        loss_action = action_weights.float() * (pred_all['action'].float() - target_action.float()).pow(2)    # shape b,l,c
                        loss_action = loss_action.mean()
                    else:
                        loss_action = 0.
                    action_loss_scale = getattr(self.args, "action_loss_scale", 1.0)

                    if joint_enabled:
                        loss = combine_joint_training_loss(
                            loss_video,
                            planner_metrics,
                            planner_loss_weight=float(
                                self.args.joint_training["planner_loss_weight"]
                            ),
                        )
                    else:
                        loss = loss_video + action_loss_scale * loss_action

                    assert torch.isnan(loss) == False, "NaN loss detected"
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and joint_enabled:
                        unwrapped_joint = accelerator.unwrap_model(self.joint_model)
                        joint_grad_metrics = {
                            "vlm_grad_norm": _gradient_norm(
                                unwrapped_joint.planner.parameters()
                            ),
                            "ltx_grad_norm": _gradient_norm(
                                unwrapped_joint.ltx.parameters()
                            ),
                        }
                        if accelerator.distributed_type != DistributedType.DEEPSPEED:
                            accelerator.clip_grad_norm_(
                                self.joint_model.parameters(),
                                self.args.max_grad_norm,
                            )
                    elif (
                        accelerator.sync_gradients
                        and accelerator.distributed_type != DistributedType.DEEPSPEED
                    ):
                        grad_norm = accelerator.clip_grad_norm_(
                            self.diffusion_model.parameters(),
                            self.args.max_grad_norm,
                        )
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                

                loss = accelerator.reduce(loss.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                    loss_action = accelerator.reduce(loss_action.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                    loss_video = accelerator.reduce(loss_video.detach(), reduction='mean')
                if joint_enabled:
                    planner_loss = accelerator.reduce(
                        planner_metrics["loss"].detach(),
                        reduction="mean",
                    )
                    semantic_metric = planner_metrics.get(
                        "semantic_mse",
                        planner_metrics.get("mse"),
                    )
                    depth_metric = planner_metrics.get(
                        "depth_wsa_loss",
                        planner_metrics.get("depth_smooth_l1"),
                    )
                    planner_semantic_mse = accelerator.reduce(
                        semantic_metric.detach(),
                        reduction="mean",
                    )
                    planner_depth_wsa_loss = accelerator.reduce(
                        depth_metric.detach(),
                        reduction="mean",
                    )

                running_loss += loss.item()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                if joint_enabled:
                    logs = {
                        "loss": loss.detach().item(),
                        "loss_video": loss_video.detach().item(),
                        "planner_loss": planner_loss.detach().item(),
                        "planner_semantic_mse": planner_semantic_mse.detach().item(),
                        "planner_depth_wsa_loss": planner_depth_wsa_loss.detach().item(),
                        "peak_memory_allocated": (
                            int(torch.cuda.max_memory_allocated(accelerator.device))
                            if torch.cuda.is_available()
                            else 0
                        ),
                    }
                    for name, value in joint_grad_metrics.items():
                        logs[name] = accelerator.reduce(
                            value.to(accelerator.device),
                            reduction="mean",
                        ).item()
                    logs.setdefault("vlm_grad_norm", 0.0)
                    logs.setdefault("ltx_grad_norm", 0.0)
                    lr_log_keys = {
                        "base_ltx": "lr/base_ltx",
                        "semantic_ltx": "lr/semantic_ltx",
                        "qwen": "lr/qwen",
                        "planner_heads": "lr/planner_heads",
                    }
                    for group in self.optimizer.param_groups:
                        group_name = group.get("name")
                        if group_name in lr_log_keys:
                            logs[lr_log_keys[group_name]] = float(group["lr"])
                    for log_key in lr_log_keys.values():
                        logs.setdefault(log_key, 0.0)
                else:
                    logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    logger.info(">>> max train step reached")
                    break

                if global_step % self.args.steps_to_log == 0:
                    if accelerator.is_main_process:
                        if self.writer is not None:
                            self.writer.add_scalar("Training Loss", loss.item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                                self.writer.add_scalar("Action loss", loss_action.mean().item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                                self.writer.add_scalar("Video loss", loss_video.item(), global_step)

                # global_step > 0 守卫: step 0 时 0 % N == 0 恒真会误触发 (grad_accum>1 时首步 global_step 停在 0)
                if accelerator.sync_gradients and global_step > 0 and global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(self.save_folder,f'Validation_step_{global_step}')
                        self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)

                
                if accelerator.sync_gradients and should_save_checkpoint(global_step, self.args):
                    if joint_enabled:
                        save_joint_checkpoint(
                            accelerator=accelerator,
                            joint_model=self.joint_model,
                            planner_provider=self.semantic_planner,
                            step_dir=Path(self.save_folder) / f"step_{global_step}",
                            args=self.args,
                            global_step=global_step,
                        )
                    else:
                        accelerator.wait_for_everyone()
                    if accelerator.is_main_process and not joint_enabled:
                        model_to_save = unwrap_model(accelerator, self.diffusion_model)
                        dtype = (
                            torch.float16
                            if self.args.mixed_precision == "fp16"
                            else torch.bfloat16
                            if self.args.mixed_precision == "bf16"
                            else torch.float32
                        )

                        model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
                        model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                        del  model_to_save
                        
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

        accelerator.wait_for_everyone()
        if joint_enabled:
            save_joint_checkpoint(
                accelerator=accelerator,
                joint_model=self.joint_model,
                planner_provider=self.semantic_planner,
                step_dir=Path(self.save_folder) / f"step_{global_step}",
                args=self.args,
                global_step=global_step,
            )
        elif accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else torch.bfloat16
                if self.args.mixed_precision == "bf16"
                else torch.float32
            )

            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

        del self.diffusion_model, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()


    def validate(
        self,
        accelerator,
        model_save_dir,
        global_step,
        n_view=1,
        n_chunk=30,
        image=None,
        prompt=None,
        cap=None,
        path=None,
        gt_actions=None,
        to_log=True,
        semantic_plan=None,
        semantic_plan_times=None,
        semantic_condition_mask=None,
        semantic_mode=None,
    ):

        os.makedirs(model_save_dir,exist_ok=True)

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer,
            unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model
        )

        batch = next(iter(self.val_dataloader))
        image = batch['video'][:,:,:,:self.args.data['train']['n_previous']].clone()  # shape b,c,v,t,h,w 
        prompt = batch['caption']
        gt_video = batch['video']
        b, c, v, t, h, w = image.shape
        negative_prompt = ''

        batch_size = 1

        image = image[:batch_size]

        if self.semantic_encoder is not None or self.semantic_planner is not None:
            semantic_mode = semantic_mode or self.args.semantic_plan.get("validation_mode", "gt")
            if semantic_mode == "gt":
                if self.semantic_encoder is None:
                    raise ValueError(
                        "GT semantic validation requires semantic_plan.source=gt_siglip2"
                    )
                mem_size = self.args.data['train']['n_previous']
                raw_future_frames = self.args.data['train']['chunk']
                semantic_future = rearrange(
                    gt_video[:batch_size, :, :, mem_size:mem_size + raw_future_frames],
                    'b c v t h w -> b v t c h w',
                ).to(accelerator.device, dtype=self.state.weight_dtype)
                semantic_plan = self.semantic_encoder.encode(
                    select_future_keyframes(
                        semantic_future,
                        indices=tuple(self.args.semantic_plan.get("keyframe_indices", (0, 3, 5, 8))),
                    )
                )
                latent_num_frames = compute_ltx_latent_frames(
                    raw_future_frames,
                    temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                    n_previous=mem_size,
                )
                semantic_plan_times = build_semantic_plan_times(
                    batch_size=batch_size,
                    n_view=v,
                    n_previous=mem_size,
                    num_future_frames=raw_future_frames,
                    num_latent_frames=latent_num_frames,
                    indices=tuple(self.args.semantic_plan.get("keyframe_indices", (0, 3, 5, 8))),
                    device=accelerator.device,
                )
                semantic_condition_mask = torch.ones(
                    batch_size * v,
                    device=accelerator.device,
                    dtype=self.state.weight_dtype,
                )
            elif semantic_mode == "planner":
                if self.semantic_planner is None:
                    raise ValueError(
                        "planner semantic validation requires semantic_plan.source=vlm_planner"
                    )
                semantic_plan, semantic_plan_times = build_vlm_semantic_condition(
                    self.semantic_planner,
                    gt_video[:batch_size],
                    prompt[:batch_size],
                    n_previous=self.args.data['train']['n_previous'],
                )
                semantic_plan = semantic_plan.to(
                    device=accelerator.device,
                    dtype=self.state.weight_dtype,
                )
                semantic_plan_times = semantic_plan_times.to(
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                semantic_condition_mask = torch.ones(
                    batch_size * v,
                    device=accelerator.device,
                    dtype=self.state.weight_dtype,
                )
            elif semantic_mode == "external":
                if semantic_plan is None or semantic_plan_times is None:
                    raise ValueError("external semantic validation requires plan tensors and times")
            elif semantic_mode == "none":
                semantic_plan = None
                semantic_plan_times = None
                semantic_condition_mask = None
            else:
                raise ValueError(f"unknown semantic validation mode: {semantic_mode}")

        image = rearrange(image, 'b c v t h w -> (b v) c t h w')
        num_denois_steps = self.args.num_inference_step

        if self.args.return_action and getattr(self.args, "add_state", False):
            history_action_state = batch['state'][:batch_size]
            if history_action_state.shape[1] > 1:
                history_action_state = history_action_state[:, self.args.data['train']['n_previous']-1:self.args.data['train']['n_previous'], :]
            history_action_state = history_action_state.contiguous()
        else:
            history_action_state = None

        preds = pipe.infer(
            image=image,
            prompt=prompt[:batch_size],
            negative_prompt=negative_prompt,
            num_inference_steps=num_denois_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=self.args.return_action,
            n_prev=self.args.data['train']['n_previous'],
            chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
            return_video=self.args.return_video,
            noise_seed=42,
            action_chunk=self.args.data['train']['action_chunk'],
            history_action_state = history_action_state,
            pixel_wise_timestep = self.args.pixel_wise_timestep,
            num_frames=self.args.data['train']['chunk'],  # Cosmos infer uses num_frames (future pixel frames = chunk); LTX ignores via kwargs
            postprocess_video=False,  # keep raw (b v) c t h w tensor so preds['video'] works; LTX ignores via kwargs
            n_chunk=n_chunk,
            action_dim=self.args.diffusion_model["config"]["action_in_channels"] if self.args.return_action else None,
            frame_rate=self.video_frame_rate,
            semantic_plan=semantic_plan,
            semantic_plan_times=semantic_plan_times,
            semantic_condition_mask=semantic_condition_mask,
        )[0]

        cap = 'Validation'
        fps = int(getattr(self.args, "basic_fps", 30) / (self.args.data['train']['action_chunk'] // self.args.data['train']['chunk']))
        save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{cap}_gt.mp4'), fps=fps)

        if self.args.return_video:
            video = preds['video'].data.cpu()
            save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], os.path.join(model_save_dir, f'{cap}.mp4'), fps=fps)

        if to_log:
            self.writer.add_text(f'step_{global_step}/{cap} prompt:', prompt[0], global_step)

        if self.args.return_action:
            # shape t, c
            gt_actions = batch['actions'][:, -self.args.data['train']['action_chunk']:]
            action_dim = gt_actions.shape[-1]

            action_logs = act_metric(
                preds['action'][:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                gt_actions[:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                prefix=cap,
                start_stop_interval=[(0,1),(1,9),(9,25),(25,self.args.data['train']['action_chunk'])]
            )

            if to_log:
                for key, value in action_logs.items():
                    self.writer.add_scalar(key, value, global_step)
