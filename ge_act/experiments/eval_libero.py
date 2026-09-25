import os, random, math
import gc
import hashlib
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from datetime import datetime, timedelta
import argparse
import json
import importlib
# ----------------------------------------------------


from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging
from PIL import Image
# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from utils.libero_sim_utils import get_libero_env, get_libero_image, get_libero_dummy_action, get_libero_state, save_rollout_video
from libero.libero import benchmark, get_libero_path
# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video
from runner.ge_trainer import (
    build_baton_semantic_condition,
    compute_ltx_latent_frames,
    prepare_baton_conditioning,
)
from runner.ge_inferencer import validate_baton_inference_source


class InferenceLibero:

    def __init__(self, config_file, output_dir=None, weight_dtype=torch.bfloat16, device="cuda:0", task_suite_name='libero_goal', model_path=None, exec_step=8, threshold=20, num_inference_steps=10, semantic_mode="config") -> None:

        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')

        # Tokenizers
        self.tokenizer = None

        # Text encoders
        self.text_encoder = None

        # Denoisers
        self.diffusion_model = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None
        self.baton_components = None
        if semantic_mode not in ("config", "disabled"):
            raise ValueError("semantic_mode must be 'config' or 'disabled'")
        ### "disabled" keeps the Baton provider in the loop but zeroes the semantic
        ### condition gate, i.e. the closed-loop counterpart of the training-time
        ### `semantic_disabled` validation mode. Always report both numbers.
        self.semantic_mode = semantic_mode
        self._latency = {"policy_ms": [], "baton_ms": [], "episode_s": []}

        self.args.output_dir = Path(self.args.output_dir)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_folder = os.path.join(self.args.output_dir, start_time)
        if getattr(self.args, "sub_folder", False):
            self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
        os.makedirs(self.save_folder, exist_ok=True)

        # import pdb; pdb.set_trace()
        args_dict = vars(deepcopy(self.args))
        for k, v in args_dict.items():
            args_dict[k] = str(v)

        self.weight_dtype = weight_dtype
        self.device = device
        self.task_suite_name = task_suite_name
        self.task_suite = self.prepare_simulator()

        with open(cd['data']['val']['stat_file'], "r") as f:
            self.StatisticInfo = json.load(f)

        ### pick the stat keys for this task suite. supports both the official single-domain
        ### file (keys "libero_eef"/"libero_state_eef") and the fastwam mixed file
        ### (keys "libero_<suite>_no_noops_lerobot_eef"). falls back to "libero".
        _dom = task_suite_name + "_no_noops_lerobot"
        if _dom + "_eef" not in self.StatisticInfo:
            _dom = "libero" if "libero_eef" in self.StatisticInfo else _dom
        self.act_mean = torch.tensor(self.StatisticInfo[_dom+"_eef"]["mean"]).unsqueeze(0)
        self.act_std = torch.tensor(self.StatisticInfo[_dom+"_eef"]["std"]).unsqueeze(0)
        self.states_mean = torch.tensor(self.StatisticInfo[_dom+"_state_eef"]['mean']).unsqueeze(0)
        self.states_std = torch.tensor(self.StatisticInfo[_dom+"_state_eef"]['std']).unsqueeze(0)
        self.act_min = torch.tensor(self.StatisticInfo[_dom+"_eef"]["q01"]).unsqueeze(0)
        self.act_max = torch.tensor(self.StatisticInfo[_dom+"_eef"]["q99"]).unsqueeze(0)
        self.states_min = torch.tensor(self.StatisticInfo[_dom+"_state_eef"]['q01']).unsqueeze(0)
        self.states_max = torch.tensor(self.StatisticInfo[_dom+"_state_eef"]['q99']).unsqueeze(0)


        self.action_chunk = cd["data"]["val"]["action_chunk"]
        self.chunk = cd["data"]["val"]["chunk"]
        self.action_dim = cd["diffusion_model"]["config"].get("action_out_channels", cd["diffusion_model"]["config"]["action_in_channels"])
        self.basic_action_dim = 7

        self.dtype = torch.bfloat16
        self.add_state = cd["add_state"]
        self.n_prev = cd["data"]['train']['n_previous']
        self.num_inference_steps = num_inference_steps

        self.sample_n_frames = cd["data"]["train"]["sample_n_frames"]

        self.excution_step = exec_step
        self.threshold = threshold

        if model_path is not None:
            cd["diffusion_model"]['model_path'] = model_path
        else:
            model_path = cd["diffusion_model"]['model_path']
        model_steps = model_path.split('/')[-2]
        self.with_state = cd["add_state"]
        _sem_tag = "" if self.semantic_mode == "config" else f"_sem{self.semantic_mode}"
        log_file_path = os.path.join(self.save_folder, f"inference_{task_suite_name}_{model_steps}_wstate{self.with_state}_execstep{self.excution_step}_thresh{self.threshold}{_sem_tag}.txt")
        self.log_file = open(log_file_path, "w")

        args_dict["model_path"] = model_path
        args_dict["task_suite_name"] = task_suite_name
        args_dict["with_state"] = self.with_state
        args_dict["excution_step"] = self.excution_step
        args_dict["threshold"] = self.threshold
        args_dict["semantic_mode"] = self.semantic_mode
        ### Pin everything that can silently change a LIBERO number: simulator assets,
        ### checkpoint bytes, code revision and the runtime. Best-effort, never fatal.
        args_dict["provenance"] = self._collect_provenance(model_path, task_suite_name)

        with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
            json.dump(args_dict, file, indent=4, sort_keys=False)
        with open(os.path.join(self.save_folder, 'provenance.json'), "w") as file:
            json.dump(args_dict["provenance"], file, indent=4, sort_keys=True)
        self.log_file.write(f"config: {args_dict}\n")
        self.log_file.flush()

    ### --- provenance -----------------------------------------------------------
    @staticmethod
    def _sha256_tree(root, limit_bytes=None):
        """SHA-256 over sorted relative paths and file contents under *root*."""
        root = Path(root)
        digest = hashlib.sha256()
        files = sorted(p for p in root.rglob("*") if p.is_file())
        total = 0
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
                    total += len(chunk)
                    if limit_bytes is not None and total > limit_bytes:
                        return {"sha256": None, "files": len(files), "note": "skipped: too large"}
        return {"sha256": digest.hexdigest(), "files": len(files)}

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _git_revision(path):
        try:
            out = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            rev = out.stdout.strip()
            if not rev:
                return None
            dirty = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                capture_output=True, text=True, timeout=30, check=False,
            ).stdout.strip()
            return {"commit": rev, "dirty": bool(dirty)}
        except Exception:
            return None

    def _collect_provenance(self, model_path, task_suite_name):
        info = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
            "numpy": np.__version__,
        }
        for module in ("robosuite", "mujoco", "libero"):
            try:
                info[module] = getattr(importlib.import_module(module), "__version__", "unknown")
            except Exception as error:
                info[module] = f"unavailable: {error}"
        try:
            benchmark_root = Path(get_libero_path("benchmark_root"))
            info["libero_benchmark_root"] = str(benchmark_root)
            info["libero_git"] = self._git_revision(benchmark_root)
            for key in ("init_states", "bddl_files"):
                suite_dir = Path(get_libero_path(key)) / task_suite_name
                info[f"libero_{key}_{task_suite_name}"] = (
                    self._sha256_tree(suite_dir) if suite_dir.is_dir() else None
                )
        except Exception as error:
            info["libero_assets_error"] = str(error)
        try:
            ckpt = Path(model_path)
            info["checkpoint"] = {
                "path": str(ckpt),
                "bytes": ckpt.stat().st_size if ckpt.exists() else None,
                "sha256": (
                    self._sha256_file(ckpt) if ckpt.is_file()
                    else self._sha256_tree(ckpt).get("sha256") if ckpt.is_dir() else None
                ),
            }
        except Exception as error:
            info["checkpoint"] = {"path": str(model_path), "error": str(error)}
        info["code_git"] = self._git_revision(Path(__file__).resolve().parent)
        return info

    ### --- prompt cache ---------------------------------------------------------
    ### The task description is fixed for a whole episode, but play() runs on every
    ### control step, so without a cache T5-XXL is re-encoded dozens to hundreds of
    ### times per episode. Encode each prompt once; once every prompt a run will
    ### need is cached, free_text_encoder() drops T5 (~9.4 GB bf16).
    @torch.no_grad()
    def _cached_prompt(self, prompt, negative_prompt):
        if not hasattr(self, "_prompt_cache"):
            self._prompt_cache = {}
        key = (prompt, negative_prompt)
        if key not in self._prompt_cache:
            if getattr(self, "_text_encoder_freed", False):
                raise RuntimeError(
                    f"prompt not pre-encoded and the text encoder was freed: {prompt!r}. "
                    "Call prefill_prompts() with every task description first."
                )
            out = self.pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                device=self.device,
            )
            ### Park the cache on the host: each entry is ~1 MB, shipping it to the
            ### GPU per call is free next to the rollout.
            self._prompt_cache[key] = tuple(
                t.detach().to("cpu") if torch.is_tensor(t) else t for t in out
            )
        return tuple(
            t.to(self.device) if torch.is_tensor(t) else t
            for t in self._prompt_cache[key]
        )

    def prefill_prompts(self, prompts, negative_prompt="", free_encoder=True):
        """Encode every prompt this process will need, then optionally release T5."""
        for pr in dict.fromkeys(prompts):
            self._cached_prompt(pr, negative_prompt)
        if free_encoder:
            self.free_text_encoder()

    def free_text_encoder(self):
        """Drop T5 from GPU. Only safe once every prompt is in the cache."""
        if getattr(self, "_text_encoder_freed", False):
            return
        self.pipeline.text_encoder = None
        self.text_encoder = None
        self._text_encoder_freed = True
        gc.collect()
        torch.cuda.empty_cache()

    def prepare_models(self,):

        print("Initializing models")
        device = self.device
        dtype = self.weight_dtype
        semantic = getattr(self.args, "semantic_plan", {})
        baton_source = validate_baton_inference_source(vars(self.args))
        if baton_source is not None:
            self.baton_components = prepare_baton_conditioning(
                vars(self.args),
                dataset=None,
                device=device,
                dtype=dtype,
                for_training=False,
            )

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
            load_weights=self.args.load_weights
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
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
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


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
        print(f'Total parameters for transformer model:{total_params}')


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

        self.pipeline = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )

    def prepare_simulator(self, ):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        return task_suite

    @torch.no_grad()
    def play(self, obs, prompt, excution_step=1, state=None):
        """
        obs: tensor of shape v*3*h*w ranging -1 to 1, v c h w
        cond_prompt: task description
        excution_step: excution step of the past play
        return the raw action
        """
        assert excution_step >= 1 and excution_step <= 100, "excution_step should be in [1, 100]"
        if obs.dtype == np.uint8:
            ### obs / 255 * 2 - 1
            obs = obs.astype(np.float32) / 255.0 * 2.0 - 1
            obs = np.transpose(obs, (0,3,1,2))

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)

        v, c, h, w = obs.shape

        # assert v == 2 and c ==3 and h == 192 and w == 256, "obs should be of shape 2*3*192*256"

        if obs.max().cpu() > 1.0:
            # normalize to [-1, 1]
            obs = obs / 255.0 * 2.0 - 1.0

        obs = obs.to(self.device, dtype=self.dtype)
        history_action_state = state
        if self.add_state  and history_action_state is not None:
            if isinstance(history_action_state, np.ndarray):
                history_action_state = torch.from_numpy(history_action_state).to(self.device, dtype=self.dtype)
            while len(history_action_state.shape) < 3:
                history_action_state = history_action_state.unsqueeze(dim=0)

            assert(len(history_action_state.shape) == 3 and history_action_state.shape[-1]==self.action_dim), f"history_action_state shape: {history_action_state.shape}, action_dim: {self.action_dim}, history_action_state.shape[-1] {history_action_state.shape[-1]}"
        else:
            history_action_state = None

        if not self.obs:
            self.obs = [obs] * self.n_prev
            self.count = self.threshold - 1
            self.buffer = [self.obs[-1]]
        else:
            if excution_step == 0:
                return self.action_buffer
            self.count += excution_step
            if self.count >= self.threshold:
                self.count = 0
                self.obs.pop(0)
                self.obs[-1] = self.buffer[0]
                self.obs.append(obs)
            else:
                self.obs[-1] = obs
            self.buffer = [self.obs[-1]]
        obs_tensor = torch.stack(self.obs, dim=1)  # from t * (v, c, h, w) to (v, t, c, h, w)

        obs_tensor = rearrange(obs_tensor, "v t c h w -> c v t h w")  # c,t,h,w
        obs_tensor = obs_tensor.unsqueeze(0)  # b,c,v,t,h,w
        semantic_plan = None
        semantic_plan_times = None
        semantic_plan_positions = None
        semantic_plan_mask = None
        semantic_plan_relevance = None
        semantic_condition_mask = None
        if self.baton_components is not None:
            _baton_t0 = time.perf_counter()
            latent_num_frames = compute_ltx_latent_frames(
                self.chunk,
                temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                n_previous=self.n_prev,
            )
            condition = build_baton_semantic_condition(
                self.baton_components,
                self.args.semantic_plan,
                obs_tensor,
                (str(prompt),),
                n_previous=self.n_prev,
                num_future_frames=self.chunk,
                num_latent_frames=latent_num_frames,
                device=self.device,
                dtype=self.dtype,
            )
            semantic_plan = condition.tokens
            semantic_plan_times = condition.times
            semantic_plan_positions = condition.positions
            semantic_plan_mask = condition.mask
            semantic_plan_relevance = condition.relevance
            _gate = 0.0 if self.semantic_mode == "disabled" else 1.0
            semantic_condition_mask = torch.full(
                (v,),
                _gate,
                device=self.device,
                dtype=self.dtype,
            )
            self._latency["baton_ms"].append((time.perf_counter() - _baton_t0) * 1000.0)
        obs_tensor = rearrange(obs_tensor, "b c v t h w -> (b v) c t h w")

        negative_prompt = ""
        pe, pam, npe, npam = self._cached_prompt(prompt, negative_prompt)
        _policy_t0 = time.perf_counter()
        pred_all = self.pipeline.infer(
            image=obs_tensor,
            prompt=None,
            negative_prompt=None,
            prompt_embeds=pe,
            prompt_attention_mask=pam,
            negative_prompt_embeds=npe,
            negative_prompt_attention_mask=npam,
            num_inference_steps=self.num_inference_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=True,
            return_video=False,
            chunk=(self.chunk-1)//self.TEMPORAL_DOWN_RATIO+1,
            action_chunk=self.action_chunk,
            history_action_state=history_action_state if self.add_state else None,
            noise_seed=42,
            pixel_wise_timestep = self.args.pixel_wise_timestep,
            n_chunk=1,
            n_prev=self.n_prev,
            action_dim=self.action_dim,
            semantic_plan=semantic_plan,
            semantic_plan_times=semantic_plan_times,
            semantic_plan_positions=semantic_plan_positions,
            semantic_plan_mask=semantic_plan_mask,
            semantic_plan_relevance=semantic_plan_relevance,
            semantic_condition_mask=semantic_condition_mask,
        )[0]
        self._latency["policy_ms"].append((time.perf_counter() - _policy_t0) * 1000.0)

        actions_pred = pred_all["action"].detach().cpu()[0]

        actions_pred = self.denormalize_actions(actions_pred[:, :self.basic_action_dim])

        self.action_buffer = actions_pred.clone()

        return actions_pred

    ### --- normalization: fastwam (lerobot_like_dataset) convention. ---
    ### The official-weight equivalents live in experiments/eval_libero_official.py; the two
    ### are deliberately kept apart because the gripper conventions are mutually destructive
    ### (applying one model's rule to the other's output silently yields garbage actions).
    def denormalize_actions(self, actions_pred):
        """Model output -> LIBERO env action space (fastwam-trained checkpoints)."""
        d = self.basic_action_dim
        actions_pred = actions_pred.float() * self.act_std[:, :d] + self.act_mean[:, :d]
        ### gripper: fastwam data-space [0,1] (0=close,1=open) -> env [-1,1] (-1=open,+1=close)
        ### equals official (g*2-1) then invert_gripper_action (*-1): g_env = 1 - 2*g
        actions_pred[:, -1] = 1.0 - 2.0 * actions_pred[:, -1]
        return actions_pred

    def normalize_state(self, state):
        """Raw env state -> model input space (fastwam-trained checkpoints)."""
        state = state.float() if torch.is_tensor(state) else torch.tensor(state).float()
        return (state - self.states_mean) / (self.states_std + 1e-6)

    def policy_memory_reset(self):
        self.obs = []
        self.buffer = []
        self.action_buffer = torch.zeros(self.action_chunk, self.action_dim, dtype=torch.bfloat16)
        self.cur_step = 0

    def infer(self, num_trails_per_task, image_shape=(256,256)):
        total_episodes, total_successes = 0, 0
        # SMOKE_MAX_TASKS caps the number of tasks for a quick smoke run (default: all tasks)
        _n_tasks = self.task_suite.n_tasks
        _smoke_max = os.environ.get("SMOKE_MAX_TASKS", "")
        if _smoke_max:
            _n_tasks = min(_n_tasks, int(_smoke_max))
        ### Encode every task description once and free T5 before the rollouts.
        ### KEEP_TEXT_ENCODER=1 keeps T5 resident (debugging only).
        self.prefill_prompts(
            [self.task_suite.get_task(i).language for i in range(_n_tasks)],
            free_encoder=os.environ.get("KEEP_TEXT_ENCODER", "") != "1",
        )
        for task_id in range( _n_tasks):
            task = self.task_suite.get_task(task_id)
            initial_states = self.task_suite.get_task_init_states(task_id)

            env, task_description = get_libero_env(
                task, image_height=image_shape[0], image_width=image_shape[1]
            )

            task_episodes, task_successes = 0, 0
            for episode_idx in range(num_trails_per_task):
                if env.env.done:
                    print(f"env.env.done, resetting...")
                    task = self.task_suite.get_task(task_id)
                    initial_states = self.task_suite.get_task_init_states(task_id)
                    env, task_description = get_libero_env(task,
                                                           image_height=image_shape[0],
                                                           image_width=image_shape[1])
                    obs = env.set_init_state(initial_states[episode_idx])

                env.reset()
                self.policy_memory_reset()


                obs = env.set_init_state(initial_states[episode_idx])
                _episode_t0 = time.perf_counter()
                t = 0
                replay_images = []
                if self.task_suite_name == "libero_spatial":
                    max_steps = 220  # longest training demo has 193 steps
                elif self.task_suite_name == "libero_object":
                    max_steps = 280  # longest training demo has 254 steps
                elif self.task_suite_name == "libero_goal":
                    max_steps = 300  # longest training demo has 270 steps
                elif self.task_suite_name == "libero_10":
                    max_steps = 520  # longest training demo has 505 steps
                elif self.task_suite_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps

                num_steps_wait = 10
                while t < max_steps + num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        print("Waiting for objects to fall...t < cfg.num_steps_wait")
                        self.log_file.write("Waiting for objects to fall...t < cfg.num_steps_wait\n")
                        continue

                    agtview_img, wrist_img = get_libero_image(obs)
                    replay_images.append(agtview_img)

                    agtview_img_tensor = torch.tensor(agtview_img.copy()).to(self.device).permute(2, 0, 1).unsqueeze(0)
                    wrist_img_tensor = torch.tensor(wrist_img.copy()).to(self.device).permute(2, 0, 1).unsqueeze(0)

                    img_obs = torch.cat([agtview_img_tensor, wrist_img_tensor], dim=0)
                    # get obs: 2 * 3 * h * w
                    # print(f"obs shape: {obs.shape}")

                    if self.with_state:
                        state = get_libero_state(obs)

                        ### mean/std normalization to match lerobot_like_dataset (fastwam) training
                        state = self.normalize_state(state)
                        state = torch.cat((torch.zeros([1,self.basic_action_dim]), state), dim=1)

                        actions = self.play(img_obs, task_description, excution_step=self.excution_step, state=state) # action_chunk * action_dim
                    else:
                        actions = self.play(img_obs, task_description, excution_step=self.excution_step, state=None) # action_chunk * action_dim
                    actions = actions.cpu().numpy()


                    if t  >= max_steps:
                        break

                    for i in range(self.excution_step):
                        cur_action = actions[i, :]
                        try:
                            obs, reward, done, info = env.step(cur_action.tolist())
                        except Exception as e:
                            print(f"Error in step {t}: {e}")
                            self.log_file.write(f"Error in step {t}: {e}\n")

                            save_rollout_video(
                                rollout_dir=self.save_folder, rollout_images=replay_images, idx=total_episodes, success=done, task_description=f"error_{e}"+task_description
                            )
                            break

                        if not done and env.env.done:
                            print(f">>>>>Task {task_id} Episode {episode_idx} is not done but env signals done.")
                            save_rollout_video(
                                rollout_dir=self.save_folder, rollout_images=replay_images, idx=total_episodes, success=done, task_description="error_{e}"+task_description
                            )

                        if done: # or env.env.done:
                            task_successes += 1
                            total_successes += 1
                            break

                    if done or env.env.done:
                        break

                    t += 1

                # Always save a rollout mp4 for the episode (success or fail)
                try:
                    save_rollout_video(
                        rollout_dir=self.save_folder, rollout_images=replay_images,
                        idx=total_episodes, success=bool(done), task_description=task_description
                    )
                except Exception as _e:
                    print(f"[warn] could not save rollout video: {_e}")

                task_episodes += 1
                total_episodes += 1
                self._latency["episode_s"].append(time.perf_counter() - _episode_t0)

                # Log current results
                print(f"Success: {done}")
                self.log_file.write(f"Success: {done}\n")
                print(f"# episodes completed so far: {total_episodes}")
                self.log_file.write(f"# episodes completed so far: {total_episodes}\n")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                self.log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                self.log_file.flush()

            # Log final results
            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            self.log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
            self.log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
            self.log_file.flush()
        self._write_latency_summary()

    def _write_latency_summary(self):
        """Per-call policy/Baton latency and per-episode wall time, for deployability."""
        summary = {}
        for name, values in self._latency.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            summary[name] = {
                "count": int(arr.size),
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(arr.max()),
            }
        summary["exec_step"] = self.excution_step
        summary["num_inference_steps"] = self.num_inference_steps
        summary["semantic_mode"] = self.semantic_mode
        with open(os.path.join(self.save_folder, "latency.json"), "w") as file:
            json.dump(summary, file, indent=4, sort_keys=True)
        line = "latency: " + ", ".join(
            f"{k} mean={v['mean']:.1f} p95={v['p95']:.1f}"
            for k, v in summary.items() if isinstance(v, dict)
        )
        print(line)
        self.log_file.write(line + "\n")
        self.log_file.flush()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="path to config file")
    parser.add_argument("--ckpt_path", type=str, help="path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default=None, help="path to output directory")
    parser.add_argument("--task_suite_name", type=str, default="libero_goal", help="task suite name")
    parser.add_argument("--exec_step", type=int, default=8, help="excution step")
    parser.add_argument("--num_trails_per_task", type=int, default=50, help="number of inference steps")
    parser.add_argument("--device", type=int, default=0, help="cuda id")
    parser.add_argument("--threshold", type=int, default=20, help="threshold")
    parser.add_argument("--semantic_mode", type=str, default="config", choices=["config", "disabled"],
                        help="'disabled' zeroes the Baton semantic gate (closed-loop semantic_disabled ablation)")

    args = parser.parse_args()

    config_file = args.config_file
    output_dir = os.path.join(args.output_dir, args.task_suite_name)


    libero_infer = InferenceLibero(
        config_file=config_file, output_dir=output_dir, task_suite_name=args.task_suite_name, model_path=args.ckpt_path, exec_step=args.exec_step, device=f"cuda:{args.device}",
        threshold=args.threshold, semantic_mode=args.semantic_mode,
    )
    libero_infer.prepare_models()
    libero_infer.infer(
        num_trails_per_task=args.num_trails_per_task, image_shape=libero_infer.args.data["train"]["sample_size"]
    )
