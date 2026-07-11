"""FastWAM with a Cosmos-Predict2.5 video backbone, coupled to an action expert
via the original-FastWAM MoT (joint masked attention).

Both streams (video DiT + action DiT) are run layer-by-layer through
``mot_block_forward`` (cosmos/mot.py), which does ONE joint self-attention over
the concatenated K/V of both streams, then per-stream cross-attn(text)+MLP. Each
stream is then read out (video -> velocity latent, action -> velocity actions)
and trained with flow matching (velocity target = noise - sample), matching the
Cosmos rectified-flow objective and the original FastWAM loss.

Requires video_expert.net.blocks and action_expert.blocks to have equal length.
UNTESTED until a GPU frees on HPC3 (QOS-blocked at time of writing).
"""
from __future__ import annotations

import os
from numbers import Integral
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger
from ..wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .couplings import build_coupling

logger = get_logger(__name__)


class FastWAMCosmos(nn.Module):
    def __setattr__(self, name, value):
        if name == "_online_semantic_planner":
            # This dependency is frozen runtime infrastructure even when a test or
            # future implementation happens to subclass nn.Module. Keep replacement
            # just as safe as constructor assignment.
            for registry_name in ("_modules", "_parameters", "_buffers"):
                registry = self.__dict__.get(registry_name)
                if registry is not None:
                    registry.pop(name, None)
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)

    def __init__(
        self,
        video_expert,
        action_expert,
        vae,
        vae_encode_fn: Optional[Callable] = None,
        vae_name: str = "CosmosWan2pt1",
        crossattn_dim: int = 1024,
        qwen_dim: int = 3584,
        coupling: str = "mot",            # "mot" (joint attention) | "cross_attn"
        mot_bidirectional: bool = False,  # MoT: True = video also attends action (default False = original FastWAM: video independent)
        feature_layer: int = -1,          # which video block's hidden state for cross_attn
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_train_shift: float = 3.0,
        action_train_shift: float = 3.0,
        num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        semantic_plan_dim: int = 0,
        semantic_plan_max_tokens: int = 0,
        semantic_plan_num_keyframes: int = 0,
        online_semantic_planner=None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.vae = vae
        self._vae_encode_fn = vae_encode_fn
        self.vae_name = vae_name
        self.crossattn_dim = int(crossattn_dim)
        self.device = device
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        for name, value in (
            ("semantic_plan_dim", semantic_plan_dim),
            ("semantic_plan_max_tokens", semantic_plan_max_tokens),
            ("semantic_plan_num_keyframes", semantic_plan_num_keyframes),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
            if value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
        self.semantic_plan_dim = int(semantic_plan_dim)
        self.semantic_plan_max_tokens = int(semantic_plan_max_tokens)
        self.semantic_plan_num_keyframes = int(semantic_plan_num_keyframes)
        # Deliberately bypass nn.Module.__setattr__: even a future provider that
        # subclasses nn.Module is frozen external infrastructure, not FastWAM state.
        object.__setattr__(
            self,
            "_online_semantic_planner",
            online_semantic_planner,
        )

        # MoT joint attention requires equal block counts (concat K/V layer-by-layer).
        # Other couplings (cross_attn, agra) run the action DiT independently of the
        # video block loop, so their action head may have a different depth (the AGRA
        # ForesightActionHead is 8 layers vs the video DiT's 28) — skip the check.
        if str(coupling) == "mot":
            nv, na = len(video_expert.net.blocks), len(action_expert.blocks)
            if nv != na:
                raise ValueError(f"MoT needs equal block counts: video={nv} action={na}")

        # Qwen2.5-VL text embeds are 3584-dim; the MiniTrainDIT crossattn wants 1024.
        # A learned projection (trained) avoids needing Cosmos' exact text projection.
        self.text_proj = (
            nn.Linear(int(qwen_dim), self.crossattn_dim).to(device=device, dtype=torch_dtype)
            if int(qwen_dim) != self.crossattn_dim else nn.Identity()
        )

        # Pluggable coupling (registry in cosmos/couplings/): the coupling owns the
        # video<->action interaction forward and any coupling-specific submodules.
        # Built-ins: "mot" (joint masked attention) | "cross_attn" (action cross-
        # attends a video hidden layer). New variants drop a file in couplings/.
        self.coupling = str(coupling)
        self.mot_bidirectional = bool(mot_bidirectional)
        self.feature_layer = int(feature_layer)
        self.video_feat_proj = None  # may be created by the coupling's setup()
        self._coupling = build_coupling(self.coupling)
        self._coupling.setup(self)

        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        # mot/cross_attn: proprio is a token appended to the text cross-attn context,
        # so the model owns a proprio_encoder. AGRA instead prepends proprio inside the
        # ForesightActionHead (which owns its own proprio encoder), so don't build one
        # here (avoids an unused Linear in the optimizer/checkpoint).
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, self.crossattn_dim).to(torch_dtype)
            if (self.proprio_dim is not None and self.coupling != "agra") else None
        )

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=video_train_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps, shift=action_train_shift
        )
        self._current_semantic_plan = None
        self._current_semantic_plan_times = None
        self._current_video_fps = None

    # The trainer optimizes exactly these (video DiT + action DiT [+ proprio]).
    @property
    def dit(self):
        return nn.ModuleDict({"video": self.video_expert, "action": self.action_expert})

    # Alias used by the AGRA coupling. For "agra" the action_expert IS a
    # ForesightActionHead (built by runtime.create_fastwam_cosmos); the coupling
    # calls model.action_head(noisy_action, t_a, proprio0, contexts).
    @property
    def action_head(self):
        return self.action_expert

    def _vae_encode(self, video):
        if self._vae_encode_fn is None:
            raise RuntimeError("`vae_encode_fn` was not provided to FastWAMCosmos.")
        return self._vae_encode_fn(self.vae_name, self.vae, video, device=self.device)

    def _reset_semantic_condition(self):
        self._current_semantic_plan = None
        self._current_semantic_plan_times = None
        self._current_video_fps = None

    def _require_semantic_geometry(self):
        for name, value in (
            ("semantic_plan_dim", self.semantic_plan_dim),
            ("semantic_plan_max_tokens", self.semantic_plan_max_tokens),
            ("semantic_plan_num_keyframes", self.semantic_plan_num_keyframes),
        ):
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive for semantic conditioning, got {value}"
                )

    def _validate_semantic_tensors(self, plan, times, batch_size):
        self._require_semantic_geometry()
        expected_plan = (
            batch_size,
            self.semantic_plan_max_tokens,
            self.semantic_plan_dim,
        )
        expected_times = (batch_size, self.semantic_plan_num_keyframes)
        if tuple(plan.shape) != expected_plan:
            raise ValueError(
                f"semantic_plan must have shape {expected_plan}, "
                f"got {tuple(plan.shape)}"
            )
        if tuple(times.shape) != expected_times:
            raise ValueError(
                f"semantic_plan_times must have shape {expected_times}, "
                f"got {tuple(times.shape)}"
            )
        if not torch.isfinite(plan).all():
            raise ValueError("semantic_plan contains non-finite values")
        if not torch.isfinite(times).all():
            raise ValueError("semantic_plan_times contains non-finite values")
        if (times < 0).any() or (times > 1).any():
            raise ValueError("semantic_plan_times must lie in [0, 1]")
        if not torch.all(times[:, 1:] > times[:, :-1]):
            raise ValueError("semantic_plan_times must be strictly increasing")

    def _validate_provider_branch(self, name, value, batch_size):
        if not torch.is_tensor(value):
            raise TypeError(f"online {name} must be a torch.Tensor")
        expected = (
            batch_size,
            self.semantic_plan_max_tokens,
            self.semantic_plan_dim,
        )
        if tuple(value.shape) != expected:
            raise ValueError(
                f"online {name} must have shape {expected}, got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"online {name} contains non-finite values")

    def _set_current_semantic_plan(
        self,
        sample,
        *,
        batch_size=None,
        device=None,
    ):
        """Validate and install one complete file-backed semantic condition."""
        offline_plan = sample.get("semantic_plan")
        offline_times = sample.get("semantic_plan_times")
        has_plan = offline_plan is not None
        has_times = offline_times is not None
        if has_plan != has_times:
            raise ValueError(
                "semantic_plan and semantic_plan_times must be provided together"
            )
        if not has_plan:
            self._current_semantic_plan = None
            self._current_semantic_plan_times = None
            return
        try:
            plan = torch.as_tensor(offline_plan).to(
                device=self.device if device is None else device,
                dtype=torch.float32,
                non_blocking=True,
            )
            times = torch.as_tensor(offline_times).to(
                device=self.device if device is None else device,
                dtype=torch.float32,
                non_blocking=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "semantic_plan and semantic_plan_times must be numeric tensors"
            ) from error
        if batch_size is None:
            batch_size = int(plan.shape[0]) if plan.ndim > 0 else 0
        self._validate_semantic_tensors(plan, times, int(batch_size))
        self._current_semantic_plan = plan
        self._current_semantic_plan_times = times

    @staticmethod
    def _validate_instructions(instructions, batch_size):
        if isinstance(instructions, str):
            instructions = [instructions]
        elif isinstance(instructions, (list, tuple)):
            instructions = list(instructions)
        else:
            raise ValueError(
                "instruction must be a string or a batch of strings for online "
                "semantic conditioning"
            )
        if len(instructions) != batch_size:
            raise ValueError(
                f"instruction batch must contain {batch_size} strings, "
                f"got {len(instructions)}"
            )
        if any(
            not isinstance(instruction, str) or not instruction.strip()
            for instruction in instructions
        ):
            raise ValueError("instruction must contain non-empty strings")
        return instructions

    def _prepare_semantic_condition(self, sample, current_rgb):
        """Prepare either online or file-backed conditioning exactly once."""
        self._reset_semantic_condition()
        if current_rgb.ndim != 4 or current_rgb.shape[1] != 3:
            raise ValueError(
                "current_rgb must have shape [B, 3, H, W], got "
                f"{tuple(current_rgb.shape)}"
            )
        batch_size = int(current_rgb.shape[0])
        offline_plan = sample.get("semantic_plan")
        offline_times = sample.get("semantic_plan_times")
        has_offline_plan = offline_plan is not None
        has_offline_times = offline_times is not None
        has_offline = has_offline_plan or has_offline_times
        has_online = self._online_semantic_planner is not None

        if has_online and has_offline:
            raise ValueError(
                "online and file-backed semantic plans are mutually exclusive"
            )
        if has_offline_plan != has_offline_times:
            raise ValueError(
                "semantic_plan and semantic_plan_times must be provided together"
            )

        raw_fps = sample.get("video_fps")
        if raw_fps is None:
            if has_online or has_offline:
                raise KeyError(
                    "video_fps is required for semantic conditioning"
                )
            video_fps = None
        else:
            try:
                video_fps = torch.as_tensor(
                    raw_fps,
                    device=current_rgb.device,
                )
            except (TypeError, ValueError) as error:
                raise ValueError("video_fps must be numeric") from error
            if video_fps.dtype == torch.bool:
                raise ValueError("video_fps must be numeric, not boolean")
            video_fps = video_fps.to(dtype=torch.float32)
            if video_fps.ndim == 0:
                video_fps = video_fps.expand(batch_size)
            if tuple(video_fps.shape) != (batch_size,):
                raise ValueError(
                    f"video_fps must have shape [{batch_size}], "
                    f"got {tuple(video_fps.shape)}"
                )
            if not torch.isfinite(video_fps).all() or (video_fps <= 0).any():
                raise ValueError(
                    "video_fps must contain finite positive values"
                )
            if video_fps.numel() > 1 and not torch.equal(
                video_fps,
                video_fps[:1].expand_as(video_fps),
            ):
                raise ValueError(
                    "video_fps must be uniform across the batch for Cosmos"
                )

        if has_online:
            self._require_semantic_geometry()
            if self.semantic_plan_num_keyframes != 4:
                raise ValueError(
                    "online semantic conditioning requires exactly 4 keyframes"
                )
            if "instruction" not in sample or sample.get("instruction") is None:
                raise KeyError(
                    "instruction is required for online semantic conditioning"
                )
            instructions = self._validate_instructions(
                sample["instruction"], batch_size
            )
            with torch.no_grad():
                result = self._online_semantic_planner.predict(
                    current_rgb,
                    instructions,
                )
            try:
                dino_plan = result.dino_plan
                depth_plan = result.depth_plan
                semantic_plan_times = result.semantic_plan_times
            except AttributeError as error:
                raise TypeError(
                    "online planner result must expose dino_plan, depth_plan, and "
                    "semantic_plan_times"
                ) from error
            self._validate_provider_branch("dino_plan", dino_plan, batch_size)
            self._validate_provider_branch("depth_plan", depth_plan, batch_size)
            if not torch.is_tensor(semantic_plan_times):
                raise TypeError(
                    "online semantic_plan_times must be a torch.Tensor"
                )
            fused_plan = self.video_expert.fuse_semantic_plan(
                dino_plan.detach(),
                depth_plan.detach(),
            )
            semantic_plan_times = semantic_plan_times.detach().to(
                device=fused_plan.device,
                dtype=torch.float32,
            )
            self._validate_semantic_tensors(
                fused_plan,
                semantic_plan_times,
                batch_size,
            )
            expected_times = semantic_plan_times.new_tensor(
                [0.25, 0.5, 0.75, 1.0]
            ).unsqueeze(0).expand(batch_size, -1)
            if not torch.allclose(
                semantic_plan_times,
                expected_times,
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    "online semantic_plan_times must equal "
                    "[0.25, 0.5, 0.75, 1.0]"
                )
            self._current_semantic_plan = fused_plan
            self._current_semantic_plan_times = semantic_plan_times
            self._current_video_fps = video_fps
            return

        if has_offline:
            self._set_current_semantic_plan(
                sample,
                batch_size=batch_size,
                device=current_rgb.device,
            )
        self._current_video_fps = video_fps

    # --------------------------------------------------------- coupling dispatch
    def couple_forward(self, noisy_latents, t_v, noisy_action, t_a, crossattn_emb):
        """Dispatch the video<->action interaction to the registered coupling.
        Returns (pred_v_latent [B,C,T,H,W], pred_a [B,Ta,action_dim])."""
        return self._coupling.forward(self, noisy_latents, t_v, noisy_action, t_a, crossattn_emb)

    # ------------------------------------------------------------- build inputs
    def build_inputs(self, sample, tiled: bool = False, video=None):
        if video is None:
            video = sample["video"].to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)
        proprio = sample.get("proprio", None)

        input_latents = self._vae_encode(video).to(self.torch_dtype)

        proprio_first = None
        # AGRA feeds proprio into the action head (its own encoder) rather than the
        # model's proprio_encoder, so gate on proprio_dim, not proprio_encoder.
        want_proprio = (self.proprio_encoder is not None) or (self.proprio_dim is not None)
        if want_proprio and proprio is not None:
            proprio_first = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool)
        return {
            "input_latents": input_latents,
            "context": context,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "proprio_first": proprio_first,
        }

    # ------------------------------------------------------------- train loss
    def training_loss(self, sample, tiled: bool = False):
        self._reset_semantic_condition()
        try:
            return self._training_loss_impl(sample, tiled=tiled)
        except Exception:
            self._reset_semantic_condition()
            raise

    def _training_loss_impl(self, sample, tiled: bool = False):
        raw_video = sample["video"].to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        self._prepare_semantic_condition(sample, raw_video[:, :, 0])
        inp = self.build_inputs(sample, tiled=tiled, video=raw_video)
        latents = inp["input_latents"]
        action = inp["action"]
        crossattn = self.text_proj(inp["context"])  # Qwen 3584 -> crossattn_dim (1024)
        B = latents.shape[0]

        # AGRA coupling: the proprio state s0 is a PREPENDED token INSIDE the action
        # head (not appended to the text context), and the foresight pass needs the
        # clean first latent frame (o0). Thread both onto the model for the coupling,
        # and skip the text-context proprio append (that is for mot/cross_attn). The
        # action head owns its own proprio encoder, so we pass the RAW proprio_first.
        if self.coupling == "agra":
            self._agra_o0_latent = latents[:, :, :1]  # [B, 16, 1, H, W] (first latent frame)
            self._agra_proprio0 = inp.get("proprio_first", None)
        else:
            # mot / cross_attn: append the proprio token to the text cross-attn context
            # (action-conditioning side info; AGRA feeds proprio into the head instead).
            if self.proprio_encoder is not None and inp["proprio_first"] is not None:
                ptok = self.proprio_encoder(inp["proprio_first"]).unsqueeze(1).to(crossattn.dtype)
                crossattn = torch.cat([crossattn, ptok], dim=1)
            # mot (FastWAM-faithful): condition the video stream on the current
            # observation by keeping latent frame 0 clean, so the action's joint
            # self-attention reads the current image (see couplings/mot.py).
            if self.coupling == "mot":
                self._mot_o0_latent = latents[:, :, :1]
                self._mot_cond_frames = 1

        noise_v = torch.randn_like(latents)
        t_v = self.train_video_scheduler.sample_training_t(B, self.device, latents.dtype)
        noisy_v = self.train_video_scheduler.add_noise(latents, noise_v, t_v)
        target_v = self.train_video_scheduler.training_target(latents, noise_v, t_v)

        noise_a = torch.randn_like(action)
        t_a = self.train_action_scheduler.sample_training_t(B, self.device, action.dtype)
        noisy_a = self.train_action_scheduler.add_noise(action, noise_a, t_a)
        target_a = self.train_action_scheduler.training_target(action, noise_a, t_a)

        pred_v, pred_a = self.couple_forward(noisy_v, t_v, noisy_a, t_a, crossattn)

        if self.coupling == "mot" and getattr(self, "_mot_o0_latent", None) is not None:
            # First-frame conditioning: the conditioning frame(s) are CLEAN inputs (the
            # current observation), not predictions -> exclude them from the video loss,
            # exactly like Wan FastWAM (pred_video[:, :, 1:] / target[:, :, 1:] when fuse
            # is on, fastwam.py:535-537). Training to "denoise" the clean cond frame is
            # wrong and wastes capacity.
            cf = int(getattr(self, "_mot_cond_frames", 1))
            loss_v = F.mse_loss(pred_v[:, :, cf:].float(), target_v[:, :, cf:].float())
        else:
            loss_v = F.mse_loss(pred_v.float(), target_v.float())
        w_v = self.train_video_scheduler.training_weight(t_v).to(loss_v)
        loss_video = (loss_v * w_v.mean())

        a_tok = F.mse_loss(pred_a.float(), target_a.float(), reduction="none").mean(dim=2)
        if inp["action_is_pad"] is not None:
            valid = (~inp["action_is_pad"]).to(a_tok)
            a_per = (a_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            a_per = a_tok.mean(dim=1)
        w_a = self.train_action_scheduler.training_weight(t_a).to(a_per)
        loss_action = (a_per * w_a).mean()

        loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss, {
            "loss": loss.detach(),
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
        }

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

    # ------------------------------------------------------------- inference
    def _load_text_context_from_cache(self, prompt: str):
        """Fetch the precomputed Qwen2.5-VL embedding for ``prompt`` (same scheme as
        the dataset: sha256(prompt) -> {context[L,qwen], mask[L]}). The cache dir is
        FASTWAM_TEXT_CACHE_DIR (falls back to the LIBERO default); context_len from
        FASTWAM_TEXT_CONTEXT_LEN (default 128). There is no live text encoder in the
        cosmos path, so all eval prompts must be pre-cached."""
        import hashlib
        cache_dir = os.environ.get(
            "FASTWAM_TEXT_CACHE_DIR", "./data/text_embeds_cache/libero_qwen"
        )
        ctx_len = int(os.environ.get("FASTWAM_TEXT_CONTEXT_LEN", "128"))
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = os.path.join(cache_dir, f"{hashed}.t5_len{ctx_len}.wan22ti2v5b.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing text-embedding cache for eval prompt: {path}. "
                "Precompute it with scripts/precompute_text_embeds_qwen.py."
            )
        payload = torch.load(path, map_location="cpu")
        return payload["context"]  # [L, qwen_dim]

    @torch.no_grad()
    def infer_action(
        self,
        prompt=None,
        input_image=None,
        action_horizon: int = 32,
        proprio=None,
        context=None,
        context_mask=None,        # accepted for API parity; unused (mask via zero rows)
        negative_prompt=None,     # accepted for API parity; CFG not used (trained w/o)
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames=None,
        cond_frames: int = 1,
        semantic_plan=None,
        semantic_plan_times=None,
        instruction: str | list[str] | None = None,
        video_fps: float | torch.Tensor | None = None,
    ):
        self._reset_semantic_condition()
        try:
            return self._infer_action_impl(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                num_video_frames=num_video_frames,
                cond_frames=cond_frames,
                semantic_plan=semantic_plan,
                semantic_plan_times=semantic_plan_times,
                instruction=instruction,
                video_fps=video_fps,
            )
        except Exception:
            self._reset_semantic_condition()
            raise

    def _infer_action_impl(
        self,
        prompt=None,
        input_image=None,
        action_horizon: int = 32,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames=None,
        cond_frames: int = 1,
        semantic_plan=None,
        semantic_plan_times=None,
        instruction: str | list[str] | None = None,
        video_fps: float | torch.Tensor | None = None,
    ):
        """Predict an action chunk for the current observation.

        cross_attn/mot coupling: the action DiT cross-attends [text ; proprio ; video
        features]. We joint-denoise the (o0-conditioned) future video and the action
        together via rectified-flow Euler steps, re-pinning the first latent frame to
        the encoded current observation each step so the video features carry the
        current image. Returns {"action": [action_horizon, action_dim] float32 cpu}.
        """
        device, dtype = self.device, self.torch_dtype
        self.eval()

        # Move/normalise the composed observation once.  Both the frozen planner
        # and the VAE consume this exact tensor; no image encode/decode round-trip.
        if input_image is None:
            raise ValueError("infer_action needs `input_image`.")
        img = input_image.to(device=device, dtype=dtype)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(
                "input_image must have shape [B, 3, H, W], got "
                f"{tuple(img.shape)}"
            )
        batch_size = int(img.shape[0])
        self._prepare_semantic_condition(
            {
                "semantic_plan": semantic_plan,
                "semantic_plan_times": semantic_plan_times,
                "instruction": instruction,
                "video_fps": video_fps,
            },
            img,
        )

        # ---- text context (Qwen cache) -> crossattn dim ----
        if context is None:
            if prompt is None:
                raise ValueError("infer_action needs `prompt` or `context`.")
            context = self._load_text_context_from_cache(prompt)
        context = context.to(device=device, dtype=dtype)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.ndim != 3:
            raise ValueError(
                "context must have shape [B, L, D] or [L, D], got "
                f"{tuple(context.shape)}"
            )
        if context.shape[0] == 1 and batch_size > 1:
            context = context.expand(batch_size, -1, -1)
        if context.shape[0] != batch_size:
            raise ValueError(
                f"context batch must equal image batch {batch_size}, "
                f"got {context.shape[0]}"
            )
        crossattn = self.text_proj(context)

        # ---- proprio token (mot/cross_attn append it to the text context) ----
        if self.proprio_encoder is not None and proprio is not None:
            p = proprio.to(device=device, dtype=dtype)
            if p.ndim == 1:
                p = p.unsqueeze(0)
            if p.ndim == 3:
                p = p[:, 0, :]                        # [1, proprio_dim]
            ptok = self.proprio_encoder(p).unsqueeze(1).to(crossattn.dtype)
            crossattn = torch.cat([crossattn, ptok], dim=1)

        # ---- encode current observation o0 -> first latent frame ----
        o0_latent = self._vae_encode(img.unsqueeze(2)).to(dtype)
        _, Cz, _, hz, wz = o0_latent.shape

        # ---- video latent temporal length (Wan VAE temporal factor 4) ----
        nvf = 9 if num_video_frames is None else int(num_video_frames)
        T_lat = max((nvf - 1) // 4 + 1, 1)

        # ---- AGRA: the foresight (FIXED tau_v=1, o0-conditioned) is independent of the
        # action and the action timestep, so compute it ONCE and reuse across all action
        # denoising steps (the generic loop below would redraw fresh noise each step ->
        # inconsistent foresight). The action head prepends the proprio state token. ----
        if self.coupling == "agra":
            cond = min(int(cond_frames), T_lat)
            gen = (
                None
                if seed is None
                else torch.Generator(device=rand_device).manual_seed(int(seed))
            )
            noise = torch.randn(
                (batch_size, Cz, T_lat, hz, wz),
                generator=gen,
                device=rand_device,
            ).to(device, dtype)
            # pure noise (sigma=1): t = num_train_timesteps (scheduler convention is
            # t in [0,N], sigma=t/N), NOT 1.0. The o0 frame is overridden to t=0 inside.
            ts1 = noise.new_full(
                (batch_size, T_lat),
                float(self.train_video_scheduler.num_train_timesteps),
            )
            feats = self.video_expert.forward_foresight(
                noise, ts1, crossattn, layers=self.agra_video_layers,
                o0_latent=o0_latent, cond_frames=cond,
                semantic_plan_B_L_D=self._current_semantic_plan,
                semantic_plan_times_B_N=self._current_semantic_plan_times)
            G = [proj(f.to(proj.weight.dtype)) for proj, f in zip(self.agra_video_projs, feats)]
            p = None
            if proprio is not None:
                p = proprio.to(device=device, dtype=dtype)
                if p.ndim == 1:
                    p = p.unsqueeze(0)
                if p.ndim == 3:
                    p = p[:, 0, :]
            action_latent = torch.randn(
                (
                    batch_size,
                    int(action_horizon),
                    self.action_expert.action_dim,
                ),
                generator=gen,
                device=rand_device,
            ).to(device, dtype)
            ts_a, da = self.train_action_scheduler.build_inference_schedule(
                num_inference_steps, device, dtype, shift_override=sigma_shift)
            for i in range(int(num_inference_steps)):
                t_a = (
                    ts_a[i]
                    .reshape(1)
                    .to(device=device, dtype=dtype)
                    .expand(batch_size)
                )
                pred_a = self.action_head(action_latent, t_a, p, G)
                action_latent = self.train_action_scheduler.step(pred_a, da[i], action_latent)
            output = action_latent.detach().to(device="cpu", dtype=torch.float32)
            return {"action": output[0] if batch_size == 1 else output}

        # ---- init noise ----
        gen = None
        if seed is not None:
            gen = torch.Generator(device=rand_device).manual_seed(int(seed))
        video_latent = torch.randn(
            (batch_size, Cz, T_lat, hz, wz), generator=gen, device=rand_device
        ).to(device=device, dtype=dtype)
        action_latent = torch.randn(
            (batch_size, int(action_horizon), self.action_expert.action_dim),
            generator=gen, device=rand_device,
        ).to(device=device, dtype=dtype)

        # ---- rectified-flow inference schedules (same scheduler as training) ----
        ts_v, dv = self.train_video_scheduler.build_inference_schedule(
            num_inference_steps, device, dtype, shift_override=sigma_shift
        )
        ts_a, da = self.train_action_scheduler.build_inference_schedule(
            num_inference_steps, device, dtype, shift_override=sigma_shift
        )

        cond = min(int(cond_frames), T_lat)
        o0_rep = o0_latent[:, :, :1].expand(batch_size, Cz, cond, hz, wz)
        # mot: feed the clean first frame through the SAME o0-conditioning path as
        # training (frame-replace + mask channel + per-frame timestep), so the action's
        # joint self-attention reads the current image exactly as it was trained.
        if self.coupling == "mot":
            self._mot_o0_latent = o0_latent
            self._mot_cond_frames = cond
        for i in range(int(num_inference_steps)):
            # re-pin the conditioning frame(s) to the clean current observation
            video_latent[:, :, :cond] = o0_rep
            t_v = ts_v[i].reshape(1).to(device=device, dtype=dtype).expand(batch_size)
            t_a = ts_a[i].reshape(1).to(device=device, dtype=dtype).expand(batch_size)
            pred_v, pred_a = self.couple_forward(video_latent, t_v, action_latent, t_a, crossattn)
            video_latent = self.train_video_scheduler.step(pred_v, dv[i], video_latent)
            action_latent = self.train_action_scheduler.step(pred_a, da[i], action_latent)

        output = action_latent.detach().to(device="cpu", dtype=torch.float32)
        return {"action": output[0] if batch_size == 1 else output}

    # ------------------------------------------------------------- checkpoint
    def state_payload(self, step=None):
        """Build the weights-checkpoint payload. ``self.dit.state_dict()`` triggers the
        FSDP FULL_STATE_DICT all-gather, which is a COLLECTIVE op — under FSDP this must
        be called on EVERY rank (the trainer does so; only rank 0 writes the file). If
        only one rank calls it, that rank's all-gather hangs until the NCCL watchdog
        times out (~30 min) and SIGABRTs the job. Also save the cross_attn/text learned
        projections so resume/eval restore the full model."""
        payload = {"dit_cosmos": self.dit.state_dict(), "step": step}
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if not isinstance(self.text_proj, nn.Identity):
            payload["text_proj"] = self.text_proj.state_dict()
        if self.video_feat_proj is not None:
            payload["video_feat_proj"] = self.video_feat_proj.state_dict()
        if getattr(self, "agra_video_projs", None) is not None:  # AGRA per-layer projs
            payload["agra_video_projs"] = self.agra_video_projs.state_dict()
        return payload

    def save_checkpoint(self, path, optimizer=None, step=None):
        # NOTE: under FSDP, prefer the trainer's all-rank state_payload() path; a bare
        # single-rank call here would hang on the FSDP gather.
        torch.save(self.state_payload(step=step), path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # Consolidated full state dict (FSDP-correct; keys are the real submodule paths
        # video_expert.* / action_expert.* / text_proj.* / ...). Accept it either wrapped
        # as {"model": sd} (our weights .pt) or RAW (accelerate's pytorch_model_fsdp.bin).
        sd = None
        if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
            sd = payload["model"]
        elif isinstance(payload, dict) and "dit_cosmos" not in payload and payload \
                and all(torch.is_tensor(v) for v in payload.values()):
            sd = payload
        if sd is not None:
            sd = {k: (v.to(self.torch_dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
                  for k, v in sd.items()}
            self.load_state_dict(sd, strict=False)
            return
        # Legacy structured format (pre-fix; root params may be corrupt under FSDP).
        self.dit.load_state_dict(payload["dit_cosmos"], strict=False)
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"])
        if "text_proj" in payload and not isinstance(self.text_proj, nn.Identity):
            self.text_proj.load_state_dict(payload["text_proj"])
        if self.video_feat_proj is not None and "video_feat_proj" in payload:
            self.video_feat_proj.load_state_dict(payload["video_feat_proj"])
        if getattr(self, "agra_video_projs", None) is not None and "agra_video_projs" in payload:
            self.agra_video_projs.load_state_dict(payload["agra_video_projs"])
