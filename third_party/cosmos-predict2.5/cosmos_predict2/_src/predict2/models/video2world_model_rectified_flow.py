# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.denoise_prediction import DenoisePrediction
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldCondition,
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)

NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(str, Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class Video2WorldModelRectifiedFlowConfig(Text2WorldModelRectifiedFlowConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    conditional_frame_timestep: float = (
        -1.0
    )  # Noise level used for conditional frames; default is -1 which will not take effective
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
    denoise_replace_gt_frames: bool = True  # Whether to denoise the ground truth frames
    conditional_frames_probs: Optional[Dict[int, float]] = None  # Probability distribution for conditional frames
    target_mask_condition_frames_only: bool = True  # Keep target mask on video-conditioning frames, TAViD-style.
    target_attention_loss_weight: float = 0.0
    target_attention_loss_eps: float = 1e-6
    target_attention_background_loss_weight: float = 0.25
    target_attention_mass_loss_weight: float = 0.1

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]


class Video2WorldModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        # generate random number of conditional frames for training
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        target_mask = data_batch.get("target_mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(device=latent_state.device, dtype=latent_state.dtype)
            target_mask = F.interpolate(target_mask, size=latent_state.shape[2:], mode="nearest")
            if self.config.target_mask_condition_frames_only:
                target_mask = target_mask * condition.condition_video_input_mask_B_C_T_H_W.type_as(target_mask)
            condition = condition.set_target_mask(target_mask)
        target_feature = data_batch.get("target_feature", None)
        if target_feature is not None:
            target_feature = target_feature.to(device=latent_state.device, dtype=latent_state.dtype)
            condition = condition.set_target_feature(target_feature)
        tgt_token_indices = data_batch.get("tgt_token_indices", None)
        if tgt_token_indices is not None:
            condition = condition.set_tgt_token_indices(tgt_token_indices.to(device=latent_state.device, dtype=torch.long))
        return raw_state, latent_state, condition

    def compute_extra_training_loss(self, condition: Video2WorldCondition) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.config.target_attention_loss_weight <= 0:
            return {}, torch.zeros((), **self.tensor_kwargs_fp32)
        target_attn_maps = getattr(self.net, "tavid_target_attn_maps", [])
        target_attn_source = getattr(self.net, "tavid_target_attn_source", "none")
        target_mask = getattr(self.net, "tavid_target_mask_B_T_H_W", None)
        if not target_attn_maps or target_mask is None:
            return {}, torch.zeros((), **self.tensor_kwargs_fp32)
        target_mask = target_mask.float().clamp(0, 1)
        frame_valid = target_mask.flatten(2).sum(dim=2) > 0
        token_valid = frame_valid[:, :, None, None].expand_as(target_mask)
        mask_flat = rearrange(target_mask, "b t h w -> b (t h w)")
        token_valid_flat = rearrange(token_valid, "b t h w -> b (t h w)")
        pos_weight = mask_flat * token_valid_flat.type_as(mask_flat)
        neg_weight = (1.0 - mask_flat).clamp(min=0.0) * token_valid_flat.type_as(mask_flat)
        pos_sum = pos_weight.sum(dim=1)
        neg_sum = neg_weight.sum(dim=1)
        valid = (pos_sum > 0) & (neg_sum > 0)
        if not bool(valid.any()):
            return {}, torch.zeros((), **self.tensor_kwargs_fp32)

        eps = self.config.target_attention_loss_eps
        supervised_area = token_valid_flat.float().sum(dim=1).clamp_min(1.0)
        mask_area_ratio = pos_sum / supervised_area

        # Target-aware loss on supervised frames only. Foreground and background
        # are averaged separately so small target masks are not diluted by the
        # much larger non-target region. The mass term directly rewards putting
        # cross-attention probability inside the target mask.
        attn_map = torch.stack([attn_map.float() for attn_map in target_attn_maps], dim=0).mean(dim=0)
        attn_flat = rearrange(attn_map, "b t h w -> b (t h w)").clamp(min=0.0)
        attn_min = torch.where(token_valid_flat, attn_flat, torch.full_like(attn_flat, float("inf"))).amin(
            dim=1, keepdim=True
        )
        attn_max = torch.where(token_valid_flat, attn_flat, torch.full_like(attn_flat, float("-inf"))).amax(
            dim=1, keepdim=True
        )
        attn_map_01 = (attn_flat - attn_min) / (attn_max - attn_min + eps)
        attn_map_01 = torch.where(token_valid_flat, attn_map_01, torch.zeros_like(attn_map_01))

        pos_mse = (((1.0 - attn_map_01) ** 2) * pos_weight).sum(dim=1) / (pos_sum + eps)
        neg_mse = ((attn_map_01**2) * neg_weight).sum(dim=1) / (neg_sum + eps)

        supervised_attn = attn_flat * token_valid_flat.type_as(attn_flat)
        attn_dist = supervised_attn / (supervised_attn.sum(dim=1, keepdim=True) + eps)
        target_mass = (attn_dist * pos_weight).sum(dim=1)
        mass_loss = -torch.log(target_mass.clamp_min(eps))

        per_sample = (
            pos_mse
            + self.config.target_attention_background_loss_weight * neg_mse
            + self.config.target_attention_mass_loss_weight * mass_loss
        )
        align_loss = per_sample[valid].mean()

        weighted_loss = align_loss * self.config.target_attention_loss_weight
        zero = torch.zeros((), device=align_loss.device, dtype=align_loss.dtype)

        outside_valid = valid & (neg_sum > 0)
        inside_mean = (attn_flat * pos_weight).sum(dim=1) / (pos_sum + eps)
        outside_mean = (attn_flat * neg_weight).sum(dim=1) / (neg_sum + eps)
        inside_outside_ratio = inside_mean / (outside_mean + eps)

        mask_area = mask_area_ratio[valid].mean()
        return {
            "target_attention_loss": align_loss.detach(),
            "target_attention_loss_weighted": weighted_loss.detach(),
            "target_attention_pos_mse": pos_mse[valid].mean().detach(),
            "target_attention_neg_mse": neg_mse[valid].mean().detach(),
            "target_attention_mass_loss": mass_loss[valid].mean().detach(),
            "target_attention_mask_valid_ratio": valid.float().mean().detach(),
            "target_attention_mask_area_ratio": mask_area.detach(),
            "target_attention_mass_in_mask": target_mass[valid].mean().detach(),
            "target_attention_mass_lift": (target_mass / (mask_area_ratio + eps))[valid].mean().detach(),
            "target_attention_inside_mean": inside_mean[valid].mean().detach(),
            "target_attention_outside_mean": (
                outside_mean[outside_valid].mean() if bool(outside_valid.any()) else zero
            ).detach(),
            "target_attention_inside_outside_ratio": (
                inside_outside_ratio[outside_valid].mean() if bool(outside_valid.any()) else zero
            ).detach(),
            "target_attention_num_maps": torch.tensor(
                float(len(target_attn_maps)), device=align_loss.device, dtype=align_loss.dtype
            ),
            "target_attention_source_is_target_branch": torch.tensor(
                float(target_attn_source == "target_branch"), device=align_loss.device, dtype=align_loss.dtype
            ),
        }, weighted_loss

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> DenoisePrediction:
        """
        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (Text2WorldCondition): conditional information, generated from self.conditioner

        Returns:
            velocity prediction
        """
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = rearrange(timesteps_B_T, "b t -> b 1 t 1 1")
                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + (
                    timesteps_B_1_T_1_1 * (1 - condition_video_mask_B_1_T_1_1)
                )
                timesteps_B_T = timesteps_B_1_T_1_1.squeeze(dim=(1, 3, 4))

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=timesteps_B_T,  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return velocity predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

        target_mask = data_batch.get("target_mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(device=x0.device, dtype=x0.dtype)
            target_mask = F.interpolate(target_mask, size=x0.shape[2:], mode="nearest")
            if self.config.target_mask_condition_frames_only:
                target_mask = target_mask * condition.condition_video_input_mask_B_C_T_H_W.type_as(target_mask)
            condition = condition.set_target_mask(target_mask)

        target_feature = data_batch.get("target_feature", None)
        if target_feature is not None:
            target_feature = target_feature.to(device=x0.device, dtype=x0.dtype)
            condition = condition.set_target_feature(target_feature)

        tgt_token_indices = data_batch.get("tgt_token_indices", None)
        if tgt_token_indices is not None:
            condition = condition.set_tgt_token_indices(tgt_token_indices.to(device=x0.device, dtype=torch.long))

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn
