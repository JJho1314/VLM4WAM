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

from typing import List, Optional, Tuple

import torch

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(
        self,
        *args,
        timestep_scale: float = 1.0,
        concat_target_mask: bool = False,
        target_mask_context_tokens: bool = False,
        **kwargs,
    ):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        if concat_target_mask:
            kwargs["in_channels"] += 1  # Add 1 for the TAViD-style target mask.
        kwargs["target_mask_concat_input"] = concat_target_mask
        kwargs["target_mask_context_tokens"] = target_mask_context_tokens
        self.concat_target_mask = concat_target_mask
        self.target_mask_context_tokens = target_mask_context_tokens
        self.timestep_scale = timestep_scale
        super().__init__(*args, **kwargs)
        if self.concat_target_mask:
            self._zero_target_mask_patch_weights()

    def _zero_target_mask_patch_weights(self) -> None:
        """Keep initial behavior identical to the pretrained model.

        Target mask is appended after latent, condition-mask, and padding-mask
        channels. Zeroing the final patch-channel weights makes the new input
        inert until fine-tuning learns to use it.
        """
        patch_dim = self.patch_spatial * self.patch_spatial * self.patch_temporal
        with torch.no_grad():
            self.x_embedder.proj[1].weight[:, -patch_dim:] = 0

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        target_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        target_feature_B_L_D: Optional[torch.Tensor] = None,
        tgt_token_indices_B: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
            if self.concat_target_mask and target_mask_B_C_T_H_W is None:
                B, _, T, H, W = x_B_C_T_H_W.shape
                target_mask_B_C_T_H_W = torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
            target_mask_B_C_T_H_W = None

        needs_target_mask_for_loss = bool(getattr(self, "tavid_attn_alignment_blocks", []))
        target_mask_for_super = (
            target_mask_B_C_T_H_W
            if (self.concat_target_mask or self.target_mask_context_tokens or needs_target_mask_for_loss)
            else None
        )
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
            target_mask_B_C_T_H_W=target_mask_for_super,
            target_feature_B_L_D=target_feature_B_L_D,
            tgt_token_indices_B=tgt_token_indices_B,
        )
