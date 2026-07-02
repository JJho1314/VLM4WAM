import math
import random
from os import PathLike
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def sample_semantic_plan_dropout(prob: float, device: torch.device) -> bool:
    """Draw the training-time semantic-plan dropout decision, synchronized across all ranks.

    The decision MUST be identical on every rank: semantic_plan_context_adapter is its own
    FSDP unit, so a rank that skips it never issues its all-gather and the collective
    schedules diverge (observed as a mesh_shard NCCL all-gather timeout on the first
    training step). Rank 0 draws the coin and broadcasts it.
    """
    if prob <= 0:
        return False
    if dist.is_available() and dist.is_initialized():
        flag = torch.zeros(1, device=device, dtype=torch.uint8)
        if dist.get_rank() == 0 and random.random() < prob:
            flag.fill_(1)
        dist.broadcast(flag, src=0)
        return bool(flag.item())
    return random.random() < prob


def _infer_square_token_grid(num_tokens: int) -> Tuple[int, bool]:
    """Infer (side, has_cls) from a ViT token count, tolerating one leading CLS token."""
    side = int(math.isqrt(num_tokens))
    if side * side == num_tokens:
        return side, False
    side = int(math.isqrt(num_tokens - 1))
    if side * side == num_tokens - 1:
        return side, True
    raise ValueError(f"Cannot infer square ViT token grid from {num_tokens} tokens")


class OnlineSemanticPlanEncoder:
    """Frozen SigLIP2 encoder producing semantic plans on the fly from training video windows.

    Replicates build_siglip2_semantic_plan_labels.encode_images: penultimate encoder-layer
    hidden states, optional CLS strip, optional adaptive average pooling to a square grid
    (grid_size<=0 keeps the native patch grid). Input video must be Cosmos-normalized to
    [-1,1], which already equals SigLIP2's mean/std=0.5 normalization, so only resizing to
    the encoder resolution is needed.

    Instances are plain objects (not nn.Module) so that holders keep the frozen encoder out
    of state_dict / EMA / FSDP sharding.
    """

    def __init__(
        self,
        model_path: str,
        *,
        num_keyframes: int = 16,
        grid_size: int = 0,
        image_size: int = 384,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = str(model_path)
        self.num_keyframes = int(num_keyframes)
        self.grid_size = int(grid_size)
        self.image_size = int(image_size)
        self.dtype = dtype
        self._vision: Optional[nn.Module] = None

    def _ensure_loaded(self, device: torch.device) -> nn.Module:
        if self._vision is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(self.model_path, torch_dtype=self.dtype, local_files_only=True)
            if not hasattr(model, "vision_model"):
                raise RuntimeError("Expected an AutoModel with a vision_model attribute for SigLIP2.")
            vision = model.vision_model
            vision.eval()
            vision.requires_grad_(False)
            self._vision = vision
        if next(self._vision.parameters()).device != torch.device(device):
            self._vision.to(device)
        return self._vision

    def keyframe_indices(self, num_frames: int) -> torch.Tensor:
        """Window positions of the encoded keyframes; matches the offline builder's
        sample_cosmos_indices (uniform over [1, T-1], excluding the conditioning frame)."""
        n = max(min(self.num_keyframes, int(num_frames)), 1)
        if num_frames <= 1:
            return torch.zeros(n, dtype=torch.long)
        return torch.linspace(1, num_frames - 1, n).round().long()

    @torch.no_grad()
    def __call__(self, video_B_C_T_H_W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a [-1,1]-normalized video window.

        Returns (semantic_plan [B, N*S, D] float32, keyframe_times [B, N] in [0,1]).
        """
        if video_B_C_T_H_W.ndim != 5:
            raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(video_B_C_T_H_W.shape)}")
        batch, _, num_frames, height, width = video_B_C_T_H_W.shape
        vision = self._ensure_loaded(video_B_C_T_H_W.device)

        idx = self.keyframe_indices(num_frames).to(video_B_C_T_H_W.device)
        n = int(idx.shape[0])
        frames = video_B_C_T_H_W.index_select(2, idx)
        frames = frames.permute(0, 2, 1, 3, 4).reshape(batch * n, -1, height, width)
        if (height, width) != (self.image_size, self.image_size):
            frames = F.interpolate(
                frames.float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        frames = frames.to(dtype=next(vision.parameters()).dtype)

        # cuDNN disabled: the training environments run with broken cuDNN sublibrary loading
        # (see COSMOS_DISABLE_CUDNN_SDPA); the patch-embed Conv2d falls back to PyTorch's own
        # kernels at negligible cost.
        with torch.backends.cudnn.flags(enabled=False):
            hidden = vision.embeddings(frames, interpolate_pos_encoding=False)
            layers = list(vision.encoder.layers)
            if len(layers) < 2:
                raise RuntimeError(f"Expected at least two SigLIP2 vision layers, got {len(layers)}.")
            penultimate = None
            for layer_idx, layer in enumerate(layers):
                hidden = layer(hidden, None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                if layer_idx == len(layers) - 2:
                    penultimate = hidden
                    break
        if penultimate is None:
            raise RuntimeError("Failed to capture SigLIP2 penultimate spatial features.")
        tokens = penultimate.float()

        side, has_cls = _infer_square_token_grid(tokens.shape[1])
        patches = tokens[:, 1:] if has_cls else tokens
        if self.grid_size > 0:
            grid = patches.reshape(batch * n, side, side, -1).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(grid, (self.grid_size, self.grid_size))
            patches = pooled.permute(0, 2, 3, 1).reshape(batch * n, self.grid_size * self.grid_size, -1)

        semantic_plan = patches.reshape(batch, n * patches.shape[1], patches.shape[-1])
        denom = max(num_frames - 1, 1)
        times = (idx.float() / float(denom)).clamp_(0.0, 1.0).unsqueeze(0).expand(batch, -1).contiguous()
        return semantic_plan, times


def infer_semantic_plan_shape(
    *,
    num_tokens: int,
    num_keyframes: int = 0,
    spatial_grid_size: int = 0,
) -> Tuple[int, int, int]:
    """Infer the time-major [N, Hs, Ws] layout of flattened semantic-plan tokens."""
    num_tokens = int(num_tokens)
    num_keyframes = int(num_keyframes)
    spatial_grid_size = int(spatial_grid_size)
    if num_tokens <= 0:
        raise ValueError(f"semantic_plan must have at least one token, got {num_tokens}")

    if num_keyframes > 0 and spatial_grid_size > 0:
        expected = num_keyframes * spatial_grid_size * spatial_grid_size
        if num_tokens < expected:
            raise ValueError(
                f"semantic_plan has {num_tokens} tokens, fewer than configured "
                f"N*Hs*Ws={expected} ({num_keyframes}*{spatial_grid_size}*{spatial_grid_size})"
            )
        return num_keyframes, spatial_grid_size, spatial_grid_size

    if num_keyframes > 0:
        if num_tokens % num_keyframes != 0:
            raise ValueError(f"semantic_plan tokens {num_tokens} are not divisible by N={num_keyframes}")
        spatial_tokens = num_tokens // num_keyframes
        side = int(math.isqrt(spatial_tokens))
        if side * side != spatial_tokens:
            raise ValueError(
                f"semantic_plan has {spatial_tokens} tokens per keyframe; expected a square spatial grid"
            )
        return num_keyframes, side, side

    side = int(math.isqrt(num_tokens))
    if side * side == num_tokens:
        return 1, side, side
    raise ValueError("semantic_plan shape is ambiguous; set num_keyframes and spatial_grid_size")


def normalize_semantic_plan_times(
    keyframe_times: Optional[torch.Tensor],
    *,
    num_keyframes: int = 0,
) -> Optional[torch.Tensor]:
    """Validate per-keyframe times normalized to [0,1]; returns [B,N] float32 or None when unusable.

    Negative values (e.g. a -1 sentinel for "unknown") invalidate the whole tensor so callers
    fall back to the uniform-spacing assumption.
    """
    if keyframe_times is None:
        return None
    times = keyframe_times
    if not isinstance(times, torch.Tensor):
        times = torch.as_tensor(times)
    times = times.float()
    if times.ndim == 1:
        times = times.unsqueeze(0)
    if times.ndim != 2 or times.shape[1] == 0:
        return None
    if num_keyframes > 0 and times.shape[1] != num_keyframes:
        return None
    if not torch.isfinite(times).all():
        return None
    if times.min() < 0.0 or times.max() > 1.0:
        return None
    return times


def semantic_plan_times_from_frame_indices(
    future_frame_indices: Optional[Sequence[int]],
    video_frame_indices: Optional[Sequence[int]],
) -> Optional[torch.Tensor]:
    """Normalized [0,1] keyframe times within the sampled clip window.

    0 maps to the clip's first (conditioning) frame and 1 to its last frame, matching how the
    video latent temporal axis spans the clip.
    """
    if not future_frame_indices or not video_frame_indices or len(video_frame_indices) < 2:
        return None
    start = float(video_frame_indices[0])
    end = float(video_frame_indices[-1])
    span = end - start
    if span <= 0:
        return None
    times = [(min(max(float(f), start), end) - start) / span for f in future_frame_indices]
    return torch.tensor(times, dtype=torch.float32)


def build_semantic_plan_coords(
    *,
    batch_size: int,
    num_tokens: int,
    num_keyframes: int = 0,
    spatial_grid_size: int = 0,
    keyframe_times_B_N: Optional[torch.Tensor] = None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build normalized time-major [t, y, x] coordinates for semantic-plan tokens.

    When keyframe_times_B_N is given, the t axis uses the actual keyframe times (mapped from
    [0,1] to [-1,1]) instead of assuming keyframes are uniformly spread over the clip.
    """
    n, h, w = infer_semantic_plan_shape(
        num_tokens=num_tokens,
        num_keyframes=num_keyframes,
        spatial_grid_size=spatial_grid_size,
    )
    total = n * h * w
    batch_size = int(batch_size)
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    times_B_N = normalize_semantic_plan_times(keyframe_times_B_N, num_keyframes=n)
    if times_B_N is None:
        t = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
        tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
        coords = torch.stack([tt, yy, xx], dim=-1).reshape(1, total, 3)
        if total < num_tokens:
            pad = torch.zeros(1, num_tokens - total, 3, device=device, dtype=dtype)
            coords = torch.cat([coords, pad], dim=1)
        return coords[:, :num_tokens].expand(batch_size, -1, -1)

    t_B_N = times_B_N.to(device=device, dtype=dtype) * 2.0 - 1.0
    if t_B_N.shape[0] == 1 and batch_size > 1:
        t_B_N = t_B_N.expand(batch_size, -1)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack(
        [
            t_B_N[:, :, None, None].expand(batch_size, n, h, w),
            yy.expand(batch_size, n, h, w),
            xx.expand(batch_size, n, h, w),
        ],
        dim=-1,
    ).reshape(batch_size, total, 3)
    if total < num_tokens:
        pad = torch.zeros(batch_size, num_tokens - total, 3, device=device, dtype=dtype)
        coords = torch.cat([coords, pad], dim=1)
    return coords[:, :num_tokens]


def build_semantic_plan_rope_positions(
    *,
    batch_size: int,
    num_tokens: int,
    num_keyframes: int = 0,
    spatial_grid_size: int = 0,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    keyframe_times_N: Optional[torch.Tensor] = None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build Baton-style semantic key positions in the latent [T,H,W] coordinate frame.

    When keyframe_times_N is given ([N] or batch-shared [B,N], normalized to [0,1]), the
    temporal positions use the actual keyframe times scaled onto the latent time axis instead
    of assuming keyframes uniformly span the clip.
    """
    n, h, w = infer_semantic_plan_shape(
        num_tokens=num_tokens,
        num_keyframes=num_keyframes,
        spatial_grid_size=spatial_grid_size,
    )
    total = n * h * w

    t_scale = max(int(latent_t) - 1, 0)
    h_scale = max(int(latent_h) - 1, 0)
    w_scale = max(int(latent_w) - 1, 0)
    times_B_N = normalize_semantic_plan_times(keyframe_times_N, num_keyframes=n)
    if times_B_N is not None:
        t = times_B_N[0].to(device=device, dtype=dtype) * float(t_scale)
    else:
        t = torch.linspace(0.0, float(t_scale), n, device=device, dtype=dtype)
    y = torch.linspace(0.0, float(h_scale), h, device=device, dtype=dtype)
    x = torch.linspace(0.0, float(w_scale), w, device=device, dtype=dtype)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    positions = torch.stack([tt, yy, xx], dim=-1).reshape(1, total, 3)
    if total < num_tokens:
        pad = torch.zeros(1, num_tokens - total, 3, device=device, dtype=dtype)
        positions = torch.cat([positions, pad], dim=1)
    return positions[:, :num_tokens].expand(int(batch_size), -1, -1)


def _semantic_plan_spatial_tokens(
    num_tokens: int,
    spatial_grid_size: int,
    source_num_keyframes: int,
) -> int:
    if spatial_grid_size > 0:
        return spatial_grid_size * spatial_grid_size
    if source_num_keyframes > 0 and num_tokens % source_num_keyframes == 0:
        return num_tokens // source_num_keyframes
    return 0


def select_semantic_plan_keyframes_and_times(
    semantic_plan_B_L_D: torch.Tensor,
    keyframe_times_B_N: Optional[torch.Tensor] = None,
    *,
    num_keyframes: int,
    spatial_grid_size: int,
    source_num_keyframes: int = 0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Uniformly sample a flattened time-major semantic plan from source N to target N.

    Per-keyframe times (when given) are selected with the same keyframe indices so downstream
    RoPE/coord builders see the true timestamps of the kept keyframes. With spatial_grid_size<=0
    the per-keyframe token count is inferred from source_num_keyframes (native-grid plans).
    """
    num_keyframes = int(num_keyframes)
    spatial_grid_size = int(spatial_grid_size)
    source_num_keyframes = int(source_num_keyframes)
    times = keyframe_times_B_N
    if num_keyframes <= 0:
        return semantic_plan_B_L_D, times

    num_tokens = semantic_plan_B_L_D.shape[1]
    spatial_tokens = _semantic_plan_spatial_tokens(num_tokens, spatial_grid_size, source_num_keyframes)
    if spatial_tokens <= 0:
        return semantic_plan_B_L_D, times
    target_tokens = num_keyframes * spatial_tokens
    if num_tokens <= target_tokens:
        return semantic_plan_B_L_D, times

    if source_num_keyframes <= 0:
        if num_tokens % spatial_tokens != 0:
            return semantic_plan_B_L_D, times
        source_num_keyframes = num_tokens // spatial_tokens
    source_tokens = source_num_keyframes * spatial_tokens
    if num_tokens < source_tokens:
        raise ValueError(
            f"semantic_plan has {num_tokens} tokens, fewer than configured "
            f"source N*Hs*Ws={source_tokens} ({source_num_keyframes}*{spatial_grid_size}*{spatial_grid_size})"
        )
    if times is not None and times.shape[-1] != source_num_keyframes:
        times = None
    if source_num_keyframes <= num_keyframes:
        if times is not None:
            times = times[..., :num_keyframes]
        return semantic_plan_B_L_D[:, :target_tokens], times

    semantic_plan = semantic_plan_B_L_D[:, :source_tokens].reshape(
        semantic_plan_B_L_D.shape[0],
        source_num_keyframes,
        spatial_tokens,
        semantic_plan_B_L_D.shape[-1],
    )
    indices = torch.linspace(
        0,
        source_num_keyframes - 1,
        num_keyframes,
        device=semantic_plan_B_L_D.device,
    ).round().long()
    semantic_plan = semantic_plan.index_select(1, indices).reshape(
        semantic_plan_B_L_D.shape[0],
        target_tokens,
        semantic_plan_B_L_D.shape[-1],
    )
    if times is not None:
        times = times.index_select(-1, indices.to(times.device))
    return semantic_plan, times


def select_semantic_plan_keyframes(
    semantic_plan_B_L_D: torch.Tensor,
    *,
    num_keyframes: int,
    spatial_grid_size: int,
    source_num_keyframes: int = 0,
) -> torch.Tensor:
    """Uniformly sample a flattened time-major semantic plan from source N to target N."""
    semantic_plan, _ = select_semantic_plan_keyframes_and_times(
        semantic_plan_B_L_D,
        None,
        num_keyframes=num_keyframes,
        spatial_grid_size=spatial_grid_size,
        source_num_keyframes=source_num_keyframes,
    )
    return semantic_plan


def normalize_semantic_plan_tensor(
    semantic_plan: object,
    *,
    semantic_plan_dim: int = 0,
    max_tokens: int = 0,
) -> torch.Tensor:
    if isinstance(semantic_plan, dict):
        for key in ("semantic_plan", "features", "feature", "embeddings"):
            if key in semantic_plan:
                semantic_plan = semantic_plan[key]
                break
    if isinstance(semantic_plan, np.ndarray):
        semantic_plan = torch.from_numpy(semantic_plan)
    if not isinstance(semantic_plan, torch.Tensor):
        semantic_plan = torch.as_tensor(semantic_plan)
    semantic_plan = semantic_plan.float()
    if semantic_plan.ndim == 1:
        semantic_plan = semantic_plan.unsqueeze(0)
    elif semantic_plan.ndim > 2:
        semantic_plan = semantic_plan.reshape(-1, semantic_plan.shape[-1])
    if semantic_plan_dim > 0 and semantic_plan.shape[-1] != semantic_plan_dim:
        raise ValueError(f"Expected semantic_plan dim {semantic_plan_dim}, got {semantic_plan.shape[-1]}")
    if max_tokens > 0:
        semantic_plan = semantic_plan[:max_tokens]
        if semantic_plan.shape[0] < max_tokens:
            pad = torch.zeros(
                max_tokens - semantic_plan.shape[0],
                semantic_plan.shape[-1],
                dtype=semantic_plan.dtype,
            )
            semantic_plan = torch.cat([semantic_plan, pad], dim=0)
    return semantic_plan.contiguous()


def load_semantic_plan_payload(
    path: str | PathLike[str],
    *,
    semantic_plan_dim: int = 0,
    max_tokens: int = 0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load a semantic plan and, when the payload records frame indices, its keyframe times."""
    path = str(path)
    keyframe_times = None
    if path.endswith(".npy"):
        semantic_plan = np.load(path)
    elif path.endswith(".npz"):
        npz = np.load(path)
        key = "semantic_plan" if "semantic_plan" in npz else npz.files[0]
        semantic_plan = npz[key]
    else:
        semantic_plan = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(semantic_plan, dict):
            keyframe_times = semantic_plan_times_from_frame_indices(
                semantic_plan.get("future_frame_indices"),
                semantic_plan.get("video_frame_indices"),
            )
    semantic_plan = normalize_semantic_plan_tensor(
        semantic_plan,
        semantic_plan_dim=semantic_plan_dim,
        max_tokens=max_tokens,
    )
    return semantic_plan, keyframe_times


def load_semantic_plan_tensor(
    path: str | PathLike[str],
    *,
    semantic_plan_dim: int = 0,
    max_tokens: int = 0,
) -> torch.Tensor:
    semantic_plan, _ = load_semantic_plan_payload(
        path,
        semantic_plan_dim=semantic_plan_dim,
        max_tokens=max_tokens,
    )
    return semantic_plan


def trim_invalid_semantic_tokens(
    tokens_B_L_D: torch.Tensor,
    valid_B_L: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep_L = valid_B_L.any(dim=0)
    return tokens_B_L_D[:, keep_L], valid_B_L[:, keep_L], keep_L


class SemanticPlanContextAdapter(nn.Module):
    """Project continuous semantic-plan features into DiT cross-attention tokens."""

    def __init__(
        self,
        *,
        in_dim: int = 1152,
        hidden_dim: int = 2048,
        out_dim: int = 1024,
        max_tokens: int = 0,
        num_keyframes: int = 0,
        source_num_keyframes: int = 0,
        spatial_grid_size: int = 0,
        coord_hidden_dim: int = 256,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.max_tokens = int(max_tokens)
        self.num_keyframes = int(num_keyframes)
        self.source_num_keyframes = int(source_num_keyframes)
        self.spatial_grid_size = int(spatial_grid_size)
        hidden_dim = int(hidden_dim) if int(hidden_dim) > 0 else out_dim
        coord_hidden_dim = int(coord_hidden_dim)

        self.norm1 = nn.LayerNorm(self.in_dim, eps=eps, elementwise_affine=False)
        self.projection = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.coord_projection = None
        if coord_hidden_dim > 0:
            self.coord_projection = nn.Sequential(
                nn.Linear(3, coord_hidden_dim),
                nn.GELU(),
                nn.Linear(coord_hidden_dim, out_dim),
            )
        self.norm2 = nn.LayerNorm(out_dim, eps=eps, elementwise_affine=False)
        self.type_token = nn.Parameter(torch.zeros(out_dim))

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.type_token)

    def coords_for_tokens(
        self,
        batch_size: int,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
        keyframe_times_B_N: Optional[torch.Tensor] = None,
    ):
        return build_semantic_plan_coords(
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_keyframes=self.num_keyframes,
            spatial_grid_size=self.spatial_grid_size,
            keyframe_times_B_N=keyframe_times_B_N,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        semantic_plan_B_L_D: torch.Tensor,
        keyframe_times_B_N: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (tokens, valid mask, keyframe times of the kept keyframes or None)."""
        if semantic_plan_B_L_D.ndim == 2:
            semantic_plan_B_L_D = semantic_plan_B_L_D.unsqueeze(0)
        if semantic_plan_B_L_D.ndim != 3:
            raise ValueError(f"Expected semantic_plan shape [B,L,D], got {tuple(semantic_plan_B_L_D.shape)}")
        if semantic_plan_B_L_D.shape[-1] != self.in_dim:
            raise ValueError(f"Expected semantic_plan dim {self.in_dim}, got {semantic_plan_B_L_D.shape[-1]}")

        keyframe_times_B_N = normalize_semantic_plan_times(keyframe_times_B_N)
        semantic_plan_B_L_D, keyframe_times_B_N = select_semantic_plan_keyframes_and_times(
            semantic_plan_B_L_D,
            keyframe_times_B_N,
            num_keyframes=self.num_keyframes,
            source_num_keyframes=self.source_num_keyframes,
            spatial_grid_size=self.spatial_grid_size,
        )
        if self.max_tokens > 0 and semantic_plan_B_L_D.shape[1] > self.max_tokens:
            if keyframe_times_B_N is not None:
                # Times stay aligned only if the trim keeps whole keyframes.
                num_tokens = semantic_plan_B_L_D.shape[1]
                if num_tokens % keyframe_times_B_N.shape[-1] == 0:
                    spatial_tokens = num_tokens // keyframe_times_B_N.shape[-1]
                    if self.max_tokens % spatial_tokens == 0:
                        keyframe_times_B_N = keyframe_times_B_N[..., : self.max_tokens // spatial_tokens]
                    else:
                        keyframe_times_B_N = None
                else:
                    keyframe_times_B_N = None
            semantic_plan_B_L_D = semantic_plan_B_L_D[:, : self.max_tokens]

        param = self.type_token
        semantic_plan = torch.nan_to_num(semantic_plan_B_L_D.to(device=param.device, dtype=param.dtype))
        valid = semantic_plan.float().abs().sum(dim=-1) > 0
        tokens = self.projection(self.norm1(semantic_plan))
        if self.coord_projection is not None:
            coords = self.coords_for_tokens(
                batch_size=tokens.shape[0],
                num_tokens=tokens.shape[1],
                device=tokens.device,
                dtype=tokens.dtype,
                keyframe_times_B_N=keyframe_times_B_N,
            )
            tokens = tokens + self.coord_projection(coords)
        tokens = self.norm2(tokens)
        tokens = tokens + self.type_token.to(device=tokens.device, dtype=tokens.dtype)
        tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        return tokens, valid, keyframe_times_B_N
