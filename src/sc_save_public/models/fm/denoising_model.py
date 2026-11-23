"""Compatibility shim re-exporting denoising models from ``sc_save.nn.flow_matching``."""

import numpy as np
import torch

from sc_save.nn.flow_matching import DiTDenoiser, MLPTimeStep, ResnetBlock, zero_init


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    dtype: torch.dtype = torch.float32,
    max_timescale: float = 10_000,
    min_timescale: float = 1,
) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    timesteps = timesteps * 1000.0
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=1)


__all__ = [
    "DiTDenoiser",
    "MLPTimeStep",
    "ResnetBlock",
    "get_timestep_embedding",
    "zero_init",
]
