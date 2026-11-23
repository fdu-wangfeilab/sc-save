"""Transformer and AdaLN conditioning building blocks for SAVE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN shift and scale modulation to sequence tensor ``(B, S, D)``."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """Two-layer feed-forward network used inside DiT blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] | callable = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention using SDPA when attention weights are not requested."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_output_attn_weight: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        is_output_attn_weight: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if not is_output_attn_weight and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).reshape(bsz, seq_len, dim)
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)

        out = self.proj(out)
        out = self.proj_drop(out)

        if is_output_attn_weight:
            return out, attn
        return out


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN-Zero condition modulation."""

    def __init__(
        self,
        feature_dim: int = 2000,
        mlp_ratio: float = 4.0,
        types_of_conds: int = 2,
        num_heads: int = 10,
        dropout: float = 0.0,
        cond_norm: str = "ln",
        **kwargs,
    ) -> None:
        super().__init__()
        self.types_of_conds = types_of_conds

        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            feature_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **kwargs,
        )
        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=feature_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )

        if cond_norm == "ln":
            self.adaLN_modulation = nn.Sequential(
                Rearrange("b c e -> b (c e)"),
                nn.Linear(feature_dim * self.types_of_conds, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, 6 * feature_dim, bias=True),
            )
        elif cond_norm == "bn":
            self.adaLN_modulation = nn.Sequential(
                Rearrange("b c e -> b (c e)"),
                nn.Linear(feature_dim * self.types_of_conds, feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, 6 * feature_dim, bias=True),
            )
        else:
            raise ValueError(f"Unsupported cond_norm={cond_norm!r}; expected 'ln' or 'bn'.")

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        is_output_attn_weight: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        msa_in = modulate(self.norm1(x), shift_msa, scale_msa)
        if is_output_attn_weight:
            attn_out, attn_weight = self.attn(msa_in, is_output_attn_weight=True)
        else:
            attn_out = self.attn(msa_in)

        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        if is_output_attn_weight:
            return x, attn_weight
        return x


DiTblock = DiTBlock


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar timestep embedder with non-persistent device frequency buffer."""

    def __init__(
        self,
        hidden_size: int = 512,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / max(1, half)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            t_freq = torch.cat(
                [t_freq, torch.zeros_like(t_freq[:, :1])], dim=-1
            )
        return self.mlp(t_freq)


__all__ = [
    "Attention",
    "DiTBlock",
    "DiTblock",
    "Mlp",
    "TimestepEmbedder",
    "modulate",
]
