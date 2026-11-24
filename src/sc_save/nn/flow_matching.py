"""Pure-PyTorch latent Flow Matching models, probability paths, and solvers."""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping, Optional

import numpy as np
import ot as pot
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from sc_save.nn.blocks import DiTBlock, TimestepEmbedder
from sc_save.nn.vae import remap_legacy_state_dict


@dataclass
class PathSample:
    """Container matching ``flow_matching.path.PathSample`` fields."""

    x_t: torch.Tensor
    dx_t: torch.Tensor
    x_1: torch.Tensor
    x_0: torch.Tensor
    t: torch.Tensor


def _expand_time_like(t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, *([1] * (target.ndim - 1)))


class CondOTPath:
    """Pure-PyTorch Conditional Optimal Transport path matching ``AffineProbPath(CondOTScheduler())``."""

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        t_exp = _expand_time_like(t, x_1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        dx_t = x_1 - x_0
        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)


class VPPath:
    """Pure-PyTorch Variance Preserving path matching ``AffineProbPath(VPScheduler())``."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        b = self.beta_min
        beta_max = self.beta_max
        one_minus_t = 1.0 - t
        T = 0.5 * (one_minus_t ** 2) * (beta_max - b) + one_minus_t * b
        dT = -one_minus_t * (beta_max - b) - b

        exp_half_T = torch.exp(-0.5 * T)
        exp_T = torch.exp(-T)
        sqrt_one_minus_exp_T = torch.sqrt(1.0 - exp_T)

        alpha_t = _expand_time_like(exp_half_T, x_1)
        sigma_t = _expand_time_like(sqrt_one_minus_exp_T, x_1)
        d_alpha_t = _expand_time_like(-0.5 * dT * exp_half_T, x_1)
        d_sigma_t = _expand_time_like(0.5 * dT * exp_T / sqrt_one_minus_exp_T, x_1)

        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1
        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)


class VEPath:
    """Pure-PyTorch Cosine path matching ``AffineProbPath(CosineScheduler())``."""

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        half_pi = 0.5 * math.pi
        angle = half_pi * t
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)

        alpha_t = _expand_time_like(sin_a, x_1)
        sigma_t = _expand_time_like(cos_a, x_1)
        d_alpha_t = _expand_time_like(half_pi * cos_a, x_1)
        d_sigma_t = _expand_time_like(-half_pi * sin_a, x_1)

        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1
        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)


class OTPlanSampler:
    """Minibatch optimal transport plan sampler wrt squared Euclidean cost."""

    def __init__(
        self,
        method: str = "exact",
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        warn: bool = True,
    ) -> None:
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown OT method: {method!r}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(
        self, pi: np.ndarray, batch_size: int, replace: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(
        self, x0: torch.Tensor, x1: torch.Tensor, replace: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_labels(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0: torch.Tensor | None = None,
        y1: torch.Tensor | None = None,
        replace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )


def zero_init(module: nn.Module) -> nn.Module:
    nn.init.constant_(module.weight.data, 0)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias.data, 0)
    return module


class ResnetBlock(nn.Module):
    """Residual MLP block for non-DiT denoiser backbones."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        dropout_prob: float = 0.0,
        embedding_dim: int | None = None,
        normalization: str = "batch",
        seq_length: int = 47,
        types_of_conds: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.out_dim = in_dim if out_dim is None else out_dim

        if normalization not in ["layer", "batch"]:
            self.net1 = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, self.out_dim))
        else:
            self.net1 = nn.Sequential(
                Rearrange("b s e -> b (s e)"),
                nn.Linear(in_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
                if normalization == "layer"
                else nn.BatchNorm1d(num_features=embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

        self.cond_proj = nn.Sequential(
            Rearrange("b c e -> b (c e)"),
            nn.Linear(embedding_dim * types_of_conds, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        )

        if normalization not in ["layer", "batch"]:
            self.net2 = nn.Sequential(
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(nn.Linear(self.out_dim, self.out_dim)),
            )
        else:
            self.net2 = nn.Sequential(
                nn.LayerNorm(embedding_dim)
                if normalization == "layer"
                else nn.BatchNorm1d(num_features=embedding_dim),
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(nn.Linear(embedding_dim, in_dim)),
                Rearrange("b (s e) -> b s e", s=seq_length),
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.net1(x)
        h = h + self.cond_proj(emb)
        h = self.net2(h)
        return x + h


class MLPTimeStep(nn.Module):
    """DiT-based conditional velocity denoiser over VAE latent sequences."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout_prob: float,
        n_blocks: int,
        seq_length: int,
        embedding_dim: int = 128,
        label_encoder_dict: Mapping[str, Any] | None = None,
        mask_id: int = 0,
        num_conds: int = 2,
        normalization: str = "layer",
        conditional: bool = True,
        is_binarized: bool = False,
        modality_list: Any = None,
        conditioning_probability: float = 0.8,
        guided_conditioning: bool = True,
        block_type: str = "DiT",
        num_heads: int = 4,
        total_cond_classes: int | None = None,
        total_classes: int | None = None,
    ) -> None:
        super().__init__()
        if total_cond_classes is None and total_classes is not None:
            total_cond_classes = total_classes
        if block_type == "DiT" and hidden_dim != embedding_dim:
            raise ValueError(
                f"For block_type='DiT', hidden_dim ({hidden_dim}) must equal embedding_dim ({embedding_dim})."
            )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.conditional = conditional
        self.is_binarized = is_binarized
        self.block_type = block_type
        self.modality_list = modality_list
        self.conditioning_probability = conditioning_probability
        self.guided_conditioning = guided_conditioning
        self.num_conds = num_conds
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, embedding_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.time_embedder = TimestepEmbedder(hidden_size=embedding_dim)
        self.net_in = nn.Linear(in_dim, self.hidden_dim)

        blocks: list[nn.Module] = []
        if block_type == "Resnet":
            for _ in range(n_blocks):
                blocks.append(
                    ResnetBlock(
                        in_dim=self.hidden_dim * seq_length,
                        dropout_prob=dropout_prob,
                        embedding_dim=embedding_dim,
                        normalization="batch",
                        seq_length=seq_length,
                        types_of_conds=num_conds,
                    )
                )
        elif block_type == "DiT":
            for _ in range(n_blocks):
                blocks.append(
                    DiTBlock(
                        feature_dim=embedding_dim,
                        mlp_ratio=4.0,
                        num_heads=num_heads,
                        types_of_conds=num_conds,
                        dropout=dropout_prob,
                        cond_norm="bn",
                    )
                )
        else:
            raise ValueError(f"Unsupported block_type={block_type!r}; expected 'DiT' or 'Resnet'.")

        self.mask_id = mask_id
        if total_cond_classes is not None:
            total_class = int(total_cond_classes)
        elif label_encoder_dict is not None:
            total_class = sum(len(le.classes_) for le in label_encoder_dict.values())
            total_class -= len(label_encoder_dict) - 1
        else:
            raise ValueError("Provide either `label_encoder_dict` or `total_cond_classes`.")
        self.cond_embedder = nn.Embedding(total_class, embedding_dim)

        self.blocks = nn.ModuleList(blocks)
        self.net_out = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, in_dim),
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        if self.block_type == "DiT":
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        target_keys = set(self.state_dict().keys())
        remapped = remap_legacy_state_dict(state_dict, target_keys=target_keys)
        return super().load_state_dict(remapped, strict=strict, assign=assign)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None,
        inference: bool = False,
        unconditional: bool = False,
        is_output_attn_weight: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t_flat = t.detach().reshape(-1)
        emb = self.time_embedder(t_flat * 1000.0).unsqueeze(1)

        if y is None:
            y = torch.zeros(
                (x.shape[0], self.num_conds, self.embedding_dim),
                device=x.device,
                dtype=emb.dtype,
            )
        elif not inference:
            random_mask = (
                torch.rand(y.shape[0], y.shape[1], device=y.device)
                < self.conditioning_probability
            )
            while random_mask.dim() < y.dim():
                random_mask = random_mask.unsqueeze(-1)
            y = torch.where(random_mask.expand_as(y), y, torch.zeros_like(y))
        elif unconditional:
            y = torch.zeros_like(y)

        emb = emb + y
        h = self.net_in(x) + self.pos_embedding

        if is_output_attn_weight and self.block_type == "DiT":
            attn_weight = None
            for i, block in enumerate(self.blocks):
                if i == len(self.blocks) - 1:
                    h, attn_weight = block(h, emb, is_output_attn_weight=True)
                else:
                    h = block(h, emb)
            return self.net_out(h), attn_weight

        for block in self.blocks:
            h = block(h, emb)
        return self.net_out(h)


DiTDenoiser = MLPTimeStep


class SimpleSDESolver:
    """Euler-Maruyama SDE solver for stochastic flow sampling."""

    def __init__(self, velocity_model: Callable[..., torch.Tensor], noise_scale: float = 1.0) -> None:
        self.velocity_model = velocity_model
        self.noise_scale = float(noise_scale)

    def sample(
        self,
        time_grid: torch.Tensor,
        x_init: torch.Tensor,
        method: str = "euler",
        step_size: float | None = None,
        return_intermediates: bool = False,
        y: torch.Tensor | None = None,
        inference: bool = True,
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor]:
        if method != "euler":
            warnings.warn(
                f"SimpleSDESolver only supports 'euler', got {method!r}; using 'euler'."
            )

        x = x_init.clone()
        intermediates = [x.clone()] if return_intermediates else None

        for i in range(len(time_grid) - 1):
            t_cur = time_grid[i]
            dt = time_grid[i + 1] - t_cur
            if t_cur.ndim == 0:
                t_batch = t_cur.unsqueeze(0).expand(x.shape[0])
            else:
                t_batch = t_cur.expand(x.shape[0])

            v = self.velocity_model(x, t_batch, y=y, inference=inference)
            diffusion = self.noise_scale * torch.sqrt(dt.abs()) * torch.randn_like(x)
            x = x + v * dt + diffusion

            if intermediates is not None:
                intermediates.append(x.clone())

        return intermediates if intermediates is not None else x


def _solve_fixed_ode(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_init: torch.Tensor,
    time_grid: torch.Tensor,
    method: str,
) -> torch.Tensor:
    x = x_init
    if method == "euler":
        for i in range(len(time_grid) - 1):
            t0 = time_grid[i]
            dt = time_grid[i + 1] - t0
            x = x + dt * velocity_fn(x, t0)
        return x

    if method == "midpoint":
        for i in range(len(time_grid) - 1):
            t0 = time_grid[i]
            dt = time_grid[i + 1] - t0
            half_dt = 0.5 * dt
            k1 = velocity_fn(x, t0)
            k2 = velocity_fn(x + half_dt * k1, t0 + half_dt)
            x = x + dt * k2
        return x

    if method == "rk4":
        for i in range(len(time_grid) - 1):
            t0 = time_grid[i]
            t1 = time_grid[i + 1]
            dt = t1 - t0
            half_dt = 0.5 * dt
            t_mid = t0 + half_dt
            k1 = velocity_fn(x, t0)
            k2 = velocity_fn(x + half_dt * k1, t_mid)
            k3 = velocity_fn(x + half_dt * k2, t_mid)
            k4 = velocity_fn(x + dt * k3, t1)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    raise ValueError(f"Unsupported native ODE method: {method!r}")


class FM(nn.Module):
    """Latent Flow Matching trainer and fused-CFG sampler."""

    def __init__(
        self,
        denoising_model: MLPTimeStep,
        antithetic_time_sampling: bool = False,
        sigma: float = 0.1,
        device: torch.device | str = torch.device("cpu"),
        use_ot: bool = True,
        is_binarized: bool = False,
        modality_list: Any = None,
        guidance_weights: float | None = None,
        path_type: str = "ot",
        path_kwargs: Optional[dict[str, Any]] = None,
        use_snr_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.denoising_model = denoising_model.to(self.device)
        self.antithetic_time_sampling = antithetic_time_sampling
        self.sigma = sigma
        self.use_ot = use_ot
        self.is_binarized = is_binarized
        self.modality_list = modality_list
        self.guidance_weights = guidance_weights
        self.path_type = path_type
        self.use_snr_weighting = use_snr_weighting

        kwargs = dict(path_kwargs or {})
        if path_type == "ot":
            self.path: CondOTPath | VPPath | VEPath = CondOTPath()
        elif path_type == "vp":
            self.path = VPPath(**kwargs)
        elif path_type == "ve":
            self.path = VEPath()
        else:
            raise ValueError(
                f"Unknown path_type: {path_type!r}. Supported types: 'ot', 'vp', 've'."
            )

        self.criterion = torch.nn.MSELoss(reduction="mean")
        if self.use_ot:
            self.ot_sampler = OTPlanSampler(method="exact")

    def _step(
        self,
        x_1: torch.Tensor,
        y_fea: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        if debug and (torch.isnan(x_1).any() or torch.isinf(x_1).any()):
            x_1 = torch.nan_to_num(x_1, nan=0.0, posinf=1.0, neginf=-1.0)

        t = self._sample_times(x_1.shape[0])
        x_0 = torch.randn_like(x_1)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        x_t = path_sample.x_t
        u_t = path_sample.dx_t
        if debug and (torch.isnan(x_t).any() or torch.isinf(x_t).any()):
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)

        v_t = self.denoising_model(x_t, path_sample.t, y_fea)

        if debug:
            if torch.isnan(v_t).any() or torch.isinf(v_t).any():
                v_t = torch.nan_to_num(v_t, nan=0.0, posinf=1.0, neginf=-1.0)
            if torch.isnan(u_t).any() or torch.isinf(u_t).any():
                u_t = torch.nan_to_num(u_t, nan=0.0, posinf=1.0, neginf=-1.0)

        loss_weights = self._compute_loss_weights(
            path_sample.t, use_snr_weighting=self.use_snr_weighting
        )

        v_t_flat = rearrange(v_t, "b s d -> (b s) d")
        u_t_flat = rearrange(u_t, "b s d -> (b s) d")

        if loss_weights is None:
            loss = nn.functional.mse_loss(v_t_flat, u_t_flat)
        else:
            if debug and (torch.isnan(loss_weights).any() or torch.isinf(loss_weights).any()):
                loss_weights = torch.nan_to_num(loss_weights, nan=1.0, posinf=10.0, neginf=0.1)
            mse_loss = nn.functional.mse_loss(v_t_flat, u_t_flat, reduction="none").mean(dim=-1)
            weights_expanded = loss_weights.repeat_interleave(x_1.shape[1])
            loss = (mse_loss * weights_expanded).mean()

        if debug and (torch.isnan(loss) or torch.isinf(loss)):
            loss = torch.tensor(1e-3, device=loss.device, requires_grad=True)

        return loss

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        use_snr_weighting: bool = False,
    ) -> torch.Tensor | None:
        if self.path_type == "vp":
            if use_snr_weighting:
                epsilon = 1e-3
                snr = ((1.0 - t + epsilon) ** 2) / (t ** 2 + epsilon)
                weights = snr / (snr + 1.0)
                return weights / weights.mean()
            epsilon = 0.01
            weights = torch.clamp(1.0 / (1.0 - t + epsilon), min=0.1, max=10.0)
            return weights / weights.mean()
        if self.path_type == "ve":
            epsilon = 1e-3
            weights = torch.clamp(1.0 / (t + epsilon), min=0.1, max=10.0)
            return weights / weights.mean()
        return None

    def _sample_times(self, batch_size: int) -> torch.Tensor:
        if self.antithetic_time_sampling:
            u0 = torch.rand((), device=self.device, dtype=torch.float32)
            return (torch.arange(batch_size, device=self.device, dtype=torch.float32) + u0) / batch_size
        return torch.rand(batch_size, device=self.device)

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        seq_length: int,
        n_sample_steps: int,
        y: torch.Tensor | None = None,
        msk_emb: torch.Tensor | None = None,
        unconditional: bool = False,
        guidance_weights: float | None = None,
        solver_type: str = "euler",
        use_sde: bool | None = None,
        x_init: torch.Tensor | None = None,
        use_amp: bool = True,
    ) -> torch.Tensor:
        self.denoising_model.eval()

        if guidance_weights is None:
            guidance_weights = 1.0 if self.guidance_weights is None else self.guidance_weights
        w = float(guidance_weights)

        if x_init is None:
            x_init = torch.randn(
                (batch_size, seq_length, self.denoising_model.in_dim),
                device=self.device,
            )
        else:
            x_init = x_init.to(self.device)

        if unconditional or y is None:
            y_cond = None
        else:
            y_cond = y.to(self.device)

        if use_sde is None:
            use_sde = self.path_type == "ve"

        use_bf16 = (
            use_amp
            and self.device.type == "cuda"
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )

        with amp_ctx:
            if y_cond is None:
                def velocity_fn(x_cur: torch.Tensor, t_cur: torch.Tensor, **extras) -> torch.Tensor:
                    return self.denoising_model(
                        x_cur, t_cur, None, inference=True, unconditional=True
                    )
            else:
                y_cat = torch.cat([torch.zeros_like(y_cond), y_cond], dim=0)

                def velocity_fn(x_cur: torch.Tensor, t_cur: torch.Tensor, **extras) -> torch.Tensor:
                    # Fused CFG: run uncond + cond branches together in a single 2*B forward pass.
                    x_cat = torch.cat([x_cur, x_cur], dim=0)
                    t_in = (
                        torch.cat([t_cur, t_cur], dim=0)
                        if t_cur.ndim > 0 and t_cur.shape[0] == x_cur.shape[0]
                        else t_cur
                    )
                    v_uncond, v_cond = self.denoising_model(
                        x_cat, t_in, y_cat, inference=True, unconditional=False
                    ).chunk(2, dim=0)
                    return (1.0 - w) * v_uncond + w * v_cond

            time_grid = torch.linspace(
                0.0, 1.0, max(1, int(n_sample_steps)) + 1, device=self.device, dtype=torch.float32
            )

            if use_sde:
                sde_solver = SimpleSDESolver(velocity_model=velocity_fn, noise_scale=1.0)
                sol = sde_solver.sample(
                    time_grid=time_grid,
                    x_init=x_init,
                    method=solver_type,
                    return_intermediates=False,
                    y=y_cond,
                    inference=True,
                )
            elif solver_type in {"euler", "midpoint", "rk4"}:
                sol = _solve_fixed_ode(velocity_fn, x_init, time_grid, method=solver_type)
            else:
                sol = self._solve_optional_adaptive_ode(
                    velocity_fn=velocity_fn,
                    x_init=x_init,
                    n_sample_steps=max(1, int(n_sample_steps)),
                    solver_type=solver_type,
                )

        return sol.float()

    def _solve_optional_adaptive_ode(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        n_sample_steps: int,
        solver_type: str,
    ) -> torch.Tensor:
        try:
            from flow_matching.solver import ODESolver
            from flow_matching.utils import ModelWrapper
        except ImportError as exc:
            raise ImportError(
                f"Adaptive ODE solver {solver_type!r} requires optional `flow-matching` / `torchdiffeq`. "
                "Install with `pip install 'sc-save[solvers]'` or use 'euler', 'midpoint', or 'rk4'."
            ) from exc

        class _WrappedVF(ModelWrapper):
            def __init__(self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
                super().__init__(None)
                self._fn = fn

            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                return self._fn(x, t)

        solver = ODESolver(velocity_model=_WrappedVF(velocity_fn))
        time_grid = torch.tensor([0.0, 1.0], device=self.device, dtype=torch.float32)
        return solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method=solver_type,
            step_size=1.0 / n_sample_steps if solver_type not in {"dopri5", "dopri8", "bosh3", "adaptive_heun"} else None,
            return_intermediates=False,
        )

    @torch.inference_mode()
    def sample_with_attn_weights(
        self,
        batch_size: int,
        seq_length: int,
        n_sample_steps: int,
        y: torch.Tensor | None = None,
        guidance_weights: float | None = None,
    ) -> dict[str, Any]:
        self.denoising_model.eval()
        if guidance_weights is None:
            guidance_weights = 1.0 if self.guidance_weights is None else self.guidance_weights
        w = float(guidance_weights)

        x = torch.randn(
            (batch_size, seq_length, self.denoising_model.in_dim),
            device=self.device,
        )
        dt = 1.0 / max(1, int(n_sample_steps))
        all_attn_weights: list[dict[str, Any]] = []

        if y is not None:
            y_cat = torch.cat([torch.zeros_like(y), y], dim=0)
        else:
            y_cat = None

        for step in range(n_sample_steps):
            t_val = step * dt
            t = torch.tensor([t_val], device=self.device, dtype=torch.float32)

            if y_cat is not None:
                x_cat = torch.cat([x, x], dim=0)
                v_cat, attn_cat = self.denoising_model(
                    x_cat, t, y_cat, inference=True, unconditional=False, is_output_attn_weight=True
                )
                v_uncond, v_cond = v_cat.chunk(2, dim=0)
                attn_uncond, attn_cond = attn_cat.chunk(2, dim=0)
            else:
                v_uncond, attn_uncond = self.denoising_model(
                    x, t, None, inference=True, unconditional=True, is_output_attn_weight=True
                )
                v_cond, attn_cond = v_uncond, attn_uncond

            v = (1.0 - w) * v_uncond + w * v_cond
            x = x + v * dt

            all_attn_weights.append(
                {
                    "step": step,
                    "time": t_val,
                    "conditional": attn_cond.detach().cpu(),
                    "unconditional": attn_uncond.detach().cpu(),
                }
            )

        return {
            "final_sample": x,
            "attn_weights": all_attn_weights,
        }


FlowMatcher = FM


__all__ = [
    "CondOTPath",
    "DiTDenoiser",
    "FM",
    "FlowMatcher",
    "MLPTimeStep",
    "OTPlanSampler",
    "PathSample",
    "ResnetBlock",
    "SimpleSDESolver",
    "VEPath",
    "VPPath",
    "zero_init",
]
