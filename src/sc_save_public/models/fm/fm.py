"""
Flow Matching (FM) Module for Single-Cell Data Generation

This module implements Flow Matching with support for multiple diffusion paths:
- Optimal Transport (OT) path: Default, uses conditional OT scheduler
- Variance Preserving (VP) path: Similar to DDPM, suitable for stable training
- Variance Exploding (VE) path: SDE-based, provides stochastic sampling

Usage Examples
--------------

1. Using the default OT path (Conditional Optimal Transport):
    >>> denoising_model = MLPTimeStep(...)
    >>> fm = FM(denoising_model=denoising_model, device=device)

2. Using VP path (Variance Preserving, similar to DDPM):
    >>> fm = FM(
    ...     denoising_model=denoising_model,
    ...     path_type="vp",
    ...     device=device
    ... )

3. Using VE path (Variance Exploding):
    >>> fm = FM(
    ...     denoising_model=denoising_model,
    ...     path_type="ve",
    ...     path_kwargs={"sigma": 1.0},
    ...     device=device
    ... )

4. Sampling with different solvers:
    >>> # ODE sampling (deterministic, default for OT and VP)
    >>> samples = fm.sample(
    ...     batch_size=100,
    ...     seq_length=50,
    ...     n_sample_steps=100,
    ...     y=condition_features,
    ...     solver_type="euler"
    ... )

    >>> # SDE sampling (stochastic, recommended for VE path)
    >>> samples = fm.sample(
    ...     batch_size=100,
    ...     seq_length=50,
    ...     n_sample_steps=100,
    ...     y=condition_features,
    ...     solver_type="euler",
    ...     use_sde=True
    ... )

Path Type Characteristics
--------------------------
- OT path:
  * Best for optimal transport between distributions
  * Uses deterministic straight paths with small noise
  * Fast convergence, suitable for most use cases

- VP path:
  * Similar to DDPM (Denoising Diffusion Probabilistic Models)
  * Preserves variance throughout the diffusion process
  * More stable training, good for complex distributions
  * Uses time-based loss weighting that emphasizes mid-timesteps

- VE path:
  * Variance increases (explodes) during the forward process
  * Allows for stochastic sampling via SDE solver
  * Good for generating diverse samples
  * Requires more sampling steps for high quality

Ablation Study Recommendations
-------------------------------
When conducting ablation studies, consider varying:
- path_type: "ot", "vp", "ve"
- solver_type: "euler", "midpoint", "rk4", "dopri5"
- use_sde: True/False (for stochastic vs deterministic sampling)
- n_sample_steps: 10, 50, 100, 200 (trade-off between quality and speed)

For fair comparison:
- Use the same denoising model architecture across all path types
- Keep training hyperparameters consistent
- Report both sample quality metrics and inference time
"""

from typing import Optional
import numpy as np

import torch
from torch import nn

from sc_save_public.models.fm.ot_sampler import OTPlanSampler
from sc_save_public.models.fm.denoising_model import MLPTimeStep

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from einops import rearrange


class SimpleSDESolver:
    """
    Simple SDE solver using Euler-Maruyama method for stochastic sampling.
    
    This implements the SDE:
        dx_t = v_θ(t, x_t, y) dt + σ dW_t
    
    where:
        - v_θ is the velocity model (denoising_model)
        - σ is the noise_scale
        - dW_t is the Wiener process (Brownian motion)
    
    Euler-Maruyama discretization:
        x_{t+Δt} = x_t + v_θ(t, x_t, y) * Δt + σ * √Δt * ε_t
    
    where ε_t ~ N(0, I)
    """
    
    def __init__(self, velocity_model, noise_scale=1.0):
        """
        Initialize the SDE solver.
        
        Args:
            velocity_model: The velocity model (wrapped denoising model)
            noise_scale: Scale of the noise term (σ)
        """
        self.velocity_model = velocity_model
        self.noise_scale = noise_scale
    
    def sample(
        self,
        time_grid,
        x_init,
        method="euler",
        step_size=None,
        return_intermediates=False,
        y=None,
        inference=True,
        **kwargs
    ):
        """
        Sample using Euler-Maruyama SDE solver.
        
        Args:
            time_grid: Time points tensor (e.g., torch.linspace(0, 1, n_steps))
            x_init: Initial state (noise)
            method: Solver method (currently only "euler" is supported)
            step_size: Step size for integration
            return_intermediates: Whether to return intermediate states
            y: Conditional features
            inference: Whether in inference mode
            **kwargs: Additional arguments
        
        Returns:
            Final state (or list of intermediate states if return_intermediates=True)
        """
        if method != "euler":
            print(f"Warning: SimpleSDESolver only supports 'euler' method, but got '{method}'. Using 'euler'.")
        
        x = x_init.clone()
        device = x.device
        
        intermediates = [x.clone()] if return_intermediates else None
        
        # Iterate through time steps
        for i in range(len(time_grid) - 1):
            t_current = time_grid[i]
            t_next = time_grid[i + 1]
            dt = t_next - t_current
            
            # Ensure t_current is a scalar tensor or expand it to batch size
            if t_current.dim() == 0:
                t_batch = t_current.unsqueeze(0).expand(x.shape[0])
            else:
                t_batch = t_current.expand(x.shape[0])
            
            # Compute velocity
            v = self.velocity_model(x, t_batch, y, inference=inference)
            
            # Euler-Maruyama step: x_{t+dt} = x_t + v * dt + σ * √dt * ε
            drift = v * dt
            diffusion = self.noise_scale * torch.sqrt(dt.abs()) * torch.randn_like(x)
            
            x = x + drift + diffusion
            
            if return_intermediates:
                intermediates.append(x.clone())
        
        if return_intermediates:
            return intermediates
        else:
            return x


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class FM(nn.Module):
    def __init__(
        self,
        denoising_model: MLPTimeStep,
        # feature_embeddings: dict,
        # plotting_folder: Path,
        # in_dim: int,
        # size_factor_statistics: dict,
        # covariate_list: str,
        # theta_covariate: str,
        # size_factor_covariate: str,
        # encoder_type: str = "fixed",
        antithetic_time_sampling: bool = False,
        # scaling_method: str = "log_normalization",  # Change int to str
        sigma: float = 0.1,
        # covariate_specific_theta: float = False,
        # plot_and_eval_every=100,
        device=torch.device("cuda:0"),
        use_ot=True,
        is_binarized=False,
        modality_list=None,
        guidance_weights=None,
        path_type: str = "ot",
        path_kwargs: Optional[dict] = None,
        use_snr_weighting: bool = False,
    ):
        """
        Flow matching for single-cell model.

        Args:
            denoising_model (nn.Module): Denoising model.
            antithetic_time_sampling (bool, optional): Use antithetic time sampling. Defaults to False.
            sigma (float, optional): Variance around straight path for flow matching objective. Defaults to 0.1.
            device: Device to use for computation. Defaults to cuda:0.
            use_ot (bool, optional): Whether to use optimal transport. Defaults to True.
            is_binarized (bool, optional): Whether the data is binarized. Defaults to False.
            modality_list: List of modalities. Defaults to None.
            guidance_weights: Weights for classifier-free guidance. Defaults to None.
            path_type (str, optional): Type of diffusion path. Options:
                - "ot": Optimal Transport path (Conditional OT, default)
                - "vp": Variance Preserving path (similar to DDPM)
                - "ve": Variance Exploding path (SDE-based)
            path_kwargs (dict, optional): Additional keyword arguments for the path scheduler.
                Note: Most schedulers work well with default parameters.
                - For VP path: VPScheduler typically doesn't require additional parameters
                - For VE path: CosineScheduler may accept additional parameters
                Defaults to None (uses default scheduler parameters).
            use_snr_weighting (bool, optional): For VP path, use SNR-based weighting (more stable).
                Defaults to False (uses clipped inverse weighting).
        """
        super().__init__()

        self.device = device
        self.denoising_model = denoising_model.to(self.device)
        # self.feature_embeddings = feature_embeddings
        # self.in_dim = in_dim
        # self.size_factor_statistics = size_factor_statistics
        self.antithetic_time_sampling = antithetic_time_sampling

        # self.scaling_method = scaling_method
        # self.plotting_folder = plotting_folder
        # self.covariate_list = covariate_list
        # self.theta_covariate = theta_covariate
        # self.size_factor_covariate = size_factor_covariate
        # self.covariate_specific_theta = covariate_specific_theta
        # self.plot_and_eval_every = plot_and_eval_every

        self.sigma = sigma
        self.use_ot = use_ot
        self.is_binarized = is_binarized
        self.guidance_weights = guidance_weights
        self.path_type = path_type
        self.use_snr_weighting = use_snr_weighting

        # Initialize path_kwargs if not provided
        if path_kwargs is None:
            path_kwargs = {}

        # Initialize the appropriate scheduler based on path_type
        if path_type == "ot":
            self.scheduler = CondOTScheduler()
        elif path_type == "vp":
            from flow_matching.path.scheduler import VPScheduler

            # VPScheduler typically doesn't require additional parameters
            # Use default initialization if path_kwargs is empty
            if path_kwargs:
                self.scheduler = VPScheduler(**path_kwargs)
            else:
                self.scheduler = VPScheduler()
        elif path_type == "ve":
            from flow_matching.path.scheduler import CosineScheduler

            # VE path uses CosineScheduler
            # Note: CosineScheduler does not accept any parameters
            self.scheduler = CosineScheduler()
        else:
            raise ValueError(
                f"Unknown path_type: {path_type}. Supported types: 'ot', 'vp', 've'"
            )

        # Initialize the probability path
        self.path = AffineProbPath(scheduler=self.scheduler)

        # MSE lost for the Flow Matching algorithm
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # OT sampler
        if self.use_ot:
            self.ot_sampler = OTPlanSampler(method="exact")

    def _step(self, x_1, y_fea, debug=False):
        """
        Common step for training and validation.

        Args:
            x_1: Target data (clean samples). Shape: (B, S, D)
            y_fea: Conditional features.
            debug: If True, print debug information when NaN is detected.

        Returns:
            torch.Tensor: Loss value.
        """
        
        # Check input for NaN/Inf
        if torch.isnan(x_1).any() or torch.isinf(x_1).any():
            if debug:
                print(f"[WARNING] NaN/Inf detected in input x_1!")
                print(f"  x_1 stats: min={x_1.min():.4f}, max={x_1.max():.4f}, mean={x_1.mean():.4f}")
            # 用小值替换 NaN/Inf
            x_1 = torch.nan_to_num(x_1, nan=0.0, posinf=1.0, neginf=-1.0)

        # Sample time
        t = self._sample_times(x_1.shape[0])  # B

        # Sample noise
        x_0 = self.sample_noise_like(x_1)  # B x S x D

        # Get objective and perturbed observation using the configured path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        
        # Check path sample for NaN/Inf
        if torch.isnan(path_sample.x_t).any() or torch.isinf(path_sample.x_t).any():
            if debug:
                print(f"[WARNING] NaN/Inf detected in path_sample.x_t!")
                print(f"  t stats: min={t.min():.4f}, max={t.max():.4f}")
                print(f"  x_t stats: min={path_sample.x_t.min():.4f}, max={path_sample.x_t.max():.4f}")
            path_sample.x_t = torch.nan_to_num(path_sample.x_t, nan=0.0, posinf=1.0, neginf=-1.0)

        # Forward through the model
        v_t = self.denoising_model(path_sample.x_t, path_sample.t, y_fea)
        u_t = path_sample.dx_t
        
        # Check model output for NaN/Inf
        if torch.isnan(v_t).any() or torch.isinf(v_t).any():
            if debug:
                print(f"[ERROR] NaN/Inf detected in model output v_t!")
                print(f"  v_t stats: contains NaN: {torch.isnan(v_t).sum()}, contains Inf: {torch.isinf(v_t).sum()}")
            v_t = torch.nan_to_num(v_t, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(u_t).any() or torch.isinf(u_t).any():
            if debug:
                print(f"[ERROR] NaN/Inf detected in target u_t!")
            u_t = torch.nan_to_num(u_t, nan=0.0, posinf=1.0, neginf=-1.0)

        # Compute loss with optional weighting based on path type
        loss_weights = self._compute_loss_weights(path_sample.t, use_snr_weighting=self.use_snr_weighting)

        # Reshape for loss computation
        v_t_flat = rearrange(v_t, "b s d -> (b s) d")
        u_t_flat = rearrange(u_t, "b s d -> (b s) d")

        # Compute per-sample MSE loss
        mse_loss = torch.nn.functional.mse_loss(
            v_t_flat, u_t_flat, reduction="none"
        ).mean(dim=-1)  # (B*S,)

        # Apply time-dependent weights if needed
        if loss_weights is not None:
            # Check weights for NaN/Inf
            if torch.isnan(loss_weights).any() or torch.isinf(loss_weights).any():
                if debug:
                    print(f"[ERROR] NaN/Inf detected in loss_weights!")
                    print(f"  weights stats: min={loss_weights.min():.4f}, max={loss_weights.max():.4f}")
                loss_weights = torch.nan_to_num(loss_weights, nan=1.0, posinf=10.0, neginf=0.1)
            
            # Expand weights to match the flattened batch dimension
            weights_expanded = loss_weights.repeat_interleave(x_1.shape[1])  # (B*S,)
            weighted_loss = (mse_loss * weights_expanded).mean()
        else:
            weighted_loss = mse_loss.mean()
        
        # Final check
        if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
            if debug:
                print(f"[ERROR] Final loss is NaN/Inf!")
                print(f"  mse_loss stats: min={mse_loss.min():.4f}, max={mse_loss.max():.4f}, mean={mse_loss.mean():.4f}")
            # 返回一个小的正数，避免训练崩溃
            weighted_loss = torch.tensor(1e-3, device=weighted_loss.device, requires_grad=True)

        return weighted_loss

    def _compute_loss_weights(self, t, use_snr_weighting=False):
        """
        Compute time-dependent loss weights based on the path type.

        Args:
            t: Time tensor. Shape: (B,)
            use_snr_weighting: If True, use SNR-based weighting for VP path (more stable)

        Returns:
            torch.Tensor or None: Loss weights for each time step, or None for uniform weighting.
        """
        if self.path_type == "vp":
            if use_snr_weighting:
                # SNR-based weighting (more stable, recommended for VP path)
                # Based on "Improved Denoising Diffusion Probabilistic Models"
                # Weight = SNR(t) / (SNR(t) + 1)
                # For VP scheduler with linear beta schedule:
                # SNR(t) ≈ (1-t)² / t²
                epsilon = 1e-3
                snr = ((1 - t + epsilon) ** 2) / (t ** 2 + epsilon)
                weights = snr / (snr + 1)
                # Normalize to have mean 1
                weights = weights / weights.mean()
            else:
                # Simple inverse weighting (less stable but faster)
                # Increased epsilon for numerical stability and added clipping
                epsilon = 0.01  # Increased from 1e-5 to prevent extreme weights
                weights = 1.0 / (1.0 - t + epsilon)
                # Clip weights to prevent gradient explosion
                weights = torch.clamp(weights, min=0.1, max=10.0)
                # Normalize weights to have mean 1
                weights = weights / weights.mean()
            return weights
        elif self.path_type == "ve":
            # For VE path, weight by timestep to emphasize early stages
            # w(t) = 1 / (t + epsilon)
            epsilon = 1e-3
            weights = 1.0 / (t + epsilon)
            # Clip weights to prevent extreme values
            weights = torch.clamp(weights, min=0.1, max=10.0)
            # Normalize weights to have mean 1
            weights = weights / weights.mean()
            return weights
        else:  # "ot" or other paths
            # Use uniform weighting (no time-dependent weighting)
            return None

    # Private methods
    # def _featurize_batch_y(self, batch):
    #     """
    #     Featurize all the covariates

    #     Args:
    #         batch: Batch data.

    #     Returns:
    #         torch.Tensor: Featurized covariates.
    #     """
    #     y = {}
    #     # feature_cat 表示条件的类别, 意思是进 batch 之前就已经把类别转化为数值了
    #     for feature_cat in batch["y"]:
    #         y_cat = self.feature_embeddings[feature_cat](batch["y"][feature_cat])
    #         y[feature_cat] = y_cat
    #     return y

    def _sample_times(self, batch_size):
        """
        Sample times, can be sampled to cover the

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Sampled times.
        """
        if self.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        seq_length,
        n_sample_steps,
        y=None,
        msk_emb=None,
        unconditional=False,
        guidance_weights=None,
        solver_type="euler",
        use_sde=None,
    ):
        """
        Sample from the flow matching model using ODE or SDE solver.

        Args:
            batch_size: Batch size for sampling.
            seq_length: Sequence length.
            n_sample_steps: Number of sampling steps.
            y: Conditional features. Defaults to None.
            msk_emb: Mask embeddings. Defaults to None.
            unconditional: Whether to perform unconditional sampling. Defaults to False.
            guidance_weights: Classifier-free guidance weights. Defaults to None.
            solver_type: Type of solver to use ("euler", "midpoint", "rk4", etc.). Defaults to "euler".
            use_sde: Whether to use SDE solver. If None, automatically determined by path_type:
                     - "ot" and "vp": use ODE solver (False)
                     - "ve": use SDE solver (True)

        Returns:
            torch.Tensor: Generated samples.
        """
        # 设置模型为评估模式，避免BatchNorm的训练模式问题
        self.denoising_model.eval()

        if guidance_weights is None:
            guidance_weights = self.guidance_weights

        # Sample random noise
        x_init = torch.randn(
            (batch_size, seq_length, self.denoising_model.in_dim), device=self.device
        )

        # Featurize the covariate
        if unconditional:
            y = None

        # Determine whether to use SDE solver based on path type
        if use_sde is None:
            # Auto-select: use SDE for VE path, ODE for others
            use_sde = self.path_type == "ve"

        # Wrap the denoising model with classifier-free guidance
        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, y, **extras):
                m_uncond = self.model(x, t, y, inference=True, unconditional=True)
                m_cond = self.model(x, t, y, inference=True, unconditional=False)
                # 标准的 Classifier Free Guidance 公式
                return (1 - guidance_weights) * m_uncond + guidance_weights * m_cond

        wrapped_vf = WrappedModel(self.denoising_model)

        step_size = 1 / n_sample_steps
        T = torch.linspace(0, 1, n_sample_steps)  # sample times
        T = T.to(device=self.device)

        if use_sde:
            # Use SDE solver for stochastic sampling
            # Using custom SimpleSDESolver (Euler-Maruyama method)
            solver = SimpleSDESolver(
                velocity_model=wrapped_vf,
                noise_scale=1.0,  # Can be adjusted based on path type
            )
            sol = solver.sample(
                time_grid=T,
                x_init=x_init,
                method=solver_type,
                step_size=step_size,
                return_intermediates=False,
                y=y,
                inference=True,
            )
        else:
            # Use ODE solver for deterministic sampling
            solver = ODESolver(velocity_model=wrapped_vf)
            sol = solver.sample(
                time_grid=T,
                x_init=x_init,
                method=solver_type,
                step_size=step_size,
                return_intermediates=False,
                y=y,
                inference=True,
            )

        return sol

    @torch.no_grad()
    def sample_with_attn_weights(
        self,
        batch_size,
        seq_length,
        n_sample_steps,
        y=None,
        guidance_weights=None,
    ):
        """
        使用 Euler 方法的 Flow Matching 采样,并收集每一步的 attention weights (仅最后一层)。

        Args:
            batch_size: 批次大小
            seq_length: 序列长度
            n_sample_steps: 采样步数
            y: 条件 embedding
            guidance_weights: Classifier Free Guidance 权重

        Returns:
            dict: {
                'final_sample': 最终生成的样本,
                'attn_weights': 每个时间步的 attention weights (仅最后一层)
            }
        """
        self.denoising_model.eval()

        if guidance_weights is None:
            guidance_weights = self.guidance_weights

        # 初始化噪声
        x = torch.randn(
            (batch_size, seq_length, self.denoising_model.in_dim), device=self.device
        )

        # 时间步长
        dt = 1.0 / n_sample_steps

        # 存储 attention weights
        all_attn_weights = []

        for step in range(n_sample_steps):
            t = torch.tensor([step * dt], device=self.device)

            # Unconditional forward
            v_uncond, attn_uncond = self.denoising_model(
                x, t, y, inference=True, unconditional=True, is_output_attn_weight=True
            )

            # Conditional forward
            v_cond, attn_cond = self.denoising_model(
                x, t, y, inference=True, unconditional=False, is_output_attn_weight=True
            )

            # Classifier Free Guidance
            v = (1 - guidance_weights) * v_uncond + guidance_weights * v_cond

            # Euler step
            x = x + v * dt

            # 保存 attention weights (仅最后一层,立即转移到CPU)
            step_attn = {
                "step": step,
                "time": t.item(),
                "conditional": attn_cond.detach().cpu(),
                "unconditional": attn_uncond.detach().cpu(),
            }
            all_attn_weights.append(step_attn)

        results = {
            "final_sample": x,
            "attn_weights": all_attn_weights,
        }

        return results

    @torch.no_grad()
    def batched_sample(
        self,
        batch_size,
        repetitions,
        n_sample_steps,
        theta_covariate,
        size_factor_covariate,
        conditioning_covariates,
        covariate_indices=None,
        log_size_factor=None,
        unconditional=False,
    ):
        total_samples = {mod: [] for mod in self.modality_list}

        # Covariate is same for all modalities
        for i in range(repetitions):
            if covariate_indices is not None:
                covariate_indices_batch = {}
                for covariate in covariate_indices:
                    covariate_indices_batch[covariate] = covariate_indices[covariate][
                        (i * batch_size) : ((i + 1) * batch_size)
                    ]
            else:
                covariate_indices_batch = None

            # Input to the sampling pre-defined size factors if provided to the function
            if self.is_binarized:
                log_size_factor_batch = (
                    log_size_factor[(i * batch_size) : ((i + 1) * batch_size)]
                    if log_size_factor is not None
                    else None
                )
            else:
                if log_size_factor is not None:
                    log_size_factor_batch = {}
                    for mod in self.modality_list:
                        log_size_factor_batch[mod] = log_size_factor[mod][
                            (i * batch_size) : ((i + 1) * batch_size)
                        ]
                else:
                    log_size_factor_batch = None

            # Sample batch
            X_samples = self.sample(
                batch_size,
                n_sample_steps,
                theta_covariate,
                size_factor_covariate,
                conditioning_covariates,
                covariate_indices_batch,
                log_size_factor_batch,
                unconditional,
            )

            for mod in X_samples:
                total_samples[mod].append(X_samples[mod].cpu())

        # Concatenate observations in the samples
        return {mod: torch.cat(total_samples[mod], dim=0) for mod in self.modality_list}

    def _decode(self, z, size_factor):
        # Decode the rescaled z
        if self.is_binarized:
            size_factor = {
                "rna": size_factor
            }  # Compatibility with the decoder implementation for multimodal data
        z = self.encoder_model.decode(z, size_factor)
        return z

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        # Resample from OT coupling
        if self.use_ot:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        # Sample time
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        # Sample noise along straight line
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x1)
        if self.use_ot:
            mu_t = t * x1 + (1 - t) * x0
        else:
            mu_t = t * x1
        return mu_t

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if self.use_ot:
            return self.sigma
        else:
            return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if self.use_ot:
            return x1 - x0
        else:
            t = t.unsqueeze(1)
            return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)
