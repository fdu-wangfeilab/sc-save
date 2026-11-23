"""Distribution distance metrics (Wasserstein and MMD) for sc_save evaluation."""

from __future__ import annotations

import gc
import math

import numpy as np
import ot
import torch


def clear_gpu_memory() -> None:
    """Explicitly trigger Python GC and empty the CUDA caching allocator."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def linear_mmd2(f_of_x: torch.Tensor, f_of_y: torch.Tensor) -> torch.Tensor:
    """Compute linear-time unbiased MMD^2 estimate clamped at zero."""
    delta = f_of_x - f_of_y
    if delta.shape[0] <= 1:
        return torch.tensor(0.0, device=delta.device)
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return torch.maximum(torch.tensor(0.0, device=delta.device), loss)


def _mix_rbf_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_list: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if X.size(0) != Y.size(0):
        raise ValueError("RBF MMD requires X and Y to have the same number of rows")

    m = X.size(0)
    Z = torch.cat((X, Y), dim=0)
    zzt = torch.mm(Z, Z.t())
    diag_zzt = torch.diag(zzt).unsqueeze(1)
    z_norm_sqr = diag_zzt.expand_as(zzt)
    exponent = z_norm_sqr - 2 * zzt + z_norm_sqr.t()

    kernel = torch.tensor(0.0, device=X.device)
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        kernel = kernel + torch.exp(-gamma * exponent)

    return kernel[:m, :m], kernel[:m, m:], kernel[m:, m:]


def _mmd2(
    K_XX: torch.Tensor,
    K_XY: torch.Tensor,
    K_YY: torch.Tensor,
    const_diagonal: float | bool = False,
    biased: bool = True,
) -> torch.Tensor:
    m = K_XX.size(0)
    if m == 0 or (not biased and m <= 1):
        return torch.tensor(0.0, device=K_XX.device, dtype=K_XX.dtype)

    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)
        diag_Y = torch.diag(K_YY)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sum = K_XY.sum()

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    if biased:
        return (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    return Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)


def mix_rbf_mmd2(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_list: list[float],
    biased: bool = True,
) -> torch.Tensor:
    """Compute multi-bandwidth RBF-kernel MMD^2 between equal-sized samples ``X`` and ``Y``."""
    K_XX, K_XY, K_YY = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def _wasserstein_from_dist_np(
    a: np.ndarray,
    b: np.ndarray,
    dist_np: np.ndarray,
    power: int,
) -> float:
    cost_np = dist_np if power == 1 else np.square(dist_np)
    result = ot.emd2(a, b, cost_np, numItermax=int(1e7))
    if power == 2:
        result = math.sqrt(max(0.0, float(result)))
    return float(result)


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    power: int = 2,
) -> float:
    """Compute exact 1-Wasserstein or 2-Wasserstein distance via POT ``ot.emd2``."""
    if power not in {1, 2}:
        raise ValueError(f"unsupported power: {power}")

    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)

    a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
    dist_np = torch.cdist(x0, x1).detach().cpu().numpy()
    return _wasserstein_from_dist_np(a, b, dist_np, power=power)


def _to_tensor(array, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.float().to(device)
    return torch.tensor(np.asarray(array)).float().to(device)


def compute_batched_linear_mmd(
    pred: torch.Tensor,
    true: torch.Tensor,
    batch_size: int,
) -> float:
    """Compute overlapping-window averaged linear MMD^2 without per-batch GC/cache sync."""
    n_samples = pred.shape[0]
    if n_samples <= batch_size:
        return linear_mmd2(pred, true).item()

    total_mmd = 0.0
    num_batches = 0
    step = max(1, batch_size // 2)
    for start_idx in range(0, n_samples - batch_size + 1, step):
        end_idx = min(start_idx + batch_size, n_samples)
        pred_batch = pred[start_idx:end_idx]
        true_batch = true[start_idx:end_idx]
        if pred_batch.shape[0] > 1:
            total_mmd += linear_mmd2(pred_batch, true_batch).item()
            num_batches += 1

    return total_mmd / max(1, num_batches)


def compute_distribution_distances_batched(
    pred: torch.Tensor | np.ndarray,
    true: torch.Tensor | np.ndarray,
    batch_size: int = 1000,
    device: torch.device = torch.device("cpu"),
    is_mmd: bool = True,
) -> dict[str, float]:
    """Compute 1-Wasserstein, 2-Wasserstein (sharing one cdist), and optional MMD metrics."""
    pred_t = _to_tensor(pred, device)
    true_t = _to_tensor(true, device)

    pred_2d = pred_t.reshape(pred_t.shape[0], -1) if pred_t.dim() > 2 else pred_t
    true_2d = true_t.reshape(true_t.shape[0], -1) if true_t.dim() > 2 else true_t
    a, b = ot.unif(pred_2d.shape[0]), ot.unif(true_2d.shape[0])
    dist_np = torch.cdist(pred_2d, true_2d).detach().cpu().numpy()

    metrics: dict[str, float] = {
        "1-Wasserstein": _wasserstein_from_dist_np(a, b, dist_np, power=1),
        "2-Wasserstein": _wasserstein_from_dist_np(a, b, dist_np, power=2),
    }

    if is_mmd:
        min_size = min(pred_t.shape[0], true_t.shape[0])
        pred_equal = pred_t[:min_size]
        true_equal = true_t[:min_size]
        metrics["Linear_MMD_unbiased"] = compute_batched_linear_mmd(
            pred_equal,
            true_equal,
            batch_size=batch_size,
        )
        metrics["RBF_MMD"] = float(
            mix_rbf_mmd2(
                pred_equal, true_equal, sigma_list=[0.01, 0.1, 1, 10, 100]
            ).item()
        )

    return metrics


def compute_evaluation_metrics_fast(
    adata_real_pca_ary,
    adata_generated_pca_ary,
    model_name: str,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 1000,
    is_mmd: bool = True,
) -> dict[str, float]:
    """Compute PCA-space distribution distance metrics keyed with ``_PCA`` suffix."""
    del model_name
    distance_metrics = compute_distribution_distances_batched(
        adata_real_pca_ary,
        adata_generated_pca_ary,
        batch_size=batch_size,
        device=device,
        is_mmd=is_mmd,
    )
    return {
        f"{metric_name}_PCA": metric_value
        for metric_name, metric_value in distance_metrics.items()
    }


__all__ = [
    "clear_gpu_memory",
    "compute_batched_linear_mmd",
    "compute_distribution_distances_batched",
    "compute_evaluation_metrics_fast",
    "linear_mmd2",
    "mix_rbf_mmd2",
    "wasserstein",
]
