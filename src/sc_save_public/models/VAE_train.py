"""Compatibility shim for Stage-1 VAE training delegating to ``sc_save.training``."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from sc_save.training.runtime import (
    add_cond_mask,
    add_cond_mask_with_emb,
    analytic_kl_div,
    enable_cuda_tf32,
    get_gpu_tensor_budget_bytes,
)
from sc_save.training.vae_trainer import train_vae


class DummyContextManager:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def kl_div(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return analytic_kl_div(mu, var)


def calculate_noise_prob(
    current_epoch: int,
    total_epochs: int,
    min_prob: float = 0.2,
    max_prob: float = 0.8,
    warmup_epochs: int = 10,
) -> float:
    if current_epoch < warmup_epochs:
        return 0
    if current_epoch > total_epochs:
        return max_prob
    ratio = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    cosine_ratio = (1 - np.cos(np.pi * ratio)) / 2
    current_prob = min_prob + (max_prob - min_prob) * cosine_ratio
    return min(max(current_prob, min_prob), max_prob)


def maybe_move_tensor_dataset_to_gpu(
    dataloader,
    device: torch.device,
    is_emb_ary: bool = False,
) -> tuple[torch.Tensor, ...] | None:
    dataset = getattr(dataloader, "dataset", None)
    if device.type != "cuda" or not isinstance(dataset, TensorDataset) or len(dataset) == 0:
        return None
    tensors = dataset.tensors
    needed = 2 if not is_emb_ary or len(tensors) < 3 else 3
    if len(tensors) < needed:
        return None

    cast_dtypes = [torch.float32, torch.long] + ([torch.float32] if needed == 3 else [])
    total_bytes = sum(
        t.nelement() * torch.empty((), dtype=dt).element_size()
        for t, dt in zip(tensors[:needed], cast_dtypes)
    )
    if total_bytes > get_gpu_tensor_budget_bytes(device):
        return None

    try:
        return tuple(
            t.to(dtype=dt, device=device) for t, dt in zip(tensors[:needed], cast_dtypes)
        )
    except RuntimeError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def iter_shuffled_batches(
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    device: torch.device,
    drop_last: bool = False,
):
    n_samples = tensors[0].shape[0]
    perm = torch.randperm(n_samples, device=device)
    limit = (
        (n_samples // batch_size) * batch_size
        if drop_last and n_samples >= batch_size
        else n_samples
    )
    for start in range(0, limit, batch_size):
        idx = perm[start : min(start + batch_size, limit)]
        yield tuple(t[idx] for t in tensors)


def vae_train(
    vae,
    dataloader,
    num_epoch,
    kl_scale=0.5,
    device=torch.device("cuda:0"),
    is_tensorboard=False,
    lr=2e-4,
    grad_clip=False,
    seed=1202,
    is_lr_scheduler=False,
    weight_decay=5e-4,
    col_msk_threshold=0.2,
    loss_monitor=None,
    max_noise_prob=0.3,
    is_col_wise_mask=False,
    tune_session=None,
    is_flash_attn=False,
    is_partial_mask=False,
    is_multi_codebook=False,
    recon_criterion="bce",
    is_enc_dec_cond_diff=False,
    lr_milestone=0,
    is_emb_ary=False,
    use_amp_vae=False,
    **kwargs,
):
    # Preserve v1 behavior where gradient clipping was applied unconditionally.
    effective_clip = 1.0 if grad_clip in (False, True) else grad_clip
    return train_vae(
        vae=vae,
        dataloader=dataloader,
        num_epoch=num_epoch,
        kl_scale=kl_scale,
        device=device,
        lr=lr,
        grad_clip=effective_clip,
        seed=seed,
        weight_decay=weight_decay,
        col_msk_threshold=col_msk_threshold,
        loss_monitor=loss_monitor,
        tune_session=tune_session,
        is_col_wise_mask=is_col_wise_mask,
        is_flash_attn=is_flash_attn,
        use_amp_vae=use_amp_vae,
        is_multi_codebook=is_multi_codebook,
        recon_criterion=recon_criterion,
        is_emb_ary=is_emb_ary,
    )


__all__ = [
    "DummyContextManager",
    "add_cond_mask",
    "add_cond_mask_with_emb",
    "calculate_noise_prob",
    "enable_cuda_tf32",
    "get_gpu_tensor_budget_bytes",
    "iter_shuffled_batches",
    "kl_div",
    "maybe_move_tensor_dataset_to_gpu",
    "vae_train",
]
