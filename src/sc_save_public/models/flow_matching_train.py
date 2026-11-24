"""Compatibility shim for Stage-2 Flow Matching training delegating to ``sc_save.training``."""

from __future__ import annotations

import torch

from sc_save.training.fm_trainer import train_fm
from sc_save.training.runtime import (
    add_cond_mask,
    add_cond_mask_with_emb,
    analytic_kl_div,
)


class DummyContextManager:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def kl_div(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return analytic_kl_div(mu, var)


def flow_matching_train(
    vae,
    fm,
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
    train_stage=1,
    lr_milestone=0,
    is_emb_ary=False,
    use_amp_fm=True,
    **kwargs,
):
    effective_clip = 1.0 if grad_clip in (False, True) else grad_clip
    return train_fm(
        vae=vae,
        fm=fm,
        dataloader=dataloader,
        num_epoch=num_epoch,
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
        use_amp_fm=use_amp_fm,
        is_multi_codebook=is_multi_codebook,
        is_emb_ary=is_emb_ary,
    )


__all__ = [
    "DummyContextManager",
    "add_cond_mask",
    "add_cond_mask_with_emb",
    "flow_matching_train",
    "kl_div",
]
