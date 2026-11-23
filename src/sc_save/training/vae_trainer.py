"""Stage 1 Attention-VAE training loop."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from tqdm import tqdm

from sc_save.training.runtime import (
    add_cond_mask,
    add_cond_mask_with_emb,
    analytic_kl_div,
    build_adamw,
    build_warmup_cosine_scheduler,
    enable_cuda_tf32,
    get_autocast_context,
    iter_resident_batches,
    optimizer_step,
    resolve_resident_source,
    seed_everything,
)


def train_vae(
    vae: nn.Module,
    x: np.ndarray | sparse.spmatrix | torch.Tensor | None = None,
    y: np.ndarray | torch.Tensor | None = None,
    *,
    dataloader: Any = None,
    emb: np.ndarray | torch.Tensor | None = None,
    num_epoch: int = 200,
    batch_size: int = 1024,
    kl_scale: float = 0.5,
    device: torch.device = torch.device("cpu"),
    lr: float = 1e-4,
    grad_clip: bool | float = 1.0,
    seed: int = 1202,
    weight_decay: float = 2.5e-5,
    col_msk_threshold: float = 1.0,
    loss_monitor: Any = None,
    tune_session: Any = None,
    is_col_wise_mask: bool = False,
    is_flash_attn: bool = False,
    use_amp_vae: bool = False,
    is_multi_codebook: bool = False,
    recon_criterion: str = "bce",
    is_emb_ary: bool = False,
    drop_last: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """Train the Stage-1 VAE with adaptive GPU residency and zero-sync KL/losses."""
    seed_everything(seed)
    enable_cuda_tf32(device)

    if num_epoch <= 0:
        return {}

    vae.to(device)
    optimizer = build_adamw(vae.parameters(), lr=lr, weight_decay=weight_decay, device=device)
    scheduler = build_warmup_cosine_scheduler(optimizer, num_epoch=num_epoch, lr=lr)

    amp_ctx, use_grad_scaler = get_autocast_context(
        device, use_amp=use_amp_vae, prefer_fp16=is_flash_attn
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler) if use_grad_scaler else None

    recon_criterion_fn = F.binary_cross_entropy if recon_criterion == "bce" else F.mse_loss
    max_grad_norm = 1.0 if grad_clip is True else (float(grad_clip) if grad_clip else 0.0)

    resident, batch_size, drop_last = resolve_resident_source(
        x,
        y,
        dataloader,
        device=device,
        batch_size=batch_size,
        drop_last=drop_last,
        emb=emb,
        is_emb_ary=is_emb_ary,
    )

    epoch_loss: dict[str, float] = {}
    epoch_iter = tqdm(range(num_epoch), ncols=80) if verbose else range(num_epoch)

    try:
        for _ in epoch_iter:
            vae.train()
            recon_accum = torch.zeros((), device=device)
            kl_accum = torch.zeros((), device=device)
            num_batches = 0

            batch_stream = (
                iter_resident_batches(resident, batch_size=batch_size, shuffle=True, drop_last=drop_last)
                if resident is not None
                else dataloader
            )

            for batch in batch_stream:
                num_batches += 1
                if resident is not None:
                    x_b, y_b = batch[0], batch[1]
                    emb_b = batch[2] if len(batch) == 3 else None
                else:
                    x_b = batch[0].float().to(device)
                    y_b = batch[1].long().to(device)
                    emb_b = batch[2].float().to(device) if (is_emb_ary and len(batch) >= 3) else None

                if emb_b is not None:
                    c = add_cond_mask_with_emb(
                        emb_b, y_b, col_msk_threshold, vae, is_col_wise_mask, is_multi_codebook
                    )
                else:
                    c = add_cond_mask(
                        y_b, col_msk_threshold, vae, is_col_wise_mask, is_multi_codebook
                    )

                with amp_ctx:
                    z, mu, var = vae.encoder(x_b, c)
                    recon_x = vae.decoder(z, c)

                # Keep BCE & KL in FP32 for exact numerical parity.
                recon_loss = recon_criterion_fn(recon_x.float(), x_b.float()) * x_b.size(-1)
                mu_flat = mu.float().reshape(-1, mu.shape[-1])
                var_flat = var.float().reshape(-1, var.shape[-1])
                kl_loss = analytic_kl_div(mu_flat, var_flat) * kl_scale

                optimizer_step(
                    optimizer,
                    recon_loss + kl_loss,
                    vae.parameters(),
                    scaler=scaler,
                    max_grad_norm=max_grad_norm,
                )
                recon_accum = recon_accum + recon_loss.detach()
                kl_accum = kl_accum + kl_loss.detach()

            scheduler.step()

            denom = max(1, num_batches)
            epoch_loss = {
                "recon_loss": (recon_accum / denom).item(),
                "kl_loss": (kl_accum / denom).item(),
            }
            if verbose and hasattr(epoch_iter, "set_postfix_str"):
                epoch_iter.set_postfix_str(
                    f"recon_loss={epoch_loss['recon_loss']:.3f},kl_loss={epoch_loss['kl_loss']:.3f}"
                )

            if loss_monitor:
                loss_monitor.log(
                    {
                        "recon_loss": epoch_loss["recon_loss"],
                        "kl_loss": epoch_loss["kl_loss"],
                        "lr": scheduler.get_last_lr()[0],
                        "cond_msk_prob": col_msk_threshold,
                    }
                )
            if tune_session:
                tune_session.report({"loss": sum(epoch_loss.values())})
    finally:
        optimizer.zero_grad(set_to_none=True)
        if resident is not None:
            del resident
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return epoch_loss


__all__ = ["train_vae"]
