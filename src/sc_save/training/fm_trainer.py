"""Stage 2 latent Flow Matching training loop with posterior precomputation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from tqdm import tqdm

from sc_save.training.runtime import (
    ResidentTensors,
    add_cond_mask_with_emb,
    build_adamw,
    build_warmup_cosine_scheduler,
    enable_cuda_tf32,
    get_autocast_context,
    get_gpu_tensor_budget_bytes,
    iter_resident_batches,
    optimizer_step,
    resolve_resident_source,
    seed_everything,
)


def _precompute_posteriors(
    vae: nn.Module,
    resident_x: ResidentTensors,
    batch_size: int,
    device: torch.device,
    amp_ctx: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.device, bool]:
    """Encode ``x`` once with masked conditions into ``(mu_all, std_all, y_all)``."""
    n_samples = resident_x.n_samples
    pre_bs = max(batch_size, 1024)
    mu_chunks: list[torch.Tensor] = []
    std_chunks: list[torch.Tensor] = []
    store_device = device

    with torch.inference_mode():
        for x_b, y_b in iter_resident_batches(resident_x, batch_size=pre_bs, shuffle=False):
            mask_ids = torch.full((y_b.numel(),), vae.mask_id, device=device, dtype=torch.long)
            mask_emb = vae.cond_embedder(mask_ids).reshape(y_b.shape[0], y_b.shape[1], -1)

            with amp_ctx:
                _, mu_b, var_b = vae.encoder(x_b, mask_emb)
            mu_b = mu_b.float()
            std_b = torch.sqrt(var_b.float())

            if not mu_chunks and device.type == "cuda":
                est_bytes = n_samples * (mu_b[0].numel() * 4 * 2 + y_b[0].numel() * 8)
                if est_bytes > get_gpu_tensor_budget_bytes(device):
                    store_device = torch.device("cpu")

            mu_chunks.append(mu_b.to(store_device))
            std_chunks.append(std_b.to(store_device))

    mu_all = torch.cat(mu_chunks, dim=0)
    std_all = torch.cat(std_chunks, dim=0)
    y_all = resident_x.y.to(store_device, dtype=torch.long)

    pinned = False
    if store_device.type == "cpu" and device.type == "cuda" and torch.cuda.is_available():
        try:
            mu_all = mu_all.pin_memory()
            std_all = std_all.pin_memory()
            y_all = y_all.pin_memory()
            pinned = True
        except RuntimeError:
            pinned = False

    return mu_all, std_all, y_all, store_device, pinned


def train_fm(
    vae: nn.Module,
    fm: nn.Module,
    x: np.ndarray | sparse.spmatrix | torch.Tensor | None = None,
    y: np.ndarray | torch.Tensor | None = None,
    *,
    dataloader: Any = None,
    emb: np.ndarray | torch.Tensor | None = None,
    num_epoch: int = 200,
    batch_size: int = 1024,
    device: torch.device = torch.device("cpu"),
    lr: float = 1e-4,
    grad_clip: bool | float = 1.0,
    seed: int = 1202,
    weight_decay: float = 2.5e-5,
    col_msk_threshold: float = 0.0,
    loss_monitor: Any = None,
    tune_session: Any = None,
    is_col_wise_mask: bool = False,
    is_flash_attn: bool = False,
    use_amp_fm: bool = True,
    is_multi_codebook: bool = False,
    is_emb_ary: bool = False,
    drop_last: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """Train Stage-2 Flow Matching on precomputed ``(mu_all, std_all, y_all)``."""
    seed_everything(seed)
    enable_cuda_tf32(device)

    if num_epoch <= 0:
        return {}

    vae.to(device)
    fm.to(device)
    fm.device = device

    optimizer = build_adamw(
        fm.denoising_model.parameters(), lr=lr, weight_decay=weight_decay, device=device
    )
    scheduler = build_warmup_cosine_scheduler(optimizer, num_epoch=num_epoch, lr=lr)

    amp_ctx, use_grad_scaler = get_autocast_context(
        device, use_amp=use_amp_fm, prefer_fp16=is_flash_attn
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler) if use_grad_scaler else None
    max_grad_norm = 1.0 if grad_clip is True else (float(grad_clip) if grad_clip else 0.0)

    vae.eval()
    fm.train()

    use_precompute = emb is None and not is_emb_ary
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

    precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.device, bool] | None = None
    resident_fallback: ResidentTensors | None = None
    if use_precompute and resident is not None:
        try:
            precomputed = _precompute_posteriors(vae, resident, batch_size, device, amp_ctx)
        finally:
            del resident
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        resident_fallback = resident

    epoch_loss: dict[str, float] = {}
    epoch_iter = tqdm(range(num_epoch), ncols=80) if verbose else range(num_epoch)

    try:
        for _ in epoch_iter:
            loss_accum = torch.zeros((), device=device)
            num_batches = 0

            if precomputed is not None:
                mu_all, std_all, y_all, store_device, pinned = precomputed
                n_samples = mu_all.shape[0]
                limit = (
                    (n_samples // batch_size) * batch_size
                    if drop_last and n_samples >= batch_size
                    else n_samples
                )
                perm = torch.randperm(n_samples, device=store_device)

                for start in range(0, limit, batch_size):
                    num_batches += 1
                    idx = perm[start : min(start + batch_size, limit)]
                    mu_b = mu_all[idx]
                    std_b = std_all[idx]
                    y_b = y_all[idx]
                    if store_device != device:
                        mu_b = mu_b.to(device, non_blocking=pinned)
                        std_b = std_b.to(device, non_blocking=pinned)
                        y_b = y_b.to(device, non_blocking=pinned)

                    z = mu_b + torch.randn_like(mu_b) * std_b
                    c = fm.denoising_model.cond_embedder(y_b.reshape(-1)).reshape(
                        y_b.shape[0], y_b.shape[1], -1
                    )
                    with amp_ctx:
                        loss_val = fm._step(z, c)

                    optimizer_step(
                        optimizer,
                        loss_val,
                        fm.denoising_model.parameters(),
                        scaler=scaler,
                        max_grad_norm=max_grad_norm,
                    )
                    loss_accum = loss_accum + loss_val.detach()
            else:
                batch_stream = (
                    iter_resident_batches(
                        resident_fallback, batch_size=batch_size, shuffle=True, drop_last=drop_last
                    )
                    if resident_fallback is not None
                    else dataloader
                )
                for batch in batch_stream:
                    num_batches += 1
                    if resident_fallback is not None:
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
                        mask_emb = c
                    else:
                        c = fm.denoising_model.cond_embedder(y_b.reshape(-1)).reshape(
                            y_b.shape[0], y_b.shape[1], -1
                        )
                        mask_ids = torch.full((y_b.numel(),), vae.mask_id, device=device, dtype=torch.long)
                        mask_emb = vae.cond_embedder(mask_ids).reshape(y_b.shape[0], y_b.shape[1], -1)

                    with amp_ctx:
                        with torch.no_grad():
                            z, _, _ = vae.encoder(x_b, mask_emb)
                        loss_val = fm._step(z, c)

                    optimizer_step(
                        optimizer,
                        loss_val,
                        fm.denoising_model.parameters(),
                        scaler=scaler,
                        max_grad_norm=max_grad_norm,
                    )
                    loss_accum = loss_accum + loss_val.detach()

            scheduler.step()

            epoch_loss = {"loss": (loss_accum / max(1, num_batches)).item()}
            if verbose and hasattr(epoch_iter, "set_postfix_str"):
                epoch_iter.set_postfix_str(f"loss={epoch_loss['loss']:.3f}")

            if loss_monitor:
                loss_monitor.log({"loss": epoch_loss["loss"], "lr": scheduler.get_last_lr()[0]})
            if tune_session:
                tune_session.report({"loss": epoch_loss["loss"]})
    finally:
        optimizer.zero_grad(set_to_none=True)
        if precomputed is not None:
            del precomputed
        if resident_fallback is not None:
            del resident_fallback
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return epoch_loss


__all__ = ["train_fm"]
