import torch
import torch.nn as nn
import numpy as np
import os
import random
from tqdm import tqdm
import scanpy as sc
import torch.nn.functional as F
from collections import defaultdict
from torch.distributions import Normal, kl_divergence
from torch.optim import lr_scheduler
from torch.distributions import Normal
from einops import rearrange, repeat
from torch.utils.data import TensorDataset
from sc_save_public.models.VAE_model import VAE
from sc_save_public.models.VAE_train import (
    enable_cuda_tf32,
    get_gpu_tensor_budget_bytes,
    iter_shuffled_batches,
)
from sc_save_public.models.utils.process_h5ad import convert_adata_to_dataloader

from sc_save_public.models.fm.fm import FM


class DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def kl_div(mu, var):
    return (
        kl_divergence(
            Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
        )
        .sum(dim=1)
        .mean()
    )


def add_cond_mask(
    y, col_msk_threshold, model, is_col_wise_mask=False, is_multi_codebook=False
):
    # 按照列来遮蔽条件
    # if random.random() < col_msk_threshold and is_col_wise_mask:
    #     # random mask condition
    #     idx = random.choice(vae.col_comb)
    #     y[:, idx] = vae.mask_id

    # 随机遮蔽条件
    if random.random() < col_msk_threshold and not is_col_wise_mask:
        y_flat = rearrange(y, "b c -> (b c)")
        # msk = torch.rand(y_flat.shape[0]).to(
        #     y_flat.device
        # ) < col_msk_threshold * (1 - 1 / self.types_of_cond)
        msk = torch.rand(y_flat.shape[0]).to(y_flat.device) < col_msk_threshold
        # msk = torch.rand(y_flat.shape[0]).to(y_flat.device) < 0.2
        y_flat = torch.where(msk, model.mask_id, y_flat)
        y = rearrange(y_flat, "(b c) -> b c", b=y.shape[0])

    y_flat = rearrange(y, "b c -> (b c)")
    c = model.cond_embedder(y_flat)
    c = rearrange(c, "(b c) e -> b c e", b=y.shape[0])

    return c


def add_cond_mask_with_emb(
    emb,
    y,
    col_msk_threshold,
    vae,
    is_col_wise_mask=False,
    is_multi_codebook=False,
):
    # 随机遮蔽条件
    if random.random() < col_msk_threshold and not is_col_wise_mask:
        y_flat = rearrange(y, "b c -> (b c)")
        # msk = torch.rand(y_flat.shape[0]).to(
        #     y_flat.device
        # ) < col_msk_threshold * (1 - 1 / self.types_of_cond)
        msk = torch.rand(y_flat.shape[0]).to(y_flat.device) < col_msk_threshold
        # msk = torch.rand(y_flat.shape[0]).to(y_flat.device) < 0.2
        y_flat = torch.where(msk, vae.mask_id, y_flat)
        y = rearrange(y_flat, "(b c) -> b c", b=y.shape[0])

    y_flat = rearrange(y, "b c -> (b c)")
    c = vae.cond_embedder(y_flat)
    c = rearrange(c, "(b c) e -> b c e", b=y.shape[0])

    embs = torch.split(emb, vae.outer_emb_dim, dim=1)

    emb_list = []
    for i in range(len(embs)):
        emb_list.append(vae.txt_emb_proj[i](embs[i]))
    emb = torch.cat(emb_list, dim=1)
    emb = rearrange(emb, "b (c e) -> b c e", c=y.shape[1])

    # 扩展mask维度以匹配embedding维度
    mask = (y == vae.mask_id).unsqueeze(-1)  # (1024, 2) -> (1024, 2, 1)
    c = torch.where(mask, c, emb)

    return c


def flow_matching_train(
    vae,
    fm: FM,
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
    **kwargs,
):
    # lr = lr / 10
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    enable_cuda_tf32(device)

    if num_epoch <= 0:
        return {}

    lr_milestone = num_epoch // 4
    optimizer = torch.optim.AdamW(
        fm.denoising_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler1 = lr_scheduler.ConstantLR(
        optimizer=optimizer, total_iters=lr_milestone, factor=1
    )

    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, last_epoch=-1, eta_min=lr / 10
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[lr_milestone]
    )

    use_bf16 = (
        device.type == "cuda"
        and not is_flash_attn
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if is_flash_attn:
        context = torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    elif use_bf16:
        context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        context = DummyContextManager()

    scaler = torch.amp.GradScaler(device=device)

    if is_flash_attn:

        def step_optimizer(loss_dict):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(sum(loss_dict.values())).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(fm.denoising_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    else:

        def step_optimizer(loss_dict):
            optimizer.zero_grad(set_to_none=True)
            sum(loss_dict.values()).backward()
            nn.utils.clip_grad_norm_(fm.denoising_model.parameters(), 1.0)
            optimizer.step()

    vae.eval()
    fm.train()

    # Precompute frozen VAE encoder posteriors (mu, var) once when possible
    # so 200 FM epochs do not re-encode full gene expression every step.
    precomputed_latents: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    resident_device = device
    dataset = getattr(dataloader, "dataset", None)
    if (
        not is_emb_ary
        and isinstance(dataset, TensorDataset)
        and len(dataset.tensors) >= 2
        and len(dataset) > 0
        and hasattr(getattr(vae, "encoder", None), "reparameterize")
    ):
        x_all, y_all = dataset.tensors[0], dataset.tensors[1]
        n_samples = x_all.shape[0]
        pre_bs = max(int(getattr(dataloader, "batch_size", None) or 1024), 1024)
        mu_chunks: list[torch.Tensor] = []
        var_chunks: list[torch.Tensor] = []
        target_store_device = device

        with torch.no_grad():
            for start in range(0, n_samples, pre_bs):
                end = min(start + pre_bs, n_samples)
                x_b = x_all[start:end].float().to(device)
                y_b = y_all[start:end].long().to(device)

                mask_ids = torch.ones_like(y_b) * vae.mask_id
                mask_ids = rearrange(mask_ids, "b c -> (b c)")
                mask_emb = vae.cond_embedder(mask_ids)
                mask_emb = rearrange(mask_emb, "(b c) e -> b c e", b=y_b.shape[0])

                with context:
                    _, mu_b, var_b = vae.encoder(x_b, mask_emb)
                mu_b = mu_b.float()
                var_b = var_b.float()

                if start == 0 and device.type == "cuda":
                    est_bytes = n_samples * (
                        mu_b[0].numel() * 4 * 2 + y_all[0].numel() * 8
                    )
                    if est_bytes > get_gpu_tensor_budget_bytes(device):
                        target_store_device = torch.device("cpu")

                mu_chunks.append(mu_b.to(target_store_device))
                var_chunks.append(var_b.to(target_store_device))

        mu_all = torch.cat(mu_chunks, dim=0)
        var_all = torch.cat(var_chunks, dim=0)
        y_resident = y_all.long().to(target_store_device)
        resident_device = target_store_device
        precomputed_latents = (mu_all, var_all, y_resident)
        del mu_chunks, var_chunks

    batch_size = int(getattr(dataloader, "batch_size", None) or len(dataloader.dataset))
    drop_last = bool(getattr(dataloader, "drop_last", False))

    tq = tqdm(range(num_epoch), ncols=80)
    epoch_loss = {}
    try:
        for epoch in tq:
            loss_accum = torch.zeros((), device=device)
            num_batches = 0

            if precomputed_latents is not None:
                for mu_b, var_b, y_b in iter_shuffled_batches(
                    precomputed_latents,
                    batch_size=batch_size,
                    device=resident_device,
                    drop_last=drop_last,
                ):
                    num_batches += 1
                    if resident_device != device:
                        mu_b = mu_b.to(device, non_blocking=True)
                        var_b = var_b.to(device, non_blocking=True)
                        y_b = y_b.to(device, non_blocking=True)

                    with torch.no_grad():
                        z = vae.encoder.reparameterize(mu_b, var_b)

                    y_flat = rearrange(y_b, "b c -> (b c)")
                    c = fm.denoising_model.cond_embedder(y_flat)
                    c = rearrange(c, "(b c) e -> b c e", b=y_b.shape[0])

                    with context:
                        loss = {"loss": fm._step(z, c)}

                    step_optimizer(loss)
                    loss_accum = loss_accum + loss["loss"].detach()
            else:
                for data in dataloader:
                    num_batches += 1
                    if is_emb_ary:
                        x, y, emb = (
                            data[0].float().to(device),
                            data[1].long().to(device),
                            data[2].float().to(device),
                        )
                        c = add_cond_mask_with_emb(
                            emb, y, col_msk_threshold, vae, is_col_wise_mask, is_multi_codebook
                        )
                    else:
                        x, y = data[0].float().to(device), data[1].long().to(device)
                        y_flat = rearrange(y, "b c -> (b c)")
                        c = fm.denoising_model.cond_embedder(y_flat)
                        c = rearrange(c, "(b c) e -> b c e", b=y.shape[0])

                        mask_ids = torch.ones_like(y).to(device) * vae.mask_id
                        mask_ids = rearrange(mask_ids, "b c -> (b c)")
                        mask_emb = vae.cond_embedder(mask_ids)
                        mask_emb = rearrange(mask_emb, "(b c) e -> b c e", b=y.shape[0])

                    with context:
                        with torch.no_grad():
                            z, _, _ = vae.encoder(x, mask_emb)
                        loss = {"loss": fm._step(z, c)}

                    step_optimizer(loss)
                    loss_accum = loss_accum + loss["loss"].detach()

            scheduler.step()

            epoch_loss = {"loss": (loss_accum / max(1, num_batches)).item()}
            epoch_info = ",".join(["{}={:.3f}".format(k, v) for k, v in epoch_loss.items()])
            tq.set_postfix_str(epoch_info)

            if loss_monitor:
                loss_monitor.log(
                    {
                        "loss": epoch_loss["loss"],
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

            if tune_session:
                tune_session.report({"loss": sum(epoch_loss.values())})
    finally:
        if precomputed_latents is not None:
            del precomputed_latents
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return epoch_loss
