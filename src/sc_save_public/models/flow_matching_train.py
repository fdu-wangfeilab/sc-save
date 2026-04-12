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
from sc_save_public.models.VAE_model import VAE
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

    context = (
        torch.amp.autocast(device_type=device.type, dtype=torch.float16)
        if is_flash_attn
        else DummyContextManager()
    )

    recon_criterion = F.binary_cross_entropy if recon_criterion == "bce" else F.mse_loss

    scaler = torch.amp.GradScaler(device=device)

    if is_flash_attn:

        def step_optimizer(loss_dict):
            optimizer.zero_grad()
            scaler.scale(sum(loss_dict.values())).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(fm.denoising_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    else:

        def step_optimizer(loss_dict):
            optimizer.zero_grad()
            sum(loss_dict.values()).backward()
            nn.utils.clip_grad_norm_(fm.denoising_model.parameters(), 1.0)
            optimizer.step()

    tq = tqdm(range(num_epoch), ncols=80)

    vae.eval()
    fm.train()

    for epoch in tq:
        epoch_loss = defaultdict(float)

        current_cond_msk_prob = col_msk_threshold

        for i, data in enumerate(dataloader):
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

                # c = add_cond_mask(
                #     y=y,
                #     col_msk_threshold=col_msk_threshold,
                #     model=fm.denoising_model,
                #     is_col_wise_mask=is_col_wise_mask,
                #     is_multi_codebook=is_multi_codebook,
                # )

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

            for k, v in loss.items():
                epoch_loss[k] += loss[k].item()

        scheduler.step()

        epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
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

    # for some config record
    return epoch_loss
