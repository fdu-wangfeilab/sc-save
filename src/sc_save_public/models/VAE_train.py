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


def calculate_noise_prob(
    current_epoch, total_epochs, min_prob=0.2, max_prob=0.8, warmup_epochs=10
):
    """
    计算当前epoch对应的噪声解码概率

    参数:
    current_epoch: 当前训练的epoch
    total_epochs: 总的训练epochs
    min_prob: 最小噪声概率
    max_prob: 最大噪声概率
    warmup_ratio: 预热阶段比例，在这个阶段内保持最小概率

    返回:
    当前epoch对应的噪声解码概率
    """
    # 预热阶段
    if current_epoch < warmup_epochs:
        return 0

    if current_epoch > total_epochs:
        return max_prob

    # 使用余弦函数增长
    ratio = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    # 余弦函数从0增长到1: (1 - cos(π*ratio))/2
    cosine_ratio = (1 - np.cos(np.pi * ratio)) / 2
    current_prob = min_prob + (max_prob - min_prob) * cosine_ratio

    # 确保概率不超过设定范围
    return min(max(current_prob, min_prob), max_prob)


def add_cond_mask(
    y, col_msk_threshold, vae, is_col_wise_mask=False, is_multi_codebook=False
):
    # 按照列来遮蔽条件
    if random.random() < col_msk_threshold and is_col_wise_mask:
        # random mask condition
        idx = random.choice(vae.col_comb)
        y[:, idx] = vae.mask_id

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

    if is_multi_codebook:
        c = torch.zeros(y.shape[0], vae.types_of_conds, vae.expand_dim).to(y.device)
        for i in range(vae.types_of_conds):
            c[:, i, :] = vae.cond_embedder[i](y[:, i])
    else:
        y_flat = rearrange(y, "b c -> (b c)")
        c = vae.cond_embedder(y_flat)
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
    **kwargs,
):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    lr_milestone = num_epoch // 4
    # if is_flash_attn:
    #     lr = lr / 10
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler1 = lr_scheduler.ConstantLR(
        optimizer=optimizer, total_iters=lr_milestone, factor=1
    )
    # scheduler1 = lr_scheduler.LinearLR(
    #     optimizer, total_iters=lr_milestone, start_factor=0.1, end_factor=1
    # )
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, last_epoch=-1, eta_min=lr / 10
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[lr_milestone]
    )

    # cls_criterion = nn.CrossEntropyLoss()

    context = (
        torch.amp.autocast(device_type=device.type, dtype=torch.float16)
        if is_flash_attn
        else DummyContextManager()
    )

    recon_criterion = F.binary_cross_entropy if recon_criterion == "bce" else F.mse_loss

    scaler = torch.amp.GradScaler(device=device)

    print(f"is col wise mask: {is_col_wise_mask}")
    # print(f"is enc dec cond diff: {is_enc_dec_cond_diff}")

    if is_flash_attn:

        def step_optimizer(loss_dict):
            optimizer.zero_grad()
            scaler.scale(sum(loss_dict.values())).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    else:

        def step_optimizer(loss_dict):
            optimizer.zero_grad()
            sum(loss_dict.values()).backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

    # print(f"is_col_wise_mask: {is_col_wise_mask}")
    # if is_flash_attn:
    #     print("-" * 10, "🚀 use flash attention", "-" * 10)
    print(f"is multi codebook: {is_multi_codebook}")

    tq = tqdm(range(num_epoch), ncols=80)
    for epoch in tq:
        vae.train()
        epoch_loss = defaultdict(float)

        # 计算当前epoch对应的噪声解码概率
        # if max_noise_prob is not None:
        #     current_noise_prob = calculate_noise_prob(
        #         current_epoch=epoch,
        #         total_epochs=num_epoch,
        #         warmup_epochs=lr_milestone,
        #         min_prob=0.2,
        #         max_prob=max_noise_prob,
        #     )
        # else:
        #     current_noise_prob = -1

        current_cond_msk_prob = col_msk_threshold

        kl_penalty = 1

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
                c = add_cond_mask(
                    y, col_msk_threshold, vae, is_col_wise_mask, is_multi_codebook
                )

            with context:
                z, mu, var = vae.encoder(x, c)
                recon_x = vae.decoder(
                    z,
                    c,
                )

                # using bce loss estimating the error
                recon_loss = recon_criterion(recon_x, x) * x.size(-1)
                # 将 mu, var 的后两维展开为一维
                kl_loss = kl_div(
                    rearrange(mu, "b s d -> (b s) d"),
                    rearrange(var, "b s d -> (b s) d"),
                )
                # recon_loss = F.mse_loss(recon_x, x) * x.size(-1)
                loss = {
                    "recon_loss": recon_loss,
                    "kl_loss": kl_scale * kl_loss * kl_penalty,
                    # "cls_loss": cls_loss,
                }

            step_optimizer(loss)

            for k, v in loss.items():
                epoch_loss[k] += loss[k].item()

        # if is_lr_scheduler:
        scheduler.step()

        epoch_loss = {k: v / (i + 1) for k, v in epoch_loss.items()}
        epoch_info = ",".join(["{}={:.3f}".format(k, v) for k, v in epoch_loss.items()])
        tq.set_postfix_str(epoch_info)

        if loss_monitor:
            loss_monitor.log(
                {
                    "recon_loss": epoch_loss["recon_loss"],
                    "kl_loss": epoch_loss["kl_loss"],
                    "lr": scheduler.get_last_lr()[0],
                    # "cls_loss": epoch_loss["cls_loss"],
                    # "noise_prob": current_noise_prob,  # 记录当前噪声概率
                    "cond_msk_prob": current_cond_msk_prob,
                }
            )

        if tune_session:
            tune_session.report({"loss": sum(epoch_loss.values())})

    # for some config record
    return epoch_loss
