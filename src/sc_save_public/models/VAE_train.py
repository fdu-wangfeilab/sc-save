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
from sc_save_public.models.utils.process_h5ad import convert_adata_to_dataloader


_MAX_GPU_RESIDENT_GB = 16.0
_MAX_GPU_FREE_VRAM_FRACTION = 0.35


def enable_cuda_tf32(device: torch.device) -> None:
    """Enable TF32 matmul/cudnn acceleration on CUDA without changing FP32 model semantics."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def get_gpu_tensor_budget_bytes(
    device: torch.device,
    vram_fraction: float = _MAX_GPU_FREE_VRAM_FRACTION,
    max_gb: float = _MAX_GPU_RESIDENT_GB,
) -> int:
    """Return max bytes safe to keep resident on GPU (`min(16 GB, 0.35 * free_vram)`)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except RuntimeError:
        return 0
    return int(min(max_gb * (1024**3), vram_fraction * free_bytes))


def maybe_move_tensor_dataset_to_gpu(
    dataloader,
    device: torch.device,
    is_emb_ary: bool = False,
) -> tuple[torch.Tensor, ...] | None:
    """Copy a TensorDataset's tensors to CUDA once if they fit safely within VRAM budget."""
    dataset = getattr(dataloader, "dataset", None)
    if device.type != "cuda" or not isinstance(dataset, TensorDataset) or len(dataset) == 0:
        return None
    tensors = dataset.tensors
    needed = (2 if not is_emb_ary or len(tensors) < 3 else 3)
    if len(tensors) < needed:
        return None

    cast_dtypes = [torch.float32, torch.long] + ([torch.float32] if needed == 3 else [])
    total_bytes = sum(t.nelement() * torch.empty((), dtype=dt).element_size() for t, dt in zip(tensors[:needed], cast_dtypes))
    if total_bytes > get_gpu_tensor_budget_bytes(device):
        return None

    try:
        return tuple(t.to(dtype=dt, device=device) for t, dt in zip(tensors[:needed], cast_dtypes))
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
    """Yield shuffled mini-batches directly from resident tensors without DataLoader IPC."""
    n_samples = tensors[0].shape[0]
    perm = torch.randperm(n_samples, device=device)
    limit = (n_samples // batch_size) * batch_size if drop_last and n_samples >= batch_size else n_samples
    for start in range(0, limit, batch_size):
        idx = perm[start : min(start + batch_size, limit)]
        yield tuple(t[idx] for t in tensors)


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

    enable_cuda_tf32(device)

    if num_epoch <= 0:
        return {}

    lr_milestone = num_epoch // 4
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)
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

    recon_criterion_fn = F.binary_cross_entropy if recon_criterion == "bce" else F.mse_loss

    scaler = torch.amp.GradScaler(device=device)

    print(f"is col wise mask: {is_col_wise_mask}")

    if is_flash_attn:

        def step_optimizer(loss_dict):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(sum(loss_dict.values())).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    else:

        def step_optimizer(loss_dict):
            optimizer.zero_grad(set_to_none=True)
            sum(loss_dict.values()).backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

    print(f"is multi codebook: {is_multi_codebook}")

    gpu_tensors = maybe_move_tensor_dataset_to_gpu(
        dataloader, device, is_emb_ary=is_emb_ary
    )
    batch_size = int(getattr(dataloader, "batch_size", None) or len(dataloader.dataset))
    drop_last = bool(getattr(dataloader, "drop_last", False))

    tq = tqdm(range(num_epoch), ncols=80)
    epoch_loss = {}
    try:
        for epoch in tq:
            vae.train()
            loss_accum = {
                "recon_loss": torch.zeros((), device=device),
                "kl_loss": torch.zeros((), device=device),
            }
            current_cond_msk_prob = col_msk_threshold
            kl_penalty = 1
            num_batches = 0

            batch_iter = (
                iter_shuffled_batches(
                    gpu_tensors, batch_size=batch_size, device=device, drop_last=drop_last
                )
                if gpu_tensors is not None
                else dataloader
            )

            for data in batch_iter:
                num_batches += 1
                if gpu_tensors is not None:
                    if is_emb_ary:
                        x, y, emb = data[0], data[1], data[2]
                        c = add_cond_mask_with_emb(
                            emb, y, col_msk_threshold, vae, is_col_wise_mask, is_multi_codebook
                        )
                    else:
                        x, y = data[0], data[1]
                        c = add_cond_mask(
                            y, col_msk_threshold, vae, is_col_wise_mask, is_multi_codebook
                        )
                else:
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
                    recon_x = vae.decoder(z, c)

                # Keep BCE & KL computations in FP32 for numerical parity.
                recon_loss = recon_criterion_fn(recon_x.float(), x.float()) * x.size(-1)
                kl_loss = kl_div(
                    rearrange(mu.float(), "b s d -> (b s) d"),
                    rearrange(var.float(), "b s d -> (b s) d"),
                )
                loss = {
                    "recon_loss": recon_loss,
                    "kl_loss": kl_scale * kl_loss * kl_penalty,
                }

                step_optimizer(loss)

                for k, v in loss.items():
                    loss_accum[k] = loss_accum[k] + v.detach()

            scheduler.step()

            denom = max(1, num_batches)
            epoch_loss = {k: (v / denom).item() for k, v in loss_accum.items()}
            epoch_info = ",".join(["{}={:.3f}".format(k, v) for k, v in epoch_loss.items()])
            tq.set_postfix_str(epoch_info)

            if loss_monitor:
                loss_monitor.log(
                    {
                        "recon_loss": epoch_loss["recon_loss"],
                        "kl_loss": epoch_loss["kl_loss"],
                        "lr": scheduler.get_last_lr()[0],
                        "cond_msk_prob": current_cond_msk_prob,
                    }
                )

            if tune_session:
                tune_session.report({"loss": sum(epoch_loss.values())})
    finally:
        if gpu_tensors is not None:
            del gpu_tensors
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    return epoch_loss
