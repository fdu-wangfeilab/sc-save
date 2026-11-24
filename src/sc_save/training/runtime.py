"""Training runtime utilities, adaptive GPU residency, and zero-sync helpers."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from einops import rearrange
from scipy import sparse
from torch.utils.data import TensorDataset

_MAX_GPU_RESIDENT_GB = 48.0
_MAX_GPU_FREE_VRAM_FRACTION = 0.50


def resolve_device(device: str | int | torch.device | None = "auto") -> torch.device:
    """Resolve ``'auto'``, integer device IDs, or strings to a ``torch.device``."""
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        if device < 0 or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(f"cuda:{device}")
    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return parsed


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_cuda_tf32(device: torch.device) -> None:
    """Enable TF32 matmul/cuDNN acceleration on CUDA without altering FP32 APIs."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def analytic_kl_div(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu, var) || N(0, I)) without Distribution validation syncs."""
    return 0.5 * (mu.pow(2) + var - var.log() - 1.0).sum(dim=1).mean()


def get_gpu_tensor_budget_bytes(
    device: torch.device,
    vram_fraction: float = _MAX_GPU_FREE_VRAM_FRACTION,
    max_gb: float = _MAX_GPU_RESIDENT_GB,
) -> int:
    """Return dynamic VRAM residency budget: ``min(48 GB, 0.50 * free_vram)``."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except RuntimeError:
        return 0
    return int(min(max_gb * (1024**3), vram_fraction * free_bytes))


def build_adamw(
    params: Any,
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.AdamW:
    """Construct AdamW with ``fused=True`` on CUDA and safe fallback."""
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epoch: int,
    lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create the v1 ConstantLR -> CosineAnnealingLR schedule."""
    lr_milestone = num_epoch // 4
    scheduler1 = torch.optim.lr_scheduler.ConstantLR(
        optimizer=optimizer, total_iters=lr_milestone, factor=1
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, last_epoch=-1, eta_min=lr / 10.0
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[lr_milestone]
    )


def add_cond_mask(
    y: torch.Tensor,
    col_msk_threshold: float,
    vae: Any,
    is_col_wise_mask: bool = False,
    is_multi_codebook: bool = False,
) -> torch.Tensor:
    """Apply random condition masking directly on ``y.device`` and embed."""
    if col_msk_threshold > 0.0:
        if is_col_wise_mask and random.random() < col_msk_threshold:
            y = y.clone()
            idx = random.choice(vae.col_comb)
            y[:, idx] = vae.mask_id
        elif not is_col_wise_mask and random.random() < col_msk_threshold:
            y_flat = y.reshape(-1)
            msk = torch.rand(y_flat.shape[0], device=y_flat.device) < col_msk_threshold
            y_flat = torch.where(msk, vae.mask_id, y_flat)
            y = y_flat.reshape(y.shape[0], y.shape[1])

    if is_multi_codebook:
        c = torch.zeros(
            (y.shape[0], vae.types_of_conds, vae.expand_dim),
            device=y.device,
            dtype=vae.cond_embedder[0].weight.dtype,
        )
        for i in range(vae.types_of_conds):
            c[:, i, :] = vae.cond_embedder[i](y[:, i])
        return c

    y_flat = y.reshape(-1)
    c = vae.cond_embedder(y_flat)
    return c.reshape(y.shape[0], y.shape[1], -1)


def add_cond_mask_with_emb(
    emb: torch.Tensor,
    y: torch.Tensor,
    col_msk_threshold: float,
    vae: Any,
    is_col_wise_mask: bool = False,
    is_multi_codebook: bool = False,
) -> torch.Tensor:
    """Apply condition masking with external text/metadata embeddings."""
    if not is_col_wise_mask and col_msk_threshold > 0.0 and random.random() < col_msk_threshold:
        y_flat = y.reshape(-1)
        msk = torch.rand(y_flat.shape[0], device=y_flat.device) < col_msk_threshold
        y_flat = torch.where(msk, vae.mask_id, y_flat)
        y = y_flat.reshape(y.shape[0], y.shape[1])

    y_flat = y.reshape(-1)
    c = vae.cond_embedder(y_flat).reshape(y.shape[0], y.shape[1], -1)

    embs = torch.split(emb, vae.outer_emb_dim, dim=1)
    emb_proj = torch.cat([vae.txt_emb_proj[i](part) for i, part in enumerate(embs)], dim=1)
    emb_proj = rearrange(emb_proj, "b (c e) -> b c e", c=y.shape[1])

    mask = (y == vae.mask_id).unsqueeze(-1)
    return torch.where(mask, c, emb_proj)


@dataclass
class ResidentTensors:
    """Resident training tensors stored either on GPU (FP32/FP16) or pinned CPU."""

    x: torch.Tensor
    y: torch.Tensor
    emb: torch.Tensor | None
    target_device: torch.device
    tensor_device: torch.device
    on_gpu: bool
    x_dtype: torch.dtype
    pinned: bool = False

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


def _materialise_x_tensor(
    x: np.ndarray | sparse.spmatrix | torch.Tensor,
    dtype: torch.dtype,
    chunk_size: int = 4096,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=dtype)
    if sparse.issparse(x):
        n_obs, n_vars = x.shape
        out = torch.empty((n_obs, n_vars), dtype=dtype)
        np_dtype = np.float16 if dtype == torch.float16 else np.float32
        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            block = np.asarray(x[start:end].toarray(), dtype=np_dtype)
            out[start:end] = torch.from_numpy(block)
        return out
    arr = np.asarray(x)
    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _materialise_long_tensor(y: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(y, torch.Tensor):
        return y.detach().to(dtype=torch.long)
    return torch.from_numpy(np.ascontiguousarray(np.asarray(y, dtype=np.int64)))


def _maybe_pin(tensor: torch.Tensor, enable: bool) -> tuple[torch.Tensor, bool]:
    if not enable or tensor.device.type != "cpu":
        return tensor, False
    try:
        return tensor.pin_memory(), True
    except RuntimeError:
        return tensor, False


def _copy_x_to_gpu_chunked(
    x: np.ndarray | sparse.spmatrix | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Allocate target GPU tensor and copy row chunks without full host expansion."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    n_obs, n_vars = x.shape
    x_gpu = torch.empty((n_obs, n_vars), dtype=dtype, device=device)
    np_dtype = np.float16 if dtype == torch.float16 else np.float32
    is_sp = sparse.issparse(x)
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        block = x[start:end].toarray() if is_sp else x[start:end]
        block_np = np.ascontiguousarray(np.asarray(block, dtype=np_dtype), dtype=np_dtype)
        x_gpu[start:end].copy_(torch.from_numpy(block_np), non_blocking=False)
    return x_gpu


def prepare_resident_tensors(
    x: np.ndarray | sparse.spmatrix | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    emb: np.ndarray | torch.Tensor | None = None,
    max_gb: float = _MAX_GPU_RESIDENT_GB,
    vram_fraction: float = _MAX_GPU_FREE_VRAM_FRACTION,
) -> ResidentTensors:
    """Keep training data resident on GPU (FP32 or FP16) or pinned CPU without DataLoader IPC."""
    n_obs, n_vars = x.shape
    y_cpu = _materialise_long_tensor(y)
    emb_cpu = (
        None
        if emb is None
        else (
            emb.detach().to(dtype=torch.float32)
            if isinstance(emb, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(np.asarray(emb, dtype=np.float32)))
        )
    )

    if device.type == "cuda" and torch.cuda.is_available():
        budget = get_gpu_tensor_budget_bytes(
            device, vram_fraction=vram_fraction, max_gb=max_gb
        )
        y_bytes = y_cpu.numel() * 8
        emb_bytes = (emb_cpu.numel() * 4) if emb_cpu is not None else 0
        fp32_bytes = n_obs * n_vars * 4 + y_bytes + emb_bytes
        fp16_bytes = n_obs * n_vars * 2 + y_bytes + emb_bytes

        for candidate_dtype, needed_bytes in (
            (torch.float32, fp32_bytes),
            (torch.float16, fp16_bytes),
        ):
            if needed_bytes <= budget:
                x_gpu: torch.Tensor | None = None
                try:
                    x_gpu = _copy_x_to_gpu_chunked(
                        x, device=device, dtype=candidate_dtype
                    )
                    y_gpu = y_cpu.to(device)
                    emb_gpu = emb_cpu.to(device) if emb_cpu is not None else None
                    return ResidentTensors(
                        x=x_gpu,
                        y=y_gpu,
                        emb=emb_gpu,
                        target_device=device,
                        tensor_device=device,
                        on_gpu=True,
                        x_dtype=candidate_dtype,
                        pinned=False,
                    )
                except RuntimeError:
                    x_gpu = None
                    torch.cuda.empty_cache()

        # Fallback: pinned-memory CPU tensor + fast direct slice gather.
        x_cpu = _materialise_x_tensor(x, dtype=torch.float32).cpu()
        y_cpu = y_cpu.cpu()
        emb_cpu = emb_cpu.cpu() if emb_cpu is not None else None
        x_cpu, x_pinned = _maybe_pin(x_cpu, True)
        y_cpu, y_pinned = _maybe_pin(y_cpu, True)
        if emb_cpu is not None:
            emb_cpu, emb_pinned = _maybe_pin(emb_cpu, True)
        else:
            emb_pinned = False
        return ResidentTensors(
            x=x_cpu,
            y=y_cpu,
            emb=emb_cpu,
            target_device=device,
            tensor_device=torch.device("cpu"),
            on_gpu=False,
            x_dtype=torch.float32,
            pinned=bool(x_pinned and y_pinned and (emb_cpu is None or emb_pinned)),
        )

    x_dev = _materialise_x_tensor(x, dtype=torch.float32).to(device)
    y_dev = y_cpu.to(device)
    emb_dev = emb_cpu.to(device) if emb_cpu is not None else None
    return ResidentTensors(
        x=x_dev,
        y=y_dev,
        emb=emb_dev,
        target_device=device,
        tensor_device=device,
        on_gpu=device.type != "cpu",
        x_dtype=torch.float32,
        pinned=False,
    )


def iter_resident_batches(
    resident: ResidentTensors,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[tuple[torch.Tensor, ...]]:
    """Yield mini-batches directly from ``ResidentTensors`` (casting FP16 -> FP32 on GPU)."""
    n_samples = resident.n_samples
    limit = (
        (n_samples // batch_size) * batch_size
        if drop_last and n_samples >= batch_size
        else n_samples
    )
    perm = (
        torch.randperm(n_samples, device=resident.tensor_device)
        if shuffle
        else None
    )

    for start in range(0, limit, batch_size):
        end = min(start + batch_size, limit)
        if perm is not None:
            idx = perm[start:end]
            x_b = resident.x[idx]
            y_b = resident.y[idx]
            emb_b = resident.emb[idx] if resident.emb is not None else None
        else:
            x_b = resident.x[start:end]
            y_b = resident.y[start:end]
            emb_b = resident.emb[start:end] if resident.emb is not None else None

        if resident.tensor_device != resident.target_device:
            x_b = x_b.to(resident.target_device, non_blocking=resident.pinned)
            y_b = y_b.to(resident.target_device, non_blocking=resident.pinned)
            if emb_b is not None:
                emb_b = emb_b.to(resident.target_device, non_blocking=resident.pinned)

        if x_b.dtype != torch.float32:
            x_b = x_b.float()
        if y_b.dtype != torch.long:
            y_b = y_b.long()

        if emb_b is not None:
            yield x_b, y_b, emb_b.float() if emb_b.dtype != torch.float32 else emb_b
        else:
            yield x_b, y_b


def extract_tensors_from_dataloader(
    dataloader: Any,
    *,
    is_emb_ary: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Extract raw tensors from a ``TensorDataset`` DataLoader to bypass IPC overhead."""
    dataset = getattr(dataloader, "dataset", None)
    if not isinstance(dataset, TensorDataset) or len(dataset) == 0:
        return None
    tensors = dataset.tensors
    if len(tensors) < 2:
        return None
    emb = tensors[2] if is_emb_ary and len(tensors) >= 3 else None
    return tensors[0], tensors[1], emb


def get_autocast_context(device: torch.device, *, use_amp: bool, prefer_fp16: bool = False):
    """Return an appropriate AMP autocast context or ``nullcontext``."""
    if not (use_amp or prefer_fp16) or device.type != "cuda":
        return nullcontext(), False
    if prefer_fp16:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16), True
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), False
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16), True


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    params: Any,
    *,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
) -> None:
    """Zero gradients, backpropagate, optionally unscale/clip, and step optimizer."""
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()


def resolve_resident_source(
    x: np.ndarray | sparse.spmatrix | torch.Tensor | None,
    y: np.ndarray | torch.Tensor | None,
    dataloader: Any,
    *,
    device: torch.device,
    batch_size: int,
    drop_last: bool,
    emb: np.ndarray | torch.Tensor | None = None,
    is_emb_ary: bool = False,
) -> tuple[ResidentTensors | None, int, bool]:
    """Build ``ResidentTensors`` from explicit arrays/tensors or a ``TensorDataset`` DataLoader."""
    if x is not None and y is not None:
        return (
            prepare_resident_tensors(x, y, device=device, emb=emb),
            batch_size,
            drop_last,
        )
    if dataloader is not None:
        eff_bs = int(getattr(dataloader, "batch_size", None) or batch_size)
        eff_drop = bool(getattr(dataloader, "drop_last", drop_last))
        extracted = extract_tensors_from_dataloader(dataloader, is_emb_ary=is_emb_ary)
        if extracted is not None:
            tx, ty, temb = extracted
            return (
                prepare_resident_tensors(tx, ty, device=device, emb=temb),
                eff_bs,
                eff_drop,
            )
        return None, eff_bs, eff_drop
    raise ValueError("Provide either `(x, y)` arrays/tensors or `dataloader`.")


__all__ = [
    "ResidentTensors",
    "add_cond_mask",
    "add_cond_mask_with_emb",
    "analytic_kl_div",
    "build_adamw",
    "build_warmup_cosine_scheduler",
    "enable_cuda_tf32",
    "extract_tensors_from_dataloader",
    "get_autocast_context",
    "get_gpu_tensor_budget_bytes",
    "iter_resident_batches",
    "optimizer_step",
    "prepare_resident_tensors",
    "resolve_device",
    "resolve_resident_source",
    "seed_everything",
]
