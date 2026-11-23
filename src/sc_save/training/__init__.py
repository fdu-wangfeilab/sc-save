"""Training runtime, Stage-1 VAE trainer, and Stage-2 FM trainer."""

from sc_save.training.fm_trainer import train_fm
from sc_save.training.runtime import (
    ResidentTensors,
    add_cond_mask,
    add_cond_mask_with_emb,
    analytic_kl_div,
    build_adamw,
    build_warmup_cosine_scheduler,
    enable_cuda_tf32,
    extract_tensors_from_dataloader,
    get_autocast_context,
    get_gpu_tensor_budget_bytes,
    iter_resident_batches,
    optimizer_step,
    prepare_resident_tensors,
    resolve_device,
    resolve_resident_source,
    seed_everything,
)
from sc_save.training.vae_trainer import train_vae

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
    "train_fm",
    "train_vae",
]
