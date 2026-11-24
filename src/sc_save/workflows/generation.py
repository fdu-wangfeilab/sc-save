"""Conditional generation workflows built on ``sc_save.model.SAVE``."""

from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.utils.sparsefuncs import inplace_column_scale
import torch

from sc_save.config import FMConfig, SAVEConfig, TrainConfig, VAEConfig
from sc_save.data.preprocessing import (
    MaxAbsExpressionScaler,
    align_genes,
    find_pad_index,
    pad_to_block_size,
)
from sc_save.model import SAVE
from sc_save.training.runtime import resolve_device, seed_everything

DEFAULT_OUTPUT_DIR = "./outputs/conditional_generation_smoke"
DEFAULT_FORMAL_OUTPUT_DIR = "./outputs/conditional_generation_formal"
DEFAULT_COND_COLS = ["batch", "cell_type"]

FORMAL_SAVE_B_TRAIN: dict[str, Any] = {
    "lr": 0.0001,
    "batch_size": 1024,
    "weight_decay": 0.000025,
    "iter": 3200,
    "epoch": 200,
    "is_lr_scheduler": True,
}

FORMAL_SAVE_B_MODEL: dict[str, Any] = {
    "kl_scale": 0.5,
    "hidden_dim": 64,
    "enc_dim": 64,
    "expand_dim": 512,
    "enc_num_heads": 4,
    "dec_num_heads": 4,
    "enc_depth": 3,
    "dec_depth": 3,
    "enc_affine": True,
    "enc_norm_type": "ln",
    "dec_norm_init": True,
    "dec_blk_type": "adaLN",
    "gene_block_size": 3200,
    "latent_block_size": 3200,
    "is_flash_attn": False,
}


def set_seed(seed: int) -> None:
    seed_everything(seed)


def get_device(device_id: int) -> torch.device:
    return resolve_device(device_id)


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "task"


def resolve_infer_batch_size(
    device: torch.device,
    batch_size: int,
    min_vram_gb: float = 40.0,
    target_batch_size: int = 4096,
) -> int:
    """Use larger inference batch sizes on >=40 GB GPUs without changing training batch_size."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            if total_gb >= min_vram_gb:
                return max(int(batch_size), int(target_batch_size))
        except RuntimeError:
            pass
    return int(batch_size)


def _align_test_to_train_vars(adata_test: ad.AnnData, train_adata: ad.AnnData) -> ad.AnnData:
    if adata_test.var_names.equals(train_adata.var_names):
        return adata_test
    train_names = [str(v) for v in train_adata.var_names]
    test_names = [str(v) for v in adata_test.var_names]
    if set(train_names).issubset(set(test_names)):
        return adata_test[:, train_names].copy()
    test_pad = find_pad_index(adata_test)
    train_pad = find_pad_index(train_adata)
    if test_pad == train_pad and adata_test.n_vars != train_adata.n_vars:
        return align_genes(adata_test[:, :test_pad].copy(), train_names, pad_missing=True)
    if adata_test.n_vars == train_adata.n_vars:
        aligned = adata_test.copy()
        aligned.var_names = train_adata.var_names.copy()
        return aligned
    raise ValueError(
        f"Test dataset gene dimension ({adata_test.n_vars}) cannot align to train ({train_adata.n_vars})"
    )


def load_and_scale_adata(
    data_path: str,
    subset_size: int,
    gene_block_size: int = 0,
) -> tuple[ad.AnnData, ad.AnnData, MaxAbsExpressionScaler, int]:
    """Read an AnnData file, optionally subsample and block-pad, and max-abs scale."""
    import scanpy as sc

    print(f"[pipeline] loading data from: {data_path}")
    adata_raw = sc.read_h5ad(data_path)
    if subset_size > 0:
        adata_ori = adata_raw[:subset_size].copy()
        del adata_raw
        print(f"[pipeline] subset shape: {adata_ori.shape}")
    else:
        adata_ori = adata_raw
        print(f"[pipeline] full shape: {adata_ori.shape}")

    if gene_block_size > 0:
        adata_ori, pad_idx = pad_to_block_size(adata_ori, gene_block_size)
    else:
        pad_idx = find_pad_index(adata_ori)

    scaler = MaxAbsExpressionScaler()
    scaled_x = scaler.fit_transform(adata_ori.X)
    adata = ad.AnnData(X=scaled_x, obs=adata_ori.obs.copy(), var=adata_ori.var.copy())
    return adata_ori, adata, scaler, pad_idx


def build_model_kwargs(seed: int, num_heads: int = 4) -> dict[str, Any]:
    return {
        "hidden_dim": 64,
        "enc_dim": 32,
        "expand_dim": 128,
        "enc_num_heads": num_heads,
        "dec_num_heads": num_heads,
        "enc_depth": 2,
        "dec_depth": 2,
        "enc_affine": True,
        "enc_norm_type": "ln",
        "dec_norm_init": True,
        "dec_blk_type": "adaLN",
        "gene_block_size": 3200,
        "latent_block_size": 3200,
        "is_flash_attn": False,
        "out_layer_dropout": 0.0,
        "is_enc_linear": False,
        "is_dec_linear": False,
        "is_out_moe": False,
        "enc_attn_dropout": 0.0,
        "dec_attn_dropout": 0.0,
        "gpu_num": 1,
        "seed": seed,
        "is_ret_val": False,
        "is_ret_model": True,
        "npy_suffix": "_smoke",
        "is_data_scaled": True,
    }


def build_train_kwargs(vae_epoch: int, batch_size: int) -> dict[str, Any]:
    return {
        "epoch": vae_epoch,
        "batch_size": batch_size,
        "lr": 1e-4,
        "weight_decay": 2.5e-5,
        "iter": 32,
        "is_lr_scheduler": True,
        "kl_scale": 0.5,
    }


def _config_from_model_kwargs(
    model_kwargs: dict[str, Any],
    *,
    n_fm_blocks: int = 5,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> SAVEConfig:
    vae_cfg = VAEConfig.from_dict(model_kwargs)
    expand_dim = int(model_kwargs.get("expand_dim", 512))
    fm_heads = int(model_kwargs.get("fm_num_heads", model_kwargs.get("enc_num_heads", 4)))
    fm_cfg = FMConfig(
        hidden_dim=expand_dim * 2,
        embedding_dim=expand_dim * 2,
        n_blocks=n_fm_blocks,
        num_heads=fm_heads,
        dropout_prob=0.0,
    )
    train_cfg = TrainConfig(
        vae_epochs=int(model_kwargs.get("epoch", 200)),
        fm_epochs=int(model_kwargs.get("epoch", 200)),
        batch_size=int(model_kwargs.get("batch_size", 1024)),
        lr=float(model_kwargs.get("lr", 1e-4)),
        weight_decay=float(model_kwargs.get("weight_decay", 2.5e-5)),
        kl_scale=float(model_kwargs.get("kl_scale", 0.5)),
        is_lr_scheduler=bool(model_kwargs.get("is_lr_scheduler", True)),
        seed=int(model_kwargs.get("seed", 1202)),
    )
    return SAVEConfig(
        vae=vae_cfg,
        fm=fm_cfg,
        train=train_cfg,
        cond_encoding=cond_encoding,
        pad_to_blocks=False,
        scale_data=False,
        seed=int(model_kwargs.get("seed", 1202)),
    )


def create_save_model(
    adata: ad.AnnData,
    device: torch.device,
    cond_cols: list[str],
    seed: int,
    num_heads: int = 4,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> tuple[SAVE, dict[str, Any]]:
    model_kwargs = build_model_kwargs(seed=seed, num_heads=num_heads)
    cfg = _config_from_model_kwargs(
        model_kwargs,
        n_fm_blocks=2,
        cond_encoding=cond_encoding,
    )
    save_model = SAVE(
        adata=adata,
        condition_cols=list(cond_cols),
        config=cfg,
        device=device,
        scale_data=False,
        pad_to_blocks=False,
        cond_encoding=cond_encoding,
    )
    return save_model, model_kwargs


def create_save_model_from_kwargs(
    adata: ad.AnnData,
    device: torch.device,
    cond_cols: list[str],
    model_kwargs: dict[str, Any],
    cond_encoding: Literal["legacy", "offset"] = "legacy",
    *,
    copy_adata: bool = True,
) -> SAVE:
    cfg = _config_from_model_kwargs(model_kwargs, cond_encoding=cond_encoding)
    return SAVE(
        adata=adata,
        condition_cols=list(cond_cols),
        config=cfg,
        device=device,
        scale_data=False,
        pad_to_blocks=False,
        cond_encoding=cond_encoding,
        copy_adata=copy_adata,
    )


def attach_fm_model(
    save_model: SAVE,
    seq_length: int,
    cond_cols: list[str],
    num_heads: int = 4,
) -> SAVE:
    fm_cfg = FMConfig(
        hidden_dim=save_model.expand_dim * 2,
        embedding_dim=save_model.expand_dim * 2,
        n_blocks=2,
        num_heads=num_heads,
        dropout_prob=0.0,
    )
    save_model.init_fm(fm_cfg, seq_length=seq_length)
    return save_model


def attach_formal_fm_model(
    save_model: SAVE,
    seq_length: int,
    cond_cols: list[str],
    num_heads: int = 4,
    n_blocks: int = 5,
) -> SAVE:
    fm_cfg = FMConfig(
        hidden_dim=save_model.expand_dim * 2,
        embedding_dim=save_model.expand_dim * 2,
        n_blocks=n_blocks,
        num_heads=num_heads,
        dropout_prob=0.0,
    )
    save_model.init_fm(fm_cfg, seq_length=seq_length)
    return save_model


def train_vae(save_model: SAVE, train_kwargs: dict[str, Any]) -> float:
    print("[pipeline] training VAE")
    start = time.time()
    save_model.fit_vae(
        epochs=int(train_kwargs["epoch"]),
        batch_size=int(train_kwargs["batch_size"]),
        lr=float(train_kwargs["lr"]),
        weight_decay=float(train_kwargs["weight_decay"]),
        kl_scale=float(train_kwargs.get("kl_scale", 0.5)),
        col_mask_prob=1.0,
        is_col_wise_mask=False,
    )
    return time.time() - start


def train_fm(save_model: SAVE, fm_epoch: int, batch_size: int) -> float:
    print("[pipeline] training FM")
    start = time.time()
    save_model.fit_fm(
        epochs=int(fm_epoch),
        batch_size=int(batch_size),
        lr=1e-4,
        weight_decay=2.5e-5,
        col_mask_prob=0.0,
    )
    return time.time() - start


def train_vae_formal(
    save_model: SAVE,
    model_kwargs: dict[str, Any],
    sample_mask_prob: float = 0.4,
    col_msk_threshold: float = 1.0,
) -> float:
    print("[pipeline] training formal VAE")
    start = time.time()
    save_model.fit_vae(
        epochs=int(model_kwargs["epoch"]),
        batch_size=int(model_kwargs["batch_size"]),
        lr=float(model_kwargs["lr"]),
        weight_decay=float(model_kwargs["weight_decay"]),
        kl_scale=float(model_kwargs.get("kl_scale", 0.5)),
        col_mask_prob=float(col_msk_threshold),
        is_col_wise_mask=False,
        seed=int(model_kwargs.get("seed", 1202)),
    )
    return time.time() - start


def train_fm_formal(
    save_model: SAVE,
    model_kwargs: dict[str, Any],
    sample_mask_prob: float = 0.4,
    col_msk_threshold: float = 0.0,
) -> float:
    print("[pipeline] training formal FM")
    start = time.time()
    save_model.fit_fm(
        epochs=int(model_kwargs["epoch"]),
        batch_size=int(model_kwargs["batch_size"]),
        lr=float(model_kwargs["lr"]),
        weight_decay=float(model_kwargs["weight_decay"]),
        col_mask_prob=float(col_msk_threshold),
        seed=int(model_kwargs.get("seed", 1202)),
    )
    return time.time() - start


def load_formal_settings(num_heads: int = 4) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    vae_setting = dict(FORMAL_SAVE_B_MODEL)
    train_setting = dict(FORMAL_SAVE_B_TRAIN)
    vae_setting["enc_num_heads"] = num_heads
    vae_setting["dec_num_heads"] = num_heads
    model_kwargs: dict[str, Any] = {
        "gpu_num": 1,
        "seed": 1202,
        "is_ret_val": False,
        "is_ret_model": True,
        "npy_suffix": "_SAVE-B-release",
        "is_data_scaled": True,
        "fm_num_heads": num_heads,
    }
    model_kwargs.update(vae_setting)
    model_kwargs.update(train_setting)
    model_kwargs["kl_scale"] = 0.5
    return model_kwargs, vae_setting, train_setting


def build_formal_ckpt_paths(
    out_dir: Path,
    vae_setting: dict[str, Any],
    gene_block_size: int,
    num_heads: int,
    task_name: str = "hca_subsampled_20k",
    ckpt_dir: Path | None = None,
) -> tuple[Path, Path]:
    ckpt_root = ckpt_dir if ckpt_dir is not None else out_dir / "ckpt"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_name(task_name)
    vae_ckpt_path = ckpt_root / (
        f"{stem}_VAE_{vae_setting['expand_dim']}_enc"
        f"{vae_setting['enc_dim']}_GBS{gene_block_size}_head{num_heads}.pt"
    )
    fm_ckpt_path = ckpt_root / (
        f"{stem}_fm_{vae_setting['expand_dim']}_enc"
        f"{vae_setting['enc_dim']}_GBS{gene_block_size}_head{num_heads}.pt"
    )
    return vae_ckpt_path, fm_ckpt_path


def run_formal_generation(
    save_model: SAVE,
    adata_ori: ad.AnnData,
    scaler: Any,
    pad_idx: int,
    batch_size: int,
    seq_length: int,
    cond_cols: list[str],
    out_dir: Path,
    guidance_weights: float = 2.0,
    sample_steps: int = 20,
    split_targets: list[tuple[ad.AnnData, Any]] | None = None,
) -> dict[str, Any]:
    """Generate cells matching each split's condition distribution in ONE batched GPU pass."""
    if split_targets is None:
        split_targets = [(adata_ori, scaler)]

    split_gen_list: list[ad.AnnData] = []
    split_target_list: list[ad.AnnData] = []
    total_z_fm: list[np.ndarray] = []
    total_z_vae: list[np.ndarray] = []
    total_groups = 0

    start = time.time()
    infer_batch_size = resolve_infer_batch_size(save_model.device, batch_size)
    for split_adata_ori, split_scaler in split_targets:
        if split_adata_ori.n_obs == 0:
            continue

        cond_info = (
            split_adata_ori.obs[list(cond_cols)]
            .value_counts()
            .rename_axis(list(cond_cols))
            .reset_index(name="count")
        )
        cond_info = cond_info[cond_info["count"] > 0].reset_index(drop=True)
        total_groups += int(cond_info.shape[0])

        split_bs = min(infer_batch_size, max(1, int(split_adata_ori.n_obs)))
        print(
            f"[pipeline] batched generation for {cond_info.shape[0]} condition groups "
            f"({split_adata_ori.n_obs} cells, batch_size={split_bs})"
        )

        # Validate condition labels eagerly before generation.
        save_model.vocab.encode_obs(split_adata_ori.obs)

        z_from_vae = save_model.encode(
            split_adata_ori,
            with_cond=False,
            latent_pos=0,
            batch_size=split_bs,
        )
        split_gen, gen_z_fm = save_model.generate_like(
            split_adata_ori,
            guidance_weight=guidance_weights,
            steps=sample_steps,
            batch_size=split_bs,
            return_latents=True,
            inverse_scale=False,
            trim_pad=True,
        )

        active_scaler = split_scaler if split_scaler is not None else save_model.scaler
        if active_scaler is not None:
            scale_trim = active_scaler.scale_[:pad_idx].astype(np.float32, copy=False)
            if sparse.issparse(split_gen.X):
                out_sp = split_gen.X.tocsr(copy=True)
                out_sp.data = out_sp.data.astype(np.float32, copy=True)
                inplace_column_scale(out_sp, scale_trim)
                split_gen.X = out_sp
            else:
                x_arr = np.asarray(split_gen.X, dtype=np.float32)
                if not x_arr.flags.writeable:
                    x_arr = x_arr.copy()
                x_arr *= scale_trim[None, :]
                split_gen.X = x_arr

        split_gen.obs_names = [
            f"{len(split_gen_list)}_{k}_gen" for k in range(split_gen.n_obs)
        ]
        split_gen_list.append(split_gen)
        split_target_list.append(split_adata_ori[:, :pad_idx].copy())
        total_z_fm.append(gen_z_fm)
        total_z_vae.append(z_from_vae)

    gen_trim = (
        ad.concat(split_gen_list) if len(split_gen_list) > 1 else split_gen_list[0]
    )
    target_trim = (
        ad.concat(split_target_list) if len(split_target_list) > 1 else split_target_list[0]
    )
    total_z_fm_arr = np.concatenate(total_z_fm, axis=0)
    total_z_vae_arr = np.concatenate(total_z_vae, axis=0)

    gen_trim.obs["type"] = "gen"
    gen_trim.obs.index = [f"{idx}_gen" for idx in range(gen_trim.shape[0])]
    target_trim.obs["type"] = "ori"

    gen_path = out_dir / "generation.h5ad"
    target_path = out_dir / "target.h5ad"
    fm_latent_path = out_dir / "gen_z_fm.npy"
    vae_latent_path = out_dir / "target_z_vae.npy"

    gen_trim.write_h5ad(gen_path)
    target_trim.write_h5ad(target_path)
    np.save(fm_latent_path, total_z_fm_arr)
    np.save(vae_latent_path, total_z_vae_arr)

    gen_shape = list(gen_trim.shape)
    target_shape = list(target_trim.shape)
    gen_z_shape = list(total_z_fm_arr.shape)
    vae_z_shape = list(total_z_vae_arr.shape)

    del (
        split_gen_list,
        split_target_list,
        gen_trim,
        target_trim,
        total_z_fm,
        total_z_vae,
        total_z_fm_arr,
        total_z_vae_arr,
    )
    gc.collect()

    return {
        "generation_seconds": round(time.time() - start, 3),
        "condition_group_count": total_groups,
        "gen_shape": gen_shape,
        "target_shape": target_shape,
        "gen_z_shape": gen_z_shape,
        "vae_z_shape": vae_z_shape,
        "generation_path": str(gen_path),
        "target_path": str(target_path),
        "gen_z_path": str(fm_latent_path),
        "vae_z_path": str(vae_latent_path),
    }


def choose_target_condition(
    adata_ori: ad.AnnData,
    cond_cols: list[str],
    generate_count: int,
) -> tuple[dict[str, Any], int, ad.AnnData]:
    cond_info = (
        adata_ori.obs[cond_cols]
        .value_counts()
        .rename_axis(cond_cols)
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    target_cond = cond_info.loc[0, cond_cols].to_dict()
    target_count = min(generate_count, int(cond_info.loc[0, "count"]))
    target_data = adata_ori[
        np.logical_and.reduce([adata_ori.obs[k] == v for k, v in target_cond.items()])
    ].copy()
    return target_cond, target_count, target_data


def run_hca_smoke(
    data_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    subset_size: int = 512,
    device_id: int = 0,
    seed: int = 1202,
    vae_epoch: int = 1,
    fm_epoch: int = 1,
    batch_size: int = 64,
    generate_count: int = 8,
    guidance_weights: float = 2.0,
    sample_steps: int = 5,
    cond_cols: list[str] | None = None,
    num_heads: int = 4,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> dict[str, Any]:
    """Run a compact smoke check for conditional generation."""
    if cond_cols is None:
        cond_cols = list(DEFAULT_COND_COLS)

    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device_id)
    adata_ori, adata, scaler, pad_idx = load_and_scale_adata(data_path, subset_size)
    save_model, model_kwargs = create_save_model(
        adata=adata,
        device=device,
        cond_cols=cond_cols,
        seed=seed,
        num_heads=num_heads,
        cond_encoding=cond_encoding,
    )
    train_kwargs = build_train_kwargs(vae_epoch=vae_epoch, batch_size=batch_size)

    vae_seconds = train_vae(save_model, train_kwargs)

    seq_length = adata_ori.shape[1] // model_kwargs["gene_block_size"]
    print("[pipeline] initializing FM")
    attach_fm_model(
        save_model,
        seq_length=seq_length,
        cond_cols=cond_cols,
        num_heads=num_heads,
    )
    fm_seconds = train_fm(save_model, fm_epoch=fm_epoch, batch_size=batch_size)

    target_cond, target_count, target_data = choose_target_condition(
        adata_ori=adata_ori,
        cond_cols=cond_cols,
        generate_count=generate_count,
    )
    print(f"[pipeline] target_cond: {target_cond}")
    print(f"[pipeline] target cells in subset: {target_data.shape[0]}")

    z_from_vae = save_model.encode(
        target_data,
        with_cond=False,
        latent_pos=0,
        batch_size=min(batch_size, max(1, target_data.n_obs)),
    )

    print("[pipeline] generating with FM")
    gen_start = time.time()
    gen_adata, gen_z_fm = save_model.generate(
        conditions=target_cond,
        n_cells=target_count,
        guidance_weight=guidance_weights,
        steps=sample_steps,
        batch_size=min(batch_size, target_count),
        return_latents=True,
        inverse_scale=False,
        trim_pad=False,
    )
    gen_seconds = time.time() - gen_start

    gen_adata.X = scaler.inverse_transform(gen_adata.X)
    gen_adata = gen_adata[:, :pad_idx].copy()
    target_trim = target_data[:, :pad_idx].copy()

    gen_path = out_dir / "generated_smoke.h5ad"
    target_path = out_dir / "target_smoke.h5ad"
    fm_latent_path = out_dir / "gen_z_fm.npy"
    vae_latent_path = out_dir / "target_z_vae.npy"
    report_path = out_dir / "report.json"

    gen_adata.write_h5ad(gen_path)
    target_trim.write_h5ad(target_path)
    np.save(fm_latent_path, gen_z_fm)
    np.save(vae_latent_path, z_from_vae)

    report = {
        "data_path": data_path,
        "subset_shape": list(adata_ori.shape),
        "trimmed_gene_dim": int(pad_idx),
        "device": str(device),
        "cond_encoding": cond_encoding,
        "target_cond": target_cond,
        "target_count": int(target_count),
        "gen_shape": list(gen_adata.shape),
        "gen_z_shape": list(gen_z_fm.shape),
        "vae_z_shape": list(z_from_vae.shape),
        "vae_seconds": round(vae_seconds, 3),
        "fm_seconds": round(fm_seconds, 3),
        "gen_seconds": round(gen_seconds, 3),
        "model_kwargs": model_kwargs,
        "train_kwargs": train_kwargs,
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("[pipeline] done")
    print(json.dumps(report, indent=2))
    print(f"[pipeline] saved generated data to: {gen_path}")
    print(f"[pipeline] saved target data to: {target_path}")
    print(f"[pipeline] saved report to: {report_path}")
    return report


def run_hca_formal(
    data_path: str,
    output_dir: str = DEFAULT_FORMAL_OUTPUT_DIR,
    device_id: int = 0,
    cond_cols: list[str] | None = None,
    num_heads: int = 4,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> dict[str, Any]:
    """Run the formal HCA dual-condition SAVE + FM preset."""
    return run_formal_conditional_generation(
        data_path=data_path,
        output_dir=output_dir,
        device_id=device_id,
        cond_cols=cond_cols,
        task_name="hca_subsampled_20k",
        gene_block_size=3200,
        num_heads=num_heads,
        cond_encoding=cond_encoding,
    )


def run_formal_conditional_generation(
    data_path: str,
    output_dir: str = DEFAULT_FORMAL_OUTPUT_DIR,
    device_id: int = 0,
    cond_cols: list[str] | None = None,
    task_name: str = "conditional_generation",
    gene_block_size: int = 3200,
    latent_block_size: int | None = None,
    vae_epoch: int = 200,
    fm_epoch: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-4,
    weight_decay: float = 2.5e-5,
    kl_scale: float = 0.5,
    expand_dim: int = 512,
    enc_dim: int = 64,
    hidden_dim: int = 64,
    num_heads: int = 4,
    n_fm_blocks: int = 5,
    guidance_weights: float = 2.0,
    sample_steps: int = 20,
    sample_mask_prob: float | None = None,
    vae_col_msk_threshold: float = 1.0,
    fm_col_msk_threshold: float = 0.0,
    test_data_path: str | None = None,
    eval_split: str = "auto",
    seed: int = 1202,
    subset_size: int = 0,
    ckpt_dir: str | None = None,
    overwrite_ckpt: bool = False,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> dict[str, Any]:
    """Run generalized SAVE + FM formal conditional generation for Tables 1-3."""
    import scanpy as sc

    if cond_cols is None:
        cond_cols = list(DEFAULT_COND_COLS)
    else:
        cond_cols = list(cond_cols)

    if latent_block_size is None:
        latent_block_size = gene_block_size
    if sample_mask_prob is None:
        effective_sample_mask_prob = 0.0 if task_name in {"cancer_lung", "lung_cancer"} else 0.4
    else:
        effective_sample_mask_prob = float(sample_mask_prob)

    if eval_split not in {"auto", "train", "test", "all"}:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device_id)

    adata_train_ori, adata_train, train_scaler, pad_idx = load_and_scale_adata(
        data_path=data_path,
        subset_size=subset_size,
        gene_block_size=gene_block_size,
    )
    for col in cond_cols:
        if col not in adata_train_ori.obs.columns:
            raise ValueError(f"Training AnnData missing required obs column: '{col}'")

    effective_batch_size = min(batch_size, max(1, adata_train_ori.n_obs))
    model_kwargs, vae_setting, train_setting = load_formal_settings(num_heads=num_heads)
    vae_setting.update(
        {
            "kl_scale": kl_scale,
            "hidden_dim": hidden_dim,
            "enc_dim": enc_dim,
            "expand_dim": expand_dim,
            "gene_block_size": gene_block_size,
            "latent_block_size": latent_block_size,
        }
    )
    train_setting.update(
        {
            "lr": lr,
            "batch_size": effective_batch_size,
            "weight_decay": weight_decay,
        }
    )
    model_kwargs.update(vae_setting)
    model_kwargs.update(train_setting)
    model_kwargs["seed"] = seed

    save_model = create_save_model_from_kwargs(
        adata=adata_train,
        device=device,
        cond_cols=cond_cols,
        model_kwargs=model_kwargs,
        cond_encoding=cond_encoding,
        copy_adata=False,
    )
    del adata_train
    train_input_shape = list(adata_train_ori.shape)
    train_n_vars = int(adata_train_ori.shape[1])

    ckpt_root = Path(ckpt_dir) if ckpt_dir is not None else out_dir / "ckpt"
    vae_ckpt_path, fm_ckpt_path = build_formal_ckpt_paths(
        out_dir=out_dir,
        vae_setting=vae_setting,
        gene_block_size=gene_block_size,
        num_heads=num_heads,
        task_name=task_name,
        ckpt_dir=ckpt_root,
    )

    vae_train_kwargs = dict(model_kwargs)
    vae_train_kwargs["epoch"] = vae_epoch
    if vae_ckpt_path.exists() and not overwrite_ckpt:
        print(f"[pipeline] loading formal VAE checkpoint: {vae_ckpt_path}")
        save_model.load_vae_ckpt(vae_ckpt_path)
        vae_seconds = 0.0
        vae_loaded = True
    else:
        vae_seconds = train_vae_formal(
            save_model,
            model_kwargs=vae_train_kwargs,
            sample_mask_prob=effective_sample_mask_prob,
            col_msk_threshold=vae_col_msk_threshold,
        )
        save_model.save_vae_ckpt(vae_ckpt_path)
        vae_loaded = False

    seq_length = train_n_vars // gene_block_size
    if seq_length < 1 or train_n_vars % gene_block_size != 0:
        raise ValueError(
            f"Input gene count {train_n_vars} must be a positive multiple of gene_block_size={gene_block_size}"
        )

    attach_formal_fm_model(
        save_model,
        seq_length=seq_length,
        cond_cols=cond_cols,
        num_heads=num_heads,
        n_blocks=n_fm_blocks,
    )

    fm_train_kwargs = dict(model_kwargs)
    fm_train_kwargs["epoch"] = fm_epoch
    if fm_ckpt_path.exists() and not overwrite_ckpt:
        print(f"[pipeline] loading formal FM checkpoint: {fm_ckpt_path}")
        save_model.load_fm_ckpt(fm_ckpt_path)
        fm_seconds = 0.0
        fm_loaded = True
    else:
        fm_seconds = train_fm_formal(
            save_model,
            model_kwargs=fm_train_kwargs,
            sample_mask_prob=effective_sample_mask_prob,
            col_msk_threshold=fm_col_msk_threshold,
        )
        save_model.save_fm_ckpt(fm_ckpt_path)
        fm_loaded = False

    # Release scaled training matrix before loading held-out test splits & generating.
    save_model.adata = None
    gc.collect()

    split_targets: list[tuple[ad.AnnData, MaxAbsExpressionScaler]] = [
        (adata_train_ori, train_scaler)
    ]
    resolved_split = eval_split
    if test_data_path is not None:
        adata_test_raw = sc.read_h5ad(test_data_path)
        for col in cond_cols:
            if col not in adata_test_raw.obs.columns:
                raise ValueError(f"Test AnnData missing required obs column: '{col}'")
        adata_test_ori = _align_test_to_train_vars(adata_test_raw, adata_train_ori)
        del adata_test_raw
        test_scaler = MaxAbsExpressionScaler().fit(adata_test_ori.X)

        if resolved_split == "auto":
            resolved_split = (
                "all"
                if (task_name in {"cancer_lung", "lung_cancer"} or len(cond_cols) >= 3)
                else "test"
            )

        if resolved_split == "all":
            split_targets = [
                (adata_train_ori, train_scaler),
                (adata_test_ori, test_scaler),
            ]
        elif resolved_split == "test":
            split_targets = [(adata_test_ori, test_scaler)]
        elif resolved_split == "train":
            split_targets = [(adata_train_ori, train_scaler)]
    elif resolved_split == "auto":
        resolved_split = "train"
    elif resolved_split in {"test", "all"}:
        raise ValueError(f"eval_split='{resolved_split}' requires test_data_path to be set")

    if resolved_split == "test":
        del adata_train_ori
        gc.collect()

    gen_report = run_formal_generation(
        save_model=save_model,
        adata_ori=split_targets[0][0],
        scaler=split_targets[0][1],
        pad_idx=pad_idx,
        batch_size=effective_batch_size,
        seq_length=seq_length,
        cond_cols=cond_cols,
        out_dir=out_dir,
        guidance_weights=guidance_weights,
        sample_steps=sample_steps,
        split_targets=split_targets,
    )

    report = {
        "task_name": task_name,
        "data_path": data_path,
        "test_data_path": test_data_path,
        "eval_split": resolved_split,
        "input_shape": train_input_shape,
        "trimmed_gene_dim": int(pad_idx),
        "device": str(device),
        "condition_cols": cond_cols,
        "cond_encoding": cond_encoding,
        "vae_loaded_from_ckpt": vae_loaded,
        "fm_loaded_from_ckpt": fm_loaded,
        "vae_seconds": round(vae_seconds, 3),
        "fm_seconds": round(fm_seconds, 3),
        "vae_ckpt_path": str(vae_ckpt_path),
        "fm_ckpt_path": str(fm_ckpt_path),
        "vae_epoch": vae_epoch,
        "fm_epoch": fm_epoch,
        "sample_mask_prob": effective_sample_mask_prob,
        "guidance_weights": guidance_weights,
        "sample_steps": sample_steps,
        "gene_block_size": gene_block_size,
        "latent_block_size": latent_block_size,
        "num_heads": num_heads,
        "model_kwargs": model_kwargs,
        "vae_setting": vae_setting,
        "train_setting": train_setting,
    }
    report.update(gen_report)

    report_path = out_dir / "report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"[pipeline] generalized formal run ({task_name}) done -> {report_path}")
    return report


__all__ = [
    "DEFAULT_COND_COLS",
    "DEFAULT_FORMAL_OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "FORMAL_SAVE_B_MODEL",
    "FORMAL_SAVE_B_TRAIN",
    "attach_fm_model",
    "attach_formal_fm_model",
    "build_formal_ckpt_paths",
    "build_model_kwargs",
    "build_train_kwargs",
    "choose_target_condition",
    "create_save_model",
    "create_save_model_from_kwargs",
    "get_device",
    "load_and_scale_adata",
    "load_formal_settings",
    "resolve_infer_batch_size",
    "run_formal_conditional_generation",
    "run_formal_generation",
    "run_hca_formal",
    "run_hca_smoke",
    "set_seed",
    "train_fm",
    "train_fm_formal",
    "train_vae",
    "train_vae_formal",
]
