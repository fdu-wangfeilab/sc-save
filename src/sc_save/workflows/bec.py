"""Batch effect correction (BEC) workflow matching the ICLR 2026 LDM_raw protocol."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

from sc_save.config import SAVEConfig
from sc_save.data.preprocessing import MaxAbsExpressionScaler, pad_to_block_size
from sc_save.eval.bec import (
    _compat_scanpy_pca_for_scib,
    _pandas_value_counts_compat,
    compute_scib_summary,
    scib_eval,
)
from sc_save.model import SAVE
from sc_save.training.runtime import resolve_device

DEFAULT_BEC_OUTPUT_DIR = "./outputs/bec_integration"

SAVE_B_PERTURB_TRAIN_CONFIG: dict[str, Any] = {
    "lr": 0.0001,
    "batch_size": 1024,
    "weight_decay": 0.000025,
    "iter": 3200,
    "epoch": 200,
    "is_lr_scheduler": True,
    "sample_mask_prob": 0.0,
    "col_msk_threshold": 0.6,
    "is_col_wise_mask": False,
    "recon_criterion": "bce",
}

SAVE_B_PERTURB_MODEL_CONFIG: dict[str, Any] = {
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
    "is_initialize": True,
    "gene_block_size": 400,
    "latent_block_size": 400,
    "is_flash_attn": False,
}


def get_device(device_id: int) -> torch.device:
    return resolve_device(device_id)


def resolve_scib_workers(scib_workers: int = 0) -> int:
    """Resolve worker process count for CPU scIB evaluation (0 = auto)."""
    if int(scib_workers) <= 0:
        return min(8, os.cpu_count() or 1)
    return max(1, int(scib_workers))


def resolve_transfer_batch_size(device: torch.device, transfer_batch_size: int) -> int:
    """Increase GPU transfer batch size on >=40 GB GPUs for better throughput."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            if total_gb >= 40.0:
                return max(int(transfer_batch_size), 2048)
        except RuntimeError:
            pass
    return int(transfer_batch_size)


_SCIB_WORKER_STATE: dict[str, Any] = {}


def _init_scib_worker(eval_h5ad_path: str, batch_key: str, cell_type_key: str) -> None:
    adata = sc.read_h5ad(eval_h5ad_path)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[cell_type_key] = adata.obs[cell_type_key].astype("category")
    _SCIB_WORKER_STATE["adata"] = adata
    _SCIB_WORKER_STATE["batch_key"] = batch_key
    _SCIB_WORKER_STATE["cell_type_key"] = cell_type_key


def _scib_eval_worker(target_batch: str, npy_path: str) -> tuple[str, dict[str, float]]:
    adata = _SCIB_WORKER_STATE["adata"]
    batch_key = _SCIB_WORKER_STATE["batch_key"]
    cell_type_key = _SCIB_WORKER_STATE["cell_type_key"]
    embed_key = f"transfer_to_{target_batch}"
    adata.obsm[embed_key] = np.load(npy_path)
    try:
        metrics = scib_eval(
            adata,
            batch_key=batch_key,
            cell_type_key=cell_type_key,
            embed_key=embed_key,
        )
    finally:
        adata.obsm.pop(embed_key, None)
    return target_batch, metrics


def _sanitize_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "batch"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _scale_max_abs(adata: ad.AnnData) -> ad.AnnData:
    scaled = adata.copy()
    x = scaled.X.toarray() if sparse.issparse(scaled.X) else np.asarray(scaled.X)
    x = np.asarray(x, dtype=np.float32)
    scaler = MaxAbsExpressionScaler()
    scaled.X = np.asarray(scaler.fit_transform(x), dtype=np.float32)
    return scaled


def _prepare_eval_adata(adata: ad.AnnData, batch_key: str, cell_type_key: str) -> ad.AnnData:
    eval_adata = adata.copy()
    if sparse.issparse(eval_adata.X):
        eval_adata.X = eval_adata.X.toarray()
    eval_adata.X = np.asarray(eval_adata.X, dtype=np.float32)
    eval_adata.obs[batch_key] = eval_adata.obs[batch_key].astype("category")
    eval_adata.obs[cell_type_key] = eval_adata.obs[cell_type_key].astype("category")
    return eval_adata


def _transfer_to_target_batch(
    save_model: SAVE,
    adata_scaled: ad.AnnData,
    target_batch: str,
    batch_key: str,
    cell_type_key: str,
    cond_cols: list[str],
    pca_components: int,
    transfer_batch_size: int,
) -> np.ndarray:
    """Transfer all non-target cells to ``target_batch`` in one batched pass and fit PCA."""
    target_mask = adata_scaled.obs[batch_key].astype(str) == str(target_batch)
    target_adata = adata_scaled[target_mask].copy()
    other_data = adata_scaled[~target_mask].copy()

    merge_parts: list[ad.AnnData] = [target_adata]
    if other_data.n_obs > 0:
        effective_bs = (
            min(transfer_batch_size, max(1, other_data.n_obs))
            if transfer_batch_size > 0
            else None
        )
        if hasattr(save_model, "transfer"):
            trans = save_model.transfer(
                other_data,
                target_cond={batch_key: target_batch},
                input_cond="obs",
                batch_size=effective_bs,
                inverse_scale=False,
                trim_pad=False,
            )
            if isinstance(trans, tuple):
                trans = trans[0]
        else:
            raise AttributeError("save_model does not expose `transfer`.")

        trans_x = trans.X.toarray() if sparse.issparse(trans.X) else np.asarray(trans.X)
        trans_adata = ad.AnnData(
            X=np.asarray(trans_x, dtype=np.float32),
            var=other_data.var.copy(),
            obs=other_data.obs.copy(),
        )
        trans_adata.obs["type"] = f"SAVE_trans_{target_batch}"
        merge_parts.append(trans_adata)

    if len(merge_parts) == 1:
        merge_data = target_adata.copy()
    else:
        merge_data = ad.concat(merge_parts, join="outer")

    merge_data.obs[batch_key] = merge_data.obs[batch_key].astype("category")
    merge_data.obs[cell_type_key] = merge_data.obs[cell_type_key].astype("category")
    if sparse.issparse(merge_data.X):
        merge_data.X = merge_data.X.toarray()
    merge_data.X = np.asarray(merge_data.X, dtype=np.float32)

    n_pcs = min(pca_components, max(1, min(merge_data.n_obs, merge_data.n_vars) - 1))
    sc.pp.pca(merge_data, n_comps=n_pcs)

    if adata_scaled.obs_names.is_unique and merge_data.obs_names.is_unique:
        latent = merge_data[adata_scaled.obs_names].obsm["X_pca"]
    else:
        pos_map = {name: pos for pos, name in enumerate(merge_data.obs_names)}
        indices = [pos_map[name] for name in adata_scaled.obs_names]
        latent = merge_data.obsm["X_pca"][indices]

    return np.asarray(latent, dtype=np.float32)


def run_bec_integration(
    data_path: str,
    task_name: str = "bec_task",
    output_dir: str = DEFAULT_BEC_OUTPUT_DIR,
    ckpt_dir: str | None = None,
    device_id: int = 0,
    batch_key: str = "batch",
    cell_type_key: str = "cell_type",
    cond_cols: list[str] | None = None,
    epoch: int = SAVE_B_PERTURB_TRAIN_CONFIG["epoch"],
    batch_size: int = SAVE_B_PERTURB_TRAIN_CONFIG["batch_size"],
    lr: float = SAVE_B_PERTURB_TRAIN_CONFIG["lr"],
    weight_decay: float = SAVE_B_PERTURB_TRAIN_CONFIG["weight_decay"],
    is_lr_scheduler: bool = SAVE_B_PERTURB_TRAIN_CONFIG["is_lr_scheduler"],
    kl_scale: float = SAVE_B_PERTURB_MODEL_CONFIG["kl_scale"],
    sample_mask_prob: float = SAVE_B_PERTURB_TRAIN_CONFIG["sample_mask_prob"],
    col_msk_threshold: float = SAVE_B_PERTURB_TRAIN_CONFIG["col_msk_threshold"],
    hidden_dim: int = SAVE_B_PERTURB_MODEL_CONFIG["hidden_dim"],
    enc_dim: int = SAVE_B_PERTURB_MODEL_CONFIG["enc_dim"],
    expand_dim: int = SAVE_B_PERTURB_MODEL_CONFIG["expand_dim"],
    enc_num_heads: int = SAVE_B_PERTURB_MODEL_CONFIG["enc_num_heads"],
    dec_num_heads: int = SAVE_B_PERTURB_MODEL_CONFIG["dec_num_heads"],
    enc_depth: int = SAVE_B_PERTURB_MODEL_CONFIG["enc_depth"],
    dec_depth: int = SAVE_B_PERTURB_MODEL_CONFIG["dec_depth"],
    gene_block_size: int = SAVE_B_PERTURB_MODEL_CONFIG["gene_block_size"],
    latent_block_size: int | None = None,
    pca_components: int = 30,
    transfer_batch_size: int = 256,
    scib_workers: int = 0,
    max_target_batches: int = 0,
    seed: int = 1202,
    overwrite_ckpt: bool = False,
    run_scib: bool = True,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> dict[str, Any]:
    """Train/load SAVE-B-perturb on GBS=400 data, transfer to every target batch, and evaluate scIB."""
    if cond_cols is None:
        cond_cols = [batch_key, cell_type_key]
    if latent_block_size is None:
        latent_block_size = gene_block_size

    out_path = Path(output_dir)
    emb_dir = out_path / "embeddings"
    ckpt_root = Path(ckpt_dir) if ckpt_dir is not None else out_path / "checkpoints"
    emb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    device = get_device(device_id)
    raw_adata = sc.read_h5ad(data_path)
    for col in cond_cols:
        if col not in raw_adata.obs.columns:
            raise ValueError(f"Input AnnData missing required obs column: '{col}'")

    raw_adata, _ = pad_to_block_size(raw_adata, gene_block_size)
    eval_adata = _prepare_eval_adata(raw_adata, batch_key=batch_key, cell_type_key=cell_type_key)
    scaled_adata = _scale_max_abs(eval_adata)

    cfg = SAVEConfig.save_b_perturb(
        gene_block_size=gene_block_size,
        latent_block_size=latent_block_size,
        hidden_dim=hidden_dim,
        enc_dim=enc_dim,
        expand_dim=expand_dim,
        num_heads=enc_num_heads,
        enc_depth=enc_depth,
        dec_depth=dec_depth,
        vae_epochs=epoch,
        fm_epochs=0,
        batch_size=min(batch_size, max(1, scaled_adata.n_obs)),
        lr=lr,
        weight_decay=weight_decay,
        kl_scale=kl_scale,
        vae_col_mask_prob=col_msk_threshold,
        cond_encoding=cond_encoding,
        seed=seed,
    )
    cfg.vae.dec_num_heads = dec_num_heads
    cfg.train.is_lr_scheduler = bool(is_lr_scheduler)
    cfg.pad_to_blocks = False
    cfg.scale_data = False

    save_model = SAVE(
        adata=scaled_adata.copy(),
        condition_cols=list(cond_cols),
        config=cfg,
        device=device,
        scale_data=False,
        pad_to_blocks=False,
        cond_encoding=cond_encoding,
    )

    ckpt_path = ckpt_root / f"{_sanitize_name(task_name)}.pt"
    loaded_existing_ckpt = ckpt_path.exists() and not overwrite_ckpt
    if loaded_existing_ckpt:
        save_model.load_vae_ckpt(ckpt_path)
    else:
        save_model.fit_vae(
            epochs=epoch,
            batch_size=min(batch_size, max(1, scaled_adata.n_obs)),
            lr=lr,
            weight_decay=weight_decay,
            kl_scale=kl_scale,
            col_mask_prob=col_msk_threshold,
            is_col_wise_mask=False,
            recon_criterion="bce",
            seed=seed,
        )
        save_model.save_vae_ckpt(ckpt_path)

    batches = list(dict.fromkeys(scaled_adata.obs[batch_key].astype(str).tolist()))
    if max_target_batches > 0:
        batches = batches[:max_target_batches]

    effective_transfer_bs = resolve_transfer_batch_size(device, transfer_batch_size)
    embedding_paths: dict[str, str] = {}
    per_batch_metrics: dict[str, dict[str, float]] = {}

    for target_batch in batches:
        latent = _transfer_to_target_batch(
            save_model=save_model,
            adata_scaled=scaled_adata,
            target_batch=target_batch,
            batch_key=batch_key,
            cell_type_key=cell_type_key,
            cond_cols=list(cond_cols),
            pca_components=pca_components,
            transfer_batch_size=effective_transfer_bs,
        )
        npy_path = emb_dir / f"transfer_to_{_sanitize_name(target_batch)}.npy"
        np.save(npy_path, latent)
        embedding_paths[target_batch] = str(npy_path)

    n_cells = int(eval_adata.n_obs)
    n_genes = int(eval_adata.n_vars)
    del save_model, scaled_adata
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if run_scib and batches:
        effective_scib_workers = resolve_scib_workers(scib_workers)
        if effective_scib_workers <= 1 or len(batches) <= 1:
            for target_batch in batches:
                embed_key = f"transfer_to_{target_batch}"
                eval_adata.obsm[embed_key] = np.load(embedding_paths[target_batch])
                per_batch_metrics[target_batch] = scib_eval(
                    eval_adata,
                    batch_key=batch_key,
                    cell_type_key=cell_type_key,
                    embed_key=embed_key,
                )
        else:
            light_adata = ad.AnnData(
                X=eval_adata.X,
                obs=eval_adata.obs[[batch_key, cell_type_key]].copy(),
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                eval_h5ad_path = str(Path(tmp_dir) / "eval_base.h5ad")
                light_adata.write_h5ad(eval_h5ad_path)
                del light_adata
                max_workers = min(effective_scib_workers, len(batches))
                mp_ctx = multiprocessing.get_context("spawn")
                raw_results: dict[str, dict[str, float]] = {}
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp_ctx,
                    initializer=_init_scib_worker,
                    initargs=(eval_h5ad_path, batch_key, cell_type_key),
                ) as executor:
                    futures = [
                        executor.submit(_scib_eval_worker, tb, embedding_paths[tb])
                        for tb in batches
                    ]
                    for fut in futures:
                        tb, metrics = fut.result()
                        raw_results[tb] = metrics
                for tb in batches:
                    per_batch_metrics[tb] = raw_results[tb]

    summary_metrics: dict[str, dict[str, float | None]] = {}
    scib_csv_path = None
    if run_scib and per_batch_metrics:
        res_df = pd.DataFrame.from_dict(per_batch_metrics)
        res_df["mean"] = res_df.mean(axis=1)
        res_df["std"] = res_df.std(axis=1)
        scib_csv = out_path / "scib_eval.csv"
        res_df.to_csv(scib_csv)
        scib_csv_path = str(scib_csv)
        for metric_name in res_df.index:
            mean_val = res_df.loc[metric_name, "mean"]
            std_val = res_df.loc[metric_name, "std"]
            summary_metrics[str(metric_name)] = {
                "mean": None if pd.isna(mean_val) else float(mean_val),
                "std": None if pd.isna(std_val) else float(std_val),
            }

    report = {
        "task_name": task_name,
        "data_path": data_path,
        "output_dir": str(out_path),
        "ckpt_path": str(ckpt_path),
        "loaded_existing_ckpt": loaded_existing_ckpt,
        "device": str(device),
        "model_preset": "SAVE-B-perturb",
        "gene_block_size": gene_block_size,
        "sample_mask_prob": sample_mask_prob,
        "col_msk_threshold": col_msk_threshold,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "batches": batches,
        "embedding_paths": embedding_paths,
        "per_batch_metrics": per_batch_metrics,
        "summary": summary_metrics,
    }
    report_path = out_path / "report.json"
    with open(report_path, "w") as handle:
        json.dump(_to_jsonable(report), handle, indent=2)

    return {
        "report_path": str(report_path),
        "scib_csv_path": scib_csv_path,
        "ckpt_path": str(ckpt_path),
        "embedding_paths": embedding_paths,
        "report": report,
    }


__all__ = [
    "DEFAULT_BEC_OUTPUT_DIR",
    "SAVE_B_PERTURB_MODEL_CONFIG",
    "SAVE_B_PERTURB_TRAIN_CONFIG",
    "_compat_scanpy_pca_for_scib",
    "_pandas_value_counts_compat",
    "compute_scib_summary",
    "get_device",
    "resolve_scib_workers",
    "resolve_transfer_batch_size",
    "run_bec_integration",
    "scib_eval",
]
