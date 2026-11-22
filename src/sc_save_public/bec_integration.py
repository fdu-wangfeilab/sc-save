"""Batch effect correction (BEC) workflow matching the ICLR 2026 LDM_raw protocol."""

from __future__ import annotations

import functools
import inspect
import json
import multiprocessing
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

from sc_save_public.models.save_model import SAVE

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
    if device_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_id}")


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


def _pad_to_block_size(adata: sc.AnnData, block_size: int) -> sc.AnnData:
    if block_size <= 0 or adata.n_vars % block_size == 0:
        return adata
    remainder = adata.n_vars % block_size
    n_pad = block_size - remainder
    if sparse.issparse(adata.X):
        pad_block = sparse.csr_matrix((adata.n_obs, n_pad), dtype=adata.X.dtype)
        new_x = sparse.hstack([adata.X, pad_block], format="csr")
    else:
        pad_block = np.zeros((adata.n_obs, n_pad), dtype=np.asarray(adata.X).dtype)
        new_x = np.concatenate([np.asarray(adata.X), pad_block], axis=1)
    pad_names = [f"pad_{i}" for i in range(n_pad)]
    new_var = pd.DataFrame(index=list(map(str, adata.var_names)) + pad_names)
    return sc.AnnData(X=new_x, obs=adata.obs.copy(), var=new_var)


def _scale_max_abs(adata: sc.AnnData) -> sc.AnnData:
    scaled = adata.copy()
    x = scaled.X.toarray() if sparse.issparse(scaled.X) else np.asarray(scaled.X)
    x = np.asarray(x, dtype=np.float32)
    scaler = MaxAbsScaler()
    scaled.X = np.asarray(scaler.fit_transform(x), dtype=np.float32)
    return scaled


def _prepare_eval_adata(adata: sc.AnnData, batch_key: str, cell_type_key: str) -> sc.AnnData:
    eval_adata = adata.copy()
    if sparse.issparse(eval_adata.X):
        eval_adata.X = eval_adata.X.toarray()
    eval_adata.X = np.asarray(eval_adata.X, dtype=np.float32)
    eval_adata.obs[batch_key] = eval_adata.obs[batch_key].astype("category")
    eval_adata.obs[cell_type_key] = eval_adata.obs[cell_type_key].astype("category")
    return eval_adata


@contextmanager
def _pandas_value_counts_compat():
    """Temporarily provide pd.value_counts on pandas >=3.0 for older scib releases."""
    had_attr = hasattr(pd, "value_counts")
    if not had_attr:
        pd.value_counts = lambda values, *args, **kwargs: pd.Series(values).value_counts(  # type: ignore[attr-defined]
            *args, **kwargs
        )
    try:
        yield
    finally:
        if not had_attr and hasattr(pd, "value_counts"):
            delattr(pd, "value_counts")


@contextmanager
def _compat_scanpy_pca_for_scib():
    """Translate removed sc.tl.pca(use_highly_variable=...) on Scanpy >=1.13 for scib."""
    orig_pca = sc.tl.pca
    params = inspect.signature(orig_pca).parameters
    if "use_highly_variable" in params:
        yield
        return

    @functools.wraps(orig_pca)
    def _pca_compat(*args, **kwargs):
        if "use_highly_variable" in kwargs:
            use_hv = kwargs.pop("use_highly_variable")
            if use_hv is not None and "mask_var" in params and "mask_var" not in kwargs:
                kwargs["mask_var"] = "highly_variable" if use_hv else None
        return orig_pca(*args, **kwargs)

    sc.tl.pca = _pca_compat
    try:
        yield
    finally:
        sc.tl.pca = orig_pca


def compute_scib_summary(metrics_dict: dict[str, Any]) -> dict[str, float]:
    """Compute paper-exact avg_bio, avg_batch, and scib_score from raw scIB metrics."""
    clean: dict[str, float] = {}
    for key, val in metrics_dict.items():
        if val is None:
            continue
        fval = float(val)
        if not np.isnan(fval):
            clean[str(key)] = fval

    bio_keys = ["NMI_cluster/label", "ARI_cluster/label", "ASW_label"]
    batch_keys = ["graph_conn", "ASW_label/batch"]

    if all(k in clean for k in bio_keys):
        clean["avg_bio"] = float(np.mean([clean[k] for k in bio_keys]))
    if all(k in clean for k in batch_keys):
        clean["avg_batch"] = float(np.mean([clean[k] for k in batch_keys]))
    if "avg_bio" in clean and "avg_batch" in clean:
        clean["scib_score"] = float(0.6 * clean["avg_bio"] + 0.4 * clean["avg_batch"])
    return clean


def scib_eval(
    adata: sc.AnnData,
    batch_key: str = "batch",
    cell_type_key: str = "cell_type",
    embed_key: str = "X_pca",
) -> dict[str, float]:
    """Run scib metrics without R dependencies and aggregate matching the ICLR 2026 paper."""
    try:
        import scib
    except ImportError as exc:
        raise ImportError(
            "BEC evaluation requires `scib`. Install with `pip install 'sc-save-public[bec]'`."
        ) from exc

    adata_int = adata.copy()
    adata_int.obs[batch_key] = adata_int.obs[batch_key].astype("category")
    adata_int.obs[cell_type_key] = adata_int.obs[cell_type_key].astype("category")

    candidate_kwargs: dict[str, Any] = {
        "batch_key": batch_key,
        "label_key": cell_type_key,
        "embed": embed_key,
        "isolated_labels_asw_": False,
        "silhouette_": True,
        "hvg_score_": False,
        "graph_conn_": True,
        "pcr_": True,
        "isolated_labels_f1_": False,
        "trajectory_": False,
        "nmi_": True,
        "ari_": True,
        "cell_cycle_": False,
        "kBET_": False,
        "lisi_": False,
        "lisi_graph_": False,
        "ilisi_": False,
        "clisi_": False,
    }
    accepted = inspect.signature(scib.metrics.metrics).parameters
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}

    with _pandas_value_counts_compat(), _compat_scanpy_pca_for_scib():
        results = scib.metrics.metrics(adata_int, adata_int=adata_int, **filtered_kwargs)

    raw_dict = results[0].to_dict() if hasattr(results[0], "to_dict") else dict(results[0])
    return compute_scib_summary(raw_dict)


def _transfer_to_target_batch(
    save_model: SAVE,
    adata_scaled: sc.AnnData,
    target_batch: str,
    batch_key: str,
    cell_type_key: str,
    cond_cols: list[str],
    pca_components: int,
    transfer_batch_size: int,
) -> np.ndarray:
    target_mask = adata_scaled.obs[batch_key].astype(str) == str(target_batch)
    target_adata = adata_scaled[target_mask].copy()
    other_data = adata_scaled[~target_mask].copy()

    other_cond_info = (
        other_data.obs[cond_cols]
        .value_counts()
        .rename_axis(cond_cols)
        .reset_index(name="count")
    )
    other_cond_info = other_cond_info[other_cond_info["count"] > 0].reset_index(drop=True)

    trans_list: list[sc.AnnData] = []
    for idx in range(len(other_cond_info)):
        cond_dict = {col: other_cond_info.loc[idx, col] for col in cond_cols}
        target_cond = dict(cond_dict)
        target_cond[batch_key] = target_batch

        source_mask = np.logical_and.reduce(
            [other_data.obs[col].astype(str) == str(val) for col, val in cond_dict.items()]
        )
        if not np.any(source_mask):
            continue
        source_adata = other_data[source_mask].copy()

        trans = save_model.transfer_cond(
            source_adata,
            input_cond=cond_dict,
            target_cond=target_cond,
            batch_size=min(transfer_batch_size, max(1, source_adata.n_obs)),
            verbose=False,
        )
        if isinstance(trans, tuple):
            trans = trans[0]
        if not isinstance(trans, sc.AnnData):
            raise TypeError(f"Unexpected transfer_cond return type: {type(trans).__name__}")

        trans_x = trans.X.toarray() if sparse.issparse(trans.X) else np.asarray(trans.X)
        trans_adata = sc.AnnData(
            X=np.asarray(trans_x, dtype=np.float32),
            var=source_adata.var.copy(),
            obs=source_adata.obs.copy(),
        )
        trans_adata.obs["type"] = f"SAVE_trans_{target_batch}"
        trans_list.append(trans_adata)

    merge_parts = [target_adata] + trans_list
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

    raw_adata = _pad_to_block_size(raw_adata, gene_block_size)
    eval_adata = _prepare_eval_adata(raw_adata, batch_key=batch_key, cell_type_key=cell_type_key)
    scaled_adata = _scale_max_abs(eval_adata)

    save_model = SAVE(
        adata=scaled_adata.copy(),
        device=device,
        is_initialized=True,
        seed=seed,
        condition_cols=list(cond_cols),
        hidden_dim=hidden_dim,
        enc_dim=enc_dim,
        expand_dim=expand_dim,
        enc_num_heads=enc_num_heads,
        dec_num_heads=dec_num_heads,
        enc_depth=enc_depth,
        dec_depth=dec_depth,
        enc_affine=SAVE_B_PERTURB_MODEL_CONFIG["enc_affine"],
        enc_norm_type=SAVE_B_PERTURB_MODEL_CONFIG["enc_norm_type"],
        dec_norm_init=SAVE_B_PERTURB_MODEL_CONFIG["dec_norm_init"],
        dec_blk_type=SAVE_B_PERTURB_MODEL_CONFIG["dec_blk_type"],
        gene_block_size=gene_block_size,
        latent_block_size=latent_block_size,
        is_flash_attn=SAVE_B_PERTURB_MODEL_CONFIG["is_flash_attn"],
    )

    ckpt_path = ckpt_root / f"{_sanitize_name(task_name)}.pt"
    loaded_existing_ckpt = ckpt_path.exists() and not overwrite_ckpt
    if loaded_existing_ckpt:
        save_model.load_ckpt(str(ckpt_path))
    else:
        save_model.train(
            epoch=epoch,
            batch_size=min(batch_size, max(1, scaled_adata.n_obs)),
            lr=lr,
            seed=seed,
            is_grad_clip=True,
            iter=SAVE_B_PERTURB_TRAIN_CONFIG["iter"],
            weight_decay=weight_decay,
            is_lr_scheduler=is_lr_scheduler,
            kl_scale=kl_scale,
            max_sample_mask_prob=sample_mask_prob,
            col_msk_threshold=col_msk_threshold,
            is_col_wise_mask=False,
            recon_criterion="bce",
        )
        save_model.save_ckpt(str(ckpt_path))

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
            light_adata = sc.AnnData(
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
    "compute_scib_summary",
    "resolve_scib_workers",
    "resolve_transfer_batch_size",
    "run_bec_integration",
    "scib_eval",
]
