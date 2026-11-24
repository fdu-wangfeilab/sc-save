"""Compatibility shim for legacy AnnData preprocessing and dataloader helpers."""

from __future__ import annotations

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sc_save.data.preprocessing import align_genes, preprocess_adata

CHUNK_SIZE = 40000


def reindex(adata: AnnData, genes, chunk_size: int = CHUNK_SIZE) -> AnnData:
    """Reindex AnnData columns to ``genes``, zero-filling missing genes."""
    return align_genes(adata, [str(g) for g in genes], pad_missing=True)


def batch_scale(
    adata: AnnData,
    chunk_size: int = CHUNK_SIZE,
    is_uniform: bool = False,
):
    """Batch-specific scaling returning ``(adata, batch_scaler)``."""
    batch_scaler = []
    for b in adata.obs["batch"].unique():
        idx = np.where(adata.obs["batch"] == b)[0]
        if is_uniform:
            scaler = StandardScaler(copy=False, with_mean=False, with_std=False).fit(
                adata.X[idx]
            )
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])

        tr = tqdm(range(len(idx) // chunk_size + 1), ncols=80)
        tr.set_description_str(desc="batch_scale")
        for i in tr:
            adata.X[idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                adata.X[idx[i * chunk_size : (i + 1) * chunk_size]]
            )
        batch_scaler.append((b, scaler))

    return (adata, batch_scaler)


def inverse_batch_scale(
    adata_ori: AnnData,
    adata_scaled: AnnData,
    chunk_size: int = CHUNK_SIZE,
    is_uniform: bool = False,
) -> AnnData:
    for b in adata_ori.obs["batch"].unique():
        idx = np.where(adata_ori.obs["batch"] == b)[0]
        if is_uniform:
            scaler = StandardScaler(copy=False, with_mean=False, with_std=False).fit(
                adata_ori.X[idx]
            )
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata_ori.X[idx])

        tr = tqdm(range(len(idx) // chunk_size + 1), ncols=80)
        tr.set_description_str(desc="inverse batch_scale")
        for i in tr:
            adata_scaled.X[idx[i * chunk_size : (i + 1) * chunk_size]] = (
                scaler.inverse_transform(
                    adata_scaled.X[idx[i * chunk_size : (i + 1) * chunk_size]]
                )
            )
    return adata_scaled


def preprocessing_rna(
    adata: AnnData,
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int = 10000,
    n_top_features=2000,
    chunk_size: int = CHUNK_SIZE,
    is_batch_scale: bool = True,
    log=None,
    is_ret_scaler: bool = False,
    **kwargs,
):
    if min_features is None:
        min_features = 600
    if n_top_features is None:
        n_top_features = 2000
    if target_sum is None:
        target_sum = 10000

    adata = preprocess_adata(
        adata,
        min_genes=min_features,
        min_cells=min_cells,
        target_sum=target_sum,
        n_top_genes=n_top_features,
        batch_key="batch" if "batch" in adata.obs.columns else None,
        preserve_raw=True,
        scale=False,
        copy=True,
    )

    if is_batch_scale:
        adata, batch_scaler = batch_scale(adata, chunk_size=chunk_size)
        if is_ret_scaler:
            return adata, batch_scaler
    return adata


def plot_hvg_umap(hvg_adata, color=["celltype"], save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3, 3))
    hvg_adata = hvg_adata.copy()
    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(
        hvg_adata,
        color=color,
        legend_fontsize=10,
        ncols=2,
        show=None,
        save=save_filename,
    )
    return hvg_adata


def plot_hvg_latent_umap(cpy_x, cols=["batch", "celltype"], save_path=None):
    cpy_x = cpy_x.copy()
    sc.set_figure_params(dpi=80, figsize=(5, 4))
    sc.pp.neighbors(cpy_x, n_neighbors=30, use_rep="latent")
    sc.tl.umap(cpy_x, min_dist=0.1)
    color = [c for c in cols if c in cpy_x.obs]
    sc.pl.umap(cpy_x, color=color, save=save_path, wspace=0.4, ncols=4, show=True)
    return cpy_x


def make_latent(model, device, adata, latent_pos=1, batch_size=1024):
    dataloader = convert_adata_to_dataloader(
        adata=adata, batch_size=batch_size, is_shuffle=False
    )
    model.eval()
    latents = []
    with torch.inference_mode():
        for x, _ in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x = x.float().to(device)
            z = model.encoder(x)[latent_pos]
            latents.append(z.detach().cpu().numpy())
    adata.obsm["latent"] = np.concatenate(latents, axis=0)


def get_data_ary(h5ad_path, n_top_features=2000):
    adata = sc.read_h5ad(h5ad_path)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    processed_adata = preprocessing_rna(
        adata,
        min_features=600,
        min_cells=3,
        target_sum=None,
        n_top_features=n_top_features,
        chunk_size=CHUNK_SIZE,
        log=None,
    )
    x_arr = (
        processed_adata.X
        if isinstance(processed_adata.X, np.ndarray)
        else processed_adata.X.toarray()
    )
    return processed_adata, x_arr, processed_adata.obs["celltype"].cat.codes.values


def convert_adata_to_dataloader(
    adata,
    batch_size,
    is_shuffle,
    is_droplast=False,
    condition_cols=["batch"],
    is_cond_reuse=False,
):
    id_max = 0
    total_idx = []
    if is_cond_reuse:
        cond_sets = set()
        for col in condition_cols:
            cond_sets.update(set(adata.obs[col]))
        label_encoder = LabelEncoder().fit(list(cond_sets))
        for col in condition_cols:
            idx = label_encoder.transform(adata.obs[col])
            total_idx.append(idx.reshape(-1, 1))
    else:
        for col in condition_cols:
            idx = adata.obs[col].cat.codes.values.astype("int16") + id_max
            total_idx.append(idx.reshape(-1, 1))
            id_max = id_max + idx.max() + 1

    total_idx_arr = np.hstack(total_idx)
    ary = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(ary, dtype=np.float32)),
        torch.from_numpy(total_idx_arr.astype(np.int32)),
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=is_shuffle,
        drop_last=is_droplast,
    )


def get_processed_data_ary(adata_path, data_ary_path, cell_type_ary_path):
    adata = sc.read(adata_path)
    data_ary = np.load(data_ary_path)
    cell_type_ary = np.load(cell_type_ary_path)
    return adata, data_ary, cell_type_ary


def get_data_loader(
    data_ary: np.ndarray,
    cell_type: np.ndarray,
    batch_size: int = 512,
    is_shuffle: bool = True,
):
    dataset = TensorDataset(
        torch.from_numpy(data_ary.astype(np.float32)),
        torch.from_numpy(cell_type.astype(np.int32)),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False
    )


def get_data_msk_loader(
    data_ary: np.ndarray,
    cell_type: np.ndarray,
    non_zero_msk,
    batch_size: int = 512,
    is_shuffle: bool = True,
):
    dataset = TensorDataset(
        torch.from_numpy(data_ary.astype(np.float32)),
        torch.from_numpy(cell_type.astype(np.int32)),
        torch.from_numpy(non_zero_msk.astype(np.int32)),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False
    )


__all__ = [
    "CHUNK_SIZE",
    "batch_scale",
    "convert_adata_to_dataloader",
    "get_data_ary",
    "get_data_loader",
    "get_data_msk_loader",
    "get_processed_data_ary",
    "inverse_batch_scale",
    "make_latent",
    "plot_hvg_latent_umap",
    "plot_hvg_umap",
    "preprocessing_rna",
    "reindex",
]
