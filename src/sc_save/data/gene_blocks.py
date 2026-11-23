"""Optimal-Transport Gene Block Construction utilities for sc_save."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import time
from typing import Any

import anndata as ad
import numpy as np
import ot
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus

from sc_save.data.preprocessing import select_hvgs_exact

DEFAULT_OUTPUT_DIR = "./outputs/gene_block_construction"
DEFAULT_OT_NUM_ITER_MAX = 1_000_000


@dataclass
class GeneBlockResult:
    """Result container returned by :func:`build_gene_blocks`."""

    adata: ad.AnnData
    n_shared_genept: int
    n_semantic_blocks: int
    n_clustered_genes: int
    n_pad: int
    block_sizes: dict[int, int]
    ot_iterations: int
    ot_converged: bool


def load_gene_embeddings(gene_embedding_path: str | Path) -> dict[str, np.ndarray]:
    """Load a gene-to-embedding-vector mapping from a pickle file."""
    with open(gene_embedding_path, "rb") as handle:
        gene_embeddings = pickle.load(handle)

    if not isinstance(gene_embeddings, dict):
        raise TypeError("gene embeddings pickle must contain a dict-like mapping")

    return gene_embeddings


def load_adata(
    data_path: str,
    gene_name_col: str | None = None,
    subset_cells: int = 0,
) -> tuple[ad.AnnData, bool]:
    """Load an AnnData file for gene block construction, applying log1p if unlogged."""
    print(f"[pipeline] loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)
    if subset_cells > 0:
        adata = adata[:subset_cells].copy()
        print(f"[pipeline] subset shape: {adata.shape}")
    else:
        print(f"[pipeline] full shape: {adata.shape}")

    adata.var = adata.var.copy()
    adata.var["source_var_name"] = adata.var_names.astype(str)

    if gene_name_col is not None:
        if gene_name_col not in adata.var.columns:
            raise ValueError(f"missing gene_name_col in adata.var: {gene_name_col}")
        adata.var_names = pd.Index(
            [str(value) for value in adata.var[gene_name_col].tolist()],
            dtype="object",
        )
    else:
        adata.var_names = pd.Index(
            [str(value) for value in adata.var_names.tolist()],
            dtype="object",
        )

    if not adata.var_names.is_unique:
        raise ValueError("gene names must be unique after applying gene_name_col")

    x_max = float(np.asarray(adata.X.max()).squeeze())
    log1p_applied = False
    if x_max > np.log1p(1e4):
        print("[pipeline] applying log1p to match preprocessing notebooks")
        sc.pp.log1p(adata)
        log1p_applied = True

    return adata, log1p_applied


def select_genept_hvgs(
    adata: ad.AnnData,
    gene_embeddings: dict[str, np.ndarray],
    gene_block_size: int,
    max_hvg_rounds: int = 8,
) -> tuple[ad.AnnData, list[str], int, int]:
    """Select embedding-covered HVGs divisible by ``gene_block_size``."""
    original_pos = {str(gene): idx for idx, gene in enumerate(adata.var_names.tolist())}
    shared_genes = [str(gene) for gene in adata.var_names if str(gene) in gene_embeddings]
    if not shared_genes:
        raise ValueError("no overlap between adata.var_names and gene embeddings")

    hvg_target = len(shared_genes) // gene_block_size * gene_block_size
    if hvg_target == 0:
        raise ValueError(
            "gene_block_size is larger than the number of embedding-covered genes"
        )

    adata_shared = adata[:, shared_genes].copy()
    hvg_rounds = 0
    if adata_shared.shape[1] != hvg_target:
        adata_shared = select_hvgs_exact(
            adata_shared,
            n_top_genes=hvg_target,
            max_rounds=max_hvg_rounds,
        )
        hvg_rounds = 1
        if adata_shared.shape[1] != hvg_target:
            raise RuntimeError(
                "failed to reach an HVG target divisible by "
                f"gene_block_size after {max_hvg_rounds} rounds"
            )

    stable_hvg_order = sorted(
        [str(g) for g in adata_shared.var_names.tolist()],
        key=lambda gene: original_pos[gene],
    )
    adata_shared = adata_shared[:, stable_hvg_order].copy()
    return adata_shared, shared_genes, hvg_target, hvg_rounds


def balanced_kmeans_ot(
    X: np.ndarray,
    n_clusters: int,
    *,
    method: str = "emd",
    reg: float = 0.1,
    max_iter: int = 20,
    num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Partition embeddings into equal-size clusters using Optimal Transport k-means."""
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    n_genes = X_arr.shape[0]
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if n_clusters == 1:
        return (
            np.zeros(n_genes, dtype=np.int64),
            X_arr.mean(axis=0, keepdims=True),
            {"iterations": 1, "converged": True},
        )

    centers, _ = kmeans_plusplus(X_arr, n_clusters=n_clusters, random_state=random_state)
    centers = np.ascontiguousarray(centers, dtype=np.float64)

    source = np.full(n_genes, 1.0 / n_genes, dtype=np.float64)
    target = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    converged = False
    labels: np.ndarray | None = None
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        old_centers = centers.copy()
        cost_matrix = np.ascontiguousarray(
            cdist(X_arr, centers, metric="sqeuclidean"),
            dtype=np.float64,
        )

        if method == "emd":
            transport_plan = ot.emd(
                source,
                target,
                cost_matrix,
                numItermax=int(num_iter_max),
            )
        elif method == "sinkhorn":
            transport_plan = ot.sinkhorn(source, target, cost_matrix, reg=reg)
        else:
            raise ValueError("method must be 'emd' or 'sinkhorn'")

        labels = np.argmax(transport_plan, axis=1).astype(np.int64)
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = X_arr[mask].mean(axis=0)

        if np.allclose(centers, old_centers):
            converged = True
            break

    if labels is None:
        raise RuntimeError("failed to produce OT cluster labels")

    return labels, centers, {"iterations": iterations, "converged": converged}


def build_gene_order(
    adata_shared: ad.AnnData,
    labels: np.ndarray,
) -> tuple[list[str], dict[str, int]]:
    """Order clustered genes stably by ``(gene_block_id, original_var_pos)``."""
    adata_shared.var = adata_shared.var.copy()
    adata_shared.var["gene_block_id"] = labels
    adata_shared.var["original_var_pos"] = np.arange(adata_shared.n_vars, dtype=int)
    order_df = pd.DataFrame(
        {
            "gene": [str(g) for g in adata_shared.var_names.tolist()],
            "gene_block_id": adata_shared.var["gene_block_id"].to_numpy(),
            "original_var_pos": adata_shared.var["original_var_pos"].to_numpy(),
        }
    ).sort_values(["gene_block_id", "original_var_pos"], kind="stable")
    sorted_idx = order_df["original_var_pos"].to_numpy()
    sorted_genes = [str(adata_shared.var_names[i]) for i in sorted_idx]
    block_id_map = {
        str(gene): int(label)
        for gene, label in zip(adata_shared.var_names.tolist(), labels.tolist())
    }
    return sorted_genes, block_id_map


def create_padded_output(
    adata: ad.AnnData,
    sorted_genes: list[str],
    block_id_map: dict[str, int],
    shared_genes: list[str],
    gene_block_size: int,
) -> tuple[ad.AnnData, list[str], list[str]]:
    """Reorder genes into semantic blocks + remainder and pad to a multiple of ``gene_block_size``."""
    sorted_gene_set = set(sorted_genes)
    shared_gene_set = set(shared_genes)
    remaining_genes = [
        str(gene) for gene in adata.var_names if str(gene) not in sorted_gene_set
    ]
    ordered_genes = sorted_genes + remaining_genes

    reordered = adata[:, ordered_genes].copy()
    pad_count = (-reordered.shape[1]) % gene_block_size
    pad_names = [f"pad_{idx}" for idx in range(pad_count)]

    if pad_count > 0:
        if sparse.issparse(reordered.X):
            pad_matrix: Any = sparse.csr_matrix(
                (reordered.n_obs, pad_count),
                dtype=reordered.X.dtype,
            )
            new_x: Any = sparse.hstack([reordered.X.tocsr(), pad_matrix], format="csr")
        else:
            src_x = np.asarray(reordered.X)
            pad_matrix = np.zeros((reordered.n_obs, pad_count), dtype=src_x.dtype)
            new_x = np.concatenate([src_x, pad_matrix], axis=1)
        processed = ad.AnnData(
            X=new_x,
            obs=reordered.obs.copy(),
            var=pd.DataFrame(index=ordered_genes + pad_names),
        )
    else:
        processed = reordered

    var_base = reordered.var.copy()
    if pad_names:
        pad_var = pd.DataFrame(index=pad_names, columns=var_base.columns)
        final_var = pd.concat([var_base, pad_var], axis=0)
    else:
        final_var = var_base

    final_index = ordered_genes + pad_names
    final_var = final_var.reindex(final_index)
    final_var["gene_block_id"] = -1
    final_var["gene_block_role"] = "remainder"
    final_var["is_genept_covered"] = False
    final_var["is_hvg_selected"] = False
    final_var["is_pad"] = False

    if block_id_map:
        for gene, block_id in block_id_map.items():
            final_var.at[gene, "gene_block_id"] = int(block_id)
            final_var.at[gene, "gene_block_role"] = "semantic_block"
            final_var.at[gene, "is_hvg_selected"] = True
            final_var.at[gene, "is_genept_covered"] = True

    for gene in remaining_genes:
        final_var.at[gene, "is_genept_covered"] = gene in shared_gene_set

    for gene in pad_names:
        final_var.at[gene, "gene_block_role"] = "pad"
        final_var.at[gene, "is_pad"] = True

    final_var.index = pd.Index([str(value) for value in final_index], dtype="object")
    processed.var = final_var
    processed.var_names = pd.Index(
        [str(value) for value in final_index],
        dtype="object",
    )
    return processed, remaining_genes, pad_names


def sanitize_var_for_h5ad(var_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``adata.var`` column dtypes serialize cleanly to HDF5/h5ad."""
    var_df = var_df.copy()
    for column in var_df.columns:
        series = var_df[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("object")
        if is_string_dtype(series.dtype) or is_object_dtype(series.dtype):
            var_df[column] = series.fillna("").map(str)
        elif is_bool_dtype(series.dtype):
            var_df[column] = series.fillna(False).astype(bool)
    var_df.index = pd.Index([str(value) for value in var_df.index.tolist()], dtype="object")
    return var_df


def _preserve_padded_layers(
    src: ad.AnnData,
    processed: ad.AnnData,
    pad_count: int,
) -> None:
    """Reapply ``src.layers`` onto ``processed`` with trailing zero padding."""
    layer_names = [name for name in src.layers.keys() if name is not None]
    if not layer_names:
        return
    real_count = processed.n_vars - pad_count
    real_vars = [str(v) for v in processed.var_names[:real_count]]
    src_aligned = src[:, real_vars]
    for layer_name in layer_names:
        layer_mat = src_aligned.layers[layer_name]
        if pad_count > 0:
            if sparse.issparse(layer_mat):
                pad_mat = sparse.csr_matrix(
                    (processed.n_obs, pad_count), dtype=layer_mat.dtype
                )
                layer_mat = sparse.hstack([layer_mat.tocsr(), pad_mat], format="csr")
            else:
                pad_mat = np.zeros((processed.n_obs, pad_count), dtype=layer_mat.dtype)
                layer_mat = np.hstack([layer_mat, pad_mat])
        processed.layers[layer_name] = layer_mat


def save_outputs(processed: ad.AnnData, out_dir: Path, report: dict[str, Any]) -> None:
    """Write ``processed.h5ad``, ``var_metadata.csv``, and ``report.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_path = out_dir / "processed.h5ad"
    report_path = out_dir / "report.json"
    var_meta_path = out_dir / "var_metadata.csv"

    processed.var = sanitize_var_for_h5ad(processed.var)
    processed.var_names = pd.Index(
        [str(value) for value in processed.var_names.tolist()],
        dtype="object",
    )
    processed.write_h5ad(processed_path)
    processed.var.to_csv(var_meta_path)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"[pipeline] saved processed data to: {processed_path}")
    print(f"[pipeline] saved var metadata to: {var_meta_path}")
    print(f"[pipeline] saved report to: {report_path}")


def build_gene_blocks(
    adata: ad.AnnData,
    gene_embeddings: dict[str, np.ndarray] | str | Path,
    *,
    block_size: int = 3200,
    gene_name_col: str | None = None,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    random_state: int = 42,
    allow_small_overlap: bool = True,
    _kmeans_fn: Any = None,
) -> GeneBlockResult:
    """Partition genes into OT-balanced semantic blocks and pad to a multiple of ``block_size``."""
    if isinstance(gene_embeddings, (str, Path)):
        embs = load_gene_embeddings(gene_embeddings)
    else:
        embs = gene_embeddings

    work_adata = adata
    if gene_name_col is not None:
        if gene_name_col not in work_adata.var.columns:
            raise ValueError(f"missing gene_name_col in adata.var: {gene_name_col}")
        work_adata = work_adata.copy()
        work_adata.var = work_adata.var.copy()
        work_adata.var["source_var_name"] = work_adata.var_names.astype(str)
        work_adata.var_names = pd.Index(
            [str(value) for value in work_adata.var[gene_name_col].tolist()],
            dtype="object",
        )
        if not work_adata.var_names.is_unique:
            raise ValueError("gene names must be unique after applying gene_name_col")

    shared_genes = [str(g) for g in work_adata.var_names if str(g) in embs]
    n_shared = len(shared_genes)

    if n_shared < block_size:
        if not allow_small_overlap:
            if n_shared == 0:
                raise ValueError("no overlap between adata.var_names and gene embeddings")
            raise ValueError(
                "gene_block_size is larger than the number of embedding-covered genes"
            )
        shared_set = set(shared_genes)
        remaining = [str(g) for g in work_adata.var_names if str(g) not in shared_set]
        reordered = work_adata[:, shared_genes + remaining].copy()
        processed, _, pad_names = create_padded_output(
            adata=reordered,
            sorted_genes=[],
            block_id_map={},
            shared_genes=shared_genes,
            gene_block_size=block_size,
        )
        processed.var["is_genept_covered"] = processed.var_names.isin(shared_set)
        processed.var = sanitize_var_for_h5ad(processed.var)
        processed.uns = dict(work_adata.uns)
        _preserve_padded_layers(reordered, processed, len(pad_names))
        return GeneBlockResult(
            adata=processed,
            n_shared_genept=n_shared,
            n_semantic_blocks=0,
            n_clustered_genes=0,
            n_pad=len(pad_names),
            block_sizes={},
            ot_iterations=0,
            ot_converged=True,
        )

    adata_shared, shared_genes, _, _ = select_genept_hvgs(
        adata=work_adata,
        gene_embeddings=embs,
        gene_block_size=block_size,
    )
    emb_matrix = np.asarray(
        [embs[str(g)] for g in adata_shared.var_names],
        dtype=np.float64,
    )
    n_blocks = emb_matrix.shape[0] // block_size

    if _kmeans_fn is None:
        labels, _, ot_stats = balanced_kmeans_ot(
            X=emb_matrix,
            n_clusters=n_blocks,
            method=ot_method,
            reg=ot_reg,
            max_iter=ot_max_iter,
            num_iter_max=ot_num_iter_max,
            random_state=random_state,
        )
    else:
        labels, _, ot_stats = _kmeans_fn(
            X=emb_matrix,
            n_clusters=n_blocks,
            method=ot_method,
            reg=ot_reg,
            max_iter=ot_max_iter,
            random_state=random_state,
        )

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(set(counts.tolist())) != 1 or int(counts[0]) != block_size:
        raise RuntimeError(
            "OT clustering did not produce equal-sized semantic blocks: "
            f"{dict(zip(unique_labels.tolist(), counts.tolist()))}"
        )

    sorted_genes, block_id_map = build_gene_order(adata_shared=adata_shared, labels=labels)
    processed, _, pad_names = create_padded_output(
        adata=work_adata,
        sorted_genes=sorted_genes,
        block_id_map=block_id_map,
        shared_genes=shared_genes,
        gene_block_size=block_size,
    )
    processed.var = sanitize_var_for_h5ad(processed.var)
    processed.uns = dict(work_adata.uns)
    _preserve_padded_layers(work_adata, processed, len(pad_names))

    block_sizes = {
        int(label): int(count)
        for label, count in zip(unique_labels.tolist(), counts.tolist())
    }
    return GeneBlockResult(
        adata=processed,
        n_shared_genept=n_shared,
        n_semantic_blocks=n_blocks,
        n_clustered_genes=len(sorted_genes),
        n_pad=len(pad_names),
        block_sizes=block_sizes,
        ot_iterations=int(ot_stats.get("iterations", 0)),
        ot_converged=bool(ot_stats.get("converged", False)),
    )


def run_gene_block_construction(
    data_path: str,
    gene_embedding_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    gene_block_size: int = 3200,
    gene_name_col: str | None = None,
    subset_cells: int = 0,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_random_state: int = 42,
    _kmeans_fn: Any = None,
) -> dict[str, Any]:
    """End-to-end file-based gene block construction entrypoint."""
    start = time.time()
    gene_embeddings = load_gene_embeddings(gene_embedding_path)
    adata, log1p_applied = load_adata(
        data_path=data_path,
        gene_name_col=gene_name_col,
        subset_cells=subset_cells,
    )
    result = build_gene_blocks(
        adata=adata,
        gene_embeddings=gene_embeddings,
        block_size=gene_block_size,
        gene_name_col=None,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        random_state=ot_random_state,
        allow_small_overlap=False,
        _kmeans_fn=_kmeans_fn,
    )

    n_remaining = int((result.adata.var["gene_block_role"] == "remainder").sum())
    report = {
        "data_path": data_path,
        "gene_embedding_path": gene_embedding_path,
        "gene_name_col": gene_name_col,
        "input_shape": list(adata.shape),
        "output_shape": list(result.adata.shape),
        "shared_gene_count": int(result.n_shared_genept),
        "selected_gene_count": int(result.n_clustered_genes),
        "remaining_gene_count": n_remaining,
        "pad_gene_count": int(result.n_pad),
        "gene_block_size": int(gene_block_size),
        "n_blocks": int(result.n_semantic_blocks),
        "block_sizes": {str(k): int(v) for k, v in result.block_sizes.items()},
        "log1p_applied": log1p_applied,
        "hvg_target": int(result.n_clustered_genes),
        "hvg_rounds": 1 if result.n_shared_genept != result.n_clustered_genes else 0,
        "ot_method": ot_method,
        "ot_reg": float(ot_reg),
        "ot_max_iter": int(ot_max_iter),
        "ot_random_state": int(ot_random_state),
        "ot_iterations": int(result.ot_iterations),
        "ot_converged": bool(result.ot_converged),
        "elapsed_seconds": round(time.time() - start, 3),
    }

    save_outputs(processed=result.adata, out_dir=Path(output_dir), report=report)
    print("[pipeline] done")
    print(json.dumps(report, indent=2))
    return report


__all__ = [
    "DEFAULT_OT_NUM_ITER_MAX",
    "DEFAULT_OUTPUT_DIR",
    "GeneBlockResult",
    "balanced_kmeans_ot",
    "build_gene_blocks",
    "build_gene_order",
    "create_padded_output",
    "load_adata",
    "load_gene_embeddings",
    "run_gene_block_construction",
    "sanitize_var_for_h5ad",
    "save_outputs",
    "select_genept_hvgs",
]
