"""Public gene block construction pipeline."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import ot
import pandas as pd
import scanpy as sc
from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus

DEFAULT_OUTPUT_DIR = "./outputs/gene_block_construction"


def load_gene_embeddings(gene_embedding_path: str) -> dict[str, np.ndarray]:
    with open(gene_embedding_path, "rb") as handle:
        gene_embeddings = pickle.load(handle)

    if not isinstance(gene_embeddings, dict):
        raise TypeError("gene embeddings pickle must contain a dict-like mapping")

    return gene_embeddings


def load_adata(
    data_path: str,
    gene_name_col: str | None = None,
    subset_cells: int = 0,
):
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
    adata,
    gene_embeddings: dict[str, np.ndarray],
    gene_block_size: int,
    max_hvg_rounds: int = 8,
):
    original_pos = {gene: idx for idx, gene in enumerate(adata.var_names.tolist())}
    shared_genes = [gene for gene in adata.var_names if gene in gene_embeddings]
    if not shared_genes:
        raise ValueError("no overlap between adata.var_names and gene embeddings")

    hvg_target = len(shared_genes) // gene_block_size * gene_block_size
    if hvg_target == 0:
        raise ValueError(
            "gene_block_size is larger than the number of embedding-covered genes"
        )

    adata_shared = adata[:, shared_genes].copy()
    hvg_rounds = 0
    while adata_shared.shape[1] != hvg_target:
        prev_dim = adata_shared.shape[1]
        sc.pp.highly_variable_genes(
            adata_shared,
            n_top_genes=hvg_target,
            inplace=True,
            subset=True,
        )
        hvg_rounds += 1
        if adata_shared.shape[1] == hvg_target:
            break
        if adata_shared.shape[1] == prev_dim or hvg_rounds >= max_hvg_rounds:
            raise RuntimeError(
                "failed to reach an HVG target divisible by "
                f"gene_block_size after {hvg_rounds} rounds"
            )

    stable_hvg_order = sorted(
        adata_shared.var_names.tolist(),
        key=lambda gene: original_pos[gene],
    )
    adata_shared = adata_shared[:, stable_hvg_order].copy()
    return adata_shared, shared_genes, hvg_target, hvg_rounds


def balanced_kmeans_with_ot(
    X: np.ndarray,
    n_clusters: int,
    method: str = "emd",
    reg: float = 0.1,
    max_iter: int = 20,
    random_state: int = 42,
):
    n_genes, _ = X.shape
    centers, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state)

    source = np.ones(n_genes) / n_genes
    target = np.ones(n_clusters) / n_clusters
    converged = False
    labels = None

    print(f"[pipeline] running balanced OT k-means with method={method}")
    for iteration in range(1, max_iter + 1):
        old_centers = centers.copy()
        cost_matrix = cdist(X, centers, metric="sqeuclidean")

        if method == "emd":
            transport_plan = ot.emd(source, target, cost_matrix)
        elif method == "sinkhorn":
            transport_plan = ot.sinkhorn(source, target, cost_matrix, reg=reg)
        else:
            raise ValueError("method must be 'emd' or 'sinkhorn'")

        labels = np.argmax(transport_plan, axis=1)
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = X[mask].mean(axis=0)

        if np.allclose(centers, old_centers):
            converged = True
            break

    if labels is None:
        raise RuntimeError("failed to produce OT cluster labels")

    return labels, centers, {"iterations": iteration, "converged": converged}


def build_gene_order(adata_shared, labels: np.ndarray):
    adata_shared.var = adata_shared.var.copy()
    adata_shared.var["gene_block_id"] = labels
    adata_shared.var["original_var_pos"] = np.arange(adata_shared.n_vars, dtype=int)
    order_df = pd.DataFrame(
        {
            "gene": adata_shared.var_names.tolist(),
            "gene_block_id": adata_shared.var["gene_block_id"].to_numpy(),
            "original_var_pos": adata_shared.var["original_var_pos"].to_numpy(),
        }
    )
    order_df = order_df.sort_values(
        ["gene_block_id", "original_var_pos"],
        kind="stable",
    )
    sorted_idx = order_df["original_var_pos"].to_numpy()
    sorted_genes = adata_shared.var_names[sorted_idx].tolist()
    block_id_map = dict(zip(adata_shared.var_names.tolist(), labels.tolist()))
    return sorted_genes, block_id_map


def create_padded_output(
    adata,
    sorted_genes: list[str],
    block_id_map: dict[str, int],
    shared_genes: list[str],
    gene_block_size: int,
):
    sorted_gene_set = set(sorted_genes)
    shared_gene_set = set(shared_genes)
    remaining_genes = [gene for gene in adata.var_names if gene not in sorted_gene_set]
    ordered_genes = sorted_genes + remaining_genes

    reordered = adata[:, ordered_genes].copy()
    pad_count = (-reordered.shape[1]) % gene_block_size
    pad_names = [f"pad_{idx}" for idx in range(pad_count)]

    if pad_count > 0:
        if sparse.issparse(reordered.X):
            pad_matrix = sparse.csr_matrix(
                (reordered.n_obs, pad_count),
                dtype=reordered.X.dtype,
            )
        else:
            pad_matrix = np.zeros((reordered.n_obs, pad_count), dtype=reordered.X.dtype)
        pad_adata = sc.AnnData(
            X=pad_matrix,
            obs=reordered.obs.copy(),
            var=pd.DataFrame(index=pad_names),
        )
        processed = sc.concat([reordered, pad_adata], axis=1)
        processed.obs = reordered.obs.copy()
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


def sanitize_var_for_h5ad(var_df: pd.DataFrame):
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


def save_outputs(processed, out_dir: Path, report: dict):
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
):
    start = time.time()
    gene_embeddings = load_gene_embeddings(gene_embedding_path)
    adata, log1p_applied = load_adata(
        data_path=data_path,
        gene_name_col=gene_name_col,
        subset_cells=subset_cells,
    )

    adata_shared, shared_genes, hvg_target, hvg_rounds = select_genept_hvgs(
        adata=adata,
        gene_embeddings=gene_embeddings,
        gene_block_size=gene_block_size,
    )

    gene_embs = np.asarray([gene_embeddings[gene] for gene in adata_shared.var_names])
    n_blocks = gene_embs.shape[0] // gene_block_size
    labels, _, ot_stats = balanced_kmeans_with_ot(
        X=gene_embs,
        n_clusters=n_blocks,
        method=ot_method,
        reg=ot_reg,
        max_iter=ot_max_iter,
        random_state=ot_random_state,
    )

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(set(counts.tolist())) != 1 or int(counts[0]) != gene_block_size:
        raise RuntimeError(
            "OT clustering did not produce equal-sized semantic blocks: "
            f"{dict(zip(unique_labels.tolist(), counts.tolist()))}"
        )

    sorted_genes, block_id_map = build_gene_order(adata_shared=adata_shared, labels=labels)
    processed, remaining_genes, pad_names = create_padded_output(
        adata=adata,
        sorted_genes=sorted_genes,
        block_id_map=block_id_map,
        shared_genes=shared_genes,
        gene_block_size=gene_block_size,
    )

    report = {
        "data_path": data_path,
        "gene_embedding_path": gene_embedding_path,
        "gene_name_col": gene_name_col,
        "input_shape": list(adata.shape),
        "output_shape": list(processed.shape),
        "shared_gene_count": int(len(shared_genes)),
        "selected_gene_count": int(len(sorted_genes)),
        "remaining_gene_count": int(len(remaining_genes)),
        "pad_gene_count": int(len(pad_names)),
        "gene_block_size": int(gene_block_size),
        "n_blocks": int(n_blocks),
        "block_sizes": {
            str(int(label)): int(count)
            for label, count in zip(unique_labels.tolist(), counts.tolist())
        },
        "log1p_applied": log1p_applied,
        "hvg_target": int(hvg_target),
        "hvg_rounds": int(hvg_rounds),
        "ot_method": ot_method,
        "ot_reg": float(ot_reg),
        "ot_max_iter": int(ot_max_iter),
        "ot_random_state": int(ot_random_state),
        "ot_iterations": int(ot_stats["iterations"]),
        "ot_converged": bool(ot_stats["converged"]),
        "elapsed_seconds": round(time.time() - start, 3),
    }

    save_outputs(processed=processed, out_dir=Path(output_dir), report=report)
    print("[pipeline] done")
    print(json.dumps(report, indent=2))
    return report


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "balanced_kmeans_with_ot",
    "build_gene_order",
    "create_padded_output",
    "load_adata",
    "load_gene_embeddings",
    "run_gene_block_construction",
    "sanitize_var_for_h5ad",
    "save_outputs",
    "select_genept_hvgs",
]
