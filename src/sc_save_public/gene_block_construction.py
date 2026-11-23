"""Backward-compatibility shim for gene block construction."""

from __future__ import annotations

import time
from pathlib import Path
import json
from typing import Any

import numpy as np

from sc_save.data.gene_blocks import (
    DEFAULT_OUTPUT_DIR,
    balanced_kmeans_ot as balanced_kmeans_with_ot,
    build_gene_blocks,
    build_gene_order,
    create_padded_output,
    load_adata,
    load_gene_embeddings,
    sanitize_var_for_h5ad,
    save_outputs,
    select_genept_hvgs,
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
) -> dict[str, Any]:
    """Run gene block construction while resolving ``balanced_kmeans_with_ot`` in module scope."""
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
        _kmeans_fn=balanced_kmeans_with_ot,
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
