"""Batch effect correction (scIB) evaluation utilities without R dependencies."""

from __future__ import annotations

from contextlib import contextmanager
import functools
import inspect
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


@contextmanager
def _pandas_value_counts_compat():
    """Temporarily provide ``pd.value_counts`` on pandas >=3.0 for older scib releases."""
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
    """Translate removed ``sc.tl.pca(use_highly_variable=...)`` on Scanpy >=1.13 for scib."""
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
    """Compute paper-exact ``avg_bio``, ``avg_batch``, and ``scib_score`` from raw scIB metrics."""
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
    adata: ad.AnnData,
    batch_key: str = "batch",
    cell_type_key: str = "cell_type",
    embed_key: str = "X_pca",
) -> dict[str, float]:
    """Run scIB metrics without R dependencies and aggregate matching the ICLR 2026 paper."""
    try:
        import scib
    except ImportError as exc:
        raise ImportError(
            "BEC evaluation requires `scib`. Install with `pip install 'sc-save[bec]'`."
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


__all__ = [
    "_compat_scanpy_pca_for_scib",
    "_pandas_value_counts_compat",
    "compute_scib_summary",
    "scib_eval",
]
