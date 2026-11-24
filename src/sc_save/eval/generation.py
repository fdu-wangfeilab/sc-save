"""Conditional generation distribution and expression evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import torch
from tqdm import tqdm

from sc_save.eval.metrics import compute_evaluation_metrics_fast

DEFAULT_OUTPUT_DIR = "./outputs/conditional_distribution_evaluation"
DEFAULT_COND_COLS = ["batch", "cell_type"]

PRIMARY_SUMMARY_COLS = ["2-Wasserstein_PCA", "RBF_MMD_PCA"]
EXTRA_DISTRIBUTION_COLS = ["1-Wasserstein_PCA", "Linear_MMD_unbiased_PCA"]
EXPRESSION_SUMMARY_COLS = ["pcc", "mse", "r2", "top_gene_overlap"]

CANCER_LUNG_MACRO_COND_COLS = ["assay", "development_stage", "disease", "uicc_stage"]
CANCER_LUNG_UNSEEN_CONDITIONS = [
    {
        "assay": "10x 3' v2",
        "development_stage": "65-year-old human stage",
        "disease": "chronic obstructive pulmonary disease",
        "uicc_stage": "non-cancer",
    },
    {
        "assay": "Smart-seq2",
        "development_stage": "70-year-old human stage",
        "disease": "squamous cell lung carcinoma",
        "uicc_stage": "I",
    },
]


def get_device(device_id: int) -> torch.device:
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def load_adata(data_path: str, data_name: str) -> ad.AnnData:
    print(f"[pipeline] loading {data_name} from: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"[pipeline] {data_name} shape: {adata.shape}")
    return adata


def validate_obs_cols(
    adata: ad.AnnData,
    required_cols: Sequence[str],
    data_name: str,
) -> None:
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"{data_name}.obs is missing required columns: {missing_cols}")


def to_numpy_matrix(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def align_generated_to_target_vars(
    gt_data: ad.AnnData,
    gen_data: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData, str]:
    if gt_data.n_vars != gen_data.n_vars:
        raise ValueError(
            "target and generated AnnData objects must have the same number of genes: "
            f"{gt_data.n_vars} != {gen_data.n_vars}"
        )

    if gt_data.var_names.equals(gen_data.var_names):
        return gt_data, gen_data, "matched"

    gt_var_names = list(map(str, gt_data.var_names))
    gen_var_names = list(map(str, gen_data.var_names))
    if len(set(gt_var_names)) == gt_data.n_vars and set(gt_var_names) == set(gen_var_names):
        print("[pipeline] reordering generated genes to match target var_names")
        gen_data = gen_data[:, gt_data.var_names].copy()
        return gt_data, gen_data, "reordered"

    print(
        "[pipeline] generated var_names do not match target; assigning target var_names "
        "by position to keep the evaluation path runnable"
    )
    gen_data = gen_data.copy()
    gen_data.var_names = gt_data.var_names.copy()
    return gt_data, gen_data, "assigned"


def _streaming_randomized_pca(
    gt_x: Any,
    gen_x: Any,
    n_components: int,
    *,
    chunk_size: int = 4096,
    n_oversamples: int = 10,
    n_iter: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact joint PCA projections via chunked randomized SVD without concatenation."""
    n_gt, n_vars = int(gt_x.shape[0]), int(gt_x.shape[1])
    n_gen = int(gen_x.shape[0])
    n_total = n_gt + n_gen
    k = min(n_components, n_total, n_vars)
    n_random = min(n_total, n_vars, k + n_oversamples)

    def _iter_chunks():
        for offset, mat, nrows in ((0, gt_x, n_gt), (n_gt, gen_x, n_gen)):
            is_sp = sparse.issparse(mat)
            for s in range(0, nrows, chunk_size):
                e = min(s + chunk_size, nrows)
                yield offset + s, offset + e, mat[s:e], is_sp

    sum_vec = np.zeros(n_vars, dtype=np.float64)
    for _, _, blk, is_sp in _iter_chunks():
        if is_sp:
            sum_vec += np.asarray(blk.sum(axis=0), dtype=np.float64).ravel()
        else:
            sum_vec += np.asarray(blk, dtype=np.float32).sum(axis=0, dtype=np.float64)
    mean = (sum_vec / float(n_total)).astype(np.float32)

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n_vars, n_random), dtype=np.float32)

    q = np.empty((n_total, n_random), dtype=np.float32)
    mean_omega = mean @ omega
    for s, e, blk, is_sp in _iter_chunks():
        prod = blk @ omega if is_sp else np.asarray(blk, dtype=np.float32) @ omega
        q[s:e] = np.asarray(prod, dtype=np.float32) - mean_omega
    q, _ = np.linalg.qr(q, mode="reduced")

    for _ in range(n_iter):
        z = np.zeros((n_vars, n_random), dtype=np.float32)
        q_sum = q.sum(axis=0, dtype=np.float64).astype(np.float32)
        for s, e, blk, is_sp in _iter_chunks():
            qs = q[s:e]
            prod = blk.T @ qs if is_sp else np.asarray(blk, dtype=np.float32).T @ qs
            z += np.asarray(prod, dtype=np.float32)
        z -= np.outer(mean, q_sum)
        z, _ = np.linalg.qr(z, mode="reduced")

        mean_z = mean @ z
        for s, e, blk, is_sp in _iter_chunks():
            prod = blk @ z if is_sp else np.asarray(blk, dtype=np.float32) @ z
            q[s:e] = np.asarray(prod, dtype=np.float32) - mean_z
        q, _ = np.linalg.qr(q, mode="reduced")

    b = np.zeros((n_random, n_vars), dtype=np.float32)
    q_sum = q.sum(axis=0, dtype=np.float64).astype(np.float32)
    for s, e, blk, is_sp in _iter_chunks():
        qs = q[s:e]
        prod = (blk.T @ qs).T if is_sp else qs.T @ np.asarray(blk, dtype=np.float32)
        b += np.asarray(prod, dtype=np.float32)
    b -= np.outer(q_sum, mean)

    u_b, s_vals, _ = np.linalg.svd(b, full_matrices=False)
    x_pca = np.asarray((q @ u_b[:, :k]) * s_vals[:k], dtype=np.float32)
    return np.ascontiguousarray(x_pca[:n_gt]), np.ascontiguousarray(x_pca[n_gt:])


def build_joint_pca(
    gt_data: ad.AnnData,
    gen_data: ad.AnnData,
    n_components: int,
    chunk_size: int = 4096,
    *,
    streaming_threshold_bytes: int = 4 * (1024**3),
) -> int:
    n_gt, n_vars = int(gt_data.n_obs), int(gt_data.n_vars)
    n_gen = int(gen_data.n_obs)
    actual_n_components = min(n_components, n_gt + n_gen, n_vars)
    if actual_n_components < 1:
        raise ValueError("PCA requires at least one sample and one feature")

    if (n_gt + n_gen) * n_vars * 4 > streaming_threshold_bytes:
        print(f"[pipeline] streaming randomized PCA with n_components={actual_n_components}")
        gt_pca, gen_pca = _streaming_randomized_pca(
            gt_data.X,
            gen_data.X,
            actual_n_components,
            chunk_size=chunk_size,
        )
        gt_data.obsm["X_pca"] = gt_pca
        gen_data.obsm["X_pca"] = gen_pca
        gc.collect()
        return actual_n_components

    combined_x = np.empty((n_gt + n_gen, n_vars), dtype=np.float32)
    for dst_offset, src_x, n_rows in (
        (0, gt_data.X, n_gt),
        (n_gt, gen_data.X, n_gen),
    ):
        is_sp = sparse.issparse(src_x)
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            block = src_x[start:end].toarray() if is_sp else src_x[start:end]
            combined_x[dst_offset + start : dst_offset + end] = np.asarray(
                block, dtype=np.float32
            )

    print(f"[pipeline] fitting PCA with n_components={actual_n_components}")
    pca = PCA(n_components=actual_n_components, copy=False)
    x_pca = pca.fit_transform(combined_x)
    del combined_x

    gt_data.obsm["X_pca"] = np.ascontiguousarray(x_pca[:n_gt])
    gen_data.obsm["X_pca"] = np.ascontiguousarray(x_pca[n_gt:])
    del x_pca
    gc.collect()
    return actual_n_components


def get_condition_distribution(
    adata: ad.AnnData,
    cond_cols: Sequence[str],
) -> pd.DataFrame:
    cols = list(cond_cols)
    return (
        adata.obs[cols]
        .value_counts()
        .rename_axis(cols)
        .reset_index(name="count")
    )


def compare_condition_distributions(
    gt_distro: pd.DataFrame,
    gen_distro: pd.DataFrame,
    cond_cols: Sequence[str],
    sort_cols: Sequence[str],
) -> pd.DataFrame:
    merged = gt_distro.rename(columns={"count": "count_gt"}).merge(
        gen_distro.rename(columns={"count": "count_gen"}),
        on=list(cond_cols),
        how="outer",
    )
    merged["count_gt"] = merged["count_gt"].fillna(0).astype(int)
    merged["count_gen"] = merged["count_gen"].fillna(0).astype(int)
    merged["present_in_gt"] = merged["count_gt"] > 0
    merged["present_in_gen"] = merged["count_gen"] > 0
    merged["is_equal"] = merged["count_gt"] == merged["count_gen"]
    merged = merged.sort_values(by=list(sort_cols)).reset_index(drop=True)
    return merged


def build_condition_mask(adata: ad.AnnData, cond_dict: dict[str, Any]) -> np.ndarray:
    masks = [adata.obs[col] == value for col, value in cond_dict.items()]
    return np.logical_and.reduce(masks)


def _group_indices(
    obs: pd.DataFrame,
    cond_cols: Sequence[str],
) -> dict[tuple[Any, ...], np.ndarray]:
    cols = list(cond_cols)
    raw_groups = obs.groupby(cols, observed=True).indices
    if len(cols) == 1:
        return {(key,): np.asarray(idx, dtype=np.int64) for key, idx in raw_groups.items()}
    return {
        (key if isinstance(key, tuple) else (key,)): np.asarray(idx, dtype=np.int64)
        for key, idx in raw_groups.items()
    }


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def resolve_eval_workers(eval_workers: int = 0) -> int:
    """Resolve CPU worker threads for per-condition distribution metric evaluation."""
    if int(eval_workers) <= 0:
        return min(8, os.cpu_count() or 1)
    return max(1, int(eval_workers))


def compute_distribution_metrics_by_condition(
    gt_data: ad.AnnData,
    gen_data: ad.AnnData,
    merged_distro: pd.DataFrame,
    cond_cols: Sequence[str],
    model_name: str,
    device: torch.device,
    max_cells_per_condition: int,
    eval_workers: int = 0,
) -> pd.DataFrame:
    cols = list(cond_cols)
    expected_columns = cols + [
        "cell_count",
        "1-Wasserstein_PCA",
        "2-Wasserstein_PCA",
        "Linear_MMD_unbiased_PCA",
        "RBF_MMD_PCA",
    ]
    gt_pca_all = np.asarray(gt_data.obsm["X_pca"])
    gen_pca_all = np.asarray(gen_data.obsm["X_pca"])

    gt_groups = _group_indices(gt_data.obs, cols)
    gen_groups = _group_indices(gen_data.obs, cols)

    tasks = []
    for idx in range(len(merged_distro)):
        if not bool(merged_distro.loc[idx, "is_equal"]):
            continue
        cond_dict = merged_distro.loc[idx, cols].to_dict()
        group_key = tuple(cond_dict[c] for c in cols)
        gt_idx = gt_groups.get(group_key)
        gen_idx = gen_groups.get(group_key)
        if gt_idx is None:
            gt_idx = np.flatnonzero(build_condition_mask(gt_data, cond_dict))
        if gen_idx is None:
            gen_idx = np.flatnonzero(build_condition_mask(gen_data, cond_dict))
        cell_count = int(merged_distro.loc[idx, "count_gt"])
        tasks.append(
            (
                cond_dict,
                cell_count,
                gt_pca_all[gt_idx].copy(),
                gen_pca_all[gen_idx].copy(),
            )
        )

    effective_workers = resolve_eval_workers(eval_workers)
    eval_device = torch.device("cpu") if effective_workers > 1 else device

    def _eval_one(task):
        cond_dict, cell_count, gt_pca, gen_pca = task
        metric = compute_evaluation_metrics_fast(
            adata_real_pca_ary=gt_pca,
            adata_generated_pca_ary=gen_pca,
            model_name=model_name,
            device=eval_device,
            batch_size=min(cell_count, max_cells_per_condition),
        )
        row = dict(cond_dict)
        row["cell_count"] = cell_count
        row.update(metric)
        return row

    if effective_workers <= 1 or len(tasks) <= 1:
        cond_metrics = [_eval_one(t) for t in tqdm(tasks, ncols=80)]
    else:
        max_workers = min(effective_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            cond_metrics = list(
                tqdm(executor.map(_eval_one, tasks), total=len(tasks), ncols=80)
            )

    return pd.DataFrame(cond_metrics, columns=expected_columns)


def compute_expression_metrics_by_condition(
    gt_data: ad.AnnData,
    gen_data: ad.AnnData,
    merged_distro: pd.DataFrame,
    cond_cols: Sequence[str],
    hvg_top_genes: int,
    top_gene_count: int,
) -> pd.DataFrame:
    cols = list(cond_cols)
    expected_columns = cols + ["top_gene_overlap", "pcc", "mse", "r2"]
    hvg_df = sc.pp.highly_variable_genes(
        adata=gt_data,
        n_top_genes=min(hvg_top_genes, gt_data.n_vars),
        subset=False,
        inplace=False,
    )
    hvg_mask = hvg_df["highly_variable"].to_numpy()
    if not hvg_mask.any():
        hvg_mask = np.ones(gt_data.n_vars, dtype=bool)

    gt_x_hvg = to_numpy_matrix(gt_data.X[:, hvg_mask])
    gen_x_hvg = to_numpy_matrix(gen_data.X[:, hvg_mask])
    hvg_var_names = np.asarray(gt_data.var_names)[hvg_mask]
    top_gene_count = min(top_gene_count, hvg_var_names.shape[0])

    gt_groups = _group_indices(gt_data.obs, cols)
    gen_groups = _group_indices(gen_data.obs, cols)

    cond_metrics = []
    for idx in tqdm(range(len(merged_distro)), ncols=80):
        if not bool(merged_distro.loc[idx, "is_equal"]):
            continue

        cond_dict = merged_distro.loc[idx, cols].to_dict()
        group_key = tuple(cond_dict[c] for c in cols)
        gt_idx = gt_groups.get(group_key)
        gen_idx = gen_groups.get(group_key)
        if gt_idx is None:
            gt_idx = np.flatnonzero(build_condition_mask(gt_data, cond_dict))
        if gen_idx is None:
            gen_idx = np.flatnonzero(build_condition_mask(gen_data, cond_dict))

        mean_gt_gene = gt_x_hvg[gt_idx].mean(axis=0)
        mean_gen_gene = gen_x_hvg[gen_idx].mean(axis=0)

        gt_top_idx = np.argsort(mean_gt_gene)[-top_gene_count:]
        gen_top_idx = np.argsort(mean_gen_gene)[-top_gene_count:]
        gt_top_genes = set(hvg_var_names[gt_top_idx])
        gen_top_genes = set(hvg_var_names[gen_top_idx])

        row = dict(cond_dict)
        row["top_gene_overlap"] = (
            len(gt_top_genes.intersection(gen_top_genes)) / max(1, top_gene_count)
        )
        row["pcc"] = safe_pearsonr(mean_gt_gene, mean_gen_gene)
        row["mse"] = mean_squared_error(mean_gt_gene, mean_gen_gene)
        row["r2"] = r2_score(mean_gt_gene, mean_gen_gene)
        cond_metrics.append(row)

    return pd.DataFrame(cond_metrics, columns=expected_columns)


def summarize_metrics(
    metrics_df: pd.DataFrame,
    metric_cols: Sequence[str],
) -> dict[str, dict[str, float]]:
    summary = {}
    for col in metric_cols:
        if col not in metrics_df.columns:
            continue
        series = pd.to_numeric(metrics_df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        summary[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
    return summary


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _build_unseen_mask(
    metrics_df: pd.DataFrame,
    unseen_conditions: Sequence[dict[str, Any]] | None,
) -> pd.Series:
    if metrics_df.empty or not unseen_conditions:
        return pd.Series(False, index=metrics_df.index, dtype=bool)
    combined_mask = pd.Series(False, index=metrics_df.index, dtype=bool)
    for cond_dict in unseen_conditions:
        present_keys = [k for k in cond_dict if k in metrics_df.columns]
        if not present_keys or len(present_keys) != len(cond_dict):
            continue
        cond_mask = pd.Series(True, index=metrics_df.index, dtype=bool)
        for key, value in cond_dict.items():
            cond_mask &= metrics_df[key].astype(str) == str(value)
        combined_mask |= cond_mask
    return combined_mask


def run_conditional_distribution_evaluation(
    gt_path: str,
    gen_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cond_cols: Sequence[str] | None = None,
    sort_cols: Sequence[str] | None = None,
    device_id: int = 0,
    pca_components: int = 30,
    max_cells_per_condition: int = 4000,
    hvg_top_genes: int = 2000,
    top_gene_count: int = 100,
    model_name: str = "generated",
    include_expression_metrics: bool = True,
    unseen_conditions: Sequence[dict[str, Any]] | None = None,
    macro_cond_cols: Sequence[str] | None = None,
    task_name: str | None = None,
    eval_workers: int = 0,
) -> dict[str, Any]:
    """Evaluate generated conditional distributions and write CSV/JSON reports."""
    if cond_cols is None:
        cond_cols = list(DEFAULT_COND_COLS)
    else:
        cond_cols = list(cond_cols)
    if sort_cols is None:
        sort_cols = list(cond_cols)
    else:
        sort_cols = list(sort_cols)
    if not set(sort_cols).issubset(set(cond_cols)):
        raise ValueError(
            f"sort_cols must be a subset of cond_cols: {sort_cols} vs {cond_cols}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device_id)
    gt_data = load_adata(gt_path, "target")
    gen_data = load_adata(gen_path, "generated")

    validate_obs_cols(gt_data, cond_cols, "target")
    validate_obs_cols(gen_data, cond_cols, "generated")
    gt_data, gen_data, var_alignment = align_generated_to_target_vars(gt_data, gen_data)

    gen_data.obs_names = [f"gen_{idx}" for idx in range(gen_data.n_obs)]

    pca_n_components = build_joint_pca(
        gt_data=gt_data,
        gen_data=gen_data,
        n_components=pca_components,
    )

    gt_distro = get_condition_distribution(gt_data, cond_cols)
    gen_distro = get_condition_distribution(gen_data, cond_cols)
    merged_distro = compare_condition_distributions(
        gt_distro=gt_distro,
        gen_distro=gen_distro,
        cond_cols=cond_cols,
        sort_cols=sort_cols,
    )

    distribution_metrics_df = compute_distribution_metrics_by_condition(
        gt_data=gt_data,
        gen_data=gen_data,
        merged_distro=merged_distro,
        cond_cols=cond_cols,
        model_name=model_name,
        device=device,
        max_cells_per_condition=max_cells_per_condition,
        eval_workers=eval_workers,
    )

    metrics_df = distribution_metrics_df.copy()
    expression_summary = {}
    if include_expression_metrics and not distribution_metrics_df.empty:
        expression_metrics_df = compute_expression_metrics_by_condition(
            gt_data=gt_data,
            gen_data=gen_data,
            merged_distro=merged_distro,
            cond_cols=cond_cols,
            hvg_top_genes=hvg_top_genes,
            top_gene_count=top_gene_count,
        )
        metrics_df = distribution_metrics_df.merge(
            expression_metrics_df,
            on=cond_cols,
            how="left",
        )
        expression_summary = summarize_metrics(metrics_df, EXPRESSION_SUMMARY_COLS)

    is_cancer_task = task_name in {"cancer_lung", "lung_cancer"}
    has_cancer_cols = set(CANCER_LUNG_MACRO_COND_COLS).issubset(set(cond_cols))
    effective_unseen = unseen_conditions
    if effective_unseen is None and (is_cancer_task or has_cancer_cols):
        candidate_mask = _build_unseen_mask(metrics_df, CANCER_LUNG_UNSEEN_CONDITIONS)
        if is_cancer_task or bool(candidate_mask.any()):
            effective_unseen = CANCER_LUNG_UNSEEN_CONDITIONS

    effective_macro_cols = list(macro_cond_cols) if macro_cond_cols is not None else None
    if effective_macro_cols is None and effective_unseen is not None and has_cancer_cols:
        effective_macro_cols = list(CANCER_LUNG_MACRO_COND_COLS)

    all_summary_cols = (
        PRIMARY_SUMMARY_COLS + EXTRA_DISTRIBUTION_COLS + EXPRESSION_SUMMARY_COLS
    )
    seen_unseen_summary: dict[str, Any] = {}
    if effective_unseen is not None and not metrics_df.empty:
        unseen_mask = _build_unseen_mask(metrics_df, effective_unseen)
        metrics_df["split"] = np.where(unseen_mask, "unseen", "seen")
        seen_df = metrics_df[~unseen_mask]
        unseen_df = metrics_df[unseen_mask]
        seen_unseen_summary = {
            "n_seen_conditions": int((~unseen_mask).sum()),
            "n_unseen_conditions": int(unseen_mask.sum()),
            "seen": summarize_metrics(seen_df, all_summary_cols),
            "unseen": summarize_metrics(unseen_df, all_summary_cols),
        }

    macro_summary: dict[str, Any] = {}
    macro_metrics_path = None
    if (
        effective_macro_cols
        and not metrics_df.empty
        and set(effective_macro_cols).issubset(set(metrics_df.columns))
    ):
        macro_df = (
            metrics_df.groupby(effective_macro_cols, observed=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        macro_file = out_dir / "macro_condition_metrics.csv"
        macro_df.to_csv(macro_file, index=False)
        macro_metrics_path = str(macro_file)
        macro_summary = {
            "macro_cond_cols": effective_macro_cols,
            "n_macro_conditions": int(len(macro_df)),
            "metrics": summarize_metrics(macro_df, all_summary_cols),
        }

    merged_distro.to_csv(out_dir / "distribution_alignment.csv", index=False)
    metrics_df.to_csv(out_dir / "per_condition_metrics.csv", index=False)

    primary_summary = summarize_metrics(metrics_df, PRIMARY_SUMMARY_COLS)
    extra_distribution_summary = summarize_metrics(metrics_df, EXTRA_DISTRIBUTION_COLS)
    summary_payload: dict[str, Any] = {
        "primary_metrics": primary_summary,
        "extra_distribution_metrics": extra_distribution_summary,
        "expression_metrics": expression_summary,
    }
    if seen_unseen_summary:
        summary_payload.update(seen_unseen_summary)
    if macro_summary:
        summary_payload["macro"] = macro_summary

    report = {
        "gt_path": gt_path,
        "gen_path": gen_path,
        "output_dir": str(out_dir),
        "cond_cols": cond_cols,
        "sort_cols": sort_cols,
        "device": str(device),
        "target_shape": list(gt_data.shape),
        "generated_shape": list(gen_data.shape),
        "var_alignment": var_alignment,
        "pca_components": pca_n_components,
        "distribution": {
            "total_conditions": int(len(merged_distro)),
            "exact_match_conditions": int(merged_distro["is_equal"].sum()),
            "mismatched_conditions": int((~merged_distro["is_equal"]).sum()),
            "all_conditions_match": bool(merged_distro["is_equal"].all()),
        },
        "summary": summary_payload,
    }

    with open(out_dir / "report.json", "w") as handle:
        json.dump(to_jsonable(report), handle, indent=2)

    print("[pipeline] primary summary:")
    for metric_name, stats in primary_summary.items():
        print(f"  - {metric_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    result_payload: dict[str, Any] = {
        "distribution_alignment_path": str(out_dir / "distribution_alignment.csv"),
        "per_condition_metrics_path": str(out_dir / "per_condition_metrics.csv"),
        "report_path": str(out_dir / "report.json"),
        "report": report,
    }
    if macro_metrics_path is not None:
        result_payload["macro_condition_metrics_path"] = macro_metrics_path
    return result_payload


__all__ = [
    "CANCER_LUNG_MACRO_COND_COLS",
    "CANCER_LUNG_UNSEEN_CONDITIONS",
    "DEFAULT_COND_COLS",
    "DEFAULT_OUTPUT_DIR",
    "EXPRESSION_SUMMARY_COLS",
    "EXTRA_DISTRIBUTION_COLS",
    "PRIMARY_SUMMARY_COLS",
    "align_generated_to_target_vars",
    "build_condition_mask",
    "build_joint_pca",
    "compare_condition_distributions",
    "compute_distribution_metrics_by_condition",
    "compute_expression_metrics_by_condition",
    "get_condition_distribution",
    "get_device",
    "load_adata",
    "resolve_eval_workers",
    "run_conditional_distribution_evaluation",
    "safe_pearsonr",
    "summarize_metrics",
    "to_jsonable",
    "to_numpy_matrix",
    "validate_obs_cols",
]
