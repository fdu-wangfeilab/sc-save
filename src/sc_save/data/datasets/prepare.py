"""Dataset standardization, preprocessing, and OT gene block preparation pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from sc_save.data.datasets.download import (
    _ensure_raw_files,
    _prepare_unique_raw_files,
    ensure_genept_embeddings,
)
from sc_save.data.datasets.registry import (
    DatasetSpec,
    GENEPT_PICKLE_NAME,
    LUNG_CANCER_AGE_STAGES,
    LUNG_CANCER_UNSEEN_CONDITIONS,
    PreparedDataset,
    get_dataset_spec,
    get_output_paths,
    resolve_dataset_names,
    resolve_workers,
)
from sc_save.data.gene_blocks import (
    DEFAULT_OT_NUM_ITER_MAX,
    balanced_kmeans_ot,
    build_gene_blocks,
    load_gene_embeddings,
    sanitize_var_for_h5ad,
)
from sc_save.data.preprocessing import (
    align_genes,
    normalize_log1p,
    select_hvgs_exact,
)

_HDF5_IO_LOCK = threading.Lock()

ensure_log1p_normalized = normalize_log1p
_select_hvg_exact = select_hvgs_exact


def balanced_kmeans_with_ot_emd(
    X: np.ndarray,
    n_clusters: int,
    *,
    method: str = "emd",
    reg: float = 0.1,
    max_iter: int = 20,
    num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition embeddings into equal-size clusters using OT balanced k-means."""
    labels, centers, _ = balanced_kmeans_ot(
        X=X,
        n_clusters=n_clusters,
        method=method,
        reg=reg,
        max_iter=max_iter,
        num_iter_max=num_iter_max,
        random_state=random_state,
    )
    return labels, centers


def apply_gene_block_construction(
    adata: ad.AnnData,
    gene_embeddings: dict[str, np.ndarray],
    gene_block_size: int,
    *,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    random_state: int = 42,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Apply OT semantic Gene Block Construction with mouse/small-overlap padding fallback."""
    res = build_gene_blocks(
        adata=adata,
        gene_embeddings=gene_embeddings,
        block_size=gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=random_state,
        allow_small_overlap=True,
    )
    return res.adata, {
        "n_shared_genept": res.n_shared_genept,
        "n_semantic_blocks": res.n_semantic_blocks,
        "n_clustered_genes": res.n_clustered_genes,
        "n_pad": res.n_pad,
    }


def align_test_to_train_genes(
    adata_test: ad.AnnData,
    train_adata: ad.AnnData,
) -> ad.AnnData:
    """Align a held-out test split to the exact ordered gene + pad_* columns of train."""
    target_vars = [str(v) for v in train_adata.var_names]
    pad_set = {v for v in target_vars if v.startswith("pad_")}
    test_non_pad = [v for v in adata_test.var_names if str(v) not in pad_set]
    src_test = (
        adata_test
        if len(test_non_pad) == adata_test.n_vars
        else adata_test[:, test_non_pad].copy()
    )
    aligned = align_genes(src_test, target_vars, pad_missing=True)
    aligned.var = sanitize_var_for_h5ad(train_adata.var.copy())
    aligned.uns = dict(adata_test.uns)
    return aligned


def _standardize_obs_column(
    adata: ad.AnnData,
    target_col: str,
    candidate_sources: Sequence[str],
    *,
    force_overwrite: bool = False,
) -> None:
    if force_overwrite or target_col not in adata.obs.columns:
        chosen = next((c for c in candidate_sources if c in adata.obs.columns), None)
        if chosen is not None and chosen != target_col:
            adata.obs[target_col] = adata.obs[chosen]
    if target_col not in adata.obs.columns:
        raise KeyError(
            f"Missing required obs column '{target_col}' (checked {candidate_sources}). "
            f"Available columns: {list(adata.obs.columns)}"
        )
    series = adata.obs[target_col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        vals = series.astype(object).fillna("unknown").astype(str)
    else:
        vals = series.fillna("unknown").astype(str)
    adata.obs[target_col] = pd.Categorical(vals)


def _standardize_dataset_obs(spec: DatasetSpec, adata: ad.AnnData) -> None:
    if spec.name == "pbmc3k":
        _standardize_obs_column(
            adata, "cell_type", ("cell_type", "celltype", "clusters")
        )
    elif spec.name == "dentategyrus":
        _standardize_obs_column(
            adata,
            "cell_type",
            ("clusters", "cell_type", "celltype"),
            force_overwrite="clusters" in adata.obs.columns,
        )
    elif spec.name == "tabula_muris":
        _standardize_obs_column(
            adata,
            "cell_type",
            ("tissue", "cell_type", "cell_ontology_class"),
            force_overwrite="tissue" in adata.obs.columns,
        )
    elif spec.name in ("heart", "heart_bec"):
        _standardize_obs_column(adata, "batch", ("donor", "batch"))
        _standardize_obs_column(adata, "cell_type", ("cell_type", "celltype"))
    elif spec.name in ("pbmc", "pbmc_bec"):
        _standardize_obs_column(adata, "batch", ("batch", "donor"))
        _standardize_obs_column(
            adata,
            "cell_type",
            ("final_annotation", "cell_type", "celltype"),
            force_overwrite="final_annotation" in adata.obs.columns,
        )
    elif spec.name in ("lung_atlas", "lung_atlas_bec"):
        _standardize_obs_column(adata, "batch", ("batch", "donor", "sample"))
        _standardize_obs_column(
            adata, "cell_type", ("cell_type", "celltype", "final_annotation")
        )
    elif spec.name == "lung_cancer":
        for col in spec.cond_cols:
            _standardize_obs_column(adata, col, (col,))
    elif spec.name == "pbmc_ifn":
        _standardize_obs_column(adata, "cell_type", ("cell_type", "celltype"))
        _standardize_obs_column(adata, "condition", ("condition", "stim"))
        cond_map = {"ctrl": "control", "stim": "stimulated"}
        normalized = [
            cond_map.get(str(v).lower(), str(v)) for v in adata.obs["condition"]
        ]
        adata.obs["condition"] = pd.Categorical(normalized)
        adata.obs["batch"] = pd.Categorical(normalized)


def _subsample_adata(
    adata: ad.AnnData,
    max_cells: int | None,
    *,
    stratify_cols: Sequence[str] = (),
    seed: int = 42,
) -> ad.AnnData:
    if max_cells is None or adata.n_obs <= max_cells:
        return adata

    rng = np.random.default_rng(seed)
    cols = [c for c in stratify_cols if c in adata.obs.columns]
    if not cols:
        idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
        return adata[idx].copy()

    keys = adata.obs[cols[0]].astype(str)
    for col in cols[1:]:
        keys = keys + "||" + adata.obs[col].astype(str)
    grouped: dict[str, list[int]] = {}
    for pos, key in enumerate(keys.tolist()):
        grouped.setdefault(key, []).append(pos)

    picked: list[int] = []
    if len(grouped) <= max_cells:
        for indices in grouped.values():
            picked.append(int(indices[int(rng.integers(0, len(indices)))]))
    picked_set = set(picked)
    remaining = [i for i in range(adata.n_obs) if i not in picked_set]
    need = max_cells - len(picked)
    if need > 0 and remaining:
        extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        picked.extend(map(int, extra.tolist()))
    return adata[sorted(picked)].copy()


def _read_h5ad_safe(path: Path) -> ad.AnnData:
    with _HDF5_IO_LOCK:
        return sc.read_h5ad(path)


def _write_h5ad_atomic(adata: ad.AnnData, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    adata.var = sanitize_var_for_h5ad(adata.var)
    tmp_path = dest.with_suffix(".h5ad.part")
    try:
        with _HDF5_IO_LOCK:
            adata.write_h5ad(tmp_path)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _make_prepared_record(
    spec: DatasetSpec,
    out_paths: dict[str, Path],
    primary_adata: ad.AnnData,
    block_stats: dict[str, Any],
) -> PreparedDataset:
    return PreparedDataset(
        name=spec.name,
        table=spec.table,
        task_type=spec.task_type,
        cond_cols=list(spec.cond_cols),
        gene_block_size=spec.gene_block_size,
        hvg_top_genes=spec.hvg_top_genes,
        primary_path=out_paths[spec.primary_key],
        files=out_paths,
        n_obs=int(primary_adata.n_obs),
        n_vars=int(primary_adata.n_vars),
        n_shared_genept=int(block_stats["n_shared_genept"]),
        n_semantic_blocks=int(block_stats["n_semantic_blocks"]),
        n_pad=int(block_stats["n_pad"]),
        source_urls=list(spec.source_urls),
    )


def _prepare_table1_cfgen(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    train_raw = _read_h5ad_safe(raw_paths[0])
    test_raw = _read_h5ad_safe(raw_paths[1])
    train_raw.var_names_make_unique()
    test_raw.var_names_make_unique()

    _standardize_dataset_obs(spec, train_raw)
    _standardize_dataset_obs(spec, test_raw)

    train_raw = _subsample_adata(
        train_raw, subset_cells, stratify_cols=spec.cond_cols, seed=seed
    )
    if subset_cells is not None:
        test_raw = _subsample_adata(
            test_raw, max(1, subset_cells // 4), stratify_cols=spec.cond_cols, seed=seed
        )

    ensure_log1p_normalized(train_raw, preserve_counts_layer=False)
    ensure_log1p_normalized(test_raw, preserve_counts_layer=False)

    train_proc, stats = apply_gene_block_construction(
        train_raw,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    test_proc = align_test_to_train_genes(test_raw, train_proc)
    _write_h5ad_atomic(train_proc, out_paths["train"])
    _write_h5ad_atomic(test_proc, out_paths["test"])
    return train_proc, stats


def _prepare_table2_dual(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    adata = _read_h5ad_safe(raw_paths[0])
    adata.var_names_make_unique()
    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(adata, subset_cells, stratify_cols=spec.cond_cols, seed=seed)
    ensure_log1p_normalized(adata, preserve_counts_layer=True, normalize_total=False)

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    _write_h5ad_atomic(processed, out_paths["full"])
    return processed, stats


def _prepare_table3_lung_cancer(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    min_genes: int | None,
    min_cells: int | None,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    with _HDF5_IO_LOCK:
        backed = ad.read_h5ad(raw_paths[0], backed="r")
        try:
            if "development_stage" not in backed.obs.columns:
                raise KeyError("Expected 'development_stage' in lung_cancer obs columns.")
            age_mask = backed.obs["development_stage"].astype(str).isin(LUNG_CANCER_AGE_STAGES)
            adata = backed[age_mask, :].to_memory()
        finally:
            if getattr(backed, "file", None) is not None:
                backed.file.close()

    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(
        adata,
        subset_cells,
        stratify_cols=("assay", "development_stage", "disease", "uicc_stage"),
        seed=seed,
    )

    eff_min_genes = spec.min_genes if min_genes is None else min_genes
    eff_min_cells = spec.min_cells if min_cells is None else min_cells
    if eff_min_genes:
        sc.pp.filter_cells(adata, min_genes=eff_min_genes)
    if eff_min_cells:
        sc.pp.filter_genes(adata, min_cells=eff_min_cells)

    ensure_log1p_normalized(adata, preserve_counts_layer=False)
    hvg_target = spec.hvg_top_genes or 2000
    adata = _select_hvg_exact(adata, n_top_genes=hvg_target)

    if "feature_name" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var_names.astype(str)
        adata.var_names = pd.Index(
            adata.var["feature_name"].astype(str).tolist(), dtype="object"
        )
    adata.var_names_make_unique()

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )

    test_mask = np.zeros(processed.n_obs, dtype=bool)
    for cond in LUNG_CANCER_UNSEEN_CONDITIONS:
        cond_mask = np.ones(processed.n_obs, dtype=bool)
        for col, val in cond.items():
            cond_mask &= processed.obs[col].astype(str).to_numpy() == val
        test_mask |= cond_mask

    processed.obs["split"] = pd.Categorical(
        np.where(test_mask, "unseen_test", "seen_train")
    )
    adata_train = processed[~test_mask].copy()
    adata_test = processed[test_mask].copy()

    _write_h5ad_atomic(processed, out_paths["full"])
    _write_h5ad_atomic(adata_train, out_paths["train"])
    _write_h5ad_atomic(adata_test, out_paths["test"])
    return processed, stats


def _prepare_table4_bec(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    min_genes: int | None,
    min_cells: int | None,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    adata = _read_h5ad_safe(raw_paths[0])
    adata.var_names_make_unique()
    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(adata, subset_cells, stratify_cols=spec.cond_cols, seed=seed)

    if spec.exclude_gene_prefixes:
        keep_genes = [
            g
            for g in adata.var_names
            if not str(g).startswith(spec.exclude_gene_prefixes)
        ]
        adata = adata[:, keep_genes].copy()

    eff_min_genes = spec.min_genes if min_genes is None else min_genes
    eff_min_cells = spec.min_cells if min_cells is None else min_cells
    if eff_min_genes:
        sc.pp.filter_cells(adata, min_genes=eff_min_genes)
    if eff_min_cells:
        sc.pp.filter_genes(adata, min_cells=eff_min_cells)

    ensure_log1p_normalized(adata, preserve_counts_layer=True)
    hvg_target = spec.hvg_top_genes or 2000
    adata = _select_hvg_exact(adata, n_top_genes=hvg_target, batch_key="batch")

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    _write_h5ad_atomic(processed, out_paths["full"])
    return processed, stats


def _prepare_table5_pbmc_ifn(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    train_adata = _read_h5ad_safe(raw_paths[0])
    valid_adata = _read_h5ad_safe(raw_paths[1])
    train_adata.var_names_make_unique()
    valid_adata.var_names_make_unique()
    if not set(train_adata.obs_names).isdisjoint(set(valid_adata.obs_names)):
        train_adata.obs_names = pd.Index(
            [f"train_{x}" for x in train_adata.obs_names], dtype="object"
        )
        valid_adata.obs_names = pd.Index(
            [f"valid_{x}" for x in valid_adata.obs_names], dtype="object"
        )

    adata = ad.concat([train_adata, valid_adata], join="inner", merge="first")
    adata.obs_names_make_unique()
    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(adata, subset_cells, stratify_cols=spec.cond_cols, seed=seed)

    ensure_log1p_normalized(adata, preserve_counts_layer=False)
    hvg_target = spec.hvg_top_genes or 2000
    adata = _select_hvg_exact(adata, n_top_genes=hvg_target, batch_key="batch")

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    _write_h5ad_atomic(processed, out_paths["full"])
    return processed, stats


def _outputs_ready(out_paths: dict[str, Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in out_paths.values())


def _inspect_existing_dataset(
    spec: DatasetSpec,
    out_paths: dict[str, Path],
) -> PreparedDataset:
    primary = _read_h5ad_safe(out_paths[spec.primary_key])
    var_df = primary.var
    n_shared = (
        int(var_df["is_genept_covered"].astype(bool).sum())
        if "is_genept_covered" in var_df.columns
        else int(primary.n_vars)
    )
    n_pad = (
        int(var_df["is_pad"].astype(bool).sum())
        if "is_pad" in var_df.columns
        else int(sum(str(v).startswith("pad_") for v in primary.var_names))
    )
    if "gene_block_id" in var_df.columns:
        valid_blocks = var_df.loc[var_df["gene_block_id"] >= 0, "gene_block_id"]
        n_blocks = int(valid_blocks.nunique())
    else:
        n_blocks = 0
    return _make_prepared_record(
        spec,
        out_paths,
        primary,
        {
            "n_shared_genept": n_shared,
            "n_semantic_blocks": n_blocks,
            "n_pad": n_pad,
        },
    )


def prepare_dataset(
    name: str,
    data_root: str | Path = "./data",
    *,
    gene_embedding_path: str | Path | None = None,
    gene_embeddings: dict[str, np.ndarray] | None = None,
    skip_download: bool = False,
    force: bool = False,
    subset_cells: int | None = None,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    min_genes: int | None = None,
    min_cells: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    """Download, standardize, normalize, and OT-partition one paper benchmark dataset."""
    spec = get_dataset_spec(name)
    root = Path(data_root)
    out_paths = get_output_paths(spec.name, root)

    if not force and subset_cells is None and _outputs_ready(out_paths):
        return _inspect_existing_dataset(spec, out_paths)

    if gene_embeddings is None:
        emb_path = ensure_genept_embeddings(
            root,
            gene_embedding_path=gene_embedding_path,
            force=force,
            skip_download=skip_download,
        )
        gene_embeddings = load_gene_embeddings(str(emb_path))

    raw_paths = _ensure_raw_files(
        spec, root, skip_download=skip_download, force=force
    )

    if spec.task_type == "single_cond":
        primary_adata, stats = _prepare_table1_cfgen(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            seed=seed,
        )
    elif spec.task_type == "dual_cond":
        primary_adata, stats = _prepare_table2_dual(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            seed=seed,
        )
    elif spec.task_type == "multi_cond":
        primary_adata, stats = _prepare_table3_lung_cancer(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            min_genes=min_genes,
            min_cells=min_cells,
            seed=seed,
        )
    elif spec.task_type == "bec":
        primary_adata, stats = _prepare_table4_bec(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            min_genes=min_genes,
            min_cells=min_cells,
            seed=seed,
        )
    elif spec.task_type == "perturb_loo":
        primary_adata, stats = _prepare_table5_pbmc_ifn(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task_type: {spec.task_type}")

    return _make_prepared_record(spec, out_paths, primary_adata, stats)


def _update_manifest(
    root: Path,
    genept_path: Path,
    records: dict[str, PreparedDataset],
) -> Path:
    manifest_path = root / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:
            existing = {}

    dataset_entries = dict(existing.get("datasets", {}))
    for name, rec in records.items():
        dataset_entries[name] = rec.to_dict(data_root=root)

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gene_embedding_path": str(genept_path),
        "datasets": dataset_entries,
    }
    tmp_path = manifest_path.with_suffix(".json.part")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, manifest_path)
    return manifest_path


def prepare_datasets(
    datasets: str | Sequence[str] = "all",
    data_root: str | Path = "./data",
    *,
    gene_embedding_path: str | Path | None = None,
    skip_download: bool = False,
    force: bool = False,
    subset_cells: int | None = None,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    min_genes: int | None = None,
    min_cells: int | None = None,
    seed: int = 42,
    workers: int | str = 1,
) -> dict[str, PreparedDataset]:
    """Prepare one or more benchmark datasets (or 'all' / 'table1'..'table5') and write manifest.json."""
    n_workers = resolve_workers(workers)
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    selected = resolve_dataset_names(datasets)

    specs = [get_dataset_spec(name) for name in selected]
    pending_specs = [
        spec
        for spec in specs
        if force
        or subset_cells is not None
        or not _outputs_ready(get_output_paths(spec.name, root))
    ]

    loaded_embeddings: dict[str, np.ndarray] | None = None
    emb_path: Path | None = (
        Path(gene_embedding_path) if gene_embedding_path is not None else None
    )

    if pending_specs:
        emb_path = ensure_genept_embeddings(
            root,
            gene_embedding_path=gene_embedding_path,
            force=force,
            skip_download=skip_download,
        )
        loaded_embeddings = load_gene_embeddings(str(emb_path))

    _prepare_unique_raw_files(
        pending_specs,
        root,
        skip_download=skip_download,
        force=force,
        workers=n_workers,
    )

    def _build_one(ds_name: str) -> PreparedDataset:
        return prepare_dataset(
            ds_name,
            data_root=root,
            gene_embedding_path=emb_path,
            gene_embeddings=loaded_embeddings,
            skip_download=True,
            force=force,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            min_genes=min_genes,
            min_cells=min_cells,
            seed=seed,
        )

    results: dict[str, PreparedDataset] = {}
    if n_workers <= 1 or len(selected) <= 1:
        for name in selected:
            results[name] = _build_one(name)
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(selected))) as pool:
            fut_to_name = {pool.submit(_build_one, name): name for name in selected}
            raw_results: dict[str, PreparedDataset] = {}
            for fut in as_completed(fut_to_name):
                raw_results[fut_to_name[fut]] = fut.result()
        for name in selected:
            results[name] = raw_results[name]

    manifest_emb = (
        emb_path if emb_path is not None else (root / "genept" / GENEPT_PICKLE_NAME)
    )
    _update_manifest(root, manifest_emb, results)
    return results


__all__ = [
    "_inspect_existing_dataset",
    "_make_prepared_record",
    "_outputs_ready",
    "_prepare_table1_cfgen",
    "_prepare_table2_dual",
    "_prepare_table3_lung_cancer",
    "_prepare_table4_bec",
    "_prepare_table5_pbmc_ifn",
    "_read_h5ad_safe",
    "_select_hvg_exact",
    "_standardize_dataset_obs",
    "_standardize_obs_column",
    "_subsample_adata",
    "_update_manifest",
    "_write_h5ad_atomic",
    "align_test_to_train_genes",
    "apply_gene_block_construction",
    "balanced_kmeans_with_ot_emd",
    "ensure_log1p_normalized",
    "prepare_dataset",
    "prepare_datasets",
]
