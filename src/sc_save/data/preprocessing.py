"""Preprocessing, scaling, alignment, and condition vocabulary utilities for sc_save."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.sparsefuncs import inplace_column_scale, min_max_axis


class ConditionVocabulary:
    """Vocabulary mapping multi-column categorical conditions to integer embedding IDs.

    Supports two modes:
    - ``"legacy"``: reproduces exact v1 ``LabelEncoder().fit(["<mask>"] + list(obs[col].unique()))``
      per-column behavior with zero column offsets (for v1 checkpoint parity).
    - ``"offset"``: allocates disjoint embedding ID ranges across condition columns starting at
      index ``1``, reserving index ``0`` exclusively for ``mask_token`` even when labels sort
      before ``"<mask>"`` (such as digit-prefixed labels like ``"10x 3' v2"``).
    """

    def __init__(
        self,
        condition_cols: Sequence[str],
        mode: Literal["legacy", "offset"] = "legacy",
        mask_token: str = "<mask>",
    ) -> None:
        if mode not in ("legacy", "offset"):
            raise ValueError(f"Unsupported ConditionVocabulary mode: {mode!r}")
        self._condition_cols: list[str] = [str(col) for col in condition_cols]
        self._mode: Literal["legacy", "offset"] = mode
        self._mask_token: str = str(mask_token)
        self._mask_id: int = 0
        self._label_encoder_dict: dict[str, LabelEncoder] = {}
        self._offsets: dict[str, int] = {}
        self._offset_maps: dict[str, dict[str, int]] = {}
        self._is_fitted: bool = False

    @property
    def condition_cols(self) -> list[str]:
        return list(self._condition_cols)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def mask_token(self) -> str:
        return self._mask_token

    @property
    def mask_id(self) -> int:
        return self._mask_id

    @property
    def label_encoder_dict(self) -> dict[str, LabelEncoder]:
        self._check_fitted()
        return dict(self._label_encoder_dict)

    @property
    def offsets(self) -> dict[str, int]:
        self._check_fitted()
        return dict(self._offsets)

    @property
    def total_classes(self) -> int:
        self._check_fitted()
        return 1 + sum(len(le.classes_) - 1 for le in self._label_encoder_dict.values())

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("ConditionVocabulary is not fitted yet. Call fit() first.")

    def fit(self, obs: pd.DataFrame) -> ConditionVocabulary:
        missing = [col for col in self._condition_cols if col not in obs.columns]
        if missing:
            raise KeyError(f"obs is missing condition columns: {missing}")

        encoders: dict[str, LabelEncoder] = {}
        offsets: dict[str, int] = {}
        offset_maps: dict[str, dict[str, int]] = {}
        running_offset = 1

        for col in self._condition_cols:
            if self._mode == "legacy":
                cond_values = [self._mask_token] + list(obs[col].unique())
                le = LabelEncoder()
                le.fit(cond_values)
                encoders[col] = le
                offsets[col] = 0
            else:
                non_mask = sorted(
                    {
                        str(val)
                        for val in obs[col].unique()
                        if str(val) != self._mask_token
                    }
                )
                non_mask_le = LabelEncoder()
                if non_mask:
                    non_mask_le.fit(non_mask)
                else:
                    non_mask_le.classes_ = np.array([], dtype=object)

                display_le = LabelEncoder()
                display_le.classes_ = np.array(
                    [self._mask_token] + list(non_mask_le.classes_),
                    dtype=object,
                )
                encoders[col] = display_le
                offsets[col] = running_offset

                col_map: dict[str, int] = {self._mask_token: self._mask_id}
                for idx, label in enumerate(non_mask_le.classes_):
                    col_map[str(label)] = running_offset + idx
                offset_maps[col] = col_map
                running_offset += len(non_mask_le.classes_)

        self._label_encoder_dict = encoders
        self._offsets = offsets
        self._offset_maps = offset_maps
        self._is_fitted = True
        return self

    def encode_obs(self, obs: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        missing = [col for col in self._condition_cols if col not in obs.columns]
        if missing:
            raise KeyError(f"obs is missing condition columns: {missing}")

        n_obs = len(obs)
        n_cols = len(self._condition_cols)
        if n_cols == 0:
            return np.empty((n_obs, 0), dtype=np.int64)
        if n_obs == 0:
            return np.empty((0, n_cols), dtype=np.int64)

        out = np.empty((n_obs, n_cols), dtype=np.int64)
        for col_idx, col in enumerate(self._condition_cols):
            series = obs[col]
            if self._mode == "legacy":
                out[:, col_idx] = self._label_encoder_dict[col].transform(series).astype(
                    np.int64, copy=False
                )
            else:
                col_map = self._offset_maps[col]
                if isinstance(series.dtype, pd.CategoricalDtype):
                    codes = series.cat.codes.to_numpy()
                    if (codes < 0).any():
                        raise ValueError(
                            f"Condition column {col!r} contains missing (NaN) values."
                        )
                    categories = series.cat.categories
                    used_codes = np.unique(codes)
                    cat_lookup = np.zeros(len(categories), dtype=np.int64)
                    for code in used_codes:
                        cat_str = str(categories[int(code)])
                        if cat_str not in col_map:
                            raise ValueError(
                                f"Unseen label {cat_str!r} in condition column {col!r}."
                            )
                        cat_lookup[int(code)] = col_map[cat_str]
                    out[:, col_idx] = cat_lookup[codes]
                else:
                    if series.isna().any():
                        raise ValueError(
                            f"Condition column {col!r} contains missing (NaN) values."
                        )
                    str_series = series.astype(str)
                    mapped = str_series.map(col_map)
                    if mapped.isna().any():
                        unseen = sorted(set(str_series[mapped.isna()].tolist()))
                        raise ValueError(
                            f"Unseen labels in condition column {col!r}: {unseen[:10]}"
                        )
                    out[:, col_idx] = mapped.to_numpy(dtype=np.int64)
        return out

    def encode_dict(
        self,
        cond: dict[str, Any],
        *,
        allow_partial: bool = False,
    ) -> np.ndarray:
        self._check_fitted()
        cond_keys = set(cond.keys())
        expected_keys = set(self._condition_cols)
        extra_keys = cond_keys - expected_keys
        if extra_keys:
            raise ValueError(f"Unknown condition columns: {sorted(extra_keys)}")
        if not allow_partial and cond_keys != expected_keys:
            missing = sorted(expected_keys - cond_keys)
            raise ValueError(f"Missing condition columns: {missing}")

        ids = np.empty(len(self._condition_cols), dtype=np.int64)
        for idx, col in enumerate(self._condition_cols):
            if col not in cond:
                ids[idx] = self._mask_id
                continue
            val = cond[col]
            if self._mode == "legacy":
                ids[idx] = int(self._label_encoder_dict[col].transform([val])[0])
            else:
                val_str = str(val)
                col_map = self._offset_maps[col]
                if val_str not in col_map:
                    raise ValueError(
                        f"Unknown value {val!r} for condition column {col!r}."
                    )
                ids[idx] = col_map[val_str]
        return ids

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        classes_payload: dict[str, list[Any]] = {}
        for col, le in self._label_encoder_dict.items():
            classes_payload[col] = [
                v.item() if isinstance(v, np.generic) else v
                for v in le.classes_.tolist()
            ]
        return {
            "condition_cols": list(self._condition_cols),
            "mode": self._mode,
            "mask_token": self._mask_token,
            "classes": classes_payload,
            "offsets": dict(self._offsets),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConditionVocabulary:
        mode = payload.get("mode", "legacy")
        vocab = cls(
            condition_cols=payload["condition_cols"],
            mode=mode,
            mask_token=payload.get("mask_token", "<mask>"),
        )
        classes_dict: dict[str, list[Any]] = payload["classes"]
        offsets_dict: dict[str, int] = payload.get("offsets", {})

        encoders: dict[str, LabelEncoder] = {}
        offsets: dict[str, int] = {}
        offset_maps: dict[str, dict[str, int]] = {}
        running_offset = 1

        for col in vocab._condition_cols:
            col_classes = list(classes_dict[col])
            le = LabelEncoder()
            if vocab._mode == "legacy":
                le.classes_ = np.asarray(col_classes)
                encoders[col] = le
                offsets[col] = int(offsets_dict.get(col, 0))
            else:
                non_mask = [str(c) for c in col_classes if str(c) != vocab._mask_token]
                le.classes_ = np.array(
                    [vocab._mask_token] + non_mask,
                    dtype=object,
                )
                encoders[col] = le
                col_offset = int(offsets_dict.get(col, running_offset))
                offsets[col] = col_offset
                col_map: dict[str, int] = {vocab._mask_token: vocab._mask_id}
                for idx, label in enumerate(non_mask):
                    col_map[label] = col_offset + idx
                offset_maps[col] = col_map
                running_offset = col_offset + len(non_mask)

        vocab._label_encoder_dict = encoders
        vocab._offsets = offsets
        vocab._offset_maps = offset_maps
        vocab._is_fitted = True
        return vocab


class MaxAbsExpressionScaler:
    """Per-gene max-abs scaler operating cleanly on dense and scipy.sparse float32 matrices."""

    def __init__(self) -> None:
        self.scale_: np.ndarray | None = None
        self.max_abs_: np.ndarray | None = None

    def _check_fitted(self, n_features: int | None = None) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("MaxAbsExpressionScaler is not fitted yet. Call fit() first.")
        if n_features is not None and int(n_features) != int(self.scale_.shape[0]):
            raise ValueError(
                f"Feature count mismatch: expected {self.scale_.shape[0]}, got {n_features}"
            )
        return self.scale_

    def fit(self, X: np.ndarray | sparse.spmatrix) -> MaxAbsExpressionScaler:
        if sparse.issparse(X):
            mat = X if (sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X)) else X.tocsr()
            if mat.shape[0] == 0:
                max_abs = np.zeros(mat.shape[1], dtype=np.float32)
            else:
                mins, maxs = min_max_axis(mat, axis=0, ignore_nan=True)
                max_abs = np.maximum(np.abs(mins), np.abs(maxs)).astype(np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
            if arr.shape[0] == 0:
                max_abs = np.zeros(arr.shape[1], dtype=np.float32)
            else:
                abs_arr = np.abs(arr)
                with np.errstate(all="ignore"):
                    max_abs = np.nanmax(abs_arr, axis=0).astype(np.float32)

        scale = max_abs.copy()
        scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
        self.max_abs_ = max_abs
        self.scale_ = scale.astype(np.float32, copy=False)
        return self

    def transform(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.spmatrix:
        if sparse.issparse(X):
            scale = self._check_fitted(X.shape[1])
            out = (
                X.copy()
                if (sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X))
                else X.tocsr(copy=True)
            )
            out.data = out.data.astype(np.float32, copy=True)
            inv_scale = (1.0 / scale).astype(np.float32, copy=False)
            inplace_column_scale(out, inv_scale)
            return out

        arr = np.array(X, dtype=np.float32, copy=True)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
        scale = self._check_fitted(arr.shape[1])
        arr /= scale[None, :]
        return arr

    def inverse_transform(
        self, X: np.ndarray | sparse.spmatrix
    ) -> np.ndarray | sparse.spmatrix:
        if sparse.issparse(X):
            scale = self._check_fitted(X.shape[1])
            out = (
                X.copy()
                if (sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X))
                else X.tocsr(copy=True)
            )
            out.data = out.data.astype(np.float32, copy=True)
            inplace_column_scale(out, scale.astype(np.float32, copy=False))
            return out

        arr = np.array(X, dtype=np.float32, copy=True)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
        scale = self._check_fitted(arr.shape[1])
        arr *= scale[None, :]
        return arr

    def fit_transform(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.spmatrix:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        scale = self._check_fitted()
        return {
            "scale": scale.astype(np.float32).tolist(),
            "max_abs": (
                self.max_abs_.astype(np.float32).tolist()
                if self.max_abs_ is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaxAbsExpressionScaler:
        scaler = cls()
        raw_scale = payload.get("scale")
        if raw_scale is None:
            raw_scale = payload.get("scale_")
        if raw_scale is None:
            raise KeyError("Scaler payload missing 'scale'")
        scale = np.asarray(raw_scale, dtype=np.float32).copy()
        scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
        scaler.scale_ = scale
        raw_max_abs = payload.get("max_abs")
        if raw_max_abs is not None:
            scaler.max_abs_ = np.asarray(raw_max_abs, dtype=np.float32).copy()
        else:
            scaler.max_abs_ = scale.copy()
        return scaler


def find_pad_index(adata: ad.AnnData) -> int:
    """Return index of the first ``pad_0`` (or ``is_pad``) column, or ``adata.n_vars``."""
    var_names = adata.var_names
    if "pad_0" in var_names:
        loc = var_names.get_loc("pad_0")
        if isinstance(loc, (int, np.integer)):
            return int(loc)
        if isinstance(loc, slice):
            return int(loc.start or 0)
        return int(np.flatnonzero(loc)[0])
    if "is_pad" in adata.var.columns:
        pad_mask = adata.var["is_pad"].astype(bool).to_numpy()
        if pad_mask.any():
            return int(np.argmax(pad_mask))
    return int(adata.shape[1])


def pad_to_block_size(adata: ad.AnnData, block_size: int) -> tuple[ad.AnnData, int]:
    """Pad ``adata`` with trailing zero ``pad_*`` columns to a multiple of ``block_size``."""
    pad_idx = find_pad_index(adata)
    if block_size <= 0 or adata.n_vars % block_size == 0:
        return adata, pad_idx

    n_pad = block_size - (adata.n_vars % block_size)
    if sparse.issparse(adata.X):
        pad_block = sparse.csr_matrix((adata.n_obs, n_pad), dtype=adata.X.dtype)
        new_x: Any = sparse.hstack([adata.X.tocsr(), pad_block], format="csr")
    else:
        src_x = np.asarray(adata.X)
        pad_block = np.zeros((adata.n_obs, n_pad), dtype=src_x.dtype)
        new_x = np.concatenate([src_x, pad_block], axis=1)

    existing_names = [str(v) for v in adata.var_names]
    start_idx = 0 if "pad_0" not in set(existing_names) else adata.n_vars
    pad_names = [f"pad_{start_idx + i}" for i in range(n_pad)]

    if len(adata.var.columns) > 0:
        pad_var = pd.DataFrame(
            index=pd.Index(pad_names, dtype="object"),
            columns=adata.var.columns,
        )
        if "is_pad" in pad_var.columns:
            pad_var["is_pad"] = True
        if "gene_block_role" in pad_var.columns:
            pad_var["gene_block_role"] = "pad"
        if "gene_block_id" in pad_var.columns:
            pad_var["gene_block_id"] = -1
        full_var = pd.concat([adata.var.copy(), pad_var], axis=0)
    else:
        full_var = pd.DataFrame(index=pd.Index(existing_names + pad_names, dtype="object"))

    full_var.index = pd.Index(existing_names + pad_names, dtype="object")
    padded = ad.AnnData(
        X=new_x,
        obs=adata.obs.copy(),
        var=full_var,
        uns=dict(adata.uns),
        obsm=dict(adata.obsm) if len(adata.obsm) > 0 else None,
    )
    return padded, pad_idx


def trim_padded_genes(adata: ad.AnnData, pad_idx: int | None = None) -> ad.AnnData:
    """Slice away trailing padded ``pad_*`` genes and return a copy."""
    idx = find_pad_index(adata) if pad_idx is None else int(pad_idx)
    idx = max(0, min(idx, int(adata.n_vars)))
    return adata[:, :idx].copy()


def _align_matrix_to_mapping(
    mat: Any,
    n_obs: int,
    n_src: int,
    n_target: int,
    src_idx: np.ndarray,
    target_idx: np.ndarray,
) -> Any:
    if sparse.issparse(mat):
        proj = sparse.csc_matrix(
            (np.ones(len(src_idx), dtype=mat.dtype), (src_idx, target_idx)),
            shape=(n_src, n_target),
            dtype=mat.dtype,
        )
        return (mat.tocsr() @ proj).tocsr()
    src_arr = np.asarray(mat)
    out = np.zeros((n_obs, n_target), dtype=src_arr.dtype)
    if len(src_idx) > 0:
        out[:, target_idx] = src_arr[:, src_idx]
    return out


def align_genes(
    adata: ad.AnnData,
    target_genes: Sequence[str],
    *,
    pad_missing: bool = True,
) -> ad.AnnData:
    """Align ``adata`` columns to match ``target_genes`` exactly, optionally zero-padding missing genes."""
    target_list = [str(g) for g in target_genes]
    src_names = [str(g) for g in adata.var_names]
    if src_names == target_list:
        return adata.copy()

    src_map: dict[str, int] = {}
    for pos, name in enumerate(src_names):
        src_map.setdefault(name, pos)

    missing = [g for g in target_list if g not in src_map]
    if not missing and len(src_map) == len(src_names):
        return adata[:, target_list].copy()

    if missing and not pad_missing:
        raise ValueError(
            f"Missing {len(missing)} required target genes (first 10): {missing[:10]}"
        )

    matched_pairs = [
        (src_map[g], tgt_pos)
        for tgt_pos, g in enumerate(target_list)
        if g in src_map
    ]
    if matched_pairs:
        src_idx = np.asarray([p[0] for p in matched_pairs], dtype=np.int64)
        target_idx = np.asarray([p[1] for p in matched_pairs], dtype=np.int64)
    else:
        src_idx = np.empty(0, dtype=np.int64)
        target_idx = np.empty(0, dtype=np.int64)

    new_x = _align_matrix_to_mapping(
        adata.X,
        adata.n_obs,
        adata.n_vars,
        len(target_list),
        src_idx,
        target_idx,
    )
    new_layers = {
        layer_name: _align_matrix_to_mapping(
            layer_mat,
            adata.n_obs,
            adata.n_vars,
            len(target_list),
            src_idx,
            target_idx,
        )
        for layer_name, layer_mat in adata.layers.items()
        if layer_name is not None
    }
    new_var = pd.DataFrame(index=pd.Index(target_list, dtype="object"))
    return ad.AnnData(
        X=new_x,
        obs=adata.obs.copy(),
        var=new_var,
        uns=dict(adata.uns),
        obsm=dict(adata.obsm) if len(adata.obsm) > 0 else None,
        layers=new_layers if new_layers else None,
    )


def _extract_matrix_values(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.data)
    return np.asarray(X).ravel()


def _looks_like_raw_counts(adata: ad.AnnData) -> bool:
    if "log1p" in adata.uns:
        return False
    vals = _extract_matrix_values(adata.X)
    if vals.size == 0:
        return False
    sample = vals[: min(vals.size, 50_000)]
    if np.any(sample < -1e-6):
        return False
    if np.allclose(sample, np.round(sample), atol=1e-6):
        return True
    return float(sample.max()) > 50.0


def normalize_log1p(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    *,
    normalize_total: bool = True,
    preserve_counts_layer: bool = True,
) -> ad.AnnData:
    """Apply log1p (and optionally normalize_total) if raw/unlogged; no-op if already log1p."""
    if adata.n_obs == 0 or adata.n_vars == 0 or "log1p" in adata.uns:
        return adata

    x_max = float(adata.X.max())
    is_raw = _looks_like_raw_counts(adata)
    if is_raw:
        if preserve_counts_layer and "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    elif x_max > float(np.log1p(target_sum)) + 1e-4:
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    return adata


def select_hvgs_exact(
    adata: ad.AnnData,
    n_top_genes: int,
    *,
    batch_key: str | None = None,
    max_rounds: int = 8,
) -> ad.AnnData:
    """Select exact ``n_top_genes`` HVGs with tie-breaking fallback across Scanpy rounds."""
    if adata.n_vars <= n_top_genes:
        return adata.copy()

    cur = adata.copy()
    if sparse.issparse(cur.X) and not sparse.isspmatrix_csr(cur.X):
        cur.X = cur.X.tocsr()

    active_batch_key = batch_key
    if active_batch_key is not None and active_batch_key in cur.obs.columns:
        counts = cur.obs[active_batch_key].value_counts()
        if counts.empty or (counts < 2).any():
            active_batch_key = None
    else:
        active_batch_key = None

    for _ in range(max_rounds):
        if cur.n_vars == n_top_genes:
            return cur
        kwargs: dict[str, Any] = {
            "n_top_genes": n_top_genes,
            "flavor": "seurat",
            "inplace": False,
        }
        if active_batch_key is not None:
            kwargs["batch_key"] = active_batch_key
        try:
            hvg_df = sc.pp.highly_variable_genes(cur, **kwargs)
        except Exception:
            if active_batch_key is not None:
                kwargs.pop("batch_key", None)
                active_batch_key = None
                hvg_df = sc.pp.highly_variable_genes(cur, **kwargs)
            else:
                raise
        selected_mask = np.asarray(hvg_df["highly_variable"], dtype=bool)
        n_sel = int(selected_mask.sum())
        if 0 < n_sel <= n_top_genes:
            cur = cur[:, selected_mask].copy()
            if cur.n_vars == n_top_genes:
                return cur
            break
        if n_sel == cur.n_vars:
            score_col = "dispersions_norm" if "dispersions_norm" in hvg_df else "dispersions"
            scores = (
                pd.to_numeric(hvg_df[score_col], errors="coerce")
                .fillna(-np.inf)
                .to_numpy()
            )
            top_idx = np.sort(np.argsort(-scores, kind="mergesort")[:n_top_genes])
            return cur[:, top_idx].copy()
        cur = cur[:, selected_mask].copy()

    if cur.n_vars > n_top_genes:
        hvg_df = sc.pp.highly_variable_genes(
            cur,
            n_top_genes=n_top_genes,
            flavor="seurat",
            inplace=False,
        )
        score_col = "dispersions_norm" if "dispersions_norm" in hvg_df else "dispersions"
        scores = (
            pd.to_numeric(hvg_df[score_col], errors="coerce")
            .fillna(-np.inf)
            .to_numpy()
        )
        top_idx = np.sort(np.argsort(-scores, kind="mergesort")[:n_top_genes])
        cur = cur[:, top_idx].copy()
    return cur


def preprocess_adata(
    adata: ad.AnnData,
    *,
    min_genes: int | None = 600,
    min_cells: int | None = 3,
    target_sum: float = 1e4,
    n_top_genes: int | Sequence[str] | None = 2000,
    batch_key: str | None = None,
    exclude_gene_prefixes: Sequence[str] = ("ERCC", "MT-", "mt-"),
    normalize_total: bool = True,
    preserve_counts_layer: bool = True,
    preserve_raw: bool = False,
    scale: bool = False,
    copy: bool = True,
) -> ad.AnnData:
    """Filter, log1p-normalize, select HVGs or align genes, and optionally scale an AnnData."""
    cur = adata.copy() if copy else adata
    if sparse.issparse(cur.X) and not sparse.isspmatrix_csr(cur.X):
        cur.X = cur.X.tocsr()

    if exclude_gene_prefixes:
        prefixes = tuple(exclude_gene_prefixes)
        keep_genes = [g for g in cur.var_names if not str(g).startswith(prefixes)]
        cur = cur[:, keep_genes].copy()

    if min_genes is not None and min_genes > 0:
        sc.pp.filter_cells(cur, min_genes=int(min_genes))
    if min_cells is not None and min_cells > 0:
        sc.pp.filter_genes(cur, min_cells=int(min_cells))

    cur = normalize_log1p(
        cur,
        target_sum=target_sum,
        normalize_total=normalize_total,
        preserve_counts_layer=preserve_counts_layer,
    )
    if preserve_raw:
        cur.raw = cur

    if isinstance(n_top_genes, (int, np.integer)) and int(n_top_genes) > 0:
        cur = select_hvgs_exact(
            cur,
            n_top_genes=int(n_top_genes),
            batch_key=batch_key,
        )
    elif n_top_genes is not None and not isinstance(n_top_genes, (int, np.integer)):
        cur = align_genes(cur, [str(g) for g in n_top_genes], pad_missing=True)

    if scale:
        scaler = MaxAbsExpressionScaler().fit(cur.X)
        cur.X = scaler.transform(cur.X)

    return cur


__all__ = [
    "ConditionVocabulary",
    "MaxAbsExpressionScaler",
    "align_genes",
    "find_pad_index",
    "normalize_log1p",
    "pad_to_block_size",
    "preprocess_adata",
    "select_hvgs_exact",
    "trim_padded_genes",
]
