"""Perturbation prediction evaluation utilities for sc_save."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import torch

from sc_save.eval.metrics import compute_evaluation_metrics_fast


def evaluate_perturbation(
    pred_data: ad.AnnData,
    test_raw: ad.AnnData,
    train_scaler: Any | None = None,
    condition_key: str = "condition",
    target_condition: str = "stimulated",
    data_scale: bool = True,
    device: torch.device = torch.device("cpu"),
    n_pca_components: int = 30,
    n_wd_mmd_repeats: int = 10,
    seed: int = 1202,
) -> dict[str, float]:
    """Compute R2, MSE, PCC, PCA-30 2-Wasserstein, and RBF-MMD vs ground-truth cells."""
    pred_x = pred_data.X
    if scipy.sparse.issparse(pred_x):
        pred_x = pred_x.toarray()
    if data_scale and train_scaler is not None:
        pred_inverse = train_scaler.inverse_transform(pred_x)
    else:
        pred_inverse = np.array(pred_x)

    test_stim = test_raw[
        test_raw.obs[condition_key].astype(str) == str(target_condition)
    ]
    stim_x = test_stim.X
    if scipy.sparse.issparse(stim_x):
        stim_x = stim_x.toarray()

    pred_arr = np.asarray(pred_inverse, dtype=np.float32)
    stim_arr = np.asarray(stim_x, dtype=np.float32)
    pred_mean = pred_arr.mean(axis=0).flatten()
    stim_mean = stim_arr.mean(axis=0).flatten()

    if pred_mean.size < 2 or np.std(pred_mean) == 0 or np.std(stim_mean) == 0:
        pcc_val = float("nan")
    else:
        pcc_val = float(pearsonr(pred_mean, stim_mean)[0])

    wd_val = float("nan")
    mmd_val = float("nan")
    combined = np.concatenate([pred_arr, stim_arr], axis=0)
    n_pcs = min(n_pca_components, combined.shape[0], combined.shape[1])
    if n_pcs >= 1 and pred_arr.shape[0] >= 1 and stim_arr.shape[0] >= 1:
        pc_func = PCA(n_components=n_pcs)
        pc_func.fit(combined)
        pred_pca = pc_func.transform(pred_arr)
        stim_pca = pc_func.transform(stim_arr)
        eval_num = min(pred_pca.shape[0], stim_pca.shape[0])
        rng = np.random.default_rng(seed)
        wd_vals: list[float] = []
        mmd_vals: list[float] = []
        for _ in range(max(1, n_wd_mmd_repeats)):
            pred_idx = rng.choice(pred_pca.shape[0], eval_num, replace=False)
            gt_idx = rng.choice(stim_pca.shape[0], eval_num, replace=False)
            dist_metric = compute_evaluation_metrics_fast(
                adata_real_pca_ary=stim_pca[gt_idx],
                adata_generated_pca_ary=pred_pca[pred_idx],
                model_name="SAVE",
                device=device,
                batch_size=eval_num,
            )
            wd_vals.append(float(dist_metric["2-Wasserstein_PCA"]))
            mmd_vals.append(float(dist_metric["RBF_MMD_PCA"]))
        wd_val = float(np.mean(wd_vals))
        mmd_val = float(np.mean(mmd_vals))

    return {
        "r2": float(r2_score(pred_mean, stim_mean)),
        "mse": float(mean_squared_error(pred_mean, stim_mean)),
        "pcc": pcc_val,
        "wd": wd_val,
        "mmd": mmd_val,
    }


__all__ = ["evaluate_perturbation"]
