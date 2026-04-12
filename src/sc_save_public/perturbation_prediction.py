"""Perturbation prediction pipeline built on the SAVE model.

Implements a leave-one-out perturbation prediction workflow:
1. Hold out one target cell type from the dataset.
2. Train (or load) a SAVE model on the remaining cells.
3. Predict the stimulated expression of the held-out cell type from its control cells.
4. Evaluate predictions against ground truth using R2, MSE, and PCC.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MaxAbsScaler

from sc_save_public.models.save_model import SAVE

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "./outputs/perturbation_prediction"
DEFAULT_LABEL_KEY = "cell_type"
DEFAULT_CONDITION_KEY = "condition"
DEFAULT_CONTROL_CONDITION = "control"
DEFAULT_TARGET_CONDITION = "stimulated"

# SAVE-B-perturb configuration (gene_block_size=400).
PERTURB_TRAIN_CONFIG = {
    "lr": 0.0001,
    "batch_size": 1024,
    "weight_decay": 0.000025,
    "iter": 3200,
    "epoch": 200,
    "is_lr_scheduler": True,
}

PERTURB_MODEL_CONFIG = {
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
    "gene_block_size": 400,
    "latent_block_size": 400,
    "is_flash_attn": False,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id: int) -> torch.device:
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_adata(data_path: str, label_key: str, condition_key: str) -> sc.AnnData:
    """Load an h5ad dataset and validate required obs columns."""
    print(f"[pipeline] loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)
    print(f"[pipeline] full shape: {adata.shape}")
    missing = [c for c in [label_key, condition_key] if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"adata.obs is missing required columns: {missing}")
    return adata


def split_adata(
    adata: sc.AnnData,
    cell_type: str,
    label_key: str,
    condition_key: str,
    control_condition: str,
    target_condition: str,
    train_with_ctrl: bool,
):
    """Split dataset into train / test / input-control for one held-out cell type."""
    if cell_type not in set(adata.obs[label_key].astype(str)):
        raise ValueError(f"cell_type not found in {label_key}: {cell_type}")

    target_mask = adata.obs[label_key].astype(str) == str(cell_type)
    target_data = adata[target_mask].copy()
    cond_counts = target_data.obs[condition_key].astype(str).value_counts().to_dict()

    for cond, name in [(control_condition, "control"), (target_condition, "target")]:
        if cond not in cond_counts:
            raise ValueError(f"{name} condition '{cond}' missing for {cell_type}")

    if train_with_ctrl:
        heldout = target_mask & (
            adata.obs[condition_key].astype(str) == str(target_condition)
        )
        train_adata = adata[~heldout].copy()
    else:
        train_adata = adata[~target_mask].copy()

    test_adata = target_data.copy()
    input_adata = test_adata[
        test_adata.obs[condition_key].astype(str) == str(control_condition)
    ].copy()
    if input_adata.n_obs == 0:
        raise ValueError(f"no control cells for {cell_type}")

    return train_adata, test_adata, input_adata, cond_counts


def scale_adata(adata: sc.AnnData):
    """Fit a MaxAbsScaler and return (scaled_copy, scaler)."""
    scaler = MaxAbsScaler()
    scaled = adata.copy()
    scaled.X = scaler.fit_transform(scaled.X)
    return scaled, scaler


# ---------------------------------------------------------------------------
# Model creation / loading
# ---------------------------------------------------------------------------


def create_or_load_model(
    train_adata: sc.AnnData,
    cell_type: str,
    ckpt_dir: str,
    device: torch.device,
    seed: int,
    train_with_ctrl: bool,
    label_key: str,
    condition_key: str,
    train_config: dict | None = None,
    model_config: dict | None = None,
):
    """Build a SAVE model, load checkpoint if available, otherwise train."""
    t_cfg = dict(train_config or PERTURB_TRAIN_CONFIG)
    m_cfg = dict(model_config or PERTURB_MODEL_CONFIG)

    save_model = SAVE(
        adata=train_adata,
        device=device,
        is_initialized=True,
        seed=seed,
        condition_cols=[condition_key, label_key],
        **m_cfg,
    )

    base = f"{cell_type}_w_ctrl_stage1" if train_with_ctrl else f"{cell_type}_stage1"
    ckpt_path = Path(ckpt_dir) / f"{base}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_existing = ckpt_path.exists()
    trained_now = False

    if loaded_existing:
        print(f"[pipeline] loading checkpoint: {ckpt_path}")
        save_model.load_ckpt(str(ckpt_path))
    else:
        print("[pipeline] training SAVE model ...")
        save_model.train(
            batch_size=t_cfg["batch_size"],
            is_grad_clip=True,
            lr=t_cfg["lr"],
            epoch=t_cfg["epoch"],
            seed=seed,
            iter=t_cfg["iter"],
            weight_decay=t_cfg["weight_decay"],
            is_lr_scheduler=t_cfg["is_lr_scheduler"],
            col_msk_threshold=0.6,
            max_sample_mask_prob=None,
            is_col_wise_mask=False,
            loss_monitor=None,
        )
        save_model.save_ckpt(str(ckpt_path))
        trained_now = True

    return save_model, ckpt_path, loaded_existing, trained_now, t_cfg, m_cfg


# ---------------------------------------------------------------------------
# Transfer prediction
# ---------------------------------------------------------------------------


def run_transfer(
    save_model: SAVE,
    input_adata: sc.AnnData,
    label_key: str,
    condition_key: str,
    control_condition: str,
    target_condition: str,
    transfer_batch_size: int,
):
    """Run conditional transfer and return (pred_adata, attn_weights)."""
    pred = save_model.transfer_cond(
        input_adata=input_adata,
        input_cond={condition_key: control_condition, label_key: "<mask>"},
        target_cond={condition_key: target_condition, label_key: "<mask>"},
        batch_size=transfer_batch_size,
        is_output_attn_weights=True,
        verbose=False,
    )
    if not isinstance(pred, tuple) or len(pred) != 2:
        raise TypeError(
            "expected transfer_cond to return (adata, attn_weights)"
        )
    pred_data, attn_weights = pred

    pred_obs = input_adata.obs.copy()
    pred_obs[condition_key] = "SAVE_pred"
    pred_obs["source_condition"] = str(control_condition)
    pred_obs["target_condition"] = str(target_condition)

    pred_data = sc.AnnData(
        X=pred_data.X,
        obs=pred_obs,
        var=input_adata.var.copy(),
    )
    return pred_data, attn_weights


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_perturbation(
    pred_data: sc.AnnData,
    test_raw: sc.AnnData,
    train_scaler: MaxAbsScaler | None,
    condition_key: str,
    target_condition: str,
    data_scale: bool,
) -> dict:
    """Compute R2, MSE, and PCC between predicted and ground truth stimulated means.

    Predictions are inverse-transformed from scaled space to original expression
    values.  Ground truth is the raw (unscaled) stimulated cells.
    """
    # Inverse-transform predictions
    pred_X = pred_data.X
    if scipy.sparse.issparse(pred_X):
        pred_X = pred_X.toarray()
    if data_scale and train_scaler is not None:
        pred_inverse = train_scaler.inverse_transform(pred_X)
    else:
        pred_inverse = np.array(pred_X)

    # Ground truth: raw stimulated cells
    test_stim = test_raw[
        test_raw.obs[condition_key].astype(str) == str(target_condition)
    ]
    stim_X = test_stim.X
    if scipy.sparse.issparse(stim_X):
        stim_X = stim_X.toarray()

    pred_mean = np.asarray(pred_inverse.mean(axis=0)).flatten()
    stim_mean = np.asarray(np.array(stim_X).mean(axis=0)).flatten()

    return {
        "r2": float(r2_score(pred_mean, stim_mean)),
        "mse": float(mean_squared_error(pred_mean, stim_mean)),
        "pcc": float(pearsonr(pred_mean, stim_mean)[0]),
    }


# ---------------------------------------------------------------------------
# Single cell-type pipeline
# ---------------------------------------------------------------------------


def run_perturbation(
    data_path: str,
    cell_type: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device_id: int = 0,
    seed: int = 1202,
    ckpt_dir: str | None = None,
    label_key: str = DEFAULT_LABEL_KEY,
    condition_key: str = DEFAULT_CONDITION_KEY,
    control_condition: str = DEFAULT_CONTROL_CONDITION,
    target_condition: str = DEFAULT_TARGET_CONDITION,
    train_with_ctrl: bool = False,
    data_scale: bool = True,
    transfer_batch_size: int = 128,
    max_input_cells: int = 0,
    train_config: dict | None = None,
    model_config: dict | None = None,
) -> dict:
    """End-to-end perturbation prediction for one held-out cell type.

    Returns a report dict that includes evaluation metrics.
    """
    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device_id)

    effective_ckpt_dir = ckpt_dir or str(out_dir / "ckpt")

    # --- data ---
    adata = load_adata(data_path, label_key, condition_key)
    train_raw, test_raw, input_raw, cond_counts = split_adata(
        adata, cell_type, label_key, condition_key,
        control_condition, target_condition, train_with_ctrl,
    )

    train_scaler = None
    if data_scale:
        print("[pipeline] scaling train split with MaxAbsScaler")
        train_scaled, train_scaler = scale_adata(train_raw)
        print("[pipeline] scaling test split with MaxAbsScaler")
        test_scaled, _ = scale_adata(test_raw)
    else:
        train_scaled = train_raw.copy()
        test_scaled = test_raw.copy()

    input_scaled = test_scaled[
        test_scaled.obs[condition_key].astype(str) == str(control_condition)
    ].copy()
    if max_input_cells > 0:
        input_scaled = input_scaled[:max_input_cells].copy()
        input_raw = input_raw[:max_input_cells].copy()

    # --- model ---
    (
        save_model, ckpt_path, loaded_existing, trained_now, t_cfg, m_cfg,
    ) = create_or_load_model(
        train_adata=train_scaled,
        cell_type=cell_type,
        ckpt_dir=effective_ckpt_dir,
        device=device,
        seed=seed,
        train_with_ctrl=train_with_ctrl,
        label_key=label_key,
        condition_key=condition_key,
        train_config=train_config,
        model_config=model_config,
    )

    # --- predict ---
    t0 = time.time()
    pred_data, attn_weights = run_transfer(
        save_model, input_scaled, label_key, condition_key,
        control_condition, target_condition, transfer_batch_size,
    )
    transfer_seconds = time.time() - t0

    # --- evaluate ---
    metrics = evaluate_perturbation(
        pred_data, test_raw, train_scaler,
        condition_key, target_condition, data_scale,
    )
    print(
        f"[pipeline] metrics: R2={metrics['r2']:.4f}  "
        f"MSE={metrics['mse']:.6f}  PCC={metrics['pcc']:.4f}"
    )

    # --- save artifacts ---
    pred_tag = "SAVE_pred_w_ctrl" if train_with_ctrl else "SAVE_pred"
    train_raw.write_h5ad(out_dir / "train.h5ad")
    test_raw.write_h5ad(out_dir / "test.h5ad")
    pred_data.write_h5ad(out_dir / f"{pred_tag}.h5ad")
    with open(out_dir / f"{pred_tag}_attn_weights.pkl", "wb") as f:
        pickle.dump(attn_weights, f)

    scaler_path = None
    if train_scaler is not None:
        scaler_path = out_dir / "train_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(train_scaler, f)

    # --- report ---
    report = {
        "cell_type": cell_type,
        "data_path": data_path,
        "output_dir": str(out_dir),
        "dataset_shape": list(adata.shape),
        "train_shape": list(train_raw.shape),
        "test_shape": list(test_raw.shape),
        "input_shape": list(input_scaled.shape),
        "pred_shape": list(pred_data.shape),
        "label_key": label_key,
        "condition_key": condition_key,
        "control_condition": control_condition,
        "target_condition": target_condition,
        "condition_counts": {str(k): int(v) for k, v in cond_counts.items()},
        "train_with_ctrl": train_with_ctrl,
        "data_scale": data_scale,
        "transfer_batch_size": transfer_batch_size,
        "max_input_cells": max_input_cells,
        "device": str(device),
        "ckpt_path": str(ckpt_path),
        "loaded_existing_ckpt": loaded_existing,
        "trained_now": trained_now,
        "transfer_seconds": round(transfer_seconds, 3),
        "metrics": metrics,
        "train_config": t_cfg,
        "model_config": m_cfg,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[pipeline] done — saved to {out_dir}")
    return report


# ---------------------------------------------------------------------------
# Leave-one-out loop
# ---------------------------------------------------------------------------


def run_leave_one_out(
    data_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device_id: int = 0,
    seed: int = 1202,
    ckpt_dir: str | None = None,
    label_key: str = DEFAULT_LABEL_KEY,
    condition_key: str = DEFAULT_CONDITION_KEY,
    control_condition: str = DEFAULT_CONTROL_CONDITION,
    target_condition: str = DEFAULT_TARGET_CONDITION,
    train_with_ctrl: bool = False,
    data_scale: bool = True,
    transfer_batch_size: int = 128,
    max_input_cells: int = 0,
    cell_types: list[str] | None = None,
    train_config: dict | None = None,
    model_config: dict | None = None,
) -> dict:
    """Run perturbation prediction for every cell type and aggregate metrics.

    If *cell_types* is ``None``, all cell types in the dataset are used.
    Returns a summary dict with per-cell-type and average metrics.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover cell types if not specified
    if cell_types is None:
        adata = load_adata(data_path, label_key, condition_key)
        cell_types = sorted(adata.obs[label_key].astype(str).unique().tolist())

    print(f"[leave-one-out] cell types: {cell_types}")
    all_metrics = {}

    for ct in cell_types:
        print(f"\n{'=' * 60}")
        print(f"[leave-one-out] {ct}")
        print(f"{'=' * 60}")
        ct_dir = str(out_root / ct.replace("+", "_plus_"))
        report = run_perturbation(
            data_path=data_path,
            cell_type=ct,
            output_dir=ct_dir,
            device_id=device_id,
            seed=seed,
            ckpt_dir=ckpt_dir,
            label_key=label_key,
            condition_key=condition_key,
            control_condition=control_condition,
            target_condition=target_condition,
            train_with_ctrl=train_with_ctrl,
            data_scale=data_scale,
            transfer_batch_size=transfer_batch_size,
            max_input_cells=max_input_cells,
            train_config=train_config,
            model_config=model_config,
        )
        all_metrics[ct] = report["metrics"]

    # Summary table
    print(f"\n{'=' * 60}")
    print("[leave-one-out] Summary")
    print(f"{'=' * 60}")
    print(f"{'Cell Type':<20} {'R2':>10} {'MSE':>12} {'PCC':>10}")
    print("-" * 56)
    for ct in cell_types:
        m = all_metrics[ct]
        print(f"{ct:<20} {m['r2']:>10.4f} {m['mse']:>12.6f} {m['pcc']:>10.4f}")

    n = len(all_metrics)
    avg = {
        k: sum(m[k] for m in all_metrics.values()) / n
        for k in ("r2", "mse", "pcc")
    }
    print("-" * 56)
    print(f"{'Average':<20} {avg['r2']:>10.4f} {avg['mse']:>12.6f} {avg['pcc']:>10.4f}")

    summary = {
        "cell_types": cell_types,
        "metrics_per_cell_type": all_metrics,
        "average_metrics": avg,
    }
    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[leave-one-out] saved summary to: {summary_path}")

    return summary
