"""Evaluation metrics and benchmark evaluators for sc_save."""

from __future__ import annotations

from sc_save.eval.bec import (
    _compat_scanpy_pca_for_scib,
    _pandas_value_counts_compat,
    compute_scib_summary,
    scib_eval,
)
from sc_save.eval.generation import (
    CANCER_LUNG_MACRO_COND_COLS,
    CANCER_LUNG_UNSEEN_CONDITIONS,
    DEFAULT_COND_COLS,
    DEFAULT_OUTPUT_DIR,
    run_conditional_distribution_evaluation,
)
from sc_save.eval.metrics import (
    clear_gpu_memory,
    compute_batched_linear_mmd,
    compute_distribution_distances_batched,
    compute_evaluation_metrics_fast,
    linear_mmd2,
    mix_rbf_mmd2,
    wasserstein,
)
from sc_save.eval.perturbation import evaluate_perturbation

__all__ = [
    "CANCER_LUNG_MACRO_COND_COLS",
    "CANCER_LUNG_UNSEEN_CONDITIONS",
    "DEFAULT_COND_COLS",
    "DEFAULT_OUTPUT_DIR",
    "_compat_scanpy_pca_for_scib",
    "_pandas_value_counts_compat",
    "clear_gpu_memory",
    "compute_batched_linear_mmd",
    "compute_distribution_distances_batched",
    "compute_evaluation_metrics_fast",
    "compute_scib_summary",
    "evaluate_perturbation",
    "linear_mmd2",
    "mix_rbf_mmd2",
    "run_conditional_distribution_evaluation",
    "scib_eval",
    "wasserstein",
]
