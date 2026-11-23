"""Backward-compatibility shim for :mod:`sc_save.eval.metrics`."""

from __future__ import annotations

from sc_save.eval.metrics import (
    clear_gpu_memory,
    compute_batched_linear_mmd,
    compute_distribution_distances_batched,
    compute_evaluation_metrics_fast,
    linear_mmd2,
    mix_rbf_mmd2,
    wasserstein,
)

__all__ = [
    "clear_gpu_memory",
    "compute_batched_linear_mmd",
    "compute_distribution_distances_batched",
    "compute_evaluation_metrics_fast",
    "linear_mmd2",
    "mix_rbf_mmd2",
    "wasserstein",
]
