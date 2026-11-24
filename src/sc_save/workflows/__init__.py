"""End-to-end benchmark and application workflows for ``sc_save``."""

from __future__ import annotations

from sc_save.workflows.bec import (
    DEFAULT_BEC_OUTPUT_DIR,
    SAVE_B_PERTURB_MODEL_CONFIG,
    SAVE_B_PERTURB_TRAIN_CONFIG,
    compute_scib_summary,
    resolve_scib_workers,
    resolve_transfer_batch_size,
    run_bec_integration,
    scib_eval,
)
from sc_save.workflows.gene_blocks import run_gene_block_construction
from sc_save.workflows.generation import (
    DEFAULT_COND_COLS,
    DEFAULT_FORMAL_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    run_formal_conditional_generation,
    run_formal_generation,
    run_hca_formal,
    run_hca_smoke,
)
from sc_save.workflows.perturbation import (
    PERTURB_MODEL_CONFIG,
    PERTURB_TRAIN_CONFIG,
    evaluate_perturbation,
    resolve_perturbation_workers,
    run_leave_one_out,
    run_perturbation,
)
from sc_save.workflows.reproduce import (
    PAPER_TARGETS,
    VALID_TABLES,
    render_summary_markdown,
    resolve_cpu_workers,
    resolve_gpu_workers,
    resolve_selected_plan,
    run_reproduction,
)

__all__ = [
    "DEFAULT_BEC_OUTPUT_DIR",
    "DEFAULT_COND_COLS",
    "DEFAULT_FORMAL_OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "PAPER_TARGETS",
    "PERTURB_MODEL_CONFIG",
    "PERTURB_TRAIN_CONFIG",
    "SAVE_B_PERTURB_MODEL_CONFIG",
    "SAVE_B_PERTURB_TRAIN_CONFIG",
    "VALID_TABLES",
    "compute_scib_summary",
    "evaluate_perturbation",
    "render_summary_markdown",
    "resolve_cpu_workers",
    "resolve_gpu_workers",
    "resolve_perturbation_workers",
    "resolve_scib_workers",
    "resolve_selected_plan",
    "resolve_transfer_batch_size",
    "run_bec_integration",
    "run_formal_conditional_generation",
    "run_formal_generation",
    "run_gene_block_construction",
    "run_hca_formal",
    "run_hca_smoke",
    "run_leave_one_out",
    "run_perturbation",
    "run_reproduction",
    "scib_eval",
]
