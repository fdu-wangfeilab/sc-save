"""Unified orchestrator entrypoint for reproducing Tables 1-5 of the SAVE ICLR 2026 paper."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sc_save.workflows import reproduce as _repro

PAPER_TARGETS = _repro.PAPER_TARGETS
PERTURB_TRAIN_CONFIG = _repro.PERTURB_TRAIN_CONFIG
TABLE_DATASETS = _repro.TABLE_DATASETS
VALID_TABLES = _repro.VALID_TABLES

_build_child_task_cmd = _repro._build_child_task_cmd
_compute_deltas = _repro._compute_deltas
_ensure_all_datasets_prepared = _repro._ensure_all_datasets_prepared
_extract_dist_summary = _repro._extract_dist_summary
_is_oom_output = _repro._is_oom_output
_metric_mean = _repro._metric_mean
_run_child_task_subprocess = _repro._run_child_task_subprocess
_run_cond_gen_and_eval = _repro._run_cond_gen_and_eval
_run_table4_bec = _repro._run_table4_bec
_run_table5_perturb = _repro._run_table5_perturb

add_reproduce_arguments = _repro.add_reproduce_arguments
build_parser = _repro.build_parser
get_dataset_spec = _repro.get_dataset_spec
get_output_paths = _repro.get_output_paths
prepare_datasets = _repro.prepare_datasets
render_summary_markdown = _repro.render_summary_markdown
resolve_cpu_workers = _repro.resolve_cpu_workers
resolve_dataset_names = _repro.resolve_dataset_names
resolve_gpu_workers = _repro.resolve_gpu_workers
resolve_selected_plan = _repro.resolve_selected_plan
run_bec_integration = _repro.run_bec_integration
run_conditional_distribution_evaluation = _repro.run_conditional_distribution_evaluation
run_formal_conditional_generation = _repro.run_formal_conditional_generation
run_leave_one_out = _repro.run_leave_one_out

_SYNCED_NAMES = (
    "PAPER_TARGETS",
    "PERTURB_TRAIN_CONFIG",
    "TABLE_DATASETS",
    "VALID_TABLES",
    "_build_child_task_cmd",
    "_compute_deltas",
    "_ensure_all_datasets_prepared",
    "_extract_dist_summary",
    "_is_oom_output",
    "_metric_mean",
    "_run_child_task_subprocess",
    "_run_cond_gen_and_eval",
    "_run_table4_bec",
    "_run_table5_perturb",
    "get_dataset_spec",
    "get_output_paths",
    "prepare_datasets",
    "render_summary_markdown",
    "resolve_cpu_workers",
    "resolve_dataset_names",
    "resolve_gpu_workers",
    "resolve_selected_plan",
    "run_bec_integration",
    "run_conditional_distribution_evaluation",
    "run_formal_conditional_generation",
    "run_leave_one_out",
    "subprocess",
    "torch",
)
_LOCAL_DEFAULTS: dict[str, Any] = {name: globals()[name] for name in _SYNCED_NAMES}


def run_reproduction(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Delegate to ``sc_save.workflows.reproduce.run_reproduction`` while honoring monkeypatches."""
    ns = globals()
    saved: dict[str, Any] = {}
    for name in _SYNCED_NAMES:
        if ns.get(name) is not _LOCAL_DEFAULTS[name]:
            saved[name] = getattr(_repro, name)
            setattr(_repro, name, ns[name])
    try:
        kwargs.setdefault("_entry_script", Path(__file__).resolve())
        return _repro.run_reproduction(*args, **kwargs)
    finally:
        for name, orig in saved.items():
            setattr(_repro, name, orig)


def main() -> None:
    args = build_parser().parse_args()
    res = run_reproduction(
        tasks=args.tasks,
        datasets=args.datasets,
        data_root=args.data_root,
        output_root=args.output_root,
        results_dir=args.results_dir,
        device=args.device,
        seed=args.seed,
        skip_prepare=args.skip_prepare,
        force_prepare=args.force_prepare,
        subset_cells=args.subset_cells,
        vae_epoch=args.vae_epoch,
        fm_epoch=args.fm_epoch,
        epoch=args.epoch,
        batch_size=args.batch_size,
        max_target_batches=args.max_target_batches,
        table1_eval_split=args.table1_eval_split,
        wd_mmd_repeats=args.wd_mmd_repeats,
        overwrite_ckpt=args.overwrite_ckpt,
        dry_run=args.dry_run,
        gpu_workers=args.gpu_workers,
        cpu_workers=args.cpu_workers,
        prep_workers=args.prep_workers,
        cond_encoding=args.cond_encoding,
    )
    print(f"Summary JSON: {res['summary_json_path']}")
    print(f"Summary MD:   {res['summary_md_path']}")


if __name__ == "__main__":
    main()
