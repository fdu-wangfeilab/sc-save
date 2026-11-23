"""Unified orchestrator for reproducing Tables 1-5 of the SAVE ICLR 2026 paper."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Literal, Sequence

import torch

from sc_save.data.datasets import (
    TABLE_DATASETS,
    get_dataset_spec,
    get_output_paths,
    prepare_datasets,
    resolve_dataset_names,
)
from sc_save.eval.generation import run_conditional_distribution_evaluation
from sc_save.workflows.bec import run_bec_integration
from sc_save.workflows.generation import run_formal_conditional_generation
from sc_save.workflows.perturbation import PERTURB_TRAIN_CONFIG, run_leave_one_out

PAPER_TARGETS: dict[str, dict[str, Any]] = {
    "table1": {
        "pbmc3k": {"wd": 19.14, "mmd": 1.80, "mse": 0.08, "pcc": 0.93, "r2": 0.84},
        "dentategyrus": {"wd": 9.16, "mmd": 0.17, "mse": 0.00, "pcc": 0.99, "r2": 0.97},
        "tabula_muris": {"wd": 10.86, "mmd": 0.04, "mse": 0.00, "pcc": 0.99, "r2": 0.95},
    },
    "table2": {
        "heart": {"wd": 8.30, "mmd": 0.63, "mse": 0.01, "pcc": 0.88, "r2": 0.63},
        "pbmc": {"wd": 5.37, "mmd": 0.29, "mse": 0.02, "pcc": 0.98, "r2": 0.95},
        "lung_atlas": {"wd": 4.37, "mmd": 1.14, "mse": 0.01, "pcc": 0.91, "r2": 0.84},
    },
    "table3": {
        "lung_cancer": {
            "seen": {"wd": 3.10, "mmd": 1.06, "mse": 0.01, "pcc": 0.94, "r2": 0.88},
            "unseen": {"wd": 4.63, "mmd": 2.11, "mse": 0.04, "pcc": 0.85, "r2": 0.70},
            "macro": {"wd": 3.15, "mmd": 1.29},
        }
    },
    "table4": {
        "lung_atlas_bec": {
            "avg_bio": 0.7340,
            "avg_batch": 0.9309,
            "scib_score": 0.8128,
            "NMI_cluster/label": 0.8285,
            "ARI_cluster/label": 0.7340,
            "ASW_label": 0.6395,
            "ASW_label/batch": 0.8693,
            "PCR_batch": 0.2134,
            "graph_conn": 0.9926,
        },
        "heart_bec": {
            "avg_bio": 0.7574,
            "avg_batch": 0.8596,
            "scib_score": 0.7983,
            "NMI_cluster/label": 0.7999,
            "ARI_cluster/label": 0.8146,
            "ASW_label": 0.6579,
            "ASW_label/batch": 0.8576,
            "PCR_batch": 0.2554,
            "graph_conn": 0.8615,
        },
        "pbmc_bec": {
            "avg_bio": 0.7516,
            "avg_batch": 0.9498,
            "scib_score": 0.8309,
            "NMI_cluster/label": 0.8637,
            "ARI_cluster/label": 0.7952,
            "ASW_label": 0.5958,
            "ASW_label/batch": 0.9128,
            "PCR_batch": 0.4996,
            "graph_conn": 0.9868,
        },
    },
    "table5": {
        "pbmc_ifn": {
            "CD8T": {"pcc": 0.98, "r2": 0.97, "wd": 4.01, "mmd": 0.11, "mse": 0.00},
            "CD14+Mono": {"pcc": 0.97, "r2": 0.89, "wd": 5.03, "mmd": 0.09, "mse": 0.00},
            "FCGR3A+Mono": {"pcc": 0.91, "r2": 0.53, "wd": 7.01, "mmd": 0.22, "mse": 0.01},
            "B": {"pcc": 0.97, "r2": 0.94, "wd": 4.40, "mmd": 0.17, "mse": 0.00},
            "CD4T": {"pcc": 0.96, "r2": 0.97, "wd": 3.98, "mmd": 0.11, "mse": 0.00},
            "Dendritic": {"pcc": 0.96, "r2": 0.81, "wd": 6.55, "mmd": 0.17, "mse": 0.01},
            "NK": {"pcc": 0.96, "r2": 0.90, "wd": 5.07, "mmd": 0.09, "mse": 0.00},
            "Average": {"pcc": 0.96, "r2": 0.86, "wd": 5.15, "mmd": 0.14, "mse": 0.00},
        }
    },
}

VALID_TABLES = ("table1", "table2", "table3", "table4", "table5")


def _normalize_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def resolve_selected_plan(
    tasks: str | Sequence[str],
    datasets: str | Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve ``(table, dataset_name)`` execution pairs from ``--tasks`` and optional ``--datasets``."""
    if isinstance(tasks, str):
        raw_tasks = [t.strip().lower() for t in tasks.split(",") if t.strip()]
    else:
        raw_tasks = [t.strip().lower() for t in tasks if str(t).strip()]

    if not raw_tasks or "all" in raw_tasks:
        selected_tables = list(VALID_TABLES)
    else:
        unknown = [t for t in raw_tasks if t not in VALID_TABLES]
        if unknown:
            raise ValueError(f"Unknown tasks {unknown}. Expected any of {VALID_TABLES} or 'all'.")
        selected_tables = list(dict.fromkeys(raw_tasks))

    allowed_datasets: set[str] | None = None
    if datasets is not None:
        if isinstance(datasets, str):
            ds_str = datasets.strip()
            if ds_str:
                allowed_datasets = set(resolve_dataset_names(ds_str))
        else:
            ds_list = [d for d in datasets if str(d).strip()]
            if ds_list:
                allowed_datasets = set(resolve_dataset_names(ds_list))

    plan: list[tuple[str, str]] = []
    for table in selected_tables:
        for ds_name in TABLE_DATASETS[table]:
            if allowed_datasets is None or ds_name in allowed_datasets:
                plan.append((table, ds_name))
    return plan


def _extract_dist_summary(summary_block: dict[str, Any]) -> dict[str, float | None]:
    primary = summary_block.get("primary_metrics", summary_block)
    expr = summary_block.get("expression_metrics", summary_block)
    return {
        "wd": _metric_mean(primary, "2-Wasserstein_PCA"),
        "mmd": _metric_mean(primary, "RBF_MMD_PCA"),
        "mse": _metric_mean(expr, "mse"),
        "pcc": _metric_mean(expr, "pcc"),
        "r2": _metric_mean(expr, "r2"),
    }


def _metric_mean(container: dict[str, Any], key: str) -> float | None:
    entry = container.get(key)
    if isinstance(entry, dict):
        val = entry.get("mean")
        return None if val is None else float(val)
    if entry is None:
        return None
    return float(entry)


def _compute_deltas(
    observed: dict[str, float | None],
    target: dict[str, float],
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for metric, target_val in target.items():
        obs_val = observed.get(metric)
        deltas[metric] = None if obs_val is None else round(float(obs_val) - float(target_val), 6)
    return deltas


def resolve_cpu_workers(cpu_workers: int | str) -> int:
    """Resolve CPU worker count ('auto' or <=0 picks min(8, os.cpu_count() or 1))."""
    if isinstance(cpu_workers, str):
        token = cpu_workers.strip().lower()
        if token == "auto":
            return min(8, os.cpu_count() or 1)
        val = int(token)
    else:
        val = int(cpu_workers)
    if val <= 0:
        return min(8, os.cpu_count() or 1)
    return max(1, val)


def resolve_gpu_workers(
    gpu_workers: int | str,
    device_id: int,
    plan: list[tuple[str, str]],
) -> int:
    """Resolve concurrent GPU task count ('auto' saturates high-VRAM GPUs cleanly)."""
    if isinstance(gpu_workers, str):
        token = gpu_workers.strip().lower()
        if token != "auto":
            return max(1, int(token))
    else:
        return max(1, int(gpu_workers))

    if device_id < 0 or not torch.cuda.is_available():
        return 1
    try:
        total_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    except RuntimeError:
        return 1

    if total_gb >= 75.0:
        all_gbs400 = bool(plan) and all(t in {"table3", "table4", "table5"} for t, _ in plan)
        return 6 if all_gbs400 else 4
    if total_gb >= 36.0:
        return 2
    return 1


def _is_oom_output(text: str) -> bool:
    low = text.lower()
    return any(
        needle in low
        for needle in (
            "out of memory",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "cuda error",
        )
    )


def _ensure_all_datasets_prepared(
    plan: list[tuple[str, str]],
    data_root: Path,
    skip_prepare: bool,
    force_prepare: bool,
    subset_cells: int | None,
    prep_workers: int | str,
) -> dict[str, dict[str, Path]]:
    unique_names = list(dict.fromkeys(ds for _, ds in plan))
    paths_by_dataset: dict[str, dict[str, Path]] = {}
    needed: list[str] = []

    for ds in unique_names:
        paths = get_output_paths(ds, data_root)
        missing = [p for p in paths.values() if not p.exists()]
        if skip_prepare:
            if missing:
                missing_str = ", ".join(str(p) for p in missing)
                raise FileNotFoundError(
                    f"Prepared dataset files missing for '{ds}' (--skip-prepare set): {missing_str}"
                )
        elif force_prepare or missing:
            needed.append(ds)
        paths_by_dataset[ds] = paths

    if needed:
        prep_kwargs: dict[str, Any] = {
            "data_root": data_root,
            "force": force_prepare,
            "subset_cells": subset_cells,
        }
        if "workers" in inspect.signature(prepare_datasets).parameters:
            prep_kwargs["workers"] = prep_workers
        prepared = prepare_datasets(needed, **prep_kwargs)
        for ds, record in prepared.items():
            paths_by_dataset[ds] = record.files

    return paths_by_dataset


def _run_cond_gen_and_eval(
    table: str,
    dataset_name: str,
    paths: dict[str, Path],
    output_root: Path,
    device_id: int,
    seed: int,
    vae_epoch: int,
    fm_epoch: int,
    batch_size: int,
    table1_eval_split: str,
    overwrite_ckpt: bool,
    eval_workers: int = 0,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_name)
    cond_cols = list(spec.cond_cols)
    run_dir = output_root / table / dataset_name
    eval_dir = run_dir / "eval"

    if table == "table1":
        data_path = paths["train"]
        if table1_eval_split == "test" and "test" in paths:
            test_path: str | None = str(paths["test"])
            eval_split = "test"
        else:
            test_path = None
            eval_split = "train"
        sample_mask_prob = 0.4
    elif table == "table2":
        data_path = paths["full"]
        test_path = None
        eval_split = "train"
        sample_mask_prob = 0.4
    else:  # table3: lung_cancer
        data_path = paths["train"]
        test_path = str(paths["test"])
        eval_split = "all"
        sample_mask_prob = 0.0

    gen_kwargs: dict[str, Any] = {
        "data_path": str(data_path),
        "output_dir": str(run_dir),
        "device_id": device_id,
        "cond_cols": cond_cols,
        "task_name": dataset_name,
        "gene_block_size": spec.gene_block_size,
        "latent_block_size": spec.gene_block_size,
        "vae_epoch": vae_epoch,
        "fm_epoch": fm_epoch,
        "batch_size": batch_size,
        "sample_mask_prob": sample_mask_prob,
        "test_data_path": test_path,
        "eval_split": eval_split,
        "seed": seed,
        "overwrite_ckpt": overwrite_ckpt,
        "cond_encoding": cond_encoding,
    }
    gen_report = run_formal_conditional_generation(**gen_kwargs)

    eval_kwargs: dict[str, Any] = {
        "gt_path": gen_report["target_path"],
        "gen_path": gen_report["generation_path"],
        "output_dir": str(eval_dir),
        "cond_cols": cond_cols,
        "sort_cols": cond_cols,
        "device_id": device_id,
        "task_name": dataset_name if table == "table3" else None,
        "include_expression_metrics": True,
    }
    if "eval_workers" in inspect.signature(run_conditional_distribution_evaluation).parameters:
        eval_kwargs["eval_workers"] = eval_workers
    eval_res = run_conditional_distribution_evaluation(**eval_kwargs)
    summary = eval_res["report"]["summary"]

    if table in {"table1", "table2"}:
        observed = _extract_dist_summary(summary)
        target = PAPER_TARGETS[table][dataset_name]
        return {
            "status": "completed",
            "generation_report": gen_report,
            "eval_report_path": eval_res["report_path"],
            "observed": observed,
            "target": target,
            "delta": _compute_deltas(observed, target),
        }

    seen_obs = _extract_dist_summary(summary.get("seen", {}))
    unseen_obs = _extract_dist_summary(summary.get("unseen", {}))
    macro_block = summary.get("macro", {}).get("metrics", {})
    macro_obs = {
        "wd": _metric_mean(macro_block, "2-Wasserstein_PCA"),
        "mmd": _metric_mean(macro_block, "RBF_MMD_PCA"),
    }
    target_t3 = PAPER_TARGETS["table3"]["lung_cancer"]
    return {
        "status": "completed",
        "generation_report": gen_report,
        "eval_report_path": eval_res["report_path"],
        "observed": {
            "seen": seen_obs,
            "unseen": unseen_obs,
            "macro": macro_obs,
        },
        "target": target_t3,
        "delta": {
            "seen": _compute_deltas(seen_obs, target_t3["seen"]),
            "unseen": _compute_deltas(unseen_obs, target_t3["unseen"]),
            "macro": _compute_deltas(macro_obs, target_t3["macro"]),
        },
    }


def _run_table4_bec(
    dataset_name: str,
    paths: dict[str, Path],
    output_root: Path,
    device_id: int,
    seed: int,
    epoch: int,
    batch_size: int,
    max_target_batches: int,
    overwrite_ckpt: bool,
    scib_workers: int = 0,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_name)
    run_dir = output_root / "table4" / dataset_name
    bec_kwargs: dict[str, Any] = {
        "data_path": str(paths["full"]),
        "task_name": dataset_name,
        "output_dir": str(run_dir),
        "device_id": device_id,
        "epoch": epoch,
        "batch_size": batch_size,
        "gene_block_size": spec.gene_block_size,
        "latent_block_size": spec.gene_block_size,
        "max_target_batches": max_target_batches,
        "seed": seed,
        "overwrite_ckpt": overwrite_ckpt,
        "run_scib": True,
        "cond_encoding": cond_encoding,
    }
    if "scib_workers" in inspect.signature(run_bec_integration).parameters:
        bec_kwargs["scib_workers"] = scib_workers
    res = run_bec_integration(**bec_kwargs)
    summary = res["report"].get("summary", {})
    observed = {k: _metric_mean(summary, k) for k in summary}
    target = PAPER_TARGETS["table4"][dataset_name]
    return {
        "status": "completed",
        "report_path": res["report_path"],
        "scib_csv_path": res["scib_csv_path"],
        "observed": observed,
        "target": target,
        "delta": _compute_deltas(observed, target),
    }


def _run_table5_perturb(
    dataset_name: str,
    paths: dict[str, Path],
    output_root: Path,
    device_id: int,
    seed: int,
    epoch: int,
    batch_size: int,
    wd_mmd_repeats: int,
    workers: int | str = 1,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
) -> dict[str, Any]:
    run_dir = output_root / "table5" / dataset_name
    train_cfg = dict(PERTURB_TRAIN_CONFIG)
    train_cfg["epoch"] = epoch
    train_cfg["batch_size"] = batch_size

    loo_kwargs: dict[str, Any] = {
        "data_path": str(paths["full"]),
        "output_dir": str(run_dir),
        "device_id": device_id,
        "seed": seed,
        "train_with_ctrl": False,
        "sample_mask_prob": 0.0,
        "col_msk_threshold": 0.6,
        "n_wd_mmd_repeats": wd_mmd_repeats,
        "train_config": train_cfg,
        "cond_encoding": cond_encoding,
    }
    if "workers" in inspect.signature(run_leave_one_out).parameters:
        loo_kwargs["workers"] = workers
    summary = run_leave_one_out(**loo_kwargs)

    targets = PAPER_TARGETS["table5"]["pbmc_ifn"]
    obs_by_ct_norm = {
        _normalize_token(ct): metrics
        for ct, metrics in summary["metrics_per_cell_type"].items()
    }
    observed_mapped: dict[str, dict[str, float | None]] = {}
    deltas_mapped: dict[str, dict[str, float | None]] = {}
    for target_ct, target_vals in targets.items():
        if target_ct == "Average":
            obs_entry = dict(summary["average_metrics"])
        else:
            obs_entry = dict(obs_by_ct_norm.get(_normalize_token(target_ct), {}))
        observed_mapped[target_ct] = obs_entry
        deltas_mapped[target_ct] = _compute_deltas(obs_entry, target_vals)

    return {
        "status": "completed",
        "summary_path": str(run_dir / "summary.json"),
        "observed": observed_mapped,
        "target": targets,
        "delta": deltas_mapped,
        "raw_observed": summary,
    }


def _fmt_num(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value):
        return None
    return value


def render_summary_markdown(summary_payload: dict[str, Any]) -> str:
    lines = [
        "# SAVE ICLR 2026 Reproduction Summary",
        "",
        f"- Generated at (UTC): `{summary_payload['metadata']['created_at_utc']}`",
        f"- Dry run: `{summary_payload['metadata']['dry_run']}`",
        f"- Data root: `{summary_payload['metadata']['data_root']}`",
        "",
    ]

    runs = summary_payload.get("runs", {})
    for table in VALID_TABLES:
        if table not in runs:
            continue
        lines.append(f"## {table.upper()}")
        lines.append("")
        table_runs = runs[table]

        if table in {"table1", "table2"}:
            lines.append("| Dataset | Metric | Observed | Paper Target | Delta |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for ds_name, entry in table_runs.items():
                obs = entry.get("observed", {})
                tgt = entry.get("target", {})
                dlt = entry.get("delta", {})
                for metric, tval in tgt.items():
                    lines.append(
                        f"| {ds_name} | {metric} | {_fmt_num(obs.get(metric), 2)} | {tval:.2f} | {_fmt_num(dlt.get(metric), 2)} |"
                    )
        elif table == "table3":
            lines.append("| Split | Metric | Observed | Paper Target | Delta |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            entry = table_runs.get("lung_cancer", {})
            for split_name in ("seen", "unseen", "macro"):
                tgt = entry.get("target", {}).get(split_name, {})
                obs = entry.get("observed", {}).get(split_name, {})
                dlt = entry.get("delta", {}).get(split_name, {})
                for metric, tval in tgt.items():
                    lines.append(
                        f"| {split_name} | {metric} | {_fmt_num(obs.get(metric), 2)} | {tval:.2f} | {_fmt_num(dlt.get(metric), 2)} |"
                    )
        elif table == "table4":
            lines.append("| Dataset | Metric | Observed | Paper Target | Delta |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for ds_name, entry in table_runs.items():
                obs = entry.get("observed", {})
                tgt = entry.get("target", {})
                dlt = entry.get("delta", {})
                for metric, tval in tgt.items():
                    lines.append(
                        f"| {ds_name} | {metric} | {_fmt_num(obs.get(metric), 4)} | {tval:.4f} | {_fmt_num(dlt.get(metric), 4)} |"
                    )
        elif table == "table5":
            lines.append("| Cell Type | Metric | Observed | Paper Target | Delta |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            entry = table_runs.get("pbmc_ifn", {})
            tgt_all = entry.get("target", {})
            obs_all = entry.get("observed", {})
            dlt_all = entry.get("delta", {})
            for ct, tgt in tgt_all.items():
                obs = obs_all.get(ct, {})
                dlt = dlt_all.get(ct, {})
                for metric, tval in tgt.items():
                    lines.append(
                        f"| {ct} | {metric} | {_fmt_num(obs.get(metric), 2)} | {tval:.2f} | {_fmt_num(dlt.get(metric), 2)} |"
                    )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _build_child_task_cmd(
    table: str,
    dataset_name: str,
    data_root_path: Path,
    output_root_path: Path,
    child_results_dir: Path,
    device: int,
    seed: int,
    vae_epoch: int | None,
    fm_epoch: int | None,
    epoch: int | None,
    batch_size: int | None,
    max_target_batches: int,
    table1_eval_split: str,
    wd_mmd_repeats: int,
    overwrite_ckpt: bool,
    cpu_workers: int,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
    entry_script: Path | None = None,
) -> list[str]:
    if entry_script is not None:
        base_cmd = [sys.executable, str(entry_script)]
    else:
        base_cmd = [sys.executable, "-m", "sc_save.cli", "reproduce"]

    cmd = base_cmd + [
        "--tasks",
        table,
        "--datasets",
        dataset_name,
        "--data-root",
        str(data_root_path),
        "--output-root",
        str(output_root_path),
        "--results-dir",
        str(child_results_dir),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--skip-prepare",
        "--max-target-batches",
        str(max_target_batches),
        "--table1-eval-split",
        table1_eval_split,
        "--wd-mmd-repeats",
        str(wd_mmd_repeats),
        "--gpu-workers",
        "1",
        "--cpu-workers",
        str(cpu_workers),
        "--prep-workers",
        "1",
    ]
    if vae_epoch is not None:
        cmd.extend(["--vae-epoch", str(vae_epoch)])
    if fm_epoch is not None:
        cmd.extend(["--fm-epoch", str(fm_epoch)])
    if epoch is not None:
        cmd.extend(["--epoch", str(epoch)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if overwrite_ckpt:
        cmd.append("--overwrite-ckpt")
    if cond_encoding != "legacy":
        cmd.extend(["--cond-encoding", cond_encoding])
    return cmd


def _run_child_task_subprocess(
    table: str,
    dataset_name: str,
    cmd: list[str],
    log_path: Path,
    child_results_dir: Path,
) -> tuple[str, str, dict[str, Any] | None, bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_results_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_text = proc.stdout or ""
    log_path.write_text(output_text)
    if proc.returncode == 0:
        summary_file = child_results_dir / "iclr2026_summary.json"
        payload = json.loads(summary_file.read_text())
        return table, dataset_name, payload["runs"][table][dataset_name], False, output_text
    if _is_oom_output(output_text):
        return table, dataset_name, None, True, output_text
    tail = "\n".join(output_text.splitlines()[-30:])
    raise RuntimeError(
        f"Parallel task {table}/{dataset_name} failed (exit={proc.returncode}, log={log_path}):\n{tail}"
    )


def run_reproduction(
    tasks: str | Sequence[str] = "all",
    datasets: str | Sequence[str] | None = None,
    data_root: str | Path = "./data",
    output_root: str | Path = "./outputs/iclr2026",
    results_dir: str | Path | None = None,
    device: int = 0,
    seed: int = 1202,
    skip_prepare: bool = False,
    force_prepare: bool = False,
    subset_cells: int | None = None,
    vae_epoch: int | None = None,
    fm_epoch: int | None = None,
    epoch: int | None = None,
    batch_size: int | None = None,
    max_target_batches: int = 0,
    table1_eval_split: str = "train",
    wd_mmd_repeats: int = 10,
    overwrite_ckpt: bool = False,
    dry_run: bool = False,
    gpu_workers: int | str = "1",
    cpu_workers: int | str = "auto",
    prep_workers: int | str = 4,
    cond_encoding: Literal["legacy", "offset"] = "legacy",
    _entry_script: str | Path | None = None,
) -> dict[str, Any]:
    """Orchestrate dataset prep -> training/inference -> evaluation -> summary comparison."""
    plan = resolve_selected_plan(tasks=tasks, datasets=datasets)
    if not plan:
        raise ValueError(f"Empty execution plan for tasks={tasks!r}, datasets={datasets!r}")

    data_root_path = Path(data_root)
    output_root_path = Path(output_root)
    results_path = Path(results_dir) if results_dir is not None else output_root_path / "results"
    output_root_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    eff_vae_epoch = vae_epoch if vae_epoch is not None else (epoch if epoch is not None else 200)
    eff_fm_epoch = fm_epoch if fm_epoch is not None else (epoch if epoch is not None else 200)
    eff_single_epoch = epoch if epoch is not None else 200
    eff_batch_size = batch_size if batch_size is not None else 1024
    eff_cpu_workers = resolve_cpu_workers(cpu_workers)
    eff_gpu_workers = resolve_gpu_workers(gpu_workers, device_id=device, plan=plan)

    runs: dict[str, dict[str, Any]] = {}
    for table, _ in plan:
        runs.setdefault(table, {})

    if dry_run:
        for table, dataset_name in plan:
            expected_paths = get_output_paths(dataset_name, data_root_path)
            runs[table][dataset_name] = {
                "status": "planned",
                "expected_paths": {k: str(v) for k, v in expected_paths.items()},
                "observed": {},
                "target": PAPER_TARGETS[table][dataset_name],
                "delta": {},
            }
    else:
        paths_by_dataset = _ensure_all_datasets_prepared(
            plan=plan,
            data_root=data_root_path,
            skip_prepare=skip_prepare,
            force_prepare=force_prepare,
            subset_cells=subset_cells,
            prep_workers=prep_workers,
        )

        def _run_serial_task(table: str, dataset_name: str) -> dict[str, Any]:
            paths = paths_by_dataset[dataset_name]
            if table in {"table1", "table2", "table3"}:
                return _run_cond_gen_and_eval(
                    table=table,
                    dataset_name=dataset_name,
                    paths=paths,
                    output_root=output_root_path,
                    device_id=device,
                    seed=seed,
                    vae_epoch=eff_vae_epoch,
                    fm_epoch=eff_fm_epoch,
                    batch_size=eff_batch_size,
                    table1_eval_split=table1_eval_split,
                    overwrite_ckpt=overwrite_ckpt,
                    eval_workers=eff_cpu_workers,
                    cond_encoding=cond_encoding,
                )
            if table == "table4":
                return _run_table4_bec(
                    dataset_name=dataset_name,
                    paths=paths,
                    output_root=output_root_path,
                    device_id=device,
                    seed=seed,
                    epoch=eff_single_epoch,
                    batch_size=eff_batch_size,
                    max_target_batches=max_target_batches,
                    overwrite_ckpt=overwrite_ckpt,
                    scib_workers=eff_cpu_workers,
                    cond_encoding=cond_encoding,
                )
            if len(plan) == 1:
                if isinstance(gpu_workers, str) and gpu_workers.strip().lower() == "auto":
                    perturb_workers: int | str = "auto"
                else:
                    perturb_workers = eff_gpu_workers
            else:
                perturb_workers = 1
            return _run_table5_perturb(
                dataset_name=dataset_name,
                paths=paths,
                output_root=output_root_path,
                device_id=device,
                seed=seed,
                epoch=eff_single_epoch,
                batch_size=eff_batch_size,
                wd_mmd_repeats=wd_mmd_repeats,
                workers=perturb_workers,
                cond_encoding=cond_encoding,
            )

        if eff_gpu_workers > 1 and len(plan) > 1:
            logs_dir = output_root_path / "logs"
            tmp_results_root = output_root_path / "tmp_results"
            oom_retries: list[tuple[str, str]] = []
            task_results: dict[tuple[str, str], dict[str, Any]] = {}
            script_path = Path(_entry_script).resolve() if _entry_script is not None else None

            with ThreadPoolExecutor(max_workers=min(eff_gpu_workers, len(plan))) as executor:
                futures = []
                for table, dataset_name in plan:
                    tag = f"{table}_{dataset_name}"
                    log_path = logs_dir / f"{tag}.log"
                    child_res_dir = tmp_results_root / tag
                    cmd = _build_child_task_cmd(
                        table=table,
                        dataset_name=dataset_name,
                        data_root_path=data_root_path,
                        output_root_path=output_root_path,
                        child_results_dir=child_res_dir,
                        device=device,
                        seed=seed,
                        vae_epoch=vae_epoch,
                        fm_epoch=fm_epoch,
                        epoch=epoch,
                        batch_size=batch_size,
                        max_target_batches=max_target_batches,
                        table1_eval_split=table1_eval_split,
                        wd_mmd_repeats=wd_mmd_repeats,
                        overwrite_ckpt=overwrite_ckpt,
                        cpu_workers=eff_cpu_workers,
                        cond_encoding=cond_encoding,
                        entry_script=script_path,
                    )
                    futures.append(
                        executor.submit(
                            _run_child_task_subprocess,
                            table,
                            dataset_name,
                            cmd,
                            log_path,
                            child_res_dir,
                        )
                    )

                for fut in futures:
                    table, dataset_name, entry, is_oom, _ = fut.result()
                    if is_oom:
                        print(
                            f"[reproduce] {table}/{dataset_name} hit CUDA OOM in parallel; queuing serial retry"
                        )
                        oom_retries.append((table, dataset_name))
                    elif entry is not None:
                        task_results[(table, dataset_name)] = entry

            if oom_retries:
                if device >= 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for table, dataset_name in oom_retries:
                    task_results[(table, dataset_name)] = _run_serial_task(table, dataset_name)

            for table, dataset_name in plan:
                runs[table][dataset_name] = task_results[(table, dataset_name)]
        else:
            for table, dataset_name in plan:
                runs[table][dataset_name] = _run_serial_task(table, dataset_name)

    summary_payload: dict[str, Any] = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": tasks if isinstance(tasks, str) else list(tasks),
            "datasets": datasets if isinstance(datasets, str) or datasets is None else list(datasets),
            "data_root": str(data_root_path),
            "output_root": str(output_root_path),
            "results_dir": str(results_path),
            "device": device,
            "seed": seed,
            "dry_run": dry_run,
            "vae_epoch": eff_vae_epoch,
            "fm_epoch": eff_fm_epoch,
            "epoch": eff_single_epoch,
            "batch_size": eff_batch_size,
            "gpu_workers": eff_gpu_workers,
            "cpu_workers": eff_cpu_workers,
            "prep_workers": prep_workers,
            "cond_encoding": cond_encoding,
        },
        "paper_targets": PAPER_TARGETS,
        "runs": runs,
    }

    summary_json_path = results_path / "iclr2026_summary.json"
    summary_md_path = results_path / "iclr2026_summary.md"
    with open(summary_json_path, "w") as handle:
        json.dump(_to_jsonable(summary_payload), handle, indent=2, allow_nan=False)
    summary_md_path.write_text(render_summary_markdown(summary_payload))

    summary_payload["summary_json_path"] = str(summary_json_path)
    summary_payload["summary_md_path"] = str(summary_md_path)
    return summary_payload


def add_reproduce_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated tables to run (e.g. table1,table2,table3,table4,table5 or all).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset filter (e.g. pbmc3k,heart,lung_atlas_bec).",
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./outputs/iclr2026")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for iclr2026_summary.json & .md (defaults to <output-root>/results).",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU index (-1 forces CPU).")
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--subset-cells", type=int, default=None)
    parser.add_argument("--vae-epoch", type=int, default=None)
    parser.add_argument("--fm-epoch", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-target-batches", type=int, default=0)
    parser.add_argument(
        "--table1-eval-split",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument("--wd-mmd-repeats", type=int, default=10)
    parser.add_argument("--overwrite-ckpt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-workers",
        type=str,
        default="1",
        help="Parallel GPU tasks ('auto' saturates >=40/75 GB GPUs, or integer >=1; default '1').",
    )
    parser.add_argument(
        "--cpu-workers",
        type=str,
        default="auto",
        help="CPU worker count for scIB and distribution evaluation ('auto' or integer; default 'auto').",
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=4,
        help="Concurrent workers passed to prepare_datasets upfront (default 4).",
    )
    parser.add_argument(
        "--cond-encoding",
        type=str,
        default="legacy",
        choices=["legacy", "offset"],
        help="Condition encoding mode ('legacy' for v1 parity, 'offset' for collision-free IDs).",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Tables 1-5 of the SAVE ICLR 2026 paper."
    )
    return add_reproduce_arguments(parser)


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


__all__ = [
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
    "add_reproduce_arguments",
    "build_parser",
    "get_dataset_spec",
    "get_output_paths",
    "main",
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
    "run_reproduction",
]
