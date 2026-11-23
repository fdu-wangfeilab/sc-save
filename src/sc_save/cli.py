"""Unified ``sc-save`` CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from sc_save import __version__


def _parse_device_arg(raw: str | int) -> str | int:
    if isinstance(raw, int):
        return raw
    token = str(raw).strip()
    try:
        return int(token)
    except ValueError:
        return token


def _parse_cond_pairs(pairs: Sequence[str]) -> dict[str, str]:
    cond: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected condition pair 'col=value', got {item!r}.")
        key, val = item.split("=", 1)
        cond[key.strip()] = val.strip()
    return cond


def _build_prepare_datasets_parser(subparsers: argparse._SubParsersAction) -> None:
    from sc_save.data.datasets import DEFAULT_OT_NUM_ITER_MAX

    p = subparsers.add_parser(
        "prepare-datasets",
        help="Download, standardize, normalize, and OT-partition benchmark datasets.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset names, aliases, table groups (table1..table5), 'genept', or 'all'.",
    )
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--gene-embedding-path", type=str, default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--subset-cells", type=int, default=None)
    p.add_argument("--ot-method", choices=["emd", "sinkhorn"], default="emd")
    p.add_argument("--ot-reg", type=float, default=0.1)
    p.add_argument("--ot-max-iter", type=int, default=20)
    p.add_argument("--ot-num-iter-max", type=int, default=DEFAULT_OT_NUM_ITER_MAX)
    p.add_argument("--min-genes", type=int, default=None)
    p.add_argument("--min-cells", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=str, default="1")
    p.add_argument("--list-datasets", action="store_true")
    p.set_defaults(handler=_handle_prepare_datasets)


def _handle_prepare_datasets(args: argparse.Namespace) -> int:
    from sc_save.data.datasets import (
        GENEPT_ZENODO_URL,
        ensure_genept_embeddings,
        get_output_paths,
        list_datasets,
        prepare_datasets,
    )

    data_root = Path(args.data_root)
    if args.list_datasets:
        print(f"GenePT v2 archive URL: {GENEPT_ZENODO_URL}")
        print(f"Data root: {data_root}\n")
        for spec in list_datasets():
            outputs = {
                k: str(v.relative_to(data_root))
                for k, v in get_output_paths(spec.name, data_root).items()
            }
            print(
                f"- {spec.name} ({spec.table} / {spec.task_type}): "
                f"GBS={spec.gene_block_size}, HVG={spec.hvg_top_genes}, "
                f"cond_cols={list(spec.cond_cols)}"
            )
            print(f"  sources: {list(spec.source_urls)}")
            print(f"  outputs: {outputs}")
        return 0

    raw_tokens: list[str] = []
    for item in args.datasets:
        raw_tokens.extend([tok.strip() for tok in item.split(",") if tok.strip()])

    non_genept = [tok for tok in raw_tokens if tok.lower() != "genept"]
    if raw_tokens and not non_genept:
        emb_path = ensure_genept_embeddings(
            data_root,
            gene_embedding_path=args.gene_embedding_path,
            force=args.force,
            skip_download=args.skip_download,
        )
        print(json.dumps({"gene_embedding_path": str(emb_path)}, indent=2))
        return 0

    results = prepare_datasets(
        datasets=non_genept or raw_tokens,
        data_root=data_root,
        gene_embedding_path=args.gene_embedding_path,
        skip_download=args.skip_download,
        force=args.force,
        subset_cells=args.subset_cells,
        ot_method=args.ot_method,
        ot_reg=args.ot_reg,
        ot_max_iter=args.ot_max_iter,
        ot_num_iter_max=args.ot_num_iter_max,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        seed=args.seed,
        workers=args.workers,
    )
    summary = {name: rec.to_dict(data_root=data_root) for name, rec in results.items()}
    print(json.dumps(summary, indent=2))
    return 0


def _build_blocks_parser(subparsers: argparse._SubParsersAction) -> None:
    from sc_save.data.gene_blocks import DEFAULT_OT_NUM_ITER_MAX, DEFAULT_OUTPUT_DIR

    p = subparsers.add_parser(
        "build-blocks",
        help="Construct balanced semantic gene blocks from precomputed embeddings.",
    )
    p.add_argument("--data-path", type=str, required=True, help="Input AnnData .h5ad file.")
    p.add_argument(
        "--gene-embedding-path",
        type=str,
        required=True,
        help="Pickle file mapping gene names to embedding vectors.",
    )
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gene-block-size", type=int, default=3200)
    p.add_argument("--gene-name-col", type=str, default=None)
    p.add_argument("--subset-cells", type=int, default=0)
    p.add_argument("--ot-method", type=str, default="emd")
    p.add_argument("--ot-reg", type=float, default=0.1)
    p.add_argument("--ot-max-iter", type=int, default=20)
    p.add_argument("--ot-num-iter-max", type=int, default=DEFAULT_OT_NUM_ITER_MAX)
    p.add_argument("--ot-random-state", type=int, default=42)
    p.set_defaults(handler=_handle_build_blocks)


def _handle_build_blocks(args: argparse.Namespace) -> int:
    from sc_save.workflows.gene_blocks import run_gene_block_construction

    run_gene_block_construction(
        data_path=args.data_path,
        gene_embedding_path=args.gene_embedding_path,
        output_dir=args.output_dir,
        gene_block_size=args.gene_block_size,
        gene_name_col=args.gene_name_col,
        subset_cells=args.subset_cells,
        ot_method=args.ot_method,
        ot_reg=args.ot_reg,
        ot_max_iter=args.ot_max_iter,
        ot_num_iter_max=args.ot_num_iter_max,
        ot_random_state=args.ot_random_state,
    )
    return 0


def _build_train_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train a SAVE model (VAE + optional Flow Matching) and save a reusable bundle.",
    )
    p.add_argument("--data-path", type=str, required=True, help="Input .h5ad file.")
    p.add_argument("--output-dir", type=str, default="./outputs/save_model")
    p.add_argument("--cond-cols", nargs="+", default=["batch", "cell_type"])
    p.add_argument(
        "--preset",
        choices=["save_b", "save_b_perturb", "smoke"],
        default="save_b",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--gene-block-size", type=int, default=None)
    p.add_argument("--latent-block-size", type=int, default=None)
    p.add_argument("--vae-epoch", type=int, default=None)
    p.add_argument("--fm-epoch", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-fm", action="store_true", help="Train Stage-1 VAE only.")
    p.add_argument("--cond-encoding", choices=["legacy", "offset"], default="legacy")
    p.add_argument("--no-scale", dest="scale_data", action="store_false")
    p.add_argument("--no-pad", dest="pad_to_blocks", action="store_false")
    p.set_defaults(scale_data=True, pad_to_blocks=True)
    p.add_argument("--seed", type=int, default=1202)
    p.set_defaults(handler=_handle_train)


def _handle_train(args: argparse.Namespace) -> int:
    import scanpy as sc

    from sc_save.config import SAVEConfig
    from sc_save.model import SAVE

    adata = sc.read_h5ad(args.data_path)
    preset_fn = {
        "save_b": SAVEConfig.save_b,
        "save_b_perturb": SAVEConfig.save_b_perturb,
        "smoke": SAVEConfig.smoke,
    }[args.preset]

    cfg_kwargs: dict[str, Any] = {
        "cond_encoding": args.cond_encoding,
        "seed": args.seed,
    }
    if args.gene_block_size is not None:
        cfg_kwargs["gene_block_size"] = args.gene_block_size
    if args.latent_block_size is not None:
        cfg_kwargs["latent_block_size"] = args.latent_block_size

    cfg = preset_fn(**cfg_kwargs)
    if args.vae_epoch is not None:
        cfg.train.vae_epochs = int(args.vae_epoch)
    if args.fm_epoch is not None:
        cfg.train.fm_epochs = int(args.fm_epoch)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)

    model = SAVE(
        adata=adata,
        condition_cols=args.cond_cols,
        config=cfg,
        device=_parse_device_arg(args.device),
        scale_data=args.scale_data,
        pad_to_blocks=args.pad_to_blocks,
        cond_encoding=args.cond_encoding,
    )
    model.fit(train_fm=not args.no_fm)
    saved_dir = model.save(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(saved_dir),
                "num_cells": model.num_cells,
                "num_genes": model.num_genes,
                "condition_cols": model.condition_cols,
                "cond_encoding": model.cond_encoding,
                "is_trained": model.is_trained,
                "is_fm_trained": model.is_fm_trained,
            },
            indent=2,
        )
    )
    return 0


def _build_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    from sc_save.workflows.generation import DEFAULT_COND_COLS, DEFAULT_FORMAL_OUTPUT_DIR

    p = subparsers.add_parser(
        "generate",
        help="Run SAVE conditional generation (from --model-dir or train+generate via --data-path).",
    )
    p.add_argument("--data-path", type=str, default=None, help="Training .h5ad file.")
    p.add_argument("--model-dir", type=str, default=None, help="Saved SAVE model directory.")
    p.add_argument(
        "--target-data-path",
        type=str,
        default=None,
        help="Target .h5ad for generate_like when using --model-dir.",
    )
    p.add_argument(
        "--cond",
        nargs="*",
        default=None,
        help="Condition pairs col=val when sampling from --model-dir.",
    )
    p.add_argument("--n-cells", type=int, default=100)
    p.add_argument("--solver", type=str, default="euler")
    p.add_argument("--save-latents", action="store_true")
    p.add_argument("--test-data-path", type=str, default=None)
    p.add_argument(
        "--eval-split",
        type=str,
        default="auto",
        choices=["auto", "train", "test", "all"],
    )
    p.add_argument("--output-dir", type=str, default=DEFAULT_FORMAL_OUTPUT_DIR)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--task-name", type=str, default="conditional_generation")
    p.add_argument("--cond-cols", nargs="+", default=list(DEFAULT_COND_COLS))
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--gene-block-size", type=int, default=3200)
    p.add_argument("--latent-block-size", type=int, default=None)
    p.add_argument("--vae-epoch", type=int, default=200)
    p.add_argument("--fm-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=2.5e-5)
    p.add_argument("--kl-scale", type=float, default=0.5)
    p.add_argument("--expand-dim", type=int, default=512)
    p.add_argument("--enc-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--n-fm-blocks", type=int, default=5)
    p.add_argument("--guidance-weights", type=float, default=2.0)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--sample-mask-prob", type=float, default=None)
    p.add_argument("--vae-col-msk-threshold", type=float, default=1.0)
    p.add_argument("--fm-col-msk-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1202)
    p.add_argument("--subset-size", type=int, default=0)
    p.add_argument("--overwrite-ckpt", action="store_true")
    p.add_argument("--cond-encoding", choices=["legacy", "offset"], default="legacy")
    p.set_defaults(handler=_handle_generate)


def _handle_generate(args: argparse.Namespace) -> int:
    if args.model_dir is not None:
        import numpy as np
        import scanpy as sc

        from sc_save.model import SAVE

        model = SAVE.load(args.model_dir, device=_parse_device_arg(args.device))
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.target_data_path is not None:
            target = sc.read_h5ad(args.target_data_path)
            result = model.generate_like(
                target,
                guidance_weight=args.guidance_weights,
                steps=args.sample_steps,
                solver=args.solver,
                batch_size=args.batch_size,
                return_latents=args.save_latents,
            )
        elif args.cond is not None:
            cond_dict = _parse_cond_pairs(args.cond)
            result = model.generate(
                conditions=cond_dict,
                n_cells=args.n_cells,
                guidance_weight=args.guidance_weights,
                steps=args.sample_steps,
                solver=args.solver,
                batch_size=args.batch_size,
                return_latents=args.save_latents,
            )
        else:
            raise ValueError("Provide --target-data-path or --cond with --model-dir.")

        if args.save_latents:
            gen_adata, z_fm = result
            np.save(out_dir / "gen_z_fm.npy", z_fm)
        else:
            gen_adata = result
        gen_path = out_dir / "generation.h5ad"
        gen_adata.write_h5ad(gen_path)
        print(json.dumps({"generation_path": str(gen_path), "shape": list(gen_adata.shape)}, indent=2))
        return 0

    if args.data_path is None:
        raise ValueError("Provide --data-path (or --model-dir) to `sc-save generate`.")

    from sc_save.workflows.generation import run_formal_conditional_generation

    dev = _parse_device_arg(args.device)
    if isinstance(dev, str):
        dev = -1 if dev == "cpu" else 0
    report = run_formal_conditional_generation(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device_id=dev,
        cond_cols=args.cond_cols,
        task_name=args.task_name,
        gene_block_size=args.gene_block_size,
        latent_block_size=args.latent_block_size,
        vae_epoch=args.vae_epoch,
        fm_epoch=args.fm_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        kl_scale=args.kl_scale,
        expand_dim=args.expand_dim,
        enc_dim=args.enc_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        n_fm_blocks=args.n_fm_blocks,
        guidance_weights=args.guidance_weights,
        sample_steps=args.sample_steps,
        sample_mask_prob=args.sample_mask_prob,
        vae_col_msk_threshold=args.vae_col_msk_threshold,
        fm_col_msk_threshold=args.fm_col_msk_threshold,
        test_data_path=args.test_data_path,
        eval_split=args.eval_split,
        seed=args.seed,
        subset_size=args.subset_size,
        ckpt_dir=args.ckpt_dir,
        overwrite_ckpt=args.overwrite_ckpt,
        cond_encoding=args.cond_encoding,
    )
    print(
        json.dumps(
            {
                "generation_path": report["generation_path"],
                "target_path": report["target_path"],
            },
            indent=2,
        )
    )
    return 0


def _build_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    from sc_save.eval.generation import DEFAULT_COND_COLS, DEFAULT_OUTPUT_DIR

    p = subparsers.add_parser(
        "evaluate",
        help="Evaluate generated conditional distributions against target AnnData.",
    )
    p.add_argument("--gt-path", type=str, required=True, help="Target AnnData .h5ad file.")
    p.add_argument("--gen-path", type=str, required=True, help="Generated AnnData .h5ad file.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--model-name", type=str, default="generated")
    p.add_argument("--cond-cols", nargs="+", default=list(DEFAULT_COND_COLS))
    p.add_argument("--sort-cols", nargs="+", default=None)
    p.add_argument("--pca-components", type=int, default=30)
    p.add_argument("--max-cells-per-condition", type=int, default=4000)
    p.add_argument("--hvg-top-genes", type=int, default=2000)
    p.add_argument("--top-gene-count", type=int, default=100)
    p.add_argument("--skip-expression-metrics", action="store_true")
    p.add_argument("--eval-workers", type=int, default=0)
    p.set_defaults(handler=_handle_evaluate)


def _handle_evaluate(args: argparse.Namespace) -> int:
    from sc_save.eval.generation import run_conditional_distribution_evaluation

    run_conditional_distribution_evaluation(
        gt_path=args.gt_path,
        gen_path=args.gen_path,
        output_dir=args.output_dir,
        cond_cols=args.cond_cols,
        sort_cols=args.sort_cols,
        device_id=args.device,
        pca_components=args.pca_components,
        max_cells_per_condition=args.max_cells_per_condition,
        hvg_top_genes=args.hvg_top_genes,
        top_gene_count=args.top_gene_count,
        model_name=args.model_name,
        include_expression_metrics=not args.skip_expression_metrics,
        eval_workers=args.eval_workers,
    )
    return 0


def _build_bec_parser(subparsers: argparse._SubParsersAction) -> None:
    from sc_save.workflows.bec import (
        DEFAULT_BEC_OUTPUT_DIR,
        SAVE_B_PERTURB_MODEL_CONFIG,
        SAVE_B_PERTURB_TRAIN_CONFIG,
    )

    p = subparsers.add_parser(
        "bec",
        help="Run SAVE batch effect correction (LDM_raw) and scIB evaluation.",
    )
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--task-name", type=str, default="bec_task")
    p.add_argument("--output-dir", type=str, default=DEFAULT_BEC_OUTPUT_DIR)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch-key", type=str, default="batch")
    p.add_argument("--cell-type-key", type=str, default="cell_type")
    p.add_argument("--epoch", type=int, default=SAVE_B_PERTURB_TRAIN_CONFIG["epoch"])
    p.add_argument("--batch-size", type=int, default=SAVE_B_PERTURB_TRAIN_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=SAVE_B_PERTURB_TRAIN_CONFIG["lr"])
    p.add_argument("--weight-decay", type=float, default=SAVE_B_PERTURB_TRAIN_CONFIG["weight_decay"])
    p.add_argument("--kl-scale", type=float, default=SAVE_B_PERTURB_MODEL_CONFIG["kl_scale"])
    p.add_argument(
        "--sample-mask-prob",
        type=float,
        default=SAVE_B_PERTURB_TRAIN_CONFIG["sample_mask_prob"],
    )
    p.add_argument(
        "--col-msk-threshold",
        type=float,
        default=SAVE_B_PERTURB_TRAIN_CONFIG["col_msk_threshold"],
    )
    p.add_argument(
        "--gene-block-size",
        type=int,
        default=SAVE_B_PERTURB_MODEL_CONFIG["gene_block_size"],
    )
    p.add_argument("--pca-components", type=int, default=30)
    p.add_argument("--transfer-batch-size", type=int, default=256)
    p.add_argument("--scib-workers", type=int, default=0)
    p.add_argument("--max-target-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=1202)
    p.add_argument("--overwrite-ckpt", action="store_true")
    p.add_argument("--skip-scib", action="store_true")
    p.add_argument("--cond-encoding", choices=["legacy", "offset"], default="legacy")
    p.set_defaults(handler=_handle_bec)


def _handle_bec(args: argparse.Namespace) -> int:
    from sc_save.workflows.bec import run_bec_integration

    result = run_bec_integration(
        data_path=args.data_path,
        task_name=args.task_name,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        device_id=args.device,
        batch_key=args.batch_key,
        cell_type_key=args.cell_type_key,
        epoch=args.epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        kl_scale=args.kl_scale,
        sample_mask_prob=args.sample_mask_prob,
        col_msk_threshold=args.col_msk_threshold,
        gene_block_size=args.gene_block_size,
        pca_components=args.pca_components,
        transfer_batch_size=args.transfer_batch_size,
        scib_workers=args.scib_workers,
        max_target_batches=args.max_target_batches,
        seed=args.seed,
        overwrite_ckpt=args.overwrite_ckpt,
        run_scib=not args.skip_scib,
        cond_encoding=args.cond_encoding,
    )
    print(f"Done. Report written to {result['report_path']}")
    summary = result["report"].get("summary", {})
    for metric in ("avg_bio", "avg_batch", "scib_score", "PCR_batch"):
        if metric in summary and summary[metric]["mean"] is not None:
            print(f"  {metric}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")
    return 0


def _build_perturb_parser(subparsers: argparse._SubParsersAction) -> None:
    from sc_save.workflows.perturbation import (
        DEFAULT_CONDITION_KEY,
        DEFAULT_CONTROL_CONDITION,
        DEFAULT_LABEL_KEY,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TARGET_CONDITION,
    )

    p = subparsers.add_parser(
        "perturb",
        help="Run single cell-type or leave-one-out perturbation prediction.",
    )
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--cell-type", type=str, default=None, help="Single held-out cell type.")
    p.add_argument("--cell-types", type=str, nargs="*", default=None)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=1202)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--label-key", type=str, default=DEFAULT_LABEL_KEY)
    p.add_argument("--condition-key", type=str, default=DEFAULT_CONDITION_KEY)
    p.add_argument("--control-condition", type=str, default=DEFAULT_CONTROL_CONDITION)
    p.add_argument("--target-condition", type=str, default=DEFAULT_TARGET_CONDITION)
    p.add_argument("--train-with-ctrl", action="store_true", default=False)
    p.add_argument("--no-data-scale", dest="data_scale", action="store_false")
    p.set_defaults(data_scale=True)
    p.add_argument("--transfer-batch-size", type=int, default=128)
    p.add_argument("--max-input-cells", type=int, default=0)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--sample-mask-prob", type=float, default=0.0)
    p.add_argument("--col-msk-threshold", type=float, default=0.6)
    p.add_argument("--n-pca-components", type=int, default=30)
    p.add_argument("--n-wd-mmd-repeats", type=int, default=10)
    p.add_argument("--workers", type=str, default="1")
    p.add_argument("--cond-encoding", choices=["legacy", "offset"], default="legacy")
    p.set_defaults(handler=_handle_perturb)


def _handle_perturb(args: argparse.Namespace) -> int:
    from sc_save.workflows.perturbation import (
        PERTURB_TRAIN_CONFIG,
        run_leave_one_out,
        run_perturbation,
    )

    train_config = None
    if args.epoch is not None:
        train_config = dict(PERTURB_TRAIN_CONFIG)
        train_config["epoch"] = args.epoch

    if args.cell_type is not None:
        run_perturbation(
            data_path=args.data_path,
            cell_type=args.cell_type,
            output_dir=args.output_dir,
            device_id=args.device,
            seed=args.seed,
            ckpt_dir=args.ckpt_dir,
            label_key=args.label_key,
            condition_key=args.condition_key,
            control_condition=args.control_condition,
            target_condition=args.target_condition,
            train_with_ctrl=args.train_with_ctrl,
            data_scale=args.data_scale,
            transfer_batch_size=args.transfer_batch_size,
            max_input_cells=args.max_input_cells,
            train_config=train_config,
            sample_mask_prob=args.sample_mask_prob,
            col_msk_threshold=args.col_msk_threshold,
            n_pca_components=args.n_pca_components,
            n_wd_mmd_repeats=args.n_wd_mmd_repeats,
            cond_encoding=args.cond_encoding,
        )
        return 0

    workers: int | str = (
        args.workers if args.workers.strip().lower() == "auto" else int(args.workers)
    )
    run_leave_one_out(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device_id=args.device,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
        label_key=args.label_key,
        condition_key=args.condition_key,
        control_condition=args.control_condition,
        target_condition=args.target_condition,
        train_with_ctrl=args.train_with_ctrl,
        data_scale=args.data_scale,
        transfer_batch_size=args.transfer_batch_size,
        max_input_cells=args.max_input_cells,
        cell_types=args.cell_types,
        train_config=train_config,
        sample_mask_prob=args.sample_mask_prob,
        col_msk_threshold=args.col_msk_threshold,
        n_pca_components=args.n_pca_components,
        n_wd_mmd_repeats=args.n_wd_mmd_repeats,
        workers=workers,
        cond_encoding=args.cond_encoding,
    )
    return 0


def _build_reproduce_parser(subparsers: argparse._SubParsersAction) -> None:
    from sc_save.workflows.reproduce import add_reproduce_arguments

    p = subparsers.add_parser(
        "reproduce",
        help="Reproduce Tables 1-5 of the SAVE ICLR 2026 paper.",
    )
    add_reproduce_arguments(p)
    p.set_defaults(handler=_handle_reproduce)


def _handle_reproduce(args: argparse.Namespace) -> int:
    from sc_save.workflows.reproduce import run_reproduction

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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sc-save",
        description="SAVE: Single-Cell Variational Flow Matching CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_prepare_datasets_parser(subparsers)
    _build_blocks_parser(subparsers)
    _build_train_parser(subparsers)
    _build_generate_parser(subparsers)
    _build_evaluate_parser(subparsers)
    _build_bec_parser(subparsers)
    _build_perturb_parser(subparsers)
    _build_reproduce_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
