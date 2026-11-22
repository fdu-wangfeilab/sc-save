"""CLI entry point for running SAVE batch effect correction (LDM_raw) + scIB evaluation."""

from __future__ import annotations

import argparse

from sc_save_public.bec_integration import (
    DEFAULT_BEC_OUTPUT_DIR,
    SAVE_B_PERTURB_MODEL_CONFIG,
    SAVE_B_PERTURB_TRAIN_CONFIG,
    run_bec_integration,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAVE-B-perturb batch effect correction (LDM_raw) and scIB evaluation."
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to processed GBS400 .h5ad file.")
    parser.add_argument("--task-name", type=str, default="bec_task", help="Task identifier and checkpoint stem.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_BEC_OUTPUT_DIR)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-key", type=str, default="batch")
    parser.add_argument("--cell-type-key", type=str, default="cell_type")
    parser.add_argument("--epoch", type=int, default=SAVE_B_PERTURB_TRAIN_CONFIG["epoch"])
    parser.add_argument("--batch-size", type=int, default=SAVE_B_PERTURB_TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=SAVE_B_PERTURB_TRAIN_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=SAVE_B_PERTURB_TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--kl-scale", type=float, default=SAVE_B_PERTURB_MODEL_CONFIG["kl_scale"])
    parser.add_argument("--sample-mask-prob", type=float, default=SAVE_B_PERTURB_TRAIN_CONFIG["sample_mask_prob"])
    parser.add_argument("--col-msk-threshold", type=float, default=SAVE_B_PERTURB_TRAIN_CONFIG["col_msk_threshold"])
    parser.add_argument("--gene-block-size", type=int, default=SAVE_B_PERTURB_MODEL_CONFIG["gene_block_size"])
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--transfer-batch-size", type=int, default=256)
    parser.add_argument("--scib-workers", type=int, default=0, help="0 auto-selects up to 8 CPU workers.")
    parser.add_argument("--max-target-batches", type=int, default=0, help="0 evaluates all batches.")
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--overwrite-ckpt", action="store_true")
    parser.add_argument("--skip-scib", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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
    )
    print(f"Done. Report written to {result['report_path']}")
    summary = result["report"].get("summary", {})
    for metric in ("avg_bio", "avg_batch", "scib_score", "PCR_batch"):
        if metric in summary and summary[metric]["mean"] is not None:
            print(f"  {metric}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")


if __name__ == "__main__":
    main()
