"""Generalized CLI entry point for SAVE formal conditional generation (Tables 1-3)."""

from __future__ import annotations

import argparse
import json

from sc_save_public.conditional_generation import (
    DEFAULT_COND_COLS,
    DEFAULT_FORMAL_OUTPUT_DIR,
    run_formal_conditional_generation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generalized SAVE VAE + FM formal conditional generation."
    )
    parser.add_argument("--data-path", type=str, required=True, help="Training .h5ad file.")
    parser.add_argument(
        "--test-data-path",
        type=str,
        default=None,
        help="Optional held-out/test .h5ad file (Table 1 test split or Table 3 Lung Cancer unseen conditions).",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="auto",
        choices=["auto", "train", "test", "all"],
        help="Which split(s) to generate when --test-data-path is given.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_FORMAL_OUTPUT_DIR)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--task-name", type=str, default="conditional_generation")
    parser.add_argument(
        "--cond-cols",
        nargs="+",
        default=DEFAULT_COND_COLS,
        help="Condition column names in adata.obs (e.g. --cond-cols cell_type).",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gene-block-size", type=int, default=3200)
    parser.add_argument("--latent-block-size", type=int, default=None)
    parser.add_argument("--vae-epoch", type=int, default=200)
    parser.add_argument("--fm-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2.5e-5)
    parser.add_argument("--kl-scale", type=float, default=0.5)
    parser.add_argument("--expand-dim", type=int, default=512)
    parser.add_argument("--enc-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-fm-blocks", type=int, default=5)
    parser.add_argument("--guidance-weights", type=float, default=2.0)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument(
        "--sample-mask-prob",
        type=float,
        default=None,
        help="Defaults to 0.0 for cancer_lung and 0.4 otherwise.",
    )
    parser.add_argument("--vae-col-msk-threshold", type=float, default=1.0)
    parser.add_argument("--fm-col-msk-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--overwrite-ckpt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_formal_conditional_generation(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device_id=args.device,
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
    )
    print(json.dumps({"generation_path": report["generation_path"], "target_path": report["target_path"]}, indent=2))


if __name__ == "__main__":
    main()
