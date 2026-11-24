import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sc_save.eval.generation import (
    DEFAULT_COND_COLS,
    DEFAULT_OUTPUT_DIR,
    run_conditional_distribution_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated conditional distributions against target AnnData.",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        required=True,
        help="Path to the target AnnData .h5ad file.",
    )
    parser.add_argument(
        "--gen-path",
        type=str,
        required=True,
        help="Path to the generated AnnData .h5ad file.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model-name", type=str, default="generated")
    parser.add_argument("--cond-cols", nargs="+", default=DEFAULT_COND_COLS)
    parser.add_argument("--sort-cols", nargs="+", default=None)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--max-cells-per-condition", type=int, default=4000)
    parser.add_argument("--hvg-top-genes", type=int, default=2000)
    parser.add_argument("--top-gene-count", type=int, default=100)
    parser.add_argument(
        "--skip-expression-metrics",
        action="store_true",
        help="Skip PCC/MSE/R2/top-gene-overlap summary metrics.",
    )
    parser.add_argument("--eval-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
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


if __name__ == "__main__":
    main()
