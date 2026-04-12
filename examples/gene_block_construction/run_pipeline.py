import argparse

from sc_save_public.gene_block_construction import (
    DEFAULT_OUTPUT_DIR,
    run_gene_block_construction,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Construct balanced semantic gene blocks from precomputed embeddings.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Input AnnData .h5ad file.",
    )
    parser.add_argument(
        "--gene-embedding-path",
        type=str,
        required=True,
        help="Pickle file mapping gene names to embedding vectors.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gene-block-size", type=int, default=3200)
    parser.add_argument("--gene-name-col", type=str, default=None)
    parser.add_argument("--subset-cells", type=int, default=0)
    parser.add_argument("--ot-method", type=str, default="emd")
    parser.add_argument("--ot-reg", type=float, default=0.1)
    parser.add_argument("--ot-max-iter", type=int, default=20)
    parser.add_argument("--ot-random-state", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
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
        ot_random_state=args.ot_random_state,
    )
