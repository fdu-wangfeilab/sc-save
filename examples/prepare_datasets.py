"""CLI entrypoint for downloading and preprocessing all SAVE ICLR 2026 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sc_save_public.datasets import (
    DEFAULT_OT_NUM_ITER_MAX,
    GENEPT_ZENODO_URL,
    ensure_genept_embeddings,
    get_output_paths,
    list_datasets,
    prepare_datasets,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for dataset preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Download, standardize, normalize, and OT-partition SAVE ICLR 2026 "
            "benchmark datasets (Tables 1-5)."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Dataset names, aliases, table groups (table1..table5), 'genept', or 'all'. "
            "Supports space- or comma-separated tokens (default: all)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory where raw/, genept/, processed/, and manifest.json are stored.",
    )
    parser.add_argument(
        "--gene-embedding-path",
        type=str,
        default=None,
        help="Optional path to an existing GenePT_gene_protein_embedding_model_3_text.pickle.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip all network downloads and read raw/GenePT inputs from --data-root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading raw inputs and recomputing processed .h5ad files.",
    )
    parser.add_argument(
        "--subset-cells",
        type=int,
        default=None,
        help="Optional cell cap per dataset for fast smoke tests.",
    )
    parser.add_argument(
        "--ot-method",
        choices=["emd", "sinkhorn"],
        default="emd",
        help="Optimal Transport assignment solver for balanced k-means (default: emd).",
    )
    parser.add_argument(
        "--ot-reg",
        type=float,
        default=0.1,
        help="Sinkhorn entropy regularization when --ot-method=sinkhorn (default: 0.1).",
    )
    parser.add_argument(
        "--ot-max-iter",
        type=int,
        default=20,
        help="Maximum OT balanced k-means iterations (default: 20).",
    )
    parser.add_argument(
        "--ot-num-iter-max",
        type=int,
        default=DEFAULT_OT_NUM_ITER_MAX,
        help=(
            "Maximum simplex iterations for POT ot.emd "
            f"(default: {DEFAULT_OT_NUM_ITER_MAX})."
        ),
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=None,
        help="Optional override for sc.pp.filter_cells(min_genes=...) in QC tasks.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=None,
        help="Optional override for sc.pp.filter_genes(min_cells=...) in QC tasks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for OT initialization and optional subsampling (default: 42).",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print all registered datasets, verified URLs, and expected output paths.",
    )
    return parser.parse_args(argv)


def print_dataset_catalog(data_root: Path) -> None:
    """Print registered datasets and output conventions in a readable summary."""
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


def main(argv: Sequence[str] | None = None) -> int:
    """Execute dataset preparation CLI."""
    args = parse_args(argv)
    data_root = Path(args.data_root)

    if args.list_datasets:
        print_dataset_catalog(data_root)
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
    )

    summary = {
        name: rec.to_dict(data_root=data_root) for name, rec in results.items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
