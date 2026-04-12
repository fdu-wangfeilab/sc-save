# Gene Block Construction

This example exposes the public `Gene Block Construction` preprocessing stage used before downstream `SAVE` training.

It implements the stable path:

1. Load an input `.h5ad` dataset.
2. Optionally replace `adata.var_names` with a column such as `feature_name`.
3. Keep genes covered by a precomputed embedding table.
4. Reduce the covered subset to a size divisible by `gene_block_size` with HVG selection.
5. Run balanced Optimal Transport k-means on the gene embeddings.
6. Reorder the selected genes by block id, append the remaining genes, and zero-pad to a full block.
7. Save the processed dataset, per-gene metadata, and a run report.

## Files

- `run_pipeline.py`: CLI entry point for one local run.
- `sc_save_public.gene_block_construction`: reusable implementation in the public package.

## Requirements

This example does not ship data or gene embeddings.

You need:

- an input `.h5ad` dataset
- a pickle file containing a mapping `gene_name -> embedding vector`

The embedding file can come from your own preprocessing pipeline or another distribution that you are allowed to use. Do not assume the private-repo paths from the research environment exist here.

## Install

From the repository root:

```bash
python -m pip install -e .
```

## Run

Generic example:

```bash
python examples/gene_block_construction/run_pipeline.py \
  --data-path /path/to/input.h5ad \
  --gene-embedding-path /path/to/gene_embeddings.pkl \
  --gene-block-size 400 \
  --output-dir ./outputs/gene_block_construction
```

If the dataset stores canonical gene symbols in `adata.var["feature_name"]`, add:

```bash
python examples/gene_block_construction/run_pipeline.py \
  --data-path /path/to/input.h5ad \
  --gene-embedding-path /path/to/gene_embeddings.pkl \
  --gene-name-col feature_name
```

Optional smoke-sized run:

```bash
python examples/gene_block_construction/run_pipeline.py \
  --data-path /path/to/input.h5ad \
  --gene-embedding-path /path/to/gene_embeddings.pkl \
  --subset-cells 512 \
  --gene-block-size 400 \
  --output-dir ./outputs/gene_block_smoke
```

## Outputs

- `processed.h5ad`: reordered and padded dataset ready for downstream training.
- `var_metadata.csv`: gene-level metadata including block ids and pad markers.
- `report.json`: shapes, HVG statistics, OT settings, and runtime.
