# Conditional Distribution Evaluation

This example evaluates whether a generated AnnData object preserves the target condition distribution and matches the target expression distribution within each condition group.

It keeps the public path:

1. Load one target `.h5ad` and one generated `.h5ad`.
2. Align genes between the two objects.
3. Count cells for every condition combination in `cond_cols`.
4. Keep only condition combinations whose target and generated counts exactly match.
5. Fit one shared PCA space on the concatenated target and generated cells.
6. Compute per-condition Wasserstein and MMD metrics in PCA space.
7. Optionally compute secondary expression-mean metrics on HVGs.
8. Save the alignment table, per-condition metric table, and a summary report.

## Files

- `run_example.py`: CLI entry point for evaluating one target/generated pair.
- `sc_save_public.conditional_distribution_evaluation`: reusable implementation in the public package.

## Install

From the repository root:

```bash
python -m pip install -e .
```

## Run

```bash
python examples/conditional_distribution_evaluation/run_example.py \
  --gt-path /path/to/target.h5ad \
  --gen-path /path/to/generated.h5ad \
  --cond-cols batch cell_type \
  --output-dir ./outputs/conditional_distribution_evaluation
```

If you only want the main Wasserstein and MMD metrics, skip the expression summary:

```bash
python examples/conditional_distribution_evaluation/run_example.py \
  --gt-path /path/to/target.h5ad \
  --gen-path /path/to/generated.h5ad \
  --cond-cols batch cell_type \
  --skip-expression-metrics
```

## Outputs

- `distribution_alignment.csv`: per-condition target/generated counts and exact-match flags.
- `per_condition_metrics.csv`: per-condition WD/MMD metrics, plus optional expression metrics.
- `report.json`: evaluation configuration, distribution-match summary, and metric means/stds.
