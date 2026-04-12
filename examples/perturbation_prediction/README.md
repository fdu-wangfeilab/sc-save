# Perturbation Prediction

Hold out one cell type at a time, train (or load) a SAVE model on the remaining cells, predict the stimulated expression from control cells, and evaluate against ground truth.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_single.py` | Predict + evaluate for **one** cell type |
| `run_leave_one_out.py` | Loop over all (or selected) cell types and produce an aggregated summary |

## Input Data

An `.h5ad` file with:

- `adata.obs["cell_type"]` — cell-type labels (configurable via `--label-key`)
- `adata.obs["condition"]` — condition labels with at least a control and a target value (configurable via `--condition-key`, `--control-condition`, `--target-condition`)

## Quick Start

Single cell type:

```bash
uv run python examples/perturbation_prediction/run_single.py \
  --data-path /path/to/data.h5ad \
  --cell-type CD8T \
  --output-dir ./outputs/perturbation_CD8T
```

Leave-one-out over all cell types:

```bash
uv run python examples/perturbation_prediction/run_leave_one_out.py \
  --data-path /path/to/data.h5ad \
  --output-dir ./outputs/perturbation_leave_one_out
```

Run only selected cell types:

```bash
uv run python examples/perturbation_prediction/run_leave_one_out.py \
  --data-path /path/to/data.h5ad \
  --cell-types CD8T NK B
```

## Outputs

Per cell type:

- `train.h5ad` / `test.h5ad` — unscaled splits
- `SAVE_pred.h5ad` — predicted stimulated cells
- `SAVE_pred_attn_weights.pkl` — attention weights
- `train_scaler.pkl` — fitted MaxAbsScaler
- `report.json` — parameters, timing, and evaluation metrics

Leave-one-out adds:

- `summary.json` — per-cell-type and average R2, MSE, PCC

## Evaluation Metrics

Predictions are inverse-transformed from scaled space to original expression values using the training scaler, then compared against raw ground-truth stimulated cells:

- **R2** — coefficient of determination (sklearn `r2_score`)
- **MSE** — mean squared error of per-gene means (sklearn `mean_squared_error`)
- **PCC** — Pearson correlation of per-gene means (scipy `pearsonr`)
