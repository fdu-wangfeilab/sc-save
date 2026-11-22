# Batch Effect Correction (BEC) Example

Runs SAVE-B-perturb (`gene_block_size=400`, `sample_mask_prob=0.0`, `col_msk_threshold=0.6`) across every target batch via `LDM_raw` conditional batch transfer, computes 30-PC PCA embeddings, and evaluates scIB metrics (`avg_bio`, `avg_batch`, `scib_score`, `PCR_batch`) without R/rpy2 dependencies.

## Install BEC extra

```bash
pip install -e ".[bec]"
```

## Run

```bash
python examples/bec_integration/run_bec.py \
  --data-path /path/to/lung_atlas_processed_GBS400.h5ad \
  --task-name lung_atlas \
  --output-dir ./outputs/bec_integration/lung_atlas \
  --device 0
```

Outputs under `--output-dir`:
- `checkpoints/<task_name>.pt`: SAVE model checkpoint (resumed automatically if present).
- `embeddings/transfer_to_<batch>.npy`: 30-PC PCA coordinates for every cell aligned to original `adata.obs_names`.
- `scib_eval.csv`: Per-target-batch scIB scores plus `mean` and `std` columns.
- `report.json`: Aggregated metrics matching Table 4 of the ICLR 2026 paper.
