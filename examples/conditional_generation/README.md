# Conditional Generation

This example exposes the public `SAVE` + Flow Matching (`FM`) conditional-generation pipelines.

It keeps the main path:

1. Load input `.h5ad` dataset(s) (with optional `--test-data-path` split alignment).
2. Scale expression values with `MaxAbsScaler`.
3. Train or resume a `SAVE` VAE with condition masking.
4. Train or resume a latent Flow Matching model (`MLPTimeStep` DiT backbone) on `--cond-cols`.
5. Sample latent codes with classifier-free guidance and decode back to gene expression, trimming padding (`pad_0...`) and inverse-scaling.

## Files

- `run_formal.py`: generalized formal CLI supporting arbitrary `--cond-cols`, `--gene-block-size`, `--test-data-path`, `--guidance-weights`, `--sample-steps`, and VAE+FM checkpoint resume (Tables 1–3).
- `run_hca_formal.py`: backwards-compatible HCA dual-condition formal entry point.
- `run_hca_smoke.py`: small smoke entry point for quick local checks.
- `sc_save_public.conditional_generation`: reusable library module.

## Install

```bash
python -m pip install -e ".[generation]"
```

## Run Generalized Formal Generation

Single-condition (Table 1, `gene_block_size=3200`):

```bash
python examples/conditional_generation/run_formal.py \
  --data-path /path/to/pbmc3k_train_GBS3200.h5ad \
  --test-data-path /path/to/pbmc3k_test_GBS3200.h5ad \
  --task-name pbmc3k \
  --cond-cols cell_type \
  --gene-block-size 3200 \
  --output-dir ./outputs/conditional_generation/pbmc3k \
  --device 0
```

Multi-condition Lung Cancer (Table 3, `gene_block_size=400`, 5 conditions, unseen conditions 13 & 24):

```bash
python examples/conditional_generation/run_formal.py \
  --data-path /path/to/cancer_lung_train_GBS400.h5ad \
  --test-data-path /path/to/cancer_lung_test_GBS400.h5ad \
  --eval-split all \
  --task-name cancer_lung \
  --cond-cols assay development_stage disease uicc_stage cell_type \
  --gene-block-size 400 \
  --sample-mask-prob 0.0 \
  --output-dir ./outputs/conditional_generation/cancer_lung \
  --device 0
```

## Outputs

- `ckpt/`: saved VAE and FM checkpoints (`<task_name>_VAE_*.pt`, `<task_name>_fm_*.pt`).
- `generation.h5ad`: generated cells after inverse scaling and pad-gene trimming.
- `target.h5ad`: aligned target cells used for evaluation.
- `gen_z_fm.npy` / `target_z_vae.npy`: sampled FM and encoded VAE latents.
- `report.json`: run configuration, timing, checkpoint status, and output paths.
