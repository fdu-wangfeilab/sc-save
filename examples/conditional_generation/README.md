# Conditional Generation

This example exposes the current public `SAVE` conditional-generation path on top of the migrated `SAVE/FM` model stack.

It keeps the main path:

1. Load an input `.h5ad` dataset.
2. Scale expression values with `MaxAbsScaler`.
3. Train a `SAVE` VAE with masked conditions.
4. Train a latent Flow Matching model conditioned on `batch` and `cell_type`.
5. Sample latent codes with FM and decode them back to gene expression.

## Files

- `run_hca_smoke.py`: small smoke entry point for one local dataset.
- `run_hca_formal.py`: formal entry point that can reuse checkpoints across runs.
- `sc_save_public.conditional_generation`: reusable implementation in the public package.

## Install

From the repository root:

```bash
python -m pip install -e .[generation]
```

`flash-attn` is optional. The default public settings keep `is_flash_attn=False`, so the smoke path can run without it.

## Run The Smoke Example

```bash
python examples/conditional_generation/run_hca_smoke.py \
  --data-path /path/to/input.h5ad \
  --device 0 \
  --subset-size 512 \
  --vae-epoch 1 \
  --fm-epoch 1 \
  --batch-size 64 \
  --generate-count 8 \
  --sample-steps 5
```

## Run The Formal Example

```bash
python examples/conditional_generation/run_hca_formal.py \
  --data-path /path/to/input.h5ad \
  --device 0 \
  --output-dir ./outputs/conditional_generation_formal
```

## Outputs

- `generated_smoke.h5ad` / `generation.h5ad`: generated cells after inverse scaling and pad-gene trimming.
- `target_smoke.h5ad` / `target.h5ad`: real target cells used for comparison.
- `gen_z_fm.npy`: FM latent samples.
- `target_z_vae.npy`: VAE target latents.
- `report.json`: run configuration, timing, condition choices, and output shapes.
