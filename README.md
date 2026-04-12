# sc-save-public

Public-facing repository for gradually open-sourcing selected `SAVE` examples and reusable utilities.

The current public examples cover:

- gene block construction
- conditional generation
- conditional distribution evaluation

## Repository Layout

```text
src/sc_save_public/   Reusable package code
examples/             Runnable example entrypoints
tests/                Public test coverage
```

## Environment Installation

This repository is pinned to the validated `uv` environment below:

- Python `3.10.16`
- `numpy 1.26.4`
- `pandas 2.2.3`
- `scanpy 1.11.0`
- `torch 2.5.1`
- generation dependencies: `einops 0.8.0`, `flow-matching 1.0.10`

From the repository root:

```bash
uv python install 3.10.16
uv sync --extra generation
```

That command creates `.venv`, installs the package in editable mode, and includes the conditional-generation stack.

Optional verification:

```bash
uv run pytest
uv run --extra generation python -c "from sc_save_public.conditional_generation import DEFAULT_COND_COLS; print(DEFAULT_COND_COLS)"
```

Notes:

- `flash-attn` is optional and is not required for the public examples. The released configs keep `is_flash_attn=False`.
- The public generation path does not require `timm` or `torchvision`.

## Input Data Requirements

For conditional generation, the input `.h5ad` should satisfy:

- gene dimension already matches the model-ready representation you want to train on
- `adata.obs["batch"]` exists
- `adata.obs["cell_type"]` exists

If your dataset already contains padded genes such as `pad_0`, the generation pipeline will detect them and trim them back out in the saved outputs.

For evaluation, the target and generated `.h5ad` files should represent the same gene space. If the gene names contain the same set in different orders, the evaluation script will reorder the generated object automatically.

## Run Conditional Generation

### Smoke Run

Use this for a fast end-to-end check on a small subset:

```bash
uv run --extra generation python examples/conditional_generation/run_hca_smoke.py \
  --data-path /path/to/input.h5ad \
  --output-dir ./outputs/conditional_generation_smoke \
  --subset-size 512 \
  --device 0 \
  --seed 1202 \
  --vae-epoch 1 \
  --fm-epoch 1 \
  --batch-size 64 \
  --generate-count 8 \
  --guidance-weights 2.0 \
  --sample-steps 5 \
  --num-heads 4
```

Main outputs:

- `generated_smoke.h5ad`
- `target_smoke.h5ad`
- `gen_z_fm.npy`
- `target_z_vae.npy`
- `report.json`

### Formal Run

Use this when you want the public formal configuration and checkpoint reuse:

```bash
uv run --extra generation python examples/conditional_generation/run_hca_formal.py \
  --data-path /path/to/input.h5ad \
  --output-dir ./outputs/conditional_generation_formal \
  --device 0 \
  --num-heads 4
```

Main outputs:

- `generation.h5ad`
- `target.h5ad`
- `gen_z_fm.npy`
- `target_z_vae.npy`
- `report.json`
- `ckpt/`

## Run Conditional Distribution Evaluation

### Evaluate Smoke Outputs

```bash
uv run python examples/conditional_distribution_evaluation/run_example.py \
  --gt-path ./outputs/conditional_generation_smoke/target_smoke.h5ad \
  --gen-path ./outputs/conditional_generation_smoke/generated_smoke.h5ad \
  --cond-cols batch cell_type \
  --output-dir ./outputs/conditional_generation_smoke_eval \
  --device 0
```

### Evaluate Formal Outputs

```bash
uv run python examples/conditional_distribution_evaluation/run_example.py \
  --gt-path ./outputs/conditional_generation_formal/target.h5ad \
  --gen-path ./outputs/conditional_generation_formal/generation.h5ad \
  --cond-cols batch cell_type \
  --output-dir ./outputs/conditional_generation_formal_eval \
  --device 0
```

If you only want the main Wasserstein and MMD metrics, skip expression summary metrics:

```bash
uv run python examples/conditional_distribution_evaluation/run_example.py \
  --gt-path /path/to/target.h5ad \
  --gen-path /path/to/generated.h5ad \
  --cond-cols batch cell_type \
  --output-dir ./outputs/conditional_distribution_evaluation \
  --skip-expression-metrics
```

Evaluation outputs:

- `distribution_alignment.csv`
- `per_condition_metrics.csv`
- `report.json`

## Citation

If you use this repository or the SAVE method in your work, please cite:

```bibtex
@inproceedings{
li2026save,
title={{SAVE}: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention},
author={Jiahao Li and Jiayi Dong and Peng Ye and Xiaochi Zhou and Haohai Lu and Fei Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=l7QEoK4uDP}
}
```
