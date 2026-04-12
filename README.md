# sc-save-public

Public-facing repository for gradually open-sourcing selected `SAVE` examples and reusable utilities.

This repository is intentionally narrower than the private research repository. It only contains code that is safe to publish and stable enough to maintain as public API or runnable examples.

## Scope

Currently migrated:

- `examples/gene_block_construction`
- `examples/conditional_distribution_evaluation`
- `examples/conditional_generation`

Planned next:

- `examples/scib_evaluation`

Modules that still depend on private research code or unpublished internals will be migrated later after dependency cleanup.

## Repository Layout

```text
src/sc_save_public/   Reusable public Python package code
examples/             Runnable public examples
tests/                Public test coverage
```

## Status

This repository is being prepared for staged release. The initial scaffold is in place, and public modules will be migrated incrementally from the private development repository.

## UV Environment

The public package is now aligned to the working `vqvae` runtime that is currently used for this repository:

- Python `3.10.16`
- `numpy 1.26.4`
- `pandas 2.2.3`
- `scanpy 1.11.0`
- `torch 2.5.1`
- generation extras: `einops 0.8.0`, `flow-matching 1.0.10`

From the repository root, create the `uv` environment and install the core package plus test tools:

```bash
uv python install 3.10.16
uv sync
```

If you also want the public conditional-generation stack, install the generation extra:

```bash
uv sync --extra generation
```

Run tests with:

```bash
uv run pytest
```

Run a generation import smoke check with:

```bash
uv run --extra generation python -c "from sc_save_public.conditional_generation import DEFAULT_COND_COLS; print(DEFAULT_COND_COLS)"
```

`flash-attn` is still optional and is not part of the default `uv` environment because the public example path keeps `is_flash_attn=False`.
The public generation path no longer depends on `timm`/`torchvision`, which keeps `uv sync --extra generation` smaller and avoids an otherwise unnecessary heavyweight transitive download.

## Legacy Pip Flow

If you specifically want a plain `pip` editable install instead of `uv`, the equivalent commands are:

```bash
python -m pip install -e .
python -m pip install -e .[generation]
pytest
```
