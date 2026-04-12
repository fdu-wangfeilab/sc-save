# sc-save-public

Public-facing repository for gradually open-sourcing selected `SAVE` examples and reusable utilities.

This repository is intentionally narrower than the private research repository. It only contains code that is safe to publish and stable enough to maintain as public API or runnable examples.

## Scope

Currently migrated:

- `examples/gene_block_construction`

Planned next:

- `examples/scib_evaluation`
- `examples/conditional_distribution_evaluation`

Modules that still depend on private research code or unpublished internals will be migrated later after dependency cleanup.

## Repository Layout

```text
src/sc_save_public/   Reusable public Python package code
examples/             Runnable public examples
tests/                Public test coverage
```

## Status

This repository is being prepared for staged release. The initial scaffold is in place, and public modules will be migrated incrementally from the private development repository.

## Development

Create an isolated environment, install the package in editable mode, and run tests:

```bash
python -m pip install -e .
pytest
```
