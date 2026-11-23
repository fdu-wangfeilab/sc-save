# Changelog

All notable changes to `sc-save` are documented in this file.

## [0.2.0] - 2026-11-23

> **Validation Status**: CPU unit, package API, and numerical-parity tests pass locally (`57` pytest tests). Full ICLR 2026 Tables 1–5 Colab GPU benchmarks for `0.2.0` have **not yet been executed**; verified ICLR 2026 v1 Colab G4 results are preserved in [`results/iclr2026_colab_repro.json`](results/iclr2026_colab_repro.json), and v2 ships `sc-save reproduce` alongside the `colab_validate_v2.py` validation driver ready for Colab GPU validation.

### Added
- **Unified `sc_save` Python package (`src/sc_save/`)**:
  - High-level `SAVE` estimator API (`fit`, `fit_vae`, `fit_fm`, `encode`, `decode`, `reconstruct`, `generate`, `generate_like`, `generate_with_attention`, `transfer`, `save`, `load`).
  - Typed configuration dataclasses (`SAVEConfig`, `VAEConfig`, `FMConfig`, `TrainConfig`) with presets (`SAVEConfig.save_b`, `SAVEConfig.save_b_perturb`, `SAVEConfig.smoke`).
  - Preprocessing and Gene Block utilities (`ConditionVocabulary` with `"legacy"` and `"offset"` modes, `MaxAbsExpressionScaler`, `preprocess_adata`, `build_gene_blocks`, `GeneBlockResult`, `load_gene_embeddings`).
  - Cleanly decomposed dataset pipeline (`sc_save.data.datasets.{registry,download,prepare}`), evaluation suite (`sc_save.eval.{metrics,generation,perturbation,bec}`), and workflow runners (`sc_save.workflows.{gene_blocks,generation,bec,perturbation,reproduce}`).
- **Unified `sc-save` CLI (`src/sc_save/cli.py`)** with subcommands: `prepare-datasets`, `build-blocks`, `train`, `generate`, `evaluate`, `bec`, `perturb`, `reproduce`, and `--version`.
- **Pure-PyTorch Flow Matching probability paths and solvers**: `CondOTPath`, `VPPath`, `VEPath`, and fixed-step solvers (`euler`, `midpoint`, `rk4`, plus SDE `euler`) with fused `[2*B]` Classifier-Free Guidance sampling—removing mandatory `flow-matching` and `torchdiffeq` runtime dependencies (moved to `[solvers]` extra).
- **Quickstart Tutorial Notebook** (`notebooks/SAVE_Quickstart_Tutorial.ipynb`) and GitHub Actions CI/CD workflows (`.github/workflows/ci.yml`, `.github/workflows/publish.yml`).

### Changed & Improved
- Renamed PyPI package distribution from `sc-save-public` to `sc-save` while preserving `src/sc_save_public/` as full backward-compatible shims and keeping all `examples/` CLI flags and output schemas identical.
- Eliminated per-step GPU-to-CPU synchronization checks (`isnan`/`isinf` and `Normal` validation syncs) in VAE training, FM training, and FM sampling.
- Replaced per-epoch `DataLoader` worker respawns with adaptive GPU residency (`min(48 GB, 0.50 * free_vram)` FP32/FP16/pinned-CPU tiers) and precomputed frozen VAE posteriors for FM training.
- Replaced per-condition loops in `run_formal_generation` and BEC integration with single-pass batched `SAVE.generate_like(...)` and `SAVE.transfer(..., input_cond="obs")`.
- Replaced $O(G \cdot N)$ condition masking in distribution evaluation with `groupby` index lookups and shared `torch.cdist` between 1-Wasserstein and 2-Wasserstein metrics without per-condition `gc.collect()` / `empty_cache()` stalls.
- Added legacy v1 `.pt` checkpoint key remapping (`tokenzier` / `pos_embding` <-> `tokenizer` / `pos_embedding`).

## [0.1.0] - 2026-04-20

### Added
- Initial public release (`sc-save-public`) containing Gene Block Construction, Conditional Generation (VAE + OT Flow Matching), Conditional Distribution Evaluation, Batch Effect Correction (`scIB`), Leave-One-Out Perturbation Prediction, dataset preparation, and unified ICLR 2026 Tables 1–5 reproduction scripts and Colab notebook.
