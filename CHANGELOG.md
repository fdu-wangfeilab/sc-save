# Changelog

All notable changes to `sc-save` are documented in this file.

## [0.2.0] - 2026-11-24

> **Validation Status**: All **64** CPU unit/API/parity tests pass locally (`pytest -v`), and full ICLR 2026 Tables 1–5 Colab GPU validation (**all 10/10 main benchmarks + `heart` `offset` A/B**) passed with **0 regression gate failures** (`all_regression_gates_passed: true`) in [`results/iclr2026_v2_colab_validation.json`](results/iclr2026_v2_colab_validation.json) alongside the v1 Blackwell G4 baseline in [`results/iclr2026_colab_repro.json`](results/iclr2026_colab_repro.json). Tier 1 and `lung_cancer` ran on Colab `NVIDIA A100-SXM4-80GB`; `tabula_muris` (`196,311 × 22,400`) and Tier 3 `heart` `offset` ran on Colab `NVIDIA L4` High-RAM (`52 GB` host RAM). Optional Tier 3 `offset` A/B reruns on `pbmc_ifn` (upstream Dropbox link temporarily disabled) and `lung_cancer` (L4 52 GB host-RAM ceiling during 17.4 GB raw HVG prep) were skipped after their main Table 3 & Table 5 `legacy` runs had already passed on A100.

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
- Replaced per-epoch `DataLoader` worker respawns with adaptive GPU residency (`min(48 GB, 0.50 * free_vram)` FP32/FP16/pinned-CPU tiers), chunked host-to-GPU transfers (`_copy_x_to_gpu_chunked`), and streaming posterior precomputation (`_precompute_posteriors_streaming`) for FM training.
- Replaced per-condition loops in `run_formal_generation` and BEC integration with single-pass batched `SAVE.generate_like(...)` and `SAVE.transfer(..., input_cond="obs")`, preallocated output arrays, and early freeing of training matrices.
- Replaced $O(G \cdot N)$ condition masking in distribution evaluation with `groupby` index lookups, shared `torch.cdist` between 1-Wasserstein and 2-Wasserstein metrics, exact streaming randomized PCA (`_streaming_randomized_pca`), and exact tiled `mix_rbf_mmd2` (`<= 16 MB` tiles) to support `196k+` cell atlases within `52 GB` host RAM.
- Added legacy v1 `.pt` checkpoint key remapping (`tokenzier` / `pos_embding` <-> `tokenizer` / `pos_embedding`).

## [0.1.0] - 2026-04-20

### Added
- Initial public release (`sc-save-public`) containing Gene Block Construction, Conditional Generation (VAE + OT Flow Matching), Conditional Distribution Evaluation, Batch Effect Correction (`scIB`), Leave-One-Out Perturbation Prediction, dataset preparation, and unified ICLR 2026 Tables 1–5 reproduction scripts and Colab notebook.
