# ICLR 2026 Paper Reproducibility Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fdu-wangfeilab/sc-save/blob/reproduce-iclr2026/notebooks/SAVE_ICLR2026_Reproduce_Colab.ipynb)

This guide documents how to reproduce all five quantitative benchmarks (Tables 1–5) from the ICLR 2026 paper:

> **SAVE: A Generalizable Framework for Multi-Condition Single-Cell Generation with Gene Block Attention**
> Jiahao Li, Jiayi Dong, Peng Ye, Xiaochi Zhou, Haohai Lu, Fei Wang (Fudan University)
> [OpenReview](https://openreview.net/forum?id=l7QEoK4uDP) · [arXiv:2604.16776](https://arxiv.org/abs/2604.16776)

---

## Table of Contents

1. [Overview of Benchmarks (Tables 1–5)](#1-overview-of-benchmarks-tables-15)
2. [Hardware, Runtime & Environment Expectations](#2-hardware-runtime--environment-expectations)
3. [Datasets & Preprocessing Pipeline](#3-datasets--preprocessing-pipeline)
4. [Model Architecture & Hyperparameter Configuration](#4-model-architecture--hyperparameter-configuration)
5. [Option A: One-Click Google Colab Reproduction (Recommended)](#5-option-a-one-click-google-colab-reproduction-recommended)
6. [Option B: Local CLI Reproduction](#6-option-b-local-cli-reproduction)
7. [Paper Target vs. Colab G4 Reproduction Results (`reproduce-iclr2026@307c925`)](#7-paper-target-vs-colab-g4-reproduction-results-reproduce-iclr2026307c925)

---

## 1. Overview of Benchmarks (Tables 1–5)

| Paper Table | Task | Datasets | Conditioning Covariates | Primary Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1** | Single-Condition Generation | `PBMC3K`, `Dentate gyrus`, `Tabula Muris` | `cell_type` | WD ($\downarrow$), MMD ($\downarrow$) |
| **Table 2** | Dual-Condition Generation | `Heart`, `PBMC`, `Lung Atlas` | `batch`, `cell_type` | WD ($\downarrow$), MMD ($\downarrow$) |
| **Table 3** | Multi-Condition Generation (Seen & Unseen) | `Lung Cancer` (Salcher et al. 2022) | `assay`, `development_stage`, `disease`, `uicc_stage`, `cell_type` | WD ($\downarrow$), MMD ($\downarrow$) |
| **Table 4** | Batch Effect Correction (BEC) | `Lung Atlas`, `Heart`, `PBMC` | `batch`, `cell_type` | Bio ($\uparrow$), Batch ($\uparrow$), scIB ($\uparrow$) |
| **Table 5** | Perturbation Prediction (LOO) | `PBMC-IFN` (7 held-out cell types) | `condition`, `cell_type` | PCC ($\uparrow$), $R^2$ ($\uparrow$) |

---

## 2. Hardware, Runtime & Environment Expectations

### Hardware Profiles

| Environment | GPU / VRAM | System RAM | Expected Runtime | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Paper Reference** | NVIDIA RTX 3090 (24 GB) | 64 GB+ | ~12.5 min per run on Heart ($K=3200, L=6$) | $191\times$ speedup over $K=1$ naive attention (2391.2 min) |
| **Google Colab Pro+ (Recommended)** | NVIDIA G4 (96 GB), H100 (80 GB), or A100 (40/80 GB) | High-RAM (51–167 GB) | ~1–4 min per dataset; full Tables 1–5 parallelized via `--gpu-workers auto` | TF32, SDPA, `bfloat16` FM autocast, GPU-resident batching, precomputed VAE latents, & Drive cache |
| **Local Workstation** | CUDA GPU ($\ge$ 16 GB VRAM) | 32 GB+ (64 GB recommended for Lung Cancer) | Comparable or faster than RTX 3090 | Use `--device 0 --gpu-workers auto --cpu-workers auto` |

### Software & Acceleration Notes

- **Python**: 3.10+ (Colab default Python 3.10/3.11+ supported; local `uv` environment pinned to `3.10.16`).
- **PyTorch**: Preserve Colab's preinstalled CUDA-enabled PyTorch (do not downgrade PyTorch when installing `sc-save`). Checkpoint loading supports PyTorch 2.6+ (`weights_only=False` where full objects/scalers are loaded).
- **GPU & CPU Acceleration (100% Paper Parity)**:
  - **VAE & FM**: CUDA TF32, PyTorch `F.scaled_dot_product_attention`, GPU-resident mini-batching (with FP32 BCE/KL), `bfloat16` Flow Matching autocast, and precomputed frozen VAE `(mu, var)` latents.
  - **Parallel Execution**: Concurrent dataset download/OT preparation (`--workers` / `--prep-workers`), multi-process GPU task scheduling with automatic serial OOM fallback (`--gpu-workers auto`), and parallel `scib` / WD+MMD evaluation (`--cpu-workers auto`, `--scib-workers`).
- **`flash-attn`**: Not required (`is_flash_attn=False` in all released configurations).

---

## 3. Datasets & Preprocessing Pipeline

### 3.1 Standardization & Gene Block Construction

All benchmarks follow the standard single-cell preprocessing and semantic Gene Block Construction pipeline:

1. **Expression Normalization**:
   - **Tables 1, 3, 4 (BEC), & 5**: Counts normalized to $10^4$ per cell (`scanpy.pp.normalize_total(target_sum=1e4)`) followed by `log1p`.
   - **Table 2 (Dual-Condition `heart`, `pbmc`, `lung_atlas`)**: Preserves raw `layers["counts"]` and applies `scanpy.pp.log1p` directly without `normalize_total(1e4)` library-size scaling, matching `iclr-ver` (`ot_split_with_pad.ipynb`).
2. **Gene Embeddings**: Semantic 1,536-dimensional gene embeddings from GenePT (`GenePT_gene_protein_embedding_model_3_text.pickle` inside `GenePT_emebdding_v2.zip`, Zenodo record `10833191`).
3. **Optimal Transport Gene Block Partitioning**:
   - Filter to genes covered by GenePT embeddings and select HVGs so total genes $G$ is divisible by block size $K$ (or pad remainder with `pad_*` columns when retaining full gene sets at $K=3200$).
   - Partition gene embeddings into $L = G / K$ balanced, non-overlapping semantic blocks via iterative Optimal Transport (`ot.emd(..., numItermax=1000000)`).
4. **Network Input Scaling**: Fit `MaxAbsScaler` on training expression to scale values into $[0, 1]$ before neural network encoding; inverse-transform predictions and trim `pad_*` columns during generation evaluation.

### 3.2 Benchmark Dataset Specifications & Verified Sources

| Dataset Key (`prepare_datasets.py`) | Paper Table(s) | Gene Block Size ($K$) | Gene Selection | Condition Columns | Verified Source |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `pbmc3k` | Table 1 | `3200` | Covered + `pad_*` | `["cell_type"]` | [CFGen Zenodo (15031364)](https://zenodo.org/api/records/15031364/files/datasets.zip/content) (`pbmc3k_{train,test}.h5ad`) |
| `dentategyrus` | Table 1 | `3200` | Covered + `pad_*` | `["cell_type"]` | [CFGen Zenodo (15031364)](https://zenodo.org/api/records/15031364/files/datasets.zip/content) (`dentategyrus_{train,test}.h5ad`) |
| `tabula_muris` | Table 1 | `3200` | Covered + `pad_*` | `["cell_type"]` | [CFGen Zenodo (15031364)](https://zenodo.org/api/records/15031364/files/datasets.zip/content) (`tabula_muris_{train,test}.h5ad`) |
| `heart` | Table 2 | `3200` | Covered + `pad_*` | `["batch", "cell_type"]` | [scVI-data HCA 20k](https://raw.githubusercontent.com/YosefLab/scVI-data/master/hca_subsampled_20k.h5ad) |
| `pbmc` | Table 2 | `3200` | Covered + `pad_*` | `["batch", "cell_type"]` | Google Drive ID `1oKpcxQSm238SMNY77TAR5bzKUMIqgGU_` (`gdown`) |
| `lung_atlas` | Table 2 | `3200` | Covered + `pad_*` | `["batch", "cell_type"]` | [Figshare (24539942)](https://ndownloader.figshare.com/files/24539942) |
| `lung_cancer` | Table 3 | `400` | `HVG=2000` | `["assay", "development_stage", "disease", "uicc_stage", "cell_type"]` | [CellxGene (Salcher et al. 2022)](https://datasets.cellxgene.cziscience.com/99040b08-7e1a-4d81-911c-4d2fd2335757.h5ad) |
| `heart_bec` / `pbmc_bec` / `lung_atlas_bec` | Table 4 | `400` | `HVG=2000` | `["batch", "cell_type"]` | Shared raw inputs above (BEC QC + `batch_key="batch"` HVG, no `batch_scale`) |
| `pbmc_ifn` | Table 5 | `400` | `HVG=2000` | `["condition", "cell_type"]` | Dropbox ([train_pbmc.h5ad](https://www.dropbox.com/s/wk5zewf2g1oat69/train_pbmc.h5ad?dl=1) + [valid_pbmc.h5ad](https://www.dropbox.com/s/nqi971n0tk4nbfj/valid_pbmc.h5ad?dl=1)) |
| GenePT v2 Embeddings | All | — | 1,536-dim | — | [Zenodo (10833191)](https://zenodo.org/api/records/10833191/files/GenePT_emebdding_v2.zip/content) (`GenePT_gene_protein_embedding_model_3_text.pickle`) |

> **Table 3 Held-Out Split**: Filtered to 55/60/65/70-year-old human stage samples spanning 27 `(assay, development_stage, disease, uicc_stage)` combinations (Appendix Table `tab:appendix_cond_info`). Conditions **13** (`10x 3' v2`, `65-year-old human stage`, `chronic obstructive pulmonary disease`, `non-cancer`) and **24** (`Smart-seq2`, `70-year-old human stage`, `squamous cell lung carcinoma`, `I`) are held out as the **Unseen** test set (`lung_cancer_hvg2000_GBS400_test.h5ad`); the remaining 25 combinations form the **Seen** training set (`lung_cancer_hvg2000_GBS400_train.h5ad`).

> **Table 5 Leave-One-Out (LOO) Split**: Iteratively holds out all stimulated and control cells of one target cell type (`CD8T`, `CD14+Mono`, `FCGR3A+Mono`, `B`, `CD4T`, `Dendritic`, `NK`), trains on the remaining 6 cell types, and predicts stimulated profiles from held-out control cells.

### 3.3 Dataset Preparation CLI & Output Paths

Inspect all registered datasets, URLs, and expected output paths:

```bash
python examples/prepare_datasets.py --list-datasets
```

Prepare all datasets for Tables 1–5 (plus GenePT embeddings) using concurrent deduplicated downloads and parallel OT partitioning:

```bash
python examples/prepare_datasets.py \
  --datasets all \
  --data-root ./data \
  --workers auto \
  --ot-num-iter-max 1000000
```

Or prepare by table group / dataset name (`table1`..`table5`, or e.g. `heart pbmc lung_atlas`):

```bash
python examples/prepare_datasets.py --datasets table2 --data-root ./data --workers 4
```

Processed `.h5ad` outputs written under `<data-root>/processed/` (alongside `<data-root>/manifest.json`):

- **Table 1**: `table1_single_cond/{pbmc3k,dentategyrus,tabula_muris}_{train,test}_GBS3200.h5ad`
- **Table 2**: `table2_dual_cond/{heart,pbmc,lung_atlas}_GBS3200.h5ad`
- **Table 3**: `table3_multi_cond/lung_cancer_hvg2000_GBS400{,_train,_test}.h5ad`
- **Table 4**: `table4_bec/{heart,pbmc,lung_atlas}_hvg2000_GBS400.h5ad`
- **Table 5**: `table5_perturb/pbmc_ifn_hvg2000_GBS400.h5ad`

---

## 4. Model Architecture & Hyperparameter Configuration

All primary benchmarks use the unified `SAVE-B` / `SAVE-B-perturb` configuration (Appendix Table `tab:param` and `model_setting.yml`):

| Category | Parameter | Conditional Generation (`SAVE-B`) | BEC (`LDM_raw`) | Perturbation LOO (`SAVE-B-perturb`) |
| :--- | :--- | :---: | :---: | :---: |
| **Gene Blocks** | `gene_block_size` ($K$) | `3200` (Tables 1–2) / `400` (Table 3) | `400` | `400` |
| **VAE Architecture** | `hidden_dim` / `enc_dim` ($d$) | `64` / `64` | `64` / `64` | `64` / `64` |
| **VAE Architecture** | `expand_dim` ($e$) | `512` | `512` | `512` |
| **VAE Architecture** | `enc_depth` / `dec_depth` | `3` / `3` | `3` / `3` | `3` / `3` |
| **VAE Architecture** | `enc_num_heads` / `dec_num_heads` | `4` / `4` | `4` / `4` | `4` / `4` |
| **VAE Architecture** | `dec_blk_type` / `kl_scale` | `adaLN` / `0.5` | `adaLN` / `0.5` | `adaLN` / `0.5` |
| **Training** | Optimizer / `lr` / `weight_decay` | `AdamW` / `1e-4` / `2.5e-5` | `AdamW` / `1e-4` / `2.5e-5` | `AdamW` / `1e-4` / `2.5e-5` |
| **Training** | `batch_size` / `epoch` | `1024` / `200` | `1024` / `200` | `1024` / `200` |
| **Condition Masking** | `sample_mask_prob` | `0.4` (Tables 1–2) / `0.0` (Table 3 `lung_cancer`) | `0.0` (`col_msk_threshold=0.6`) | `0.0` (`col_msk_threshold=0.6`) |
| **Condition Masking** | `train_with_ctrl` | — | — | `False` |
| **Flow Matching** | `n_blocks` / `guidance_weights` / `sample_steps` | `5` / `2.0` / `20` | — (VAE latent transfer) | — (VAE latent transfer) |

### Evaluation Metric Definitions

- **Conditional Generation (Tables 1–3)**:
  - Computed in shared 30-dimensional PCA space fitted on concatenated real and generated cells (capped at 4,000 cells per condition group).
  - **WD**: Per-condition 2-Wasserstein / Earth Mover's Distance averaged across matched conditions.
  - **MMD**: Per-condition multi-scale RBF Maximum Mean Discrepancy averaged across matched conditions.
- **Batch Effect Correction (Table 4)**:
  - Evaluated via `scib` metrics on 30-dim PCA of batch-transferred integrated AnnData:
    - $\text{avg\_bio} = \frac{1}{3}(\text{NMI}_{\text{cluster/label}} + \text{ARI}_{\text{cluster/label}} + \text{ASW}_{\text{label}})$
    - $\text{avg\_batch} = \frac{1}{2}(\text{graph\_conn} + \text{ASW}_{\text{label/batch}})$
    - $\text{scIB} = 0.6 \times \text{avg\_bio} + 0.4 \times \text{avg\_batch}$
- **Perturbation Prediction (Table 5)**:
  - Evaluated on inverse-scaled (original log-normalized) per-gene mean expression across all genes for each held-out cell type: **PCC** (`scipy.stats.pearsonr`), **$R^2$** (`sklearn.metrics.r2_score`), and **MSE** (`sklearn.metrics.mean_squared_error`), plus cell-distribution **WD** and **MMD**.

---

## 5. Option A: One-Click Google Colab Reproduction (Recommended)

Launch [`notebooks/SAVE_ICLR2026_Reproduce_Colab.ipynb`](notebooks/SAVE_ICLR2026_Reproduce_Colab.ipynb) in Google Colab Pro+ with a **G4 (96 GB)**, **H100 (80 GB)**, or **A100 (40/80 GB)** High-RAM runtime:

1. **Cell 1**: Verify GPU & CUDA PyTorch environment and install `sc-save` dependencies without downgrading Colab's PyTorch.
2. **Cell 2**: Clone (or `git pull --ff-only` if present) `fdu-wangfeilab/sc-save` (`reproduce-iclr2026` branch), set parallel defaults (`PREP_WORKERS = 4`, `GPU_WORKERS = "auto"`, `CPU_WORKERS = "auto"`), and optionally enable Google Drive caching (`USE_GOOGLE_DRIVE = False` by default).
3. **Cell 3**: Download & preprocess datasets + GenePT embeddings concurrently via `--workers <PREP_WORKERS>`.
4. **Fast Parallel All-Tables Runner (Optional)**: Set `RUN_ALL_PARALLEL = True` to run **all 5 tables concurrently** (`--tasks all --gpu-workers auto --cpu-workers auto --prep-workers 4`), then jump directly to Cell 9.
5. **Cells 4–8**: Or run Tables 2, 1, 3, 4, and 5 individually or selectively (also using `GPU_WORKERS = "auto"` and `CPU_WORKERS = "auto"`), printing live comparison against paper targets.
6. **Cell 9**: Aggregate all reproduced metrics into a consolidated Paper vs. Colab table and export `iclr2026_colab_repro.json`.

---

## 6. Option B: Local CLI Reproduction

### 6.1 Environment Setup

From the repository root:

```bash
git clone --branch reproduce-iclr2026 https://github.com/fdu-wangfeilab/sc-save.git
cd sc-save

# Option 1: Using uv (recommended)
uv sync --extra all

# Option 2: Using pip inside an existing CUDA PyTorch >= 2.2 environment
pip install -e ".[all]"
```

Run offline unit tests to verify the installation:

```bash
pytest -v
```

### 6.2 Unified Reproduction CLI (`examples/reproduce_iclr2026.py`)

Preview the execution plan and expected output file paths without downloading or training:

```bash
python examples/reproduce_iclr2026.py --tasks all --data-root ./data --output-root ./outputs/iclr2026 --dry-run
```

Run **all 5 tables** end-to-end with full GPU & CPU parallelism (automatically downloads/prepares datasets concurrently, schedules parallel GPU subprocesses with automatic serial OOM fallback, evaluates metrics, and writes `<output-root>/results/iclr2026_summary.{json,md}` and per-task logs under `<output-root>/logs/`):

```bash
python examples/reproduce_iclr2026.py \
  --tasks all \
  --data-root ./data \
  --output-root ./outputs/iclr2026 \
  --device 0 \
  --seed 1202 \
  --gpu-workers auto \
  --cpu-workers auto \
  --prep-workers 4
```

Or run individual tables selectively (using distinct `--results-dir` paths if you want per-table summary files):

```bash
# Table 1: Single-Condition Generation (PBMC3K, Dentate gyrus, Tabula Muris)
python examples/reproduce_iclr2026.py --tasks table1 --data-root ./data --output-root ./outputs/iclr2026 --results-dir ./outputs/iclr2026/results/table1 --gpu-workers auto --cpu-workers auto --device 0

# Table 2: Dual-Condition Generation (Heart, PBMC, Lung Atlas)
python examples/reproduce_iclr2026.py --tasks table2 --data-root ./data --output-root ./outputs/iclr2026 --results-dir ./outputs/iclr2026/results/table2 --gpu-workers auto --cpu-workers auto --device 0

# Table 3: Multi-Condition Generation (Lung Cancer Seen & Unseen 13, 24)
python examples/reproduce_iclr2026.py --tasks table3 --data-root ./data --output-root ./outputs/iclr2026 --results-dir ./outputs/iclr2026/results/table3 --cpu-workers auto --device 0

# Table 4: Batch Effect Correction (Lung Atlas, Heart, PBMC + scIB)
python examples/reproduce_iclr2026.py --tasks table4 --data-root ./data --output-root ./outputs/iclr2026 --results-dir ./outputs/iclr2026/results/table4 --gpu-workers auto --cpu-workers auto --device 0

# Table 5: Leave-One-Out Perturbation Prediction (PBMC-IFN, 7 cell types)
python examples/reproduce_iclr2026.py --tasks table5 --data-root ./data --output-root ./outputs/iclr2026 --results-dir ./outputs/iclr2026/results/table5 --gpu-workers auto --device 0
```

Useful flags:
- `--gpu-workers <auto|int>`: Parallel GPU child processes (`"auto"` runs 4–6 concurrent jobs on $\ge$75 GB VRAM G4/H100, 2 on $\ge$36 GB A100, or up to 7 LOO folds on single-dataset `table5`; defaults to `"1"` for serial compatibility).
- `--cpu-workers <auto|int>`: Concurrent CPU workers for `scib` and WD/MMD distribution evaluation (`"auto"` -> `min(8, os.cpu_count() or 1)`).
- `--prep-workers <int>`: Upfront dataset download & OT workers passed to `prepare_datasets` (default `4`).
- `--skip-prepare`: Assert that datasets under `--data-root/processed/` are already prepared.
- `--datasets pbmc3k,heart`: Restrict execution to specific datasets.
- `--subset-cells 512 --vae-epoch 1 --fm-epoch 1 --epoch 1`: Fast end-to-end smoke check.
- `--overwrite-ckpt`: Retrain even if checkpoints already exist under `--output-root`.

### 6.3 Standalone Task Entrypoints (Alternative)

Each benchmark can also be invoked directly via its dedicated script:

```bash
# Conditional Generation (Tables 1-3) + Evaluation
python examples/conditional_generation/run_formal.py \
  --data-path ./data/processed/table2_dual_cond/heart_GBS3200.h5ad \
  --task-name heart \
  --cond-cols batch cell_type \
  --gene-block-size 3200 \
  --output-dir ./outputs/iclr2026/table2/heart \
  --device 0

python examples/conditional_distribution_evaluation/run_example.py \
  --gt-path ./outputs/iclr2026/table2/heart/target.h5ad \
  --gen-path ./outputs/iclr2026/table2/heart/generation.h5ad \
  --cond-cols batch cell_type \
  --sort-cols batch cell_type \
  --output-dir ./outputs/iclr2026/table2/heart/eval \
  --device 0

# Batch Effect Correction (Table 4)
python examples/bec_integration/run_bec.py \
  --data-path ./data/processed/table4_bec/lung_atlas_hvg2000_GBS400.h5ad \
  --task-name lung_atlas_bec \
  --output-dir ./outputs/iclr2026/table4/lung_atlas_bec \
  --scib-workers 8 \
  --device 0

# Perturbation Prediction LOO (Table 5)
python examples/perturbation_prediction/run_leave_one_out.py \
  --data-path ./data/processed/table5_perturb/pbmc_ifn_hvg2000_GBS400.h5ad \
  --output-dir ./outputs/iclr2026/table5/pbmc_ifn \
  --workers auto \
  --device 0
```

---

## 7. Paper Target vs. Colab G4 Reproduction Results (`reproduce-iclr2026@307c925`)

> **Verification Status**: **Completed** across all 5 tables (11 datasets) on Google Colab Pro+ (`reproduce-iclr2026@307c925`). Full raw JSON output is archived in [`results/iclr2026_colab_repro.json`](results/iclr2026_colab_repro.json).

### 7.1 Colab G4 Runtime Environment, Dataset Dimensions & Wall-Clock Timings

- **Hardware & Software**: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (**94.97 GB VRAM**; Colab Pro+ G4 selected while H100/A100 instances were busy), Python `3.13.16`, PyTorch `2.14.1+cu130`, CUDA `13.0`, `prep_workers=4`, `gpu_workers="auto"` (4 concurrent GPU worker processes), `cpu_workers="auto"` (8 CPU workers).
- **Verified `manifest.json` Dataset Dimensions**:

| Dataset Key | Table | Shape (`n_obs × n_vars`) | `n_shared_genept` | `n_semantic_blocks` | `n_pad` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `pbmc3k` | Table 1 | `2,110 × 9,600` | `8,248` | `2` | `1,027` |
| `dentategyrus` | Table 1 | `14,570 × 19,200` | `78` | `0` | `2,198` |
| `tabula_muris` | Table 1 | `196,311 × 22,400` | `97` | `0` | `2,666` |
| `heart` | Table 2 | `20,000 × 28,800` | `19,112` | `5` | `2,138` |
| `pbmc` | Table 2 | `33,506 × 12,800` | `10,757` | `3` | `1,949` |
| `lung_atlas` | Table 2 | `32,472 × 16,000` | `13,714` | `4` | `852` |
| `lung_cancer` | Table 3 | `114,692 × 2,000` | `1,925` | `4` | `0` |
| `heart_bec` | Table 4 | `16,159 × 2,000` | `1,769` | `4` | `0` |
| `pbmc_bec` | Table 4 | `32,603 × 2,000` | `1,984` | `4` | `0` |
| `lung_atlas_bec` | Table 4 | `25,486 × 2,000` | `1,901` | `4` | `0` |
| `pbmc_ifn` | Table 5 | `18,868 × 2,000` | `1,834` | `4` | `0` |

- **Observed Wall-Clock Timings (Colab G4)**:
  - Single-dataset `heart` (Table 2) accelerated from `~16m` to **`~7m`** (`vae=109.03s`, `fm=131.50s`, `gen=52.94s`).
  - Parallel Tables 1–4 batch finished all **10 tasks in `5,295s` (`88m 15s`)** wall-clock (`tabula_muris` `88m 15s`, `lung_cancer` `70m 32s`, `pbmc_bec` `51m 04s`, `lung_atlas_bec` `47m 00s`, `lung_atlas` `30m 14s`, `pbmc` `25m 07s`, `dentategyrus` `9m 38s`, `pbmc3k` `1m 51s`).

> **Reproduction Transparency Notes**:
> 1. **Step A vs. Parallel-Batch Rerun (`heart` & `heart_bec`)**: [`results/iclr2026_colab_repro.json`](results/iclr2026_colab_repro.json) follows Cell 9's merge order where per-table `table2/heart` (fresh Step A run: `wd=8.8298` / `8.83`, `mmd=0.6416` / `0.64`, `mse=0.0158`, `pcc=0.8710`, `r2=0.5828`) and `table4/heart_bec` (fresh Step A run: `avg_bio=0.7539`, `avg_batch=0.8740`, `scib_score=0.8019`) took precedence over the parallel Tables 1–4 batch re-evaluation (`table2/heart` reusing Step A checkpoints: `wd=8.9382` / `8.94`, `mmd=0.6394` / `0.64`, `mse=0.0155` / `0.02`, `pcc=0.8684` / `0.87`, `r2=0.5830` / `0.58`; `table4/heart_bec` parallel rerun: `avg_bio=0.7527`, `avg_batch=0.8649`, `scib_score=0.7975`).
> 2. **Hardware / Stochastic Variance**: Minor variation relative to RTX 3090 FP32 paper runs is expected under Blackwell CUDA 13 `bfloat16` Flow Matching autocast and stochastic OT/Leiden/FM sampling—notably `lung_cancer` unseen split `(13, 24)` (`WD 5.39 / PCC 0.82 / R2 0.65` vs. paper `4.63 / 0.85 / 0.70`) and `pbmc_ifn` LOO `CD14+Mono` (`WD 7.68 / PCC 0.91 / R2 0.71` vs. paper `5.03 / 0.97 / 0.89`, yielding `Average` `PCC 0.943 / R2 0.807 / WD 5.77` vs. paper `0.96 / 0.86 / 5.15`), while `table1/pbmc3k` (`WD 12.76` vs. `19.14`) and `dentategyrus` (`WD 8.08` vs. `9.16`) match or outperform paper targets.

---

### Table 1: Single-Condition Generation (`cell_type`)

| Dataset | Metric | scVI | scDiffusion | CFGen | SAVE (Paper) | SAVE (Colab G4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PBMC3K** | WD ($\downarrow$) | 17.66 | 22.41 | 16.94 | 19.14 | **12.76** (`12.7560`) |
| **PBMC3K** | MMD ($\downarrow$) | 0.94 | 1.27 | 0.85 | 1.80 | **0.48** (`0.4794`) |
| **Dentate gyrus** | WD ($\downarrow$) | 22.61 | 22.56 | 21.55 | 9.16 | **8.08** (`8.0844`) |
| **Dentate gyrus** | MMD ($\downarrow$) | 1.15 | 1.22 | 1.12 | 0.17 | **0.11** (`0.1055`) |
| **Tabula Muris** | WD ($\downarrow$) | 9.76 | 7.89 | **7.39** | 10.86 | 10.65 (`10.6501`) |
| **Tabula Muris** | MMD ($\downarrow$) | 0.26 | 0.24 | 0.19 | **0.04** | **0.04** (`0.0356`) |

<details>
<summary>Appendix Table: Table 1 Gene-Level Expression Metrics (SAVE Paper vs. Colab G4)</summary>

| Dataset | Paper MSE ($\downarrow$) | Paper PCC ($\uparrow$) | Paper $R^2$ ($\uparrow$) | Colab G4 MSE | Colab G4 PCC | Colab G4 $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PBMC3K** | 0.08 ± 0.18 | 0.93 ± 0.12 | 0.84 ± 0.27 | `0.0406` | `0.9428` | `0.8919` |
| **Dentate gyrus** | 0.00 ± 0.00 | 0.99 ± 0.01 | 0.97 ± 0.02 | `0.0048` | `0.9910` | `0.9736` |
| **Tabula Muris** | 0.00 ± 0.00 | 0.99 ± 0.01 | 0.95 ± 0.05 | `0.0049` | `0.9892` | `0.9520` |

</details>

---

### Table 2: Dual-Condition Generation (`batch`, `cell_type`)

| Dataset | Metric | scVI | CFGen | scDiffusion | SAVE (Paper) | SAVE (Colab G4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Heart** | WD ($\downarrow$) | 19.18 | 12.57 | 20.82 | **8.30** | 8.83 (`8.8298`) |
| **Heart** | MMD ($\downarrow$) | 1.19 | 0.66 | 0.94 | **0.63** | 0.64 (`0.6416`) |
| **PBMC** | WD ($\downarrow$) | 17.75 | 17.41 | 11.38 | **5.37** | 5.39 (`5.3851`) |
| **PBMC** | MMD ($\downarrow$) | 0.95 | 0.65 | 0.48 | **0.29** | 0.30 (`0.3027`) |
| **Lung Atlas** | WD ($\downarrow$) | 22.20 | 28.02 | 13.89 | **4.37** | 4.42 (`4.4174`) |
| **Lung Atlas** | MMD ($\downarrow$) | 2.27 | 1.76 | 1.71 | **1.14** | 1.16 (`1.1579`) |

<details>
<summary>Appendix Table: Table 2 Gene-Level Expression Metrics (SAVE Paper vs. Colab G4)</summary>

| Dataset | Paper MSE ($\downarrow$) | Paper PCC ($\uparrow$) | Paper $R^2$ ($\uparrow$) | Colab G4 MSE | Colab G4 PCC | Colab G4 $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Heart** | 0.01 ± 0.02 | 0.88 ± 0.13 | 0.63 ± 0.26 | `0.0158` | `0.8710` | `0.5828` |
| **PBMC** | 0.02 ± 0.02 | 0.98 ± 0.03 | 0.95 ± 0.07 | `0.0061` | `0.9857` | `0.9689` |
| **Lung Atlas** | 0.01 ± 0.01 | 0.91 ± 0.11 | 0.84 ± 0.18 | `0.0089` | `0.9121` | `0.8308` |

</details>

---

### Table 3: Multi-Condition Generation on Lung Cancer (5 Covariates)

| Split | Metric | CFGen | scDiffusion | SAVE (Paper) | SAVE (Colab G4) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Seen** (25 combos) | WD ($\downarrow$) | 16.94 ± 4.19 | 5.27 ± 2.14 | 3.10 ± 0.96 | **3.07** (`3.0661`) |
| **Seen** (25 combos) | MMD ($\downarrow$) | 1.49 ± 1.80 | 1.06 ± 1.59 | **1.06** ± 1.48 | 1.08 (`1.0762`) |
| **Unseen** (combos 13 & 24) | WD ($\downarrow$) | 23.67 ± 3.08 | 5.29 ± 2.03 | **4.63** ± 0.95 | 5.39 (`5.3855`) |
| **Unseen** (combos 13 & 24) | MMD ($\downarrow$) | 2.76 ± 2.64 | **1.87 ± 2.31** | 2.11 ± 2.11 | 2.16 (`2.1557`) |
| **Macro** (all 27 combos) | WD ($\downarrow$) | — | — | **3.15** | 3.23 (`3.2299`) |
| **Macro** (all 27 combos) | MMD ($\downarrow$) | — | — | **1.29** | 1.31 (`1.3057`) |

<details>
<summary>Appendix Tables: Table 3 Gene-Level & Held-Out Per-Condition Breakdown (SAVE Paper vs. Colab G4)</summary>

| Subset | Paper MSE ($\downarrow$) | Paper PCC ($\uparrow$) | Paper $R^2$ ($\uparrow$) | Colab G4 MSE | Colab G4 PCC | Colab G4 $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seen** | 0.01 ± 0.02 | 0.94 ± 0.07 | 0.88 ± 0.14 | `0.0117` | `0.9443` | `0.8812` |
| **Unseen** | 0.04 ± 0.03 | 0.85 ± 0.08 | 0.70 ± 0.14 | `0.0441` | `0.8168` | `0.6467` |

Paper per-condition distribution metrics (`tab:multi_cond_performance`):
- Condition 13 (Unseen): WD = **4.60**, MMD = **2.15**
- Condition 24 (Unseen): WD = **3.36**, MMD = **1.86**
- All 27 conditions mean: WD = **3.15**, MMD = **1.29** (Colab G4 macro: WD = `3.2299`, MMD = `1.3057`)

</details>

---

### Table 4: Batch Effect Correction (scIB Evaluation)

| Dataset | Metric | Scanorama | Harmony | scVI | trVAE | SAVE (Paper) | SAVE (Colab G4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lung Atlas** | Bio ($\uparrow$) | 0.70 | 0.65 | 0.58 | 0.69 | **0.73** (`0.7340`) | **0.73** (`0.7342`) |
| **Lung Atlas** | Batch ($\uparrow$) | 0.88 | **0.93** | 0.83 | 0.90 | **0.93** (`0.9309`) | 0.92 (`0.9231`) |
| **Lung Atlas** | scIB ($\uparrow$) | 0.77 | 0.76 | 0.68 | 0.78 | **0.81** (`0.8128`) | **0.81** (`0.8097`) |
| **Heart** | Bio ($\uparrow$) | 0.72 | **0.76** | 0.66 | 0.75 | **0.76** (`0.7574`) | 0.75 (`0.7539`) |
| **Heart** | Batch ($\uparrow$) | 0.82 | **0.89** | 0.81 | 0.83 | 0.86 (`0.8596`) | 0.87 (`0.8740`) |
| **Heart** | scIB ($\uparrow$) | 0.76 | **0.81** | 0.72 | 0.78 | 0.80 (`0.7983`) | 0.80 (`0.8019`) |
| **PBMC** | Bio ($\uparrow$) | 0.71 | 0.71 | 0.47 | **0.75** | **0.75** (`0.7516`) | 0.73 (`0.7330`) |
| **PBMC** | Batch ($\uparrow$) | 0.93 | **0.95** | 0.78 | 0.91 | **0.95** (`0.9498`) | 0.93 (`0.9317`) |
| **PBMC** | scIB ($\uparrow$) | 0.80 | 0.80 | 0.59 | 0.82 | **0.83** (`0.8309`) | 0.81 (`0.8125`) |

<details>
<summary>Appendix Tables: Detailed scIB Sub-Metrics for SAVE (Paper Target vs. Colab G4)</summary>

| Dataset | Source | NMI | ARI | ASW_label | ASW_batch | PCR_batch | graph_conn | avg_bio | avg_batch | scib_score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PBMC** | Paper | 0.8637 | 0.7952 | 0.5958 | 0.9128 | 0.4996 | 0.9868 | 0.7516 | 0.9498 | 0.8309 |
| **PBMC** | Colab G4 | 0.8385 | 0.7476 | 0.6128 | 0.8789 | 0.5709 | 0.9845 | 0.7330 | 0.9317 | 0.8125 |
| **Lung Atlas** | Paper | 0.8285 | 0.7340 | 0.6395 | 0.8693 | 0.2134 | 0.9926 | 0.7340 | 0.9309 | 0.8128 |
| **Lung Atlas** | Colab G4 | 0.8257 | 0.7186 | 0.6583 | 0.8542 | 0.1798 | 0.9919 | 0.7342 | 0.9231 | 0.8097 |
| **Heart** | Paper | 0.7999 | 0.8146 | 0.6579 | 0.8576 | 0.2554 | 0.8615 | 0.7574 | 0.8596 | 0.7983 |
| **Heart** | Colab G4 | 0.7986 | 0.8138 | 0.6492 | 0.8497 | 0.1640 | 0.8983 | 0.7539 | 0.8740 | 0.8019 |

</details>

---

### Table 5: Leave-One-Out Perturbation Prediction (`PBMC-IFN`)

| Held-Out Cell Type | Paper PCC ($\uparrow$) | Colab G4 PCC | Paper $R^2$ ($\uparrow$) | Colab G4 $R^2$ | Paper WD / MMD ($\downarrow$) | Colab G4 WD / MMD | Paper MSE ($\downarrow$) | Colab G4 MSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CD8T** | 0.98 | `0.9840` | 0.97 | `0.9642` | 4.01 / 0.11 | `4.06 / 0.03` | 0.00 | `0.0012` |
| **CD14+Mono** | 0.97 | `0.9093` | 0.89 | `0.7098` | 5.03 / 0.09 | `7.68 / 0.28` | 0.00 | `0.0189` |
| **FCGR3A+Mono** | 0.91 | `0.9090` | 0.53 | `0.5612` | 7.01 / 0.22 | `8.02 / 0.30` | 0.01 | `0.0206` |
| **B** | 0.97 | `0.9491` | 0.94 | `0.8664` | 4.40 / 0.17 | `4.76 / 0.09` | 0.00 | `0.0043` |
| **CD4T** | 0.96 | `0.9665` | 0.97 | `0.9312` | 3.98 / 0.11 | `3.57 / 0.05` | 0.00 | `0.0020` |
| **Dendritic** | 0.96 | `0.9246` | 0.81 | `0.7233` | 6.55 / 0.17 | `7.36 / 0.22` | 0.01 | `0.0137` |
| **NK** | 0.96 | `0.9605` | 0.90 | `0.8952` | 5.07 / 0.09 | `4.97 / 0.07` | 0.00 | `0.0036` |
| **Average** | **0.96** | **`0.9433`** | **0.86** | **`0.8073`** | **5.15 / 0.14** | **`5.77 / 0.15`** | **0.00** | **`0.0092`** |
