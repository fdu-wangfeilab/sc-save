"""Dataset specifications, paper metadata, aliases, and path resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Verified Public Source Constants & Paper Metadata
# ---------------------------------------------------------------------------

GENEPT_ZENODO_URL = (
    "https://zenodo.org/api/records/10833191/files/GenePT_emebdding_v2.zip/content"
)
GENEPT_ZENODO_BACKUP_URL = (
    "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
)
GENEPT_PICKLE_NAME = "GenePT_gene_protein_embedding_model_3_text.pickle"

CFGEN_ZENODO_URL = (
    "https://zenodo.org/api/records/15031364/files/datasets.zip/content"
)
CFGEN_ZENODO_BACKUP_URL = (
    "https://zenodo.org/records/15031364/files/datasets.zip?download=1"
)

HEART_URL = (
    "https://raw.githubusercontent.com/YosefLab/scVI-data/master/hca_subsampled_20k.h5ad"
)
PBMC_GDRIVE_ID = "1oKpcxQSm238SMNY77TAR5bzKUMIqgGU_"
PBMC_URL = f"https://drive.google.com/uc?id={PBMC_GDRIVE_ID}"
LUNG_ATLAS_URL = "https://ndownloader.figshare.com/files/24539942"
LUNG_CANCER_URL = (
    "https://datasets.cellxgene.cziscience.com/"
    "99040b08-7e1a-4d81-911c-4d2fd2335757.h5ad"
)
PBMC_IFN_TRAIN_URL = (
    "https://www.dropbox.com/s/wk5zewf2g1oat69/train_pbmc.h5ad?dl=1"
)
PBMC_IFN_VALID_URL = (
    "https://www.dropbox.com/s/nqi971n0tk4nbfj/valid_pbmc.h5ad?dl=1"
)

LUNG_CANCER_AGE_STAGES: tuple[str, ...] = (
    "55-year-old human stage",
    "60-year-old human stage",
    "65-year-old human stage",
    "70-year-old human stage",
)

# Appendix Table tab:appendix_cond_info held-out combinations (indices 13 and 24)
LUNG_CANCER_UNSEEN_CONDITIONS: tuple[dict[str, str], ...] = (
    {
        "development_stage": "65-year-old human stage",
        "assay": "10x 3' v2",
        "disease": "chronic obstructive pulmonary disease",
        "uicc_stage": "non-cancer",
    },
    {
        "development_stage": "70-year-old human stage",
        "assay": "Smart-seq2",
        "disease": "squamous cell lung carcinoma",
        "uicc_stage": "I",
    },
)


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable specification of a benchmark dataset and its paper preprocessing."""

    name: str
    table: str
    task_type: str
    description: str
    cond_cols: tuple[str, ...]
    gene_block_size: int
    hvg_top_genes: int | None
    source_urls: tuple[str, ...]
    backup_urls: tuple[str, ...]
    download_kind: str
    raw_subdir: str
    raw_filenames: tuple[str, ...]
    processed_subdir: str
    output_filenames: dict[str, str]
    primary_key: str
    aliases: tuple[str, ...] = ()
    gdrive_id: str | None = None
    archive_members: tuple[str, ...] = ()
    min_genes: int | None = None
    min_cells: int | None = None
    exclude_gene_prefixes: tuple[str, ...] = ()


@dataclass
class PreparedDataset:
    """Result record returned after preparing a dataset and written to manifest.json."""

    name: str
    table: str
    task_type: str
    cond_cols: list[str]
    gene_block_size: int
    hvg_top_genes: int | None
    primary_path: Path
    files: dict[str, Path]
    n_obs: int
    n_vars: int
    n_shared_genept: int
    n_semantic_blocks: int
    n_pad: int
    source_urls: list[str]

    def to_dict(self, data_root: Path | None = None) -> dict[str, Any]:
        """Serialize metadata to JSON-friendly dictionary."""

        def _fmt(path: Path) -> str:
            if data_root is not None:
                try:
                    return str(path.resolve().relative_to(data_root.resolve()))
                except ValueError:
                    pass
            return str(path)

        return {
            "name": self.name,
            "table": self.table,
            "task_type": self.task_type,
            "cond_cols": list(self.cond_cols),
            "gene_block_size": int(self.gene_block_size),
            "hvg_top_genes": self.hvg_top_genes,
            "primary_path": _fmt(self.primary_path),
            "files": {k: _fmt(v) for k, v in self.files.items()},
            "n_obs": int(self.n_obs),
            "n_vars": int(self.n_vars),
            "n_shared_genept": int(self.n_shared_genept),
            "n_semantic_blocks": int(self.n_semantic_blocks),
            "n_pad": int(self.n_pad),
            "source_urls": list(self.source_urls),
        }


DATASET_SPECS: dict[str, DatasetSpec] = {
    # Table 1: Single-Condition Generation (CFGen splits, GBS=3200 + pad_*)
    "pbmc3k": DatasetSpec(
        name="pbmc3k",
        table="table1",
        task_type="single_cond",
        description="Table 1 single-condition PBMC3K dataset (CFGen train/test split, GBS=3200).",
        cond_cols=("cell_type",),
        gene_block_size=3200,
        hvg_top_genes=None,
        source_urls=(CFGEN_ZENODO_URL,),
        backup_urls=(CFGEN_ZENODO_BACKUP_URL,),
        download_kind="cfgen_zip64",
        raw_subdir="raw/table1_single_cond/pbmc3k",
        raw_filenames=("pbmc3k_train.h5ad", "pbmc3k_test.h5ad"),
        processed_subdir="processed/table1_single_cond",
        output_filenames={
            "train": "pbmc3k_train_GBS3200.h5ad",
            "test": "pbmc3k_test_GBS3200.h5ad",
        },
        primary_key="train",
        aliases=("pbmc3k_gen",),
        archive_members=(
            "datasets/pbmc3k/pbmc3k_train.h5ad",
            "datasets/pbmc3k/pbmc3k_test.h5ad",
        ),
    ),
    "dentategyrus": DatasetSpec(
        name="dentategyrus",
        table="table1",
        task_type="single_cond",
        description="Table 1 single-condition Dentate gyrus dataset (CFGen split, GBS=3200).",
        cond_cols=("cell_type",),
        gene_block_size=3200,
        hvg_top_genes=None,
        source_urls=(CFGEN_ZENODO_URL,),
        backup_urls=(CFGEN_ZENODO_BACKUP_URL,),
        download_kind="cfgen_zip64",
        raw_subdir="raw/table1_single_cond/dentategyrus",
        raw_filenames=("dentategyrus_train.h5ad", "dentategyrus_test.h5ad"),
        processed_subdir="processed/table1_single_cond",
        output_filenames={
            "train": "dentategyrus_train_GBS3200.h5ad",
            "test": "dentategyrus_test_GBS3200.h5ad",
        },
        primary_key="train",
        aliases=("dentate_gyrus",),
        archive_members=(
            "datasets/dentategyrus/dentategyrus_train.h5ad",
            "datasets/dentategyrus/dentategyrus_test.h5ad",
        ),
    ),
    "tabula_muris": DatasetSpec(
        name="tabula_muris",
        table="table1",
        task_type="single_cond",
        description="Table 1 single-condition Tabula Muris dataset (CFGen split, GBS=3200).",
        cond_cols=("cell_type",),
        gene_block_size=3200,
        hvg_top_genes=None,
        source_urls=(CFGEN_ZENODO_URL,),
        backup_urls=(CFGEN_ZENODO_BACKUP_URL,),
        download_kind="cfgen_zip64",
        raw_subdir="raw/table1_single_cond/tabula_muris",
        raw_filenames=("tabula_muris_train.h5ad", "tabula_muris_test.h5ad"),
        processed_subdir="processed/table1_single_cond",
        output_filenames={
            "train": "tabula_muris_train_GBS3200.h5ad",
            "test": "tabula_muris_test_GBS3200.h5ad",
        },
        primary_key="train",
        aliases=("tabulamuris",),
        archive_members=(
            "datasets/tabula_muris/tabula_muris_train.h5ad",
            "datasets/tabula_muris/tabula_muris_test.h5ad",
        ),
    ),
    # Table 2: Dual-Condition Generation (batch + cell_type, GBS=3200 + pad_*)
    "heart": DatasetSpec(
        name="heart",
        table="table2",
        task_type="dual_cond",
        description="Table 2 dual-condition Heart Cell Atlas (20k) dataset (GBS=3200).",
        cond_cols=("batch", "cell_type"),
        gene_block_size=3200,
        hvg_top_genes=None,
        source_urls=(HEART_URL,),
        backup_urls=(),
        download_kind="direct",
        raw_subdir="raw/dual_cond",
        raw_filenames=("hca_subsampled_20k.h5ad",),
        processed_subdir="processed/table2_dual_cond",
        output_filenames={"full": "heart_GBS3200.h5ad"},
        primary_key="full",
        aliases=("hca_subsampled_20k", "heart_gen"),
    ),
    "pbmc": DatasetSpec(
        name="pbmc",
        table="table2",
        task_type="dual_cond",
        description="Table 2 dual-condition multi-platform PBMC dataset (GBS=3200).",
        cond_cols=("batch", "cell_type"),
        gene_block_size=3200,
        hvg_top_genes=None,
        source_urls=(PBMC_URL,),
        backup_urls=(),
        download_kind="gdrive",
        raw_subdir="raw/dual_cond",
        raw_filenames=("pbmc_vars_sb.h5ad",),
        processed_subdir="processed/table2_dual_cond",
        output_filenames={"full": "pbmc_GBS3200.h5ad"},
        primary_key="full",
        aliases=("pbmc_vars_sb", "pbmc_gen"),
        gdrive_id=PBMC_GDRIVE_ID,
    ),
    "lung_atlas": DatasetSpec(
        name="lung_atlas",
        table="table2",
        task_type="dual_cond",
        description="Table 2 dual-condition Lung Atlas dataset (GBS=3200).",
        cond_cols=("batch", "cell_type"),
        gene_block_size=3200,
        hvg_top_genes=None,
        source_urls=(LUNG_ATLAS_URL,),
        backup_urls=(),
        download_kind="direct",
        raw_subdir="raw/dual_cond",
        raw_filenames=("lung_atlas.h5ad",),
        processed_subdir="processed/table2_dual_cond",
        output_filenames={"full": "lung_atlas_GBS3200.h5ad"},
        primary_key="full",
        aliases=("lung_atlas_gen",),
    ),
    # Table 3: Multi-Condition Generation on Salcher Lung Cancer (HVG=2000, GBS=400)
    "lung_cancer": DatasetSpec(
        name="lung_cancer",
        table="table3",
        task_type="multi_cond",
        description=(
            "Table 3 multi-condition Lung Cancer atlas filtered to ages 55/60/65/70 "
            "with seen/unseen split (conditions 13 & 24 held out, HVG=2000, GBS=400)."
        ),
        cond_cols=(
            "assay",
            "development_stage",
            "disease",
            "uicc_stage",
            "cell_type",
        ),
        gene_block_size=400,
        hvg_top_genes=2000,
        source_urls=(LUNG_CANCER_URL,),
        backup_urls=(),
        download_kind="direct",
        raw_subdir="raw/table3_multi_cond",
        raw_filenames=("cancer_lung.h5ad",),
        processed_subdir="processed/table3_multi_cond",
        output_filenames={
            "full": "lung_cancer_hvg2000_GBS400.h5ad",
            "train": "lung_cancer_hvg2000_GBS400_train.h5ad",
            "test": "lung_cancer_hvg2000_GBS400_test.h5ad",
        },
        primary_key="full",
        aliases=("cancer_lung", "cancer_lung_55_70"),
        min_genes=600,
        min_cells=3,
    ),
    # Table 4: Batch Effect Correction (HVG=2000, GBS=400, without batch_scale)
    "heart_bec": DatasetSpec(
        name="heart_bec",
        table="table4",
        task_type="bec",
        description="Table 4 BEC Heart dataset (QC + 2000 HVG, GBS=400).",
        cond_cols=("batch", "cell_type"),
        gene_block_size=400,
        hvg_top_genes=2000,
        source_urls=(HEART_URL,),
        backup_urls=(),
        download_kind="direct",
        raw_subdir="raw/dual_cond",
        raw_filenames=("hca_subsampled_20k.h5ad",),
        processed_subdir="processed/table4_bec",
        output_filenames={"full": "heart_hvg2000_GBS400.h5ad"},
        primary_key="full",
        aliases=("hca_subsampled_20k_bec",),
        min_genes=600,
        min_cells=3,
        exclude_gene_prefixes=("ERCC", "MT-", "mt-"),
    ),
    "pbmc_bec": DatasetSpec(
        name="pbmc_bec",
        table="table4",
        task_type="bec",
        description="Table 4 BEC PBMC dataset (QC + 2000 HVG, GBS=400).",
        cond_cols=("batch", "cell_type"),
        gene_block_size=400,
        hvg_top_genes=2000,
        source_urls=(PBMC_URL,),
        backup_urls=(),
        download_kind="gdrive",
        raw_subdir="raw/dual_cond",
        raw_filenames=("pbmc_vars_sb.h5ad",),
        processed_subdir="processed/table4_bec",
        output_filenames={"full": "pbmc_hvg2000_GBS400.h5ad"},
        primary_key="full",
        aliases=("pbmc_vars_sb_bec",),
        gdrive_id=PBMC_GDRIVE_ID,
        min_genes=600,
        min_cells=3,
        exclude_gene_prefixes=("ERCC", "MT-", "mt-"),
    ),
    "lung_atlas_bec": DatasetSpec(
        name="lung_atlas_bec",
        table="table4",
        task_type="bec",
        description="Table 4 BEC Lung Atlas dataset (QC + 2000 HVG, GBS=400).",
        cond_cols=("batch", "cell_type"),
        gene_block_size=400,
        hvg_top_genes=2000,
        source_urls=(LUNG_ATLAS_URL,),
        backup_urls=(),
        download_kind="direct",
        raw_subdir="raw/dual_cond",
        raw_filenames=("lung_atlas.h5ad",),
        processed_subdir="processed/table4_bec",
        output_filenames={"full": "lung_atlas_hvg2000_GBS400.h5ad"},
        primary_key="full",
        aliases=(),
        min_genes=600,
        min_cells=3,
        exclude_gene_prefixes=("ERCC", "MT-", "mt-"),
    ),
    # Table 5: Leave-One-Out Perturbation Prediction (PBMC-IFN, HVG=2000, GBS=400)
    "pbmc_ifn": DatasetSpec(
        name="pbmc_ifn",
        table="table5",
        task_type="perturb_loo",
        description="Table 5 PBMC-IFN perturbation LOO dataset (HVG=2000, GBS=400).",
        cond_cols=("condition", "cell_type"),
        gene_block_size=400,
        hvg_top_genes=2000,
        source_urls=(PBMC_IFN_TRAIN_URL, PBMC_IFN_VALID_URL),
        backup_urls=(),
        download_kind="multi_direct",
        raw_subdir="raw/table5_perturb",
        raw_filenames=("train_pbmc.h5ad", "valid_pbmc.h5ad"),
        processed_subdir="processed/table5_perturb",
        output_filenames={"full": "pbmc_ifn_hvg2000_GBS400.h5ad"},
        primary_key="full",
        aliases=("pbmc-ifn", "kang_pbmc", "pbmc_perturb"),
    ),
}

TABLE_DATASETS: dict[str, tuple[str, ...]] = {
    "table1": ("pbmc3k", "dentategyrus", "tabula_muris"),
    "table2": ("heart", "pbmc", "lung_atlas"),
    "table3": ("lung_cancer",),
    "table4": ("heart_bec", "pbmc_bec", "lung_atlas_bec"),
    "table5": ("pbmc_ifn",),
}

DATASET_ALIASES: dict[str, str] = {}
for _name, _spec in DATASET_SPECS.items():
    DATASET_ALIASES[_name] = _name
    for _alias in _spec.aliases:
        DATASET_ALIASES[_alias.lower()] = _name


def resolve_workers(workers: int | str = 1) -> int:
    """Resolve worker count ('auto' -> min(8, cpu_count), or positive integer)."""
    if isinstance(workers, str):
        stripped = workers.strip().lower()
        if stripped == "auto":
            return max(1, min(8, os.cpu_count() or 1))
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Invalid workers value '{workers}'. Expected 'auto' or positive integer."
            ) from exc
        if parsed < 1:
            raise ValueError(f"workers must be >= 1, got {parsed}")
        return parsed
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise TypeError(f"workers must be int or 'auto', got {type(workers).__name__}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return workers


def list_datasets() -> list[DatasetSpec]:
    """Return all canonical dataset specifications in paper table order."""
    return list(DATASET_SPECS.values())


def get_dataset_spec(name: str) -> DatasetSpec:
    """Resolve a canonical dataset name or alias to its DatasetSpec."""
    key = DATASET_ALIASES.get(name.strip().lower())
    if key is None:
        valid = ", ".join(list(DATASET_SPECS.keys()) + list(TABLE_DATASETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {valid}, all.")
    return DATASET_SPECS[key]


def resolve_dataset_names(datasets: str | Sequence[str]) -> list[str]:
    """Expand 'all', 'table1'..'table5', aliases, and comma strings to canonical names."""
    if isinstance(datasets, str):
        tokens = [tok.strip() for tok in datasets.split(",") if tok.strip()]
    else:
        tokens = []
        for item in datasets:
            tokens.extend([tok.strip() for tok in str(item).split(",") if tok.strip()])

    if not tokens:
        return list(DATASET_SPECS.keys())

    resolved: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        low = token.lower()
        if low == "all":
            candidates = list(DATASET_SPECS.keys())
        elif low in TABLE_DATASETS:
            candidates = list(TABLE_DATASETS[low])
        else:
            candidates = [get_dataset_spec(token).name]
        for cand in candidates:
            if cand not in seen:
                seen.add(cand)
                resolved.append(cand)
    return resolved


def get_output_paths(name: str, data_root: str | Path) -> dict[str, Path]:
    """Return expected processed .h5ad output paths for a dataset under data_root."""
    spec = get_dataset_spec(name)
    root = Path(data_root)
    return {
        key: root / spec.processed_subdir / fname
        for key, fname in spec.output_filenames.items()
    }


def _get_raw_paths(spec: DatasetSpec, data_root: Path) -> list[Path]:
    return [data_root / spec.raw_subdir / fname for fname in spec.raw_filenames]


__all__ = [
    "CFGEN_ZENODO_BACKUP_URL",
    "CFGEN_ZENODO_URL",
    "DATASET_ALIASES",
    "DATASET_SPECS",
    "DatasetSpec",
    "GENEPT_PICKLE_NAME",
    "GENEPT_ZENODO_BACKUP_URL",
    "GENEPT_ZENODO_URL",
    "HEART_URL",
    "LUNG_ATLAS_URL",
    "LUNG_CANCER_AGE_STAGES",
    "LUNG_CANCER_UNSEEN_CONDITIONS",
    "LUNG_CANCER_URL",
    "PBMC_GDRIVE_ID",
    "PBMC_IFN_TRAIN_URL",
    "PBMC_IFN_VALID_URL",
    "PBMC_URL",
    "PreparedDataset",
    "TABLE_DATASETS",
    "_get_raw_paths",
    "get_dataset_spec",
    "get_output_paths",
    "list_datasets",
    "resolve_dataset_names",
    "resolve_workers",
]
