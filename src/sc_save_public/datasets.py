"""Dataset download, standardization, preprocessing, and Gene Block Construction
utilities for reproducing Tables 1-5 of the SAVE ICLR 2026 paper.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import threading
import time
from typing import Any, Sequence
import urllib.error
import urllib.request
import zipfile
import zlib

import anndata as ad
import numpy as np
import ot
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm

from sc_save_public.gene_block_construction import (
    create_padded_output,
    load_gene_embeddings,
    sanitize_var_for_h5ad,
)

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

DEFAULT_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": "curl/8.5.0",
    "Accept": "*/*",
}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"
DEFAULT_OT_NUM_ITER_MAX = 1_000_000
_HDF5_IO_LOCK = threading.Lock()


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


# ---------------------------------------------------------------------------
# Registry Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Download & HTTP-Range ZIP64 Utilities
# ---------------------------------------------------------------------------


def _build_request(
    url: str,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> urllib.request.Request:
    req_headers = dict(DEFAULT_HTTP_HEADERS)
    if headers:
        req_headers.update(headers)
    return urllib.request.Request(url, headers=req_headers, method=method)


def _http_range_get(
    url: str,
    byte_range: str,
    timeout: int = 120,
    max_retries: int = 4,
) -> tuple[bytes, str | None, int]:
    """Execute a single HTTP Range GET request returning (body, Content-Range, status)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = _build_request(url, headers={"Range": f"bytes={byte_range}"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                content_range = resp.headers.get("Content-Range")
                data = resp.read()
                if status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {status} for {url}")
                return data, content_range, status
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last_exc = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404):
                raise
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Range request failed for {url} ({byte_range}): {last_exc}")


def _parse_total_size_from_content_range(content_range: str | None) -> int | None:
    if not content_range or "/" not in content_range:
        return None
    total_str = content_range.split("/")[-1].strip()
    if total_str == "*":
        return None
    return int(total_str)


@dataclass(frozen=True)
class _ZipEntryMeta:
    name: str
    compress_method: int
    crc32: int
    comp_size: int
    uncomp_size: int
    local_header_offset: int


def _parse_zip64_extra(
    extra: bytes,
    need_uncomp: bool,
    need_comp: bool,
    need_offset: bool,
) -> tuple[int | None, int | None, int | None]:
    pos = 0
    uncomp_val = comp_val = offset_val = None
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        payload = extra[pos : pos + data_size]
        pos += data_size
        if header_id != 0x0001:
            continue
        idx = 0
        if need_uncomp and idx + 8 <= len(payload):
            (uncomp_val,) = struct.unpack_from("<Q", payload, idx)
            idx += 8
        if need_comp and idx + 8 <= len(payload):
            (comp_val,) = struct.unpack_from("<Q", payload, idx)
            idx += 8
        if need_offset and idx + 8 <= len(payload):
            (offset_val,) = struct.unpack_from("<Q", payload, idx)
        break
    return uncomp_val, comp_val, offset_val


def _read_remote_zip_directory(
    url: str,
    timeout: int = 120,
) -> dict[str, _ZipEntryMeta]:
    """Fetch only the EOCD + Central Directory of a remote ZIP/ZIP64 archive via HTTP Range."""
    tail_size = 65560
    tail_bytes, content_range, status = _http_range_get(
        url, f"-{tail_size}", timeout=timeout
    )
    eocd_pos = tail_bytes.rfind(b"PK\x05\x06")
    if eocd_pos == -1 or eocd_pos + 22 > len(tail_bytes):
        raise RuntimeError(f"Could not find ZIP EOCD signature in tail of {url}")

    total_archive_size = _parse_total_size_from_content_range(content_range)
    if total_archive_size is None:
        if status == 200:
            total_archive_size = len(tail_bytes)
        else:
            raise RuntimeError(
                f"Server did not provide Content-Range total size for {url}: {content_range}"
            )
    tail_start = max(0, total_archive_size - len(tail_bytes))

    _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack_from(
        "<4sHHHHIIH", tail_bytes, eocd_pos
    )
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        loc_pos = tail_bytes.rfind(b"PK\x06\x07", 0, eocd_pos)
        if loc_pos == -1 or loc_pos + 20 > len(tail_bytes):
            raise RuntimeError(f"ZIP64 marker set, but ZIP64 EOCD locator missing for {url}")
        _, _, zip64_eocd_offset, _ = struct.unpack_from("<4sIQH", tail_bytes, loc_pos)
        if zip64_eocd_offset >= tail_start:
            rel = zip64_eocd_offset - tail_start
            z64 = tail_bytes[rel : rel + 56]
        else:
            z64, _, _ = _http_range_get(
                url,
                f"{zip64_eocd_offset}-{zip64_eocd_offset + 55}",
                timeout=timeout,
            )
        if not z64.startswith(b"PK\x06\x06") or len(z64) < 56:
            raise RuntimeError(f"Invalid ZIP64 EOCD header in {url}")
        _, _, _, _, _, _, _, _, cd_size, cd_offset = struct.unpack_from(
            "<4sQHHIIQQQQ", z64, 0
        )

    if cd_offset >= tail_start and cd_offset + cd_size <= tail_start + len(tail_bytes):
        rel = cd_offset - tail_start
        cd_bytes = tail_bytes[rel : rel + cd_size]
    else:
        cd_bytes, _, _ = _http_range_get(
            url, f"{cd_offset}-{cd_offset + cd_size - 1}", timeout=timeout
        )

    entries: dict[str, _ZipEntryMeta] = {}
    pos = 0
    while pos + 46 <= len(cd_bytes):
        if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
            break
        (
            _,
            _,
            _,
            _,
            method,
            _,
            _,
            crc32,
            comp_size,
            uncomp_size,
            fname_len,
            extra_len,
            comment_len,
            _,
            _,
            _,
            local_offset,
        ) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd_bytes, pos)
        pos += 46
        name = cd_bytes[pos : pos + fname_len].decode("utf-8", errors="replace")
        pos += fname_len
        extra = cd_bytes[pos : pos + extra_len]
        pos += extra_len + comment_len

        need_u = uncomp_size == 0xFFFFFFFF
        need_c = comp_size == 0xFFFFFFFF
        need_o = local_offset == 0xFFFFFFFF
        if need_u or need_c or need_o:
            u64, c64, o64 = _parse_zip64_extra(extra, need_u, need_c, need_o)
            if u64 is not None:
                uncomp_size = u64
            if c64 is not None:
                comp_size = c64
            if o64 is not None:
                local_offset = o64

        entries[name] = _ZipEntryMeta(
            name=name,
            compress_method=method,
            crc32=crc32,
            comp_size=comp_size,
            uncomp_size=uncomp_size,
            local_header_offset=local_offset,
        )
    return entries


def _stream_remote_zip_entry(
    url: str,
    meta: _ZipEntryMeta,
    dest: Path,
    timeout: int = 120,
    chunk_size: int = 4 * 1024 * 1024,
) -> None:
    """Stream one compressed entry from a remote ZIP/ZIP64 archive to dest atomically."""
    lh_bytes, _, _ = _http_range_get(
        url,
        f"{meta.local_header_offset}-{meta.local_header_offset + 29}",
        timeout=timeout,
    )
    if len(lh_bytes) < 30 or lh_bytes[:4] != ZIP_MAGIC:
        raise RuntimeError(f"Invalid local ZIP header for {meta.name} in {url}")
    _, _, _, _, _, _, _, _, _, lf_fname_len, lf_extra_len = struct.unpack_from(
        "<4sHHHHHIIIHH", lh_bytes, 0
    )
    data_start = meta.local_header_offset + 30 + lf_fname_len + lf_extra_len
    if meta.comp_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"")
        return
    data_end = data_start + meta.comp_size - 1

    if meta.compress_method not in (0, 8):
        raise RuntimeError(
            f"Unsupported ZIP compression method {meta.compress_method} for {meta.name}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    decomp = zlib.decompressobj(-zlib.MAX_WBITS) if meta.compress_method == 8 else None
    crc = 0
    written = 0

    req = _build_request(url, headers={"Range": f"bytes={data_start}-{data_end}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(
            tmp_path, "wb"
        ) as out_f, tqdm(
            total=meta.comp_size,
            unit="B",
            unit_scale=True,
            desc=f"Extracting {dest.name}",
            leave=True,
        ) as pbar:
            status = getattr(resp, "status", 200)
            if status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP status {status} streaming {meta.name}")
            # If server ignored Range and returned 200, skip ahead manually once.
            if status == 200 and data_start > 0:
                remaining_skip = data_start
                while remaining_skip > 0:
                    discard = resp.read(min(chunk_size, remaining_skip))
                    if not discard:
                        raise RuntimeError(f"Unexpected EOF skipping to {data_start}")
                    remaining_skip -= len(discard)

            remaining = meta.comp_size
            while remaining > 0:
                chunk = resp.read(min(chunk_size, remaining))
                if not chunk:
                    raise RuntimeError(
                        f"Truncated stream for {meta.name}: {remaining} bytes left"
                    )
                remaining -= len(chunk)
                pbar.update(len(chunk))
                out_chunk = decomp.decompress(chunk) if decomp is not None else chunk
                if out_chunk:
                    out_f.write(out_chunk)
                    crc = zlib.crc32(out_chunk, crc)
                    written += len(out_chunk)

            if decomp is not None:
                tail = decomp.flush()
                if tail:
                    out_f.write(tail)
                    crc = zlib.crc32(tail, crc)
                    written += len(tail)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if written != meta.uncomp_size:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Size mismatch for {meta.name}: expected {meta.uncomp_size}, got {written}"
        )
    if (crc & 0xFFFFFFFF) != (meta.crc32 & 0xFFFFFFFF):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"CRC32 mismatch for {meta.name}: expected {meta.crc32:#x}, got {crc:#x}"
        )
    os.replace(tmp_path, dest)


def extract_zip_entries_via_http_range(
    url: str,
    member_to_dest: dict[str, Path],
    backup_urls: Sequence[str] = (),
    force: bool = False,
    timeout: int = 120,
    max_workers: int = 1,
) -> dict[str, Path]:
    """Download only requested entries from a remote ZIP/ZIP64 archive via HTTP Range."""
    pending = {
        member: dest
        for member, dest in member_to_dest.items()
        if force or not (dest.exists() and dest.stat().st_size > 0)
    }
    if not pending:
        return dict(member_to_dest)

    last_err: Exception | None = None
    for candidate_url in (url, *backup_urls):
        try:
            entries = _read_remote_zip_directory(candidate_url, timeout=timeout)
            missing = [m for m in pending if m not in entries]
            if missing:
                raise KeyError(
                    f"Archive entries not found in {candidate_url}: {missing}"
                )
            n_workers = max(1, min(int(max_workers), len(pending)))
            if n_workers == 1 or len(pending) == 1:
                for member, dest in pending.items():
                    _stream_remote_zip_entry(
                        candidate_url, entries[member], dest, timeout=timeout
                    )
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = [
                        pool.submit(
                            _stream_remote_zip_entry,
                            candidate_url,
                            entries[member],
                            dest,
                            timeout=timeout,
                        )
                        for member, dest in pending.items()
                    ]
                    for fut in as_completed(futures):
                        fut.result()
            return dict(member_to_dest)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        f"Failed to extract ZIP entries {list(pending.keys())} from {url}: {last_err}"
    )


def _download_file(
    url: str,
    dest: Path,
    *,
    backup_urls: Sequence[str] = (),
    expected_magic: bytes | None = None,
    force: bool = False,
    timeout: int = 120,
    max_retries: int = 4,
    chunk_size: int = 4 * 1024 * 1024,
) -> Path:
    """Download a file atomically with HTTP Range resume support and header checks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and dest.exists() and dest.stat().st_size > 0:
        if expected_magic is None:
            return dest
        with open(dest, "rb") as f:
            if f.read(len(expected_magic)) == expected_magic:
                return dest

    part_path = dest.with_suffix(dest.suffix + ".part")
    if force and part_path.exists():
        part_path.unlink()

    last_err: Exception | None = None
    for candidate_url in (url, *backup_urls):
        for attempt in range(max_retries):
            try:
                existing_bytes = part_path.stat().st_size if part_path.exists() else 0
                headers = (
                    {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}
                )
                req = _build_request(candidate_url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status = getattr(resp, "status", 200)
                    resuming = status == 206 and existing_bytes > 0
                    mode = "ab" if resuming else "wb"
                    content_len = resp.headers.get("Content-Length")
                    total_bytes = (
                        existing_bytes + int(content_len)
                        if resuming and content_len is not None
                        else (int(content_len) if content_len is not None else None)
                    )
                    initial = existing_bytes if resuming else 0
                    with open(part_path, mode) as out_f, tqdm(
                        total=total_bytes,
                        initial=initial,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {dest.name}",
                        leave=True,
                    ) as pbar:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            out_f.write(chunk)
                            pbar.update(len(chunk))

                if expected_magic is not None:
                    with open(part_path, "rb") as f:
                        head = f.read(len(expected_magic))
                    if head != expected_magic:
                        part_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Downloaded {dest.name} from {candidate_url} failed magic check "
                            f"(expected {expected_magic!r}, got {head!r})."
                        )
                os.replace(part_path, dest)
                return dest
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, RuntimeError) as exc:
                last_err = exc
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {dest.name} from {url}: {last_err}")


def _download_gdrive_file(
    file_id: str,
    dest: Path,
    *,
    expected_magic: bytes | None = HDF5_MAGIC,
    force: bool = False,
) -> Path:
    """Download a large Google Drive file via gdown and verify its magic bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and dest.exists() and dest.stat().st_size > 0:
        if expected_magic is None:
            return dest
        with open(dest, "rb") as f:
            if f.read(len(expected_magic)) == expected_magic:
                return dest

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download Google Drive datasets (PBMC). "
            "Install it with `pip install gdown` or pass an already downloaded "
            f"raw file at {dest}."
        ) from exc

    part_path = dest.with_suffix(dest.suffix + ".part")
    if force and part_path.exists():
        part_path.unlink()

    result = gdown.download(
        id=file_id,
        output=str(part_path),
        quiet=False,
        resume=not force,
    )
    if result is None or not part_path.exists() or part_path.stat().st_size == 0:
        raise RuntimeError(
            f"gdown failed to download Google Drive ID '{file_id}' to {dest}."
        )
    if expected_magic is not None:
        with open(part_path, "rb") as f:
            head = f.read(len(expected_magic))
        if head != expected_magic:
            part_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded Google Drive file '{file_id}' is not a valid HDF5/.h5ad "
                f"(got header {head!r}); Drive quota or link sharing may be blocking it."
            )
    os.replace(part_path, dest)
    return dest


def _select_genept_zip_member(zf: zipfile.ZipFile, zip_dest: Path) -> str:
    """Locate the GenePT v2 model-3 text embedding pickle inside the downloaded ZIP."""
    candidates = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and "/__MACOSX/" not in info.filename
        and PurePosixPath(info.filename).name.rstrip(".") == GENEPT_PICKLE_NAME
    ]
    if not candidates:
        raise KeyError(
            f"Could not find {GENEPT_PICKLE_NAME} inside {zip_dest}; "
            f"entries preview: {[info.filename for info in zf.infolist()[:20]]}"
        )
    exact = [
        member
        for member in candidates
        if PurePosixPath(member).name == GENEPT_PICKLE_NAME
    ]
    return exact[0] if exact else candidates[0]


def ensure_genept_embeddings(
    data_root: str | Path,
    gene_embedding_path: str | Path | None = None,
    force: bool = False,
    skip_download: bool = False,
) -> Path:
    """Ensure GenePT_gene_protein_embedding_model_3_text.pickle exists and return its path."""
    if gene_embedding_path is not None:
        explicit = Path(gene_embedding_path)
        if not explicit.exists():
            raise FileNotFoundError(
                f"Specified gene_embedding_path does not exist: {explicit}"
            )
        return explicit

    root = Path(data_root)
    pickle_dest = root / "genept" / GENEPT_PICKLE_NAME
    if not force and pickle_dest.exists() and pickle_dest.stat().st_size > 0:
        return pickle_dest

    if skip_download:
        raise FileNotFoundError(
            f"Missing GenePT embeddings with skip_download=True: {pickle_dest}"
        )

    zip_dest = root / "raw" / "genept" / "GenePT_emebdding_v2.zip"
    _download_file(
        GENEPT_ZENODO_URL,
        zip_dest,
        backup_urls=(GENEPT_ZENODO_BACKUP_URL,),
        expected_magic=ZIP_MAGIC,
        force=force,
    )

    pickle_dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_pickle = pickle_dest.with_suffix(".pickle.part")
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            selected_member = _select_genept_zip_member(zf, zip_dest)
            with zf.open(selected_member) as src_f, open(tmp_pickle, "wb") as dst_f:
                while True:
                    buf = src_f.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    dst_f.write(buf)
        os.replace(tmp_pickle, pickle_dest)
    except Exception:
        tmp_pickle.unlink(missing_ok=True)
        raise
    return pickle_dest


def _ensure_raw_files(
    spec: DatasetSpec,
    data_root: Path,
    *,
    skip_download: bool = False,
    force: bool = False,
) -> list[Path]:
    raw_paths = _get_raw_paths(spec, data_root)
    if skip_download:
        missing = [p for p in raw_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing raw files with --skip-download for '{spec.name}': {missing}"
            )
        return raw_paths

    if spec.download_kind == "cfgen_zip64":
        mapping = dict(zip(spec.archive_members, raw_paths))
        extract_zip_entries_via_http_range(
            spec.source_urls[0],
            mapping,
            backup_urls=spec.backup_urls,
            force=force,
        )
    elif spec.download_kind == "direct":
        _download_file(
            spec.source_urls[0],
            raw_paths[0],
            backup_urls=spec.backup_urls,
            expected_magic=HDF5_MAGIC,
            force=force,
        )
    elif spec.download_kind == "multi_direct":
        for url, dest in zip(spec.source_urls, raw_paths):
            _download_file(
                url,
                dest,
                expected_magic=HDF5_MAGIC,
                force=force,
            )
    elif spec.download_kind == "gdrive":
        assert spec.gdrive_id is not None
        _download_gdrive_file(
            spec.gdrive_id,
            raw_paths[0],
            expected_magic=HDF5_MAGIC,
            force=force,
        )
    else:
        raise ValueError(f"Unsupported download_kind: {spec.download_kind}")
    return raw_paths


# ---------------------------------------------------------------------------
# Preprocessing, Normalization, HVG Selection & OT Gene Block Construction
# ---------------------------------------------------------------------------


def _extract_matrix_values(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.data)
    return np.asarray(X).ravel()


def _looks_like_raw_counts(adata: ad.AnnData) -> bool:
    if "log1p" in adata.uns:
        return False
    vals = _extract_matrix_values(adata.X)
    if vals.size == 0:
        return False
    sample = vals[: min(vals.size, 50_000)]
    if np.any(sample < -1e-6):
        return False
    if np.allclose(sample, np.round(sample), atol=1e-6):
        return True
    return float(sample.max()) > 50.0


def ensure_log1p_normalized(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    *,
    preserve_counts_layer: bool = True,
    normalize_total: bool = True,
) -> ad.AnnData:
    """Apply log1p (and optionally normalize_total) if raw/unlogged; no-op if already log1p."""
    if adata.n_obs == 0 or adata.n_vars == 0 or "log1p" in adata.uns:
        return adata

    x_max = float(adata.X.max())
    is_raw = _looks_like_raw_counts(adata)
    if is_raw:
        if preserve_counts_layer and "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    elif x_max > float(np.log1p(target_sum)) + 1e-4:
        if normalize_total:
            sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    return adata


def _select_hvg_exact(
    adata: ad.AnnData,
    n_top_genes: int,
    *,
    batch_key: str | None = None,
    max_rounds: int = 8,
) -> ad.AnnData:
    """Select exact n_top_genes HVGs with tie-breaking fallback across Scanpy rounds."""
    if adata.n_vars <= n_top_genes:
        return adata.copy()

    cur = adata.copy()
    if sparse.issparse(cur.X) and not sparse.isspmatrix_csr(cur.X):
        cur.X = cur.X.tocsr()

    active_batch_key = batch_key
    if active_batch_key is not None and active_batch_key in cur.obs.columns:
        counts = cur.obs[active_batch_key].value_counts()
        if counts.empty or (counts < 2).any():
            active_batch_key = None
    else:
        active_batch_key = None

    for _ in range(max_rounds):
        if cur.n_vars == n_top_genes:
            return cur
        kwargs: dict[str, Any] = {
            "n_top_genes": n_top_genes,
            "flavor": "seurat",
            "inplace": False,
        }
        if active_batch_key is not None:
            kwargs["batch_key"] = active_batch_key
        try:
            hvg_df = sc.pp.highly_variable_genes(cur, **kwargs)
        except Exception:
            if active_batch_key is not None:
                kwargs.pop("batch_key", None)
                active_batch_key = None
                hvg_df = sc.pp.highly_variable_genes(cur, **kwargs)
            else:
                raise
        selected_mask = np.asarray(hvg_df["highly_variable"], dtype=bool)
        n_sel = int(selected_mask.sum())
        if n_sel <= n_top_genes and n_sel > 0:
            cur = cur[:, selected_mask].copy()
            if cur.n_vars == n_top_genes:
                return cur
            break
        if n_sel == cur.n_vars:
            score_col = "dispersions_norm" if "dispersions_norm" in hvg_df else "dispersions"
            scores = (
                pd.to_numeric(hvg_df[score_col], errors="coerce")
                .fillna(-np.inf)
                .to_numpy()
            )
            top_idx = np.sort(np.argsort(-scores, kind="mergesort")[:n_top_genes])
            return cur[:, top_idx].copy()
        cur = cur[:, selected_mask].copy()

    if cur.n_vars > n_top_genes:
        hvg_df = sc.pp.highly_variable_genes(
            cur,
            n_top_genes=n_top_genes,
            flavor="seurat",
            inplace=False,
        )
        score_col = "dispersions_norm" if "dispersions_norm" in hvg_df else "dispersions"
        scores = (
            pd.to_numeric(hvg_df[score_col], errors="coerce")
            .fillna(-np.inf)
            .to_numpy()
        )
        top_idx = np.sort(np.argsort(-scores, kind="mergesort")[:n_top_genes])
        cur = cur[:, top_idx].copy()
    return cur


def balanced_kmeans_with_ot_emd(
    X: np.ndarray,
    n_clusters: int,
    *,
    method: str = "emd",
    reg: float = 0.1,
    max_iter: int = 20,
    num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition embeddings into equal-size clusters using OT balanced k-means."""
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    n_samples = X_arr.shape[0]
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    if n_clusters == 1:
        return np.zeros(n_samples, dtype=np.int64), X_arr.mean(axis=0, keepdims=True)

    centers, _ = kmeans_plusplus(X_arr, n_clusters=n_clusters, random_state=random_state)
    centers = np.ascontiguousarray(centers, dtype=np.float64)
    source = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
    target = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)

    labels = np.zeros(n_samples, dtype=np.int64)
    for _ in range(max_iter):
        old_centers = centers.copy()
        cost_matrix = np.ascontiguousarray(
            cdist(X_arr, centers, "sqeuclidean"), dtype=np.float64
        )
        if method == "emd":
            transport_plan = ot.emd(
                source,
                target,
                cost_matrix,
                numItermax=int(num_iter_max),
            )
        elif method == "sinkhorn":
            transport_plan = ot.sinkhorn(source, target, cost_matrix, reg=reg)
        else:
            raise ValueError(f"Unsupported OT method: {method}")

        labels = np.argmax(transport_plan, axis=1).astype(np.int64)
        for j in range(n_clusters):
            members = X_arr[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
        if np.allclose(centers, old_centers):
            break

    return labels, centers


def _preserve_padded_layers(
    src: ad.AnnData,
    processed: ad.AnnData,
    pad_count: int,
) -> None:
    """Reapply src.layers (e.g. 'counts') onto processed with trailing zero pad."""
    if not src.layers:
        return
    real_count = processed.n_vars - pad_count
    real_vars = [str(v) for v in processed.var_names[:real_count]]
    src_aligned = src[:, real_vars]
    for layer_name in list(src.layers.keys()):
        layer_mat = src_aligned.layers[layer_name]
        if pad_count > 0:
            if sparse.issparse(layer_mat):
                pad_mat = sparse.csr_matrix(
                    (processed.n_obs, pad_count), dtype=layer_mat.dtype
                )
                layer_mat = sparse.hstack([layer_mat.tocsr(), pad_mat], format="csr")
            else:
                pad_mat = np.zeros((processed.n_obs, pad_count), dtype=layer_mat.dtype)
                layer_mat = np.hstack([layer_mat, pad_mat])
        processed.layers[layer_name] = layer_mat


def apply_gene_block_construction(
    adata: ad.AnnData,
    gene_embeddings: dict[str, np.ndarray],
    gene_block_size: int,
    *,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    random_state: int = 42,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Apply OT semantic Gene Block Construction with mouse/small-overlap padding fallback."""
    shared_genes = [g for g in adata.var_names if g in gene_embeddings]
    n_shared = len(shared_genes)

    if n_shared < gene_block_size:
        # Matches iclr-ver ot_split_CFGen_data.ipynb cell 6 for mouse datasets
        # (Dentate gyrus / Tabula Muris) whose GenePT symbol overlap < 3200.
        remaining = [g for g in adata.var_names if g not in set(shared_genes)]
        reordered = adata[:, shared_genes + remaining].copy()
        processed, _, pad_names = create_padded_output(
            adata=reordered,
            sorted_genes=[],
            block_id_map={},
            shared_genes=shared_genes,
            gene_block_size=gene_block_size,
        )
        processed.var["is_genept_covered"] = processed.var_names.isin(set(shared_genes))
        processed.var = sanitize_var_for_h5ad(processed.var)
        processed.uns = dict(adata.uns)
        _preserve_padded_layers(reordered, processed, len(pad_names))
        return processed, {
            "n_shared_genept": n_shared,
            "n_semantic_blocks": 0,
            "n_clustered_genes": 0,
            "n_pad": len(pad_names),
        }

    n_blocks = n_shared // gene_block_size
    target_clustered = n_blocks * gene_block_size
    adata_shared = adata[:, shared_genes].copy()
    if adata_shared.n_vars != target_clustered:
        adata_shared = _select_hvg_exact(adata_shared, n_top_genes=target_clustered)

    emb_matrix = np.asarray(
        [gene_embeddings[g] for g in adata_shared.var_names], dtype=np.float64
    )
    labels, _ = balanced_kmeans_with_ot_emd(
        emb_matrix,
        n_clusters=n_blocks,
        method=ot_method,
        reg=ot_reg,
        max_iter=ot_max_iter,
        num_iter_max=ot_num_iter_max,
        random_state=random_state,
    )
    sorted_idx = np.argsort(labels, kind="stable")
    sorted_genes = [str(adata_shared.var_names[i]) for i in sorted_idx]
    block_id_map = {
        str(adata_shared.var_names[i]): int(labels[i]) for i in sorted_idx
    }

    processed, _, pad_names = create_padded_output(
        adata=adata,
        sorted_genes=sorted_genes,
        block_id_map=block_id_map,
        shared_genes=shared_genes,
        gene_block_size=gene_block_size,
    )
    processed.var = sanitize_var_for_h5ad(processed.var)
    processed.uns = dict(adata.uns)
    _preserve_padded_layers(adata, processed, len(pad_names))
    return processed, {
        "n_shared_genept": n_shared,
        "n_semantic_blocks": n_blocks,
        "n_clustered_genes": len(sorted_genes),
        "n_pad": len(pad_names),
    }


def align_test_to_train_genes(
    adata_test: ad.AnnData,
    train_adata: ad.AnnData,
) -> ad.AnnData:
    """Align a held-out test split to the exact ordered gene + pad_* columns of train."""
    target_vars = [str(v) for v in train_adata.var_names]
    pad_set = {v for v in target_vars if v.startswith("pad_")}
    real_vars = [v for v in target_vars if v not in pad_set]

    test_var_set = set(map(str, adata_test.var_names))
    if all(v in test_var_set for v in real_vars):
        reordered = adata_test[:, real_vars].copy()
    else:
        present = [v for v in real_vars if v in test_var_set]
        col_map = {v: i for i, v in enumerate(present)}
        src_X = adata_test[:, present].X
        if sparse.issparse(src_X):
            src_csc = src_X.tocsc()
            new_X = sparse.lil_matrix((adata_test.n_obs, len(real_vars)), dtype=src_X.dtype)
            for out_idx, g in enumerate(real_vars):
                if g in col_map:
                    new_X[:, out_idx] = src_csc[:, col_map[g]]
            reordered_X: Any = new_X.tocsr()
        else:
            reordered_X = np.zeros((adata_test.n_obs, len(real_vars)), dtype=src_X.dtype)
            for out_idx, g in enumerate(real_vars):
                if g in col_map:
                    reordered_X[:, out_idx] = src_X[:, col_map[g]]
        reordered = ad.AnnData(
            X=reordered_X,
            obs=adata_test.obs.copy(),
            var=pd.DataFrame(index=real_vars),
        )

    pad_names = [v for v in target_vars if v in pad_set]
    if pad_names:
        if sparse.issparse(reordered.X):
            pad_mat: Any = sparse.csr_matrix(
                (reordered.n_obs, len(pad_names)), dtype=reordered.X.dtype
            )
        else:
            pad_mat = np.zeros((reordered.n_obs, len(pad_names)), dtype=reordered.X.dtype)
        pad_adata = ad.AnnData(
            X=pad_mat,
            obs=reordered.obs.copy(),
            var=pd.DataFrame(index=pad_names),
        )
        aligned = ad.concat([reordered, pad_adata], axis=1)
        aligned.obs = reordered.obs.copy()
    else:
        aligned = reordered

    aligned = aligned[:, target_vars].copy()
    aligned.var = sanitize_var_for_h5ad(train_adata.var.copy())
    aligned.uns = dict(adata_test.uns)
    return aligned


# ---------------------------------------------------------------------------
# Dataset-Specific Standardization & Table Pipelines
# ---------------------------------------------------------------------------


def _standardize_obs_column(
    adata: ad.AnnData,
    target_col: str,
    candidate_sources: Sequence[str],
    *,
    force_overwrite: bool = False,
) -> None:
    if force_overwrite or target_col not in adata.obs.columns:
        chosen = next((c for c in candidate_sources if c in adata.obs.columns), None)
        if chosen is not None and chosen != target_col:
            adata.obs[target_col] = adata.obs[chosen]
    if target_col not in adata.obs.columns:
        raise KeyError(
            f"Missing required obs column '{target_col}' (checked {candidate_sources}). "
            f"Available columns: {list(adata.obs.columns)}"
        )
    series = adata.obs[target_col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        vals = series.astype(object).fillna("unknown").astype(str)
    else:
        vals = series.fillna("unknown").astype(str)
    adata.obs[target_col] = pd.Categorical(vals)


def _standardize_dataset_obs(spec: DatasetSpec, adata: ad.AnnData) -> None:
    if spec.name == "pbmc3k":
        _standardize_obs_column(
            adata, "cell_type", ("cell_type", "celltype", "clusters")
        )
    elif spec.name == "dentategyrus":
        _standardize_obs_column(
            adata,
            "cell_type",
            ("clusters", "cell_type", "celltype"),
            force_overwrite="clusters" in adata.obs.columns,
        )
    elif spec.name == "tabula_muris":
        _standardize_obs_column(
            adata,
            "cell_type",
            ("tissue", "cell_type", "cell_ontology_class"),
            force_overwrite="tissue" in adata.obs.columns,
        )
    elif spec.name in ("heart", "heart_bec"):
        _standardize_obs_column(adata, "batch", ("donor", "batch"))
        _standardize_obs_column(adata, "cell_type", ("cell_type", "celltype"))
    elif spec.name in ("pbmc", "pbmc_bec"):
        _standardize_obs_column(adata, "batch", ("batch", "donor"))
        _standardize_obs_column(
            adata,
            "cell_type",
            ("final_annotation", "cell_type", "celltype"),
            force_overwrite="final_annotation" in adata.obs.columns,
        )
    elif spec.name in ("lung_atlas", "lung_atlas_bec"):
        _standardize_obs_column(adata, "batch", ("batch", "donor", "sample"))
        _standardize_obs_column(
            adata, "cell_type", ("cell_type", "celltype", "final_annotation")
        )
    elif spec.name == "lung_cancer":
        for col in spec.cond_cols:
            _standardize_obs_column(adata, col, (col,))
    elif spec.name == "pbmc_ifn":
        _standardize_obs_column(adata, "cell_type", ("cell_type", "celltype"))
        _standardize_obs_column(adata, "condition", ("condition", "stim"))
        cond_map = {"ctrl": "control", "stim": "stimulated"}
        normalized = [
            cond_map.get(str(v).lower(), str(v)) for v in adata.obs["condition"]
        ]
        adata.obs["condition"] = pd.Categorical(normalized)
        adata.obs["batch"] = pd.Categorical(normalized)


def _subsample_adata(
    adata: ad.AnnData,
    max_cells: int | None,
    *,
    stratify_cols: Sequence[str] = (),
    seed: int = 42,
) -> ad.AnnData:
    if max_cells is None or adata.n_obs <= max_cells:
        return adata

    rng = np.random.default_rng(seed)
    cols = [c for c in stratify_cols if c in adata.obs.columns]
    if not cols:
        idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
        return adata[idx].copy()

    keys = adata.obs[cols[0]].astype(str)
    for col in cols[1:]:
        keys = keys + "||" + adata.obs[col].astype(str)
    grouped: dict[str, list[int]] = {}
    for pos, key in enumerate(keys.tolist()):
        grouped.setdefault(key, []).append(pos)

    picked: list[int] = []
    if len(grouped) <= max_cells:
        for indices in grouped.values():
            picked.append(int(indices[int(rng.integers(0, len(indices)))]))
    picked_set = set(picked)
    remaining = [i for i in range(adata.n_obs) if i not in picked_set]
    need = max_cells - len(picked)
    if need > 0 and remaining:
        extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        picked.extend(map(int, extra.tolist()))
    return adata[sorted(picked)].copy()


def _read_h5ad_safe(path: Path) -> ad.AnnData:
    with _HDF5_IO_LOCK:
        return sc.read_h5ad(path)


def _write_h5ad_atomic(adata: ad.AnnData, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    adata.var = sanitize_var_for_h5ad(adata.var)
    tmp_path = dest.with_suffix(".h5ad.part")
    try:
        with _HDF5_IO_LOCK:
            adata.write_h5ad(tmp_path)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _make_prepared_record(
    spec: DatasetSpec,
    out_paths: dict[str, Path],
    primary_adata: ad.AnnData,
    block_stats: dict[str, Any],
) -> PreparedDataset:
    return PreparedDataset(
        name=spec.name,
        table=spec.table,
        task_type=spec.task_type,
        cond_cols=list(spec.cond_cols),
        gene_block_size=spec.gene_block_size,
        hvg_top_genes=spec.hvg_top_genes,
        primary_path=out_paths[spec.primary_key],
        files=out_paths,
        n_obs=int(primary_adata.n_obs),
        n_vars=int(primary_adata.n_vars),
        n_shared_genept=int(block_stats["n_shared_genept"]),
        n_semantic_blocks=int(block_stats["n_semantic_blocks"]),
        n_pad=int(block_stats["n_pad"]),
        source_urls=list(spec.source_urls),
    )


def _prepare_table1_cfgen(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    train_raw = _read_h5ad_safe(raw_paths[0])
    test_raw = _read_h5ad_safe(raw_paths[1])
    train_raw.var_names_make_unique()
    test_raw.var_names_make_unique()

    _standardize_dataset_obs(spec, train_raw)
    _standardize_dataset_obs(spec, test_raw)

    train_raw = _subsample_adata(
        train_raw, subset_cells, stratify_cols=spec.cond_cols, seed=seed
    )
    if subset_cells is not None:
        test_raw = _subsample_adata(
            test_raw, max(1, subset_cells // 4), stratify_cols=spec.cond_cols, seed=seed
        )

    ensure_log1p_normalized(train_raw, preserve_counts_layer=False)
    ensure_log1p_normalized(test_raw, preserve_counts_layer=False)

    train_proc, stats = apply_gene_block_construction(
        train_raw,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    test_proc = align_test_to_train_genes(test_raw, train_proc)
    _write_h5ad_atomic(train_proc, out_paths["train"])
    _write_h5ad_atomic(test_proc, out_paths["test"])
    return train_proc, stats


def _prepare_table2_dual(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    adata = _read_h5ad_safe(raw_paths[0])
    adata.var_names_make_unique()
    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(adata, subset_cells, stratify_cols=spec.cond_cols, seed=seed)
    # Match iclr-ver ot_split_with_pad.ipynb & gene_block_construction.py: log1p only
    # without normalize_total(1e4) for Table 2 dual-condition generation.
    ensure_log1p_normalized(adata, preserve_counts_layer=True, normalize_total=False)

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    _write_h5ad_atomic(processed, out_paths["full"])
    return processed, stats


def _prepare_table3_lung_cancer(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    min_genes: int | None,
    min_cells: int | None,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    # Open huge 17 GB CellxGene atlas backed="r" and subset ages 55/60/65/70 before .to_memory()
    with _HDF5_IO_LOCK:
        backed = ad.read_h5ad(raw_paths[0], backed="r")
        try:
            if "development_stage" not in backed.obs.columns:
                raise KeyError("Expected 'development_stage' in lung_cancer obs columns.")
            age_mask = backed.obs["development_stage"].astype(str).isin(LUNG_CANCER_AGE_STAGES)
            adata = backed[age_mask, :].to_memory()
        finally:
            if getattr(backed, "file", None) is not None:
                backed.file.close()

    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(
        adata,
        subset_cells,
        stratify_cols=("assay", "development_stage", "disease", "uicc_stage"),
        seed=seed,
    )

    eff_min_genes = spec.min_genes if min_genes is None else min_genes
    eff_min_cells = spec.min_cells if min_cells is None else min_cells
    if eff_min_genes:
        sc.pp.filter_cells(adata, min_genes=eff_min_genes)
    if eff_min_cells:
        sc.pp.filter_genes(adata, min_cells=eff_min_cells)

    ensure_log1p_normalized(adata, preserve_counts_layer=False)
    hvg_target = spec.hvg_top_genes or 2000
    adata = _select_hvg_exact(adata, n_top_genes=hvg_target)

    if "feature_name" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var_names.astype(str)
        adata.var_names = pd.Index(
            adata.var["feature_name"].astype(str).tolist(), dtype="object"
        )
    adata.var_names_make_unique()

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )

    test_mask = np.zeros(processed.n_obs, dtype=bool)
    for cond in LUNG_CANCER_UNSEEN_CONDITIONS:
        cond_mask = np.ones(processed.n_obs, dtype=bool)
        for col, val in cond.items():
            cond_mask &= processed.obs[col].astype(str).to_numpy() == val
        test_mask |= cond_mask

    processed.obs["split"] = pd.Categorical(
        np.where(test_mask, "unseen_test", "seen_train")
    )
    adata_train = processed[~test_mask].copy()
    adata_test = processed[test_mask].copy()

    _write_h5ad_atomic(processed, out_paths["full"])
    _write_h5ad_atomic(adata_train, out_paths["train"])
    _write_h5ad_atomic(adata_test, out_paths["test"])
    return processed, stats


def _prepare_table4_bec(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    min_genes: int | None,
    min_cells: int | None,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    adata = _read_h5ad_safe(raw_paths[0])
    adata.var_names_make_unique()
    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(adata, subset_cells, stratify_cols=spec.cond_cols, seed=seed)

    if spec.exclude_gene_prefixes:
        keep_genes = [
            g
            for g in adata.var_names
            if not str(g).startswith(spec.exclude_gene_prefixes)
        ]
        adata = adata[:, keep_genes].copy()

    eff_min_genes = spec.min_genes if min_genes is None else min_genes
    eff_min_cells = spec.min_cells if min_cells is None else min_cells
    if eff_min_genes:
        sc.pp.filter_cells(adata, min_genes=eff_min_genes)
    if eff_min_cells:
        sc.pp.filter_genes(adata, min_cells=eff_min_cells)

    ensure_log1p_normalized(adata, preserve_counts_layer=True)
    hvg_target = spec.hvg_top_genes or 2000
    adata = _select_hvg_exact(adata, n_top_genes=hvg_target, batch_key="batch")

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    _write_h5ad_atomic(processed, out_paths["full"])
    return processed, stats


def _prepare_table5_pbmc_ifn(
    spec: DatasetSpec,
    raw_paths: list[Path],
    out_paths: dict[str, Path],
    gene_embeddings: dict[str, np.ndarray],
    *,
    subset_cells: int | None,
    ot_method: str,
    ot_reg: float,
    ot_max_iter: int,
    ot_num_iter_max: int,
    seed: int,
) -> tuple[ad.AnnData, dict[str, Any]]:
    train_adata = _read_h5ad_safe(raw_paths[0])
    valid_adata = _read_h5ad_safe(raw_paths[1])
    train_adata.var_names_make_unique()
    valid_adata.var_names_make_unique()
    if not set(train_adata.obs_names).isdisjoint(set(valid_adata.obs_names)):
        train_adata.obs_names = pd.Index(
            [f"train_{x}" for x in train_adata.obs_names], dtype="object"
        )
        valid_adata.obs_names = pd.Index(
            [f"valid_{x}" for x in valid_adata.obs_names], dtype="object"
        )

    adata = ad.concat([train_adata, valid_adata], join="inner", merge="first")
    adata.obs_names_make_unique()
    _standardize_dataset_obs(spec, adata)
    adata = _subsample_adata(adata, subset_cells, stratify_cols=spec.cond_cols, seed=seed)

    ensure_log1p_normalized(adata, preserve_counts_layer=False)
    hvg_target = spec.hvg_top_genes or 2000
    adata = _select_hvg_exact(adata, n_top_genes=hvg_target, batch_key="batch")

    processed, stats = apply_gene_block_construction(
        adata,
        gene_embeddings,
        spec.gene_block_size,
        ot_method=ot_method,
        ot_reg=ot_reg,
        ot_max_iter=ot_max_iter,
        ot_num_iter_max=ot_num_iter_max,
        random_state=seed,
    )
    _write_h5ad_atomic(processed, out_paths["full"])
    return processed, stats


# ---------------------------------------------------------------------------
# Public Entrypoints
# ---------------------------------------------------------------------------


def _outputs_ready(out_paths: dict[str, Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in out_paths.values())


def _inspect_existing_dataset(
    spec: DatasetSpec,
    out_paths: dict[str, Path],
) -> PreparedDataset:
    primary = _read_h5ad_safe(out_paths[spec.primary_key])
    var_df = primary.var
    n_shared = (
        int(var_df["is_genept_covered"].astype(bool).sum())
        if "is_genept_covered" in var_df.columns
        else int(primary.n_vars)
    )
    n_pad = (
        int(var_df["is_pad"].astype(bool).sum())
        if "is_pad" in var_df.columns
        else int(sum(str(v).startswith("pad_") for v in primary.var_names))
    )
    if "gene_block_id" in var_df.columns:
        valid_blocks = var_df.loc[var_df["gene_block_id"] >= 0, "gene_block_id"]
        n_blocks = int(valid_blocks.nunique())
    else:
        n_blocks = 0
    return _make_prepared_record(
        spec,
        out_paths,
        primary,
        {
            "n_shared_genept": n_shared,
            "n_semantic_blocks": n_blocks,
            "n_pad": n_pad,
        },
    )


def prepare_dataset(
    name: str,
    data_root: str | Path = "./data",
    *,
    gene_embedding_path: str | Path | None = None,
    gene_embeddings: dict[str, np.ndarray] | None = None,
    skip_download: bool = False,
    force: bool = False,
    subset_cells: int | None = None,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    min_genes: int | None = None,
    min_cells: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    """Download, standardize, normalize, and OT-partition one paper benchmark dataset."""
    spec = get_dataset_spec(name)
    root = Path(data_root)
    out_paths = get_output_paths(spec.name, root)

    if not force and subset_cells is None and _outputs_ready(out_paths):
        return _inspect_existing_dataset(spec, out_paths)

    if gene_embeddings is None:
        emb_path = ensure_genept_embeddings(
            root,
            gene_embedding_path=gene_embedding_path,
            force=force,
            skip_download=skip_download,
        )
        gene_embeddings = load_gene_embeddings(str(emb_path))

    raw_paths = _ensure_raw_files(
        spec, root, skip_download=skip_download, force=force
    )

    if spec.task_type == "single_cond":
        primary_adata, stats = _prepare_table1_cfgen(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            seed=seed,
        )
    elif spec.task_type == "dual_cond":
        primary_adata, stats = _prepare_table2_dual(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            seed=seed,
        )
    elif spec.task_type == "multi_cond":
        primary_adata, stats = _prepare_table3_lung_cancer(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            min_genes=min_genes,
            min_cells=min_cells,
            seed=seed,
        )
    elif spec.task_type == "bec":
        primary_adata, stats = _prepare_table4_bec(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            min_genes=min_genes,
            min_cells=min_cells,
            seed=seed,
        )
    elif spec.task_type == "perturb_loo":
        primary_adata, stats = _prepare_table5_pbmc_ifn(
            spec,
            raw_paths,
            out_paths,
            gene_embeddings,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task_type: {spec.task_type}")

    return _make_prepared_record(spec, out_paths, primary_adata, stats)


def _update_manifest(
    root: Path,
    genept_path: Path,
    records: dict[str, PreparedDataset],
) -> Path:
    manifest_path = root / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:
            existing = {}

    dataset_entries = dict(existing.get("datasets", {}))
    for name, rec in records.items():
        dataset_entries[name] = rec.to_dict(data_root=root)

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gene_embedding_path": str(genept_path),
        "datasets": dataset_entries,
    }
    tmp_path = manifest_path.with_suffix(".json.part")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, manifest_path)
    return manifest_path


def _prepare_unique_raw_files(
    pending_specs: Sequence[DatasetSpec],
    data_root: Path,
    *,
    skip_download: bool,
    force: bool,
    workers: int,
) -> None:
    """Ensure all unique raw inputs required by pending_specs exist without duplicate downloads."""
    if not pending_specs:
        return

    seen_dests: set[Path] = set()
    ordered_dests: list[Path] = []
    cfgen_groups: dict[tuple[str, tuple[str, ...]], dict[str, Path]] = {}
    download_jobs: list[Any] = []

    for spec in pending_specs:
        raw_paths = _get_raw_paths(spec, data_root)
        if spec.download_kind == "cfgen_zip64":
            group_key = (spec.source_urls[0], spec.backup_urls)
            member_map = cfgen_groups.setdefault(group_key, {})
            for member, dest in zip(spec.archive_members, raw_paths):
                resolved = dest.resolve()
                if resolved not in seen_dests:
                    seen_dests.add(resolved)
                    ordered_dests.append(dest)
                    member_map[member] = dest
        elif spec.download_kind == "direct":
            dest = raw_paths[0]
            resolved = dest.resolve()
            if resolved not in seen_dests:
                seen_dests.add(resolved)
                ordered_dests.append(dest)
                url = spec.source_urls[0]
                backups = spec.backup_urls
                download_jobs.append(
                    lambda u=url, b=backups, d=dest: _download_file(
                        u,
                        d,
                        backup_urls=b,
                        expected_magic=HDF5_MAGIC,
                        force=force,
                    )
                )
        elif spec.download_kind == "multi_direct":
            for url, dest in zip(spec.source_urls, raw_paths):
                resolved = dest.resolve()
                if resolved not in seen_dests:
                    seen_dests.add(resolved)
                    ordered_dests.append(dest)
                    download_jobs.append(
                        lambda u=url, d=dest: _download_file(
                            u,
                            d,
                            expected_magic=HDF5_MAGIC,
                            force=force,
                        )
                    )
        elif spec.download_kind == "gdrive":
            dest = raw_paths[0]
            resolved = dest.resolve()
            if resolved not in seen_dests:
                seen_dests.add(resolved)
                ordered_dests.append(dest)
                gid = spec.gdrive_id
                assert gid is not None
                download_jobs.append(
                    lambda g=gid, d=dest: _download_gdrive_file(
                        g,
                        d,
                        expected_magic=HDF5_MAGIC,
                        force=force,
                    )
                )
        else:
            raise ValueError(f"Unsupported download_kind: {spec.download_kind}")

    if skip_download:
        missing = [p for p in ordered_dests if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing raw files with --skip-download: {missing}"
            )
        return

    for (url, backups), member_map in cfgen_groups.items():
        if member_map:
            download_jobs.append(
                lambda u=url, b=backups, m=member_map: extract_zip_entries_via_http_range(
                    u,
                    m,
                    backup_urls=b,
                    force=force,
                    max_workers=workers,
                )
            )

    if not download_jobs:
        return
    if workers <= 1 or len(download_jobs) == 1:
        for job in download_jobs:
            job()
        return

    with ThreadPoolExecutor(max_workers=min(workers, len(download_jobs))) as pool:
        futures = [pool.submit(job) for job in download_jobs]
        for fut in as_completed(futures):
            fut.result()


def prepare_datasets(
    datasets: str | Sequence[str] = "all",
    data_root: str | Path = "./data",
    *,
    gene_embedding_path: str | Path | None = None,
    skip_download: bool = False,
    force: bool = False,
    subset_cells: int | None = None,
    ot_method: str = "emd",
    ot_reg: float = 0.1,
    ot_max_iter: int = 20,
    ot_num_iter_max: int = DEFAULT_OT_NUM_ITER_MAX,
    min_genes: int | None = None,
    min_cells: int | None = None,
    seed: int = 42,
    workers: int | str = 1,
) -> dict[str, PreparedDataset]:
    """Prepare one or more benchmark datasets (or 'all' / 'table1'..'table5') and write manifest.json."""
    n_workers = resolve_workers(workers)
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    selected = resolve_dataset_names(datasets)

    specs = [get_dataset_spec(name) for name in selected]
    pending_specs = [
        spec
        for spec in specs
        if force
        or subset_cells is not None
        or not _outputs_ready(get_output_paths(spec.name, root))
    ]

    loaded_embeddings: dict[str, np.ndarray] | None = None
    emb_path: Path | None = (
        Path(gene_embedding_path) if gene_embedding_path is not None else None
    )

    # Step A: Ensure and load GenePT embeddings once upfront when any dataset needs building.
    if pending_specs:
        emb_path = ensure_genept_embeddings(
            root,
            gene_embedding_path=gene_embedding_path,
            force=force,
            skip_download=skip_download,
        )
        loaded_embeddings = load_gene_embeddings(str(emb_path))

    # Step B: Deduplicate raw files by destination path and download concurrently.
    _prepare_unique_raw_files(
        pending_specs,
        root,
        skip_download=skip_download,
        force=force,
        workers=n_workers,
    )

    # Step C: Preprocess and run OT Gene Block Construction concurrently (all raw files ready).
    def _build_one(ds_name: str) -> PreparedDataset:
        return prepare_dataset(
            ds_name,
            data_root=root,
            gene_embedding_path=emb_path,
            gene_embeddings=loaded_embeddings,
            skip_download=True,
            force=force,
            subset_cells=subset_cells,
            ot_method=ot_method,
            ot_reg=ot_reg,
            ot_max_iter=ot_max_iter,
            ot_num_iter_max=ot_num_iter_max,
            min_genes=min_genes,
            min_cells=min_cells,
            seed=seed,
        )

    results: dict[str, PreparedDataset] = {}
    if n_workers <= 1 or len(selected) <= 1:
        for name in selected:
            results[name] = _build_one(name)
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(selected))) as pool:
            fut_to_name = {pool.submit(_build_one, name): name for name in selected}
            raw_results: dict[str, PreparedDataset] = {}
            for fut in as_completed(fut_to_name):
                raw_results[fut_to_name[fut]] = fut.result()
        for name in selected:
            results[name] = raw_results[name]

    manifest_emb = emb_path if emb_path is not None else (root / "genept" / GENEPT_PICKLE_NAME)
    _update_manifest(root, manifest_emb, results)
    return results
