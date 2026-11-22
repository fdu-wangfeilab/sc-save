"""Offline unit tests for sc_save_public.datasets and examples/prepare_datasets.py."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import pickle
import struct
import sys
import threading
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from sc_save_public import datasets as ds


# ---------------------------------------------------------------------------
# Helpers for synthetic AnnData, embeddings, and HTTP Range ZIP server
# ---------------------------------------------------------------------------


def _make_fake_embeddings(gene_names: list[str], dim: int = 8, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        gene: rng.normal(size=dim).astype(np.float32)
        for gene in gene_names
    }


def _write_fake_genept_pickle(path: Path, gene_names: list[str], dim: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    embs = _make_fake_embeddings(gene_names, dim=dim)
    with open(path, "wb") as handle:
        pickle.dump(embs, handle)
    return path


def _make_synthetic_adata(
    n_obs: int,
    gene_names: list[str],
    obs_df: pd.DataFrame,
    *,
    raw_counts: bool = False,
    log1p_uns: bool = False,
    sparse_x: bool = True,
    seed: int = 0,
    var_df: pd.DataFrame | None = None,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    if raw_counts:
        X = rng.poisson(lam=2.5, size=(n_obs, len(gene_names))).astype(np.float32)
        # Ensure high expression diversity across genes for HVG selection.
        scale = np.linspace(1.0, 5.0, len(gene_names), dtype=np.float32)
        X = np.round(X * scale[None, :]).astype(np.float32)
    else:
        counts = rng.poisson(lam=2.0, size=(n_obs, len(gene_names))).astype(np.float32)
        lib = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        X = np.log1p(counts / lib * 1e4).astype(np.float32)

    mat = sparse.csr_matrix(X) if sparse_x else X
    obs = obs_df.copy()
    obs.index = pd.Index([f"cell_{i}" for i in range(n_obs)], dtype="object")
    if var_df is None:
        var = pd.DataFrame(index=pd.Index(gene_names, dtype="object"))
    else:
        var = var_df.copy()
        var.index = pd.Index(gene_names, dtype="object")
    adata = ad.AnnData(X=mat, obs=obs, var=var)
    if log1p_uns:
        adata.uns["log1p"] = {"base": None}
    return adata


class _RangeRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler supporting Range requests without external dependencies."""

    recorded_ranges: list[str | None] = []

    def do_GET(self) -> None:
        range_header = self.headers.get("Range")
        self.__class__.recorded_ranges.append(range_header)
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return

        size = path.stat().st_size
        if not range_header or not range_header.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            self.wfile.write(path.read_bytes())
            return

        spec = range_header[len("bytes=") :]
        if spec.startswith("-"):
            length = int(spec[1:])
            start = max(0, size - length)
            end = size - 1
        else:
            start_s, end_s = spec.split("-", 1)
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            payload = f.read(length)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _convert_zip_to_zip64(src_zip: Path, dst_zip: Path) -> None:
    """Wrap an existing ZIP file with explicit ZIP64 EOCD + locator records."""
    with zipfile.ZipFile(src_zip, "r") as zf:
        infos = zf.infolist()

    raw = src_zip.read_bytes()
    eocd_pos = raw.rfind(b"PK\x05\x06")
    assert eocd_pos != -1
    _, disk_no, cd_disk, n_entries_disk, n_entries, cd_size, cd_offset, _ = (
        struct.unpack_from("<4sHHHHIIH", raw, eocd_pos)
    )
    prefix = raw[:eocd_pos]
    z64_eocd_offset = len(prefix)
    z64_eocd = struct.pack(
        "<4sQHHIIQQQQ",
        b"PK\x06\x06",
        44,
        45,
        45,
        disk_no,
        cd_disk,
        len(infos),
        len(infos),
        cd_size,
        cd_offset,
    )
    z64_loc = struct.pack(
        "<4sIQH",
        b"PK\x06\x07",
        0,
        z64_eocd_offset,
        1,
    )
    eocd = struct.pack(
        "<4sHHHHIIH",
        b"PK\x05\x06",
        0,
        0,
        min(n_entries_disk, 0xFFFF),
        min(n_entries, 0xFFFF),
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    dst_zip.write_bytes(prefix + z64_eocd + z64_loc + eocd)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_registry_specs_and_aliases() -> None:
    specs = {s.name: s for s in ds.list_datasets()}
    assert len(specs) == 11
    assert ds.resolve_dataset_names("all") == list(specs.keys())
    assert ds.resolve_dataset_names("table1") == ["pbmc3k", "dentategyrus", "tabula_muris"]
    assert ds.resolve_dataset_names("table2") == ["heart", "pbmc", "lung_atlas"]
    assert ds.resolve_dataset_names("table3") == ["lung_cancer"]
    assert ds.resolve_dataset_names("table4") == [
        "heart_bec",
        "pbmc_bec",
        "lung_atlas_bec",
    ]
    assert ds.resolve_dataset_names("table5") == ["pbmc_ifn"]
    assert ds.resolve_dataset_names("hca_subsampled_20k,pbmc-ifn") == [
        "heart",
        "pbmc_ifn",
    ]
    assert ds.get_dataset_spec("cancer_lung").cond_cols == (
        "assay",
        "development_stage",
        "disease",
        "uicc_stage",
        "cell_type",
    )
    assert ds.get_dataset_spec("heart").gene_block_size == 3200
    assert ds.get_dataset_spec("heart_bec").gene_block_size == 400
    assert ds.get_dataset_spec("heart_bec").hvg_top_genes == 2000


def test_zip64_http_range_extraction_offline(tmp_path: Path) -> None:
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    plain_zip = srv_dir / "plain.zip"
    zip64_path = srv_dir / "archive64.zip"

    wanted_a = b"HDF5_SIM_A_" * 500
    wanted_b = b"HDF5_SIM_B_" * 700
    huge_unused = b"UNUSED_DATASET_" * 20000

    with zipfile.ZipFile(plain_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("datasets/unused/big.h5ad", huge_unused)
        zf.writestr("datasets/pbmc3k/pbmc3k_train.h5ad", wanted_a)
        zf.writestr(
            "datasets/pbmc3k/pbmc3k_test.h5ad",
            wanted_b,
            compress_type=zipfile.ZIP_STORED,
        )
    _convert_zip_to_zip64(plain_zip, zip64_path)

    _RangeRequestHandler.recorded_ranges = []
    handler = partial(_RangeRequestHandler, directory=str(srv_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/archive64.zip"
        dest_a = tmp_path / "out" / "pbmc3k_train.h5ad"
        dest_b = tmp_path / "out" / "pbmc3k_test.h5ad"
        ds.extract_zip_entries_via_http_range(
            url,
            {
                "datasets/pbmc3k/pbmc3k_train.h5ad": dest_a,
                "datasets/pbmc3k/pbmc3k_test.h5ad": dest_b,
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert dest_a.read_bytes() == wanted_a
    assert dest_b.read_bytes() == wanted_b
    assert all(r is not None for r in _RangeRequestHandler.recorded_ranges)


def test_ot_emd_receives_large_num_iter_max_and_balances_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_num_iters: list[int] = []
    real_emd = ds.ot.emd

    def spy_emd(a, b, M, **kwargs):
        recorded_num_iters.append(int(kwargs.get("numItermax", -1)))
        return real_emd(a, b, M, **kwargs)

    monkeypatch.setattr(ds.ot, "emd", spy_emd)

    rng = np.random.default_rng(123)
    X = rng.normal(size=(24, 6))
    labels, _ = ds.balanced_kmeans_with_ot_emd(X, n_clusters=3, max_iter=5)

    assert recorded_num_iters
    assert all(val == ds.DEFAULT_OT_NUM_ITER_MAX for val in recorded_num_iters)
    assert ds.DEFAULT_OT_NUM_ITER_MAX >= 1_000_000
    unique, counts = np.unique(labels, return_counts=True)
    assert list(unique) == [0, 1, 2]
    assert list(counts) == [8, 8, 8]


def test_apply_gene_block_construction_multiblock_and_small_overlap() -> None:
    genes = [f"G_{i}" for i in range(25)] + [f"UNCOV_{j}" for j in range(5)]
    obs = pd.DataFrame({"cell_type": ["A"] * 6 + ["B"] * 6})
    adata = _make_synthetic_adata(12, genes, obs, raw_counts=False, log1p_uns=True)
    embs = _make_fake_embeddings([f"G_{i}" for i in range(25)], dim=6)

    # 25 covered with GBS=10 -> 2 blocks of 10 + 10 remainder -> multiple of 10 (0 pad)
    proc_2blk, stats = ds.apply_gene_block_construction(adata, embs, gene_block_size=10)
    assert stats["n_semantic_blocks"] == 2
    assert proc_2blk.n_vars == 30
    assert (proc_2blk.var["gene_block_id"][:10] == 0).all()
    assert (proc_2blk.var["gene_block_id"][10:20] == 1).all()
    assert (proc_2blk.var["gene_block_id"][20:] == -1).all()

    # Add 2 more uncovered genes -> 32 real genes -> padded with 8 pad_* to 40
    adata_extra = _make_synthetic_adata(
        12, genes + ["UNCOV_5", "UNCOV_6"], obs, raw_counts=False, log1p_uns=True
    )
    proc_padded, pad_stats = ds.apply_gene_block_construction(
        adata_extra, embs, gene_block_size=10
    )
    assert proc_padded.n_vars == 40
    assert pad_stats["n_pad"] == 8
    assert list(proc_padded.var_names[-8:]) == [f"pad_{i}" for i in range(8)]

    # Small overlap < gene_block_size fallback (matches mouse Table 1 datasets)
    small_embs = _make_fake_embeddings(["G_0", "G_1", "G_2"], dim=6)
    proc_small, small_stats = ds.apply_gene_block_construction(
        adata[:, :22].copy(), small_embs, gene_block_size=32
    )
    assert small_stats["n_semantic_blocks"] == 0
    assert proc_small.n_vars == 32
    assert small_stats["n_pad"] == 10
    assert list(proc_small.var_names[:3]) == ["G_0", "G_1", "G_2"]


def test_table1_cfgen_datasets_and_test_alignment(tmp_path: Path) -> None:
    shared_genes = [f"HUMAN_G_{i}" for i in range(3300)]
    extra_genes = [f"EXTRA_G_{i}" for i in range(150)]
    all_pbmc_genes = shared_genes + extra_genes

    data_root = tmp_path / "data"
    emb_path = _write_fake_genept_pickle(
        data_root / "genept" / ds.GENEPT_PICKLE_NAME, shared_genes
    )

    # 1. PBMC3K (shared >= 3200)
    raw_pbmc = data_root / "raw" / "table1_single_cond" / "pbmc3k"
    raw_pbmc.mkdir(parents=True)
    obs_tr = pd.DataFrame({"cell_type": ["B", "T"] * 6})
    obs_te = pd.DataFrame({"cell_type": ["B", "T"] * 2})
    _make_synthetic_adata(12, all_pbmc_genes, obs_tr, log1p_uns=True).write_h5ad(
        raw_pbmc / "pbmc3k_train.h5ad"
    )
    _make_synthetic_adata(4, all_pbmc_genes, obs_te, log1p_uns=True).write_h5ad(
        raw_pbmc / "pbmc3k_test.h5ad"
    )

    # 2. Dentate gyrus (mouse overlap < 3200, standardize clusters -> cell_type)
    mouse_genes = [f"MouseGene{i}" for i in range(200)] + ["HUMAN_G_0", "HUMAN_G_1"]
    raw_dg = data_root / "raw" / "table1_single_cond" / "dentategyrus"
    raw_dg.mkdir(parents=True)
    obs_dg_tr = pd.DataFrame({"clusters": ["Granule", "Astro"] * 5})
    obs_dg_te = pd.DataFrame({"clusters": ["Granule", "Astro"] * 2})
    _make_synthetic_adata(10, mouse_genes, obs_dg_tr, log1p_uns=True).write_h5ad(
        raw_dg / "dentategyrus_train.h5ad"
    )
    _make_synthetic_adata(4, mouse_genes, obs_dg_te, log1p_uns=True).write_h5ad(
        raw_dg / "dentategyrus_test.h5ad"
    )

    # 3. Tabula Muris (tissue must overwrite cell_type, matching paper Section 4.1)
    raw_tm = data_root / "raw" / "table1_single_cond" / "tabula_muris"
    raw_tm.mkdir(parents=True)
    obs_tm_tr = pd.DataFrame(
        {"tissue": ["Lung", "Spleen"] * 5, "cell_type": ["wrong"] * 10}
    )
    obs_tm_te = pd.DataFrame(
        {"tissue": ["Lung", "Spleen"] * 2, "cell_type": ["wrong"] * 4}
    )
    _make_synthetic_adata(10, mouse_genes, obs_tm_tr, log1p_uns=True).write_h5ad(
        raw_tm / "tabula_muris_train.h5ad"
    )
    _make_synthetic_adata(4, mouse_genes, obs_tm_te, log1p_uns=True).write_h5ad(
        raw_tm / "tabula_muris_test.h5ad"
    )

    res = ds.prepare_datasets(
        "table1",
        data_root=data_root,
        gene_embedding_path=emb_path,
        skip_download=True,
    )

    pbmc_tr = sc.read_h5ad(res["pbmc3k"].files["train"])
    pbmc_te = sc.read_h5ad(res["pbmc3k"].files["test"])
    assert pbmc_tr.n_vars == 6400
    assert list(pbmc_tr.var_names) == list(pbmc_te.var_names)
    assert res["pbmc3k"].n_semantic_blocks == 1
    assert pbmc_tr.var_names[-1].startswith("pad_")

    dg_tr = sc.read_h5ad(res["dentategyrus"].files["train"])
    dg_te = sc.read_h5ad(res["dentategyrus"].files["test"])
    assert dg_tr.n_vars == 3200
    assert list(dg_tr.var_names) == list(dg_te.var_names)
    assert set(dg_tr.obs["cell_type"]) == {"Granule", "Astro"}

    tm_tr = sc.read_h5ad(res["tabula_muris"].files["train"])
    assert tm_tr.n_vars == 3200
    assert set(tm_tr.obs["cell_type"]) == {"Lung", "Spleen"}


def test_table2_and_table4_heart_pbmc_lung_atlas_parity(tmp_path: Path) -> None:
    genept_genes = [f"GENE_{i}" for i in range(3300)]
    extra_genes = [f"NOVEL_{i}" for i in range(50)] + ["MT-CO1", "mt-Nd1", "ERCC-0001"]
    all_genes = genept_genes + extra_genes

    data_root = tmp_path / "data"
    emb_path = _write_fake_genept_pickle(
        data_root / "genept" / ds.GENEPT_PICKLE_NAME, genept_genes
    )
    raw_dir = data_root / "raw" / "dual_cond"
    raw_dir.mkdir(parents=True)

    # Heart: raw integer counts, donor -> batch
    heart_obs = pd.DataFrame(
        {"donor": ["D1", "D2"] * 10, "cell_type": ["CM", "FB"] * 10}
    )
    _make_synthetic_adata(20, all_genes, heart_obs, raw_counts=True, seed=1).write_h5ad(
        raw_dir / "hca_subsampled_20k.h5ad"
    )

    # PBMC: final_annotation -> cell_type
    pbmc_obs = pd.DataFrame(
        {"batch": ["10x", "SmartSeq"] * 10, "final_annotation": ["CD4", "Mono"] * 10}
    )
    _make_synthetic_adata(20, all_genes, pbmc_obs, log1p_uns=True, seed=2).write_h5ad(
        raw_dir / "pbmc_vars_sb.h5ad"
    )

    # Lung Atlas: batch + cell_type
    lung_obs = pd.DataFrame(
        {"batch": ["B1", "B2"] * 10, "cell_type": ["AT2", "Macrophage"] * 10}
    )
    _make_synthetic_adata(20, all_genes, lung_obs, log1p_uns=True, seed=3).write_h5ad(
        raw_dir / "lung_atlas.h5ad"
    )

    res_t2 = ds.prepare_datasets(
        "table2",
        data_root=data_root,
        gene_embedding_path=emb_path,
        skip_download=True,
    )
    heart_t2 = sc.read_h5ad(res_t2["heart"].primary_path)
    assert heart_t2.n_vars == 6400
    assert set(heart_t2.obs["batch"]) == {"D1", "D2"}
    assert "counts" in heart_t2.layers
    assert "log1p" in heart_t2.uns
    assert float(heart_t2.X.max()) <= np.log1p(1e4) + 1e-4

    pbmc_t2 = sc.read_h5ad(res_t2["pbmc"].primary_path)
    assert set(pbmc_t2.obs["cell_type"]) == {"CD4", "Mono"}

    res_t4 = ds.prepare_datasets(
        "table4",
        data_root=data_root,
        gene_embedding_path=emb_path,
        skip_download=True,
        min_genes=10,
        min_cells=1,
    )
    for key in ("heart_bec", "pbmc_bec", "lung_atlas_bec"):
        bec = sc.read_h5ad(res_t4[key].primary_path)
        assert bec.n_vars == 2000
        assert res_t4[key].gene_block_size == 400
        assert not any(
            str(g).startswith(("ERCC", "MT-", "mt-")) for g in bec.var_names
        )
        # BEC leaves expression log1p-normalized (unscaled, MaxAbsScaler applied at train time)
        assert float(bec.X.max()) > 1.0


def test_table3_lung_cancer_age_filter_and_unseen_split(tmp_path: Path) -> None:
    # 2200 Ensembl IDs mapped to feature_name symbols covered by GenePT + 300 uncovered
    ensembl_ids = [f"ENSG{i:06d}" for i in range(2500)]
    feature_names = [f"LC_GENE_{i}" for i in range(2200)] + [
        f"LC_UNCOV_{i}" for i in range(300)
    ]

    data_root = tmp_path / "data"
    emb_path = _write_fake_genept_pickle(
        data_root / "genept" / ds.GENEPT_PICKLE_NAME, feature_names[:2200]
    )
    raw_dir = data_root / "raw" / "table3_multi_cond"
    raw_dir.mkdir(parents=True)

    rows = [
        # Out-of-age-range rows (must be filtered out during backed slice)
        ("10x 3' v2", "40-year-old human stage", "normal", "non-cancer", "T cell"),
        ("10x 3' v3", "80-year-old human stage", "lung adenocarcinoma", "III", "AT2"),
    ]
    # Seen 55-70 rows
    for _ in range(8):
        rows.append(
            ("10x 3' v2", "55-year-old human stage", "lung adenocarcinoma", "II", "AT2")
        )
        rows.append(
            ("10x 3' v3", "60-year-old human stage", "normal", "non-cancer", "NK")
        )
    # Unseen condition 13
    for _ in range(4):
        rows.append(
            (
                "10x 3' v2",
                "65-year-old human stage",
                "chronic obstructive pulmonary disease",
                "non-cancer",
                "Macrophage",
            )
        )
    # Unseen condition 24
    for _ in range(4):
        rows.append(
            (
                "Smart-seq2",
                "70-year-old human stage",
                "squamous cell lung carcinoma",
                "I",
                "B cell",
            )
        )

    obs_df = pd.DataFrame(
        rows,
        columns=["assay", "development_stage", "disease", "uicc_stage", "cell_type"],
    )
    var_df = pd.DataFrame({"feature_name": feature_names})
    _make_synthetic_adata(
        len(obs_df),
        ensembl_ids,
        obs_df,
        raw_counts=False,
        log1p_uns=True,
        var_df=var_df,
    ).write_h5ad(raw_dir / "cancer_lung.h5ad")

    res = ds.prepare_datasets(
        "lung_cancer",
        data_root=data_root,
        gene_embedding_path=emb_path,
        skip_download=True,
        min_genes=10,
        min_cells=1,
    )

    rec = res["lung_cancer"]
    full = sc.read_h5ad(rec.files["full"])
    train = sc.read_h5ad(rec.files["train"])
    test = sc.read_h5ad(rec.files["test"])

    assert full.n_obs == 24  # 26 minus 2 out-of-age rows
    assert train.n_obs == 16
    assert test.n_obs == 8
    assert full.n_vars == 2000
    assert not any(v.startswith("ENSG") for v in full.var_names)
    assert not set(train.obs["development_stage"]).intersection(
        {"40-year-old human stage", "80-year-old human stage"}
    )
    assert set(test.obs["development_stage"]) == {
        "65-year-old human stage",
        "70-year-old human stage",
    }


def test_table5_pbmc_ifn_and_cli_manifest(tmp_path: Path) -> None:
    genept_genes = [f"IFN_G_{i}" for i in range(1800)]
    all_genes = genept_genes + [f"IFN_OTHER_{i}" for i in range(600)]

    data_root = tmp_path / "data"
    emb_path = _write_fake_genept_pickle(
        data_root / "genept" / ds.GENEPT_PICKLE_NAME, genept_genes
    )
    raw_dir = data_root / "raw" / "table5_perturb"
    raw_dir.mkdir(parents=True)

    obs_tr = pd.DataFrame(
        {"condition": ["ctrl", "stim"] * 6, "cell_type": ["CD4T", "B"] * 6}
    )
    obs_va = pd.DataFrame(
        {"condition": ["control", "stimulated"] * 3, "cell_type": ["CD8T", "NK"] * 3}
    )
    _make_synthetic_adata(12, all_genes, obs_tr, log1p_uns=True, seed=4).write_h5ad(
        raw_dir / "train_pbmc.h5ad"
    )
    _make_synthetic_adata(6, all_genes, obs_va, log1p_uns=True, seed=5).write_h5ad(
        raw_dir / "valid_pbmc.h5ad"
    )

    script_path = Path(__file__).resolve().parents[1] / "examples" / "prepare_datasets.py"
    spec = importlib.util.spec_from_file_location("prepare_datasets", script_path)
    assert spec is not None and spec.loader is not None
    cli_mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_datasets"] = cli_mod
    spec.loader.exec_module(cli_mod)
    cli_main = cli_mod.main

    assert cli_main(["--list-datasets", "--data-root", str(data_root)]) == 0
    exit_code = cli_main(
        [
            "--datasets",
            "pbmc_ifn",
            "--data-root",
            str(data_root),
            "--gene-embedding-path",
            str(emb_path),
            "--skip-download",
        ]
    )
    assert exit_code == 0

    out_path = ds.get_output_paths("pbmc_ifn", data_root)["full"]
    adata_ifn = sc.read_h5ad(out_path)
    assert adata_ifn.n_obs == 18
    assert adata_ifn.n_vars == 2000
    assert set(adata_ifn.obs["condition"]) == {"control", "stimulated"}
    assert set(adata_ifn.obs["cell_type"]) == {"CD4T", "B", "CD8T", "NK"}

    manifest_path = data_root / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "pbmc_ifn" in manifest["datasets"]
    assert manifest["datasets"]["pbmc_ifn"]["n_vars"] == 2000


def test_ensure_genept_embeddings_trailing_dot_and_exact_preference(
    tmp_path: Path,
) -> None:
    # 1. Exact Zenodo archive layout with trailing dot on model_3_text.pickle.
    root_dot = tmp_path / "dot_case"
    zip_dot = root_dot / "raw" / "genept" / "GenePT_emebdding_v2.zip"
    zip_dot.parent.mkdir(parents=True, exist_ok=True)
    expected_embs = _make_fake_embeddings(["TP53", "GAPDH"], dim=4, seed=9)
    with zipfile.ZipFile(zip_dot, "w") as zf:
        zf.writestr("GenePT_emebdding_v2/", b"")
        zf.writestr(
            f"__MACOSX/GenePT_emebdding_v2/{ds.GENEPT_PICKLE_NAME}",
            b"macos-resource-fork",
        )
        zf.writestr("GenePT_emebdding_v2/NCBI_UniProt_summary_of_genes.json", b"{}")
        zf.writestr("GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle", b"ada")
        zf.writestr(
            f"GenePT_emebdding_v2/{ds.GENEPT_PICKLE_NAME}.",
            pickle.dumps(expected_embs),
        )
        zf.writestr("GenePT_emebdding_v2/NCBI_summary_of_genes.json", b"{}")

    extracted = ds.ensure_genept_embeddings(root_dot)
    assert extracted == root_dot / "genept" / ds.GENEPT_PICKLE_NAME
    loaded = pickle.loads(extracted.read_bytes())
    assert set(loaded.keys()) == {"TP53", "GAPDH"}
    assert np.allclose(loaded["TP53"], expected_embs["TP53"])

    # 2. When both trailing-dot and exact match exist, prefer exact match.
    root_exact = tmp_path / "exact_case"
    zip_exact = root_exact / "raw" / "genept" / "GenePT_emebdding_v2.zip"
    zip_exact.parent.mkdir(parents=True, exist_ok=True)
    exact_embs = _make_fake_embeddings(["EXACT_GENE"], dim=4, seed=11)
    with zipfile.ZipFile(zip_exact, "w") as zf:
        zf.writestr(
            f"GenePT_emebdding_v2/{ds.GENEPT_PICKLE_NAME}.",
            pickle.dumps(expected_embs),
        )
        zf.writestr(
            f"GenePT_emebdding_v2/{ds.GENEPT_PICKLE_NAME}",
            pickle.dumps(exact_embs),
        )

    extracted_exact = ds.ensure_genept_embeddings(root_exact)
    loaded_exact = pickle.loads(extracted_exact.read_bytes())
    assert set(loaded_exact.keys()) == {"EXACT_GENE"}
