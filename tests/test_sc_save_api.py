"""Public v2 API, ConditionVocabulary, preprocessing, end-to-end workflow, and CLI tests."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import LabelEncoder


def test_condition_vocabulary_legacy_vs_offset_digit_prefixed_and_disjoint():
    from sc_save.data.preprocessing import ConditionVocabulary

    obs = pd.DataFrame(
        {
            "assay": ["10x 3' v2", "10x 3' v3", "Smart-seq2", "shared_tag"],
            "cell_type": ["T cell", "B cell", "10x 3' v2", "shared_tag"],
        },
        index=[f"c_{i}" for i in range(4)],
    )

    legacy = ConditionVocabulary(["assay", "cell_type"], mode="legacy").fit(obs)
    offset = ConditionVocabulary(["assay", "cell_type"], mode="offset").fit(obs)

    # Identical total_classes count in both modes
    assert legacy.total_classes == offset.total_classes
    assert offset.mask_id == 0
    assert legacy.offsets == {"assay": 0, "cell_type": 0}

    # In legacy mode, digit-prefixed "10x 3' v2" sorts before "<mask>", reproducing v1 quirk
    le_assay_v1 = LabelEncoder().fit(["<mask>"] + list(obs["assay"].unique()))
    assert list(legacy.label_encoder_dict["assay"].classes_) == list(le_assay_v1.classes_)
    assert int(le_assay_v1.transform(["10x 3' v2"])[0]) == 0
    assert int(le_assay_v1.transform(["<mask>"])[0]) != 0

    # In offset mode, "<mask>" is ALWAYS 0 even when digit-prefixed labels exist
    mask_vec = offset.encode_dict({"assay": "<mask>", "cell_type": "<mask>"})
    assert mask_vec.dtype == np.int64
    assert mask_vec.tolist() == [0, 0]

    partial_vec = offset.encode_dict({"assay": "10x 3' v2"}, allow_partial=True)
    assert partial_vec.shape == (2,)
    assert partial_vec[0] > 0
    assert partial_vec[1] == 0

    # Cross-column IDs are strictly disjoint even for identical string values across columns
    enc_offset = offset.encode_obs(obs)
    assert enc_offset.shape == (4, 2)
    assert enc_offset.dtype == np.int64
    col0_ids = set(enc_offset[:, 0].tolist())
    col1_ids = set(enc_offset[:, 1].tolist())
    assert 0 not in col0_ids and 0 not in col1_ids
    assert col0_ids.isdisjoint(col1_ids)
    assert max(col1_ids) < offset.total_classes
    assert set(col0_ids) == {1, 2, 3, 4}
    assert set(col1_ids) == {5, 6, 7, 8}

    # Roundtrip serialization parity
    reloaded_offset = ConditionVocabulary.from_dict(offset.to_dict())
    reloaded_legacy = ConditionVocabulary.from_dict(legacy.to_dict())
    np.testing.assert_array_equal(reloaded_offset.encode_obs(obs), enc_offset)
    np.testing.assert_array_equal(
        reloaded_legacy.encode_obs(obs), legacy.encode_obs(obs)
    )


def test_maxabs_scaler_and_preprocessing_helpers():
    from sc_save.data.preprocessing import (
        MaxAbsExpressionScaler,
        align_genes,
        find_pad_index,
        normalize_log1p,
        pad_to_block_size,
        select_hvgs_exact,
        trim_padded_genes,
    )

    # Dense + sparse MaxAbsExpressionScaler with all-zero column
    x_dense = np.array(
        [[0.0, 2.0, 0.0], [4.0, 0.0, 0.0], [2.0, 4.0, 0.0]],
        dtype=np.float32,
    )
    scaler_dense = MaxAbsExpressionScaler().fit(x_dense)
    np.testing.assert_allclose(scaler_dense.scale_, np.array([4.0, 4.0, 1.0], dtype=np.float32))
    scaled_dense = scaler_dense.transform(x_dense)
    assert scaled_dense.dtype == np.float32
    np.testing.assert_allclose(scaler_dense.inverse_transform(scaled_dense), x_dense, atol=1e-6)

    x_sparse = sparse.csr_matrix(x_dense)
    scaler_sparse = MaxAbsExpressionScaler()
    scaled_sparse = scaler_sparse.fit_transform(x_sparse)
    assert isinstance(scaled_sparse, np.ndarray) or sparse.issparse(scaled_sparse)
    scaled_sparse_arr = (
        scaled_sparse.toarray() if sparse.issparse(scaled_sparse) else np.asarray(scaled_sparse)
    )
    assert scaled_sparse_arr.dtype == np.float32
    np.testing.assert_allclose(scaled_sparse_arr, scaled_dense, atol=1e-6)

    scaler_reloaded = MaxAbsExpressionScaler.from_dict(scaler_dense.to_dict())
    np.testing.assert_allclose(scaler_reloaded.scale_, scaler_dense.scale_)

    # Padding, trimming, and gene alignment
    rng = np.random.default_rng(12)
    adata = ad.AnnData(
        X=rng.poisson(lam=3.0, size=(20, 15)).astype(np.float32),
        obs=pd.DataFrame({"batch": ["b1", "b2"] * 10}, index=[f"c_{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"G_{i}" for i in range(15)]),
    )
    normed = normalize_log1p(adata, target_sum=1e4, normalize_total=True, preserve_counts_layer=True)
    assert "counts" in normed.layers
    assert "log1p" in normed.uns

    hvg_exact = select_hvgs_exact(normed, n_top_genes=10)
    assert hvg_exact.n_vars == 10

    padded, pad_idx = pad_to_block_size(hvg_exact, block_size=8)
    assert pad_idx == 10
    assert padded.n_vars == 16
    assert find_pad_index(padded) == 10
    assert list(padded.var_names[-6:]) == [f"pad_{i}" for i in range(6)]

    trimmed = trim_padded_genes(padded)
    assert trimmed.n_vars == 10
    assert list(trimmed.var_names) == list(hvg_exact.var_names)

    # Align subset / reordered genes with missing padded genes
    partial = hvg_exact[:, list(hvg_exact.var_names[:5])].copy()
    aligned = align_genes(partial, target_genes=list(padded.var_names), pad_missing=True)
    assert list(aligned.var_names) == list(padded.var_names)


def test_save_config_presets_and_serialization():
    from sc_save import FMConfig, SAVEConfig, TrainConfig, VAEConfig

    for preset in (
        SAVEConfig.save_b(),
        SAVEConfig.save_b_perturb(),
        SAVEConfig.smoke(),
    ):
        payload = preset.to_dict()
        restored = SAVEConfig.from_dict(payload)
        assert restored.to_dict() == payload
        assert isinstance(restored.vae, VAEConfig)
        assert isinstance(restored.fm, FMConfig)
        assert isinstance(restored.train, TrainConfig)

    assert SAVEConfig.save_b().vae.gene_block_size == 3200
    assert SAVEConfig.save_b_perturb().vae.gene_block_size == 400
    assert SAVEConfig.save_b().train.use_amp_vae is False


def test_end_to_end_sc_save_package_workflow_save_load(tmp_path):
    import sc_save
    from sc_save import (
        SAVE,
        ConditionVocabulary,
        GeneBlockResult,
        MaxAbsExpressionScaler,
        SAVEConfig,
        build_gene_blocks,
        load_gene_embeddings,
        preprocess_adata,
    )

    assert sc_save.__version__ == "0.2.0"
    assert ConditionVocabulary is not None
    assert MaxAbsExpressionScaler is not None
    assert preprocess_adata is not None

    # Create synthetic raw dataset + GenePT-style embeddings pickle
    rng = np.random.default_rng(77)
    n_obs = 32
    genes = [f"GENE_{i}" for i in range(25)] + [f"OTHER_{i}" for i in range(5)]
    counts = rng.poisson(lam=2.5, size=(n_obs, len(genes))).astype(np.float32)
    scale = np.linspace(1.0, 4.0, len(genes), dtype=np.float32)
    x = np.log1p(counts * scale[None, :]).astype(np.float32)
    obs = pd.DataFrame(
        {
            "batch": ["10x_v2", "10x_v3"] * (n_obs // 2),
            "cell_type": (["T", "T", "B", "B"] * (n_obs // 4)),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    raw_adata = ad.AnnData(
        X=sparse.csr_matrix(x),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )

    embs = {f"GENE_{i}": rng.normal(size=8).astype(np.float32) for i in range(25)}
    emb_path = tmp_path / "gene_embs.pkl"
    with open(emb_path, "wb") as handle:
        pickle.dump(embs, handle)

    loaded_embs = load_gene_embeddings(emb_path)
    assert set(loaded_embs.keys()) == set(embs.keys())

    block_res = build_gene_blocks(
        raw_adata,
        loaded_embs,
        block_size=10,
        ot_max_iter=5,
    )
    assert isinstance(block_res, GeneBlockResult)
    # 25 shared -> 2 semantic blocks of 10 + 5 shared remainder + 5 OTHER = 30 genes -> 0 pad
    assert block_res.n_semantic_blocks == 2
    assert block_res.adata.n_vars == 30

    cfg = SAVEConfig.smoke(gene_block_size=10, latent_block_size=10)
    cfg.fm.n_blocks = 1
    cfg.fm.num_heads = 2
    cfg.vae.enc_num_heads = 2
    cfg.vae.dec_num_heads = 2

    model = SAVE(
        block_res.adata,
        condition_cols=("batch", "cell_type"),
        config=cfg,
        device="cpu",
        cond_encoding="offset",
    )
    ret = model.fit(vae_epochs=1, fm_epochs=1)
    assert ret is model

    latents = model.encode(with_cond=False)
    assert latents.shape == (n_obs, 3, cfg.vae.enc_dim)

    decoded = model.decode(latents[:5], cond={"batch": "10x_v2", "cell_type": "T"}, trim_pad=True)
    assert decoded.shape == (5, 30)

    recon = model.reconstruct(trim_pad=True)
    assert recon.shape == (n_obs, 30)

    gen = model.generate({"batch": "10x_v2", "cell_type": "T"}, n_cells=6, steps=3)
    assert gen.shape == (6, 30)
    assert list(gen.obs["batch"]) == ["10x_v2"] * 6
    assert list(gen.obs["cell_type"]) == ["T"] * 6

    gen_like = model.generate_like(block_res.adata[:8], steps=3)
    assert gen_like.shape == (8, 30)

    attn_res = model.generate_with_attention(
        {"batch": "10x_v3", "cell_type": "B"},
        n_cells=4,
        steps=2,
        batch_size=4,
    )
    assert attn_res["adata"].shape == (4, 30)
    assert len(attn_res["attn_weights"]) == 2

    transferred = model.transfer(
        block_res.adata[block_res.adata.obs["batch"] == "10x_v2"][:4],
        target_cond={"batch": "10x_v3"},
        input_cond="obs",
        trim_pad=True,
    )
    assert transferred.shape == (4, 30)
    assert set(transferred.obs["batch"]) == {"10x_v3"}

    save_dir = tmp_path / "saved_model"
    model.save(save_dir)
    for fname in ("config.json", "vocab.json", "metadata.json", "scaler.npz", "vae.pt", "fm.pt"):
        assert (save_dir / fname).exists(), f"missing {fname} in saved bundle"
    meta = json.loads((save_dir / "metadata.json").read_text())
    assert meta["cond_encoding"] == "offset"

    reloaded = SAVE.load(save_dir, device="cpu")
    assert reloaded.cond_encoding == "offset"

    import torch

    np.testing.assert_allclose(
        reloaded.encode(block_res.adata[:6], with_cond=False, latent_pos=1),
        model.encode(block_res.adata[:6], with_cond=False, latent_pos=1),
        atol=1e-6,
    )
    torch.manual_seed(777)
    recon_orig = model.reconstruct(block_res.adata[:6])
    torch.manual_seed(777)
    recon_reload = reloaded.reconstruct(block_res.adata[:6])
    np.testing.assert_allclose(
        np.asarray(recon_reload.X),
        np.asarray(recon_orig.X),
        atol=1e-6,
    )
    torch.manual_seed(888)
    trans_orig = model.transfer(
        block_res.adata[:6],
        target_cond={"batch": "10x_v3"},
        input_cond="obs",
    )
    torch.manual_seed(888)
    trans_reload = reloaded.transfer(
        block_res.adata[:6],
        target_cond={"batch": "10x_v3"},
        input_cond="obs",
    )
    np.testing.assert_allclose(
        np.asarray(trans_reload.X),
        np.asarray(trans_orig.X),
        atol=1e-6,
    )

    torch.manual_seed(999)
    g1 = model.generate({"batch": "10x_v2", "cell_type": "T"}, n_cells=4, steps=3)
    torch.manual_seed(999)
    g2 = reloaded.generate({"batch": "10x_v2", "cell_type": "T"}, n_cells=4, steps=3)
    np.testing.assert_allclose(np.asarray(g1.X), np.asarray(g2.X), atol=1e-5)


@pytest.mark.parametrize(
    "args",
    [
        ["--version"],
        ["--help"],
        ["prepare-datasets", "--help"],
        ["build-blocks", "--help"],
        ["train", "--help"],
        ["generate", "--help"],
        ["evaluate", "--help"],
        ["bec", "--help"],
        ["perturb", "--help"],
        ["reproduce", "--help"],
    ],
)
def test_sc_save_cli_help_and_subcommands(args: list[str]):
    cli_bin = shutil.which("sc-save") or str(Path(sys.executable).parent / "sc-save")
    cmd = [cli_bin, *args] if Path(cli_bin).exists() else [sys.executable, "-m", "sc_save.cli", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"{cmd} failed:\n{proc.stderr}"
    if args == ["--version"]:
        assert "0.2.0" in proc.stdout
    else:
        assert "usage:" in proc.stdout.lower() or "options:" in proc.stdout.lower()
