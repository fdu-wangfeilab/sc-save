"""Offline unit and synthetic pipeline tests for BEC, conditional generation, perturbation WD/MMD, and ICLR 2026 reproduction."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import scanpy as sc
from sc_save_public.bec_integration import (
    _compat_scanpy_pca_for_scib,
    compute_scib_summary,
    run_bec_integration,
    scib_eval,
)
from sc_save_public.conditional_distribution_evaluation import (
    CANCER_LUNG_UNSEEN_CONDITIONS,
    run_conditional_distribution_evaluation,
)
from sc_save_public.conditional_generation import run_formal_conditional_generation
from sc_save_public.distribution_metrics import mix_rbf_mmd2, wasserstein
from sc_save_public.models.save_model import SAVE
from sc_save_public.perturbation_prediction import evaluate_perturbation, run_leave_one_out

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_repro_module():
    path = REPO_ROOT / "examples" / "reproduce_iclr2026.py"
    spec = importlib.util.spec_from_file_location("reproduce_iclr2026", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_adata(
    n_obs: int = 48,
    n_vars: int = 40,
    obs_df: pd.DataFrame | None = None,
    seed: int = 42,
    pad_tail: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0, size=(n_obs, n_vars)).astype(np.float32)
    var_names = [f"gene_{i}" for i in range(n_vars)]
    if pad_tail > 0:
        x_pad = np.zeros((n_obs, pad_tail), dtype=np.float32)
        x = np.concatenate([x, x_pad], axis=1)
        var_names += [f"pad_{i}" for i in range(pad_tail)]
    if obs_df is None:
        obs_df = pd.DataFrame(
            {
                "batch": ["b1"] * (n_obs // 2) + ["b2"] * (n_obs - n_obs // 2),
                "cell_type": (["T", "B"] * (n_obs // 2 + 1))[:n_obs],
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )
    return ad.AnnData(X=x, obs=obs_df, var=pd.DataFrame(index=var_names))


def test_compute_scib_summary_exact_paper_formula():
    raw = {
        "NMI_cluster/label": 0.80,
        "ARI_cluster/label": 0.70,
        "ASW_label": 0.60,
        "graph_conn": 0.90,
        "ASW_label/batch": 0.80,
        "PCR_batch": 0.25,  # must NOT enter avg_batch
        "kBET": float("nan"),
    }
    summary = compute_scib_summary(raw)
    assert summary["avg_bio"] == pytest.approx((0.80 + 0.70 + 0.60) / 3.0)
    assert summary["avg_batch"] == pytest.approx((0.90 + 0.80) / 2.0)
    assert summary["scib_score"] == pytest.approx(
        0.6 * summary["avg_bio"] + 0.4 * summary["avg_batch"]
    )
    assert summary["PCR_batch"] == pytest.approx(0.25)
    assert "kBET" not in summary


def test_scib_eval_runs_without_r_on_synthetic_adata():
    pytest.importorskip("scib")
    adata = _make_synthetic_adata(n_obs=60, n_vars=40, seed=7)
    # Give clusters clear structure in PCA embedding
    rng = np.random.default_rng(7)
    emb = rng.normal(0.0, 0.2, size=(60, 10)).astype(np.float32)
    is_t = (adata.obs["cell_type"] == "T").to_numpy()
    emb[is_t, 0] += 3.0
    adata.obsm["X_emb"] = emb

    res = scib_eval(adata, batch_key="batch", cell_type_key="cell_type", embed_key="X_emb")
    assert "avg_bio" in res
    assert "avg_batch" in res
    assert "scib_score" in res
    assert "PCR_batch" in res
    assert res["scib_score"] == pytest.approx(0.6 * res["avg_bio"] + 0.4 * res["avg_batch"])


def test_compat_scanpy_pca_for_scib_translates_and_restores(monkeypatch):
    calls = []

    def mock_pca_no_use_hv(adata, mask_var="UNSET"):
        calls.append((adata, mask_var))
        return "ok"

    monkeypatch.setattr(sc.tl, "pca", mock_pca_no_use_hv)

    with pytest.raises(RuntimeError, match="boom"):
        with _compat_scanpy_pca_for_scib():
            assert sc.tl.pca("a1", use_highly_variable=False) == "ok"
            assert sc.tl.pca("a2", use_highly_variable=True) == "ok"
            assert sc.tl.pca("a3", use_highly_variable=True, mask_var="custom") == "ok"
            raise RuntimeError("boom")

    assert calls == [
        ("a1", None),
        ("a2", "highly_variable"),
        ("a3", "custom"),
    ]
    # Restored cleanly even after an exception inside the context
    assert sc.tl.pca is mock_pca_no_use_hv


def test_run_bec_integration_smoke_and_ckpt_resume(tmp_path):
    pytest.importorskip("scib")
    adata = _make_synthetic_adata(n_obs=48, n_vars=45, seed=11)
    data_path = tmp_path / "bec_input.h5ad"
    adata.write_h5ad(data_path)

    out_dir = tmp_path / "bec_out"
    res1 = run_bec_integration(
        data_path=str(data_path),
        task_name="synth_bec",
        output_dir=str(out_dir),
        device_id=-1,
        epoch=1,
        batch_size=16,
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        gene_block_size=40,  # forces padding 45 -> 80
        latent_block_size=40,
        pca_components=8,
        transfer_batch_size=16,
    )

    assert Path(res1["ckpt_path"]).exists()
    assert not res1["report"]["loaded_existing_ckpt"]
    assert Path(res1["scib_csv_path"]).exists()
    assert Path(res1["report_path"]).exists()
    assert len(res1["embedding_paths"]) == 2
    for emb_path in res1["embedding_paths"].values():
        arr = np.load(emb_path)
        assert arr.shape == (48, 8)

    scib_df = pd.read_csv(res1["scib_csv_path"], index_col=0)
    assert "mean" in scib_df.columns and "std" in scib_df.columns
    assert "scib_score" in scib_df.index

    # Second run loads checkpoint cleanly (PyTorch 2.6+ weights_only=False compat)
    res2 = run_bec_integration(
        data_path=str(data_path),
        task_name="synth_bec",
        output_dir=str(out_dir),
        device_id=-1,
        epoch=1,
        batch_size=16,
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        gene_block_size=40,
        latent_block_size=40,
        pca_components=8,
        transfer_batch_size=16,
    )
    assert res2["report"]["loaded_existing_ckpt"] is True
    assert 0.0 <= res2["report"]["summary"]["scib_score"]["mean"] <= 1.0


def test_distribution_metrics_biased_mmd_and_wasserstein_parity():
    torch.manual_seed(0)
    x = torch.randn(12, 6)
    y = torch.randn(12, 6) + 0.5
    sigmas = [0.1, 1.0, 10.0]

    biased = float(mix_rbf_mmd2(x, y, sigmas).item())
    unbiased = float(mix_rbf_mmd2(x, y, sigmas, biased=False).item())
    assert biased > 0.0
    assert biased != pytest.approx(unbiased)

    # Single-sample biased MMD must not NaN
    single = mix_rbf_mmd2(x[:1], y[:1], sigmas, biased=True)
    assert torch.isfinite(single)
    # Identical distributions have zero 2-Wasserstein without sqrt(-eps) error
    assert wasserstein(x, x, power=2) == pytest.approx(0.0, abs=1e-6)


def test_conditional_distribution_evaluation_lung_cancer_seen_unseen_and_macro(tmp_path):
    rows = []
    # Seen condition combinations (2 macro groups x 2 cell types = 4 groups)
    seen_specs = [
        {"assay": "10x 3' v2", "development_stage": "55-year-old human stage", "disease": "normal", "uicc_stage": "non-cancer"},
        {"assay": "Smart-seq2", "development_stage": "60-year-old human stage", "disease": "lung adenocarcinoma", "uicc_stage": "IV"},
    ]
    for spec in seen_specs + list(CANCER_LUNG_UNSEEN_CONDITIONS):
        for ct in ("T", "Macrophage"):
            for _ in range(5):
                row = dict(spec)
                row["cell_type"] = ct
                rows.append(row)

    obs = pd.DataFrame(rows, index=[f"cell_{i}" for i in range(len(rows))])
    rng = np.random.default_rng(12)
    x_gt = rng.uniform(0.1, 3.0, size=(len(rows), 20)).astype(np.float32)
    x_gen = np.clip(x_gt + rng.normal(0.0, 0.05, size=x_gt.shape), 0.0, None).astype(np.float32)
    var = pd.DataFrame(index=[f"g_{i}" for i in range(20)])

    gt = ad.AnnData(X=x_gt, obs=obs.copy(), var=var.copy())
    gen = ad.AnnData(X=x_gen, obs=obs.copy(), var=var.copy())

    gt_path = tmp_path / "gt.h5ad"
    gen_path = tmp_path / "gen.h5ad"
    gt.write_h5ad(gt_path)
    gen.write_h5ad(gen_path)

    cond_cols = ["assay", "development_stage", "disease", "uicc_stage", "cell_type"]
    res = run_conditional_distribution_evaluation(
        gt_path=str(gt_path),
        gen_path=str(gen_path),
        output_dir=str(tmp_path / "eval_out"),
        cond_cols=cond_cols,
        pca_components=8,
        hvg_top_genes=15,
        top_gene_count=5,
        task_name="lung_cancer",
    )

    summary = res["report"]["summary"]
    assert summary["n_seen_conditions"] == 4
    assert summary["n_unseen_conditions"] == 4
    assert "2-Wasserstein_PCA" in summary["seen"]
    assert "2-Wasserstein_PCA" in summary["unseen"]
    assert summary["macro"]["n_macro_conditions"] == 4
    per_cond_df = pd.read_csv(res["per_condition_metrics_path"])
    assert set(per_cond_df["split"].unique()) == {"seen", "unseen"}
    assert Path(res["macro_condition_metrics_path"]).exists()


def test_formal_conditional_generation_with_test_split_and_ckpt_resume(tmp_path):
    obs_train = pd.DataFrame(
        {"cell_type": ["A"] * 12 + ["B"] * 12},
        index=[f"tr_{i}" for i in range(24)],
    )
    obs_test = pd.DataFrame(
        {"cell_type": ["A"] * 6 + ["B"] * 6},
        index=[f"te_{i}" for i in range(12)],
    )
    train_adata = _make_synthetic_adata(n_obs=24, n_vars=32, obs_df=obs_train, seed=3, pad_tail=8)
    test_adata = _make_synthetic_adata(n_obs=12, n_vars=32, obs_df=obs_test, seed=4, pad_tail=8)

    train_path = tmp_path / "train.h5ad"
    test_path = tmp_path / "test.h5ad"
    train_adata.write_h5ad(train_path)
    test_adata.write_h5ad(test_path)

    out_dir = tmp_path / "cond_gen_out"
    report1 = run_formal_conditional_generation(
        data_path=str(train_path),
        test_data_path=str(test_path),
        eval_split="all",
        output_dir=str(out_dir),
        device_id=-1,
        cond_cols=["cell_type"],
        task_name="synth_single",
        gene_block_size=40,
        latent_block_size=40,
        vae_epoch=1,
        fm_epoch=1,
        batch_size=12,
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        num_heads=2,
        n_fm_blocks=1,
        sample_steps=3,
    )

    assert not report1["vae_loaded_from_ckpt"]
    assert not report1["fm_loaded_from_ckpt"]
    assert Path(report1["vae_ckpt_path"]).exists()
    assert Path(report1["fm_ckpt_path"]).exists()
    gen_adata = ad.read_h5ad(report1["generation_path"])
    target_adata = ad.read_h5ad(report1["target_path"])
    # pad_0..pad_7 trimmed -> 32 real genes; all train (24) + test (12) = 36 cells
    assert gen_adata.shape == (36, 32)
    assert target_adata.shape == (36, 32)

    # Resume both VAE + FM checkpoints on second call
    report2 = run_formal_conditional_generation(
        data_path=str(train_path),
        test_data_path=str(test_path),
        eval_split="test",
        output_dir=str(out_dir),
        device_id=-1,
        cond_cols=["cell_type"],
        task_name="synth_single",
        gene_block_size=40,
        latent_block_size=40,
        vae_epoch=1,
        fm_epoch=1,
        batch_size=12,
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        num_heads=2,
        n_fm_blocks=1,
        sample_steps=3,
    )
    assert report2["vae_loaded_from_ckpt"]
    assert report2["fm_loaded_from_ckpt"]
    assert report2["gen_shape"] == [12, 32]


def test_save_load_ckpt_weights_only_compat(tmp_path):
    adata = _make_synthetic_adata(n_obs=16, n_vars=40, seed=5)
    model = SAVE(
        adata=adata,
        device=torch.device("cpu"),
        condition_cols=["batch", "cell_type"],
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        gene_block_size=40,
        latent_block_size=40,
    )
    model.is_trained = True
    ckpt = tmp_path / "model.pt"
    model.save_ckpt(str(ckpt))
    model.is_trained = False
    model.load_ckpt(str(ckpt))
    assert model.is_trained


def test_perturbation_wd_mmd_and_loo_summary(tmp_path):
    rows = []
    for ct in ("CD8T", "B"):
        for cond in ("control", "stimulated"):
            for i in range(10):
                rows.append({"cell_type": ct, "condition": cond})
    obs = pd.DataFrame(rows, index=[f"pcell_{i}" for i in range(len(rows))])
    adata = _make_synthetic_adata(n_obs=40, n_vars=40, obs_df=obs, seed=19)
    data_path = tmp_path / "perturb.h5ad"
    adata.write_h5ad(data_path)

    # Direct evaluate_perturbation check
    test_raw = adata[adata.obs["cell_type"] == "CD8T"].copy()
    pred_adata = test_raw[test_raw.obs["condition"] == "control"].copy()
    direct = evaluate_perturbation(
        pred_data=pred_adata,
        test_raw=test_raw,
        train_scaler=None,
        condition_key="condition",
        target_condition="stimulated",
        data_scale=False,
        n_pca_components=8,
        n_wd_mmd_repeats=3,
    )
    assert {"r2", "mse", "pcc", "wd", "mmd"} == set(direct.keys())
    assert direct["wd"] >= 0.0 and np.isfinite(direct["mmd"])

    out_dir = tmp_path / "perturb_loo"
    summary = run_leave_one_out(
        data_path=str(data_path),
        output_dir=str(out_dir),
        device_id=-1,
        train_with_ctrl=False,
        transfer_batch_size=8,
        train_config={"epoch": 1, "batch_size": 16, "lr": 1e-4, "weight_decay": 2.5e-5, "iter": 10, "is_lr_scheduler": True},
        model_config={
            "kl_scale": 0.5,
            "hidden_dim": 16,
            "enc_dim": 16,
            "expand_dim": 32,
            "enc_num_heads": 2,
            "dec_num_heads": 2,
            "enc_depth": 1,
            "dec_depth": 1,
            "gene_block_size": 40,
            "latent_block_size": 40,
            "is_flash_attn": False,
        },
        sample_mask_prob=0.0,
        col_msk_threshold=0.6,
        n_pca_components=8,
        n_wd_mmd_repeats=2,
    )
    assert {"r2", "mse", "pcc", "wd", "mmd"}.issubset(summary["average_metrics"].keys())
    for ct in ("CD8T", "B"):
        assert {"r2", "mse", "pcc", "wd", "mmd"}.issubset(summary["metrics_per_cell_type"][ct].keys())
    disk_summary = json.loads((out_dir / "summary.json").read_text())
    assert "wd" in disk_summary["average_metrics"]
    assert "mmd" in disk_summary["average_metrics"]


def test_reproduce_iclr2026_dry_run_and_mocked_summary(tmp_path, monkeypatch):
    repro = _load_repro_module()
    assert set(repro.PAPER_TARGETS.keys()) == {"table1", "table2", "table3", "table4", "table5"}
    assert repro.PAPER_TARGETS["table4"]["lung_atlas_bec"]["scib_score"] == pytest.approx(0.8128)
    assert repro.PAPER_TARGETS["table5"]["pbmc_ifn"]["Average"]["wd"] == pytest.approx(5.15)

    dry = repro.run_reproduction(
        tasks="table1,table2,table3,table4,table5",
        data_root=tmp_path / "data",
        output_root=tmp_path / "out_dry",
        device=-1,
        dry_run=True,
    )
    assert Path(dry["summary_json_path"]).exists()
    assert Path(dry["summary_md_path"]).exists()
    md_text = Path(dry["summary_md_path"]).read_text()
    assert "## TABLE1" in md_text and "## TABLE5" in md_text

    # Mocked pipeline run with dataset filtering to test orchestration & delta calculation
    data_root = tmp_path / "mock_data"
    for ds in ("pbmc3k", "heart_bec", "pbmc_ifn"):
        for path in repro.get_output_paths(ds, data_root).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub")

    monkeypatch.setattr(
        repro,
        "run_formal_conditional_generation",
        lambda **kw: {"target_path": "t.h5ad", "generation_path": "g.h5ad"},
    )
    monkeypatch.setattr(
        repro,
        "run_conditional_distribution_evaluation",
        lambda **kw: {
            "report_path": "eval.json",
            "report": {
                "summary": {
                    "primary_metrics": {
                        "2-Wasserstein_PCA": {"mean": 19.0},
                        "RBF_MMD_PCA": {"mean": 1.8},
                    },
                    "expression_metrics": {
                        "mse": {"mean": 0.08},
                        "pcc": {"mean": 0.93},
                        "r2": {"mean": 0.84},
                    },
                }
            },
        },
    )
    monkeypatch.setattr(
        repro,
        "run_bec_integration",
        lambda **kw: {
            "report_path": "bec.json",
            "scib_csv_path": "scib.csv",
            "report": {"summary": {"avg_bio": {"mean": 0.75}, "avg_batch": {"mean": 0.86}, "scib_score": {"mean": 0.794}}},
        },
    )
    monkeypatch.setattr(
        repro,
        "run_leave_one_out",
        lambda **kw: {
            "metrics_per_cell_type": {"CD8T": {"pcc": 0.98, "r2": 0.97, "wd": 4.0, "mmd": 0.1, "mse": 0.0}},
            "average_metrics": {"pcc": 0.96, "r2": 0.86, "wd": 5.15, "mmd": 0.14, "mse": 0.0},
        },
    )

    out_mock = tmp_path / "out_mock"
    res = repro.run_reproduction(
        tasks="table1,table4,table5",
        datasets="pbmc3k,heart_bec,pbmc_ifn",
        data_root=data_root,
        output_root=out_mock,
        device=-1,
        skip_prepare=True,
    )
    assert res["runs"]["table1"]["pbmc3k"]["delta"]["wd"] == pytest.approx(-0.14)
    assert res["runs"]["table4"]["heart_bec"]["observed"]["scib_score"] == pytest.approx(0.794)
    assert res["runs"]["table5"]["pbmc_ifn"]["delta"]["Average"]["wd"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "script_rel",
    [
        "examples/bec_integration/run_bec.py",
        "examples/conditional_generation/run_formal.py",
        "examples/perturbation_prediction/run_single.py",
        "examples/perturbation_prediction/run_leave_one_out.py",
        "examples/reproduce_iclr2026.py",
    ],
)
def test_example_cli_help(script_rel):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script_rel), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_ditblock_sdpa_and_single_attention_call():
    from sc_save_public.models.VAE_model import Attention, DiTblock

    torch.manual_seed(0)
    attn = Attention(dim=16, num_heads=2, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
    attn.eval()
    x = torch.randn(4, 5, 16)
    with torch.no_grad():
        out_sdpa = attn(x, is_output_attn_weight=False)
        out_manual, _ = attn(x, is_output_attn_weight=True)
    assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-5)

    blk = DiTblock(feature_dim=16, types_of_conds=2, num_heads=2)
    blk.eval()
    call_count = {"n": 0}
    orig_attn_forward = blk.attn.forward

    def _counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return orig_attn_forward(*args, **kwargs)

    blk.attn.forward = _counting_forward
    c = torch.randn(4, 2, 16)
    with torch.no_grad():
        out_blk = blk(x, c)
    assert out_blk.shape == x.shape
    assert call_count["n"] == 1


def test_precomputed_latent_fm_train_matches_posterior_and_is_finite():
    from torch.utils.data import DataLoader, TensorDataset
    from sc_save_public.models.flow_matching_train import flow_matching_train
    from sc_save_public.models.fm.denoising_model import MLPTimeStep
    from sc_save_public.models.fm.fm import FM

    adata = _make_synthetic_adata(n_obs=20, n_vars=40, seed=21)
    model = SAVE(
        adata=adata,
        device=torch.device("cpu"),
        condition_cols=["batch", "cell_type"],
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        gene_block_size=40,
        latent_block_size=40,
    )
    denoising = MLPTimeStep(
        in_dim=16,
        conditional=True,
        seq_length=1,
        hidden_dim=64,
        embedding_dim=64,
        num_conds=2,
        dropout_prob=0.0,
        n_blocks=1,
        label_encoder_dict=model.label_encoder_dict,
        mask_id=model.mask_id,
        block_type="DiT",
        num_heads=2,
    )
    fm = FM(denoising_model=denoising, device=torch.device("cpu"))

    x_tensor = torch.rand(20, 40)
    y_tensor = torch.ones(20, 2, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=8, shuffle=True)

    # Confirm encoder dropout is off in eval() so precomputed (mu, var) is deterministic
    model.model.eval()
    mask_emb = torch.zeros(20, 2, 32)
    with torch.no_grad():
        _, mu1, var1 = model.model.encoder(x_tensor, mask_emb)
        _, mu2, var2 = model.model.encoder(x_tensor, mask_emb)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(var1, var2)

    losses = flow_matching_train(
        vae=model.model,
        fm=fm,
        dataloader=loader,
        num_epoch=2,
        device=torch.device("cpu"),
        lr=1e-4,
        weight_decay=2.5e-5,
        seed=1202,
        is_emb_ary=False,
    )
    assert "loss" in losses and np.isfinite(losses["loss"])


def test_parallel_eval_workers_matches_serial(tmp_path):
    obs = pd.DataFrame(
        {
            "batch": ["b1"] * 12 + ["b2"] * 12,
            "cell_type": (["T", "T", "T", "B", "B", "B"] * 4),
        },
        index=[f"c_{i}" for i in range(24)],
    )
    gt = _make_synthetic_adata(n_obs=24, n_vars=20, obs_df=obs.copy(), seed=31)
    gen = _make_synthetic_adata(n_obs=24, n_vars=20, obs_df=obs.copy(), seed=32)
    gt_path = tmp_path / "gt.h5ad"
    gen_path = tmp_path / "gen.h5ad"
    gt.write_h5ad(gt_path)
    gen.write_h5ad(gen_path)

    res_serial = run_conditional_distribution_evaluation(
        gt_path=str(gt_path),
        gen_path=str(gen_path),
        output_dir=str(tmp_path / "eval_s"),
        pca_components=5,
        include_expression_metrics=False,
        eval_workers=1,
    )
    res_parallel = run_conditional_distribution_evaluation(
        gt_path=str(gt_path),
        gen_path=str(gen_path),
        output_dir=str(tmp_path / "eval_p"),
        pca_components=5,
        include_expression_metrics=False,
        eval_workers=2,
    )
    df_s = pd.read_csv(res_serial["per_condition_metrics_path"])
    df_p = pd.read_csv(res_parallel["per_condition_metrics_path"])
    pd.testing.assert_frame_equal(df_s, df_p, rtol=1e-5, atol=1e-5)


def test_parallel_scib_workers_matches_serial(tmp_path):
    pytest.importorskip("scib")
    adata = _make_synthetic_adata(n_obs=48, n_vars=40, seed=41)
    data_path = tmp_path / "bec.h5ad"
    adata.write_h5ad(data_path)

    common = dict(
        data_path=str(data_path),
        task_name="par_bec",
        device_id=-1,
        epoch=1,
        batch_size=16,
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        gene_block_size=40,
        latent_block_size=40,
        pca_components=8,
        transfer_batch_size=16,
        seed=1202,
    )
    ckpt_dir = tmp_path / "shared_ckpt"
    # Train once so both serial and parallel runs start from identical loaded checkpoint & RNG state
    run_bec_integration(
        output_dir=str(tmp_path / "out_warmup"),
        ckpt_dir=str(ckpt_dir),
        scib_workers=1,
        run_scib=False,
        **common,
    )
    res_s = run_bec_integration(
        output_dir=str(tmp_path / "out_s"),
        ckpt_dir=str(ckpt_dir),
        scib_workers=1,
        **common,
    )
    res_p = run_bec_integration(
        output_dir=str(tmp_path / "out_p"),
        ckpt_dir=str(ckpt_dir),
        scib_workers=2,
        **common,
    )
    assert res_p["report"]["batches"] == res_s["report"]["batches"]
    for b in res_s["report"]["batches"]:
        np.testing.assert_allclose(
            np.load(res_s["embedding_paths"][b]),
            np.load(res_p["embedding_paths"][b]),
            rtol=1e-5,
            atol=1e-5,
        )
    for metric in ("avg_bio", "avg_batch", "scib_score", "PCR_batch"):
        assert res_p["report"]["summary"][metric]["mean"] == pytest.approx(
            res_s["report"]["summary"][metric]["mean"], abs=1e-4
        )


def test_parallel_perturbation_loo_matches_serial_and_oom_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sc_save_public.perturbation_prediction as pp

    # Verify VRAM-tier worker resolution helper
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda dev: SimpleNamespace(total_memory=96 * (1024**3)),
    )
    assert pp.resolve_perturbation_workers("auto", torch.device("cuda:0"), 7) == 7
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda dev: SimpleNamespace(total_memory=48 * (1024**3)),
    )
    assert pp.resolve_perturbation_workers("auto", torch.device("cuda:0"), 7) == 4
    assert pp.resolve_perturbation_workers("auto", torch.device("cpu"), 7) == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    rows = []
    for ct in ("CD8T", "B"):
        for cond in ("control", "stimulated"):
            for _ in range(10):
                rows.append({"cell_type": ct, "condition": cond})
    obs = pd.DataFrame(rows, index=[f"pc_{i}" for i in range(len(rows))])
    adata = _make_synthetic_adata(n_obs=40, n_vars=40, obs_df=obs, seed=55)
    data_path = tmp_path / "perturb.h5ad"
    adata.write_h5ad(data_path)

    common = dict(
        data_path=str(data_path),
        device_id=-1,
        train_with_ctrl=False,
        transfer_batch_size=8,
        train_config={"epoch": 1, "batch_size": 16, "lr": 1e-4, "weight_decay": 2.5e-5, "iter": 10, "is_lr_scheduler": True},
        model_config={
            "kl_scale": 0.5,
            "hidden_dim": 16,
            "enc_dim": 16,
            "expand_dim": 32,
            "enc_num_heads": 2,
            "dec_num_heads": 2,
            "enc_depth": 1,
            "dec_depth": 1,
            "gene_block_size": 40,
            "latent_block_size": 40,
            "is_flash_attn": False,
        },
        sample_mask_prob=0.0,
        col_msk_threshold=0.6,
        n_pca_components=8,
        n_wd_mmd_repeats=2,
    )
    res_s = run_leave_one_out(output_dir=str(tmp_path / "loo_s"), workers=1, **common)
    res_p = run_leave_one_out(output_dir=str(tmp_path / "loo_p"), workers=2, **common)
    for k in ("r2", "mse", "pcc", "wd", "mmd"):
        assert res_p["average_metrics"][k] == pytest.approx(res_s["average_metrics"][k], abs=1e-4)

    # Simulate worker OOM exception -> serial retry succeeds cleanly
    class _FakeFuture:
        def __init__(self, fail_oom: bool):
            self.fail_oom = fail_oom

        def result(self):
            if self.fail_oom:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return ("CD8T", {"r2": 0.9, "mse": 0.01, "pcc": 0.95, "wd": 4.0, "mmd": 0.1})

    class _FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, kw):
            return _FakeFuture(fail_oom=(kw["cell_type"] == "B"))

    serial_called = []
    monkeypatch.setattr(pp, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(
        pp,
        "run_perturbation",
        lambda **kw: (serial_called.append(kw["cell_type"]), {"metrics": {"r2": 0.8, "mse": 0.02, "pcc": 0.9, "wd": 5.0, "mmd": 0.2}})[1],
    )
    res_retry = run_leave_one_out(output_dir=str(tmp_path / "loo_oom"), workers=2, **common)
    assert serial_called == ["B"]
    assert res_retry["metrics_per_cell_type"]["B"]["r2"] == pytest.approx(0.8)


def test_reproduce_iclr2026_parallel_scheduler_and_oom_retry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    repro = _load_repro_module()

    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        repro.torch.cuda,
        "get_device_properties",
        lambda dev: SimpleNamespace(total_memory=95 * (1024**3)),
    )
    assert repro.resolve_gpu_workers("auto", 0, [("table1", "pbmc3k"), ("table4", "heart_bec")]) == 4
    assert repro.resolve_gpu_workers("auto", 0, [("table4", "heart_bec"), ("table5", "pbmc_ifn")]) == 6
    monkeypatch.setattr(
        repro.torch.cuda,
        "get_device_properties",
        lambda dev: SimpleNamespace(total_memory=40 * (1024**3)),
    )
    assert repro.resolve_gpu_workers("auto", 0, [("table1", "pbmc3k")]) == 2
    monkeypatch.setattr(repro.torch.cuda, "is_available", lambda: False)
    assert repro.resolve_gpu_workers("auto", 0, [("table1", "pbmc3k")]) == 1

    data_root = tmp_path / "data"
    for ds in ("pbmc3k", "dentategyrus"):
        for path in repro.get_output_paths(ds, data_root).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub")

    subproc_calls = []

    def _fake_subprocess_run(cmd, stdout=None, stderr=None, text=False, check=False):
        ds_idx = cmd.index("--datasets") + 1
        ds_name = cmd[ds_idx]
        res_idx = cmd.index("--results-dir") + 1
        child_res_dir = Path(cmd[res_idx])
        subproc_calls.append(ds_name)
        if ds_name == "dentategyrus":
            return SimpleNamespace(returncode=1, stdout="torch.cuda.OutOfMemoryError: CUDA out of memory")
        child_res_dir.mkdir(parents=True, exist_ok=True)
        (child_res_dir / "iclr2026_summary.json").write_text(
            json.dumps(
                {
                    "runs": {
                        "table1": {
                            "pbmc3k": {
                                "status": "completed",
                                "observed": {"wd": 19.1, "mmd": 1.8, "mse": 0.08, "pcc": 0.93, "r2": 0.84},
                                "target": repro.PAPER_TARGETS["table1"]["pbmc3k"],
                                "delta": {"wd": -0.04},
                            }
                        }
                    }
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="ok")

    serial_retried = []
    monkeypatch.setattr(repro.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(
        repro,
        "_run_cond_gen_and_eval",
        lambda **kw: (
            serial_retried.append(kw["dataset_name"]),
            {
                "status": "completed",
                "observed": {"wd": 9.2, "mmd": 0.17, "mse": 0.0, "pcc": 0.99, "r2": 0.97},
                "target": repro.PAPER_TARGETS["table1"]["dentategyrus"],
                "delta": {"wd": 0.04},
            },
        )[1],
    )

    out = repro.run_reproduction(
        tasks="table1",
        datasets="pbmc3k,dentategyrus",
        data_root=data_root,
        output_root=tmp_path / "out_par",
        device=-1,
        skip_prepare=True,
        gpu_workers=2,
        cpu_workers=2,
    )
    assert set(subproc_calls) == {"pbmc3k", "dentategyrus"}
    assert serial_retried == ["dentategyrus"]
    assert out["runs"]["table1"]["pbmc3k"]["observed"]["wd"] == pytest.approx(19.1)
    assert out["runs"]["table1"]["dentategyrus"]["observed"]["wd"] == pytest.approx(9.2)
    assert Path(out["summary_json_path"]).exists()
