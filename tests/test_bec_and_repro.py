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
