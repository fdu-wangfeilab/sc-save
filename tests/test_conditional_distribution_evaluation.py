import json

import anndata as ad
import numpy as np
import pandas as pd

from sc_save_public.conditional_distribution_evaluation import (
    run_conditional_distribution_evaluation,
)


def test_run_conditional_distribution_evaluation_writes_outputs(tmp_path):
    obs = pd.DataFrame(
        {
            "batch": ["b1", "b1", "b2", "b2"],
            "cell_type": ["t1", "t1", "t2", "t2"],
        },
        index=[f"cell_{idx}" for idx in range(4)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])

    gt = ad.AnnData(
        X=np.array(
            [
                [1.0, 0.0, 0.5],
                [1.1, 0.1, 0.4],
                [0.0, 1.0, 0.2],
                [0.1, 1.1, 0.3],
            ]
        ),
        obs=obs.copy(),
        var=var.copy(),
    )
    gen = ad.AnnData(
        X=np.array(
            [
                [1.0, 0.0, 0.45],
                [1.05, 0.05, 0.35],
                [0.05, 1.0, 0.25],
                [0.15, 1.05, 0.2],
            ]
        ),
        obs=obs.copy(),
        var=var.copy(),
    )

    gt_path = tmp_path / "target.h5ad"
    gen_path = tmp_path / "generated.h5ad"
    gt.write_h5ad(gt_path)
    gen.write_h5ad(gen_path)

    output_dir = tmp_path / "output"
    result = run_conditional_distribution_evaluation(
        gt_path=str(gt_path),
        gen_path=str(gen_path),
        output_dir=str(output_dir),
        include_expression_metrics=False,
        pca_components=2,
    )

    alignment = pd.read_csv(output_dir / "distribution_alignment.csv")
    metrics = pd.read_csv(output_dir / "per_condition_metrics.csv")
    report = json.loads((output_dir / "report.json").read_text())

    assert result["distribution_alignment_path"].endswith("distribution_alignment.csv")
    assert alignment["is_equal"].tolist() == [True, True]
    assert sorted(metrics.columns.tolist()) == sorted(
        [
            "batch",
            "cell_type",
            "cell_count",
            "1-Wasserstein_PCA",
            "2-Wasserstein_PCA",
            "Linear_MMD_unbiased_PCA",
            "RBF_MMD_PCA",
        ]
    )
    assert report["distribution"]["exact_match_conditions"] == 2
    assert report["distribution"]["all_conditions_match"] is True
