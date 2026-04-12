import json
import pickle

import anndata as ad
import numpy as np
import pandas as pd

import sc_save_public.gene_block_construction as gbc


def test_run_gene_block_construction_writes_expected_outputs(tmp_path, monkeypatch):
    adata = ad.AnnData(
        X=np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        ),
        obs=pd.DataFrame(index=["cell_1", "cell_2"]),
        var=pd.DataFrame(index=["g1", "g2", "g3", "g4", "g5"]),
    )

    data_path = tmp_path / "toy.h5ad"
    adata.write_h5ad(data_path)

    gene_embeddings = {
        "g1": np.array([0.0, 0.0]),
        "g2": np.array([0.1, 0.0]),
        "g3": np.array([10.0, 0.0]),
        "g4": np.array([10.1, 0.0]),
    }
    embedding_path = tmp_path / "gene_embeddings.pkl"
    with open(embedding_path, "wb") as handle:
        pickle.dump(gene_embeddings, handle)

    def fake_balanced_kmeans_with_ot(**kwargs):
        labels = np.array([0, 0, 1, 1])
        stats = {"iterations": 1, "converged": True}
        return labels, None, stats

    monkeypatch.setattr(gbc, "balanced_kmeans_with_ot", fake_balanced_kmeans_with_ot)

    output_dir = tmp_path / "output"
    report = gbc.run_gene_block_construction(
        data_path=str(data_path),
        gene_embedding_path=str(embedding_path),
        output_dir=str(output_dir),
        gene_block_size=2,
    )

    processed = ad.read_h5ad(output_dir / "processed.h5ad")
    saved_report = json.loads((output_dir / "report.json").read_text())
    var_metadata = pd.read_csv(output_dir / "var_metadata.csv", index_col=0)

    assert report["input_shape"] == [2, 5]
    assert report["output_shape"] == [2, 6]
    assert report["pad_gene_count"] == 1
    assert saved_report["block_sizes"] == {"0": 2, "1": 2}
    assert processed.var_names.tolist() == ["g1", "g2", "g3", "g4", "g5", "pad_0"]
    assert var_metadata.loc["g1", "gene_block_role"] == "semantic_block"
    assert var_metadata.loc["g5", "gene_block_role"] == "remainder"
    assert bool(var_metadata.loc["pad_0", "is_pad"])
