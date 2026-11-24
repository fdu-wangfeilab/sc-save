"""Compatibility package re-exporting core ``sc_save`` APIs for v1 callers."""

from __future__ import annotations

from sc_save import (
    ConditionVocabulary,
    FMConfig,
    GeneBlockResult,
    MaxAbsExpressionScaler,
    SAVE,
    SAVEConfig,
    TrainConfig,
    VAEConfig,
    __version__,
    build_gene_blocks,
    load_gene_embeddings,
    preprocess_adata,
)

__all__ = [
    "ConditionVocabulary",
    "FMConfig",
    "GeneBlockResult",
    "MaxAbsExpressionScaler",
    "SAVE",
    "SAVEConfig",
    "TrainConfig",
    "VAEConfig",
    "__version__",
    "build_gene_blocks",
    "load_gene_embeddings",
    "preprocess_adata",
]
