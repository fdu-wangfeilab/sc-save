"""Compatibility shim re-exporting VAE building blocks from ``sc_save.nn``."""

from sc_save.nn.blocks import Attention, DiTBlock, DiTblock, Mlp, TimestepEmbedder, modulate
from sc_save.nn.vae import (
    VAE,
    AttentionDecoder,
    AttentionEncoder,
    load_state_dict_compat,
    remap_legacy_state_dict,
)

__all__ = [
    "Attention",
    "AttentionDecoder",
    "AttentionEncoder",
    "DiTBlock",
    "DiTblock",
    "Mlp",
    "TimestepEmbedder",
    "VAE",
    "load_state_dict_compat",
    "modulate",
    "remap_legacy_state_dict",
]
