"""Attention VAE encoder, decoder, and legacy checkpoint loader for SAVE."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from sc_save.nn.blocks import DiTBlock


def remap_legacy_state_dict(
    state_dict: Mapping[str, Any],
    target_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Remap legacy checkpoint typos (``tokenzier`` <-> ``tokenizer``, ``pos_embding`` <-> ``pos_embedding``)."""
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if target_keys is not None and key in target_keys:
            remapped[key] = value
            continue

        to_v2 = key.replace("tokenzier", "tokenizer").replace("pos_embding", "pos_embedding")
        if target_keys is None or to_v2 in target_keys:
            remapped[to_v2] = value
            continue

        to_v1 = key.replace("tokenizer", "tokenzier").replace("pos_embedding", "pos_embding")
        if to_v1 in target_keys:
            remapped[to_v1] = value
            continue

        remapped[key] = value
    return remapped


def load_state_dict_compat(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Any:
    """Load a ``state_dict`` into ``module``, remapping v1/v2 key spellings seamlessly."""
    target_keys = set(module.state_dict().keys())
    remapped = remap_legacy_state_dict(state_dict, target_keys=target_keys)
    return nn.Module.load_state_dict(module, remapped, strict=strict)


class AttentionEncoder(nn.Module):
    """Block-tokenized Transformer encoder outputting posterior ``(z, mu, var)``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        enc_dim: int,
        bn_affine: bool = True,
        norm_type: str = "bn",
        expand_dim: int = 32,
        is_reshape_first: bool = False,
        gene_block_size: int = 40,
        is_linear: bool = False,
        dropout: float = 0.2,
        is_enc_cond: bool = False,
        types_of_conds: int = 2,
        **kwargs,
    ) -> None:
        if kwargs.get("is_linear_replace_attn", False):
            raise ValueError("Linear replacement encoder is removed in SAVE v2.")
        super().__init__()
        self.expand_dim = expand_dim
        self.types_of_conds = types_of_conds
        self.is_enc_cond = is_enc_cond
        self.is_reshape_first = is_reshape_first

        if is_reshape_first:
            if gene_block_size <= 0 or input_dim % gene_block_size != 0:
                raise ValueError(
                    f"input_dim ({input_dim}) must be a positive multiple of gene_block_size ({gene_block_size})."
                )
            num_gene_block = input_dim // gene_block_size
            self.tokenizer = nn.Sequential(
                Rearrange("b (m n) -> b m n", m=num_gene_block, n=gene_block_size),
                nn.Linear(gene_block_size, expand_dim),
            )
            self.pos_embedding = nn.Parameter(torch.zeros(num_gene_block, expand_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        else:
            self.tokenizer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * expand_dim),
                Rearrange("c (h e) -> c h e", h=hidden_dim, e=expand_dim),
            )
            self.pos_embedding = nn.Parameter(torch.zeros(hidden_dim, expand_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.blks = nn.ModuleList()
        if is_enc_cond:
            for _ in range(depth):
                self.blks.append(
                    DiTBlock(
                        feature_dim=expand_dim,
                        mlp_ratio=4.0,
                        num_heads=num_heads,
                        dropout=dropout,
                        types_of_conds=types_of_conds,
                    )
                )

        # Preserve v1 module creation order (var_enc before mu_enc) for RNG parity.
        if is_reshape_first:
            self.var_enc = nn.Sequential(
                nn.Linear(expand_dim, expand_dim * 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(expand_dim * 4, enc_dim),
            )
            self.mu_enc = nn.Sequential(
                nn.Linear(expand_dim, expand_dim * 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(expand_dim * 4, enc_dim),
            )
        else:
            self.var_enc = nn.Sequential(
                Rearrange("c h e -> c (h e)"),
                nn.Linear(expand_dim * hidden_dim, enc_dim),
            )
            self.mu_enc = nn.Sequential(
                Rearrange("c h e -> c (h e)"),
                nn.Linear(expand_dim * hidden_dim, enc_dim),
            )

    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.sqrt(var)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.tokenizer(x) + self.pos_embedding

        if self.is_enc_cond:
            for blk in self.blks:
                h = blk(h, c)
        else:
            for blk in self.blks:
                h = blk(h)

        mu = self.mu_enc(h)
        var = torch.clamp(torch.exp(self.var_enc(h)), min=1e-20)
        z = self.reparameterize(mu, var)
        return z, mu, var


class AttentionDecoder(nn.Module):
    """AdaLN-Zero Transformer decoder mapping latents + condition embeddings to gene space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        out_dim: int,
        expand_dim: int = 32,
        is_norm_init: bool = False,
        block_type: str = "adaLN",
        len_col_comb: int | None = None,
        types_of_cond: int = 2,
        is_reshape_first: bool = False,
        latent_block_size: int = 4,
        out_layer_dropout: float = 0.4,
        is_linear: bool = False,
        attn_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        if kwargs.get("is_linear_replace_attn", False):
            raise ValueError("Linear replacement decoder is removed in SAVE v2.")
        if block_type != "adaLN":
            raise ValueError(f"Unsupported decoder block_type={block_type!r}; expected 'adaLN'.")
        super().__init__()
        self.block_type = block_type
        self.blks = nn.ModuleList()
        self.input_dim = input_dim
        self.expand_dim = expand_dim
        self.types_of_cond = types_of_cond
        self.is_reshape_first = is_reshape_first

        if is_reshape_first:
            if latent_block_size <= 0 or out_dim % latent_block_size != 0:
                raise ValueError(
                    f"out_dim ({out_dim}) must be a positive multiple of latent_block_size ({latent_block_size})."
                )
            num_latent_block = out_dim // latent_block_size
            self.reshape_layer = nn.Sequential(
                nn.Linear(input_dim, expand_dim),
            )
            self.pos_embedding = nn.Parameter(
                torch.zeros(num_latent_block, expand_dim)
            )
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        else:
            self.reshape_layer = nn.Sequential(
                Rearrange("b h -> b h 1"),
                nn.Linear(1, expand_dim),
            )
            self.pos_embedding = nn.Parameter(torch.zeros(hidden_dim, expand_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        for _ in range(depth):
            self.blks.append(
                DiTBlock(
                    feature_dim=expand_dim,
                    mlp_ratio=4.0,
                    num_heads=num_heads,
                    types_of_conds=types_of_cond,
                    dropout=attn_dropout,
                )
            )

        if is_reshape_first:
            self.out_layer = nn.Sequential(
                nn.Linear(expand_dim, out_dim // num_latent_block, bias=False),
                Rearrange("b m n -> b (m n)"),
                nn.Sigmoid(),
            )
        else:
            self.out_layer = nn.Sequential(
                Rearrange("c h e -> c (h e)"),
                nn.Linear(hidden_dim * expand_dim, out_dim),
                nn.Sigmoid(),
            )

        if is_norm_init:
            self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        for block in self.blks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        return_logits: bool = False,
        is_output_attn_weight: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        h = self.reshape_layer(x) + self.pos_embedding

        if is_output_attn_weight:
            attn_weights = []
            for blk in self.blks:
                h, attn_weight = blk(h, c, is_output_attn_weight=True)
                attn_weights.append(attn_weight)
            return self.out_layer(h), attn_weights

        for blk in self.blks:
            h = blk(h, c)

        if return_logits:
            return self.out_layer(h), h
        return self.out_layer(h)


class VAE(nn.Module):
    """Conditional block-attention VAE used as the first stage of SAVE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        enc_num_heads: int,
        dec_num_heads: int,
        enc_depth: int,
        dec_depth: int,
        enc_dim: int = 10,
        expand_dim: int = 32,
        enc_affine: bool = True,
        enc_norm_type: str = "bn",
        dec_norm_init: bool = False,
        dec_blk_type: str = "adaLN",
        is_initialize: bool = False,
        len_col_comb: int | None = None,
        mask_id: int = 0,
        label_encoder_dict: Mapping[str, Any] | None = None,
        is_multi_codebook: bool = False,
        is_reshape_first: bool = False,
        gene_block_size: int = 40,
        latent_block_size: int = 4,
        out_layer_dropout: float = 0.4,
        is_enc_linear: bool = False,
        is_dec_linear: bool = False,
        enc_attn_dropout: float = 0.2,
        dec_attn_dropout: float = 0.0,
        is_enc_cond: bool = False,
        outer_emb_dim: Sequence[int] | int = (768, 256),
        total_cond_classes: int | None = None,
        total_classes: int | None = None,
        types_of_conds: int | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get("is_linear_replace_attn", False):
            raise ValueError("Linear replacement VAE is removed in SAVE v2.")
        super().__init__()
        if total_cond_classes is None and total_classes is not None:
            total_cond_classes = total_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = enc_dim
        self.enc_heads = enc_num_heads
        self.dec_heads = dec_num_heads
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_affine = enc_affine
        self.enc_norm_type = enc_norm_type
        self.expand_dim = expand_dim
        self.dec_blk_type = dec_blk_type
        self.is_multi_codebook = is_multi_codebook
        self.mask_id = mask_id

        if label_encoder_dict is not None:
            self.types_of_conds = len(label_encoder_dict)
        elif types_of_conds is not None:
            self.types_of_conds = int(types_of_conds)
        elif len_col_comb is not None:
            self.types_of_conds = int(len_col_comb)
        else:
            raise ValueError("Provide either `label_encoder_dict` or `types_of_conds`.")

        if len_col_comb:
            col_comb: list[Any] = []
            for i in range(1, len_col_comb + 1):
                combs = list(combinations(range(len_col_comb), i))
                col_comb.extend(combs)
            self.col_comb = [list(x) if len(x) > 1 else x[0] for x in col_comb]

        if is_multi_codebook:
            if label_encoder_dict is None:
                raise ValueError("`label_encoder_dict` is required when `is_multi_codebook=True`.")
            self.cond_embedder: nn.Module = nn.ModuleList(
                [
                    nn.Embedding(len(le.classes_), expand_dim)
                    for le in label_encoder_dict.values()
                ]
            )
        else:
            if total_cond_classes is not None:
                total_class = int(total_cond_classes)
            elif label_encoder_dict is not None:
                total_class = sum(len(le.classes_) for le in label_encoder_dict.values())
                total_class -= len(label_encoder_dict) - 1
            else:
                raise ValueError("Provide either `label_encoder_dict` or `total_cond_classes`.")
            self.cond_embedder = nn.Embedding(total_class, expand_dim)

        if isinstance(outer_emb_dim, int):
            outer_dims = (outer_emb_dim,)
        else:
            outer_dims = tuple(int(d) for d in outer_emb_dim)
        self.outer_emb_dim = list(outer_dims)
        self.txt_emb_proj = nn.ModuleList(
            [nn.Linear(dim, expand_dim) for dim in self.outer_emb_dim]
        )

        self.encoder = AttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            expand_dim=expand_dim,
            num_heads=enc_num_heads,
            depth=enc_depth,
            enc_dim=enc_dim,
            bn_affine=enc_affine,
            norm_type=enc_norm_type,
            is_reshape_first=is_reshape_first,
            gene_block_size=gene_block_size,
            is_linear=is_enc_linear,
            dropout=enc_attn_dropout,
            is_enc_cond=is_enc_cond,
            types_of_conds=self.types_of_conds,
        )

        self.decoder = AttentionDecoder(
            input_dim=enc_dim,
            hidden_dim=enc_dim,
            expand_dim=expand_dim,
            num_heads=dec_num_heads,
            depth=dec_depth,
            out_dim=input_dim,
            is_norm_init=dec_norm_init,
            block_type=dec_blk_type,
            len_col_comb=len_col_comb,
            is_reshape_first=is_reshape_first,
            latent_block_size=latent_block_size,
            out_layer_dropout=out_layer_dropout,
            is_linear=is_dec_linear,
            attn_dropout=dec_attn_dropout,
            types_of_cond=self.types_of_conds,
        )

        # Preserve exact v1 init order: AttentionDecoder.initialize_weights() then VAE.initialize_weights().
        if is_initialize:
            self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        target_keys = set(self.state_dict().keys())
        remapped = remap_legacy_state_dict(state_dict, target_keys=target_keys)
        return super().load_state_dict(remapped, strict=strict, assign=assign)


__all__ = [
    "AttentionDecoder",
    "AttentionEncoder",
    "VAE",
    "load_state_dict_compat",
    "remap_legacy_state_dict",
]
