"""Typed configuration dataclasses for SAVE models and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Mapping


def _filter_fields(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in payload.items() if k in valid}


@dataclass
class VAEConfig:
    """Architecture hyperparameters for the block-tokenized attention VAE."""

    hidden_dim: int = 64
    enc_dim: int = 64
    expand_dim: int = 512
    enc_num_heads: int = 4
    dec_num_heads: int = 4
    enc_depth: int = 3
    dec_depth: int = 3
    enc_affine: bool = True
    enc_norm_type: str = "ln"
    dec_norm_init: bool = True
    dec_blk_type: str = "adaLN"
    is_initialize: bool = True
    is_reshape_first: bool = True
    gene_block_size: int = 3200
    latent_block_size: int = 3200
    is_enc_cond: bool = True
    is_multi_codebook: bool = False
    enc_attn_dropout: float = 0.2
    dec_attn_dropout: float = 0.0
    out_layer_dropout: float = 0.4
    is_flash_attn: bool = False
    outer_emb_dim: tuple[int, ...] = (768, 256)

    def __post_init__(self) -> None:
        if isinstance(self.outer_emb_dim, int):
            self.outer_emb_dim = (int(self.outer_emb_dim),)
        else:
            self.outer_emb_dim = tuple(int(v) for v in self.outer_emb_dim)
        if self.dec_blk_type != "adaLN":
            raise ValueError(f"Unsupported dec_blk_type={self.dec_blk_type!r}; expected 'adaLN'.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outer_emb_dim"] = list(self.outer_emb_dim)
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VAEConfig:
        filtered = _filter_fields(cls, payload)
        if "outer_emb_dim" in filtered:
            val = filtered["outer_emb_dim"]
            filtered["outer_emb_dim"] = (int(val),) if isinstance(val, int) else tuple(int(v) for v in val)
        return cls(**filtered)


@dataclass
class FMConfig:
    """Architecture and sampling hyperparameters for latent Flow Matching."""

    hidden_dim: int = 1024
    embedding_dim: int = 1024
    n_blocks: int = 5
    num_heads: int = 4
    dropout_prob: float = 0.0
    conditioning_probability: float = 0.8
    block_type: Literal["DiT", "Resnet"] = "DiT"
    path_type: Literal["ot", "vp", "ve"] = "ot"
    path_kwargs: dict[str, Any] = field(default_factory=dict)
    sigma: float = 0.1
    use_ot: bool = True
    use_snr_weighting: bool = False
    antithetic_time_sampling: bool = False
    guidance_weight: float = 2.0
    sample_steps: int = 20
    solver: str = "euler"
    use_sde: bool | None = None

    def __post_init__(self) -> None:
        if self.block_type not in ("DiT", "Resnet"):
            raise ValueError(f"Unsupported block_type={self.block_type!r}; expected 'DiT' or 'Resnet'.")
        if self.path_type not in ("ot", "vp", "ve"):
            raise ValueError(f"Unsupported path_type={self.path_type!r}; expected 'ot', 'vp', or 've'.")
        self.path_kwargs = dict(self.path_kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FMConfig:
        return cls(**_filter_fields(cls, payload))


@dataclass
class TrainConfig:
    """Optimization and conditioning schedule hyperparameters."""

    vae_epochs: int = 200
    fm_epochs: int = 200
    batch_size: int = 1024
    lr: float = 1e-4
    fm_lr: float | None = None
    weight_decay: float = 2.5e-5
    kl_scale: float = 0.5
    grad_clip: float = 1.0
    is_lr_scheduler: bool = True
    recon_criterion: Literal["bce", "mse"] = "bce"
    vae_col_mask_prob: float = 1.0
    fm_col_mask_prob: float = 0.0
    is_col_wise_mask: bool = False
    use_amp_vae: bool = False
    use_amp_fm: bool = True
    seed: int = 1202

    def __post_init__(self) -> None:
        if self.recon_criterion not in ("bce", "mse"):
            raise ValueError(
                f"Unsupported recon_criterion={self.recon_criterion!r}; expected 'bce' or 'mse'."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainConfig:
        return cls(**_filter_fields(cls, payload))


@dataclass
class SAVEConfig:
    """Top-level configuration bundle for SAVE."""

    vae: VAEConfig = field(default_factory=VAEConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cond_encoding: Literal["legacy", "offset"] = "legacy"
    mask_token: str = "<mask>"
    pad_to_blocks: bool = True
    scale_data: bool = True
    seed: int = 1202

    def __post_init__(self) -> None:
        if self.cond_encoding not in ("legacy", "offset"):
            raise ValueError(
                f"Unsupported cond_encoding={self.cond_encoding!r}; expected 'legacy' or 'offset'."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vae": self.vae.to_dict(),
            "fm": self.fm.to_dict(),
            "train": self.train.to_dict(),
            "cond_encoding": self.cond_encoding,
            "mask_token": self.mask_token,
            "pad_to_blocks": self.pad_to_blocks,
            "scale_data": self.scale_data,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SAVEConfig:
        vae_cfg = VAEConfig.from_dict(payload.get("vae", {}))
        fm_cfg = FMConfig.from_dict(payload.get("fm", {}))
        train_cfg = TrainConfig.from_dict(payload.get("train", {}))
        return cls(
            vae=vae_cfg,
            fm=fm_cfg,
            train=train_cfg,
            cond_encoding=payload.get("cond_encoding", "legacy"),
            mask_token=payload.get("mask_token", "<mask>"),
            pad_to_blocks=bool(payload.get("pad_to_blocks", True)),
            scale_data=bool(payload.get("scale_data", True)),
            seed=int(payload.get("seed", train_cfg.seed)),
        )

    @classmethod
    def save_b(
        cls,
        gene_block_size: int = 3200,
        *,
        latent_block_size: int | None = None,
        hidden_dim: int = 64,
        enc_dim: int = 64,
        expand_dim: int = 512,
        num_heads: int = 4,
        enc_depth: int = 3,
        dec_depth: int = 3,
        fm_n_blocks: int = 5,
        fm_num_heads: int | None = None,
        vae_epochs: int = 200,
        fm_epochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-4,
        weight_decay: float = 2.5e-5,
        kl_scale: float = 0.5,
        vae_col_mask_prob: float = 1.0,
        fm_col_mask_prob: float = 0.0,
        guidance_weight: float = 2.0,
        sample_steps: int = 20,
        solver: str = "euler",
        cond_encoding: Literal["legacy", "offset"] = "legacy",
        use_amp_vae: bool = False,
        seed: int = 1202,
    ) -> SAVEConfig:
        """Preset matching SAVE-B (formal conditional generation, ICLR 2026)."""
        lbs = gene_block_size if latent_block_size is None else latent_block_size
        fm_heads = num_heads if fm_num_heads is None else fm_num_heads
        return cls(
            vae=VAEConfig(
                hidden_dim=hidden_dim,
                enc_dim=enc_dim,
                expand_dim=expand_dim,
                enc_num_heads=num_heads,
                dec_num_heads=num_heads,
                enc_depth=enc_depth,
                dec_depth=dec_depth,
                gene_block_size=gene_block_size,
                latent_block_size=lbs,
                enc_attn_dropout=0.2,
                dec_attn_dropout=0.0,
                out_layer_dropout=0.4,
            ),
            fm=FMConfig(
                hidden_dim=expand_dim * 2,
                embedding_dim=expand_dim * 2,
                n_blocks=fm_n_blocks,
                num_heads=fm_heads,
                guidance_weight=guidance_weight,
                sample_steps=sample_steps,
                solver=solver,
            ),
            train=TrainConfig(
                vae_epochs=vae_epochs,
                fm_epochs=fm_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                kl_scale=kl_scale,
                vae_col_mask_prob=vae_col_mask_prob,
                fm_col_mask_prob=fm_col_mask_prob,
                use_amp_vae=use_amp_vae,
                seed=seed,
            ),
            cond_encoding=cond_encoding,
            seed=seed,
        )

    @classmethod
    def save_b_perturb(
        cls,
        gene_block_size: int = 400,
        *,
        latent_block_size: int | None = None,
        hidden_dim: int = 64,
        enc_dim: int = 64,
        expand_dim: int = 512,
        num_heads: int = 4,
        enc_depth: int = 3,
        dec_depth: int = 3,
        vae_epochs: int = 200,
        fm_epochs: int = 0,
        batch_size: int = 1024,
        lr: float = 1e-4,
        weight_decay: float = 2.5e-5,
        kl_scale: float = 0.5,
        vae_col_mask_prob: float = 0.6,
        cond_encoding: Literal["legacy", "offset"] = "legacy",
        use_amp_vae: bool = False,
        seed: int = 1202,
    ) -> SAVEConfig:
        """Preset matching SAVE-B-perturb (BEC integration & perturbation prediction)."""
        lbs = gene_block_size if latent_block_size is None else latent_block_size
        return cls(
            vae=VAEConfig(
                hidden_dim=hidden_dim,
                enc_dim=enc_dim,
                expand_dim=expand_dim,
                enc_num_heads=num_heads,
                dec_num_heads=num_heads,
                enc_depth=enc_depth,
                dec_depth=dec_depth,
                gene_block_size=gene_block_size,
                latent_block_size=lbs,
                enc_attn_dropout=0.2,
                dec_attn_dropout=0.0,
                out_layer_dropout=0.4,
            ),
            fm=FMConfig(
                hidden_dim=expand_dim * 2,
                embedding_dim=expand_dim * 2,
                n_blocks=5,
                num_heads=num_heads,
            ),
            train=TrainConfig(
                vae_epochs=vae_epochs,
                fm_epochs=fm_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                kl_scale=kl_scale,
                vae_col_mask_prob=vae_col_mask_prob,
                fm_col_mask_prob=0.0,
                use_amp_vae=use_amp_vae,
                seed=seed,
            ),
            cond_encoding=cond_encoding,
            seed=seed,
        )

    @classmethod
    def smoke(
        cls,
        gene_block_size: int = 3200,
        *,
        latent_block_size: int | None = None,
        hidden_dim: int = 64,
        enc_dim: int = 32,
        expand_dim: int = 128,
        num_heads: int = 4,
        enc_depth: int = 2,
        dec_depth: int = 2,
        fm_n_blocks: int = 2,
        vae_epochs: int = 1,
        fm_epochs: int = 1,
        batch_size: int = 64,
        guidance_weight: float = 2.0,
        sample_steps: int = 5,
        cond_encoding: Literal["legacy", "offset"] = "legacy",
        use_amp_vae: bool = False,
        seed: int = 1202,
    ) -> SAVEConfig:
        """Compact preset for fast smoke runs and unit tests."""
        lbs = gene_block_size if latent_block_size is None else latent_block_size
        return cls(
            vae=VAEConfig(
                hidden_dim=hidden_dim,
                enc_dim=enc_dim,
                expand_dim=expand_dim,
                enc_num_heads=num_heads,
                dec_num_heads=num_heads,
                enc_depth=enc_depth,
                dec_depth=dec_depth,
                gene_block_size=gene_block_size,
                latent_block_size=lbs,
                enc_attn_dropout=0.0,
                dec_attn_dropout=0.0,
                out_layer_dropout=0.0,
            ),
            fm=FMConfig(
                hidden_dim=expand_dim * 2,
                embedding_dim=expand_dim * 2,
                n_blocks=fm_n_blocks,
                num_heads=num_heads,
                guidance_weight=guidance_weight,
                sample_steps=sample_steps,
            ),
            train=TrainConfig(
                vae_epochs=vae_epochs,
                fm_epochs=fm_epochs,
                batch_size=batch_size,
                lr=1e-4,
                weight_decay=2.5e-5,
                kl_scale=0.5,
                vae_col_mask_prob=1.0,
                fm_col_mask_prob=0.0,
                use_amp_vae=use_amp_vae,
                seed=seed,
            ),
            cond_encoding=cond_encoding,
            seed=seed,
        )


__all__ = [
    "FMConfig",
    "SAVEConfig",
    "TrainConfig",
    "VAEConfig",
]
