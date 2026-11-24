"""High-level SAVE model interface for single-cell representation, generation, and transfer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from sc_save.config import FMConfig, SAVEConfig
from sc_save.data.preprocessing import (
    ConditionVocabulary,
    MaxAbsExpressionScaler,
    align_genes,
    find_pad_index,
    pad_to_block_size,
)
from sc_save.nn.flow_matching import FM, MLPTimeStep
from sc_save.nn.vae import VAE, load_state_dict_compat
from sc_save.training.fm_trainer import train_fm
from sc_save.training.runtime import resolve_device, seed_everything
from sc_save.training.vae_trainer import train_vae


def _resolve_infer_batch_size(
    device: torch.device,
    batch_size: int | None,
    default_batch_size: int,
) -> int:
    if batch_size is not None:
        return max(1, int(batch_size))
    base = max(1, int(default_batch_size))
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            if total_gb >= 40.0:
                return max(base, 4096)
        except RuntimeError:
            pass
    return base


def _load_torch_state_dict(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


class SAVE:
    """Single-Cell Variational Flow Matching (SAVE) model."""

    def __init__(
        self,
        adata: ad.AnnData | None = None,
        condition_cols: Sequence[str] = ("batch", "cell_type"),
        config: SAVEConfig | None = None,
        device: str | int | torch.device = "auto",
        scale_data: bool = True,
        pad_to_blocks: bool = True,
        cond_encoding: Literal["legacy", "offset"] | None = None,
        *,
        copy_adata: bool = True,
        vocab: ConditionVocabulary | None = None,
        scaler: MaxAbsExpressionScaler | None = None,
        n_genes: int | None = None,
        var_names: Sequence[str] | None = None,
        pad_idx: int | None = None,
        outer_embeddings: np.ndarray | None = None,
    ) -> None:
        cfg = (
            SAVEConfig.from_dict(config.to_dict())
            if config is not None
            else SAVEConfig()
        )
        if cond_encoding is not None:
            if cond_encoding not in ("legacy", "offset"):
                raise ValueError(
                    f"Unsupported cond_encoding={cond_encoding!r}; expected 'legacy' or 'offset'."
                )
            cfg.cond_encoding = cond_encoding
        cfg.scale_data = bool(scale_data)
        cfg.pad_to_blocks = bool(pad_to_blocks)

        self.config = cfg
        self.device = resolve_device(device)
        self.outer_embeddings = outer_embeddings
        seed_everything(cfg.seed)

        if adata is not None:
            missing_cols = [c for c in condition_cols if c not in adata.obs.columns]
            if missing_cols:
                raise ValueError(
                    f"Input AnnData missing required obs columns: {missing_cols}"
                )

            base_adata = adata
            did_pad = False
            if (
                pad_to_blocks
                and cfg.vae.is_reshape_first
                and cfg.vae.gene_block_size > 0
                and adata.n_vars % cfg.vae.gene_block_size != 0
            ):
                base_adata, detected_pad_idx = pad_to_block_size(
                    adata, cfg.vae.gene_block_size
                )
                did_pad = True
            else:
                detected_pad_idx = find_pad_index(adata)

            self.pad_idx = int(pad_idx if pad_idx is not None else detected_pad_idx)
            self.var_names = [str(v) for v in base_adata.var_names]
            self.num_genes = int(base_adata.n_vars)
            self.num_cells = int(base_adata.n_obs)

            if vocab is not None:
                self.vocab = vocab
            else:
                self.vocab = ConditionVocabulary(
                    condition_cols=condition_cols,
                    mode=cfg.cond_encoding,
                    mask_token=cfg.mask_token,
                ).fit(base_adata.obs)

            if scaler is not None:
                self.scaler: MaxAbsExpressionScaler | None = scaler
                if scale_data:
                    scaled_x = self.scaler.transform(base_adata.X)
                    working_adata = ad.AnnData(
                        X=scaled_x,
                        obs=base_adata.obs.copy(),
                        var=base_adata.var.copy(),
                    )
                elif did_pad or not copy_adata:
                    working_adata = base_adata
                else:
                    working_adata = base_adata.copy()
            elif scale_data:
                self.scaler = MaxAbsExpressionScaler().fit(base_adata.X)
                scaled_x = self.scaler.transform(base_adata.X)
                working_adata = ad.AnnData(
                    X=scaled_x,
                    obs=base_adata.obs.copy(),
                    var=base_adata.var.copy(),
                )
            else:
                self.scaler = None
                if did_pad or not copy_adata:
                    working_adata = base_adata
                else:
                    working_adata = base_adata.copy()

            self.adata: ad.AnnData | None = working_adata
        else:
            if vocab is None:
                raise ValueError("Provide `adata` or an existing fitted `vocab`.")
            if n_genes is None and var_names is None:
                raise ValueError("Provide `adata` or `n_genes`/`var_names`.")
            self.vocab = vocab
            self.var_names = (
                [str(v) for v in var_names]
                if var_names is not None
                else [f"gene_{i}" for i in range(int(n_genes))]
            )
            self.num_genes = int(n_genes if n_genes is not None else len(self.var_names))
            self.pad_idx = int(pad_idx if pad_idx is not None else self.num_genes)
            self.num_cells = 0
            self.scaler = scaler
            self.adata = None

        self.condition_cols = self.vocab.condition_cols

        if cfg.vae.is_reshape_first:
            if (
                cfg.vae.gene_block_size <= 0
                or self.num_genes % cfg.vae.gene_block_size != 0
            ):
                raise ValueError(
                    f"Input gene count ({self.num_genes}) must be a multiple of "
                    f"gene_block_size={cfg.vae.gene_block_size} (set pad_to_blocks=True to auto-pad)."
                )
            if (
                cfg.vae.latent_block_size <= 0
                or self.num_genes % cfg.vae.latent_block_size != 0
            ):
                raise ValueError(
                    f"Input gene count ({self.num_genes}) must be a multiple of "
                    f"latent_block_size={cfg.vae.latent_block_size}."
                )
            self.seq_length = self.num_genes // cfg.vae.gene_block_size
        else:
            self.seq_length = cfg.vae.hidden_dim

        self.vae = VAE(
            input_dim=self.num_genes,
            hidden_dim=cfg.vae.hidden_dim,
            enc_dim=cfg.vae.enc_dim,
            expand_dim=cfg.vae.expand_dim,
            enc_num_heads=cfg.vae.enc_num_heads,
            dec_num_heads=cfg.vae.dec_num_heads,
            enc_depth=cfg.vae.enc_depth,
            dec_depth=cfg.vae.dec_depth,
            enc_affine=cfg.vae.enc_affine,
            enc_norm_type=cfg.vae.enc_norm_type,
            dec_norm_init=cfg.vae.dec_norm_init,
            dec_blk_type=cfg.vae.dec_blk_type,
            is_initialize=cfg.vae.is_initialize,
            len_col_comb=len(self.condition_cols),
            mask_id=self.vocab.mask_id,
            label_encoder_dict=self.vocab.label_encoder_dict,
            total_cond_classes=self.vocab.total_classes,
            is_multi_codebook=cfg.vae.is_multi_codebook,
            is_reshape_first=cfg.vae.is_reshape_first,
            gene_block_size=cfg.vae.gene_block_size,
            latent_block_size=cfg.vae.latent_block_size,
            out_layer_dropout=cfg.vae.out_layer_dropout,
            enc_attn_dropout=cfg.vae.enc_attn_dropout,
            dec_attn_dropout=cfg.vae.dec_attn_dropout,
            is_enc_cond=cfg.vae.is_enc_cond,
            outer_emb_dim=cfg.vae.outer_emb_dim,
        ).to(self.device)

        self.fm: FM | None = None
        self.is_trained: bool = False
        self.is_fm_trained: bool = False

    @property
    def model(self) -> VAE:
        return self.vae

    @property
    def label_encoder_dict(self):
        return self.vocab.label_encoder_dict

    @property
    def mask_id(self) -> int:
        return self.vocab.mask_id

    @property
    def enc_dim(self) -> int:
        return self.config.vae.enc_dim

    @property
    def expand_dim(self) -> int:
        return self.config.vae.expand_dim

    @property
    def cond_encoding(self) -> str:
        return self.config.cond_encoding

    def init_fm(
        self,
        fm_config: FMConfig | None = None,
        *,
        seq_length: int | None = None,
    ) -> FM:
        """Instantiate the Stage-2 Flow Matching denoiser and wrapper on ``self.device``."""
        fm_cfg = fm_config or self.config.fm
        if fm_config is not None:
            self.config.fm = FMConfig.from_dict(fm_config.to_dict())
        eff_seq_length = int(self.seq_length if seq_length is None else seq_length)

        denoiser = MLPTimeStep(
            in_dim=self.config.vae.enc_dim,
            conditional=True,
            seq_length=eff_seq_length,
            hidden_dim=fm_cfg.hidden_dim,
            embedding_dim=fm_cfg.embedding_dim,
            num_conds=len(self.condition_cols),
            dropout_prob=fm_cfg.dropout_prob,
            n_blocks=fm_cfg.n_blocks,
            label_encoder_dict=self.vocab.label_encoder_dict,
            total_cond_classes=self.vocab.total_classes,
            mask_id=self.vocab.mask_id,
            conditioning_probability=fm_cfg.conditioning_probability,
            block_type=fm_cfg.block_type,
            num_heads=fm_cfg.num_heads,
        )
        self.fm = FM(
            denoising_model=denoiser,
            antithetic_time_sampling=fm_cfg.antithetic_time_sampling,
            sigma=fm_cfg.sigma,
            device=self.device,
            use_ot=fm_cfg.use_ot,
            guidance_weights=fm_cfg.guidance_weight,
            path_type=fm_cfg.path_type,
            path_kwargs=fm_cfg.path_kwargs,
            use_snr_weighting=fm_cfg.use_snr_weighting,
        )
        return self.fm

    def fit_vae(
        self,
        epochs: int | None = None,
        *,
        batch_size: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        kl_scale: float | None = None,
        col_mask_prob: float | None = None,
        is_col_wise_mask: bool | None = None,
        recon_criterion: str | None = None,
        use_amp: bool | None = None,
        seed: int | None = None,
        verbose: bool = True,
        loss_monitor: Any = None,
        tune_session: Any = None,
    ) -> dict[str, float]:
        """Train Stage-1 Attention VAE on ``self.adata``."""
        if self.adata is None:
            raise RuntimeError("No training AnnData attached to SAVE instance.")

        t_cfg = self.config.train
        eff_epochs = int(t_cfg.vae_epochs if epochs is None else epochs)
        eff_bs = int(t_cfg.batch_size if batch_size is None else batch_size)
        eff_bs = min(eff_bs, max(1, self.adata.n_obs))
        eff_lr = float(t_cfg.lr if lr is None else lr)
        eff_wd = float(t_cfg.weight_decay if weight_decay is None else weight_decay)
        eff_kl = float(t_cfg.kl_scale if kl_scale is None else kl_scale)
        eff_mask = float(t_cfg.vae_col_mask_prob if col_mask_prob is None else col_mask_prob)
        eff_col_wise = bool(t_cfg.is_col_wise_mask if is_col_wise_mask is None else is_col_wise_mask)
        eff_crit = str(t_cfg.recon_criterion if recon_criterion is None else recon_criterion)
        eff_amp = bool(t_cfg.use_amp_vae if use_amp is None else use_amp)
        eff_seed = int(t_cfg.seed if seed is None else seed)

        y = self.vocab.encode_obs(self.adata.obs)
        losses = train_vae(
            vae=self.vae,
            x=self.adata.X,
            y=y,
            emb=self.outer_embeddings,
            num_epoch=eff_epochs,
            batch_size=eff_bs,
            kl_scale=eff_kl,
            device=self.device,
            lr=eff_lr,
            grad_clip=t_cfg.grad_clip,
            seed=eff_seed,
            weight_decay=eff_wd,
            col_msk_threshold=eff_mask,
            loss_monitor=loss_monitor,
            tune_session=tune_session,
            is_col_wise_mask=eff_col_wise,
            is_flash_attn=self.config.vae.is_flash_attn,
            use_amp_vae=eff_amp,
            is_multi_codebook=self.config.vae.is_multi_codebook,
            recon_criterion=eff_crit,
            is_emb_ary=self.outer_embeddings is not None,
            verbose=verbose,
        )
        self.is_trained = True
        return losses

    def fit_fm(
        self,
        epochs: int | None = None,
        *,
        batch_size: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        col_mask_prob: float | None = None,
        use_amp: bool | None = None,
        seed: int | None = None,
        verbose: bool = True,
        loss_monitor: Any = None,
        tune_session: Any = None,
    ) -> dict[str, float]:
        """Train Stage-2 Flow Matching on precomputed VAE posteriors."""
        if self.adata is None:
            raise RuntimeError("No training AnnData attached to SAVE instance.")
        if self.fm is None:
            self.init_fm()

        t_cfg = self.config.train
        eff_epochs = int(t_cfg.fm_epochs if epochs is None else epochs)
        eff_bs = int(t_cfg.batch_size if batch_size is None else batch_size)
        eff_bs = min(eff_bs, max(1, self.adata.n_obs))
        default_fm_lr = t_cfg.lr if t_cfg.fm_lr is None else t_cfg.fm_lr
        eff_lr = float(default_fm_lr if lr is None else lr)
        eff_wd = float(t_cfg.weight_decay if weight_decay is None else weight_decay)
        eff_mask = float(t_cfg.fm_col_mask_prob if col_mask_prob is None else col_mask_prob)
        eff_amp = bool(t_cfg.use_amp_fm if use_amp is None else use_amp)
        eff_seed = int(t_cfg.seed if seed is None else seed)

        y = self.vocab.encode_obs(self.adata.obs)
        losses = train_fm(
            vae=self.vae,
            fm=self.fm,
            x=self.adata.X,
            y=y,
            emb=self.outer_embeddings,
            num_epoch=eff_epochs,
            batch_size=eff_bs,
            device=self.device,
            lr=eff_lr,
            grad_clip=t_cfg.grad_clip,
            seed=eff_seed,
            weight_decay=eff_wd,
            col_msk_threshold=eff_mask,
            loss_monitor=loss_monitor,
            tune_session=tune_session,
            is_col_wise_mask=t_cfg.is_col_wise_mask,
            is_flash_attn=self.config.vae.is_flash_attn,
            use_amp_fm=eff_amp,
            is_multi_codebook=self.config.vae.is_multi_codebook,
            is_emb_ary=self.outer_embeddings is not None,
            verbose=verbose,
        )
        self.is_fm_trained = True
        self.is_trained = True
        return losses

    def fit(
        self,
        vae_epochs: int | None = None,
        fm_epochs: int | None = None,
        *,
        train_fm: bool = True,
        batch_size: int | None = None,
        lr: float | None = None,
        fm_lr: float | None = None,
        weight_decay: float | None = None,
        kl_scale: float | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> SAVE:
        """Fit Stage-1 VAE and optionally Stage-2 Flow Matching, returning ``self``."""
        eff_vae_epochs = self.config.train.vae_epochs if vae_epochs is None else int(vae_epochs)
        if eff_vae_epochs > 0:
            self.fit_vae(
                epochs=eff_vae_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                kl_scale=kl_scale,
                seed=seed,
                verbose=verbose,
            )

        eff_fm_epochs = self.config.train.fm_epochs if fm_epochs is None else int(fm_epochs)
        if train_fm and eff_fm_epochs > 0:
            self.fit_fm(
                epochs=eff_fm_epochs,
                batch_size=batch_size,
                lr=fm_lr if fm_lr is not None else lr,
                weight_decay=weight_decay,
                seed=seed,
                verbose=verbose,
            )
        return self

    def _align_external_adata(self, adata: ad.AnnData) -> tuple[ad.AnnData, Any]:
        src_names = [str(g) for g in adata.var_names]
        if src_names == self.var_names:
            aligned = adata
        elif (
            adata.n_vars == self.pad_idx
            and self.pad_idx < self.num_genes
            and src_names == self.var_names[: self.pad_idx]
        ):
            aligned = align_genes(adata, self.var_names, pad_missing=True)
        elif set(self.var_names[: self.pad_idx]).issubset(set(src_names)):
            aligned = align_genes(adata, self.var_names, pad_missing=True)
        elif adata.n_vars == self.num_genes:
            aligned = adata
        elif adata.n_vars < self.num_genes and self.config.pad_to_blocks:
            padded, _ = pad_to_block_size(adata, self.config.vae.gene_block_size)
            if padded.n_vars != self.num_genes:
                raise ValueError(
                    f"Input AnnData has {adata.n_vars} genes (padded to {padded.n_vars}), "
                    f"expected {self.num_genes}."
                )
            aligned = padded
        else:
            raise ValueError(
                f"Input AnnData has {adata.n_vars} genes, expected {self.num_genes} "
                f"(or {self.pad_idx} unpadded)."
            )

        x = aligned.X
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return aligned, x

    def _prepare_input(self, adata: ad.AnnData | None) -> tuple[ad.AnnData, Any]:
        if adata is None:
            if self.adata is None:
                raise RuntimeError("Pass `adata` explicitly when SAVE has no attached AnnData.")
            return self.adata, self.adata.X
        return self._align_external_adata(adata)

    def _slice_x_batch(self, x: Any, start: int, end: int) -> torch.Tensor:
        if sparse.issparse(x):
            arr = x[start:end].toarray()
        elif isinstance(x, torch.Tensor):
            return x[start:end].to(device=self.device, dtype=torch.float32)
        else:
            arr = np.asarray(x[start:end])
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(self.device)

    def _embed_vae_cond(self, cond_ids: torch.Tensor) -> torch.Tensor:
        if self.config.vae.is_multi_codebook:
            cols = [self.vae.cond_embedder[i](cond_ids[:, i]) for i in range(cond_ids.shape[1])]
            return torch.stack(cols, dim=1)
        flat = cond_ids.reshape(-1)
        return self.vae.cond_embedder(flat).reshape(cond_ids.shape[0], cond_ids.shape[1], -1)

    def _postprocess_output(
        self,
        x_mat: np.ndarray,
        obs: pd.DataFrame,
        *,
        inverse_scale: bool,
        trim_pad: bool,
        inplace_scale: bool = False,
    ) -> ad.AnnData:
        x_out = np.asarray(x_mat, dtype=np.float32)
        if inverse_scale and self.scaler is not None:
            x_out = np.asarray(
                self.scaler.inverse_transform(x_out, inplace=inplace_scale),
                dtype=np.float32,
            )
        var = pd.DataFrame(index=pd.Index(self.var_names, dtype="object"))
        if trim_pad and self.pad_idx < self.num_genes:
            x_out = x_out[:, : self.pad_idx]
            var = var.iloc[: self.pad_idx].copy()
        return ad.AnnData(X=x_out, obs=obs.copy(), var=var)

    @torch.inference_mode()
    def encode(
        self,
        adata: ad.AnnData | None = None,
        *,
        with_cond: bool = False,
        latent_pos: int = 0,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode cells into VAE latent space ``(z, mu, var)[latent_pos]``."""
        src_adata, x_mat = self._prepare_input(adata)
        self.vae.to(self.device).eval()

        n_obs = int(src_adata.n_obs)
        bs = _resolve_infer_batch_size(self.device, batch_size, self.config.train.batch_size)
        if with_cond:
            cond_ids_np = self.vocab.encode_obs(src_adata.obs)
        else:
            cond_ids_np = np.full(
                (n_obs, len(self.condition_cols)), self.vocab.mask_id, dtype=np.int64
            )

        latents: list[np.ndarray] = []
        for start in range(0, n_obs, bs):
            end = min(start + bs, n_obs)
            x_b = self._slice_x_batch(x_mat, start, end)
            y_b = torch.from_numpy(cond_ids_np[start:end]).to(self.device, dtype=torch.long)
            c_b = self._embed_vae_cond(y_b)
            out = self.vae.encoder(x_b, c_b)[latent_pos]
            latents.append(out.detach().cpu().numpy())

        return np.concatenate(latents, axis=0) if latents else np.empty(
            (0, self.seq_length, self.config.vae.enc_dim), dtype=np.float32
        )

    def _resolve_cond_array_and_obs(
        self,
        cond: Any,
        n_rows: int,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        n_cols = len(self.condition_cols)
        if cond is None:
            ids = np.full((n_rows, n_cols), self.vocab.mask_id, dtype=np.int64)
            obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_rows)])
            return ids, obs
        if isinstance(cond, dict):
            row = self.vocab.encode_dict(cond, allow_partial=True)
            ids = np.repeat(row[None, :], n_rows, axis=0)
            obs = pd.DataFrame(
                {k: [v] * n_rows for k, v in cond.items()},
                index=[f"cell_{i}" for i in range(n_rows)],
            )
            return ids, obs
        if isinstance(cond, pd.DataFrame):
            if len(cond) != n_rows:
                raise ValueError(f"cond DataFrame length ({len(cond)}) != n_rows ({n_rows}).")
            return self.vocab.encode_obs(cond), cond.copy()
        if isinstance(cond, (list, tuple)):
            df = pd.DataFrame(list(cond))
            if len(df) != n_rows:
                raise ValueError(f"cond sequence length ({len(df)}) != n_rows ({n_rows}).")
            return self.vocab.encode_obs(df), df
        arr = cond.detach().cpu().numpy() if isinstance(cond, torch.Tensor) else np.asarray(cond)
        ids = np.asarray(arr, dtype=np.int64)
        if ids.ndim == 1:
            ids = np.repeat(ids[None, :], n_rows, axis=0)
        if ids.shape != (n_rows, n_cols):
            raise ValueError(f"Expected condition ID shape {(n_rows, n_cols)}, got {ids.shape}.")
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_rows)])
        return ids, obs

    @torch.inference_mode()
    def decode(
        self,
        latents: np.ndarray | torch.Tensor,
        cond: Any = None,
        *,
        batch_size: int | None = None,
        inverse_scale: bool = True,
        trim_pad: bool = True,
    ) -> ad.AnnData:
        """Decode latent vectors back into AnnData gene expression."""
        self.vae.to(self.device).eval()

        z_tensor = (
            latents.detach().to(dtype=torch.float32)
            if isinstance(latents, torch.Tensor)
            else torch.from_numpy(np.asarray(latents, dtype=np.float32))
        )
        if z_tensor.ndim == 2:
            if z_tensor.shape[1] == self.config.vae.enc_dim and self.seq_length == 1:
                z_tensor = z_tensor.unsqueeze(1)
            elif z_tensor.shape[1] == self.seq_length * self.config.vae.enc_dim:
                z_tensor = z_tensor.reshape(
                    z_tensor.shape[0], self.seq_length, self.config.vae.enc_dim
                )
            else:
                raise ValueError(
                    f"Cannot reshape 2D latents {tuple(z_tensor.shape)} to "
                    f"(N, {self.seq_length}, {self.config.vae.enc_dim})."
                )
        elif z_tensor.ndim != 3:
            raise ValueError(f"Expected 2D or 3D latents, got shape {tuple(z_tensor.shape)}.")

        n_obs = int(z_tensor.shape[0])
        cond_ids_np, obs = self._resolve_cond_array_and_obs(cond, n_obs)
        bs = _resolve_infer_batch_size(self.device, batch_size, self.config.train.batch_size)

        outputs: list[np.ndarray] = []
        for start in range(0, n_obs, bs):
            end = min(start + bs, n_obs)
            z_b = z_tensor[start:end].to(self.device)
            y_b = torch.from_numpy(cond_ids_np[start:end]).to(self.device, dtype=torch.long)
            c_b = self._embed_vae_cond(y_b)
            recon_b = torch.clamp(self.vae.decoder(z_b, c_b), 0.0, 1.0)
            outputs.append(recon_b.detach().cpu().numpy())

        x_all = (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, self.num_genes), dtype=np.float32)
        )
        return self._postprocess_output(
            x_all,
            obs=obs,
            inverse_scale=inverse_scale,
            trim_pad=trim_pad,
            inplace_scale=True,
        )

    @torch.inference_mode()
    def reconstruct(
        self,
        adata: ad.AnnData | None = None,
        *,
        with_cond: bool = True,
        batch_size: int | None = None,
        inverse_scale: bool = True,
        trim_pad: bool = True,
    ) -> ad.AnnData:
        """Encode and reconstruct an AnnData through the Stage-1 VAE."""
        src_adata, x_mat = self._prepare_input(adata)
        self.vae.to(self.device).eval()

        n_obs = int(src_adata.n_obs)
        bs = _resolve_infer_batch_size(self.device, batch_size, self.config.train.batch_size)
        if with_cond:
            cond_ids_np = self.vocab.encode_obs(src_adata.obs)
        else:
            cond_ids_np = np.full(
                (n_obs, len(self.condition_cols)), self.vocab.mask_id, dtype=np.int64
            )

        outputs: list[np.ndarray] = []
        for start in range(0, n_obs, bs):
            end = min(start + bs, n_obs)
            x_b = self._slice_x_batch(x_mat, start, end)
            y_b = torch.from_numpy(cond_ids_np[start:end]).to(self.device, dtype=torch.long)
            c_b = self._embed_vae_cond(y_b)
            z_b, _, _ = self.vae.encoder(x_b, c_b)
            recon_b = torch.clamp(self.vae.decoder(z_b, c_b), 0.0, 1.0)
            outputs.append(recon_b.detach().cpu().numpy())

        x_all = (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, self.num_genes), dtype=np.float32)
        )
        return self._postprocess_output(
            x_all,
            obs=src_adata.obs,
            inverse_scale=inverse_scale,
            trim_pad=trim_pad,
            inplace_scale=True,
        )

    def _expand_generation_conditions(
        self,
        conditions: dict[str, Any] | pd.DataFrame | Sequence[dict[str, Any]],
        n_cells: int | None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if isinstance(conditions, dict):
            if n_cells is None or int(n_cells) <= 0:
                raise ValueError("`n_cells` (> 0) is required when `conditions` is a dict.")
            n = int(n_cells)
            row_ids = self.vocab.encode_dict(conditions, allow_partial=True)
            cond_ids = np.repeat(row_ids[None, :], n, axis=0)
            obs = pd.DataFrame(
                {k: [v] * n for k, v in conditions.items()},
                index=[f"{i}_gen" for i in range(n)],
            )
            return cond_ids, obs

        if isinstance(conditions, pd.DataFrame):
            df = conditions[self.condition_cols].copy()
        elif isinstance(conditions, (list, tuple)):
            df = pd.DataFrame(list(conditions))
        else:
            raise TypeError(
                f"Expected conditions as dict, DataFrame, or sequence of dicts, got {type(conditions).__name__}."
            )

        if n_cells is not None:
            n = int(n_cells)
            if n <= 0:
                raise ValueError(f"n_cells must be positive, got {n}.")
            df = df.loc[df.index.repeat(n)].copy()

        df = df.reset_index(drop=True)
        df.index = pd.Index([f"{i}_gen" for i in range(len(df))], dtype="object")
        cond_ids = self.vocab.encode_obs(df)
        return cond_ids, df

    @torch.inference_mode()
    def generate(
        self,
        conditions: dict[str, Any] | pd.DataFrame | Sequence[dict[str, Any]],
        n_cells: int | None = None,
        *,
        guidance_weight: float | None = None,
        steps: int | None = None,
        solver: str | None = None,
        batch_size: int | None = None,
        return_latents: bool = False,
        inverse_scale: bool = True,
        trim_pad: bool = True,
        min_clip: float = 1e-4,
    ) -> ad.AnnData | tuple[ad.AnnData, np.ndarray]:
        """Generate synthetic single-cell expression profiles via Flow Matching + VAE."""
        if self.fm is None:
            raise RuntimeError("Flow Matching model is not initialized. Call fit() or fit_fm() first.")

        cond_ids_np, obs = self._expand_generation_conditions(conditions, n_cells)
        n_total = len(obs)

        eff_w = float(
            guidance_weight
            if guidance_weight is not None
            else (
                self.fm.guidance_weights
                if self.fm.guidance_weights is not None
                else self.config.fm.guidance_weight
            )
        )
        eff_steps = int(self.config.fm.sample_steps if steps is None else steps)
        eff_solver = str(self.config.fm.solver if solver is None else solver)
        bs = _resolve_infer_batch_size(self.device, batch_size, self.config.train.batch_size)

        self.vae.to(self.device).eval()
        self.fm.to(self.device).eval()
        self.fm.denoising_model.eval()

        out_cols = (
            self.pad_idx
            if (trim_pad and self.pad_idx < self.num_genes)
            else self.num_genes
        )
        x_all = np.empty((n_total, out_cols), dtype=np.float32)
        z_all = (
            np.empty(
                (n_total, self.seq_length, self.config.vae.enc_dim),
                dtype=np.float32,
            )
            if return_latents
            else None
        )
        scale_slice = (
            self.scaler.scale_[:out_cols]
            if (inverse_scale and self.scaler is not None)
            else None
        )

        for start in range(0, n_total, bs):
            end = min(start + bs, n_total)
            y_b = torch.from_numpy(cond_ids_np[start:end]).to(self.device, dtype=torch.long)
            c_fm = self.fm.denoising_model.cond_embedder(y_b.reshape(-1)).reshape(
                y_b.shape[0], y_b.shape[1], -1
            )
            mask_ids = torch.full(
                (y_b.numel(),), self.vocab.mask_id, device=self.device, dtype=torch.long
            )
            mask_emb_de = self.vae.cond_embedder(mask_ids).reshape(
                y_b.shape[0], y_b.shape[1], -1
            )

            z_b = self.fm.sample(
                batch_size=c_fm.shape[0],
                seq_length=self.seq_length,
                n_sample_steps=eff_steps,
                y=c_fm,
                guidance_weights=eff_w,
                solver_type=eff_solver,
                use_sde=self.config.fm.use_sde,
            )
            recon_b = torch.clamp(self.vae.decoder(z_b, mask_emb_de), 0.0, 1.0)
            if min_clip is not None and min_clip > 0:
                recon_b[recon_b < min_clip] = 0.0

            x_all[start:end] = recon_b[:, :out_cols].detach().cpu().numpy()
            if z_all is not None:
                z_all[start:end] = z_b.detach().cpu().numpy()

        if scale_slice is not None:
            x_all *= scale_slice[None, :]

        var = pd.DataFrame(
            index=pd.Index(self.var_names[:out_cols], dtype="object")
        )
        gen_adata = ad.AnnData(X=x_all, obs=obs.copy(), var=var)
        if z_all is not None:
            return gen_adata, z_all
        return gen_adata

    @torch.inference_mode()
    def generate_like(
        self,
        target_adata: ad.AnnData,
        *,
        guidance_weight: float | None = None,
        steps: int | None = None,
        solver: str | None = None,
        batch_size: int | None = None,
        return_latents: bool = False,
        inverse_scale: bool = True,
        trim_pad: bool = True,
    ) -> ad.AnnData | tuple[ad.AnnData, np.ndarray]:
        """Generate cells matching ``target_adata.obs[condition_cols]`` in one batched pass."""
        missing = [c for c in self.condition_cols if c not in target_adata.obs.columns]
        if missing:
            raise ValueError(f"target_adata.obs missing required condition columns: {missing}")

        return self.generate(
            conditions=target_adata.obs[self.condition_cols],
            n_cells=None,
            guidance_weight=guidance_weight,
            steps=steps,
            solver=solver,
            batch_size=batch_size,
            return_latents=return_latents,
            inverse_scale=inverse_scale,
            trim_pad=trim_pad,
        )

    @torch.inference_mode()
    def generate_with_attention(
        self,
        conditions: dict[str, Any],
        n_cells: int,
        *,
        guidance_weight: float | None = None,
        steps: int = 50,
        batch_size: int = 128,
        inverse_scale: bool = True,
        trim_pad: bool = True,
    ) -> dict[str, Any]:
        """Generate cells and collect per-step FM DiT attention weights."""
        if self.fm is None:
            raise RuntimeError("Flow Matching model is not initialized. Call fit() or fit_fm() first.")

        cond_ids_np, obs = self._expand_generation_conditions(conditions, n_cells)
        eff_w = float(
            guidance_weight
            if guidance_weight is not None
            else (
                self.fm.guidance_weights
                if self.fm.guidance_weights is not None
                else self.config.fm.guidance_weight
            )
        )
        bs = max(1, int(batch_size))

        self.vae.to(self.device).eval()
        self.fm.to(self.device).eval()
        self.fm.denoising_model.eval()

        x_chunks: list[np.ndarray] = []
        all_attn: list[dict[str, Any]] = []

        for start in range(0, len(obs), bs):
            end = min(start + bs, len(obs))
            y_b = torch.from_numpy(cond_ids_np[start:end]).to(self.device, dtype=torch.long)
            c_fm = self.fm.denoising_model.cond_embedder(y_b.reshape(-1)).reshape(
                y_b.shape[0], y_b.shape[1], -1
            )
            mask_ids = torch.full(
                (y_b.numel(),), self.vocab.mask_id, device=self.device, dtype=torch.long
            )
            mask_emb_de = self.vae.cond_embedder(mask_ids).reshape(
                y_b.shape[0], y_b.shape[1], -1
            )

            res = self.fm.sample_with_attn_weights(
                batch_size=c_fm.shape[0],
                seq_length=self.seq_length,
                n_sample_steps=int(steps),
                y=c_fm,
                guidance_weights=eff_w,
            )
            z_b = res["final_sample"]
            all_attn.extend(res["attn_weights"])

            recon_b = torch.clamp(self.vae.decoder(z_b, mask_emb_de), 0.0, 1.0)
            recon_b[recon_b < 1e-4] = 0.0
            x_chunks.append(recon_b.detach().cpu().numpy())

        x_all = np.concatenate(x_chunks, axis=0)
        adata_out = self._postprocess_output(
            x_all, obs=obs, inverse_scale=inverse_scale, trim_pad=trim_pad
        )
        return {"adata": adata_out, "attn_weights": all_attn}

    @torch.inference_mode()
    def transfer(
        self,
        adata: ad.AnnData,
        target_cond: dict[str, Any] | pd.DataFrame | np.ndarray | torch.Tensor,
        input_cond: Literal["obs"] | dict[str, Any] | pd.DataFrame | np.ndarray | torch.Tensor | None = None,
        *,
        batch_size: int | None = None,
        min_clip: float = 1e-4,
        return_attention: bool = False,
        inverse_scale: bool = False,
        trim_pad: bool = False,
    ) -> ad.AnnData | tuple[ad.AnnData, dict[str, np.ndarray]]:
        """Transfer cells from source condition(s) (or ``input_cond='obs'``) to ``target_cond``."""
        aligned, x_mat = self._prepare_input(adata)
        self.vae.to(self.device).eval()

        n_obs = int(aligned.n_obs)
        n_cols = len(self.condition_cols)
        bs = _resolve_infer_batch_size(self.device, batch_size, self.config.train.batch_size)

        if input_cond is None:
            input_ids_np = np.full((n_obs, n_cols), self.vocab.mask_id, dtype=np.int64)
        elif isinstance(input_cond, str) and input_cond == "obs":
            input_ids_np = self.vocab.encode_obs(aligned.obs)
        else:
            input_ids_np, _ = self._resolve_cond_array_and_obs(input_cond, n_obs)

        out_obs = aligned.obs.copy()
        if isinstance(target_cond, dict):
            if (
                isinstance(input_cond, str)
                and input_cond == "obs"
                and set(target_cond.keys()) != set(self.condition_cols)
            ):
                unknown = set(target_cond.keys()) - set(self.condition_cols)
                if unknown:
                    raise ValueError(f"Unknown condition columns: {sorted(unknown)}")
                target_obs = aligned.obs[self.condition_cols].copy()
                for col, val in target_cond.items():
                    target_obs[col] = val
                    out_obs[col] = val
                target_ids_np = self.vocab.encode_obs(target_obs)
            else:
                row_ids = self.vocab.encode_dict(target_cond, allow_partial=True)
                target_ids_np = np.repeat(row_ids[None, :], n_obs, axis=0)
                for col, val in target_cond.items():
                    out_obs[col] = val
        elif isinstance(target_cond, pd.DataFrame):
            target_ids_np = self.vocab.encode_obs(target_cond)
            for col in self.condition_cols:
                if col in target_cond.columns:
                    out_obs[col] = target_cond[col].values
        else:
            target_ids_np, _ = self._resolve_cond_array_and_obs(target_cond, n_obs)

        recon_chunks: list[np.ndarray] = []
        attn_accum: dict[str, list[np.ndarray]] = {}

        for start in range(0, n_obs, bs):
            end = min(start + bs, n_obs)
            x_b = self._slice_x_batch(x_mat, start, end)
            y_in_b = torch.from_numpy(input_ids_np[start:end]).to(self.device, dtype=torch.long)
            y_tgt_b = torch.from_numpy(target_ids_np[start:end]).to(self.device, dtype=torch.long)

            c_in = self._embed_vae_cond(y_in_b)
            c_tgt = self._embed_vae_cond(y_tgt_b)
            z_b, _, _ = self.vae.encoder(x_b, c_in)

            if return_attention:
                recon_b, attn_b = self.vae.decoder(z_b, c_tgt, is_output_attn_weight=True)
                for layer_idx, layer_attn in enumerate(attn_b):
                    key = f"layer_{layer_idx}"
                    attn_accum.setdefault(key, []).append(layer_attn.detach().cpu().numpy())
            else:
                recon_b = self.vae.decoder(z_b, c_tgt)

            recon_b = torch.clamp(recon_b, 0.0, 1.0)
            if min_clip is not None and min_clip > 0:
                recon_b[recon_b <= min_clip] = 0.0
            recon_chunks.append(recon_b.detach().cpu().numpy())

        x_all = (
            np.concatenate(recon_chunks, axis=0)
            if recon_chunks
            else np.empty((0, self.num_genes), dtype=np.float32)
        )
        out_adata = self._postprocess_output(
            x_all, obs=out_obs, inverse_scale=inverse_scale, trim_pad=trim_pad
        )

        if return_attention:
            attn_out = {k: np.concatenate(v, axis=0) for k, v in attn_accum.items()}
            return out_adata, attn_out
        return out_adata

    def save_vae_ckpt(self, path: str | Path) -> Path:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.vae.state_dict(), ckpt_path)
        return ckpt_path

    def load_vae_ckpt(self, path: str | Path) -> None:
        state_dict = _load_torch_state_dict(Path(path), self.device)
        load_state_dict_compat(self.vae, state_dict, strict=True)
        self.vae.to(self.device)
        self.is_trained = True

    def save_fm_ckpt(self, path: str | Path) -> Path:
        if self.fm is None:
            raise RuntimeError("No Flow Matching model initialized to save.")
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.fm.denoising_model.state_dict(), ckpt_path)
        return ckpt_path

    def load_fm_ckpt(self, path: str | Path) -> None:
        if self.fm is None:
            self.init_fm()
        state_dict = _load_torch_state_dict(Path(path), self.device)
        load_state_dict_compat(self.fm.denoising_model, state_dict, strict=True)
        self.fm.to(self.device)
        self.is_fm_trained = True

    def save(self, path: str | Path) -> Path:
        """Save full SAVE bundle (config, vocab, metadata, optional scaler, weights) to ``path``."""
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.fm is not None:
            dm = self.fm.denoising_model
            self.config.fm.hidden_dim = int(dm.hidden_dim)
            self.config.fm.embedding_dim = int(dm.embedding_dim)
            self.config.fm.n_blocks = int(getattr(dm, "n_blocks", len(dm.blocks)))
            self.config.fm.num_heads = int(getattr(dm, "num_heads", self.config.fm.num_heads))
            self.config.fm.dropout_prob = float(getattr(dm, "dropout_prob", self.config.fm.dropout_prob))
            self.config.fm.conditioning_probability = float(dm.conditioning_probability)
            self.config.fm.block_type = dm.block_type
            self.config.fm.path_type = self.fm.path_type
            self.config.fm.sigma = float(self.fm.sigma)
            self.config.fm.use_ot = bool(self.fm.use_ot)
            self.config.fm.use_snr_weighting = bool(self.fm.use_snr_weighting)
            self.config.fm.antithetic_time_sampling = bool(self.fm.antithetic_time_sampling)

        with open(out_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        with open(out_dir / "vocab.json", "w") as f:
            json.dump(self.vocab.to_dict(), f, indent=2)

        if self.scaler is not None:
            np.savez(
                out_dir / "scaler.npz",
                scale=self.scaler.scale_.astype(np.float32),
                max_abs=(
                    self.scaler.max_abs_.astype(np.float32)
                    if self.scaler.max_abs_ is not None
                    else self.scaler.scale_.astype(np.float32)
                ),
            )

        metadata = {
            "version": "0.2.0",
            "cond_encoding": self.config.cond_encoding,
            "num_genes": int(self.num_genes),
            "pad_idx": int(self.pad_idx),
            "var_names": list(self.var_names),
            "condition_cols": list(self.condition_cols),
            "is_trained": bool(self.is_trained),
            "is_fm_trained": bool(self.is_fm_trained),
            "has_scaler": self.scaler is not None,
            "has_fm": self.fm is not None,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.save_vae_ckpt(out_dir / "vae.pt")
        if self.fm is not None:
            self.save_fm_ckpt(out_dir / "fm.pt")

        return out_dir

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | int | torch.device = "auto",
    ) -> SAVE:
        """Load a saved SAVE bundle from directory ``path``."""
        src_dir = Path(path)
        with open(src_dir / "config.json") as f:
            cfg = SAVEConfig.from_dict(json.load(f))
        with open(src_dir / "vocab.json") as f:
            vocab = ConditionVocabulary.from_dict(json.load(f))
        with open(src_dir / "metadata.json") as f:
            metadata = json.load(f)

        scaler: MaxAbsExpressionScaler | None = None
        scaler_path = src_dir / "scaler.npz"
        if scaler_path.exists():
            with np.load(scaler_path) as npz:
                scaler = MaxAbsExpressionScaler.from_dict(
                    {
                        "scale": npz["scale"],
                        "max_abs": npz["max_abs"] if "max_abs" in npz else npz["scale"],
                    }
                )

        instance = cls(
            adata=None,
            condition_cols=vocab.condition_cols,
            config=cfg,
            device=device,
            scale_data=bool(cfg.scale_data),
            pad_to_blocks=bool(cfg.pad_to_blocks),
            cond_encoding=cfg.cond_encoding,
            vocab=vocab,
            scaler=scaler,
            n_genes=int(metadata["num_genes"]),
            var_names=metadata.get("var_names"),
            pad_idx=int(metadata.get("pad_idx", metadata["num_genes"])),
        )
        instance.load_vae_ckpt(src_dir / "vae.pt")
        instance.is_trained = bool(metadata.get("is_trained", True))

        fm_path = src_dir / "fm.pt"
        if fm_path.exists():
            instance.load_fm_ckpt(fm_path)
            instance.is_fm_trained = bool(metadata.get("is_fm_trained", True))

        return instance


__all__ = ["SAVE"]
