"""Compatibility shim providing legacy v1 ``SAVE`` and checkpoint helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from einops import rearrange
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sc_save.config import FMConfig, SAVEConfig, TrainConfig, VAEConfig
from sc_save.data.preprocessing import align_genes
from sc_save.model import SAVE as V2SAVE
from sc_save.nn.vae import VAE
from sc_save.nn.flow_matching import FM
from sc_save.training.fm_trainer import train_fm
from sc_save.training.runtime import analytic_kl_div
from sc_save.training.vae_trainer import train_vae
from sc_save_public.models.VAE_train import vae_train
from sc_save_public.models.flow_matching_train import flow_matching_train
from sc_save_public.models.utils.process_h5ad import convert_adata_to_dataloader

CHUNK_SIZE = 40000


class DummyContextManager:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def kl_div(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return analytic_kl_div(mu, var)


def reindex(adata: AnnData, genes, chunk_size: int = CHUNK_SIZE) -> AnnData:
    return align_genes(adata, [str(g) for g in genes], pad_missing=True)


def batch_scale(adata: AnnData, chunk_size: int = CHUNK_SIZE, is_uniform: bool = False):
    for b in adata.obs["batch"].unique():
        idx = np.where(adata.obs["batch"] == b)[0]
        if is_uniform:
            scaler = StandardScaler(copy=False, with_mean=False, with_std=False).fit(
                adata.X[idx]
            )
        else:
            scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])

        tr = tqdm(range(len(idx) // chunk_size + 1), ncols=80)
        tr.set_description_str(desc="batch_scale")
        for i in tr:
            adata.X[idx[i * chunk_size : (i + 1) * chunk_size]] = scaler.transform(
                adata.X[idx[i * chunk_size : (i + 1) * chunk_size]]
            )
    return adata


def load_ckpt(save_model: V2SAVE, ckpt_path: str) -> None:
    save_model.load_vae_ckpt(ckpt_path)


def save_ckpt(save_model: V2SAVE, ckpt_path: str) -> None:
    save_model.save_vae_ckpt(ckpt_path)


class SAVE(V2SAVE):
    """Backward-compatible v1 wrapper around ``sc_save.SAVE``."""

    def __init__(
        self,
        adata: sc.AnnData,
        device,
        is_initialized: bool = True,
        is_print_model: bool = False,
        seed: int = 1202,
        hidden_dim: int = 64,
        enc_dim: int = 22,
        expand_dim: int = 256,
        enc_num_heads: int = 4,
        dec_num_heads: int = 4,
        enc_depth: int = 3,
        dec_depth: int = 3,
        enc_affine: bool = True,
        enc_norm_type: str = "ln",
        dec_norm_init: bool = True,
        dec_blk_type: str = "adaLN",
        condition_cols=("batch",),
        is_cond_reuse: bool = False,
        is_reshape_first: bool = True,
        gene_block_size: int = 40,
        latent_block_size: int = 1,
        is_flash_attn: bool = False,
        out_layer_dropout: float = 0.4,
        is_enc_linear: bool = False,
        is_dec_linear: bool = False,
        is_out_moe: bool = False,
        enc_attn_dropout: float = 0.2,
        dec_attn_dropout: float = 0.0,
        is_enc_cond: bool = True,
        is_multi_codebook: bool = False,
        is_linear_replace_attn: bool = False,
        emb_ary: np.ndarray | None = None,
        outer_emb_dim=(768, 256),
        fm: FM | None = None,
        cond_encoding: str = "legacy",
        **kwargs,
    ) -> None:
        cfg = SAVEConfig(
            vae=VAEConfig(
                hidden_dim=hidden_dim,
                enc_dim=enc_dim,
                expand_dim=expand_dim,
                enc_num_heads=enc_num_heads,
                dec_num_heads=dec_num_heads,
                enc_depth=enc_depth,
                dec_depth=dec_depth,
                enc_affine=enc_affine,
                enc_norm_type=enc_norm_type,
                dec_norm_init=dec_norm_init,
                dec_blk_type=dec_blk_type,
                is_initialize=is_initialized,
                is_reshape_first=is_reshape_first,
                gene_block_size=gene_block_size,
                latent_block_size=latent_block_size,
                is_enc_cond=is_enc_cond,
                is_multi_codebook=is_multi_codebook,
                enc_attn_dropout=enc_attn_dropout,
                dec_attn_dropout=dec_attn_dropout,
                out_layer_dropout=out_layer_dropout,
                is_flash_attn=is_flash_attn,
                outer_emb_dim=outer_emb_dim,
            ),
            fm=FMConfig(
                hidden_dim=expand_dim * 2,
                embedding_dim=expand_dim * 2,
                num_heads=kwargs.get("fm_num_heads", enc_num_heads),
            ),
            train=TrainConfig(seed=seed),
            cond_encoding=cond_encoding,
            pad_to_blocks=False,
            scale_data=False,
            seed=seed,
        )
        super().__init__(
            adata=adata,
            condition_cols=list(condition_cols),
            config=cfg,
            device=device,
            scale_data=False,
            pad_to_blocks=False,
            cond_encoding=cond_encoding,
            outer_embeddings=emb_ary,
        )
        self.is_flash_attn = is_flash_attn
        self.is_cond_reuse = is_cond_reuse
        self.is_multi_codebook = is_multi_codebook
        self.types_of_conds = len(self.condition_cols)
        self.emb_ary = emb_ary
        self.fm = fm
        self.context = (
            torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if is_flash_attn and self.device.type == "cuda"
            else DummyContextManager()
        )
        if is_print_model:
            print(self.model)

    def get_cond_label_encoders(self):
        return self.vocab.label_encoder_dict

    def convert_adata_to_dataloader(
        self,
        adata,
        batch_size,
        is_shuffle,
        is_droplast=False,
        condition_cols=("batch",),
        is_multi_codebook=False,
    ):
        total_idx = self.vocab.encode_obs(adata.obs)
        ary = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        if self.emb_ary is not None:
            dataset = TensorDataset(
                torch.from_numpy(np.asarray(ary, dtype=np.float32)),
                torch.from_numpy(total_idx.astype(np.int32)),
                torch.from_numpy(np.asarray(self.emb_ary, dtype=np.float32)),
            )
        else:
            dataset = TensorDataset(
                torch.from_numpy(np.asarray(ary, dtype=np.float32)),
                torch.from_numpy(total_idx.astype(np.int32)),
            )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=is_shuffle,
            drop_last=is_droplast,
        )

    def train(
        self,
        epoch: int = 15,
        batch_size: int = 64,
        lr: float = 0.00015,
        seed: int = 1202,
        is_grad_clip: bool = True,
        iter: int = 8000,
        weight_decay: float = 5e-4,
        is_lr_scheduler: bool = False,
        col_msk_threshold: float = 0.6,
        loss_monitor=None,
        kl_scale: float = 0.5,
        tune_session=None,
        max_sample_mask_prob=None,
        is_col_wise_mask: bool = False,
        recon_criterion: str = "bce",
        is_partial_sample_mask: bool = True,
        is_enc_dec_cond_diff: bool = False,
        use_amp_vae: bool = False,
        **kwargs,
    ):
        y = self.vocab.encode_obs(self.adata.obs)
        train_vae(
            vae=self.vae,
            x=self.adata.X,
            y=y,
            emb=self.emb_ary,
            num_epoch=epoch,
            batch_size=min(int(batch_size), max(1, self.adata.n_obs)),
            kl_scale=kl_scale,
            device=self.device,
            lr=lr,
            grad_clip=1.0 if is_grad_clip else 0.0,
            seed=seed,
            weight_decay=weight_decay,
            col_msk_threshold=col_msk_threshold,
            loss_monitor=loss_monitor,
            tune_session=tune_session,
            is_col_wise_mask=is_col_wise_mask,
            is_flash_attn=self.is_flash_attn,
            use_amp_vae=use_amp_vae,
            is_multi_codebook=self.is_multi_codebook,
            recon_criterion=recon_criterion,
            is_emb_ary=self.emb_ary is not None,
        )
        self.is_trained = True

    def fm_train(
        self,
        epoch: int = 15,
        batch_size: int = 64,
        lr: float = 0.00015,
        seed: int = 1202,
        is_grad_clip: bool = True,
        iter: int = 8000,
        weight_decay: float = 5e-4,
        is_lr_scheduler: bool = False,
        col_msk_threshold: float = 0.6,
        loss_monitor=None,
        kl_scale: float = 0.5,
        tune_session=None,
        max_sample_mask_prob=None,
        is_col_wise_mask: bool = False,
        recon_criterion: str = "bce",
        is_partial_sample_mask: bool = True,
        is_enc_dec_cond_diff: bool = False,
        use_amp_fm: bool = True,
        **kwargs,
    ):
        if self.fm is None:
            self.init_fm()
        y = self.vocab.encode_obs(self.adata.obs)
        train_fm(
            vae=self.vae,
            fm=self.fm,
            x=self.adata.X,
            y=y,
            emb=self.emb_ary,
            num_epoch=epoch,
            batch_size=min(int(batch_size), max(1, self.adata.n_obs)),
            device=self.device,
            lr=lr,
            grad_clip=1.0 if is_grad_clip else 0.0,
            seed=seed,
            weight_decay=weight_decay,
            col_msk_threshold=col_msk_threshold,
            loss_monitor=loss_monitor,
            tune_session=tune_session,
            is_col_wise_mask=is_col_wise_mask,
            is_flash_attn=self.is_flash_attn,
            use_amp_fm=use_amp_fm,
            is_multi_codebook=self.is_multi_codebook,
            is_emb_ary=self.emb_ary is not None,
        )
        self.is_fm_trained = True
        self.is_trained = True

    def get_latent(self, batch_size: int = 1024, latent_pos: int = 1):
        return self.encode(with_cond=False, latent_pos=latent_pos, batch_size=batch_size)

    def get_latent_from_adata(
        self,
        batch_size: int = 256,
        adata: sc.AnnData | None = None,
        is_encode_with_cond: bool = False,
        latent_pos: int = 0,
    ):
        return self.encode(
            adata=adata,
            with_cond=is_encode_with_cond,
            latent_pos=latent_pos,
            batch_size=batch_size,
        )

    def get_recon(
        self,
        batch_size: int = 256,
        is_monitor_mu_var: bool = False,
        recon_with_cond: bool = True,
    ):
        recon_data = self.reconstruct(
            adata=self.adata,
            with_cond=recon_with_cond,
            batch_size=batch_size,
            inverse_scale=False,
            trim_pad=False,
        )
        if not is_monitor_mu_var:
            return recon_data

        with torch.inference_mode():
            y = self.vocab.encode_obs(self.adata.obs)
            total_kl = []
            for start in range(0, self.adata.n_obs, batch_size):
                end = min(start + batch_size, self.adata.n_obs)
                x_b = self._slice_x_batch(self.adata.X, start, end)
                y_b = torch.from_numpy(y[start:end]).to(self.device, dtype=torch.long)
                if not recon_with_cond:
                    y_b = torch.full_like(y_b, self.mask_id)
                c_b = self._embed_vae_cond(y_b)
                _, mu, var = self.vae.encoder(x_b, c_b)
                kl = analytic_kl_div(mu.reshape(-1, mu.shape[-1]), var.reshape(-1, var.shape[-1]))
                total_kl.append(kl.item())
        kl_arr = np.asarray(total_kl, dtype=np.float64)
        return recon_data, (float(kl_arr.mean()), float(kl_arr.std()))

    def save_ckpt(self, ckpt_path: str):
        if not self.is_trained:
            print("model not trained!")
            return
        self.save_vae_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path: str):
        self.load_vae_ckpt(ckpt_path)

    def convert_cond_to_idx(self, cond: dict, num_cells: int):
        ids = self.vocab.encode_dict(cond, allow_partial=True)
        return torch.from_numpy(np.repeat(ids[None, :], num_cells, axis=0)).to(
            self.device, dtype=torch.long
        )

    def transfer_cond(
        self,
        input_adata: sc.AnnData,
        target_cond: dict,
        input_cond,
        batch_size: int = 1024,
        latent_pos: int = 0,
        min_clip: float = 1e-4,
        transfer_cond_suffix: str = "",
        verbose: bool = True,
        is_output_attn_weights: bool = False,
    ):
        res = self.transfer(
            adata=input_adata,
            target_cond=target_cond,
            input_cond=input_cond,
            batch_size=batch_size,
            min_clip=min_clip,
            return_attention=is_output_attn_weights,
            inverse_scale=False,
            trim_pad=False,
        )
        if is_output_attn_weights:
            recon_data, attn_weights = res
        else:
            recon_data = res

        if transfer_cond_suffix:
            for k, v in target_cond.items():
                recon_data.obs[k] = str(v) + transfer_cond_suffix

        if is_output_attn_weights:
            return (recon_data, attn_weights)
        return (recon_data,)

    def sampling_noise(self, generate_count: int, mean: float = 0, std: float = 1):
        rand_z = torch.randn(generate_count, self.model.latent_dim, device=self.device)
        return rand_z * std + mean

    def cond_generate_with_guidance(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 1024,
        mean: float = 0,
        std: float = 1,
        omega: float = 0.0,
        input_noise=None,
        verbose: bool = True,
    ):
        if input_noise is not None:
            rand_z = input_noise.to(self.device)
        else:
            rand_z = self.sampling_noise(generate_count, mean, std)
        out = self.decode(
            rand_z,
            cond=target_cond,
            batch_size=batch_size,
            inverse_scale=False,
            trim_pad=False,
        )
        out.X[out.X < 1e-4] = 0
        return out

    def cond_generate_with_guidance_fm(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 1024,
        verbose: bool = True,
        guidance_weights=None,
        seq_length: int = 47,
        sample_steps: int = 5,
        is_output_attn_weights: bool = False,
        solver_type: str = "euler",
        is_print_memory: bool = False,
        use_sde=None,
    ):
        if self.fm is None:
            raise RuntimeError("FM model is not attached.")
        prev_use_sde = self.config.fm.use_sde
        self.config.fm.use_sde = use_sde
        try:
            gen_adata, z_fm = self.generate(
                conditions=target_cond,
                n_cells=generate_count,
                guidance_weight=guidance_weights,
                steps=sample_steps,
                solver=solver_type,
                batch_size=batch_size,
                return_latents=True,
                inverse_scale=False,
                trim_pad=False,
            )
        finally:
            self.config.fm.use_sde = prev_use_sde
        return gen_adata, z_fm

    def cond_generate_with_guidance_fm_attn(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 128,
        verbose: bool = True,
        guidance_weights=None,
        seq_length: int = 47,
        sample_steps: int = 50,
    ):
        return self.generate_with_attention(
            conditions=target_cond,
            n_cells=generate_count,
            guidance_weight=guidance_weights,
            steps=sample_steps,
            batch_size=batch_size,
            inverse_scale=False,
            trim_pad=False,
        )

    @torch.inference_mode()
    def cond_generate_emb(
        self,
        generate_count: int,
        target_cond: dict,
        total_emb: dict[str, Any],
        batch_size: int = 1024,
        mean: float = 0,
        std: float = 1,
        omega: float = 0.0,
        input_noise=None,
        verbose: bool = True,
    ):
        self.vae.to(self.device).eval()
        cond_emb = [
            np.array(total_emb[k][v]).reshape(1, -1).astype(np.float32)
            for k, v in target_cond.items()
        ]
        cond_emb_arr = np.repeat(np.concatenate(cond_emb, axis=1), generate_count, axis=0)
        cond_tensor = torch.from_numpy(cond_emb_arr).to(self.device)
        rand_z = (
            input_noise.to(self.device)
            if input_noise is not None
            else self.sampling_noise(generate_count, mean, std)
        )

        total_recon = []
        for start in range(0, generate_count, batch_size):
            end = min(start + batch_size, generate_count)
            z_b = rand_z[start:end]
            c_b = cond_tensor[start:end]
            embs = torch.split(c_b, self.vae.outer_emb_dim, dim=1)
            emb_cat = torch.cat([self.vae.txt_emb_proj[i](embs[i]) for i in range(len(embs))], dim=1)
            c_proj = rearrange(emb_cat, "b (c e) -> b c e", c=len(self.vae.outer_emb_dim))
            recon_x = torch.clamp(self.vae.decoder(z_b, c_proj), 0.0, 1.0)
            recon_x[recon_x < 1e-4] = 0.0
            total_recon.append(recon_x.detach().cpu().numpy())

        return sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=self.adata.var.copy(),
            obs={k: v for k, v in target_cond.items()},
        )

    def preprocessing_rna(
        self,
        adata: AnnData,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,
        chunk_size: int = CHUNK_SIZE,
        is_batch_scale: bool = True,
        log=None,
    ):
        from sc_save_public.models.utils.process_h5ad import preprocessing_rna as _prep_rna

        return _prep_rna(
            adata=adata,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            n_top_features=n_top_features,
            chunk_size=chunk_size,
            is_batch_scale=is_batch_scale,
            is_ret_scaler=False,
        )

    @torch.inference_mode()
    def zero_shot_embeding(
        self,
        input_adata: AnnData,
        is_data_scaled: bool = False,
        embedding_name: str = "SAVE",
        batch_size: int = 1024,
        latent_pos: int = 1,
    ):
        if not is_data_scaled:
            input_adata = input_adata[:, self.adata.var_names]
            input_adata = self.preprocessing_rna(
                input_adata,
                is_batch_scale=True,
                n_top_features=self.num_genes,
                min_features=0,
                min_cells=0,
            )

        self.vae.to(self.device).eval()
        latents = []
        for start in range(0, input_adata.n_obs, batch_size):
            end = min(start + batch_size, input_adata.n_obs)
            x_b = self._slice_x_batch(input_adata.X, start, end)
            z = self.vae.encoder(x_b)[latent_pos]
            latents.append(z.detach().cpu().numpy())
        input_adata.obsm[embedding_name] = np.concatenate(latents, axis=0)
        return input_adata

    def label_predict(
        self,
        input_adata: AnnData,
        is_data_scaled: bool = False,
        label_key: str = "cell_type",
        embedding_name: str = "SAVE",
        batch_size: int = 1024,
    ):
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import SVC

        test_data = self.zero_shot_embeding(
            input_adata, is_data_scaled=is_data_scaled, embedding_name=embedding_name
        )
        test_latent = test_data.obsm[embedding_name]
        train_latent = self.get_latent()
        train_label = self.adata.obs[label_key].cat.codes.values
        clf = make_pipeline(SVC(gamma="auto")).fit(train_latent, train_label)

        y_pred = clf.predict(test_latent)
        categories = self.adata.obs[label_key].cat.categories
        input_adata.obs["pred_labels"] = categories[y_pred]
        return input_adata

    def del_adata(self) -> None:
        self.adata = None


__all__ = [
    "CHUNK_SIZE",
    "DummyContextManager",
    "FM",
    "SAVE",
    "VAE",
    "batch_scale",
    "convert_adata_to_dataloader",
    "flow_matching_train",
    "kl_div",
    "load_ckpt",
    "reindex",
    "save_ckpt",
    "vae_train",
]
