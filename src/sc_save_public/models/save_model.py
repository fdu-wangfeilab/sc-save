import torch
import scanpy as sc
import numpy as np

from tqdm import tqdm
import scipy
from scipy.sparse import csr
from anndata import AnnData
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
from torch.distributions import Normal, kl_divergence
import time

from sc_save_public.models.utils.process_h5ad import convert_adata_to_dataloader
from sc_save_public.models.VAE_train import vae_train
from sc_save_public.models.flow_matching_train import flow_matching_train
from sc_save_public.models.VAE_model import VAE
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sc_save_public.models.fm.fm import FM
from einops import rearrange

CHUNK_SIZE = 40000


class DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def kl_div(mu, var):
    return (
        kl_divergence(
            Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
        )
        .sum(dim=1)
        .mean()
    )


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list

    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print("There are {} gene in selected genes".format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0] // chunk_size + 1):
            new_X[i * chunk_size : (i + 1) * chunk_size, idx] = adata[
                i * chunk_size : (i + 1) * chunk_size, genes[idx]
            ].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={"var_names": genes})
    return adata


def batch_scale(adata, chunk_size=CHUNK_SIZE, is_uniform=False):
    """
    Batch-specific scale data

    Parameters
    ----------
    adata
        AnnData
    chunk_size
        chunk large data into small chunks

    Return
    ------
    AnnData
    """
    # 就是 b 就是 batchname，如 rna，atac 的数据，b就是('rna','atac')
    for b in adata.obs["batch"].unique():
        # 取某个batch 所有数据的 index
        idx = np.where(adata.obs["batch"] == b)[0]

        # 对某个batch中的数据做归一化, 这样也是对列做归一化
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


class SAVE:
    def __init__(
        self,
        adata: sc.AnnData,
        device,
        is_initialized=True,
        is_print_model=False,
        seed=1202,
        hidden_dim=64,
        enc_dim=22,
        expand_dim=256,
        enc_num_heads=4,
        dec_num_heads=4,
        enc_depth=3,
        dec_depth=3,
        enc_affine=True,
        enc_norm_type="ln",
        dec_norm_init=True,
        dec_blk_type="adaLN",
        condition_cols=["batch"],
        is_cond_reuse=False,
        is_reshape_first=True,
        gene_block_size=40,
        latent_block_size=1,
        is_flash_attn=False,
        out_layer_dropout=0.4,
        is_enc_linear=False,
        is_dec_linear=False,
        is_out_moe=False,
        enc_attn_dropout=0.2,
        dec_attn_dropout=0.0,
        is_enc_cond=True,
        is_multi_codebook=False,
        is_linear_replace_attn=False,
        emb_ary: np.ndarray = None,
        outer_emb_dim: list[int] = [768, 256],
        fm: FM = None,
        **kwargs,
    ) -> None:
        self.adata = adata.copy()
        self.num_cells = adata.shape[0]
        self.num_genes = adata.shape[1]
        self.device = device
        self.is_trained = False
        self.emb_ary = emb_ary

        self.is_flash_attn = is_flash_attn
        self.context = (
            torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
            if is_flash_attn
            else DummyContextManager()
        )

        self.enc_dim = enc_dim
        self.expand_dim = expand_dim
        self.is_cond_reuse = is_cond_reuse

        self.condition_cols = condition_cols
        self.label_encoder_dict = self.get_cond_label_encoders()
        self.is_multi_codebook = is_multi_codebook
        self.mask_id = 0

        self.types_of_conds = len(condition_cols)

        # to stablize model initial performance
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.fm = fm

        self.model = VAE(
            input_dim=self.num_genes,
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
            len_col_comb=len(self.condition_cols),
            mask_id=self.mask_id,
            is_multi_codebook=self.is_multi_codebook,
            label_encoder_dict=self.label_encoder_dict,
            is_reshape_first=is_reshape_first,
            gene_block_size=gene_block_size,
            latent_block_size=latent_block_size,
            out_layer_dropout=out_layer_dropout,
            is_enc_linear=is_enc_linear,
            is_dec_linear=is_dec_linear,
            is_out_moe=is_out_moe,
            is_enc_cond=is_enc_cond,
            enc_attn_dropout=enc_attn_dropout,
            dec_attn_dropout=dec_attn_dropout,
            is_linear_replace_attn=is_linear_replace_attn,
            outer_emb_dim=outer_emb_dim,
        )

        if is_initialized:
            print(f"initialze model weight with seed {seed}")
        if is_print_model:
            print(self.model)

    def get_cond_label_encoders(self):
        condition_cols = self.condition_cols
        label_encoder_dict = {}

        for col in condition_cols:
            # get all cond in one col
            cond_name = ["<mask>"] + list(self.adata.obs[col].unique())
            label_encoder = LabelEncoder()
            label_encoder.fit(cond_name)
            # print("--------------------------------")
            # print(f"col: {col}")
            # print(f"label_encoder.classes_: {label_encoder.classes_}")
            # print("--------------------------------")
            label_encoder_dict[col] = label_encoder

        return label_encoder_dict

    def convert_adata_to_dataloader(
        self,
        adata,
        batch_size,
        is_shuffle,
        is_droplast=False,
        condition_cols=["batch"],
        is_multi_codebook=False,
        # is_cond_reuse=False,
    ):
        # simple converter
        id_max = 0
        total_idx = []
        if is_multi_codebook:
            for col in condition_cols:
                label_encoder = self.label_encoder_dict[col]
                idx = label_encoder.transform(adata.obs[col])
                total_idx.append(idx.reshape(-1, 1))
        else:
            for col in condition_cols:
                label_encoder = self.label_encoder_dict[col]
                idx = label_encoder.transform(adata.obs[col])
                # idx = adata.obs[col].cat.codes.values.astype("int16") + id_max
                total_idx.append(idx.reshape(-1, 1))
                id_max = id_max + len(label_encoder.classes_)

        total_idx = np.hstack(total_idx)

        if isinstance(adata.X, np.ndarray):
            ary = adata.X
        else:
            ary = adata.X.toarray()

        if self.emb_ary is not None:
            dataset = TensorDataset(
                torch.from_numpy(ary.astype(np.float32)),
                torch.from_numpy(total_idx.astype(np.int32)),
                torch.from_numpy(self.emb_ary.astype(np.float32)),
            )
        else:
            dataset = TensorDataset(
                torch.from_numpy(ary.astype(np.float32)),
                torch.from_numpy(total_idx.astype(np.int32)),
            )

        # 其实这里 id_max 是所有条件编码的长度
        print(f"cond encoding max: {id_max}")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=4,
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
        col_msk_threshold: int = 0.6,
        loss_monitor=None,
        kl_scale: float = 0.5,
        tune_session=None,
        max_sample_mask_prob=None,
        is_col_wise_mask=False,
        recon_criterion="bce",
        is_partial_sample_mask=True,
        is_enc_dec_cond_diff=False,
        **kwargs,
    ):
        dataloader = self.convert_adata_to_dataloader(
            self.adata,
            batch_size=batch_size,
            is_shuffle=True,
            condition_cols=self.condition_cols,
            # is_cond_reuse=self.is_cond_reuse,
            is_multi_codebook=self.is_multi_codebook,
        )

        # if epoch * len(dataloader) < iter:
        #     epoch = int(iter / len(dataloader))

        # print(f"total iter: {epoch * len(dataloader)}")

        self.model.to(self.device)

        vae_train(
            vae=self.model,
            dataloader=dataloader,
            num_epoch=epoch,
            device=self.device,
            kl_scale=kl_scale,
            is_tensorboard=False,
            grad_clip=is_grad_clip,
            lr=lr,
            seed=seed,
            weight_decay=weight_decay,
            is_lr_scheduler=is_lr_scheduler,
            col_msk_threshold=col_msk_threshold,
            loss_monitor=loss_monitor,
            tune_session=tune_session,
            max_noise_prob=max_sample_mask_prob,
            is_col_wise_mask=is_col_wise_mask,
            is_flash_attn=self.is_flash_attn,
            is_partial_mask=is_partial_sample_mask,
            is_multi_codebook=self.is_multi_codebook,
            recon_criterion=recon_criterion,
            is_enc_dec_cond_diff=is_enc_dec_cond_diff,
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
        col_msk_threshold: int = 0.6,
        loss_monitor=None,
        kl_scale: float = 0.5,
        tune_session=None,
        max_sample_mask_prob=None,
        is_col_wise_mask=False,
        recon_criterion="bce",
        is_partial_sample_mask=True,
        is_enc_dec_cond_diff=False,
        **kwargs,
    ):
        dataloader = self.convert_adata_to_dataloader(
            self.adata,
            batch_size=batch_size,
            is_shuffle=True,
            condition_cols=self.condition_cols,
            # is_cond_reuse=self.is_cond_reuse,
            is_multi_codebook=self.is_multi_codebook,
        )

        self.model.to(self.device)

        flow_matching_train(
            fm=self.fm,
            vae=self.model,
            dataloader=dataloader,
            num_epoch=epoch,
            device=self.device,
            kl_scale=kl_scale,
            is_tensorboard=False,
            grad_clip=is_grad_clip,
            lr=lr,
            seed=seed,
            weight_decay=weight_decay,
            is_lr_scheduler=is_lr_scheduler,
            col_msk_threshold=col_msk_threshold,
            loss_monitor=loss_monitor,
            tune_session=tune_session,
            max_noise_prob=max_sample_mask_prob,
            is_col_wise_mask=is_col_wise_mask,
            is_flash_attn=self.is_flash_attn,
            is_partial_mask=is_partial_sample_mask,
            is_multi_codebook=self.is_multi_codebook,
            recon_criterion=recon_criterion,
            is_enc_dec_cond_diff=is_enc_dec_cond_diff,
            is_emb_ary=self.emb_ary is not None,
        )

        self.is_trained = True

    def get_latent(self, batch_size: int = 1024, latent_pos: int = 1):
        dataloader = convert_adata_to_dataloader(
            adata=self.adata,
            batch_size=batch_size,
            is_shuffle=False,
            condition_cols=self.condition_cols,
        )
        self.model.to(self.device)
        self.model.eval()
        latents = []
        for x, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x, idx = x.float().to(self.device), idx.long().to(self.device)
            z = self.model.encoder(x)[latent_pos]
            latents.append(z.detach().cpu().numpy())
        return np.concatenate(latents, axis=0)

    def get_latent_from_adata(
        self,
        batch_size: int = 256,
        adata: sc.AnnData = None,
        is_encode_with_cond: bool = False,
        latent_pos: int = 0,
    ):
        dataloader = self.convert_adata_to_dataloader(
            adata=adata,
            batch_size=batch_size,
            is_shuffle=False,
            condition_cols=self.condition_cols,
        )

        self.model.to(self.device)
        self.model.eval()

        total_recon = []
        for x, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x, idx = x.float().to(self.device), idx.long().to(self.device)
            y_input = rearrange(idx, "b c -> (b c)")

            with torch.no_grad():
                if is_encode_with_cond:
                    c_input = self.model.cond_embedder(y_input)
                    c_input = rearrange(c_input, "(b c) e -> b c e", b=idx.shape[0])
                else:
                    y_input = (
                        torch.ones_like(y_input).to(self.device) * self.model.mask_id
                    )
                    c_input = self.model.cond_embedder(y_input)
                    c_input = rearrange(c_input, "(b c) e -> b c e", b=idx.shape[0])

                z = self.model.encoder(x, c_input)[latent_pos]

            total_recon.append(z.detach().cpu().numpy())

        return np.concatenate(total_recon, axis=0)

    def get_recon(
        self,
        batch_size: int = 256,
        is_monitor_mu_var: bool = False,
        recon_with_cond: bool = True,
    ):
        dataloader = self.convert_adata_to_dataloader(
            adata=self.adata,
            batch_size=batch_size,
            is_shuffle=False,
            condition_cols=self.condition_cols,
        )

        self.model.to(self.device)
        self.model.eval()

        total_recon = []
        total_kl = []
        for x, idx in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x, idx = x.float().to(self.device), idx.long().to(self.device)
            y_input = rearrange(idx, "b c -> (b c)")

            with torch.no_grad():
                if recon_with_cond:
                    c_input = self.model.cond_embedder(y_input)
                    c_input = rearrange(c_input, "(b c) e -> b c e", b=idx.shape[0])
                else:
                    y_input = (
                        torch.ones_like(y_input).to(self.device) * self.model.mask_id
                    )
                    c_input = self.model.cond_embedder(y_input)
                    c_input = rearrange(c_input, "(b c) e -> b c e", b=idx.shape[0])

                z, mu, var = self.model.encoder(x, c_input)

                if is_monitor_mu_var:
                    kl = kl_div(mu, var)
                    total_kl.append(kl.detach().cpu().item())

                recon = self.model.decoder(z, c_input)
            recon = torch.clamp(recon, 0, 1)
            total_recon.append(recon.detach().cpu().numpy())

        total_kl = np.array(total_kl)
        recon_data = sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=self.adata.var,
            obs=self.adata.obs,
        )
        if is_monitor_mu_var:
            return recon_data, (total_kl.mean(), total_kl.std())
        else:
            return recon_data

    def save_ckpt(self, ckpt_path: str):
        if not self.is_trained:
            print("model not trained!")
            return

        torch.save(self.model.state_dict(), ckpt_path)
        print(f"save ckpt to {ckpt_path}!")

    def load_ckpt(self, ckpt_path: str):
        try:
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.is_trained = True
        print(f"successfully load {ckpt_path}!")

    def convert_cond_to_idx(self, cond: dict, num_cells: int):
        target_keys = set(cond.keys())
        cond_dict_keys = set(self.label_encoder_dict.keys())
        target_cond_ids = []
        if target_keys == cond_dict_keys:
            assert cond.keys() == self.label_encoder_dict.keys(), (
                "input keys don't match condition keys's order"
            )

            for k in self.label_encoder_dict.keys():
                target_cond_ids.append(
                    self.label_encoder_dict[k].transform([cond[k]])[0]
                )
        elif target_keys.issubset(cond_dict_keys):
            # 只转换一部分条件, 其他条件用 mask 代替
            print("cond subset transfer")
            for k in self.label_encoder_dict.keys():
                if k in target_keys:
                    target_cond_ids.append(
                        self.label_encoder_dict[k].transform([cond[k]])[0]
                    )
                else:
                    target_cond_ids.append(self.mask_id)

        total_idx = []
        for target_batch_id in target_cond_ids:
            idx = (
                torch.ones(num_cells, dtype=torch.long).to(self.device)
                * target_batch_id
            )
            total_idx.append(idx.reshape(-1, 1))
        # col combine
        total_idx = torch.hstack(total_idx)
        return total_idx

    def transfer_cond(
        self,
        input_adata: sc.AnnData,
        target_cond: dict,
        input_cond,
        batch_size: int = 1024,
        latent_pos=0,
        min_clip=1e-4,
        transfer_cond_suffix="",
        verbose=True,
        is_output_attn_weights=False,
    ):
        self.model.to(self.device)
        self.model.eval()

        if isinstance(input_adata.X, np.ndarray):
            ary = input_adata.X
        else:
            ary = input_adata.X.toarray()

        target_cond_id = self.convert_cond_to_idx(target_cond, ary.shape[0])

        if input_cond is None:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(ary.astype(np.float32)), target_cond_id
            )
        else:
            input_cond_id = self.convert_cond_to_idx(input_cond, ary.shape[0])
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(ary.astype(np.float32)), input_cond_id, target_cond_id
            )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        if verbose:
            dataloader = tqdm(dataloader, ncols=80, total=len(dataloader))

        total_recon = []
        if is_output_attn_weights:
            total_attn_weights = {}

        if input_cond is None:
            for x, idx in dataloader:
                x, idx = (
                    x.float().to(self.device),
                    idx.long().to(self.device),
                )
                y_flat = rearrange(idx, "b c -> (b c)")
                y_uncond = torch.ones_like(y_flat).to(self.device) * self.mask_id

                c = self.model.cond_embedder(y_flat)
                c = rearrange(c, "(b c) e -> b c e", b=idx.shape[0])

                c_uncond = self.model.cond_embedder(y_uncond)
                c_uncond = rearrange(c_uncond, "(b c) e -> b c e", b=idx.shape[0])

                z, _, _ = self.model.encoder(x, c_uncond)

                recon_x = self.model.decoder(z, c)

                recon_x = torch.clamp(recon_x, 0, 1)
                total_recon.append(recon_x.detach().cpu().numpy())
        else:
            for x, input_cond_id, target_cond_id in dataloader:
                x, input_cond_id, target_cond_id = (
                    x.float().to(self.device),
                    input_cond_id.long().to(self.device),
                    target_cond_id.long().to(self.device),
                )
                y_input = rearrange(input_cond_id, "b c -> (b c)")
                y_target = rearrange(target_cond_id, "b c -> (b c)")

                c_input = self.model.cond_embedder(y_input)
                c_input = rearrange(
                    c_input, "(b c) e -> b c e", b=input_cond_id.shape[0]
                )

                c_target = self.model.cond_embedder(y_target)
                c_target = rearrange(
                    c_target, "(b c) e -> b c e", b=target_cond_id.shape[0]
                )

                z, mu, var = self.model.encoder(x, c_input)
                if is_output_attn_weights:
                    recon_x, attn_weights = self.model.decoder(
                        z, c_target, is_output_attn_weight=True
                    )
                    attn_weights = {
                        "layer_" + str(i): attn_weights[i].detach().cpu().numpy()
                        for i in range(len(attn_weights))
                    }

                    for k in attn_weights.keys():
                        if k not in total_attn_weights:
                            total_attn_weights[k] = []
                        total_attn_weights[k].append(attn_weights[k])

                else:
                    recon_x = self.model.decoder(z, c_target)
                recon_x = torch.clamp(recon_x, 0, 1)
                total_recon.append(recon_x.detach().cpu().numpy())

        recon_data = sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=input_adata.var,
            obs={k: v + transfer_cond_suffix for k, v in target_cond.items()},
        )

        # clip the value near to zero
        recon_data.X[recon_data.X <= min_clip] = 0
        if is_output_attn_weights:
            for k in total_attn_weights.keys():
                total_attn_weights[k] = np.concatenate(total_attn_weights[k], axis=0)
            return (recon_data, total_attn_weights)
        else:
            return (recon_data,)

    def sampling_noise(self, generate_count: int, mean=0, std=1):
        # sampling noise
        rand_z = torch.randn(
            generate_count,
            self.model.latent_dim,
        ).to(self.device)
        rand_z = rand_z * std + mean
        return rand_z

    def cond_generate_with_guidance(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 1024,
        mean=0,
        std=1,
        omega=0.0,
        input_noise=None,
        verbose=True,
    ):
        self.model.to(self.device)
        self.model.eval()
        total_recon = []
        assert target_cond.keys() == self.label_encoder_dict.keys(), (
            "input keys don't match condition keys"
        )

        # input cond to id
        target_cond_ids = []
        for k in self.label_encoder_dict.keys():
            target_cond_ids.append(
                self.label_encoder_dict[k].transform([target_cond[k]])[0]
            )

        # sampling noise
        if input_noise is not None:
            rand_z = input_noise.to(self.device)
        else:
            # print("random sampling noise!")
            rand_z = self.sampling_noise(generate_count, mean, std)

        # construct dataloader
        total_idx = []
        for target_batch_id in target_cond_ids:
            idx = (
                torch.ones(generate_count, dtype=torch.long).to(self.device)
                * target_batch_id
            )
            total_idx.append(idx.reshape(-1, 1))
        total_idx = torch.hstack(total_idx)

        dataset = torch.utils.data.TensorDataset(rand_z, total_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        if verbose:
            dataloader = tqdm(dataloader, ncols=80, total=len(dataloader))
        else:
            dataloader = dataloader

        # generate
        for z, idx in dataloader:
            y_flat = rearrange(idx, "b c -> (b c)")
            # y_uncond = torch.ones_like(y_flat).to(self.device) * self.mask_id

            # 条件 embedding
            c = self.model.cond_embedder(y_flat)
            c = rearrange(c, "(b c) e -> b c e", b=idx.shape[0])

            # 全部是 mask embedding
            # c_uncond = self.model.cond_embedder(y_uncond)
            # c_uncond = rearrange(c_uncond, "(b c) e -> b c e", b=idx.shape[0])

            # recon_x_condi = self.model.decoder(z, c)
            recon_x = self.model.decoder(z, c)
            # recon_x_uncondi = self.model.decoder(z, c_uncond)

            # recon_x = (1 + omega) * recon_x_condi - omega * recon_x_uncondi
            recon_x = torch.clamp(recon_x, 0, 1)
            recon_x[recon_x < 1e-4] = 0

            total_recon.append(recon_x.detach().cpu().numpy())

        return sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=self.adata.var,
            obs={k: v for k, v in target_cond.items()},
        )

    def cond_generate_with_guidance_fm(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 1024,
        verbose=True,
        guidance_weights=None,
        seq_length=47,
        sample_steps=5,
        is_output_attn_weights=False,
        solver_type="euler",
        is_print_memory=False,
        use_sde=None,
    ):
        self.model.to(self.device)
        self.model.eval()

        self.fm.denoising_model.eval()
        self.fm.eval()

        total_recon = []
        assert target_cond.keys() == self.label_encoder_dict.keys(), (
            "input keys don't match condition keys"
        )

        # input cond to id
        target_cond_ids = []
        for k in self.label_encoder_dict.keys():
            target_cond_ids.append(
                self.label_encoder_dict[k].transform([target_cond[k]])[0]
            )

        # construct dataloader
        total_idx = []
        for target_batch_id in target_cond_ids:
            idx = (
                torch.ones(generate_count, dtype=torch.long).to(self.device)
                * target_batch_id
            )
            total_idx.append(idx.reshape(-1, 1))
        total_idx = torch.hstack(total_idx)

        dataset = torch.utils.data.TensorDataset(total_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        if verbose:
            dataloader = tqdm(dataloader, ncols=80, total=len(dataloader))
        else:
            dataloader = dataloader

        # generate
        total_z_fm = []
        # 显存统计
        fm_memory_list = []
        vae_memory_list = []

        for idx in dataloader:
            y_flat = rearrange(idx[0], "b c -> (b c)")
            msk_id = torch.ones_like(y_flat).to(self.device) * self.mask_id

            # 条件 embedding
            c = self.fm.denoising_model.cond_embedder(y_flat)
            c = rearrange(c, "(b c) e -> b c e", b=idx[0].shape[0])

            msk_emb_fm = self.fm.denoising_model.cond_embedder(msk_id)
            msk_emb_fm = rearrange(msk_emb_fm, "(b c) e -> b c e", b=idx[0].shape[0])

            mask_emb_de = self.model.cond_embedder(msk_id)
            mask_emb_de = rearrange(mask_emb_de, "(b c) e -> b c e", b=idx[0].shape[0])

            # FM 采样前记录显存
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                mem_before_fm = torch.cuda.memory_allocated(self.device) / 1024**3
            else:
                mem_before_fm = 0.0

            start_time = time.time()
            z = self.fm.sample(
                batch_size=c.shape[0],
                seq_length=seq_length,
                n_sample_steps=sample_steps,
                y=c,
                msk_emb=msk_emb_fm,
                guidance_weights=guidance_weights,
                solver_type=solver_type,
                use_sde=use_sde,
            )

            # FM 采样后记录显存
            if self.device.type == "cuda":
                mem_after_fm = torch.cuda.max_memory_allocated(self.device) / 1024**3
                fm_memory = mem_after_fm - mem_before_fm
            else:
                fm_memory = 0.0
            fm_memory_list.append(fm_memory)

            total_z_fm.append(z.detach().cpu().numpy())

            # VAE 解码前记录显存
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                mem_before_vae = torch.cuda.memory_allocated(self.device) / 1024**3
            else:
                mem_before_vae = 0.0

            with torch.no_grad():
                recon_x = self.model.decoder(z, mask_emb_de)
            end_time = time.time()

            # VAE 解码后记录显存
            if self.device.type == "cuda":
                mem_after_vae = torch.cuda.max_memory_allocated(self.device) / 1024**3
                vae_memory = mem_after_vae - mem_before_vae
            else:
                vae_memory = 0.0
            vae_memory_list.append(vae_memory)

            recon_x = torch.clamp(recon_x, 0, 1)
            recon_x[recon_x < 1e-4] = 0

            total_recon.append(recon_x.detach().cpu().numpy())

        # 打印显存统计
        if is_print_memory:
            print(f"FM 采样时间: {end_time - start_time:.2f} 秒")
            print(f"\n{'=' * 60}")
            print(f"显存统计 (batch_size={batch_size}):")
            print(f"  FM 模型:")
            print(f"    - 平均显存: {np.mean(fm_memory_list):.3f} GB")
            print(f"    - 最大显存: {np.max(fm_memory_list):.3f} GB")
            print(f"    - 最小显存: {np.min(fm_memory_list):.3f} GB")
            print(f"  VAE 解码器:")
            print(f"    - 平均显存: {np.mean(vae_memory_list):.3f} GB")
            print(f"    - 最大显存: {np.max(vae_memory_list):.3f} GB")
            print(f"    - 最小显存: {np.min(vae_memory_list):.3f} GB")
            print(f"  总计:")
            print(
                f"    - 平均显存: {np.mean(fm_memory_list) + np.mean(vae_memory_list):.3f} GB"
            )
            print(f"{'=' * 60}\n")

        return sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=self.adata.var,
            obs={k: v for k, v in target_cond.items()},
        ), np.concatenate(total_z_fm, axis=0)

    def cond_generate_with_guidance_fm_attn(
        self,
        generate_count: int,
        target_cond: dict,
        batch_size: int = 128,
        verbose=True,
        guidance_weights=None,
        seq_length=47,
        sample_steps=50,
    ):
        """
        使用 Flow Matching 生成数据并收集每一步的 attention weights (仅最后一层)。

        Args:
            generate_count: 生成样本数量
            target_cond: 目标条件 dict,如 {'batch': 'batch1', 'cell_type': 'T cells'}
            batch_size: 批次大小
            verbose: 是否显示进度条
            guidance_weights: Classifier Free Guidance 权重
            seq_length: 序列长度
            sample_steps: 采样步数

        Returns:
            dict: {
                'adata': 生成的 AnnData 对象,
                'attn_weights': 所有 batch 的 attention weights 列表
            }
        """
        self.model.to(self.device)
        self.model.eval()
        self.fm.denoising_model.eval()
        self.fm.eval()

        total_recon = []
        all_attn_weights = []

        assert target_cond.keys() == self.label_encoder_dict.keys(), (
            "input keys don't match condition keys"
        )

        # input cond to id
        target_cond_ids = []
        for k in self.label_encoder_dict.keys():
            target_cond_ids.append(
                self.label_encoder_dict[k].transform([target_cond[k]])[0]
            )

        # construct dataloader
        total_idx = []
        for target_batch_id in target_cond_ids:
            idx = (
                torch.ones(generate_count, dtype=torch.long).to(self.device)
                * target_batch_id
            )
            total_idx.append(idx.reshape(-1, 1))
        total_idx = torch.hstack(total_idx)

        dataset = torch.utils.data.TensorDataset(total_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        if verbose:
            dataloader = tqdm(dataloader, ncols=80, total=len(dataloader))
        else:
            dataloader = dataloader

        # generate
        for idx in dataloader:
            y_flat = rearrange(idx[0], "b c -> (b c)")
            msk_id = torch.ones_like(y_flat).to(self.device) * self.mask_id

            # 条件 embedding
            c = self.fm.denoising_model.cond_embedder(y_flat)
            c = rearrange(c, "(b c) e -> b c e", b=idx[0].shape[0])

            mask_emb_de = self.model.cond_embedder(msk_id)
            mask_emb_de = rearrange(mask_emb_de, "(b c) e -> b c e", b=idx[0].shape[0])

            # 使用新的采样方法收集 attention weights
            results = self.fm.sample_with_attn_weights(
                batch_size=c.shape[0],
                seq_length=seq_length,
                n_sample_steps=sample_steps,
                y=c,
                guidance_weights=guidance_weights,
            )

            z = results["final_sample"]
            all_attn_weights.extend(results["attn_weights"])

            with torch.no_grad():
                recon_x = self.model.decoder(z, mask_emb_de)

            recon_x = torch.clamp(recon_x, 0, 1)
            recon_x[recon_x < 1e-4] = 0

            total_recon.append(recon_x.detach().cpu().numpy())

        return {
            "adata": sc.AnnData(
                X=np.concatenate(total_recon, axis=0),
                var=self.adata.var,
                obs={k: v for k, v in target_cond.items()},
            ),
            "attn_weights": all_attn_weights,
        }

    def cond_generate_emb(
        self,
        generate_count: int,
        target_cond: dict,
        total_emb: np.ndarray,
        batch_size: int = 1024,
        mean=0,
        std=1,
        omega=0.0,
        input_noise=None,
        verbose=True,
    ):
        self.model.to(self.device)
        self.model.eval()
        total_recon = []

        cond_emb = [
            np.array(total_emb[k][v]).reshape(1, -1).astype(np.float32)
            for k, v in target_cond.items()
        ]

        cond_emb = np.concatenate(cond_emb, axis=1)

        # if len(cond_emb.shape) == 2:
        #     cond_emb = np.repeat(cond_emb[np.newaxis, :, :], generate_count, axis=0)
        # elif len(cond_emb.shape) == 1:
        #     cond_emb = np.repeat(
        #         cond_emb[np.newaxis, np.newaxis, :], generate_count, axis=0
        #     )
        cond_emb = np.repeat(cond_emb, generate_count, axis=0)
        cond_emb = torch.from_numpy(cond_emb).to(self.device)

        # sampling noise
        if input_noise is not None:
            rand_z = input_noise.to(self.device)
        else:
            # print("random sampling noise!")
            rand_z = self.sampling_noise(generate_count, mean, std)

        # construct dataloader

        dataset = torch.utils.data.TensorDataset(rand_z, cond_emb)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        if verbose:
            dataloader = tqdm(dataloader, ncols=80, total=len(dataloader))
        else:
            dataloader = dataloader

        # generate
        for z, c in dataloader:
            embs = torch.split(c, self.model.outer_emb_dim, dim=1)

            emb_list = []
            for i in range(len(embs)):
                emb_list.append(self.model.txt_emb_proj[i](embs[i]))
            emb = torch.cat(emb_list, dim=1)
            c = rearrange(emb, "b (c e) -> b c e", c=len(self.model.outer_emb_dim))
            recon_x = self.model.decoder(z, c)
            # recon_x_uncondi = self.model.decoder(z, c_uncond)

            # recon_x = (1 + omega) * recon_x_condi - omega * recon_x_uncondi
            recon_x = torch.clamp(recon_x, 0, 1)
            recon_x[recon_x < 1e-4] = 0

            total_recon.append(recon_x.detach().cpu().numpy())

        return sc.AnnData(
            X=np.concatenate(total_recon, axis=0),
            var=self.adata.var,
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
        # 重新设置一遍预处理的参数
        if min_features is None:
            min_features = 600
        if n_top_features is None:
            n_top_features = 2000
        if target_sum is None:
            target_sum = 10000
        # 如果不是系数矩阵的格式则转换为稀疏矩阵
        if not isinstance(adata.X, csr.csr_matrix):
            tmp = scipy.sparse.csr_matrix(adata.X)
            adata.X = None
            adata.X = tmp

        # 筛选出 不是['ERCC', 'MT-', 'mt-'] 开头的 gene
        adata = adata[
            :,
            [
                gene
                for gene in adata.var_names
                if not str(gene).startswith(tuple(["ERCC", "MT-", "mt-"]))
            ],
        ]  # type: ignore

        # 直接调用 scanpy 来过滤 cell,gene 并且标准化数据
        # 要求细胞中至少有 min_features 个基因表达
        sc.pp.filter_cells(adata, min_genes=min_features)
        # 要求基因至少在 min cell 中表达
        sc.pp.filter_genes(adata, min_cells=min_cells)
        # 把每一行的数据放缩到 10000，但是保持和为 10000
        # 将每个细胞的总测量量标准化为相同的值，有助于消除细胞之间的测量偏差，

        # 因为在细胞总数相等的情况下，每个基因在不同细胞之间的相对表达量会更加可比。
        sc.pp.normalize_total(adata, target_sum=target_sum)

        # 使数据更加符合正态分布
        sc.pp.log1p(adata)

        # 保存一份数据在 adata.raw 中
        adata.raw = adata

        # 此处 n_top_features 是 2000
        if isinstance(n_top_features, int) and n_top_features > 0:
            # 此处是取 2000 个HVG
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_features,
                batch_key="batch",
                inplace=False,
                subset=True,
            )
        elif not isinstance(n_top_features, int):
            adata = reindex(adata, n_top_features)

        # batch 采用 maxabs 归一化,因为数据全部>0， 最后范围为 [0，1],
        # 分 chunck 归一化每次相当于 max 取chunk中的最大值
        if is_batch_scale:
            adata = batch_scale(adata, chunk_size=chunk_size)

        return adata

    def zero_shot_embeding(
        self,
        input_adata,
        is_data_scaled=False,
        embedding_name="SAVE",
        batch_size=1024,
        latent_pos=1,
    ):
        if not is_data_scaled:
            input_adata = input_adata[:, self.adata.var_names]
            print(input_adata.shape)
            input_adata = self.preprocessing_rna(
                input_adata,
                is_batch_scale=True,
                n_top_features=self.num_genes,
                min_features=0,
                min_cells=0,
            )

        if isinstance(input_adata.X, np.ndarray):
            ary = input_adata.X
        else:
            ary = input_adata.X.toarray()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(ary.astype(np.float32))
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        self.model.to(self.device)
        self.model.eval()
        latents = []
        for x in tqdm(dataloader, ncols=80, total=len(dataloader)):
            x = x[0].float().to(self.device)
            z = self.model.encoder(x)[latent_pos]
            latents.append(z.detach().cpu().numpy())

        input_adata.obsm[embedding_name] = np.concatenate(latents, axis=0)
        return input_adata

    def label_predict(
        self,
        input_adata,
        is_data_scaled=False,
        label_key="cell_type",
        embedding_name="SAVE",
        batch_size=1024,
    ):
        test_data = self.zero_shot_embeding(
            input_adata, is_data_scaled=is_data_scaled, embedding_name=embedding_name
        )

        test_latent = test_data.obsm[embedding_name]
        train_latent = self.get_latent()
        train_label = self.adata.obs[label_key].cat.codes.values

        # train svm as classifier
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline

        clf = make_pipeline(SVC(gamma="auto"))
        clf.fit(train_latent, train_label)

        n_test = test_latent.shape[0]
        it = n_test // batch_size
        y_pred = []
        for i in range(it):
            input = test_latent[i * batch_size : (i + 1) * batch_size]
            print(input.shape)
            y_p = clf.predict(input)
            y_pred.append(y_p)
        y_pred.append(clf.predict(test_latent[it * batch_size :]))
        y_pred = np.concatenate(y_pred)

        label_encoder = self.adata.obs[label_key].cat
        pred_labels = label_encoder.categories[y_pred]

        input_adata.obs["pred_labels"] = pred_labels
        return input_adata

    def del_adata(self):
        del self.adata
        print("adata deleted!")
