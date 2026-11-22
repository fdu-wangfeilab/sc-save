"""Public conditional generation pipelines built on the SAVE model stack."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

from sc_save_public.models.fm.denoising_model import MLPTimeStep
from sc_save_public.models.fm.fm import FM
from sc_save_public.models.save_model import SAVE

DEFAULT_OUTPUT_DIR = "./outputs/conditional_generation_smoke"
DEFAULT_FORMAL_OUTPUT_DIR = "./outputs/conditional_generation_formal"
DEFAULT_COND_COLS = ["batch", "cell_type"]

FORMAL_SAVE_B_TRAIN = {
    "lr": 0.0001,
    "batch_size": 1024,
    "weight_decay": 0.000025,
    "iter": 3200,
    "epoch": 200,
    "is_lr_scheduler": True,
}

FORMAL_SAVE_B_MODEL = {
    "kl_scale": 0.5,
    "hidden_dim": 64,
    "enc_dim": 64,
    "expand_dim": 512,
    "enc_num_heads": 4,
    "dec_num_heads": 4,
    "enc_depth": 3,
    "dec_depth": 3,
    "enc_affine": True,
    "enc_norm_type": "ln",
    "dec_norm_init": True,
    "dec_blk_type": "adaLN",
    "gene_block_size": 3200,
    "latent_block_size": 3200,
    "is_flash_attn": False,
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id: int) -> torch.device:
    if device_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_id}")


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "task"


def _load_torch_state_dict(ckpt_path: Path, device: torch.device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _find_pad_index(adata: sc.AnnData) -> int:
    try:
        return list(adata.var_names).index("pad_0")
    except ValueError:
        return int(adata.shape[1])


def _pad_adata_to_block_size(adata: sc.AnnData, gene_block_size: int):
    pad_idx = _find_pad_index(adata)
    if gene_block_size <= 0 or adata.n_vars % gene_block_size == 0:
        return adata, pad_idx
    n_pad = gene_block_size - (adata.n_vars % gene_block_size)
    if sparse.issparse(adata.X):
        pad_block = sparse.csr_matrix((adata.n_obs, n_pad), dtype=adata.X.dtype)
        new_x = sparse.hstack([adata.X, pad_block], format="csr")
    else:
        pad_block = np.zeros((adata.n_obs, n_pad), dtype=np.asarray(adata.X).dtype)
        new_x = np.concatenate([np.asarray(adata.X), pad_block], axis=1)
    start_idx = 0 if "pad_0" not in set(map(str, adata.var_names)) else adata.n_vars
    pad_names = [f"pad_{start_idx + i}" for i in range(n_pad)]
    new_var = sc.AnnData(X=new_x[:, :0], var=adata.var).var
    import pandas as pd  # local import keeps top-level minimal

    full_var = pd.DataFrame(index=list(map(str, adata.var_names)) + pad_names)
    del new_var
    padded = sc.AnnData(X=new_x, obs=adata.obs.copy(), var=full_var)
    return padded, pad_idx


def _align_test_to_train_vars(adata_test: sc.AnnData, train_adata: sc.AnnData) -> sc.AnnData:
    if adata_test.var_names.equals(train_adata.var_names):
        return adata_test
    train_names = list(map(str, train_adata.var_names))
    test_names = list(map(str, adata_test.var_names))
    if set(train_names).issubset(set(test_names)):
        return adata_test[:, train_names].copy()
    test_pad = _find_pad_index(adata_test)
    train_pad = _find_pad_index(train_adata)
    if test_pad == train_pad and adata_test.n_vars != train_adata.n_vars:
        # Align unpadded gene positions and adjust trailing pad block to match train n_vars.
        test_core = adata_test[:, :test_pad]
        x_core = test_core.X.toarray() if sparse.issparse(test_core.X) else np.asarray(test_core.X)
        diff = train_adata.n_vars - test_pad
        if diff > 0:
            pad_block = np.zeros((adata_test.n_obs, diff), dtype=x_core.dtype)
            new_x = np.concatenate([x_core, pad_block], axis=1)
            return sc.AnnData(X=new_x, obs=adata_test.obs.copy(), var=train_adata.var.copy())
    if adata_test.n_vars == train_adata.n_vars:
        aligned = adata_test.copy()
        aligned.var_names = train_adata.var_names.copy()
        return aligned
    raise ValueError(
        f"Test dataset gene dimension ({adata_test.n_vars}) cannot align to train ({train_adata.n_vars})"
    )


def load_and_scale_adata(data_path: str, subset_size: int, gene_block_size: int = 0):
    print(f"[pipeline] loading data from: {data_path}")
    adata_ori = sc.read_h5ad(data_path)
    if subset_size > 0:
        adata_ori = adata_ori[:subset_size].copy()
        print(f"[pipeline] subset shape: {adata_ori.shape}")
    else:
        adata_ori = adata_ori.copy()
        print(f"[pipeline] full shape: {adata_ori.shape}")

    if gene_block_size > 0:
        adata_ori, pad_idx = _pad_adata_to_block_size(adata_ori, gene_block_size)
    else:
        pad_idx = _find_pad_index(adata_ori)

    scaler = MaxAbsScaler()
    adata = adata_ori.copy()
    adata.X = scaler.fit_transform(adata.X)
    return adata_ori, adata, scaler, pad_idx


def build_model_kwargs(seed: int, num_heads: int = 4):
    return {
        "hidden_dim": 64,
        "enc_dim": 32,
        "expand_dim": 128,
        "enc_num_heads": num_heads,
        "dec_num_heads": num_heads,
        "enc_depth": 2,
        "dec_depth": 2,
        "enc_affine": True,
        "enc_norm_type": "ln",
        "dec_norm_init": True,
        "dec_blk_type": "adaLN",
        "gene_block_size": 3200,
        "latent_block_size": 3200,
        "is_flash_attn": False,
        "out_layer_dropout": 0.0,
        "is_enc_linear": False,
        "is_dec_linear": False,
        "is_out_moe": False,
        "enc_attn_dropout": 0.0,
        "dec_attn_dropout": 0.0,
        "gpu_num": 1,
        "seed": seed,
        "is_ret_val": False,
        "is_ret_model": True,
        "npy_suffix": "_smoke",
        "is_data_scaled": True,
    }


def build_train_kwargs(vae_epoch: int, batch_size: int):
    return {
        "epoch": vae_epoch,
        "batch_size": batch_size,
        "lr": 1e-4,
        "weight_decay": 2.5e-5,
        "iter": 32,
        "is_lr_scheduler": True,
        "kl_scale": 0.5,
    }


def create_save_model(adata, device, cond_cols, seed: int, num_heads: int = 4):
    model_kwargs = build_model_kwargs(seed=seed, num_heads=num_heads)
    save_model = SAVE(
        adata=adata,
        device=device,
        is_initialized=True,
        condition_cols=cond_cols,
        is_reshape_first=True,
        is_enc_cond=True,
        **model_kwargs,
    )
    return save_model, model_kwargs


def create_save_model_from_kwargs(adata, device, cond_cols, model_kwargs):
    return SAVE(
        adata=adata,
        device=device,
        is_initialized=True,
        condition_cols=cond_cols,
        is_reshape_first=True,
        is_enc_cond=True,
        **model_kwargs,
    )


def attach_fm_model(save_model: SAVE, seq_length: int, cond_cols, num_heads: int = 4):
    denoising_model = MLPTimeStep(
        in_dim=save_model.enc_dim,
        conditional=True,
        seq_length=seq_length,
        hidden_dim=save_model.expand_dim * 2,
        embedding_dim=save_model.expand_dim * 2,
        num_conds=len(cond_cols),
        dropout_prob=0.0,
        n_blocks=2,
        label_encoder_dict=save_model.label_encoder_dict,
        mask_id=save_model.mask_id,
        block_type="DiT",
        num_heads=num_heads,
    )
    save_model.fm = FM(
        denoising_model=denoising_model,
        device=save_model.device,
    )
    return save_model


def attach_formal_fm_model(
    save_model: SAVE,
    seq_length: int,
    cond_cols,
    num_heads: int = 4,
    n_blocks: int = 5,
):
    denoising_model = MLPTimeStep(
        in_dim=save_model.enc_dim,
        conditional=True,
        seq_length=seq_length,
        hidden_dim=save_model.expand_dim * 2,
        embedding_dim=save_model.expand_dim * 2,
        num_conds=len(cond_cols),
        dropout_prob=0.0,
        n_blocks=n_blocks,
        label_encoder_dict=save_model.label_encoder_dict,
        mask_id=save_model.mask_id,
        block_type="DiT",
        num_heads=num_heads,
    )
    save_model.fm = FM(
        denoising_model=denoising_model,
        device=save_model.device,
    )
    return save_model


def train_vae(save_model: SAVE, train_kwargs):
    print("[pipeline] training VAE")
    start = time.time()
    save_model.train(
        col_msk_threshold=1.0,
        loss_monitor=None,
        max_sample_mask_prob=0.0,
        is_col_wise_mask=False,
        **train_kwargs,
    )
    return time.time() - start


def train_fm(save_model: SAVE, fm_epoch: int, batch_size: int):
    print("[pipeline] training FM")
    start = time.time()
    save_model.fm_train(
        epoch=fm_epoch,
        batch_size=batch_size,
        lr=1e-4,
        weight_decay=2.5e-5,
        iter=32,
        is_lr_scheduler=True,
        col_msk_threshold=0.0,
        loss_monitor=None,
        max_sample_mask_prob=0.0,
        is_col_wise_mask=False,
        kl_scale=0.5,
    )
    return time.time() - start


def train_vae_formal(
    save_model: SAVE,
    model_kwargs,
    sample_mask_prob: float = 0.4,
    col_msk_threshold: float = 1.0,
):
    print("[pipeline] training formal VAE")
    start = time.time()
    save_model.train(
        col_msk_threshold=col_msk_threshold,
        loss_monitor=None,
        max_sample_mask_prob=sample_mask_prob,
        is_col_wise_mask=False,
        **model_kwargs,
    )
    return time.time() - start


def train_fm_formal(
    save_model: SAVE,
    model_kwargs,
    sample_mask_prob: float = 0.4,
    col_msk_threshold: float = 0.0,
):
    print("[pipeline] training formal FM")
    start = time.time()
    save_model.fm_train(
        col_msk_threshold=col_msk_threshold,
        loss_monitor=None,
        max_sample_mask_prob=sample_mask_prob,
        is_col_wise_mask=False,
        **model_kwargs,
    )
    return time.time() - start


def load_formal_settings(num_heads: int = 4):
    vae_setting = dict(FORMAL_SAVE_B_MODEL)
    train_setting = dict(FORMAL_SAVE_B_TRAIN)
    vae_setting["enc_num_heads"] = num_heads
    vae_setting["dec_num_heads"] = num_heads
    model_kwargs = {
        "gpu_num": 1,
        "seed": 1202,
        "is_ret_val": False,
        "is_ret_model": True,
        "npy_suffix": "_SAVE-B-release",
        "is_data_scaled": True,
        "fm_num_heads": num_heads,
    }
    model_kwargs.update(vae_setting)
    model_kwargs.update(train_setting)
    model_kwargs["kl_scale"] = 0.5
    return model_kwargs, vae_setting, train_setting


def build_formal_ckpt_paths(
    out_dir: Path,
    vae_setting: dict,
    gene_block_size: int,
    num_heads: int,
    task_name: str = "hca_subsampled_20k",
    ckpt_dir: Path | None = None,
):
    ckpt_root = ckpt_dir if ckpt_dir is not None else out_dir / "ckpt"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_name(task_name)
    vae_ckpt_path = ckpt_root / (
        f"{stem}_VAE_{vae_setting['expand_dim']}_enc"
        f"{vae_setting['enc_dim']}_GBS{gene_block_size}_head{num_heads}.pt"
    )
    fm_ckpt_path = ckpt_root / (
        f"{stem}_fm_{vae_setting['expand_dim']}_enc"
        f"{vae_setting['enc_dim']}_GBS{gene_block_size}_head{num_heads}.pt"
    )
    return vae_ckpt_path, fm_ckpt_path


def run_formal_generation(
    save_model: SAVE,
    adata_ori,
    scaler,
    pad_idx: int,
    batch_size: int,
    seq_length: int,
    cond_cols,
    out_dir: Path,
    guidance_weights: float = 2.0,
    sample_steps: int = 20,
    split_targets: list[tuple[sc.AnnData, Any]] | None = None,
):
    if split_targets is None:
        split_targets = [(adata_ori, scaler)]

    split_gen_list = []
    split_target_list = []
    total_z_fm = []
    total_z_vae = []
    total_groups = 0

    start = time.time()
    for split_adata_ori, split_scaler in split_targets:
        cond_info = (
            split_adata_ori.obs[cond_cols]
            .value_counts()
            .rename_axis(cond_cols)
            .reset_index(name="count")
        )
        cond_info = cond_info[cond_info["count"] > 0].reset_index(drop=True)
        total_groups += int(cond_info.shape[0])

        split_gen_chunks = []
        split_target_chunks = []
        print(f"[pipeline] generating for {cond_info.shape[0]} condition groups")
        for i in range(cond_info.shape[0]):
            cond_dict = cond_info.loc[i, cond_cols].to_dict()
            for col, val in cond_dict.items():
                classes = set(save_model.label_encoder_dict[col].classes_.tolist())
                if val not in classes:
                    raise ValueError(
                        f"Category {val!r} in condition column {col!r} was not seen during training."
                    )

            mask = np.logical_and.reduce(
                [split_adata_ori.obs[k] == v for k, v in cond_dict.items()]
            )
            sub_data = split_adata_ori[mask].copy()
            print(
                f"[pipeline] generating group {i + 1}/{cond_info.shape[0]}: "
                f"{cond_dict} count={sub_data.shape[0]}"
            )

            z_from_vae = save_model.get_latent_from_adata(
                adata=sub_data,
                is_encode_with_cond=False,
                latent_pos=0,
            )
            gen_data, gen_z_fm = save_model.cond_generate_with_guidance_fm(
                generate_count=int(cond_info.loc[i, "count"]),
                target_cond=cond_dict,
                batch_size=min(batch_size, max(1, sub_data.n_obs)),
                guidance_weights=guidance_weights,
                seq_length=seq_length,
                sample_steps=sample_steps,
                verbose=True,
            )

            gen_data.obs_names = [
                f"{len(split_gen_list)}_{i}_{k}_gen" for k in range(gen_data.shape[0])
            ]
            split_gen_chunks.append(gen_data)
            split_target_chunks.append(sub_data)
            total_z_fm.append(gen_z_fm)
            total_z_vae.append(z_from_vae)

        if split_gen_chunks:
            split_gen = sc.concat(split_gen_chunks)
            split_gen.X = split_scaler.inverse_transform(split_gen.X)
            split_target = sc.concat(split_target_chunks)
            split_gen_list.append(split_gen)
            split_target_list.append(split_target)

    gen_adata = sc.concat(split_gen_list) if len(split_gen_list) > 1 else split_gen_list[0].copy()
    target_adata = (
        sc.concat(split_target_list) if len(split_target_list) > 1 else split_target_list[0].copy()
    )
    total_z_fm_arr = np.concatenate(total_z_fm, axis=0)
    total_z_vae_arr = np.concatenate(total_z_vae, axis=0)

    gen_adata.obs["type"] = "gen"
    gen_adata.obs.index = [f"{idx}_gen" for idx in range(gen_adata.shape[0])]
    target_adata.obs["type"] = "ori"

    gen_trim = gen_adata[:, :pad_idx].copy()
    target_trim = target_adata[:, :pad_idx].copy()

    gen_path = out_dir / "generation.h5ad"
    target_path = out_dir / "target.h5ad"
    fm_latent_path = out_dir / "gen_z_fm.npy"
    vae_latent_path = out_dir / "target_z_vae.npy"

    gen_trim.write_h5ad(gen_path)
    target_trim.write_h5ad(target_path)
    np.save(fm_latent_path, total_z_fm_arr)
    np.save(vae_latent_path, total_z_vae_arr)

    return {
        "generation_seconds": round(time.time() - start, 3),
        "condition_group_count": total_groups,
        "gen_shape": list(gen_trim.shape),
        "target_shape": list(target_trim.shape),
        "gen_z_shape": list(total_z_fm_arr.shape),
        "vae_z_shape": list(total_z_vae_arr.shape),
        "generation_path": str(gen_path),
        "target_path": str(target_path),
        "gen_z_path": str(fm_latent_path),
        "vae_z_path": str(vae_latent_path),
    }


def choose_target_condition(adata_ori, cond_cols, generate_count: int):
    cond_info = (
        adata_ori.obs[cond_cols]
        .value_counts()
        .rename_axis(cond_cols)
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    target_cond = cond_info.loc[0, cond_cols].to_dict()
    target_count = min(generate_count, int(cond_info.loc[0, "count"]))
    target_data = adata_ori[
        np.logical_and.reduce([adata_ori.obs[k] == v for k, v in target_cond.items()])
    ].copy()
    return target_cond, target_count, target_data


def run_hca_smoke(
    data_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    subset_size: int = 512,
    device_id: int = 0,
    seed: int = 1202,
    vae_epoch: int = 1,
    fm_epoch: int = 1,
    batch_size: int = 64,
    generate_count: int = 8,
    guidance_weights: float = 2.0,
    sample_steps: int = 5,
    cond_cols=None,
    num_heads: int = 4,
):
    if cond_cols is None:
        cond_cols = DEFAULT_COND_COLS

    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device_id)
    adata_ori, adata, scaler, pad_idx = load_and_scale_adata(data_path, subset_size)
    save_model, model_kwargs = create_save_model(
        adata=adata,
        device=device,
        cond_cols=cond_cols,
        seed=seed,
        num_heads=num_heads,
    )
    train_kwargs = build_train_kwargs(vae_epoch=vae_epoch, batch_size=batch_size)

    vae_seconds = train_vae(save_model, train_kwargs)

    seq_length = adata_ori.shape[1] // model_kwargs["gene_block_size"]
    print("[pipeline] initializing FM")
    attach_fm_model(
        save_model,
        seq_length=seq_length,
        cond_cols=cond_cols,
        num_heads=num_heads,
    )
    fm_seconds = train_fm(save_model, fm_epoch=fm_epoch, batch_size=batch_size)

    target_cond, target_count, target_data = choose_target_condition(
        adata_ori=adata_ori,
        cond_cols=cond_cols,
        generate_count=generate_count,
    )
    print(f"[pipeline] target_cond: {target_cond}")
    print(f"[pipeline] target cells in subset: {target_data.shape[0]}")

    z_from_vae = save_model.get_latent_from_adata(
        adata=target_data,
        is_encode_with_cond=False,
        latent_pos=0,
    )

    print("[pipeline] generating with FM")
    gen_start = time.time()
    gen_adata, gen_z_fm = save_model.cond_generate_with_guidance_fm(
        generate_count=target_count,
        target_cond=target_cond,
        batch_size=min(batch_size, target_count),
        guidance_weights=guidance_weights,
        seq_length=seq_length,
        sample_steps=sample_steps,
        verbose=True,
    )
    gen_seconds = time.time() - gen_start

    gen_adata.X = scaler.inverse_transform(gen_adata.X)
    gen_adata = gen_adata[:, :pad_idx].copy()
    target_trim = target_data[:, :pad_idx].copy()

    gen_path = out_dir / "generated_smoke.h5ad"
    target_path = out_dir / "target_smoke.h5ad"
    fm_latent_path = out_dir / "gen_z_fm.npy"
    vae_latent_path = out_dir / "target_z_vae.npy"
    report_path = out_dir / "report.json"

    gen_adata.write_h5ad(gen_path)
    target_trim.write_h5ad(target_path)
    np.save(fm_latent_path, gen_z_fm)
    np.save(vae_latent_path, z_from_vae)

    report = {
        "data_path": data_path,
        "subset_shape": list(adata_ori.shape),
        "trimmed_gene_dim": int(pad_idx),
        "device": str(device),
        "target_cond": target_cond,
        "target_count": int(target_count),
        "gen_shape": list(gen_adata.shape),
        "gen_z_shape": list(gen_z_fm.shape),
        "vae_z_shape": list(z_from_vae.shape),
        "vae_seconds": round(vae_seconds, 3),
        "fm_seconds": round(fm_seconds, 3),
        "gen_seconds": round(gen_seconds, 3),
        "model_kwargs": model_kwargs,
        "train_kwargs": train_kwargs,
    }
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("[pipeline] done")
    print(json.dumps(report, indent=2))
    print(f"[pipeline] saved generated data to: {gen_path}")
    print(f"[pipeline] saved target data to: {target_path}")
    print(f"[pipeline] saved report to: {report_path}")
    return report


def run_hca_formal(
    data_path: str,
    output_dir: str = DEFAULT_FORMAL_OUTPUT_DIR,
    device_id: int = 0,
    cond_cols=None,
    num_heads: int = 4,
):
    if cond_cols is None:
        cond_cols = DEFAULT_COND_COLS

    set_seed(1202)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(device_id)
    adata_ori, adata, scaler, pad_idx = load_and_scale_adata(
        data_path=data_path,
        subset_size=0,
    )
    model_kwargs, vae_setting, train_setting = load_formal_settings(num_heads=num_heads)
    save_model = create_save_model_from_kwargs(
        adata=adata,
        device=device,
        cond_cols=cond_cols,
        model_kwargs=model_kwargs,
    )

    vae_ckpt_path, fm_ckpt_path = build_formal_ckpt_paths(
        out_dir=out_dir,
        vae_setting=vae_setting,
        gene_block_size=model_kwargs["gene_block_size"],
        num_heads=num_heads,
    )

    if vae_ckpt_path.exists():
        print(f"[pipeline] loading formal VAE checkpoint: {vae_ckpt_path}")
        save_model.load_ckpt(str(vae_ckpt_path))
        vae_seconds = 0.0
        vae_loaded = True
    else:
        vae_seconds = train_vae_formal(save_model, model_kwargs=model_kwargs)
        save_model.save_ckpt(str(vae_ckpt_path))
        vae_loaded = False

    seq_length = adata_ori.shape[1] // model_kwargs["gene_block_size"]
    print("[pipeline] initializing formal FM")
    attach_formal_fm_model(
        save_model,
        seq_length=seq_length,
        cond_cols=cond_cols,
        num_heads=num_heads,
    )

    if fm_ckpt_path.exists():
        print(f"[pipeline] loading formal FM checkpoint: {fm_ckpt_path}")
        save_model.fm.denoising_model.load_state_dict(
            _load_torch_state_dict(fm_ckpt_path, device)
        )
        fm_seconds = 0.0
        fm_loaded = True
    else:
        fm_seconds = train_fm_formal(save_model, model_kwargs=model_kwargs)
        torch.save(save_model.fm.denoising_model.state_dict(), fm_ckpt_path)
        fm_loaded = False

    gen_report = run_formal_generation(
        save_model=save_model,
        adata_ori=adata_ori,
        scaler=scaler,
        pad_idx=pad_idx,
        batch_size=model_kwargs["batch_size"],
        seq_length=seq_length,
        cond_cols=cond_cols,
        out_dir=out_dir,
    )

    report = {
        "data_path": data_path,
        "input_shape": list(adata_ori.shape),
        "trimmed_gene_dim": int(pad_idx),
        "device": str(device),
        "condition_cols": cond_cols,
        "vae_loaded_from_ckpt": vae_loaded,
        "fm_loaded_from_ckpt": fm_loaded,
        "vae_seconds": round(vae_seconds, 3),
        "fm_seconds": round(fm_seconds, 3),
        "vae_ckpt_path": str(vae_ckpt_path),
        "fm_ckpt_path": str(fm_ckpt_path),
        "sample_mask_prob": 0.4,
        "guidance_weights": 2.0,
        "sample_steps": 20,
        "num_heads": num_heads,
        "model_kwargs": model_kwargs,
        "vae_setting": vae_setting,
        "train_setting": train_setting,
    }
    report.update(gen_report)

    report_path = out_dir / "report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("[pipeline] formal run done")
    print(json.dumps(report, indent=2))
    print(f"[pipeline] saved report to: {report_path}")
    return report


def run_formal_conditional_generation(
    data_path: str,
    output_dir: str = DEFAULT_FORMAL_OUTPUT_DIR,
    device_id: int = 0,
    cond_cols: list[str] | None = None,
    task_name: str = "conditional_generation",
    gene_block_size: int = 3200,
    latent_block_size: int | None = None,
    vae_epoch: int = 200,
    fm_epoch: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-4,
    weight_decay: float = 2.5e-5,
    kl_scale: float = 0.5,
    expand_dim: int = 512,
    enc_dim: int = 64,
    hidden_dim: int = 64,
    num_heads: int = 4,
    n_fm_blocks: int = 5,
    guidance_weights: float = 2.0,
    sample_steps: int = 20,
    sample_mask_prob: float | None = None,
    vae_col_msk_threshold: float = 1.0,
    fm_col_msk_threshold: float = 0.0,
    test_data_path: str | None = None,
    eval_split: str = "auto",
    seed: int = 1202,
    subset_size: int = 0,
    ckpt_dir: str | None = None,
    overwrite_ckpt: bool = False,
) -> dict[str, Any]:
    """Run generalized SAVE + FM formal conditional generation for Tables 1-3."""
    if cond_cols is None:
        cond_cols = list(DEFAULT_COND_COLS)
    else:
        cond_cols = list(cond_cols)

    if latent_block_size is None:
        latent_block_size = gene_block_size
    if sample_mask_prob is None:
        effective_sample_mask_prob = 0.0 if task_name in {"cancer_lung", "lung_cancer"} else 0.4
    else:
        effective_sample_mask_prob = float(sample_mask_prob)

    if eval_split not in {"auto", "train", "test", "all"}:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device_id)

    adata_train_ori, adata_train, train_scaler, pad_idx = load_and_scale_adata(
        data_path=data_path,
        subset_size=subset_size,
        gene_block_size=gene_block_size,
    )
    for col in cond_cols:
        if col not in adata_train_ori.obs.columns:
            raise ValueError(f"Training AnnData missing required obs column: '{col}'")

    effective_batch_size = min(batch_size, max(1, adata_train_ori.n_obs))
    model_kwargs, vae_setting, train_setting = load_formal_settings(num_heads=num_heads)
    vae_setting.update(
        {
            "kl_scale": kl_scale,
            "hidden_dim": hidden_dim,
            "enc_dim": enc_dim,
            "expand_dim": expand_dim,
            "gene_block_size": gene_block_size,
            "latent_block_size": latent_block_size,
        }
    )
    train_setting.update(
        {
            "lr": lr,
            "batch_size": effective_batch_size,
            "weight_decay": weight_decay,
        }
    )
    model_kwargs.update(vae_setting)
    model_kwargs.update(train_setting)
    model_kwargs["seed"] = seed

    save_model = create_save_model_from_kwargs(
        adata=adata_train,
        device=device,
        cond_cols=cond_cols,
        model_kwargs=model_kwargs,
    )

    ckpt_root = Path(ckpt_dir) if ckpt_dir is not None else out_dir / "ckpt"
    vae_ckpt_path, fm_ckpt_path = build_formal_ckpt_paths(
        out_dir=out_dir,
        vae_setting=vae_setting,
        gene_block_size=gene_block_size,
        num_heads=num_heads,
        task_name=task_name,
        ckpt_dir=ckpt_root,
    )

    vae_train_kwargs = dict(model_kwargs)
    vae_train_kwargs["epoch"] = vae_epoch
    if vae_ckpt_path.exists() and not overwrite_ckpt:
        print(f"[pipeline] loading formal VAE checkpoint: {vae_ckpt_path}")
        save_model.load_ckpt(str(vae_ckpt_path))
        vae_seconds = 0.0
        vae_loaded = True
    else:
        vae_seconds = train_vae_formal(
            save_model,
            model_kwargs=vae_train_kwargs,
            sample_mask_prob=effective_sample_mask_prob,
            col_msk_threshold=vae_col_msk_threshold,
        )
        save_model.save_ckpt(str(vae_ckpt_path))
        vae_loaded = False

    seq_length = adata_train_ori.shape[1] // gene_block_size
    if seq_length < 1 or adata_train_ori.shape[1] % gene_block_size != 0:
        raise ValueError(
            f"Input gene count {adata_train_ori.shape[1]} must be a positive multiple of gene_block_size={gene_block_size}"
        )

    attach_formal_fm_model(
        save_model,
        seq_length=seq_length,
        cond_cols=cond_cols,
        num_heads=num_heads,
        n_blocks=n_fm_blocks,
    )

    fm_train_kwargs = dict(model_kwargs)
    fm_train_kwargs["epoch"] = fm_epoch
    if fm_ckpt_path.exists() and not overwrite_ckpt:
        print(f"[pipeline] loading formal FM checkpoint: {fm_ckpt_path}")
        save_model.fm.denoising_model.load_state_dict(
            _load_torch_state_dict(fm_ckpt_path, device)
        )
        fm_seconds = 0.0
        fm_loaded = True
    else:
        fm_seconds = train_fm_formal(
            save_model,
            model_kwargs=fm_train_kwargs,
            sample_mask_prob=effective_sample_mask_prob,
            col_msk_threshold=fm_col_msk_threshold,
        )
        torch.save(save_model.fm.denoising_model.state_dict(), fm_ckpt_path)
        fm_loaded = False

    split_targets: list[tuple[sc.AnnData, Any]] = [(adata_train_ori, train_scaler)]
    resolved_split = eval_split
    if test_data_path is not None:
        adata_test_raw = sc.read_h5ad(test_data_path)
        for col in cond_cols:
            if col not in adata_test_raw.obs.columns:
                raise ValueError(f"Test AnnData missing required obs column: '{col}'")
        adata_test_ori = _align_test_to_train_vars(adata_test_raw, adata_train_ori)
        test_scaler = MaxAbsScaler()
        test_scaler.fit(adata_test_ori.X)

        if resolved_split == "auto":
            resolved_split = (
                "all"
                if (task_name in {"cancer_lung", "lung_cancer"} or len(cond_cols) >= 3)
                else "test"
            )

        if resolved_split == "all":
            split_targets = [
                (adata_train_ori, train_scaler),
                (adata_test_ori, test_scaler),
            ]
        elif resolved_split == "test":
            split_targets = [(adata_test_ori, test_scaler)]
        elif resolved_split == "train":
            split_targets = [(adata_train_ori, train_scaler)]
    elif resolved_split == "auto":
        resolved_split = "train"
    elif resolved_split in {"test", "all"}:
        raise ValueError(f"eval_split='{resolved_split}' requires test_data_path to be set")

    gen_report = run_formal_generation(
        save_model=save_model,
        adata_ori=adata_train_ori,
        scaler=train_scaler,
        pad_idx=pad_idx,
        batch_size=effective_batch_size,
        seq_length=seq_length,
        cond_cols=cond_cols,
        out_dir=out_dir,
        guidance_weights=guidance_weights,
        sample_steps=sample_steps,
        split_targets=split_targets,
    )

    report = {
        "task_name": task_name,
        "data_path": data_path,
        "test_data_path": test_data_path,
        "eval_split": resolved_split,
        "input_shape": list(adata_train_ori.shape),
        "trimmed_gene_dim": int(pad_idx),
        "device": str(device),
        "condition_cols": cond_cols,
        "vae_loaded_from_ckpt": vae_loaded,
        "fm_loaded_from_ckpt": fm_loaded,
        "vae_seconds": round(vae_seconds, 3),
        "fm_seconds": round(fm_seconds, 3),
        "vae_ckpt_path": str(vae_ckpt_path),
        "fm_ckpt_path": str(fm_ckpt_path),
        "vae_epoch": vae_epoch,
        "fm_epoch": fm_epoch,
        "sample_mask_prob": effective_sample_mask_prob,
        "guidance_weights": guidance_weights,
        "sample_steps": sample_steps,
        "gene_block_size": gene_block_size,
        "latent_block_size": latent_block_size,
        "num_heads": num_heads,
        "model_kwargs": model_kwargs,
        "vae_setting": vae_setting,
        "train_setting": train_setting,
    }
    report.update(gen_report)

    report_path = out_dir / "report.json"
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"[pipeline] generalized formal run ({task_name}) done -> {report_path}")
    return report


__all__ = [
    "DEFAULT_COND_COLS",
    "DEFAULT_FORMAL_OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "run_formal_conditional_generation",
    "run_hca_formal",
    "run_hca_smoke",
]
