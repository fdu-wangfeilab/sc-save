"""Numerical parity, zero-grad inference, checkpoint compat, and batching tests for sc_save."""

from __future__ import annotations

import numpy as np
import ot
import pandas as pd
import pytest
import torch
from anndata import AnnData
from torch.distributions import Normal, kl_divergence


def _make_padded_adata(
    n_obs: int = 36,
    n_real_genes: int = 32,
    pad_tail: int = 8,
    seed: int = 42,
) -> AnnData:
    rng = np.random.default_rng(seed)
    x_real = rng.uniform(0.05, 2.5, size=(n_obs, n_real_genes)).astype(np.float32)
    var_names = [f"gene_{i}" for i in range(n_real_genes)]
    if pad_tail > 0:
        x = np.concatenate(
            [x_real, np.zeros((n_obs, pad_tail), dtype=np.float32)], axis=1
        )
        var_names += [f"pad_{i}" for i in range(pad_tail)]
    else:
        x = x_real

    batches = ["b1", "b2", "b3"]
    cell_types = ["T", "B"]
    obs = pd.DataFrame(
        {
            "batch": [batches[i % len(batches)] for i in range(n_obs)],
            "cell_type": [cell_types[(i // len(batches)) % len(cell_types)] for i in range(n_obs)],
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    return AnnData(X=x, obs=obs, var=pd.DataFrame(index=var_names))


def test_analytic_kl_div_matches_torch_distributions():
    from sc_save.training.runtime import analytic_kl_div

    torch.manual_seed(7)
    for shape in ((16, 24), (64, 12)):
        mu = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
        log_var = torch.randn(*shape, dtype=torch.float32) * 1.5
        var = torch.clamp(torch.exp(log_var), min=1e-20).detach().requires_grad_(True)

        mu_ref = mu.detach().clone().requires_grad_(True)
        var_ref = var.detach().clone().requires_grad_(True)

        kl_fast = analytic_kl_div(mu, var)
        kl_ref = (
            kl_divergence(
                Normal(mu_ref, var_ref.sqrt()),
                Normal(torch.zeros_like(mu_ref), torch.ones_like(var_ref)),
            )
            .sum(dim=1)
            .mean()
        )
        assert torch.allclose(kl_fast, kl_ref, atol=1e-5, rtol=1e-5)

        kl_fast.backward()
        kl_ref.backward()
        assert torch.allclose(mu.grad, mu_ref.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(var.grad, var_ref.grad, atol=1e-5, rtol=1e-5)


def test_cond_ot_path_and_native_euler_parity_against_flow_matching(monkeypatch):
    flow_matching = pytest.importorskip("flow_matching")
    del flow_matching
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, VPScheduler
    from flow_matching.solver import ODESolver
    from flow_matching.utils import ModelWrapper
    from sc_save.data.preprocessing import ConditionVocabulary
    from sc_save.nn.flow_matching import CondOTPath, FM, MLPTimeStep, VEPath, VPPath

    torch.manual_seed(11)
    x_0 = torch.randn(10, 4, 16)
    x_1 = torch.randn(10, 4, 16)
    t = torch.rand(10)

    for native_path, ref_scheduler in (
        (CondOTPath(), CondOTScheduler()),
        (VPPath(), VPScheduler()),
        (VEPath(), CosineScheduler()),
    ):
        ref_path = AffineProbPath(ref_scheduler)
        native_sample = native_path.sample(t=t, x_0=x_0, x_1=x_1)
        ref_sample = ref_path.sample(t=t, x_0=x_0, x_1=x_1)
        assert torch.allclose(native_sample.x_t, ref_sample.x_t, atol=1e-5)
        assert torch.allclose(native_sample.dx_t, ref_sample.dx_t, atol=1e-5)
        assert torch.allclose(native_sample.t, ref_sample.t, atol=1e-6)

    obs = pd.DataFrame({"batch": ["b1", "b2"] * 4, "cell_type": ["T", "B"] * 4})
    vocab = ConditionVocabulary(["batch", "cell_type"], mode="offset").fit(obs)
    denoiser = MLPTimeStep(
        in_dim=16,
        hidden_dim=32,
        dropout_prob=0.0,
        n_blocks=2,
        seq_length=3,
        embedding_dim=32,
        label_encoder_dict=vocab.label_encoder_dict,
        total_classes=vocab.total_classes,
        mask_id=vocab.mask_id,
        num_conds=2,
        num_heads=2,
    )
    denoiser.eval()
    fm = FM(denoising_model=denoiser, device=torch.device("cpu"), path_type="ot")

    x_init = torch.randn(6, 3, 16)
    cond_ids = torch.tensor(vocab.encode_obs(obs.iloc[:6]), dtype=torch.long)
    cond_emb = denoiser.cond_embedder(cond_ids)

    class _RefWrappedModel(ModelWrapper):
        def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **extras):
            m_uncond = self.model(x, t, y, inference=True, unconditional=True)
            m_cond = self.model(x, t, y, inference=True, unconditional=False)
            return (1.0 - 2.0) * m_uncond + 2.0 * m_cond

    n_steps = 8
    ref_solver = ODESolver(velocity_model=_RefWrappedModel(denoiser))
    with torch.no_grad():
        ref_out = ref_solver.sample(
            time_grid=torch.linspace(0.0, 1.0, n_steps),
            x_init=x_init.clone(),
            method="euler",
            step_size=1.0 / n_steps,
            return_intermediates=False,
            y=cond_emb,
        )

    orig_randn = torch.randn
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *args, **kwargs: x_init.clone()
        if tuple(args[0]) == tuple(x_init.shape) and isinstance(args[0], tuple)
        else x_init.clone()
        if args == tuple(x_init.shape)
        else orig_randn(*args, **kwargs),
    )
    native_out = fm.sample(
        batch_size=6,
        seq_length=3,
        n_sample_steps=n_steps,
        y=cond_emb,
        guidance_weights=2.0,
        solver_type="euler",
    )
    assert torch.allclose(native_out, ref_out, atol=1e-4, rtol=1e-4)

    # Also verify pure-PyTorch VP & VE paths work without flow_matching
    for path_type in ("vp", "ve"):
        fm_alt = FM(denoising_model=denoiser, device=torch.device("cpu"), path_type=path_type)
        loss = fm_alt._step(x_init, cond_emb)
        assert torch.isfinite(loss)
        samples = fm_alt.sample(
            batch_size=4,
            seq_length=3,
            n_sample_steps=4,
            y=cond_emb[:4],
            guidance_weights=1.5,
            solver_type="euler",
        )
        assert samples.shape == (4, 3, 16)
        assert torch.isfinite(samples).all()


def test_fused_cfg_single_forward_matches_separate_passes(monkeypatch):
    from sc_save.data.preprocessing import ConditionVocabulary
    from sc_save.nn.flow_matching import FM, MLPTimeStep

    torch.manual_seed(19)
    obs = pd.DataFrame({"batch": ["b1", "b2"] * 4, "cell_type": ["T", "B"] * 4})
    vocab = ConditionVocabulary(["batch", "cell_type"], mode="offset").fit(obs)
    denoiser = MLPTimeStep(
        in_dim=12,
        hidden_dim=32,
        dropout_prob=0.0,
        n_blocks=2,
        seq_length=2,
        embedding_dim=32,
        label_encoder_dict=vocab.label_encoder_dict,
        total_classes=vocab.total_classes,
        mask_id=vocab.mask_id,
        num_conds=2,
        num_heads=2,
    )
    denoiser.eval()
    fm = FM(denoising_model=denoiser, device=torch.device("cpu"), path_type="ot")

    batch_size = 5
    seq_len = 2
    steps = 6
    guidance = 2.5
    x_init = torch.randn(batch_size, seq_len, 12)
    cond_ids = torch.tensor(vocab.encode_obs(obs.iloc[:batch_size]), dtype=torch.long)
    cond_emb = denoiser.cond_embedder(cond_ids)

    # Compute explicit separate uncond / cond Euler trajectory
    dt = 1.0 / steps
    x_sep = x_init.clone()
    with torch.no_grad():
        for step in range(steps):
            t_batch = torch.full((batch_size,), step * dt, dtype=torch.float32)
            v_uncond = denoiser(x_sep, t_batch, cond_emb, inference=True, unconditional=True)
            v_cond = denoiser(x_sep, t_batch, cond_emb, inference=True, unconditional=False)
            v = (1.0 - guidance) * v_uncond + guidance * v_cond
            x_sep = x_sep + v * dt

    # Spy on denoiser.forward during fused CFG sampling
    forward_batch_sizes: list[int] = []
    orig_forward = denoiser.forward

    def _spy_forward(x, t, y, *args, **kwargs):
        forward_batch_sizes.append(int(x.shape[0]))
        return orig_forward(x, t, y, *args, **kwargs)

    monkeypatch.setattr(denoiser, "forward", _spy_forward)
    orig_randn = torch.randn
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *args, **kwargs: x_init.clone()
        if (args and args[0] == (batch_size, seq_len, 12)) or args == (batch_size, seq_len, 12)
        else orig_randn(*args, **kwargs),
    )

    x_fused = fm.sample(
        batch_size=batch_size,
        seq_length=seq_len,
        n_sample_steps=steps,
        y=cond_emb,
        guidance_weights=guidance,
        solver_type="euler",
    )

    assert len(forward_batch_sizes) == steps
    assert all(bs == 2 * batch_size for bs in forward_batch_sizes)
    assert torch.allclose(x_fused, x_sep, atol=1e-4, rtol=1e-4)


def test_inference_methods_create_zero_autograd_graphs(monkeypatch):
    from sc_save import SAVE, SAVEConfig

    adata = _make_padded_adata(n_obs=24, n_real_genes=32, pad_tail=8, seed=23)
    cfg = SAVEConfig.smoke(gene_block_size=40, latent_block_size=40)
    model = SAVE(
        adata=adata,
        condition_cols=("batch", "cell_type"),
        config=cfg,
        device="cpu",
        cond_encoding="offset",
    ).fit(vae_epochs=1, fm_epochs=1)

    grad_states: list[tuple[str, bool, bool]] = []

    def _wrap_forward(name: str, mod: torch.nn.Module):
        orig = mod.forward

        def _wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            first = out[0] if isinstance(out, tuple) else out
            grad_states.append(
                (name, torch.is_grad_enabled(), bool(getattr(first, "requires_grad", False)))
            )
            return out

        monkeypatch.setattr(mod, "forward", _wrapped)

    _wrap_forward("encoder", model.vae.encoder)
    _wrap_forward("decoder", model.vae.decoder)
    _wrap_forward("denoiser", model.fm.denoising_model)

    latents = model.encode()
    recon = model.reconstruct()
    gen = model.generate({"batch": "b1", "cell_type": "T"}, n_cells=4, steps=2)
    gen_like = model.generate_like(adata[:6], steps=2)
    trans = model.transfer(adata[:6], target_cond={"batch": "b2"}, input_cond="obs")

    assert latents.shape[0] == adata.n_obs
    assert recon.n_obs == adata.n_obs
    assert gen.n_obs == 4
    assert gen_like.n_obs == 6
    assert trans.n_obs == 6

    assert grad_states, "expected forward passes to be recorded"
    assert all(not grad_on and not req_grad for _, grad_on, req_grad in grad_states)
    for mod in (model.vae, model.fm.denoising_model):
        assert all(p.grad is None for p in mod.parameters())


def test_legacy_v1_checkpoint_loading_into_v2_models(tmp_path):
    from sc_save import SAVE, SAVEConfig
    from sc_save.data.preprocessing import ConditionVocabulary
    from sc_save.nn.blocks import TimestepEmbedder
    from sc_save.nn.flow_matching import MLPTimeStep
    from sc_save.nn.vae import VAE, load_state_dict_compat

    # 1. TimestepEmbedder keeps freqs non-persistent while moving across devices
    te = TimestepEmbedder(hidden_size=32, frequency_embedding_size=16)
    assert "freqs" not in te.state_dict()
    assert hasattr(te, "freqs")
    assert te(torch.tensor([0.0, 0.5, 1.0])).shape == (3, 32)

    # 2. Legacy v1 checkpoint key remapping (tokenzier -> tokenizer, pos_embding -> pos_embedding)
    obs = pd.DataFrame({"batch": ["b1", "b2"] * 4, "cell_type": ["T", "B"] * 4})
    vocab = ConditionVocabulary(["batch", "cell_type"], mode="legacy").fit(obs)

    torch.manual_seed(101)
    vae_src = VAE(
        input_dim=40,
        hidden_dim=16,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        enc_dim=16,
        expand_dim=32,
        enc_norm_type="ln",
        dec_norm_init=True,
        is_initialize=True,
        label_encoder_dict=vocab.label_encoder_dict,
        total_classes=vocab.total_classes,
        mask_id=vocab.mask_id,
        is_reshape_first=True,
        gene_block_size=40,
        latent_block_size=40,
        is_enc_cond=True,
    )
    legacy_vae_sd = {
        k.replace("tokenizer", "tokenzier").replace("pos_embedding", "pos_embding"): v.clone()
        for k, v in vae_src.state_dict().items()
    }
    assert any("tokenzier" in k for k in legacy_vae_sd)
    assert any("pos_embding" in k for k in legacy_vae_sd)

    torch.manual_seed(202)
    vae_dst = VAE(
        input_dim=40,
        hidden_dim=16,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        enc_dim=16,
        expand_dim=32,
        enc_norm_type="ln",
        dec_norm_init=True,
        is_initialize=True,
        label_encoder_dict=vocab.label_encoder_dict,
        total_classes=vocab.total_classes,
        mask_id=vocab.mask_id,
        is_reshape_first=True,
        gene_block_size=40,
        latent_block_size=40,
        is_enc_cond=True,
    )
    load_state_dict_compat(vae_dst, legacy_vae_sd)
    for k, v in vae_src.state_dict().items():
        assert torch.equal(vae_dst.state_dict()[k], v)

    # 3. Legacy MLPTimeStep state_dict with pos_embding
    fm_src = MLPTimeStep(
        in_dim=16,
        hidden_dim=64,
        dropout_prob=0.0,
        n_blocks=1,
        seq_length=1,
        embedding_dim=64,
        label_encoder_dict=vocab.label_encoder_dict,
        total_classes=vocab.total_classes,
        mask_id=vocab.mask_id,
        num_conds=2,
        num_heads=2,
    )
    legacy_fm_sd = {
        k.replace("pos_embedding", "pos_embding"): v.clone()
        for k, v in fm_src.state_dict().items()
    }
    assert "pos_embding" in legacy_fm_sd

    fm_dst = MLPTimeStep(
        in_dim=16,
        hidden_dim=64,
        dropout_prob=0.0,
        n_blocks=1,
        seq_length=1,
        embedding_dim=64,
        label_encoder_dict=vocab.label_encoder_dict,
        total_classes=vocab.total_classes,
        mask_id=vocab.mask_id,
        num_conds=2,
        num_heads=2,
    )
    load_state_dict_compat(fm_dst, legacy_fm_sd)
    for k, v in fm_src.state_dict().items():
        assert torch.equal(fm_dst.state_dict()[k], v)

    # 4. End-to-end SAVE load_vae_ckpt / load_fm_ckpt with legacy .pt state_dicts
    vae_pt = tmp_path / "v1_vae.pt"
    fm_pt = tmp_path / "v1_fm.pt"
    torch.save(legacy_vae_sd, vae_pt)
    torch.save(legacy_fm_sd, fm_pt)

    adata = _make_padded_adata(n_obs=16, n_real_genes=32, pad_tail=8, seed=9)
    # Match obs batch count (b1, b2) & architecture of vae_src/fm_src above
    adata.obs["batch"] = ["b1", "b2"] * 8
    cfg = SAVEConfig.smoke(
        gene_block_size=40,
        latent_block_size=40,
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        num_heads=2,
        enc_depth=1,
        dec_depth=1,
        fm_n_blocks=1,
    )
    cfg.fm.hidden_dim = 64
    cfg.fm.embedding_dim = 64
    save_model = SAVE(
        adata=adata,
        condition_cols=("batch", "cell_type"),
        config=cfg,
        device="cpu",
        cond_encoding="legacy",
    )
    save_model.load_vae_ckpt(vae_pt)
    save_model.init_fm(seq_length=1)
    save_model.load_fm_ckpt(fm_pt)
    out = save_model.generate({"batch": "b1", "cell_type": "T"}, n_cells=2, steps=2)
    assert out.shape == (2, 32)

    # 5. Legacy shim SAVE.load_ckpt loads legacy v1 .pt directly
    from sc_save_public.models.save_model import SAVE as LegacySAVE

    legacy_model = LegacySAVE(
        adata=adata,
        device=torch.device("cpu"),
        condition_cols=["batch", "cell_type"],
        hidden_dim=16,
        enc_dim=16,
        expand_dim=32,
        enc_num_heads=2,
        dec_num_heads=2,
        enc_depth=1,
        dec_depth=1,
        gene_block_size=40,
        latent_block_size=40,
    )
    legacy_model.load_ckpt(str(vae_pt))
    assert legacy_model.is_trained
    for k, v in vae_src.state_dict().items():
        assert torch.equal(legacy_model.model.state_dict()[k], v)


def test_batched_generate_like_shape_obs_pad_trim_and_single_pass(monkeypatch):
    from sc_save import SAVE, SAVEConfig

    adata = _make_padded_adata(n_obs=36, n_real_genes=32, pad_tail=8, seed=31)
    cfg = SAVEConfig.smoke(gene_block_size=40, latent_block_size=40)
    model = SAVE(
        adata=adata,
        condition_cols=("batch", "cell_type"),
        config=cfg,
        device="cpu",
        cond_encoding="offset",
    ).fit(vae_epochs=1, fm_epochs=1)

    target_obs = pd.DataFrame(
        {
            "batch": ["b1"] * 4 + ["b2"] * 2 + ["b3"] * 5,
            "cell_type": ["T", "T", "B", "B", "T", "B", "T", "T", "B", "B", "T"],
        },
        index=[f"t_{i}" for i in range(11)],
    )
    target_adata = AnnData(
        X=np.zeros((11, 32), dtype=np.float32),
        obs=target_obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(32)]),
    )

    sample_calls = []
    orig_sample = model.fm.sample

    def _spy_sample(*args, **kwargs):
        sample_calls.append(kwargs.get("batch_size", args[0] if args else None))
        return orig_sample(*args, **kwargs)

    monkeypatch.setattr(model.fm, "sample", _spy_sample)

    gen_trimmed, latents = model.generate_like(
        target_adata,
        guidance_weight=2.0,
        steps=3,
        batch_size=32,
        return_latents=True,
        trim_pad=True,
    )
    # Single batched GPU pass across all 6 condition groups
    assert len(sample_calls) == 1
    assert sample_calls[0] == 11
    assert gen_trimmed.shape == (11, 32)
    assert latents.shape == (11, 1, cfg.vae.enc_dim)
    assert list(gen_trimmed.var_names) == [f"gene_{i}" for i in range(32)]

    # Exact condition count distribution parity
    expected_counts = target_obs.groupby(["batch", "cell_type"], observed=True).size().to_dict()
    observed_counts = gen_trimmed.obs.groupby(["batch", "cell_type"], observed=True).size().to_dict()
    assert observed_counts == expected_counts

    gen_untrimmed = model.generate_like(target_adata, steps=2, trim_pad=False)
    assert gen_untrimmed.shape == (11, 40)
    assert list(gen_untrimmed.var_names[-8:]) == [f"pad_{i}" for i in range(8)]


def test_batched_bec_transfer_input_cond_obs_matches_per_group(monkeypatch):
    from sc_save import SAVE, SAVEConfig

    adata = _make_padded_adata(n_obs=36, n_real_genes=32, pad_tail=8, seed=41)
    cfg = SAVEConfig.smoke(gene_block_size=40, latent_block_size=40)
    model = SAVE(
        adata=adata,
        condition_cols=("batch", "cell_type"),
        config=cfg,
        device="cpu",
        cond_encoding="offset",
    ).fit(vae_epochs=1, train_fm=False)

    # Disable posterior reparameterization noise so batched vs per-group comparison is deterministic
    monkeypatch.setattr(model.vae.encoder, "reparameterize", lambda mu, var: mu)

    other_data = model.adata[model.adata.obs["batch"].astype(str) != "b1"].copy()
    batched = model.transfer(
        other_data,
        target_cond={"batch": "b1"},
        input_cond="obs",
        batch_size=8,
        min_clip=1e-4,
        inverse_scale=False,
        trim_pad=False,
    )
    assert list(batched.obs_names) == list(other_data.obs_names)
    assert set(batched.obs["batch"].astype(str)) == {"b1"}

    # Compare against per-(batch, cell_type) group transfer
    per_group_rows: dict[str, np.ndarray] = {}
    for (b_val, ct_val), idx in other_data.obs.groupby(["batch", "cell_type"], observed=True).indices.items():
        sub = other_data[idx].copy()
        res = model.transfer(
            sub,
            target_cond={"batch": "b1", "cell_type": ct_val},
            input_cond={"batch": b_val, "cell_type": ct_val},
            batch_size=8,
            min_clip=1e-4,
            inverse_scale=False,
            trim_pad=False,
        )
        for row_i, cell_name in enumerate(sub.obs_names):
            per_group_rows[cell_name] = np.asarray(res.X[row_i])

    per_group_x = np.stack([per_group_rows[name] for name in other_data.obs_names], axis=0)
    np.testing.assert_allclose(np.asarray(batched.X), per_group_x, atol=1e-5, rtol=1e-5)


def test_shared_cdist_wasserstein_and_mix_rbf_mmd2_parity(monkeypatch):
    from sc_save.eval.metrics import (
        compute_distribution_distances_batched,
        mix_rbf_mmd2,
        wasserstein,
    )
    from sc_save_public import distribution_metrics as legacy_dm

    torch.manual_seed(55)
    x = torch.randn(14, 6)
    y = torch.randn(14, 6) + 0.4

    cdist_calls = {"n": 0}
    orig_cdist = torch.cdist

    def _spy_cdist(*args, **kwargs):
        cdist_calls["n"] += 1
        return orig_cdist(*args, **kwargs)

    monkeypatch.setattr(torch, "cdist", _spy_cdist)
    metrics = compute_distribution_distances_batched(x, y, batch_size=14, is_mmd=True)
    # 1-Wasserstein and 2-Wasserstein share one torch.cdist call
    assert cdist_calls["n"] == 1

    w1_ref = wasserstein(x, y, power=1)
    w2_ref = wasserstein(x, y, power=2)
    assert metrics["1-Wasserstein"] == pytest.approx(w1_ref, abs=1e-5)
    assert metrics["2-Wasserstein"] == pytest.approx(w2_ref, abs=1e-5)

    # Also verify exact agreement with POT emd2 directly
    a, b = ot.unif(x.shape[0]), ot.unif(y.shape[0])
    cost = orig_cdist(x, y).numpy()
    assert w1_ref == pytest.approx(float(ot.emd2(a, b, cost, numItermax=int(1e7))), abs=1e-5)

    sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
    assert float(mix_rbf_mmd2(x, y, sigmas, biased=True).item()) == pytest.approx(
        float(legacy_dm.mix_rbf_mmd2(x, y, sigmas, biased=True).item()),
        abs=1e-6,
    )
    assert float(mix_rbf_mmd2(x, y, sigmas, biased=False).item()) == pytest.approx(
        float(legacy_dm.mix_rbf_mmd2(x, y, sigmas, biased=False).item()),
        abs=1e-6,
    )


def test_resident_tensors_sparse_and_dense_iteration():
    from scipy import sparse
    from sc_save.training.runtime import iter_resident_batches, prepare_resident_tensors

    rng = np.random.default_rng(101)
    x_dense = rng.uniform(0.0, 1.0, size=(15, 8)).astype(np.float32)
    x_sparse = sparse.csr_matrix(x_dense)
    y = rng.integers(0, 4, size=(15, 2), dtype=np.int64)

    res_dense = prepare_resident_tensors(x_dense, y, device=torch.device("cpu"))
    res_sparse = prepare_resident_tensors(x_sparse, y, device=torch.device("cpu"))
    assert torch.allclose(res_dense.x, res_sparse.x)
    assert torch.equal(res_dense.y, res_sparse.y)

    batches = list(iter_resident_batches(res_sparse, batch_size=6, shuffle=False))
    assert [b[0].shape[0] for b in batches] == [6, 6, 3]
    recon = torch.cat([b[0] for b in batches], dim=0)
    assert torch.allclose(recon, res_dense.x)


def test_scaler_inplace_dense_and_sparse_parity():
    from scipy import sparse
    from sc_save.data.preprocessing import MaxAbsExpressionScaler

    rng = np.random.default_rng(202)
    x_raw = rng.uniform(0.0, 5.0, size=(12, 6)).astype(np.float32)
    x_raw[:, 2] = 0.0
    scaler = MaxAbsExpressionScaler().fit(x_raw)

    ref_scaled = scaler.transform(x_raw, inplace=False)
    x_in = x_raw.copy()
    got_scaled = scaler.transform(x_in, inplace=True)
    assert got_scaled is x_in
    np.testing.assert_allclose(got_scaled, ref_scaled, atol=1e-7)

    ref_inv = scaler.inverse_transform(ref_scaled, inplace=False)
    got_inv = scaler.inverse_transform(got_scaled, inplace=True)
    assert got_inv is x_in
    np.testing.assert_allclose(got_inv, ref_inv, atol=1e-6)
    np.testing.assert_allclose(got_inv, x_raw, atol=1e-6)

    sp_raw = sparse.csr_matrix(x_raw)
    ref_sp_scaled = scaler.transform(sp_raw, inplace=False)
    sp_in = sp_raw.copy()
    got_sp_scaled = scaler.transform(sp_in, inplace=True)
    assert got_sp_scaled is sp_in
    np.testing.assert_allclose(got_sp_scaled.toarray(), ref_sp_scaled.toarray(), atol=1e-7)

    ref_sp_inv = scaler.inverse_transform(ref_sp_scaled, inplace=False)
    got_sp_inv = scaler.inverse_transform(got_sp_scaled, inplace=True)
    assert got_sp_inv is sp_in
    np.testing.assert_allclose(got_sp_inv.toarray(), ref_sp_inv.toarray(), atol=1e-6)


def test_copy_x_to_gpu_chunked_and_materialise_parity():
    from scipy import sparse
    from sc_save.training.runtime import _copy_x_to_gpu_chunked, _materialise_x_tensor

    rng = np.random.default_rng(303)
    x_dense64 = rng.uniform(0.0, 2.0, size=(11, 7))
    x_sparse = sparse.csr_matrix(x_dense64.astype(np.float32))
    x_tensor = torch.from_numpy(x_dense64.astype(np.float32))

    ref32 = _materialise_x_tensor(x_dense64, dtype=torch.float32, chunk_size=4)
    for src in (x_dense64, x_sparse, x_tensor):
        got = _copy_x_to_gpu_chunked(src, device=torch.device("cpu"), dtype=torch.float32, chunk_size=4)
        assert torch.allclose(got, ref32, atol=1e-6)

    ref16 = _materialise_x_tensor(x_dense64, dtype=torch.float16, chunk_size=4)
    got16 = _copy_x_to_gpu_chunked(x_sparse, device=torch.device("cpu"), dtype=torch.float16, chunk_size=4)
    assert got16.dtype == torch.float16
    assert torch.allclose(got16, ref16, atol=1e-3)


def test_train_fm_streaming_posterior_precomputation_parity():
    from scipy import sparse
    from sc_save import FMConfig, SAVE, SAVEConfig, TrainConfig, VAEConfig
    from sc_save.training.fm_trainer import _precompute_posteriors, _precompute_posteriors_streaming
    from sc_save.training.runtime import get_autocast_context, prepare_resident_tensors

    adata_dense = _make_padded_adata(n_obs=14, n_real_genes=8, pad_tail=0, seed=41)
    adata_sp = adata_dense.copy()
    adata_sp.X = sparse.csr_matrix(adata_dense.X)

    cfg = SAVEConfig(
        vae=VAEConfig(
            hidden_dim=8,
            enc_dim=4,
            expand_dim=16,
            enc_num_heads=2,
            dec_num_heads=2,
            enc_depth=1,
            dec_depth=1,
            gene_block_size=8,
            latent_block_size=8,
            is_enc_cond=False,
        ),
        fm=FMConfig(hidden_dim=16, embedding_dim=16, n_blocks=1, num_heads=2),
        train=TrainConfig(vae_epochs=1, fm_epochs=2, batch_size=5, use_amp_vae=False, use_amp_fm=False),
        seed=99,
    )

    m_sp = SAVE(adata_sp, condition_cols=["batch", "cell_type"], config=cfg, device="cpu")
    m_dn = SAVE(adata_dense, condition_cols=["batch", "cell_type"], config=cfg, device="cpu")
    m_dn.vae.load_state_dict(m_sp.vae.state_dict())
    m_sp.init_fm()
    m_dn.init_fm()
    m_dn.fm.denoising_model.load_state_dict(m_sp.fm.denoising_model.state_dict())

    y_ids = m_sp.vocab.encode_obs(m_sp.adata.obs)
    amp_ctx, _ = get_autocast_context(torch.device("cpu"), use_amp=False)
    resident = prepare_resident_tensors(m_dn.adata.X, y_ids, device=torch.device("cpu"))
    mu_ref, std_ref, y_ref, _, _ = _precompute_posteriors(m_sp.vae, resident, 5, torch.device("cpu"), amp_ctx)
    mu_stream, std_stream, y_stream, _, _ = _precompute_posteriors_streaming(
        m_sp.vae, m_sp.adata.X, y_ids, 5, torch.device("cpu"), amp_ctx
    )
    assert torch.allclose(mu_stream, mu_ref, atol=1e-6)
    assert torch.allclose(std_stream, std_ref, atol=1e-6)
    assert torch.equal(y_stream, y_ref)

    loss_sp = m_sp.fit_fm(epochs=2, batch_size=5, seed=123, verbose=False)
    loss_dn = m_dn.fit_fm(epochs=2, batch_size=5, seed=123, verbose=False)
    assert loss_sp["loss"] == pytest.approx(loss_dn["loss"], abs=1e-6)


def test_save_copy_adata_false_and_preallocated_generate_parity():
    from sc_save import FMConfig, SAVE, SAVEConfig, TrainConfig, VAEConfig
    from sc_save.data.preprocessing import MaxAbsExpressionScaler

    adata_pad = _make_padded_adata(n_obs=12, n_real_genes=10, pad_tail=6, seed=55)
    pad_idx = 10
    scaler = MaxAbsExpressionScaler().fit(adata_pad.X)
    adata_pad.X = scaler.transform(adata_pad.X)

    cfg = SAVEConfig(
        vae=VAEConfig(
            hidden_dim=8,
            enc_dim=4,
            expand_dim=16,
            enc_num_heads=2,
            dec_num_heads=2,
            enc_depth=1,
            dec_depth=1,
            gene_block_size=8,
            latent_block_size=8,
            is_enc_cond=False,
        ),
        fm=FMConfig(hidden_dim=16, embedding_dim=16, n_blocks=1, num_heads=2, sample_steps=3),
        train=TrainConfig(vae_epochs=1, fm_epochs=1, batch_size=4, use_amp_vae=False, use_amp_fm=False),
        seed=17,
    )

    model_nocopy = SAVE(
        adata_pad,
        condition_cols=["batch", "cell_type"],
        config=cfg,
        device="cpu",
        scale_data=False,
        pad_to_blocks=False,
        copy_adata=False,
        scaler=scaler,
        pad_idx=pad_idx,
    )
    assert model_nocopy.adata is adata_pad
    model_nocopy.fit(vae_epochs=1, fm_epochs=1, verbose=False)

    cond = {"batch": "b1", "cell_type": "T"}
    torch.manual_seed(777)
    gen_untrimmed = model_nocopy.generate(
        conditions=cond,
        n_cells=8,
        batch_size=3,
        steps=3,
        inverse_scale=False,
        trim_pad=False,
    )
    torch.manual_seed(777)
    gen_trimmed_scaled = model_nocopy.generate(
        conditions=cond,
        n_cells=8,
        batch_size=3,
        steps=3,
        inverse_scale=True,
        trim_pad=True,
    )
    assert gen_untrimmed.shape == (8, 16)
    assert gen_trimmed_scaled.shape == (8, 10)
    expected = scaler.inverse_transform(gen_untrimmed.X)[:, :10]
    np.testing.assert_allclose(np.asarray(gen_trimmed_scaled.X), expected, atol=1e-5)


def test_build_joint_pca_chunked_and_hvg_no_mutation():
    import anndata as ad
    from scipy import sparse
    from sklearn.decomposition import PCA
    from sc_save.eval.generation import (
        build_joint_pca,
        compare_condition_distributions,
        compute_expression_metrics_by_condition,
        get_condition_distribution,
    )

    rng = np.random.default_rng(404)
    gt_x = rng.uniform(0.0, 3.0, size=(20, 12)).astype(np.float32)
    gen_x = rng.uniform(0.0, 3.0, size=(20, 12)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(["b0"] * 10 + ["b1"] * 10),
            "cell_type": pd.Categorical((["ct0"] * 5 + ["ct1"] * 5) * 2),
        },
        index=[f"c_{i}" for i in range(20)],
    )
    var = pd.DataFrame(index=pd.Index([f"g_{i}" for i in range(12)], dtype="object"))

    gt_sp = ad.AnnData(X=sparse.csr_matrix(gt_x), obs=obs.copy(), var=var.copy())
    gen_sp = ad.AnnData(X=gen_x.copy(), obs=obs.copy(), var=var.copy())

    n_comp = build_joint_pca(gt_sp, gen_sp, n_components=5, chunk_size=7)
    assert n_comp == 5

    pca_ref = PCA(n_components=5)
    pca_ref.fit(np.concatenate([gt_x, gen_x], axis=0))
    np.testing.assert_allclose(gt_sp.obsm["X_pca"], pca_ref.transform(gt_x), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(gen_sp.obsm["X_pca"], pca_ref.transform(gen_x), atol=1e-4, rtol=1e-4)

    merged = compare_condition_distributions(
        get_condition_distribution(gt_sp, ["batch", "cell_type"]),
        get_condition_distribution(gen_sp, ["batch", "cell_type"]),
        ["batch", "cell_type"],
        ["batch", "cell_type"],
    )
    var_cols_before = list(gt_sp.var.columns)
    expr_df = compute_expression_metrics_by_condition(
        gt_sp, gen_sp, merged, ["batch", "cell_type"], hvg_top_genes=6, top_gene_count=4
    )
    assert len(expr_df) == 4
    assert list(gt_sp.var.columns) == var_cols_before


def test_streaming_randomized_pca_matches_sklearn_pca():
    import anndata as ad
    from scipy import sparse
    from sklearn.decomposition import PCA
    from sc_save.eval.generation import _streaming_randomized_pca, build_joint_pca

    rng = np.random.default_rng(505)
    low_rank = rng.normal(size=(48, 6)).astype(np.float32)
    proj = rng.normal(size=(6, 40)).astype(np.float32)
    base = np.maximum(0.0, low_rank @ proj + 0.05 * rng.normal(size=(48, 40))).astype(np.float32)
    gt_x = base[:24]
    gen_x = base[24:]
    gt_sp_mat = sparse.csr_matrix(gt_x)

    gt_pca, gen_pca = _streaming_randomized_pca(
        gt_sp_mat, gen_x, n_components=5, chunk_size=7, n_oversamples=12, n_iter=6, seed=0
    )
    comb_pca = np.concatenate([gt_pca, gen_pca], axis=0)

    ref_pca = PCA(n_components=5, svd_solver="full").fit_transform(np.concatenate([gt_x, gen_x], axis=0))
    # Singular values / Euclidean pairwise distances are invariant to eigenvector sign flips.
    np.testing.assert_allclose(np.var(comb_pca, axis=0), np.var(ref_pca, axis=0), atol=1e-3, rtol=1e-3)
    d_stream = np.linalg.norm(comb_pca[:, None, :] - comb_pca[None, :, :], axis=-1)
    d_ref = np.linalg.norm(ref_pca[:, None, :] - ref_pca[None, :, :], axis=-1)
    np.testing.assert_allclose(d_stream, d_ref, atol=1e-3, rtol=1e-3)

    obs = pd.DataFrame({"cell_type": ["T"] * 24}, index=[f"c_{i}" for i in range(24)])
    var = pd.DataFrame(index=pd.Index([f"g_{i}" for i in range(40)], dtype="object"))
    gt_ad = ad.AnnData(X=gt_sp_mat, obs=obs.copy(), var=var.copy())
    gen_ad = ad.AnnData(X=gen_x.copy(), obs=obs.copy(), var=var.copy())
    k = build_joint_pca(gt_ad, gen_ad, n_components=5, chunk_size=7, streaming_threshold_bytes=1)
    assert k == 5
    np.testing.assert_allclose(np.abs(gt_ad.obsm["X_pca"]), np.abs(gt_pca), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.abs(gen_ad.obsm["X_pca"]), np.abs(gen_pca), atol=1e-4, rtol=1e-4)
