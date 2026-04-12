import torch
import math
import einops
import torch.nn as nn
import random
import numpy as np
from torch.distributions import Normal
try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None
from itertools import combinations
from einops.layers.torch import Rearrange
from torch.nn import functional as F


def modulate(x, shift, scale):
    # 对数据做 shift 和 scale 的方法
    # scale.unsqueeze(1) 将 scale 转换成列向量
    res = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return res


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_output_attn_weight=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, is_output_attn_weight=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if is_output_attn_weight:
            return x, attn
        else:
            return x


class AttentionLinear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # x = q + k + v
        # x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if flash_attn_qkvpacked_func is None:
            raise ImportError(
                "flash_attn is not installed, but FlashAttention was requested."
            )
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        x = flash_attn_qkvpacked_func(qkv)
        x = einops.rearrange(x, "b n h d -> b n (h d)")
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTblock(nn.Module):
    # adaLN -> attn -> mlp
    def __init__(
        self,
        feature_dim=2000,
        mlp_ratio=4.0,
        types_of_conds=2,
        num_heads=10,
        is_linear_replace=False,
        dropout=0.0,
        cond_norm="ln",
        **kwargs,
    ) -> None:
        super().__init__()

        self.types_of_conds = types_of_conds

        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)

        if is_linear_replace:
            self.attn = AttentionLinear(
                feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs
            )
        else:
            self.attn = Attention(
                feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs
            )

        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=feature_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=dropout,
        )

        if cond_norm == "ln":
            self.adaLN_modulation = nn.Sequential(
                Rearrange("b c e -> b (c e)"),
                # block 1
                nn.Linear(feature_dim * self.types_of_conds, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                # block 2
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                # block 3
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                # final out
                nn.Linear(feature_dim, 6 * feature_dim, bias=True),
            )

        elif cond_norm == "bn":
            self.adaLN_modulation = nn.Sequential(
                Rearrange("b c e -> b (c e)"),
                nn.Linear(feature_dim * self.types_of_conds, feature_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, 6 * feature_dim, bias=True),
            )

    def forward(self, x, c, is_output_attn_weight=False):
        # 将 condition 投影到 6 * hiddensize 之后沿列切成 6 份
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # attention blk
        if is_output_attn_weight:
            attn_out, attn_weight = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                is_output_attn_weight=True,
            )
        else:
            attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        x = x + gate_msa.unsqueeze(1) * attn_out
        # mlp blk 采用 ViT 中实现的版本
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        if is_output_attn_weight:
            return x, attn_weight
        else:
            return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    将 time emb 成 frequency_embedding_size 维，再投影到 hidden_size
    """

    def __init__(self, hidden_size=512, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # 将输入 emb 成 frequency_embedding_size 维
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        # 最后一维加 None 相当于拓展一维
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # 采用 pos emb 之后过 mlp
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AttentionBlock(nn.Module):
    def __init__(
        self,
        feature_dim=2000,
        mlp_ratio=4.0,
        num_heads=10,
        norm_type="bn",
        affine=True,
        expand_dim=32,
        is_flash_attn=False,
        dropout=0,
        **kwargs,
    ) -> None:
        super().__init__()

        if norm_type == "bn":
            self.norm1 = nn.BatchNorm1d(feature_dim, affine=affine, eps=1e-6)
            self.norm2 = nn.BatchNorm1d(feature_dim, affine=affine, eps=1e-6)
        elif norm_type == "ln":
            self.norm1 = nn.LayerNorm(expand_dim, elementwise_affine=affine, eps=1e-6)
            self.norm2 = nn.LayerNorm(expand_dim, elementwise_affine=affine, eps=1e-6)

        if is_flash_attn:
            self.attn = FlashAttention(
                expand_dim, num_heads=num_heads, qkv_bias=True, **kwargs
            )
        else:
            self.attn = Attention(
                expand_dim, num_heads=num_heads, qkv_bias=True, **kwargs
            )

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_dim = int(expand_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=expand_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=dropout,
        )

    def forward(self, x, c=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LinearBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=1024,
        depth=3,
        dropout=0.5,
        types_of_conds=2,
        feature_dim=256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.types_of_conds = types_of_conds
        self.network = nn.ModuleList()
        for _ in range(depth):
            self.network.append(
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.PReLU(),
                )
            )

        self.adaLN_modulation = nn.Sequential(
            Rearrange("b c e -> b (c e)"),
            # block 1
            nn.Linear(feature_dim * self.types_of_conds, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            # block 2
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            # block 3
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            # final out
            nn.Linear(feature_dim, hidden_dim, bias=True),
        )

    def forward(self, x, c):
        # shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        c = self.adaLN_modulation(c)
        # x = x + c
        for i, blk in enumerate(self.network):
            if i < len(self.network) - 1:
                x = blk(x + c) + x
            else:
                x = blk(x + c)
        return x


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        depth,
        num_heads,
        enc_dim,
        bn_affine=True,
        norm_type="bn",
        expand_dim=32,
        is_linear_replace_attn=False,
        is_reshape_first=False,
        gene_block_size=40,
        is_linear=False,
        dropout=0.2,
        is_enc_cond=False,
        types_of_conds=2,
        mu_var_mlp_ratio=7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.expand_dim = expand_dim

        # self.input_layer = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        # )

        # self.expand_layer = nn.Linear(1, expand_dim)
        self.types_of_conds = types_of_conds
        self.is_linear_replace_attn = is_linear_replace_attn
        self.is_enc_cond = is_enc_cond

        if not is_linear_replace_attn:
            if is_reshape_first:
                num_gene_block = input_dim // gene_block_size
                self.tokenzier = nn.Sequential(
                    Rearrange("b (m n) -> b m n", m=num_gene_block, n=gene_block_size),
                    # nn.Linear(gene_block_size, expand_dim, bias=False),
                    # nn.Dropout(p=0.4),
                    nn.Linear(gene_block_size, expand_dim),
                    # Rearrange("c (h e) -> c h e", h=hidden_dim, e=expand_dim),
                )
                self.pos_embding = nn.Parameter(torch.zeros(num_gene_block, expand_dim))
                nn.init.trunc_normal_(self.pos_embding, std=0.02)
            else:
                self.tokenzier = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * expand_dim),
                    Rearrange("c (h e) -> c h e", h=hidden_dim, e=expand_dim),
                )
                self.pos_embding = nn.Parameter(torch.zeros(hidden_dim, expand_dim))
                nn.init.trunc_normal_(self.pos_embding, std=0.02)

            self.blks = nn.ModuleList()
            if is_enc_cond:
                for _ in range(depth):
                    self.blks.append(
                        DiTblock(
                            feature_dim=expand_dim,
                            mlp_ratio=4.0,
                            num_heads=num_heads,
                            # is_linear_replace=is_linear_replace_attn,
                            dropout=dropout,
                            types_of_conds=types_of_conds,
                        )
                    )

            if is_reshape_first:
                self.var_enc = nn.Sequential(
                    # Rearrange("b m n -> b (m n)"),
                    nn.Linear(expand_dim, expand_dim * 4),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(expand_dim * 4, enc_dim),
                )
                self.mu_enc = nn.Sequential(
                    # Rearrange("b m n -> b (m n)"),
                    nn.Linear(expand_dim, expand_dim * 4),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(expand_dim * 4, enc_dim),
                )
                # 对齐 mu var predictor 之前的维度
                # self.var_enc = nn.Sequential(
                #     nn.Linear(
                #         expand_dim, gene_block_size * mu_var_mlp_ratio, bias=False
                #     ),
                #     Rearrange("b m n -> b (m n)"),
                #     nn.Linear(
                #         num_gene_block * gene_block_size * mu_var_mlp_ratio, enc_dim
                #     ),
                # )
                # self.mu_enc = nn.Sequential(
                #     nn.Linear(
                #         expand_dim, gene_block_size * mu_var_mlp_ratio, bias=False
                #     ),
                #     Rearrange("b m n -> b (m n)"),
                #     nn.Linear(
                #         num_gene_block * gene_block_size * mu_var_mlp_ratio, enc_dim
                #     ),
                # )
            else:
                self.var_enc = nn.Sequential(
                    Rearrange("c h e -> c (h e)"),
                    nn.Linear(expand_dim * hidden_dim, enc_dim),
                )
                self.mu_enc = nn.Sequential(
                    Rearrange("c h e -> c (h e)"),
                    nn.Linear(expand_dim * hidden_dim, enc_dim),
                )
        else:
            linear_dim = 1024
            self.tokenzier = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(input_dim, linear_dim),
                nn.BatchNorm1d(linear_dim),
                nn.PReLU(),
            )
            self.blks = nn.ModuleList()
            for _ in range(2):
                self.blks.append(
                    LinearBlock(
                        hidden_dim=linear_dim,
                        depth=depth,
                        types_of_conds=types_of_conds,
                        dropout=dropout,
                    )
                )

            self.mu_enc = nn.Sequential(
                nn.Linear(linear_dim, enc_dim),
            )
            self.var_enc = nn.Sequential(
                nn.Linear(linear_dim, enc_dim),
            )

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, c=None):
        if self.is_linear_replace_attn:
            h = self.tokenzier(x)
        else:
            h = self.tokenzier(x) + self.pos_embding

        if self.is_enc_cond:
            for blk in self.blks:
                h = blk(h, c)
        else:
            for blk in self.blks:
                h = blk(h)

        mu = self.mu_enc(h)

        # make sure var>0
        var = torch.clamp(torch.exp(self.var_enc(h)), min=1e-20)
        z = self.reparameterize(mu, var)

        # return torch.tanh(z), mu, var
        return z, mu, var


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, out_dim, bias=True),
            nn.Sigmoid(),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        depth,
        out_dim,
        expand_dim=32,
        is_norm_init=False,
        block_type="adaLN",
        len_col_comb=None,
        types_of_cond=2,
        is_reshape_first=False,
        latent_block_size=4,
        is_linear_replace_attn=False,
        out_layer_dropout=0.4,
        is_linear=False,
        is_out_moe=False,
        attn_dropout=0.0,
        mlp_ratio=32,
    ) -> None:
        super().__init__()

        self.block_type = block_type
        self.blks = nn.ModuleList()
        self.input_dim = input_dim
        self.expand_dim = expand_dim
        self.types_of_cond = types_of_cond
        self.is_linear_replace_attn = is_linear_replace_attn

        if not is_linear_replace_attn:
            if is_reshape_first:
                num_latent_block = out_dim // latent_block_size
                # self.reshape_layer = nn.Sequential(
                #     Rearrange(
                #         "b (m n) -> b m n", m=num_latent_block, n=latent_block_size
                #     ),
                #     nn.Linear(latent_block_size, expand_dim),
                # )
                self.reshape_layer = nn.Sequential(
                    nn.Linear(input_dim, expand_dim),
                )

                self.pos_embding = nn.Parameter(
                    torch.zeros(num_latent_block, expand_dim)
                )
                nn.init.trunc_normal_(self.pos_embding, std=0.02)

            else:
                self.reshape_layer = nn.Sequential(
                    Rearrange("b h -> b h 1"),
                    nn.Linear(1, expand_dim),
                )
                self.pos_embding = nn.Parameter(torch.zeros(hidden_dim, expand_dim))
                nn.init.trunc_normal_(self.pos_embding, std=0.02)

            # self.batch_embedder = TimestepEmbedder(hidden_size=expand_dim)

            for _ in range(depth):
                self.blks.append(
                    DiTblock(
                        feature_dim=expand_dim,
                        mlp_ratio=4.0,
                        num_heads=num_heads,
                        types_of_conds=types_of_cond,
                        dropout=attn_dropout,
                    )
                )

            # logits_layer = nn.Identity() if is_flash_attn else nn.Sigmoid()
            logits_layer = nn.Sigmoid()

            if is_reshape_first:
                # 保持最后一个线性层的信息总量为 out_dim_max
                # if expand_dim * num_latent_block > out_dim_max:
                #     out_middle_dim = out_dim_max // num_latent_block
                #     self.out_layer = nn.Sequential(
                #         nn.Linear(expand_dim, out_middle_dim),
                #         Rearrange("b m n -> b (m n)"),
                #         nn.Linear(out_middle_dim * num_latent_block, out_dim),
                #         logits_layer,
                #     )
                # else:
                #     self.out_layer = nn.Sequential(
                #         Rearrange("b m n -> b (m n)"),
                #         nn.Linear(expand_dim * num_latent_block, out_dim),
                #         logits_layer,
                #     )
                self.out_layer = nn.Sequential(
                    nn.Linear(expand_dim, out_dim // num_latent_block, bias=False),
                    Rearrange("b m n -> b (m n)"),
                    # nn.Linear(
                    #     latent_block_size * num_latent_block * mlp_ratio, out_dim
                    # ),
                    logits_layer,
                )
            else:
                self.out_layer = nn.Sequential(
                    Rearrange("c h e -> c (h e)"),
                    nn.Linear(hidden_dim * expand_dim, out_dim),
                    logits_layer,
                )
        else:
            linear_dim = 1024
            self.reshape_layer = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(input_dim, linear_dim),
                nn.BatchNorm1d(linear_dim),
                nn.PReLU(),
            )
            self.blks = nn.ModuleList()
            for _ in range(3):
                self.blks.append(
                    LinearBlock(
                        hidden_dim=linear_dim,
                        depth=depth,
                        types_of_conds=types_of_cond,
                        dropout=out_layer_dropout,
                    )
                )
            self.out_layer = nn.Sequential(
                nn.Linear(linear_dim, out_dim),
                nn.Sigmoid(),
            )
        # self.final_cond_layer = FinalLayer(hidden_size=expand_dim, out_dim=1)
        # self.out_layer = nn.Sequential(
        #     Rearrange("c h e -> c (h e)"), nn.Linear(hidden_dim, out_dim), nn.Sigmoid()
        # )

        if is_norm_init:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize batch embedding MLP:
        # nn.init.normal_(self.batch_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.batch_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:

        # nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.out_layer.linear[0].weight, 0)
        # nn.init.constant_(self.out_layer.linear[0].bias, 0)

    def forward(
        self,
        x,
        c,
        # col_msk_threshold=0.8,
        return_logits=False,
        is_output_attn_weight=False,
        # is_col_wise_mask=True,
    ):
        if self.block_type == "adaLN":
            # y_flat = einops.rearrange(y, "b c -> (b c)")
            # c = self.batch_embedder(y_flat)
            # c_stk = einops.rearrange(c, "(b c) e -> b c e", b=y.shape[0])
            # c_sum = einops.reduce(c, "b c e -> b e", reduction="sum")

            if self.is_linear_replace_attn:
                h = self.reshape_layer(x)
            else:
                h = self.reshape_layer(x) + self.pos_embding

            if is_output_attn_weight:
                attn_weights = []
                for blk in self.blks:
                    h, attn_weight = blk(h, c, is_output_attn_weight=True)
                    attn_weights.append(attn_weight)
            else:
                for blk in self.blks:
                    h = blk(h, c)

        elif self.block_type == "dsbn":
            h = self.input_layer(x, c)
            h = self.proj(h)
            for blk in self.blks:
                h = blk(h)

        # h = einops.rearrange(h, "c h e -> c (h e)")
        if return_logits:
            return self.out_layer(h), h

        if is_output_attn_weight:
            return self.out_layer(h), attn_weights

        return self.out_layer(h)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        enc_num_heads,
        dec_num_heads,
        enc_depth,
        dec_depth,
        enc_dim=10,
        expand_dim=32,
        enc_affine=True,
        enc_norm_type="bn",
        dec_norm_init=False,
        dec_blk_type="adaLN",
        is_initialize=False,
        len_col_comb=None,
        mask_id=0,
        label_encoder_dict=None,
        is_multi_codebook=False,
        # types_of_conds=2,
        is_reshape_first=False,
        gene_block_size=40,
        latent_block_size=4,
        # is_flash_attn=False,
        out_layer_dropout=0.4,
        is_enc_linear=False,
        is_dec_linear=False,
        is_out_moe=False,
        enc_attn_dropout=0.2,
        dec_attn_dropout=0.0,
        is_enc_cond=False,
        is_linear_replace_attn=False,
        outer_emb_dim=768,
    ) -> None:
        super().__init__()
        if is_reshape_first:
            print(f"is_reshape_first: {is_reshape_first}")
        # for parameter record
        self.hidden_dim = hidden_dim
        self.latent_dim = enc_dim
        # self.num_class = num_conds

        self.enc_heads = enc_num_heads
        self.dec_heads = dec_num_heads

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth

        self.enc_affine = enc_affine
        self.enc_norm_type = enc_norm_type
        self.expand_dim = expand_dim

        self.dec_blk_type = dec_blk_type

        self.types_of_conds = len(label_encoder_dict.keys())
        self.is_multi_codebook = is_multi_codebook
        if len_col_comb:
            col_comb = []
            leave_one_comb = []
            for i in range(1, len_col_comb + 1):
                combs = list(combinations(range(len_col_comb), i))
                col_comb.extend(combs)
                if i == len_col_comb - 1:
                    leave_one_comb = [list(x) for x in combs]
            col_comb = [list(x) if len(x) > 1 else x[0] for x in col_comb]

            # other_comb_size = len(col_comb) - len(leave_one_comb)
            # 其他组合数量 / (1+n) * 只留一个组合数量 = 0.2, 尽量优化单条件的情况
            # other_comb_size /  (1+n) * len(leave_one_comb) = 0.2
            # n = 4 * other_comb_size / len(leave_one_comb) - 1
            # print("total col comb is:", len(col_comb))
            # for _ in range(int(n)):
            #     # col_comb.extend(leave_one_comb)
            #     pass
            self.col_comb = col_comb

        print(f"is_enc_cond: {is_enc_cond}")
        self.mask_id = mask_id
        if is_multi_codebook:
            self.cond_embedder = nn.ModuleList(
                [
                    nn.Embedding(len(label_encoder.classes_), expand_dim)
                    for label_encoder in label_encoder_dict.values()
                ]
            )
        else:
            total_class = sum(
                [
                    len(label_encoder.classes_)
                    for label_encoder in label_encoder_dict.values()
                ]
            )
            # 减掉 n-1 个重复的 mask 类别
            total_class -= len(label_encoder_dict.keys()) - 1
            self.cond_embedder = nn.Embedding(total_class, expand_dim)

        self.outer_emb_dim = outer_emb_dim
        self.txt_emb_proj = nn.ModuleList(
            [nn.Linear(outer_emb_dim[i], expand_dim) for i in range(len(outer_emb_dim))]
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
            is_linear_replace_attn=is_linear_replace_attn,
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
            # is_flash_attn=is_flash_attn,
            is_linear_replace_attn=is_linear_replace_attn,
            out_layer_dropout=out_layer_dropout,
            is_linear=is_dec_linear,
            is_out_moe=is_out_moe,
            attn_dropout=dec_attn_dropout,
            types_of_cond=self.types_of_conds,
        )

        print(f"enc_dropout: {enc_attn_dropout}, dec_dropout: {dec_attn_dropout}")

        # self.cond_predictor = Classifier(
        #     input_dim=expand_dim, num_classes=num_conds, types_of_conds=types_of_conds
        # )

        if is_initialize:
            self.initialize_weights()

        ### 统计参数量
        # encoder param
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"encoder params: {total_params / 1e6:.2f}M")

        # decoder param
        total_params = sum(p.numel() for p in self.decoder.parameters())
        # for name, param in self.decoder.named_parameters():
        #     print(name, param.numel() / 1e6, "M")
        print(f"decoder params: {total_params / 1e6:.2f}M")

        # total param
        total_params = sum(p.numel() for p in self.parameters())
        print(f"total params: {total_params / 1e6:.2f}M")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
