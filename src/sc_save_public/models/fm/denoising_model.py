import numpy as np
import torch
from torch import nn

from sc_save_public.models.VAE_model import DiTblock
from sc_save_public.models.fm.layer_utils import Linear
from einops.layers.torch import Rearrange
from sc_save_public.models.VAE_model import TimestepEmbedder


# Util functions
def zero_init(module):
    """
    Initializes the weights and biases of a PyTorch module with zero values.

    Args:
        module (torch.nn.Module): PyTorch module for weight and bias initialization.

    Returns:
        torch.nn.Module: The input module with weights and biases initialized to zero.
    """
    nn.init.constant_(module.weight.data, 0)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias.data, 0)
    return module


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    """
    Generates a sinusoidal embedding for a sequence of timesteps.

    Args:
        timesteps (torch.Tensor): 1-dimensional tensor representing the input timesteps.
        embedding_dim (int): Dimensionality of the embedding. It must be an even number.
        dtype (torch.dtype, optional): Data type for the resulting tensor. Default is torch.float32.
        max_timescale (float, optional): Maximum timescale value for the sinusoidal embedding. Default is 10,000.
        min_timescale (float, optional): Minimum timescale value for the sinusoidal embedding. Default is 1.

    Returns:
        torch.Tensor: Sinusoidal embedding tensor for the input timesteps with the specified embedding_dim.
    """
    # Adapted from tensor2tensor and VDM codebase.
    # 这里 assert timesteps.ndim == 1 是为了确保 timesteps 是一维张量。
    # 因为后续的嵌入计算（如 timesteps.to(dtype)[:, None]）假设 timesteps 是一维的，
    # 这样才能正确地扩展维度并与 inv_timescales 进行广播运算。
    # 如果 timesteps 不是一维的，后续的操作会报错或得到意外的结果。
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    assert (
        timesteps.ndim == 1
    ), f"timesteps must be 1-dimensional, but got {timesteps.ndim} {timesteps}"
    assert embedding_dim % 2 == 0
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


# ResNet MLP
class MLPTimeStep(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout_prob: int,
        n_blocks: int,
        seq_length: int,
        embedding_dim=128,
        label_encoder_dict=None,
        mask_id=0,
        num_conds=2,
        normalization="layer",
        conditional=True,
        is_binarized=False,
        modality_list=None,
        conditioning_probability=0.8,
        guided_conditioning=True,
        block_type="DiT",
        num_heads=4,
    ):
        super().__init__()

        # Gene expression dimension
        self.in_dim = in_dim

        # The network downsizes the input multiple times
        self.hidden_dim = hidden_dim

        # Initialize attributes
        # self.size_factor_min = size_factor_min
        # self.size_factor_max = size_factor_max
        # self.embed_size_factor = embed_size_factor
        self.embedding_dim = embedding_dim
        self.conditional = conditional
        self.is_binarized = is_binarized
        self.block_type = block_type
        # self.covariate_list = covariate_list
        self.modality_list = modality_list
        self.conditioning_probability = (
            conditioning_probability  # Conditioning probability during the guiding
        )
        self.guided_conditioning = guided_conditioning

        self.pos_embding = nn.Parameter(torch.zeros(seq_length, embedding_dim))
        nn.init.trunc_normal_(self.pos_embding, std=0.02)

        # Time embedding network
        # self.time_embedder = nn.Sequential(
        #     Linear(embedding_dim, embedding_dim),  # Time embedding
        #     nn.SiLU(),
        #     Linear(embedding_dim, embedding_dim),
        # )
        self.time_embedder = TimestepEmbedder(hidden_size=embedding_dim)

        # Size factor embeddings
        # if embed_size_factor:
        #     self.size_factor_embedder = nn.Sequential(
        #         Linear(embedding_dim, embedding_dim),  # Size factor embedding
        #         nn.SiLU(),
        #         Linear(embedding_dim, embedding_dim),
        #     )

        # Initial convolution
        self.net_in = Linear(in_dim, self.hidden_dim)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.blocks = []

        # Dimensionality preserving Resnet in the bottleneck
        if block_type == "Resnet":
            for _ in range(n_blocks):
                self.blocks.append(
                    ResnetBlock(
                        in_dim=self.hidden_dim * seq_length,
                        dropout_prob=dropout_prob,
                        embedding_dim=embedding_dim,
                        normalization="batch",
                        seq_length=seq_length,
                        types_of_conds=num_conds,
                    )
                )

        elif block_type == "DiT":
            for _ in range(n_blocks):
                self.blocks.append(
                    DiTblock(
                        feature_dim=embedding_dim,
                        mlp_ratio=4.0,
                        num_heads=num_heads,
                        types_of_conds=num_conds,
                        dropout=dropout_prob,
                        cond_norm="bn",
                    )
                )

        self.mask_id = mask_id
        total_class = sum(
            [
                len(label_encoder.classes_)
                for label_encoder in label_encoder_dict.values()
            ]
        )
        # 减掉 n-1 个重复的 mask 类别
        total_class -= len(label_encoder_dict.keys()) - 1
        self.cond_embedder = nn.Embedding(total_class, embedding_dim)

        # self.cond_proj = nn.Sequential(
        #     Rearrange("b c e -> b (c e)"),
        #     nn.Linear(embedding_dim * num_conds, embedding_dim),
        #     nn.SiLU(),
        #     nn.Linear(embedding_dim, embedding_dim),
        # )

        # Set up blocks
        self.blocks = nn.ModuleList(self.blocks)

        # if normalization not in ["layer", "batch"]:
        #     self.net_out = nn.Sequential(nn.SiLU(), Linear(self.hidden_dim, in_dim))
        # else:
        #     self.net_out = nn.Sequential(
        #         nn.LayerNorm(self.hidden_dim)
        #         if normalization == "layer"
        #         else nn.BatchNorm1d(num_features=self.hidden_dim),
        #         nn.SiLU(),
        #         Linear(self.hidden_dim, in_dim),
        #     )

        self.net_out = nn.Sequential(
            nn.GELU(approximate="tanh"),
            Linear(self.hidden_dim, in_dim),
            # nn.Tanh(),
        )

        self.count_parameters_in_M()
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
        if self.block_type == "DiT":
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # 统计模型总参数量，单位为 M（百万）
    def count_parameters_in_M(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"FM params: {total_params / 1e6:.2f} M")
        return total_params / 1e6

    def forward(self, x, t, y, inference=False, unconditional=False, is_output_attn_weight=False):
        # Make a copy of time for using in time embeddings
        t_for_embeddings = t.clone().detach().squeeze()
        # y = self.cond_proj(y)

        # Time embedding
        # emb = self.time_embedder(
        #     get_timestep_embedding(t_for_embeddings, self.embedding_dim)
        # )
        # emb = emb.unsqueeze(1).repeat(1, y.shape[1], 1)
        if t_for_embeddings.ndim == 0:
            t_for_embeddings = t_for_embeddings.unsqueeze(0)

        emb = self.time_embedder(t_for_embeddings * 1000.0)
        emb = emb.unsqueeze(1)

        # Embed condition
        if not inference:
            random_mask = (
                torch.rand(y.shape[0], y.shape[1]).to(y.device)
                < self.conditioning_probability
            )  # (batch_size,)

            while random_mask.dim() < y.dim():
                random_mask = random_mask.unsqueeze(-1)
            random_mask = random_mask.expand_as(y)
            y = torch.where(random_mask, y, torch.zeros_like(y))

        elif inference:
            if not unconditional:
                pass
            else:
                y = torch.zeros_like(y)

        emb = emb + y

        # if self.guided_conditioning:
        #     is_conditioned = (
        #         torch.bernoulli(torch.tensor([self.conditioning_probability])).item()
        #         if not inference
        #         else 1.0
        #     )  # Bernoulli variable to decide whether to condition or not

        #     if self.conditional and is_conditioned and not unconditional:
        #         print(is_conditioned)
        #         emb = emb + y
        # else:
        #     # Normal conditioning
        #     emb = emb + y

        # emb = emb + y
        # Compute prediction

        h = self.net_in(x)
        h = h + self.pos_embding

        if is_output_attn_weight and self.block_type == "DiT":
            # Only output attention weight from the last layer
            for i, block in enumerate(self.blocks):
                if i == len(self.blocks) - 1:  # Last layer
                    h, attn_weight = block(h, emb, is_output_attn_weight=True)
                else:
                    h = block(h, emb)
            pred = self.net_out(h)
            return pred, attn_weight
        else:
            for block in self.blocks:  # n_blocks times
                h = block(h, emb)
            pred = self.net_out(h)
            return pred


class ResnetBlock(nn.Module):
    """
    A block for a Multi-Layer Perceptron (MLP) with skip connection.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output features. Defaults to None, in which case it's set equal to input_dim.
        condition_dim (int, optional): Dimension of the conditional input. Defaults to None.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.0.
        norm_groups (int, optional): Number of groups for layer normalization. Defaults to 32.
    """

    def __init__(
        self,
        in_dim,
        out_dim=None,
        dropout_prob=0.0,
        embedding_dim=None,
        normalization="batch",
        seq_length=47,
        types_of_conds=2,
    ):
        super().__init__()

        # Variables controlling if time and size factor should be embedded
        self.embedding_dim = embedding_dim

        # Set output_dim to input_dim if not provided
        out_dim = in_dim if out_dim is None else out_dim
        self.out_dim = out_dim

        # First linear block with LayerNorm and SiLU activation
        if normalization not in ["layer", "batch"]:
            self.net1 = nn.Sequential(nn.SiLU(), Linear(in_dim, out_dim))
        else:
            self.net1 = nn.Sequential(
                Rearrange("b s e -> b (s e)"),
                nn.Linear(in_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
                if normalization == "layer"
                else nn.BatchNorm1d(num_features=embedding_dim),
                nn.SiLU(),
                Linear(embedding_dim, embedding_dim),
            )

        # Projections for conditions
        self.cond_proj = nn.Sequential(
            Rearrange("b c e -> b (c e)"),
            # block 1
            nn.Linear(embedding_dim * types_of_conds, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            # block 2
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            # block 3
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            # final out
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        )

        # Second linear block with LayerNorm, SiLU activation, and optional dropout
        if normalization not in ["layer", "batch"]:
            self.net2 = nn.Sequential(
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(Linear(out_dim, out_dim)),
            )
        else:
            self.net2 = nn.Sequential(
                nn.LayerNorm(embedding_dim)
                if normalization == "layer"
                else nn.BatchNorm1d(num_features=embedding_dim),
                nn.SiLU(),
                *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
                zero_init(Linear(embedding_dim, in_dim)),
                Rearrange("b (s e) -> b s e", s=seq_length),
            )

        # Linear projection for skip connection if input_dim and output_dim differ
        if in_dim != out_dim:
            self.skip_proj = Linear(in_dim, out_dim)

    def forward(self, x, emb):
        """
        Forward pass of the MLP block.

        Args:
            x (torch.Tensor): Input features.
            condition (torch.Tensor, optional): Conditional input. Defaults to None.

        Returns:
            torch.Tensor: Output features.
        """
        # Forward pass through the first linear block
        h = self.net1(x)

        # Condition time and library size
        emb = self.cond_proj(emb)
        h = h + emb

        # Forward pass through the second linear block
        h = self.net2(h)

        # Linear projection for skip connection if input_dim and output_dim differ
        # if x.shape[1] != self.out_dim:
        #     x = self.skip_proj(x)

        # Add skip connection to the output
        assert x.shape == h.shape

        return x + h
