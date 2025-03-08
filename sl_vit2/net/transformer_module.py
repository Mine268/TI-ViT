import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """PE module"""
    def __init__(self, d_model: int, max_len: int = 512, mode: str = 'absolute'):
        super(PositionalEncoding, self).__init__()
        self.mode = mode
        self.d_model = d_model

        if mode == 'absolute':
            self.pe = nn.Embedding(max_len, d_model)
            self.register_buffer('positions', torch.arange(max_len))
        elif mode == 'relative':
            self.max_rel_dist = max_len
            self.rel_k = nn.Parameter(torch.randn(2*max_len+1, d_model)//math.sqrt(d_model))
        else:
            raise ValueError(f"Unsupported position mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [batch, seq, dim]"""
        seq_len = x.size(1)

        if self.mode == 'absolute':
            positions = self.positions[:seq_len].unsqueeze(0)  # [1, seq]
            pos_embed = self.pe(positions)  # [1, seq, dim]
            return x + pos_embed
        elif self.mode == 'relative':
            rel_dist = torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]  # [seq, seq]
            rel_dist = rel_dist.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
            rel_bias = self.rel_k[rel_dist]  # [seq, seq, dim]
            return x + rel_bias[None,:,:,:].sum(dim=2)


class ContinuousAngleEmbedding(nn.Module):
    def __init__(
        self,
        output_dim=64,
        num_freq=16,
        learnable_freq=True,
        max_angle=2*math.pi,
        epsilon=1e-6
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_freq = num_freq
        self.max_angle = max_angle
        self.epsilon = epsilon

        self.freq_base = nn.Parameter(
            torch.logspace(0, 1, num_freq, base=10).float(),
            requires_grad=learnable_freq
        )

        self.proj = nn.Sequential(
            nn.Linear(2 * num_freq, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, angles):
        """
        Args:
            angles (torch.Tensor): Size=[...]

        Returns:
            (torch.Tensor): Size=[..., output_dim]
        """
        # normalize to [0,1], then scale to [0,2 pi]
        angles %= self.max_angle
        angles = angles / self.max_angle * 2 * math.pi

        scaled_angles = angles.unsqueeze(-1) * self.freq_base

        sin_enc = torch.sin(scaled_angles)  # [..., num_freq]
        cos_enc = torch.cos(scaled_angles)  # [..., num_freq]
        raw_enc = torch.cat([sin_enc, cos_enc], dim=-1)

        embeddings = self.proj(raw_enc)
        return embeddings


class LoraCompatibleMHA(nn.Module):
    """LoRA compatible multi-head attention
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super(LoraCompatibleMHA, self).__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads,
            kdim=embed_dim, vdim=embed_dim,
            batch_first=True
        )

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        return self.mha(q, k, v, need_weights=False)[0]


class FeedForwardNetwork(nn.Module):
    """FFN
    """
    def __init__(self, dim):
        super(FeedForwardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super(TransformerBlock, self).__init__()
        self.pe = PositionalEncoding(dim, mode='absolute')
        self.attn = LoraCompatibleMHA(dim, num_heads)
        self.ffn = FeedForwardNetwork(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.pe(x)

        y = self.attn(x, x, x)
        y = self.norm1(y)
        x = x + y

        y = self.ffn(x)
        y = self.norm2(y)
        x = x + y
        return x

