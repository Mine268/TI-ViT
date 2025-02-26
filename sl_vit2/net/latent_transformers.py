from typing import *
from functools import partial

import math
import torch
import torch.nn as nn
from einops import *


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


class ImageLatentTransformerGroup(nn.Module):
    def __init__(
        self,
        num_layers: int=1,
        embed_dim: int=768,
        num_heads: int=12,
    ):
        """
        Args:
            num_layers (int): Number of transformer block for latent transformations. Defaults to 3.
            embed_dim (int): Image token dim. Defaults to 768.
        """
        super(ImageLatentTransformerGroup, self).__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # HF: horizontal flip
        self.hf = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)])
        # CR: center-oriented rotation
        self.cr = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)])
        # HR: horizontal flip then center-oriented rotation
        self.hr = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)])

        # Rotation embedding
        self.angle_embedder = ContinuousAngleEmbedding(output_dim=embed_dim, num_freq=32)

        # key: (first_op, second_op)
        self.composition_law = {
            (self.do_hf, self.do_hf): (self.do_cr, 0, 0),
            (self.do_hf, self.do_cr): (self.do_hr, 0, 1),
            (self.do_hf, self.do_hr): (self.do_cr, 0, 1),
            (self.do_cr, self.do_hf): (self.do_hr, -1, 0),
            (self.do_cr, self.do_cr): (self.do_cr, 1, 1),
            (self.do_cr, self.do_hr): (self.do_hr, -1, 1),
            (self.do_hr, self.do_hf): (self.do_cr, -1, 0),
            (self.do_hr, self.do_cr): (self.do_hr, 1, 1),
            (self.do_hr, self.do_hr): (self.do_cr, -1, 1),
        }

    def __repr__(self):
        return f"ImageLatentTransformerGroup(" + \
            f"num_layer={self.num_layers}, embed_dim={self.embed_dim}, num_heads={self.num_heads})"

    def do_hf(self, patches: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _ = args, kwargs
        """
        Args:
            patches (torch.Tensor): Size=(N,L,D), output size=(N,L,D).
        """
        return self.hf(patches)

    def do_cr(self,
        patches: torch.Tensor, angle_rad: Union[torch.Tensor, List, None], *args, **kwargs
    ) -> torch.Tensor:
        _, _ = args, kwargs
        """
        Args:
            patches (torch.Tensor): As `do_hf`.
            angle_rad (Union[torch.Tensor, List, None]): Angles rotated for each image in radius, \
                size=(N,)
        """
        if angle_rad is None:
            angle_rad = torch.zeros(
                size=(patches.shape[0],),
                device=patches.device,
                dtype=patches.dtype)
        elif isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad, device=patches.device, dtype=patches.dtype)

        angle_embeds = self.angle_embedder(angle_rad)
        patches = torch.concat([angle_embeds[:, None], patches], dim=1)

        return self.cr(patches)[:, 1:]

    def do_hr(self,
        patches: torch.Tensor, angle_rad: Union[torch.Tensor, List, None], *args, **kwargs
    ) -> torch.Tensor:
        _, _ = args, kwargs
        """
        Args:
            patches (torch.Tensor): As `do_hf`.
            angle_rad (Union[torch.Tensor, List, None]): Angles rotated for each image in radius, \
                size=(N,)
        """
        if angle_rad is None:
            angle_rad = torch.zeros(
                size=(patches.shape[0],),
                device=patches.device,
                dtype=patches.dtype)
        elif isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad, device=patches.device, dtype=patches.device)

        angle_embeds = self.angle_embedder(angle_rad)
        patches = torch.concat([angle_embeds[:, None], patches], dim=1)

        return self.hr(patches)[:, 1:]


    def _unwrap_partial(self, op: partial) -> Tuple[Callable, torch.Tensor]:
        if op.func != self.do_hf:
            return op.func, op.keywords['angle_rad']
        else:
            return op.func, None

    def _mix_param(self,
        factor1: int, param1: Optional[torch.Tensor],
        factor2: int, param2: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if param1 is not None and param2 is not None:
            return factor1 * param1 + factor2 * param2
        if param1 is None and param2 is None:
            return None
        if param1 is not None:
            return factor1 * param1
        return factor2 * param2

    def compose(self,
        first_op: partial,
        second_op: partial,
    ) -> partial:
        """Compose the latent transformation to produce new one

        Args:
            first_op/second_op (partial): one of `do_{hf,cf,hr}`.

        Returns:
            partial: one of `do_{hf,cf,hr}`, with argument being partialed.
        """
        op1, param1 = self._unwrap_partial(first_op)
        op2, param2 = self._unwrap_partial(second_op)

        result_op, factor1, factor2 = self.composition_law[(op1, op2)]
        result_param = self._mix_param(factor1, param1, factor2, param2)

        return partial(result_op, angle_rad=result_param)

    def get_parameterized_hf(self) -> partial:
        return partial(self.do_hf, angle_rad=None)

    def get_parameterized_cr(self, angle_rad: Union[torch.Tensor, List]) -> partial:
        if isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad)
        return partial(self.do_cr, angle_rad=angle_rad)

    def get_parameterized_hr(self, angle_rad: Union[torch.Tensor, List]) -> partial:
        if isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad)
        return partial(self.do_hr, angle_rad=angle_rad)