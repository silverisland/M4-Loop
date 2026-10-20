"""Local patch, shared-depth Transformer, RevIN, and prediction-head layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RevINStats:
    """Statistics belonging to one input window, never stored on the module."""

    mean: Tensor
    std: Tensor


class RevIN(nn.Module):
    """Non-affine instance normalization over the final (time) dimension."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("RevIN eps must be positive")
        self.eps = eps

    def normalize(self, x: Tensor) -> tuple[Tensor, RevINStats]:
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
        stats = RevINStats(mean=mean, std=std)
        return (x - stats.mean) / stats.std, stats

    def denormalize(self, x: Tensor, stats: RevINStats) -> Tensor:
        return x * stats.std + stats.mean


class PreNormTransformerBlock(nn.Module):
    """Standard x + Attention(LN(x)), followed by x + FFN(LN(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm_attn(x)
        attended, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        x = x + self.dropout_attn(attended)
        return x + self.dropout_ffn(self.ff(self.norm_ffn(x)))


class LoopedTransformerEncoder(nn.Module):
    """Execute the same ordered block stack a fixed number of times.

    Blocks are registered once. Every iteration participates in autograd, and
    training and evaluation use the same loop count. Attention scores are not
    carried between layers or loops.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        block_layers: int = 2,
        loop_steps: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if block_layers < 1 or loop_steps < 1:
            raise ValueError("block_layers and loop_steps must be positive")
        self.loop_steps = loop_steps
        self.blocks = nn.ModuleList(
            [
                PreNormTransformerBlock(d_model, n_heads, d_ff, dropout, attn_dropout, activation)
                for _ in range(block_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.loop_steps):
            for block in self.blocks:
                x = block(x)
        return self.final_norm(x)


class FlattenPredictionHead(nn.Module):
    """Flatten [B,C,D,N] and return prediction plus pre-normalization embedding."""

    def __init__(
        self,
        nf: int,
        target_window: int,
        embedding_dim: int = 256,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.embedding_linear = nn.Linear(nf, embedding_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(head_dropout)
        self.linear2 = nn.Linear(embedding_dim, 256)
        self.norm2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(head_dropout)
        self.linear3 = nn.Linear(256, head_hidden_dim)
        self.norm3 = nn.LayerNorm(head_hidden_dim)
        self.dropout3 = nn.Dropout(head_dropout)
        self.pred = nn.Linear(head_hidden_dim, target_window)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.embedding_linear(self.flatten(x))
        x = self.embedding_dropout(self.relu(self.embedding_norm(embedding)))
        x = self.dropout2(self.relu(self.norm2(self.linear2(x))))
        x = self.dropout3(self.relu(self.norm3(self.linear3(x))))
        return self.pred(x), embedding
