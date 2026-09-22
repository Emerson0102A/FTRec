"""Explicit-projection causal attention used by SASRec."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, self.num_heads, self.head_size).transpose(1, 2)

    def forward(
        self, inputs: torch.Tensor, valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        if inputs.ndim != 3 or valid_tokens.shape != inputs.shape[:2]:
            raise ValueError("attention expects [batch, length, hidden] and [batch, length]")
        _, length, _ = inputs.shape
        queries = self._split_heads(self.q_proj(inputs))
        keys = self._split_heads(self.k_proj(inputs))
        values = self._split_heads(self.v_proj(inputs))
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=inputs.device
        ).tril()
        allowed = causal.view(1, 1, length, length) & valid_tokens.view(
            valid_tokens.shape[0], 1, 1, length
        )
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        merged = attended.transpose(1, 2).contiguous().view_as(inputs)
        output = self.out_proj(merged)
        return output.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.first = nn.Linear(hidden_size, hidden_size)
        self.second = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.second(self.dropout(F.relu(self.first(inputs)))))


class SASRecBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.attention = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_size, dropout)
        # Parameter-efficient adapters are injected only after a pretrained
        # checkpoint has been loaded. Keeping these slots empty preserves the
        # original checkpoint schema and model output exactly.
        self.attention_adapter: nn.Module | None = None
        self.ffn_adapter: nn.Module | None = None

    def forward(self, inputs: torch.Tensor, valid_tokens: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.attention_norm(inputs), valid_tokens)
        if self.attention_adapter is not None:
            attention_output = self.attention_adapter(attention_output)
        outputs = inputs + attention_output
        outputs = outputs.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        ffn_output = self.ffn(self.ffn_norm(outputs))
        if self.ffn_adapter is not None:
            ffn_output = self.ffn_adapter(ffn_output)
        outputs = outputs + ffn_output
        return outputs.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)


class PostNormSASRecBlock(nn.Module):
    """MyModel4-compatible post-norm SASRec block.

    The projections deliberately keep the same explicit module names as the
    existing block so later LoRA experiments can target Q/K/V/O and FFN
    layers without a second adapter implementation.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.attention = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_size, dropout)
        self.attention_adapter: nn.Module | None = None
        self.ffn_adapter: nn.Module | None = None

    def forward(self, inputs: torch.Tensor, valid_tokens: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(inputs, valid_tokens)
        if self.attention_adapter is not None:
            attention_output = self.attention_adapter(attention_output)
        outputs = self.attention_norm(inputs + attention_output)
        outputs = outputs.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        ffn_output = self.ffn(outputs)
        if self.ffn_adapter is not None:
            ffn_output = self.ffn_adapter(ffn_output)
        outputs = self.ffn_norm(outputs + ffn_output)
        return outputs.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
