"""Vanilla SASRec with explicit attention projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import SASRecBlock

if False:  # pragma: no cover - imported only for static type checkers
    from .embedding_adapter import TargetEmbeddingAdapter


@dataclass(frozen=True)
class SASRecConfig:
    num_items: int
    hidden_size: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    maxlen: int = 50

    def __post_init__(self) -> None:
        if self.num_items < 1 or self.hidden_size < 1 or self.num_blocks < 1:
            raise ValueError("num_items, hidden_size, and num_blocks must be positive")
        if self.num_heads < 1 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.maxlen < 1:
            raise ValueError("maxlen must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class SASRec(nn.Module):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.hidden_size,
            padding_idx=0,
            sparse=True,
        )
        self.item_embedding_adapter: TargetEmbeddingAdapter | None = None
        self.position_embedding = nn.Embedding(
            config.maxlen + 1, config.hidden_size, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        if item_ids.ndim != 2:
            raise ValueError("item_ids must have shape [batch, length]")
        if item_ids.shape[1] > self.config.maxlen:
            raise ValueError("sequence exceeds configured maxlen")
        valid = item_ids.ne(0)
        positions = valid.long().cumsum(dim=1) * valid.long()
        item_vectors = self.item_embedding(item_ids)
        if self.item_embedding_adapter is not None:
            item_vectors = item_vectors + self.item_embedding_adapter(item_ids)
        outputs = item_vectors * (self.config.hidden_size**0.5)
        outputs = outputs + self.position_embedding(positions)
        outputs = self.embedding_dropout(outputs)
        outputs = outputs.masked_fill(~valid.unsqueeze(-1), 0.0)
        for block in self.blocks:
            outputs = block(outputs, valid)
        outputs = self.final_norm(outputs)
        return outputs.masked_fill(~valid.unsqueeze(-1), 0.0)

    def final_state(self, item_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(item_ids)
        lengths = item_ids.ne(0).sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("every context must contain at least one non-padding item")
        indices = item_ids.shape[1] - torch.ones_like(lengths)
        return encoded[torch.arange(encoded.shape[0], device=encoded.device), indices]

    def score(self, contexts: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        states = self.final_state(contexts)
        candidates = self.item_embedding(candidate_ids)
        if self.item_embedding_adapter is not None:
            candidates = candidates + self.item_embedding_adapter(candidate_ids)
        if candidates.ndim == 2:
            return states @ candidates.transpose(0, 1)
        if candidates.ndim == 3:
            return torch.einsum("bd,bcd->bc", states, candidates)
        raise ValueError("candidate_ids must have shape [candidates] or [batch, candidates]")

    def scoring_weight(self) -> torch.Tensor:
        if self.item_embedding_adapter is None:
            return self.item_embedding.weight
        item_ids = torch.arange(
            self.config.num_items + 1, device=self.item_embedding.weight.device
        )
        return self.item_embedding.weight + self.item_embedding_adapter(item_ids)

    def optimizer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        sparse_names = {"item_embedding.weight"}
        if self.item_embedding_adapter is not None:
            sparse_names.add("item_embedding_adapter.delta.weight")
        return {
            "sparse": [
                parameter
                for name, parameter in self.named_parameters()
                if name in sparse_names
            ],
            "dense": [
                parameter
                for name, parameter in self.named_parameters()
                if name not in sparse_names
            ],
        }

    def logging_parameter_groups(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, tuple[str, ...]] = {
            "item_embedding": ("item_embedding.weight",),
            "position_embedding": ("position_embedding.weight",),
        }
        for index in range(len(self.blocks)):
            # LoRA changes the projection weight matrices, not their biases.
            query = f"blocks.{index}.attention.q_proj.weight"
            value = f"blocks.{index}.attention.v_proj.weight"
            groups[f"block_{index}_attention"] = (
                f"blocks.{index}.attention_norm",
                f"blocks.{index}.attention",
            )
            groups[f"block_{index}_ffn"] = (
                f"blocks.{index}.ffn_norm",
                f"blocks.{index}.ffn",
            )
            groups[f"block_{index}_q"] = (query,)
            groups[f"block_{index}_v"] = (value,)
            groups[f"block_{index}_qv"] = (query, value)
        return groups
