"""Target-domain residual item embeddings for adaptation diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .sasrec import SASRec


class TargetEmbeddingAdapter(nn.Module):
    """A zero-initialized residual table containing only target-domain items."""

    def __init__(
        self,
        *,
        num_items: int,
        hidden_size: int,
        target_item_ids: Iterable[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        item_ids = tuple(sorted({int(item_id) for item_id in target_item_ids}))
        if not item_ids:
            raise ValueError("target-domain item ids must not be empty")
        if item_ids[0] < 1 or item_ids[-1] > num_items:
            raise ValueError("target-domain item id is outside the model catalog")

        index_map = torch.zeros(num_items + 1, dtype=torch.long, device=device)
        index_map[torch.tensor(item_ids, dtype=torch.long, device=device)] = torch.arange(
            1, len(item_ids) + 1, dtype=torch.long, device=device
        )
        self.register_buffer("index_map", index_map, persistent=False)
        self.delta = nn.Embedding(
            len(item_ids) + 1,
            hidden_size,
            padding_idx=0,
            sparse=True,
            device=device,
            dtype=dtype,
        )
        nn.init.zeros_(self.delta.weight)

    @property
    def num_target_items(self) -> int:
        return self.delta.num_embeddings - 1

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.delta(self.index_map[item_ids])


def inject_target_embedding_adapter(
    model: SASRec,
    target_item_ids: Iterable[int],
    *,
    freeze_existing: bool,
) -> TargetEmbeddingAdapter:
    if model.item_embedding_adapter is not None:
        raise ValueError("target embedding adapter has already been injected")
    if freeze_existing:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    adapter = TargetEmbeddingAdapter(
        num_items=model.config.num_items,
        hidden_size=model.config.hidden_size,
        target_item_ids=target_item_ids,
        device=model.item_embedding.weight.device,
        dtype=model.item_embedding.weight.dtype,
    )
    model.item_embedding_adapter = adapter
    adapter.delta.weight.requires_grad_(True)
    return adapter


def target_embedding_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("item_embedding_adapter.")
    )
