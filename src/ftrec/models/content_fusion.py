"""Provider-neutral fusion of frozen LLM content with trainable item IDs."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ftrec.attributes.artifacts import load_attribute_artifact


class ContentFusion(nn.Module):
    """Fuse ID, title, and attribute representations with a learned residual gate.

    Attribute banks are frozen, excluded from checkpoints, and supplied by either
    extractor through the same validated artifact contract.  Only this small
    projection/gating module is learned by the recommender.
    """

    def __init__(
        self,
        *,
        num_items: int,
        hidden_size: int,
        artifact_dir: str,
        dropout: float,
        use_title: bool,
    ) -> None:
        super().__init__()
        artifact = load_attribute_artifact(
            artifact_dir, expected_item_count=num_items, mmap_mode="c"
        )
        self.provider = artifact.provider
        self.use_title = use_title
        source_dim = artifact.embedding_dim

        # mmap_mode="c" produces writable copy-on-write arrays, which avoids
        # PyTorch's read-only NumPy warning without duplicating the whole bank.
        title = torch.from_numpy(
            np.asarray(artifact.title_embeddings, dtype=np.float16)
        )
        attributes = torch.from_numpy(
            np.asarray(artifact.attribute_embeddings, dtype=np.float16)
        )
        present = torch.from_numpy(
            np.asarray(artifact.present_mask, dtype=np.bool_)
        )
        self.register_buffer("title_bank", title, persistent=False)
        self.register_buffer("attribute_bank", attributes, persistent=False)
        self.register_buffer("content_present", present, persistent=False)

        bottleneck = max(hidden_size, min(256, max(1, source_dim // 4)))
        self.title_adapter = nn.Sequential(
            nn.Linear(source_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_size),
        )
        self.attribute_key = nn.Linear(source_dim, hidden_size)
        self.attribute_query = nn.Parameter(torch.empty(hidden_size))
        self.attribute_adapter = nn.Sequential(
            nn.Linear(source_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_size),
        )
        self.content_norm = nn.LayerNorm(hidden_size)
        self.content_gate = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.normal_(self.attribute_query, std=hidden_size**-0.5)

    def forward(
        self, item_ids: torch.Tensor, id_representations: torch.Tensor
    ) -> torch.Tensor:
        ids = item_ids.long()
        projection_dtype = self.title_adapter[0].weight.dtype
        title = F.normalize(
            self.title_bank[ids].to(projection_dtype), p=2, dim=-1
        )
        attributes = F.normalize(
            self.attribute_bank[ids].to(projection_dtype), p=2, dim=-1
        )

        title_representation = (
            self.title_adapter(title)
            if self.use_title
            else torch.zeros_like(id_representations)
        )
        attribute_keys = self.attribute_key(attributes)
        attribute_valid = attributes.abs().sum(dim=-1).gt(0)
        logits = torch.einsum(
            "...kh,h->...k", attribute_keys, self.attribute_query
        )
        logits = logits.masked_fill(~attribute_valid, -1e4)
        weights = torch.softmax(logits, dim=-1) * attribute_valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled_attributes = torch.einsum(
            "...k,...kd->...d", weights, attributes
        )
        content = self.content_norm(
            title_representation + self.attribute_adapter(pooled_attributes)
        )
        gate = torch.sigmoid(
            self.content_gate(torch.cat((id_representations, content), dim=-1))
        )
        available = self.content_present[ids].unsqueeze(-1).to(gate.dtype)
        fused = id_representations + available * gate * content
        return fused * ids.ne(0).unsqueeze(-1).to(fused.dtype)


def content_fusion_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("content_fusion.")
    )


def unfreeze_content_fusion(model: nn.Module) -> tuple[str, ...]:
    fusion = getattr(model, "content_fusion", None)
    if fusion is None:
        raise ValueError(
            "content-aware adaptation requires model.attribute_artifact"
        )
    for parameter in fusion.parameters():
        parameter.requires_grad_(True)
    return content_fusion_parameter_names(model)
