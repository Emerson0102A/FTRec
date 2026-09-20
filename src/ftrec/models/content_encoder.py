"""MyRec-compatible, content-only item encoders."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ftrec.attributes.artifacts import AttributeArtifact, load_attribute_artifact


def load_content_artifact(artifact_dir: str, num_items: int) -> AttributeArtifact:
    """Load the frozen item-aligned content banks used by both SASRec towers."""

    return load_attribute_artifact(
        artifact_dir,
        expected_item_count=num_items,
        mmap_mode="c",
    )


class TitleItemEncoder(nn.Module):
    """MyRec ``TitleItemEmb`` with an explicit padding/missing-content mask."""

    def __init__(
        self,
        artifact: AttributeArtifact,
        *,
        hidden_size: int,
    ) -> None:
        super().__init__()
        title = torch.from_numpy(
            np.asarray(artifact.title_embeddings, dtype=np.float16)
        )
        present = torch.from_numpy(
            np.asarray(artifact.present_mask, dtype=np.bool_)
        )
        self.register_buffer("title_bank", title, persistent=False)
        self.register_buffer("content_present", present, persistent=False)
        source_dim = artifact.embedding_dim
        bottleneck = max(1, source_dim // 4)
        self.adapter = nn.Sequential(
            nn.Linear(source_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_size),
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        ids = item_ids.long()
        dtype = self.adapter[0].weight.dtype
        title = F.normalize(self.title_bank[ids].to(dtype), p=2, dim=-1)
        encoded = self.adapter(title)
        available = self.content_present[ids] & ids.ne(0)
        return encoded * available.unsqueeze(-1).to(encoded.dtype)


class AttributeItemEncoder(nn.Module):
    """MyRec ``AttrItemEmb`` with hard attribute selection."""

    def __init__(
        self,
        artifact: AttributeArtifact,
        *,
        hidden_size: int,
    ) -> None:
        super().__init__()
        attributes = torch.from_numpy(
            np.asarray(artifact.attribute_embeddings, dtype=np.float16)
        )
        present = torch.from_numpy(
            np.asarray(artifact.present_mask, dtype=np.bool_)
        )
        self.register_buffer("attribute_bank", attributes, persistent=False)
        self.register_buffer("content_present", present, persistent=False)
        source_dim = artifact.embedding_dim
        bottleneck = max(1, source_dim // 4)
        self.node_query = nn.Linear(source_dim, source_dim)
        self.node_key = nn.Linear(source_dim, source_dim)
        self.adapter = nn.Sequential(
            nn.Linear(source_dim, bottleneck),
            nn.ReLU(),
            nn.LayerNorm(bottleneck),
            nn.Linear(bottleneck, hidden_size),
        )
        self.source_dim = source_dim

    def _select_attribute(self, attributes: torch.Tensor) -> torch.Tensor:
        valid = attributes.abs().sum(dim=-1).gt(0)
        query = self.node_query(attributes)
        key = self.node_key(attributes)
        attention = torch.matmul(query, key.transpose(-2, -1)) / (
            self.source_dim**0.5
        )
        logits = attention.mean(dim=-1).masked_fill(~valid, -1e4)
        if self.training:
            weights = F.gumbel_softmax(logits, dim=-1, tau=1.0, hard=True)
        else:
            # MyRec uses hard Gumbel selection.  Argmax preserves that hard
            # choice at evaluation time without making ranks depend on RNG or
            # selecting the target differently in separate scoring calls.
            selected = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits).scatter_(-1, selected, 1.0)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("...k,...kd->...d", weights, attributes)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        ids = item_ids.long()
        dtype = self.node_query.weight.dtype
        attributes = F.normalize(
            self.attribute_bank[ids].to(dtype), p=2, dim=-1
        )
        encoded = self.adapter(self._select_attribute(attributes))
        available = self.content_present[ids] & ids.ne(0)
        return encoded * available.unsqueeze(-1).to(encoded.dtype)


class FusedContentItemEncoder(nn.Module):
    """MyRec-style Title+Attr item embedding used by the single-tower ablation."""

    def __init__(
        self,
        artifact: AttributeArtifact,
        *,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.title_encoder = TitleItemEncoder(artifact, hidden_size=hidden_size)
        self.attribute_encoder = AttributeItemEncoder(
            artifact, hidden_size=hidden_size
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        ids = item_ids.long()
        fused = self.output_norm(
            self.title_encoder(ids) + self.attribute_encoder(ids)
        )
        available = self.title_encoder.content_present[ids] & ids.ne(0)
        return fused * available.unsqueeze(-1).to(fused.dtype)
