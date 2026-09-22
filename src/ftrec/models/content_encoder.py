"""MyRec-compatible, content-only item encoders."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ftrec.attributes.artifacts import AttributeArtifact, load_attribute_artifact


ATTRIBUTE_POOLING_MODES = (
    "hard_top1",
    "mean_all",
    "soft_attention",
    "domain_title_attention",
)


def load_content_artifact(artifact_dir: str, num_items: int) -> AttributeArtifact:
    """Load the frozen item-aligned content banks used by both SASRec towers."""

    return load_attribute_artifact(
        artifact_dir,
        expected_item_count=num_items,
        mmap_mode="c",
    )


def load_item_domain_ids(
    path: str | Path, num_items: int
) -> tuple[torch.Tensor, int]:
    """Load an item-aligned domain vector, reserving zero for padding."""

    aligned, domain_values = _load_item_domain_index(path, num_items)
    return aligned, len(domain_values)


def _load_item_domain_index(
    path: str | Path, num_items: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return mapped item domains plus their ordered raw domain identifiers."""

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    raw_domains = np.full(num_items + 1, -1, dtype=np.int64)
    raw_domains[0] = 0
    with opener(source, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"item_id", "domain_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"item domain file must contain {sorted(required)} columns: {source}"
            )
        for row in reader:
            item_id = int(row["item_id"])
            domain_id = int(row["domain_id"])
            if not 1 <= item_id <= num_items:
                raise ValueError(
                    f"item_id {item_id} is outside 1..{num_items} in {source}"
                )
            if raw_domains[item_id] != -1:
                raise ValueError(f"duplicate item_id {item_id} in {source}")
            raw_domains[item_id] = domain_id
    missing = np.flatnonzero(raw_domains[1:] < 0) + 1
    if missing.size:
        preview = ", ".join(str(value) for value in missing[:5])
        raise ValueError(
            f"item domain file is missing {missing.size} items; first: {preview}"
        )
    domain_values = tuple(sorted(int(value) for value in np.unique(raw_domains[1:])))
    remapping = {value: index + 1 for index, value in enumerate(domain_values)}
    aligned = np.zeros(num_items + 1, dtype=np.int64)
    for raw, mapped in remapping.items():
        aligned[raw_domains == raw] = mapped
    return torch.from_numpy(aligned), domain_values


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
    """Build one Title+Attr token before the single SASRec tower.

    ``hard_top1`` is the original MyRec-compatible baseline. The other modes
    pool all valid attributes, with the most expressive mode borrowing
    MyModel4's domain-specific title-conditioned attention.
    """

    def __init__(
        self,
        artifact: AttributeArtifact,
        *,
        hidden_size: int,
        attribute_pooling: str = "hard_top1",
        item_domain_file: str | None = None,
        attribute_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if attribute_pooling not in ATTRIBUTE_POOLING_MODES:
            raise ValueError(
                "attribute_pooling must be one of "
                f"{ATTRIBUTE_POOLING_MODES}, got {attribute_pooling!r}"
            )
        if attribute_temperature <= 0:
            raise ValueError("attribute_temperature must be positive")
        self.title_encoder = TitleItemEncoder(artifact, hidden_size=hidden_size)
        self.attribute_pooling = attribute_pooling
        self.attribute_temperature = attribute_temperature
        self.attribute_encoder: AttributeItemEncoder | None = None
        self.attribute_adapter: nn.Sequential | None = None
        self.attribute_title_query: nn.Linear | None = None
        self.domain_title_queries: nn.ModuleList | None = None
        self.domain_queries: nn.Parameter | None = None
        self.num_domains = 0

        if attribute_pooling == "hard_top1":
            # Preserve the exact module layout and state-dict keys used by
            # existing fused checkpoints.
            self.attribute_encoder = AttributeItemEncoder(
                artifact, hidden_size=hidden_size
            )
        else:
            attributes = torch.from_numpy(
                np.asarray(artifact.attribute_embeddings, dtype=np.float16)
            )
            self.register_buffer("attribute_bank", attributes, persistent=False)
            source_dim = artifact.embedding_dim
            bottleneck = max(1, source_dim // 4)
            self.attribute_adapter = nn.Sequential(
                nn.Linear(source_dim, bottleneck),
                nn.ReLU(),
                nn.LayerNorm(bottleneck),
                nn.Linear(bottleneck, hidden_size),
            )
            if attribute_pooling == "soft_attention":
                self.attribute_title_query = nn.Linear(
                    hidden_size, hidden_size, bias=False
                )
            elif attribute_pooling == "domain_title_attention":
                if item_domain_file is None:
                    raise ValueError(
                        "domain_title_attention requires item_domain_file"
                    )
                domain_ids, self.num_domains = load_item_domain_ids(
                    item_domain_file, artifact.item_count
                )
                self.register_buffer("item_domain_ids", domain_ids, persistent=False)
                self.domain_queries = nn.Parameter(
                    torch.zeros(self.num_domains + 1, hidden_size)
                )
                self.domain_title_queries = nn.ModuleList(
                    nn.Linear(hidden_size, hidden_size, bias=False)
                    for _ in range(self.num_domains)
                )
        self.output_norm = nn.LayerNorm(hidden_size)

    def reset_pooling_parameters(self) -> None:
        """Start learned attention from valid-attribute mean pooling."""

        with torch.no_grad():
            if self.attribute_title_query is not None:
                self.attribute_title_query.weight.zero_()
            if self.domain_queries is not None:
                self.domain_queries.zero_()
            if self.domain_title_queries is not None:
                for projection in self.domain_title_queries:
                    projection.weight.zero_()

    @staticmethod
    def _masked_weights(
        logits: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.softmax(logits.masked_fill(~valid, -1e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def _attention_query(
        self, item_ids: torch.Tensor, title: torch.Tensor
    ) -> torch.Tensor:
        if self.attribute_pooling == "soft_attention":
            assert self.attribute_title_query is not None
            return self.attribute_title_query(title)
        assert self.attribute_pooling == "domain_title_attention"
        assert self.domain_queries is not None
        assert self.domain_title_queries is not None
        domains = self.item_domain_ids[item_ids]
        query = self.domain_queries[domains]
        title_conditioning = torch.zeros_like(title)
        for domain_id, projection in enumerate(
            self.domain_title_queries, start=1
        ):
            mask = domains.eq(domain_id)
            if torch.any(mask):
                title_conditioning[mask] = projection(title[mask])
        return query + title_conditioning

    def pool_attribute_embeddings(
        self, item_ids: torch.Tensor, title: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled attributes and inspectable per-attribute weights."""

        if self.attribute_pooling == "hard_top1":
            raise RuntimeError("hard_top1 pooling is owned by AttributeItemEncoder")
        assert self.attribute_adapter is not None
        ids = item_ids.long()
        dtype = self.attribute_adapter[0].weight.dtype
        raw = self.attribute_bank[ids].to(dtype)
        valid = raw.abs().sum(dim=-1).gt(0)
        attributes = self.attribute_adapter(F.normalize(raw, p=2, dim=-1))
        if self.attribute_pooling == "mean_all":
            weights = valid.to(attributes.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            query = self._attention_query(ids, title)
            logits = torch.einsum("...ah,...h->...a", attributes, query)
            logits = logits / (
                attributes.shape[-1] ** 0.5 * self.attribute_temperature
            )
            weights = self._masked_weights(logits, valid)
        pooled = torch.einsum("...a,...ah->...h", weights, attributes)
        return pooled, weights

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        ids = item_ids.long()
        title = self.title_encoder(ids)
        if self.attribute_pooling == "hard_top1":
            assert self.attribute_encoder is not None
            attributes = self.attribute_encoder(ids)
        else:
            attributes, _ = self.pool_attribute_embeddings(ids, title)
        fused = self.output_norm(title + attributes)
        available = self.title_encoder.content_present[ids] & ids.ne(0)
        return fused * available.unsqueeze(-1).to(fused.dtype)


class _SharedPrivateProjection(nn.Module):
    """Shared semantic compression followed by one output map per domain."""

    def __init__(
        self,
        input_size: int,
        projection_size: int,
        hidden_size: int,
        num_domains: int,
        *,
        shared_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_size, projection_size),
            nn.ReLU(),
        ]
        if shared_norm:
            layers.append(nn.LayerNorm(projection_size))
        self.shared = nn.Sequential(*layers)
        self.private = nn.ModuleList(
            nn.Linear(projection_size, hidden_size) for _ in range(num_domains)
        )

    def forward(self, values: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        compressed = self.shared(values)
        output = compressed.new_zeros(*compressed.shape[:-1], self.private[0].out_features)
        for mapped_domain, projection in enumerate(self.private, start=1):
            mask = domain_ids.eq(mapped_domain)
            if torch.any(mask):
                output[mask] = projection(compressed[mask])
        return output


class SharedPrivateBehaviorItemEncoder(nn.Module):
    """The complete content-only Behavior item encoder from MyModel4.

    Raw title and attribute embeddings are frozen. Semantic compression is
    shared, while output projections, attribute attention, and final
    normalization are private to each item domain.
    """

    def __init__(
        self,
        artifact: AttributeArtifact,
        *,
        hidden_size: int,
        item_domain_file: str,
        projection_size: int = 0,
        domain_embedding_scale: float = 0.1,
        attribute_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if attribute_temperature <= 0:
            raise ValueError("attribute_temperature must be positive")
        if domain_embedding_scale < 0:
            raise ValueError("domain_embedding_scale must be non-negative")
        title = torch.from_numpy(np.asarray(artifact.title_embeddings, dtype=np.float16))
        attributes = torch.from_numpy(np.asarray(artifact.attribute_embeddings, dtype=np.float16))
        present = torch.from_numpy(np.asarray(artifact.present_mask, dtype=np.bool_))
        domain_ids, raw_domains = _load_item_domain_index(item_domain_file, artifact.item_count)
        self.register_buffer("title_bank", title, persistent=False)
        self.register_buffer("attribute_bank", attributes, persistent=False)
        self.register_buffer("content_present", present, persistent=False)
        self.register_buffer("item_domain_ids", domain_ids, persistent=False)
        self.raw_domains = raw_domains
        self.num_domains = len(raw_domains)
        self.hidden_size = hidden_size
        self.domain_embedding_scale = float(domain_embedding_scale)
        self.attribute_temperature = float(attribute_temperature)
        source_size = artifact.embedding_dim
        bottleneck = (
            int(projection_size) if projection_size > 0 else max(source_size // 4, hidden_size)
        )
        self.title_projection = _SharedPrivateProjection(
            source_size,
            bottleneck,
            hidden_size,
            self.num_domains,
            shared_norm=False,
        )
        self.attribute_projection = _SharedPrivateProjection(
            source_size,
            bottleneck,
            hidden_size,
            self.num_domains,
            shared_norm=True,
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(self.num_domains))
        self.domain_embedding = nn.Embedding(self.num_domains + 1, hidden_size, padding_idx=0)
        self.attribute_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden_size)) for _ in range(self.num_domains)
        )
        self.attribute_title_queries = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(self.num_domains)
        )

    def reset_attribute_conditioning(self) -> None:
        """Begin from uniform pooling while preserving trainable attention."""

        with torch.no_grad():
            for query in self.attribute_queries:
                query.zero_()
            for projection in self.attribute_title_queries:
                projection.weight.zero_()

    def get_domain_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_domain_ids[item_ids.long()]

    def raw_domain_for_mapped(self, mapped_domain: int) -> int:
        if not 1 <= mapped_domain <= self.num_domains:
            raise ValueError(f"mapped domain must be in 1..{self.num_domains}")
        return self.raw_domains[mapped_domain - 1]

    def components(self, item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = item_ids.long()
        dtype = self.title_projection.shared[0].weight.dtype
        raw_title = self.title_bank[ids].to(dtype)
        raw_attributes = self.attribute_bank[ids].to(dtype)
        attribute_valid = raw_attributes.abs().sum(dim=-1).gt(0)
        domains = self.get_domain_ids(ids)
        title = self.title_projection(F.normalize(raw_title, p=2, dim=-1), domains)
        attribute_domains = domains.unsqueeze(-1).expand_as(attribute_valid)
        attributes = self.attribute_projection(
            F.normalize(raw_attributes, p=2, dim=-1), attribute_domains
        )
        attributes = attributes * attribute_valid.unsqueeze(-1).to(attributes.dtype)
        return title, attributes, attribute_valid

    def pool_attributes(
        self,
        title: torch.Tensor,
        attributes: torch.Tensor,
        valid: torch.Tensor,
        domains: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = torch.zeros_like(title)
        for mapped_domain in range(1, self.num_domains + 1):
            mask = domains.eq(mapped_domain)
            if torch.any(mask):
                query[mask] = self.attribute_queries[
                    mapped_domain - 1
                ] + self.attribute_title_queries[mapped_domain - 1](title[mask])
        logits = torch.einsum("...ah,...h->...a", attributes, query)
        logits = logits / (self.hidden_size**0.5 * self.attribute_temperature)
        weights = FusedContentItemEncoder._masked_weights(logits, valid)
        pooled = torch.einsum("...a,...ah->...h", weights, attributes)
        return pooled, weights

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        ids = item_ids.long()
        domains = self.get_domain_ids(ids)
        title, attributes, valid = self.components(ids)
        pooled, _ = self.pool_attributes(title, attributes, valid, domains)
        values = title + pooled + self.domain_embedding_scale * self.domain_embedding(domains)
        result = torch.zeros_like(values)
        for mapped_domain, norm in enumerate(self.norms, start=1):
            mask = domains.eq(mapped_domain)
            if torch.any(mask):
                result[mask] = norm(values[mask])
        available = self.content_present[ids] & ids.ne(0)
        return result * available.unsqueeze(-1).to(result.dtype)

    def private_parameters(self, mapped_domain: int) -> tuple[nn.Parameter, ...]:
        index = mapped_domain - 1
        modules: tuple[nn.Module, ...] = (
            self.title_projection.private[index],
            self.attribute_projection.private[index],
            self.norms[index],
            self.attribute_title_queries[index],
        )
        parameters = [parameter for module in modules for parameter in module.parameters()]
        parameters.append(self.attribute_queries[index])
        return tuple(parameters)
