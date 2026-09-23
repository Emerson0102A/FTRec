"""SASRec with ID, MyRec dual-tower, and content-fused item encoders."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .attention import PostNormSASRecBlock, SASRecBlock
from .content_encoder import (
    ATTRIBUTE_POOLING_MODES,
    AttributeItemEncoder,
    FusedContentItemEncoder,
    SharedPrivateBehaviorItemEncoder,
    TitleItemEncoder,
    load_content_artifact,
)
from .content_ablation import ablate_content_artifact

if False:  # pragma: no cover - imported only for static type checkers
    from .embedding_adapter import TargetEmbeddingAdapter


ITEM_EMBEDDING_MODES = (
    "id",
    "content_dual",
    "content_fused",
    "mymodel4_behavior",
)


@dataclass(frozen=True)
class SASRecConfig:
    num_items: int
    hidden_size: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    maxlen: int = 50
    item_embedding_mode: str = "id"
    attribute_artifact: str | None = None
    attribute_pooling: str = "hard_top1"
    item_domain_file: str | None = None
    attribute_temperature: float = 1.0
    shared_behavior_blocks: int = 1
    content_projection_size: int = 0
    domain_embedding_scale: float = 0.1
    content_ablation: str = "none"
    content_ablation_seed: int = 42
    content_ablation_domain_file: str | None = None

    def __post_init__(self) -> None:
        if self.num_items < 1 or self.hidden_size < 1 or self.num_blocks < 1:
            raise ValueError("num_items, hidden_size, and num_blocks must be positive")
        if self.num_heads < 1 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.maxlen < 1:
            raise ValueError("maxlen must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.item_embedding_mode not in ITEM_EMBEDDING_MODES:
            raise ValueError(
                "item_embedding_mode must be one of "
                f"{ITEM_EMBEDDING_MODES}, got {self.item_embedding_mode!r}"
            )
        if self.content_ablation not in {"none", "random", "shuffled", "attribute_only"}:
            raise ValueError("unsupported content_ablation")
        if self.content_ablation != "none" and self.item_embedding_mode != "content_fused":
            raise ValueError("content_ablation requires content_fused")
        if self.content_ablation == "shuffled" and self.content_ablation_domain_file is None:
            raise ValueError("shuffled content requires content_ablation_domain_file")
        if self.content_ablation != "shuffled" and self.content_ablation_domain_file is not None:
            raise ValueError("content_ablation_domain_file is only used by shuffled content")
        if self.item_embedding_mode == "id" and self.attribute_artifact is not None:
            raise ValueError(
                "attribute_artifact cannot be combined with ID embeddings; "
                "choose content_dual or content_fused"
            )
        if self.item_embedding_mode != "id" and self.attribute_artifact is None:
            raise ValueError(
                f"{self.item_embedding_mode} requires attribute_artifact"
            )
        if self.attribute_pooling not in ATTRIBUTE_POOLING_MODES:
            raise ValueError(
                "attribute_pooling must be one of "
                f"{ATTRIBUTE_POOLING_MODES}, got {self.attribute_pooling!r}"
            )
        if self.attribute_temperature <= 0:
            raise ValueError("attribute_temperature must be positive")
        if self.content_projection_size < 0:
            raise ValueError("content_projection_size must be non-negative")
        if self.domain_embedding_scale < 0:
            raise ValueError("domain_embedding_scale must be non-negative")
        if (
            self.attribute_pooling != "hard_top1"
            and self.item_embedding_mode != "content_fused"
        ):
            raise ValueError(
                "non-hard attribute pooling is currently defined only for "
                "content_fused"
            )
        needs_domains = (
            self.attribute_pooling == "domain_title_attention"
            or self.item_embedding_mode == "mymodel4_behavior"
        )
        if needs_domains and self.item_domain_file is None:
            raise ValueError(
                "domain-conditioned content models require item_domain_file"
            )
        if not needs_domains and self.item_domain_file is not None:
            raise ValueError(
                "item_domain_file is only used by domain-conditioned content models"
            )
        if self.item_embedding_mode == "mymodel4_behavior" and not (
            0 <= self.shared_behavior_blocks < self.num_blocks
        ):
            raise ValueError(
                "mymodel4_behavior requires shared_behavior_blocks in "
                "[0, num_blocks)"
            )


def model_config_dict(config: SASRecConfig) -> dict[str, object]:
    """Serialize configs without changing hashes of legacy checkpoints."""

    values = asdict(config)
    if (
        config.attribute_pooling == "hard_top1"
        and config.item_domain_file is None
        and config.attribute_temperature == 1.0
    ):
        values.pop("attribute_pooling")
        values.pop("item_domain_file")
        values.pop("attribute_temperature")
    if config.item_embedding_mode != "mymodel4_behavior":
        values.pop("shared_behavior_blocks")
        values.pop("content_projection_size")
        values.pop("domain_embedding_scale")
    else:
        # MyModel4 always uses its own domain/title-conditioned soft attention;
        # the legacy fused-encoder pooling switch is not part of this model.
        values.pop("attribute_pooling")
    if config.content_ablation == "none":
        values.pop("content_ablation")
        values.pop("content_ablation_seed")
        values.pop("content_ablation_domain_file")
    elif config.content_ablation != "shuffled":
        values.pop("content_ablation_domain_file")
    return values


class _ContentSASRecTower(nn.Module):
    """A complete SASRec tower driven only by a content item encoder."""

    def __init__(self, config: SASRecConfig, item_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.item_encoder = item_encoder
        self._evaluation_item_cache: torch.Tensor | None = None
        self.position_embedding = nn.Embedding(
            config.maxlen + 1, config.hidden_size, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=1e-8)

    def prepare_evaluation_cache(self, *, chunk_size: int = 4096) -> None:
        """Encode the catalog once for deterministic evaluation scoring."""

        if self.training:
            raise RuntimeError("evaluation cache requires eval mode")
        if self._evaluation_item_cache is not None:
            return
        if chunk_size < 1:
            raise ValueError("evaluation cache chunk_size must be positive")
        device = self.position_embedding.weight.device
        cache: torch.Tensor | None = None
        with torch.no_grad():
            for start in range(0, self.config.num_items + 1, chunk_size):
                stop = min(start + chunk_size, self.config.num_items + 1)
                item_ids = torch.arange(start, stop, device=device)
                encoded = self.item_encoder(item_ids)
                if cache is None:
                    cache = torch.empty(
                        (self.config.num_items + 1, encoded.shape[-1]),
                        dtype=encoded.dtype,
                        device=encoded.device,
                    )
                cache[start:stop].copy_(encoded)
        if cache is None:  # pragma: no cover - num_items is validated positive
            raise RuntimeError("failed to construct evaluation item cache")
        self._evaluation_item_cache = cache

    def clear_evaluation_cache(self) -> None:
        self._evaluation_item_cache = None

    def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        if item_ids.ndim != 2:
            raise ValueError("item_ids must have shape [batch, length]")
        if item_ids.shape[1] > self.config.maxlen:
            raise ValueError("sequence exceeds configured maxlen")
        valid = item_ids.ne(0)
        positions = valid.long().cumsum(dim=1) * valid.long()
        outputs = self.item_encoder(item_ids) * (self.config.hidden_size**0.5)
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
        # FTRec sequences are left padded, so every non-empty sequence ends at
        # the final column regardless of its unpadded length.
        indices = item_ids.shape[1] - torch.ones_like(lengths)
        return encoded[
            torch.arange(encoded.shape[0], device=encoded.device), indices
        ]

    def score_prepared(
        self, states: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        candidates = (
            self._evaluation_item_cache[candidate_ids]
            if self._evaluation_item_cache is not None
            else self.item_encoder(candidate_ids)
        )
        if candidates.ndim == 2:
            return states @ candidates.transpose(0, 1)
        if candidates.ndim == 3:
            return torch.einsum("bd,bcd->bc", states, candidates)
        raise ValueError(
            "candidate_ids must have shape [candidates] or [batch, candidates]"
        )


class _SharedPrivateContentSASRecTower(nn.Module):
    """MyModel4 Behavior backbone with target-domain private upper blocks."""

    def __init__(
        self, config: SASRecConfig, item_encoder: SharedPrivateBehaviorItemEncoder
    ) -> None:
        super().__init__()
        self.config = config
        self.item_encoder = item_encoder
        self._evaluation_item_cache: torch.Tensor | None = None
        self.position_embedding = nn.Embedding(config.maxlen + 1, config.hidden_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.shared_blocks = nn.ModuleList(
            PostNormSASRecBlock(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.shared_behavior_blocks)
        )
        private_count = config.num_blocks - config.shared_behavior_blocks
        self.private_blocks = nn.ModuleList(
            nn.ModuleList(
                PostNormSASRecBlock(config.hidden_size, config.num_heads, config.dropout)
                for _ in range(private_count)
            )
            for _ in range(item_encoder.num_domains)
        )
        self.final_norms = nn.ModuleList(
            nn.LayerNorm(config.hidden_size, eps=1e-8) for _ in range(item_encoder.num_domains)
        )

    def prepare_evaluation_cache(self, *, chunk_size: int = 4096) -> None:
        if self.training:
            raise RuntimeError("evaluation cache requires eval mode")
        if self._evaluation_item_cache is not None:
            return
        if chunk_size < 1:
            raise ValueError("evaluation cache chunk_size must be positive")
        device = self.position_embedding.weight.device
        cache: torch.Tensor | None = None
        with torch.no_grad():
            for start in range(0, self.config.num_items + 1, chunk_size):
                stop = min(start + chunk_size, self.config.num_items + 1)
                item_ids = torch.arange(start, stop, device=device)
                encoded = self.item_encoder(item_ids)
                if cache is None:
                    cache = torch.empty(
                        (self.config.num_items + 1, encoded.shape[-1]),
                        dtype=encoded.dtype,
                        device=encoded.device,
                    )
                cache[start:stop].copy_(encoded)
        if cache is None:  # pragma: no cover - num_items is positive
            raise RuntimeError("failed to construct evaluation item cache")
        self._evaluation_item_cache = cache

    def clear_evaluation_cache(self) -> None:
        self._evaluation_item_cache = None

    def target_domains(self, candidate_ids: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        domains = self.item_encoder.get_domain_ids(candidate_ids)
        if candidate_ids.ndim == 1:
            present = torch.unique(domains[domains.ne(0)])
            if present.numel() != 1:
                raise ValueError("shared candidate vector must belong to one domain")
            return present.expand(batch_size)
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != batch_size:
            raise ValueError("candidate_ids must have shape [candidates] or [batch, candidates]")
        target = domains.max(dim=1).values
        if torch.any(target.eq(0)):
            raise ValueError("each candidate row must contain a non-padding item")
        compatible = domains.eq(0) | domains.eq(target.unsqueeze(1))
        if not torch.all(compatible):
            raise ValueError("each candidate row must contain one target domain")
        return target

    def encode(self, item_ids: torch.Tensor, target_domains: torch.Tensor) -> torch.Tensor:
        if item_ids.ndim != 2:
            raise ValueError("item_ids must have shape [batch, length]")
        if item_ids.shape[1] > self.config.maxlen:
            raise ValueError("sequence exceeds configured maxlen")
        if target_domains.shape != (item_ids.shape[0],):
            raise ValueError("target_domains must have shape [batch]")
        if torch.any(target_domains < 1) or torch.any(
            target_domains > self.item_encoder.num_domains
        ):
            raise ValueError("target domain is outside the content catalog")
        valid = item_ids.ne(0)
        positions = torch.arange(1, item_ids.shape[1] + 1, device=item_ids.device).unsqueeze(0)
        positions = positions.expand_as(item_ids) * valid.long()
        outputs = self.item_encoder(item_ids) * math.sqrt(self.config.hidden_size)
        outputs = self.embedding_dropout(outputs + self.position_embedding(positions))
        outputs = outputs.masked_fill(~valid.unsqueeze(-1), 0.0)
        for block in self.shared_blocks:
            outputs = block(outputs, valid)
        routed = torch.zeros_like(outputs)
        for mapped_domain in range(1, self.item_encoder.num_domains + 1):
            rows = target_domains.eq(mapped_domain)
            if not torch.any(rows):
                continue
            private = outputs[rows]
            for block in self.private_blocks[mapped_domain - 1]:
                private = block(private, valid[rows])
            routed[rows] = self.final_norms[mapped_domain - 1](private)
        return routed.masked_fill(~valid.unsqueeze(-1), 0.0)

    def final_state(self, item_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        target_domains = self.target_domains(candidate_ids, batch_size=item_ids.shape[0])
        encoded = self.encode(item_ids, target_domains)
        if torch.any(item_ids.ne(0).sum(dim=1).eq(0)):
            raise ValueError("every context must contain at least one non-padding item")
        return encoded[:, -1, :]

    def score_prepared(self, states: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        candidates = (
            self._evaluation_item_cache[candidate_ids]
            if self._evaluation_item_cache is not None
            else self.item_encoder(candidate_ids)
        )
        if candidates.ndim == 2:
            return states @ candidates.transpose(0, 1)
        if candidates.ndim == 3:
            return torch.einsum("bd,bcd->bc", states, candidates)
        raise ValueError("candidate_ids must have shape [candidates] or [batch, candidates]")

    def private_parameter_owners(self) -> dict[str, int]:
        owners_by_identity: dict[int, int] = {}
        for mapped_domain in range(1, self.item_encoder.num_domains + 1):
            raw_domain = self.item_encoder.raw_domain_for_mapped(mapped_domain)
            parameters = list(self.item_encoder.private_parameters(mapped_domain))
            parameters.extend(
                parameter
                for block in self.private_blocks[mapped_domain - 1]
                for parameter in block.parameters()
            )
            parameters.extend(self.final_norms[mapped_domain - 1].parameters())
            for parameter in parameters:
                owners_by_identity[id(parameter)] = raw_domain
        return {
            name: owners_by_identity[id(parameter)]
            for name, parameter in self.named_parameters()
            if id(parameter) in owners_by_identity
        }


class SASRec(nn.Module):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding_adapter: TargetEmbeddingAdapter | None = None
        self.item_embedding: nn.Embedding | None = None
        self.title_tower: _ContentSASRecTower | None = None
        self.attribute_tower: _ContentSASRecTower | None = None
        self.fused_tower: _ContentSASRecTower | None = None
        self.shared_private_tower: _SharedPrivateContentSASRecTower | None = None

        if config.item_embedding_mode == "id":
            # Keep these names unchanged for existing baseline checkpoints.
            self.item_embedding = nn.Embedding(
                config.num_items + 1,
                config.hidden_size,
                padding_idx=0,
                sparse=True,
            )
            self.position_embedding = nn.Embedding(
                config.maxlen + 1, config.hidden_size, padding_idx=0
            )
            self.embedding_dropout = nn.Dropout(config.dropout)
            self.blocks = nn.ModuleList(
                SASRecBlock(config.hidden_size, config.num_heads, config.dropout)
                for _ in range(config.num_blocks)
            )
            self.final_norm = nn.LayerNorm(config.hidden_size, eps=1e-8)
        else:
            assert config.attribute_artifact is not None
            artifact = load_content_artifact(
                config.attribute_artifact, config.num_items
            )
            if config.content_ablation != "none":
                artifact = ablate_content_artifact(
                    artifact,
                    control=config.content_ablation,
                    seed=config.content_ablation_seed,
                    domain_file=config.content_ablation_domain_file,
                )
            if config.item_embedding_mode == "content_dual":
                self.title_tower = _ContentSASRecTower(
                    config,
                    TitleItemEncoder(artifact, hidden_size=config.hidden_size),
                )
                self.attribute_tower = _ContentSASRecTower(
                    config,
                    AttributeItemEncoder(artifact, hidden_size=config.hidden_size),
                )
            elif config.item_embedding_mode == "content_fused":
                self.fused_tower = _ContentSASRecTower(
                    config,
                    FusedContentItemEncoder(
                        artifact,
                        hidden_size=config.hidden_size,
                        attribute_pooling=config.attribute_pooling,
                        item_domain_file=config.item_domain_file,
                        attribute_temperature=config.attribute_temperature,
                        attribute_only=config.content_ablation == "attribute_only",
                    ),
                )
            else:
                assert config.item_domain_file is not None
                self.shared_private_tower = _SharedPrivateContentSASRecTower(
                    config,
                    SharedPrivateBehaviorItemEncoder(
                        artifact,
                        hidden_size=config.hidden_size,
                        item_domain_file=config.item_domain_file,
                        projection_size=config.content_projection_size,
                        domain_embedding_scale=config.domain_embedding_scale,
                        attribute_temperature=config.attribute_temperature,
                    ),
                )
        self.reset_parameters()
        if self.fused_tower is not None:
            encoder = self.fused_tower.item_encoder
            if isinstance(encoder, FusedContentItemEncoder):
                encoder.reset_pooling_parameters()
        if self.shared_private_tower is not None:
            self.shared_private_tower.item_encoder.reset_attribute_conditioning()

    @property
    def is_content_model(self) -> bool:
        return self.config.item_embedding_mode != "id"

    def prepare_evaluation_cache(self, *, chunk_size: int = 4096) -> None:
        if self.config.item_embedding_mode == "content_dual":
            assert self.title_tower is not None and self.attribute_tower is not None
            self.title_tower.prepare_evaluation_cache(chunk_size=chunk_size)
            self.attribute_tower.prepare_evaluation_cache(chunk_size=chunk_size)
        elif self.config.item_embedding_mode == "content_fused":
            assert self.fused_tower is not None
            self.fused_tower.prepare_evaluation_cache(chunk_size=chunk_size)
        elif self.config.item_embedding_mode == "mymodel4_behavior":
            assert self.shared_private_tower is not None
            self.shared_private_tower.prepare_evaluation_cache(chunk_size=chunk_size)

    def clear_evaluation_cache(self) -> None:
        for tower in (
            self.title_tower,
            self.attribute_tower,
            self.fused_tower,
            self.shared_private_tower,
        ):
            if tower is not None:
                tower.clear_evaluation_cache()

    def train(self, mode: bool = True) -> SASRec:
        if mode:
            self.clear_evaluation_cache()
        return super().train(mode)

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

    def _require_id_mode(self) -> None:
        if self.config.item_embedding_mode != "id":
            raise RuntimeError(
                "this operation is only defined for the ID-embedding baseline"
            )

    def embed_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        self._require_id_mode()
        assert self.item_embedding is not None
        item_vectors = self.item_embedding(item_ids)
        if self.item_embedding_adapter is not None:
            item_vectors = item_vectors + self.item_embedding_adapter(item_ids)
        return item_vectors

    def encode(self, item_ids: torch.Tensor) -> torch.Tensor:
        self._require_id_mode()
        if item_ids.ndim != 2:
            raise ValueError("item_ids must have shape [batch, length]")
        if item_ids.shape[1] > self.config.maxlen:
            raise ValueError("sequence exceeds configured maxlen")
        valid = item_ids.ne(0)
        positions = valid.long().cumsum(dim=1) * valid.long()
        outputs = self.embed_items(item_ids) * (self.config.hidden_size**0.5)
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
        return encoded[
            torch.arange(encoded.shape[0], device=encoded.device), indices
        ]

    def prepare_scoring(
        self, contexts: torch.Tensor, candidate_ids: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.config.item_embedding_mode == "id":
            return self.final_state(contexts)
        if self.config.item_embedding_mode == "content_dual":
            assert self.title_tower is not None and self.attribute_tower is not None
            return (
                self.title_tower.final_state(contexts),
                self.attribute_tower.final_state(contexts),
            )
        if self.config.item_embedding_mode == "content_fused":
            assert self.fused_tower is not None
            return self.fused_tower.final_state(contexts)
        assert self.shared_private_tower is not None
        if candidate_ids is None:
            raise ValueError("mymodel4_behavior scoring requires candidate_ids")
        return self.shared_private_tower.final_state(contexts, candidate_ids)

    def score_prepared(
        self,
        prepared: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.item_embedding_mode == "id":
            if not isinstance(prepared, torch.Tensor):
                raise TypeError("ID scoring expects one prepared state tensor")
            candidates = self.embed_items(candidate_ids)
            if candidates.ndim == 2:
                return prepared @ candidates.transpose(0, 1)
            if candidates.ndim == 3:
                return torch.einsum("bd,bcd->bc", prepared, candidates)
            raise ValueError(
                "candidate_ids must have shape [candidates] or [batch, candidates]"
            )
        if self.config.item_embedding_mode == "content_dual":
            if not isinstance(prepared, tuple) or len(prepared) != 2:
                raise TypeError("dual scoring expects title and attribute states")
            assert self.title_tower is not None and self.attribute_tower is not None
            title_score = self.title_tower.score_prepared(
                prepared[0], candidate_ids
            )
            attribute_score = self.attribute_tower.score_prepared(
                prepared[1], candidate_ids
            )
            return 0.5 * title_score + 0.5 * attribute_score
        if not isinstance(prepared, torch.Tensor):
            raise TypeError("fused scoring expects one prepared state tensor")
        if self.config.item_embedding_mode == "content_fused":
            assert self.fused_tower is not None
            return self.fused_tower.score_prepared(prepared, candidate_ids)
        assert self.shared_private_tower is not None
        return self.shared_private_tower.score_prepared(prepared, candidate_ids)

    def score_components(
        self, contexts: torch.Tensor, candidate_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        prepared = self.prepare_scoring(contexts, candidate_ids)
        if self.config.item_embedding_mode != "content_dual":
            return (self.score_prepared(prepared, candidate_ids),)
        assert isinstance(prepared, tuple)
        assert self.title_tower is not None and self.attribute_tower is not None
        return (
            self.title_tower.score_prepared(prepared[0], candidate_ids),
            self.attribute_tower.score_prepared(prepared[1], candidate_ids),
        )

    def score(self, contexts: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        components = self.score_components(contexts, candidate_ids)
        if len(components) == 1:
            return components[0]
        return 0.5 * components[0] + 0.5 * components[1]

    def scoring_weight(self) -> torch.Tensor:
        self._require_id_mode()
        assert self.item_embedding is not None
        if self.item_embedding_adapter is None:
            return self.item_embedding.weight
        item_ids = torch.arange(
            self.config.num_items + 1, device=self.item_embedding.weight.device
        )
        return self.embed_items(item_ids)

    def lora_block_groups(self) -> tuple[nn.ModuleList, ...]:
        if self.config.item_embedding_mode == "id":
            return (self.blocks,)
        if self.config.item_embedding_mode == "content_dual":
            assert self.title_tower is not None and self.attribute_tower is not None
            return (self.title_tower.blocks, self.attribute_tower.blocks)
        if self.config.item_embedding_mode == "content_fused":
            assert self.fused_tower is not None
            return (self.fused_tower.blocks,)
        assert self.shared_private_tower is not None
        return (
            self.shared_private_tower.shared_blocks,
            *tuple(self.shared_private_tower.private_blocks),
        )

    def private_parameter_owners(self) -> dict[str, int]:
        if self.shared_private_tower is None:
            return {}
        return {
            f"shared_private_tower.{name}": domain
            for name, domain in self.shared_private_tower.private_parameter_owners().items()
        }

    def optimizer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        if self.config.item_embedding_mode != "id":
            return {
                "sparse": [],
                "dense": list(self.parameters()),
            }
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
        if self.config.item_embedding_mode == "id":
            groups: dict[str, tuple[str, ...]] = {
                "item_embedding": ("item_embedding.weight",),
                "position_embedding": ("position_embedding.weight",),
            }
            prefixes = (("", self.blocks),)
        elif self.config.item_embedding_mode == "content_dual":
            assert self.title_tower is not None and self.attribute_tower is not None
            groups = {
                "title_content_encoder": ("title_tower.item_encoder",),
                "attribute_content_encoder": ("attribute_tower.item_encoder",),
                "title_position_embedding": ("title_tower.position_embedding.weight",),
                "attribute_position_embedding": (
                    "attribute_tower.position_embedding.weight",
                ),
            }
            prefixes = (
                ("title_tower.", self.title_tower.blocks),
                ("attribute_tower.", self.attribute_tower.blocks),
            )
        elif self.config.item_embedding_mode == "content_fused":
            assert self.fused_tower is not None
            groups = {
                "fused_content_encoder": ("fused_tower.item_encoder",),
                "fused_position_embedding": (
                    "fused_tower.position_embedding.weight",
                ),
            }
            prefixes = (("fused_tower.", self.fused_tower.blocks),)
        else:
            assert self.shared_private_tower is not None
            groups = {
                "shared_private_content_encoder": ("shared_private_tower.item_encoder",),
                "shared_private_position_embedding": (
                    "shared_private_tower.position_embedding.weight",
                ),
                "shared_behavior_blocks": ("shared_private_tower.shared_blocks",),
            }
            for domain_index in range(len(self.shared_private_tower.private_blocks)):
                groups[f"private_behavior_domain_{domain_index}"] = (
                    f"shared_private_tower.private_blocks.{domain_index}",
                    f"shared_private_tower.final_norms.{domain_index}",
                )
            prefixes = ()

        for prefix, blocks in prefixes:
            label_prefix = prefix.replace(".", "_")
            for index in range(len(blocks)):
                path = f"{prefix}blocks.{index}"
                query = f"{path}.attention.q_proj.weight"
                value = f"{path}.attention.v_proj.weight"
                groups[f"{label_prefix}block_{index}_attention"] = (
                    f"{path}.attention_norm",
                    f"{path}.attention",
                )
                groups[f"{label_prefix}block_{index}_ffn"] = (
                    f"{path}.ffn_norm",
                    f"{path}.ffn",
                )
                groups[f"{label_prefix}block_{index}_q"] = (query,)
                groups[f"{label_prefix}block_{index}_v"] = (value,)
                groups[f"{label_prefix}block_{index}_qv"] = (query, value)
        return groups
