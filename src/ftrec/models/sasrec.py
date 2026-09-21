"""SASRec with ID, MyRec dual-tower, and content-fused item encoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .attention import SASRecBlock
from .content_encoder import (
    ATTRIBUTE_POOLING_MODES,
    AttributeItemEncoder,
    FusedContentItemEncoder,
    TitleItemEncoder,
    load_content_artifact,
)

if False:  # pragma: no cover - imported only for static type checkers
    from .embedding_adapter import TargetEmbeddingAdapter


ITEM_EMBEDDING_MODES = ("id", "content_dual", "content_fused")


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
        if (
            self.attribute_pooling != "hard_top1"
            and self.item_embedding_mode != "content_fused"
        ):
            raise ValueError(
                "non-hard attribute pooling is currently defined only for "
                "content_fused"
            )
        needs_domains = self.attribute_pooling == "domain_title_attention"
        if needs_domains and self.item_domain_file is None:
            raise ValueError(
                "domain_title_attention requires item_domain_file"
            )
        if not needs_domains and self.item_domain_file is not None:
            raise ValueError(
                "item_domain_file is only used by domain_title_attention"
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


class SASRec(nn.Module):
    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding_adapter: TargetEmbeddingAdapter | None = None
        self.item_embedding: nn.Embedding | None = None
        self.title_tower: _ContentSASRecTower | None = None
        self.attribute_tower: _ContentSASRecTower | None = None
        self.fused_tower: _ContentSASRecTower | None = None

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
            if config.item_embedding_mode == "content_dual":
                self.title_tower = _ContentSASRecTower(
                    config,
                    TitleItemEncoder(artifact, hidden_size=config.hidden_size),
                )
                self.attribute_tower = _ContentSASRecTower(
                    config,
                    AttributeItemEncoder(artifact, hidden_size=config.hidden_size),
                )
            else:
                self.fused_tower = _ContentSASRecTower(
                    config,
                    FusedContentItemEncoder(
                        artifact,
                        hidden_size=config.hidden_size,
                        attribute_pooling=config.attribute_pooling,
                        item_domain_file=config.item_domain_file,
                        attribute_temperature=config.attribute_temperature,
                    ),
                )
        self.reset_parameters()
        if self.fused_tower is not None:
            encoder = self.fused_tower.item_encoder
            if isinstance(encoder, FusedContentItemEncoder):
                encoder.reset_pooling_parameters()

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

    def clear_evaluation_cache(self) -> None:
        for tower in (self.title_tower, self.attribute_tower, self.fused_tower):
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
        self, contexts: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.config.item_embedding_mode == "id":
            return self.final_state(contexts)
        if self.config.item_embedding_mode == "content_dual":
            assert self.title_tower is not None and self.attribute_tower is not None
            return (
                self.title_tower.final_state(contexts),
                self.attribute_tower.final_state(contexts),
            )
        assert self.fused_tower is not None
        return self.fused_tower.final_state(contexts)

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
        assert self.fused_tower is not None
        return self.fused_tower.score_prepared(prepared, candidate_ids)

    def score_components(
        self, contexts: torch.Tensor, candidate_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        prepared = self.prepare_scoring(contexts)
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
        assert self.fused_tower is not None
        return (self.fused_tower.blocks,)

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
        else:
            assert self.fused_tower is not None
            groups = {
                "fused_content_encoder": ("fused_tower.item_encoder",),
                "fused_position_embedding": (
                    "fused_tower.position_embedding.weight",
                ),
            }
            prefixes = (("fused_tower.", self.fused_tower.blocks),)

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
