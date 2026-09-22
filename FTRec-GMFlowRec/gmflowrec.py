"""Paper-driven reproduction of GMFlowRec.

The implementation follows arXiv:2510.21021.  The paper leaves the target of
the GMM likelihood and the sign of the reverse ODE ambiguous.  This module uses
the mathematically consistent reverse velocity ``x_item - x_user`` and applies
the recommendation objective directly to the expected GMM velocity, as written
in Equation (11).  See
``docs/gmflowrec-reproduction.md`` for the exact correspondence and assumptions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GMFlowRecConfig:
    item_count: int
    domain_offsets: Mapping[int, tuple[int, int]]
    maxlen: int = 50
    hidden_units: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    dropout_rate: float = 0.1
    num_mixtures: int = 8
    fusion_weight: float = 0.9
    prior_weight: float = 0.1
    gmm_weight: float = 1e-4
    min_scale: float = 1e-3

    def __post_init__(self) -> None:
        if self.item_count <= 0 or self.maxlen <= 0 or self.hidden_units <= 0:
            raise ValueError("item_count, maxlen, and hidden_units must be positive")
        if self.num_blocks <= 0 or self.num_heads <= 0 or self.num_mixtures <= 0:
            raise ValueError("encoder and mixture counts must be positive")
        if self.hidden_units % self.num_heads:
            raise ValueError("hidden_units must be divisible by num_heads")
        if not 0.0 <= self.fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")
        keys = sorted(int(key) for key in self.domain_offsets)
        if keys != list(range(len(keys))):
            raise ValueError("domain IDs must be contiguous and start at zero")
        previous_end = 0
        for domain_id in keys:
            start, end = self.domain_offsets[domain_id]
            if start != previous_end or end <= start:
                raise ValueError("domain offsets must be contiguous non-empty ranges")
            previous_end = end
        if previous_end != self.item_count:
            raise ValueError("domain offsets must cover every item exactly once")

    @property
    def domain_count(self) -> int:
        return len(self.domain_offsets)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["domain_offsets"] = {
            str(key): list(value) for key, value in self.domain_offsets.items()
        }
        return result


@dataclass
class GaussianMixtureOutput:
    logits: Tensor
    means: Tensor
    scales: Tensor

    @property
    def weights(self) -> Tensor:
        return self.logits.softmax(dim=-1)

    @property
    def mean(self) -> Tensor:
        return torch.sum(self.weights.unsqueeze(-1) * self.means, dim=1)


def gaussian_mixture_nll(output: GaussianMixtureOutput, target: Tensor) -> Tensor:
    """Mean NLL under a mixture with one spherical covariance per component."""

    if target.ndim != 2 or output.means.ndim != 3:
        raise ValueError("target must be [B,D] and mixture means must be [B,K,D]")
    if output.means.shape[0] != target.shape[0] or output.means.shape[2] != target.shape[1]:
        raise ValueError("target and mixture shapes are incompatible")
    if output.scales.shape != output.logits.shape:
        raise ValueError("mixture scales and logits must have shape [B,K]")

    residual = (target.unsqueeze(1) - output.means) / output.scales.unsqueeze(-1)
    dimension = target.shape[-1]
    component_log_prob = -0.5 * residual.square().sum(dim=-1)
    component_log_prob -= dimension * output.scales.log()
    component_log_prob -= 0.5 * dimension * math.log(2.0 * math.pi)
    log_prob = torch.logsumexp(
        output.logits.log_softmax(dim=-1) + component_log_prob, dim=-1
    )
    return -log_prob.mean()


class SharedDualMaskedEncoder(nn.Module):
    """One Transformer reused with the paper's two attention masks."""

    def __init__(self, config: GMFlowRecConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_units,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_units,
            dropout=config.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_blocks,
            norm=nn.LayerNorm(config.hidden_units),
            enable_nested_tensor=False,
        )
        self.num_heads = config.num_heads

    def _attention_mask(self, domains: Tensor, domain_specific: bool) -> Tensor:
        """Return a boolean blocked-attention mask shaped ``[B*H,L,L]``."""

        if domains.ndim != 2:
            raise ValueError("domains must have shape [batch, sequence]")
        batch_size, length = domains.shape
        device = domains.device
        padding = domains.lt(0)
        query_index = torch.arange(length, device=device).view(1, length, 1)
        key_index = torch.arange(length, device=device).view(1, 1, length)

        blocked = key_index.gt(query_index).expand(batch_size, -1, -1).clone()
        blocked |= padding.unsqueeze(1).expand(-1, length, -1)
        if domain_specific:
            blocked |= domains.unsqueeze(2).ne(domains.unsqueeze(1))

        # A fully masked padding query produces NaNs in softmax.  Let each
        # padding position attend only to itself; valid queries still cannot
        # attend to padding keys.
        if padding.any():
            pad_batch, pad_position = padding.nonzero(as_tuple=True)
            blocked[pad_batch, pad_position, :] = True
            blocked[pad_batch, pad_position, pad_position] = False

        return (
            blocked.unsqueeze(1)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(batch_size * self.num_heads, length, length)
        )

    def forward(self, inputs: Tensor, domains: Tensor) -> tuple[Tensor, Tensor]:
        padding = domains.lt(0)
        invariant_mask = self._attention_mask(domains, domain_specific=False)
        specific_mask = self._attention_mask(domains, domain_specific=True)
        invariant = self.encoder(inputs, mask=invariant_mask, is_causal=False)
        specific = self.encoder(inputs, mask=specific_mask, is_causal=False)
        invariant = invariant.masked_fill(padding.unsqueeze(-1), 0.0)
        specific = specific.masked_fill(padding.unsqueeze(-1), 0.0)
        return invariant, specific


class GMFlowRec(nn.Module):
    """Gaussian-mixture flow model for multi-domain sequential recommendation."""

    def __init__(self, config: GMFlowRecConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_units
        mixtures = config.num_mixtures

        self.item_embedding = nn.Embedding(
            config.item_count + 1, hidden, padding_idx=0
        )
        self.domain_embedding = nn.Embedding(
            config.domain_count + 1, hidden, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            config.maxlen + 1, hidden, padding_idx=0
        )
        self.embedding_norm = nn.LayerNorm(hidden)
        self.embedding_dropout = nn.Dropout(config.dropout_rate)
        self.encoder = SharedDualMaskedEncoder(config)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.flow_trunk = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.SiLU(),
        )
        self.mixture_logits = nn.Linear(2 * hidden, mixtures)
        self.mixture_means = nn.Linear(2 * hidden, mixtures * hidden)
        self.mixture_scales = nn.Linear(2 * hidden, mixtures)

        # Convert the dataset's zero-based [start,end) ranges to slices over
        # the one-based embedding table (item 0 is padding).
        self.domain_item_slices = {
            int(domain): (int(start) + 1, int(end) + 1)
            for domain, (start, end) in config.domain_offsets.items()
        }
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.item_embedding.weight.device

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.domain_embedding.weight[0].zero_()
            self.position_embedding.weight[0].zero_()

    def _validate_inputs(self, items: Tensor, domains: Tensor) -> None:
        if items.ndim != 2 or domains.shape != items.shape:
            raise ValueError("items and domains must have identical [B,L] shapes")
        if items.shape[1] > self.config.maxlen:
            raise ValueError("input sequence exceeds configured maxlen")
        if torch.any(items.lt(0)) or torch.any(items.gt(self.config.item_count)):
            raise ValueError("item ID outside the embedding table")
        valid = items.ne(0)
        if torch.any(valid.sum(dim=1).eq(0)):
            raise ValueError("every sequence needs at least one context item")
        if torch.any(domains[valid].lt(0)) or torch.any(
            domains[valid].ge(self.config.domain_count)
        ):
            raise ValueError("domain ID outside the configured domains")
        if torch.any(domains[~valid].ne(-1)):
            raise ValueError("padding items must use domain ID -1")

    def encode(self, items: Tensor, domains: Tensor) -> tuple[Tensor, Tensor]:
        """Compute domain-invariant and domain-specific sequence priors."""

        items = items.to(device=self.device, dtype=torch.long)
        domains = domains.to(device=self.device, dtype=torch.long)
        self._validate_inputs(items, domains)
        valid = items.ne(0)
        positions = valid.long().cumsum(dim=1) * valid.long()
        shifted_domains = torch.where(valid, domains + 1, torch.zeros_like(domains))
        embedded = self.item_embedding(items)
        embedded = embedded + self.domain_embedding(shifted_domains)
        embedded = embedded + self.position_embedding(positions)
        embedded = self.embedding_dropout(self.embedding_norm(embedded))
        return self.encoder(embedded, domains)

    @staticmethod
    def _last_valid_state(states: Tensor, items: Tensor) -> Tensor:
        positions = torch.arange(items.shape[1], device=items.device).unsqueeze(0)
        last_index = positions.masked_fill(items.eq(0), -1).max(dim=1).values
        return states.gather(
            1, last_index.view(-1, 1, 1).expand(-1, 1, states.shape[-1])
        ).squeeze(1)

    def domain_aligned_prior(
        self, domain_specific_states: Tensor, domains: Tensor, target_domains: Tensor
    ) -> Tensor:
        """Equation (3): latest in-domain state plus the target-domain embedding."""

        domains = domains.to(device=self.device, dtype=torch.long)
        target_domains = target_domains.to(device=self.device, dtype=torch.long)
        if target_domains.ndim != 1 or target_domains.shape[0] != domains.shape[0]:
            raise ValueError("target_domains must have shape [batch]")
        if torch.any(target_domains.lt(0)) or torch.any(
            target_domains.ge(self.config.domain_count)
        ):
            raise ValueError("target domain outside the configured domains")

        positions = torch.arange(domains.shape[1], device=self.device).unsqueeze(0)
        matches = domains.eq(target_domains.unsqueeze(1)) & domains.ge(0)
        latest = positions.expand_as(domains).masked_fill(~matches, -1).max(dim=1).values
        has_history = latest.ge(0)
        safe_latest = latest.clamp_min(0)
        gathered = domain_specific_states.gather(
            1,
            safe_latest.view(-1, 1, 1).expand(
                -1, 1, domain_specific_states.shape[-1]
            ),
        ).squeeze(1)
        gathered = torch.where(has_history.unsqueeze(1), gathered, torch.zeros_like(gathered))
        return gathered + self.domain_embedding(target_domains + 1)

    def mixture(
        self, latent_state: Tensor, invariant_prior: Tensor, aligned_prior: Tensor, time: Tensor
    ) -> GaussianMixtureOutput:
        """Equations (6)--(9), with explicit time conditioning."""

        if time.ndim == 1:
            time = time.unsqueeze(1)
        if time.shape != (latent_state.shape[0], 1):
            raise ValueError("time must have shape [batch] or [batch,1]")
        time = time.to(device=latent_state.device, dtype=latent_state.dtype)
        fused = (
            self.config.fusion_weight * latent_state
            + (1.0 - self.config.fusion_weight) * invariant_prior
        )
        time_features = self.time_embedding(time)
        trunk = self.flow_trunk(torch.cat([fused, aligned_prior, time_features], dim=-1))
        means = self.mixture_means(trunk).view(
            -1, self.config.num_mixtures, self.config.hidden_units
        )
        scales = F.softplus(self.mixture_scales(trunk)) + self.config.min_scale
        return GaussianMixtureOutput(
            logits=self.mixture_logits(trunk), means=means, scales=scales
        )

    def _domain_softmax_nll(
        self, representations: Tensor, targets: Tensor, target_domains: Tensor
    ) -> Tensor:
        """Equations (4), (5), and (11), using each domain's full vocabulary."""

        losses = []
        for domain, (start, end) in self.domain_item_slices.items():
            selected = target_domains.eq(domain)
            if not torch.any(selected):
                continue
            domain_targets = targets[selected] - start
            if torch.any(domain_targets.lt(0)) or torch.any(domain_targets.ge(end - start)):
                raise ValueError("target item does not belong to its target domain")
            item_weights = self.item_embedding.weight[start:end]
            logits = representations[selected] @ item_weights.transpose(0, 1)
            losses.append(F.cross_entropy(logits, domain_targets, reduction="none"))
        if not losses:
            raise ValueError("batch contains no recognized target domains")
        return torch.cat(losses).mean()

    def training_objective(
        self,
        items: Tensor,
        domains: Tensor,
        targets: Tensor,
        target_domains: Tensor,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return the weighted paper objective and its three components."""

        items = items.to(device=self.device, dtype=torch.long)
        domains = domains.to(device=self.device, dtype=torch.long)
        targets = targets.to(device=self.device, dtype=torch.long)
        target_domains = target_domains.to(device=self.device, dtype=torch.long)
        invariant_states, specific_states = self.encode(items, domains)
        invariant_prior = self._last_valid_state(invariant_states, items)
        aligned_prior = self.domain_aligned_prior(
            specific_states, domains, target_domains
        )
        target_embedding = self.item_embedding(targets)

        if time is None:
            time = torch.rand(items.shape[0], 1, device=self.device)
        else:
            time = time.to(device=self.device, dtype=invariant_prior.dtype)
            if time.ndim == 1:
                time = time.unsqueeze(1)
        latent_state = (1.0 - time) * target_embedding + time * invariant_prior
        mixture = self.mixture(latent_state, invariant_prior, aligned_prior, time)

        # Along x_t=(1-t)x_0+t*x_1, a reverse step uses x_0-x_1.
        reverse_velocity = target_embedding - invariant_prior
        gmm_loss = gaussian_mixture_nll(mixture, reverse_velocity)
        recommendation_loss = self._domain_softmax_nll(
            mixture.mean, targets, target_domains
        )
        prior_loss = self._domain_softmax_nll(
            aligned_prior, targets, target_domains
        )
        loss = (
            recommendation_loss
            + self.config.prior_weight * prior_loss
            + self.config.gmm_weight * gmm_loss
        )
        return {
            "loss": loss,
            "recommendation_loss": recommendation_loss,
            "prior_loss": prior_loss,
            "gmm_loss": gmm_loss,
        }

    def generate(
        self, items: Tensor, domains: Tensor, target_domains: Tensor, steps: int = 8
    ) -> Tensor:
        """Run the first-order reverse Euler solver from user state to item state."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        items = items.to(device=self.device, dtype=torch.long)
        domains = domains.to(device=self.device, dtype=torch.long)
        target_domains = target_domains.to(device=self.device, dtype=torch.long)
        invariant_states, specific_states = self.encode(items, domains)
        invariant_prior = self._last_valid_state(invariant_states, items)
        aligned_prior = self.domain_aligned_prior(
            specific_states, domains, target_domains
        )

        state = invariant_prior
        step_size = 1.0 / steps
        for step in range(steps):
            current_time = 1.0 - step * step_size
            time = torch.full(
                (items.shape[0], 1),
                current_time,
                device=self.device,
                dtype=state.dtype,
            )
            velocity = self.mixture(state, invariant_prior, aligned_prior, time).mean
            state = state + step_size * velocity
        return state

    @torch.no_grad()
    def predict(
        self,
        items: Tensor,
        domains: Tensor,
        target_domains: Tensor,
        candidates: Tensor,
        steps: int = 8,
    ) -> Tensor:
        generated = self.generate(items, domains, target_domains, steps=steps)
        candidates = candidates.to(device=self.device, dtype=torch.long)
        candidate_embeddings = self.item_embedding(candidates)
        return torch.einsum("bch,bh->bc", candidate_embeddings, generated)

