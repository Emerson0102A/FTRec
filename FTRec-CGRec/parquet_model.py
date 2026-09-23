"""Single-device CGRec wrapper for the five-domain Parquet benchmark.

The published model is kept under ``src/``. This wrapper retains its encoder,
per-domain recommendation loss, and exhaustive Shapley comparisons. Category
inputs can be restored from an ASIN-aligned Amazon metadata catalog.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn


SOURCE_DIR = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from models import CausalModel  # noqa: E402


class CGRecParquetModel(CausalModel):
    """Run the released CGRec item path on CPU or one CUDA device."""

    def __init__(
        self,
        item_count: int,
        maxlen: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        device: str | torch.device,
        shapley: bool = True,
        cat1_size: int = 1,
        cat2_size: int = 1,
        hierarchical: bool = False,
    ):
        if min(item_count, maxlen, hidden_size, num_layers, num_heads) <= 0:
            raise ValueError("model sizes must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if hierarchical and min(cat1_size, cat2_size) <= 1:
            raise ValueError("hierarchical CGRec requires two nonempty category vocabularies")
        args = SimpleNamespace(
            # CGRec reserves 0..4; original item IDs are shifted by five.
            item_size=item_count + 5,
            type_size=10,
            cat1_size=cat1_size,
            cat2_size=cat2_size,
            max_seq_length=maxlen,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_act="gelu",
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout,
            initializer_range=0.02,
            loss_type="negative",
            hierarhical="y" if hierarchical else "n",
            shaply_value="y" if shapley else "n",
            local_rank=0,
        )
        super().__init__(args)
        # This full-vocabulary output projection is never used by the published
        # item-level CGRec loss or sampled ranking and otherwise doubles memory.
        self.mlm_output = nn.Identity()
        old_state = self.shaply_values_update
        del self.shaply_values_update
        self.register_buffer("shaply_values_update", old_state)
        self.device = torch.device(device)
        self.to(self.device)

    def shaply_layer(self, *args, **kwargs):
        # The released code detaches all subset losses while creating its weight
        # tensor. Avoid retaining 30 unused autograd graphs per training batch.
        with torch.no_grad():
            weights = super().shaply_layer(*args, **kwargs)
        # The released implementation rebuilds the softmax input with
        # torch.tensor(...), which places its weights on CPU even on CUDA.
        return {domain: weight.to(self.device) for domain, weight in weights.items()}

    def softmax_with_temperature(self, preds: Tensor, temperature: float) -> Tensor:
        return torch.softmax(preds / temperature, dim=0)

    def train_loss(
        self,
        items: Tensor,
        positives: Tensor,
        negatives: Tensor,
        cat1_input: Tensor,
        cat1_pos: Tensor,
        cat1_neg: Tensor,
        cat2_input: Tensor,
        cat2_pos: Tensor,
        cat2_neg: Tensor,
        domains: Tensor,
    ) -> Tensor:
        zeros = torch.zeros_like(items)
        loss, _, _, _ = self.pretrain_seq(
            items,
            positives,
            negatives,
            zeros,  # test negatives: unused in training
            zeros,  # test answer: unused in training
            cat1_input,
            cat1_pos,
            cat1_neg,
            cat2_input,
            cat2_pos,
            cat2_neg,
            domains,
            self.args.hierarhical,
        )
        return loss

    def score(
        self, items: Tensor, cat1: Tensor, cat2: Tensor,
        domains: Tensor, candidates: Tensor,
    ) -> Tensor:
        zeros = torch.zeros_like(items)
        _, _, recommendation = self.get_last_emb(
            items, cat1, cat2, domains, zeros, zeros,
            self.args.hierarhical, cuda_yn="y",
        )
        return torch.sum(
            self.item_embeddings(candidates) * recommendation.unsqueeze(1), dim=-1
        )
