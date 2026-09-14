"""SASRec sampled next-item objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sampled_bce_loss(
    positive_logits: torch.Tensor, negative_logits: torch.Tensor
) -> torch.Tensor:
    if positive_logits.shape != negative_logits.shape:
        raise ValueError("positive and negative logits must have identical shapes")
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_logits, torch.zeros_like(negative_logits)
    )
    return positive_loss + negative_loss

