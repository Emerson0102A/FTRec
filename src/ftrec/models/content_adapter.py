"""Target-domain residual adaptation for fused content item representations."""

from __future__ import annotations

from torch import nn

from .adapters import BottleneckAdapter
from .content_encoder import FusedContentItemEncoder
from .sasrec import SASRec


CONTENT_ADAPTER_PREFIX = "fused_tower.item_encoder.content_adapter."


def inject_fused_content_adapter(
    model: SASRec,
    *,
    bottleneck_size: int,
    freeze_existing: bool,
) -> BottleneckAdapter:
    """Add a zero-initialized residual adapter after fused item encoding.

    The adapter transforms both history and candidate representations, so the
    target-domain scoring space remains shared.  Its zero-initialized up
    projection makes the injected model exactly equal to the base checkpoint
    before the first optimizer step.
    """

    if model.config.item_embedding_mode != "content_fused":
        raise ValueError("fused content adaptation requires item_embedding_mode=content_fused")
    if model.fused_tower is None or not isinstance(
        model.fused_tower.item_encoder, FusedContentItemEncoder
    ):
        raise RuntimeError("content_fused model is missing its fused item encoder")
    encoder = model.fused_tower.item_encoder
    if encoder.content_adapter is not None:
        raise ValueError("a fused content adapter has already been injected")
    if freeze_existing:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    reference = encoder.output_norm.weight
    adapter = BottleneckAdapter(
        model.config.hidden_size,
        bottleneck_size,
        device=reference.device,
        dtype=reference.dtype,
    )
    encoder.content_adapter = adapter
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    return adapter


def content_adapter_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(CONTENT_ADAPTER_PREFIX)
    )
