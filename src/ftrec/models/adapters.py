"""Residual bottleneck adapters for domain-specific SASRec adaptation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .sasrec import SASRec


ADAPTER_METHODS = ("houlsby", "pfeiffer")


class BottleneckAdapter(nn.Module):
    """A zero-initialized residual bottleneck that preserves base output."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or bottleneck_size < 1:
            raise ValueError("hidden_size and bottleneck_size must be positive")
        self.down = nn.Linear(
            hidden_size, bottleneck_size, device=device, dtype=dtype
        )
        self.up = nn.Linear(
            bottleneck_size, hidden_size, device=device, dtype=dtype
        )
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.up(F.gelu(self.down(inputs)))


def freeze_for_adapter(model: nn.Module) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, BottleneckAdapter):
            continue
        for parameter_name, parameter in module.named_parameters():
            parameter.requires_grad_(True)
            trainable.append(f"{name}.{parameter_name}")
    if not trainable:
        raise ValueError("model contains no bottleneck adapters")
    return tuple(trainable)


def inject_adapters(
    model: SASRec, *, method: str, bottleneck_size: int
) -> SASRec:
    if method not in ADAPTER_METHODS:
        raise ValueError(
            f"unknown adapter method {method!r}; expected one of {ADAPTER_METHODS}"
        )
    for blocks in model.lora_block_groups():
        for block in blocks:
            if block.attention_adapter is not None or block.ffn_adapter is not None:
                raise ValueError("an adapter has already been injected")
            reference = block.attention.q_proj.weight
            if method == "houlsby":
                block.attention_adapter = BottleneckAdapter(
                    model.config.hidden_size,
                    bottleneck_size,
                    device=reference.device,
                    dtype=reference.dtype,
                )
            block.ffn_adapter = BottleneckAdapter(
                model.config.hidden_size,
                bottleneck_size,
                device=reference.device,
                dtype=reference.dtype,
            )
    freeze_for_adapter(model)
    return model


def adapter_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (".attention_adapter." in name or ".ffn_adapter." in name)
    )


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".attention_adapter." in name or ".ffn_adapter." in name
    }
