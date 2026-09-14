"""Q/V-only low-rank adapters for the explicit-projection SASRec."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from ftrec.training.checkpoint import CheckpointMismatchError

from .sasrec import SASRec


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapter = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return self.base(inputs) + adapter * self.scaling


def freeze_for_lora(model: nn.Module) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        module.lora_A.requires_grad_(True)
        module.lora_B.requires_grad_(True)
        trainable.extend((f"{name}.lora_A", f"{name}.lora_B"))
    if not trainable:
        raise ValueError("model contains no LoRA modules")
    return tuple(trainable)


def inject_qv_lora(model: SASRec, rank: int, alpha: float) -> SASRec:
    for block in model.blocks:
        attention = block.attention
        if isinstance(attention.q_proj, LoRALinear) or isinstance(
            attention.v_proj, LoRALinear
        ):
            raise ValueError("LoRA has already been injected")
        attention.q_proj = LoRALinear(attention.q_proj, rank, alpha)
        attention.v_proj = LoRALinear(attention.v_proj, rank, alpha)
    freeze_for_lora(model)
    return model


def lora_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in name
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".lora_" in name
    }


def save_adapter_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    metadata: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter": lora_state_dict(model),
        "metadata": dict(metadata),
        "schema_version": 1,
        "training_state": dict(training_state or {}),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_adapter_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    expected: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise CheckpointMismatchError("unsupported adapter checkpoint schema_version")
    metadata = dict(payload["metadata"])
    for key, expected_value in dict(expected or {}).items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise CheckpointMismatchError(
                f"adapter {key} mismatch: expected {expected_value!r}, got {actual!r}"
            )
    unexpected = set(payload["adapter"]) - set(model.state_dict())
    if unexpected:
        raise CheckpointMismatchError(
            f"adapter contains unknown tensors: {sorted(unexpected)}"
        )
    model.load_state_dict(payload["adapter"], strict=False)
    return metadata
