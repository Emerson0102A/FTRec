"""Fingerprint-checked, CPU-portable experiment checkpoints."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class CheckpointMismatchError(RuntimeError):
    """Raised when a checkpoint belongs to another data/config lineage."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    metadata: dict[str, Any]
    training_state: dict[str, Any]
    optimizer_state: dict[str, Any]


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    metadata: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {**dict(metadata), "model_state_hash": model_state_hash(model)},
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": dict(optimizer_state or {}),
        "rng": {
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        },
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


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    expected: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> LoadedCheckpoint:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != 1:
        raise CheckpointMismatchError("unsupported checkpoint schema_version")
    metadata = dict(payload["metadata"])
    for key, expected_value in dict(expected or {}).items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise CheckpointMismatchError(
                f"checkpoint {key} mismatch: expected {expected_value!r}, got {actual!r}"
            )
    model.load_state_dict(payload["model"], strict=True)
    actual_model_hash = model_state_hash(model)
    expected_model_hash = metadata.get("model_state_hash")
    if expected_model_hash != actual_model_hash:
        raise CheckpointMismatchError(
            "checkpoint model_state_hash mismatch: "
            f"expected {expected_model_hash!r}, got {actual_model_hash!r}"
        )
    if restore_rng:
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        # map_location may move every tensor in the payload to CUDA, but the
        # default CPU generator only accepts a CPU ByteTensor state.
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if torch.cuda.is_available() and payload["rng"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["rng"]["torch_cuda"]]
            )
    return LoadedCheckpoint(
        metadata,
        dict(payload.get("training_state", {})),
        dict(payload.get("optimizer", {})),
    )
