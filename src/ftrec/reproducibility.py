"""Deterministic runtime helpers shared by every experiment stage."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


class DeviceError(RuntimeError):
    """Raised when a requested accelerator is unavailable."""


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)


def resolve_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise DeviceError("CUDA was requested but is not available")
    if normalized not in {"cpu", "cuda"} and not normalized.startswith("cuda:"):
        raise DeviceError(f"unsupported device: {name}")
    return torch.device(normalized)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

