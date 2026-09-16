from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch


RANKS = (1, 2, 4, 8, 16)
INPUTS = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)


def _tiny_sasrec(*, blocks: int = 2, hidden: int = 8):
    from ftrec.models.sasrec import SASRec, SASRecConfig

    torch.manual_seed(42)
    return SASRec(
        SASRecConfig(
            num_items=20,
            hidden_size=hidden,
            num_blocks=blocks,
            num_heads=1,
            dropout=0,
            maxlen=3,
        )
    )


@pytest.mark.parametrize("rank", RANKS)
def test_zero_initialized_qv_lora_preserves_output_and_trainable_set(rank: int) -> None:
    from ftrec.models.lora import inject_qv_lora

    base = _tiny_sasrec().eval()
    adapted = inject_qv_lora(copy.deepcopy(base), rank=rank, alpha=rank).eval()

    torch.testing.assert_close(adapted.encode(INPUTS), base.encode(INPUTS))
    trainable = [
        name for name, parameter in adapted.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(
        ("q_proj" in name or "v_proj" in name) and ".lora_" in name
        for name in trainable
    )


def test_qv_lora_parameter_count_is_exact_and_monotonic() -> None:
    from ftrec.models.lora import count_trainable_parameters, inject_qv_lora

    counts = [
        count_trainable_parameters(inject_qv_lora(_tiny_sasrec(), rank, rank))
        for rank in RANKS
    ]

    assert counts == [4 * 2 * 8 * rank for rank in RANKS]
    assert counts == sorted(counts)


def test_lora_parameters_inherit_base_projection_device_and_dtype() -> None:
    """Catch adapters being created on CPU/float32 after a model moved to CUDA/bf16."""
    from torch import nn

    from ftrec.models.lora import LoRALinear

    base = nn.Linear(4, 4, device="meta", dtype=torch.float64)
    adapted = LoRALinear(base, rank=2, alpha=2)

    assert adapted.lora_A.device == base.weight.device
    assert adapted.lora_B.device == base.weight.device
    assert adapted.lora_A.dtype == base.weight.dtype
    assert adapted.lora_B.dtype == base.weight.dtype


def test_lora_step_changes_adapter_but_preserves_every_base_tensor() -> None:
    from ftrec.models.lora import inject_qv_lora

    model = inject_qv_lora(_tiny_sasrec(blocks=1), rank=2, alpha=2)
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    before_b = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.endswith("lora_B")
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    loss = model.score(INPUTS, torch.tensor([[6, 7], [8, 9]])).sum()
    loss.backward()
    optimizer.step()

    for name, expected in frozen.items():
        assert torch.equal(dict(model.named_parameters())[name], expected), name
    assert any(
        not torch.equal(dict(model.named_parameters())[name], expected)
        for name, expected in before_b.items()
    )


def test_adapter_round_trip_rejects_wrong_base_hash(tmp_path: Path) -> None:
    from ftrec.models.lora import (
        inject_qv_lora,
        load_adapter_checkpoint,
        lora_state_dict,
        save_adapter_checkpoint,
    )
    from ftrec.training.checkpoint import CheckpointMismatchError

    model = inject_qv_lora(_tiny_sasrec(blocks=1), rank=2, alpha=2)
    path = save_adapter_checkpoint(
        tmp_path / "adapter.pt",
        model,
        metadata={"base_hash": "base-a", "rank": 2, "alpha": 2},
    )
    expected = lora_state_dict(model)
    reloaded = inject_qv_lora(_tiny_sasrec(blocks=1), rank=2, alpha=2)
    load_adapter_checkpoint(path, reloaded, expected={"base_hash": "base-a"})
    for name, value in expected.items():
        assert torch.equal(lora_state_dict(reloaded)[name], value)

    with pytest.raises(CheckpointMismatchError, match="base_hash"):
        load_adapter_checkpoint(path, reloaded, expected={"base_hash": "base-b"})
