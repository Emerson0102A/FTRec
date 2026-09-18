from __future__ import annotations

import copy

import pytest
import torch


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


@pytest.mark.parametrize(
    "method,adapters_per_block", (("houlsby", 2), ("pfeiffer", 1))
)
@pytest.mark.parametrize("bottleneck", (1, 2, 4))
def test_zero_initialized_adapter_preserves_output_and_exact_capacity(
    method: str, adapters_per_block: int, bottleneck: int
) -> None:
    from ftrec.models.adapters import adapter_parameter_names, inject_adapters

    base = _tiny_sasrec().eval()
    adapted = inject_adapters(
        copy.deepcopy(base), method=method, bottleneck_size=bottleneck
    ).eval()

    torch.testing.assert_close(adapted.encode(INPUTS), base.encode(INPUTS))
    names = adapter_parameter_names(adapted)
    assert names == tuple(
        name for name, parameter in adapted.named_parameters() if parameter.requires_grad
    )
    expected_per_adapter = 2 * 8 * bottleneck + bottleneck + 8
    actual = sum(
        parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad
    )
    assert actual == 2 * adapters_per_block * expected_per_adapter


@pytest.mark.parametrize("method", ("houlsby", "pfeiffer"))
def test_adapter_step_changes_adapter_but_preserves_backbone(method: str) -> None:
    from ftrec.models.adapters import inject_adapters

    model = inject_adapters(
        _tiny_sasrec(blocks=1), method=method, bottleneck_size=2
    )
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    before_up = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.endswith("up.weight")
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    model.score(INPUTS, torch.tensor([[6, 7], [8, 9]])).sum().backward()
    optimizer.step()

    parameters = dict(model.named_parameters())
    for name, expected in frozen.items():
        assert torch.equal(parameters[name], expected), name
    assert any(
        not torch.equal(parameters[name], expected)
        for name, expected in before_up.items()
    )
