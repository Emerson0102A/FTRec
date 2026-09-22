from __future__ import annotations

import torch

from gmflowrec import (
    GMFlowRec,
    GMFlowRecConfig,
    GaussianMixtureOutput,
    gaussian_mixture_nll,
)


def _config() -> GMFlowRecConfig:
    return GMFlowRecConfig(
        item_count=8,
        domain_offsets={0: (0, 4), 1: (4, 8)},
        maxlen=5,
        hidden_units=8,
        num_blocks=1,
        num_heads=2,
        dropout_rate=0.0,
        num_mixtures=2,
    )


def test_dual_attention_masks_are_causal_and_domain_specific():
    model = GMFlowRec(_config())
    domains = torch.tensor([[-1, 0, 1, 0]])
    invariant = model.encoder._attention_mask(domains, domain_specific=False)[0]
    specific = model.encoder._attention_mask(domains, domain_specific=True)[0]

    assert invariant[3].tolist() == [True, False, False, False]
    assert specific[3].tolist() == [True, False, True, False]
    assert invariant[1, 2]
    assert not invariant[1, 1]


def test_domain_aligned_prior_uses_latest_matching_state_and_cold_start():
    model = GMFlowRec(_config())
    states = torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8)
    domains = torch.tensor([[-1, 0, 1, 0], [-1, 0, 0, 0]])
    targets = torch.tensor([0, 1])
    result = model.domain_aligned_prior(states, domains, targets)

    torch.testing.assert_close(
        result[0], states[0, 3] + model.domain_embedding.weight[1]
    )
    torch.testing.assert_close(result[1], model.domain_embedding.weight[2])


def test_gmm_nll_rewards_a_mean_near_the_target():
    target = torch.tensor([[1.0, -1.0]])
    close = GaussianMixtureOutput(
        logits=torch.tensor([[5.0, -5.0]]),
        means=torch.tensor([[[1.0, -1.0], [10.0, 10.0]]]),
        scales=torch.ones(1, 2),
    )
    far = GaussianMixtureOutput(
        logits=close.logits,
        means=close.means + 4.0,
        scales=close.scales,
    )
    assert gaussian_mixture_nll(close, target) < gaussian_mixture_nll(far, target)


def test_training_objective_and_inference_are_finite_and_differentiable():
    torch.manual_seed(3)
    model = GMFlowRec(_config())
    items = torch.tensor([[0, 1, 5, 2, 6], [0, 0, 3, 4, 7]])
    domains = torch.tensor([[-1, 0, 1, 0, 1], [-1, -1, 0, 0, 1]])
    targets = torch.tensor([3, 8])
    target_domains = torch.tensor([0, 1])
    losses = model.training_objective(
        items, domains, targets, target_domains, time=torch.tensor([0.25, 0.75])
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.mixture_means.weight.grad is not None
    assert torch.isfinite(model.mixture_means.weight.grad).all()

    candidates = torch.tensor([[3, 1, 4], [8, 5, 6]])
    scores = model.predict(items, domains, target_domains, candidates, steps=2)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()

