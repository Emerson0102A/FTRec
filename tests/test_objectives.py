import torch
import torch.nn.functional as F


def test_sampled_bce_matches_manual_logits() -> None:
    from ftrec.training.objectives import sampled_bce_loss

    positive = torch.tensor([2.0, 0.0])
    negative = torch.tensor([-1.0, 1.0])
    expected = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
    expected += F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))

    torch.testing.assert_close(sampled_bce_loss(positive, negative), expected)

