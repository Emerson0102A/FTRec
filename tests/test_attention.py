import copy

import torch


def test_future_token_cannot_change_previous_attention_states() -> None:
    from ftrec.models.attention import CausalSelfAttention

    torch.manual_seed(3)
    attention = CausalSelfAttention(hidden_size=4, num_heads=1, dropout=0.0).eval()
    first = torch.randn(1, 4, 4)
    second = first.clone()
    second[:, 3] = torch.randn(4)
    valid = torch.ones(1, 4, dtype=torch.bool)

    first_output = attention(first, valid)
    second_output = attention(second, valid)

    torch.testing.assert_close(first_output[:, :3], second_output[:, :3])


def test_padding_key_and_query_have_no_effect_or_output() -> None:
    from ftrec.models.attention import CausalSelfAttention

    torch.manual_seed(4)
    attention = CausalSelfAttention(hidden_size=4, num_heads=2, dropout=0.0).eval()
    values = torch.randn(1, 4, 4)
    changed_padding = values.clone()
    changed_padding[:, :2] = 1000
    valid = torch.tensor([[False, False, True, True]])

    output = attention(values, valid)
    changed = attention(changed_padding, valid)

    assert torch.count_nonzero(output[:, :2]) == 0
    torch.testing.assert_close(output[:, 2:], changed[:, 2:])

