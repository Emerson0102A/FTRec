from __future__ import annotations

import math

import pytest


def test_recovery_is_signed_unclipped_and_marks_zero_denominator() -> None:
    from ftrec.analysis.recovery import recovery

    assert recovery(pretrain=0.2, lora=0.5, fullft=0.4).value == pytest.approx(1.5)
    assert recovery(pretrain=0.4, lora=0.3, fullft=0.2).value == pytest.approx(0.5)
    undefined = recovery(pretrain=0.2, lora=0.3, fullft=0.2)

    assert math.isnan(undefined.value)
    assert undefined.undefined_recovery


def test_recovery_warns_when_fullft_is_below_pretrain() -> None:
    from ftrec.analysis.recovery import recovery

    result = recovery(pretrain=0.4, lora=0.3, fullft=0.2)
    assert "below" in result.warning
