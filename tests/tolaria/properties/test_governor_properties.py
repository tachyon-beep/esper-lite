"""Property-based tests for TolariaGovernor safety invariants."""

from __future__ import annotations

import pytest
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.leyline import MIN_GOVERNOR_HISTORY_SAMPLES
from esper.tolaria import TolariaGovernor

pytestmark = pytest.mark.property


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):  # type: ignore[override]
        return self.linear(x)


@given(
    losses=st.lists(
        st.floats(min_value=0.0, max_value=9.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=50,
    )
)
@settings(max_examples=50)
def test_check_vital_signs_never_panics_below_absolute_threshold(losses: list[float]) -> None:
    gov = TolariaGovernor(DummyModel(), random_guess_loss=1e9)

    for loss in losses:
        assert gov.check_vital_signs(loss) is False

    assert gov.consecutive_panics == 0


@given(
    min_panics=st.integers(min_value=1, max_value=5),
    extra_anomalies=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=30)
def test_check_vital_signs_requires_consecutive_panics(
    min_panics: int, extra_anomalies: int
) -> None:
    gov = TolariaGovernor(
        DummyModel(),
        history_window=MIN_GOVERNOR_HISTORY_SAMPLES,
        min_panics_before_rollback=min_panics,
        random_guess_loss=1e9,
    )

    for _ in range(MIN_GOVERNOR_HISTORY_SAMPLES):
        assert gov.check_vital_signs(1.0) is False

    anomaly_loss = gov.absolute_threshold * 1000.0
    total_anomalies = min_panics + extra_anomalies
    for i in range(total_anomalies):
        should_panic = i + 1 >= min_panics
        assert gov.check_vital_signs(anomaly_loss) is should_panic

    assert gov.consecutive_panics == total_anomalies

    assert gov.check_vital_signs(1.0) is False
    assert gov.consecutive_panics == 0

