"""G1 fossilize-rate guard (pre-A/B build WI-3; gate criterion 3).

The fast in-run RUNAWAY breaker for the committed-Shapley ON arm: per PPO
batch (12 env-episodes), >= FOSSILIZE_RATE_GUARD_TRIP_COUNT successful
fossilizations for FOSSILIZE_RATE_GUARD_CONSECUTIVE_BATCHES consecutive
batches aborts the run fail-loud. It deliberately does NOT adjudicate
correctness (k=3 concentration is invisible to it — drl review MAJOR-2);
G3 + dcorr(reward, J) do that at scoring time.
"""

from esper.leyline import (
    FOSSILIZE_RATE_GUARD_CONSECUTIVE_BATCHES,
    FOSSILIZE_RATE_GUARD_TRIP_COUNT,
)
from esper.simic.training.fossilize_rate_guard import FossilizeRateGuard


def test_constants_pinned_to_review_values():
    # 5x the banked control baseline (0.207 fossilize/ep * 12 env-episodes
    # = 12.42/batch) pinned to the integer ABOVE it (drl review NOTE-6).
    assert FOSSILIZE_RATE_GUARD_TRIP_COUNT == 13
    assert FOSSILIZE_RATE_GUARD_CONSECUTIVE_BATCHES == 2


def test_below_threshold_never_trips():
    guard = FossilizeRateGuard()
    for _ in range(50):
        assert guard.observe_batch(12) is False  # 12 < 13: under the line


def test_single_batch_spike_does_not_trip():
    guard = FossilizeRateGuard()
    assert guard.observe_batch(36) is False  # structural max, one batch only


def test_two_consecutive_over_threshold_batches_trip():
    guard = FossilizeRateGuard()
    assert guard.observe_batch(13) is False
    assert guard.observe_batch(13) is True


def test_calm_batch_resets_the_consecutive_counter():
    guard = FossilizeRateGuard()
    assert guard.observe_batch(20) is False
    assert guard.observe_batch(3) is False  # reset
    assert guard.observe_batch(20) is False  # counting from 1 again
    assert guard.observe_batch(20) is True


def test_consecutive_count_exposed_for_telemetry():
    guard = FossilizeRateGuard()
    guard.observe_batch(15)
    assert guard.consecutive == 1
    guard.observe_batch(15)
    assert guard.consecutive == 2
    guard.observe_batch(0)
    assert guard.consecutive == 0
