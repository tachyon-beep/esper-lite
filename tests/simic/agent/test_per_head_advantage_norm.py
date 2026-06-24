"""End-to-end wiring for per-head advantage normalization.

The numerics (unit-std on the active subset, fp16 safety, low-count fallback) are
covered by ``tests/simic/test_advantages.py``. This module checks the live PPO update
path: that ``per_head_advantage_norm`` is read by the agent, that
``head_advantage_norm_stats`` flows through the metrics builder, and that the update
stays finite — including the low-count fallback (a 4-step buffer keeps every head below
MIN_HEAD_NORM_COUNT, so the guard must send them all back to the global scale).
"""

from __future__ import annotations

from esper.leyline import HEAD_NAMES, MIN_HEAD_NORM_COUNT

from tests.simic.agent.test_q_finiteness_and_contract import _build_agent, _fill_buffer

_STAT_KEYS = {"pre_norm_std", "post_norm_std", "n_active", "fellback", "normalized"}


def _run_update(per_head_advantage_norm: bool) -> dict:
    agent, slot_config = _build_agent()
    # The flag is a plain attribute; flip it post-construction to reuse the helper.
    agent.per_head_advantage_norm = per_head_advantage_norm
    _fill_buffer(agent, slot_config)
    return agent.update()


def test_stats_present_and_finite_with_flag_on():
    metrics = _run_update(per_head_advantage_norm=True)

    assert metrics["ppo_update_performed"] is True
    stats = metrics["head_advantage_norm_stats"]
    assert set(stats.keys()) == set(HEAD_NAMES)
    for head, s in stats.items():
        assert set(s.keys()) == _STAT_KEYS, f"{head} stat keys mismatch"
        for key, value in s.items():
            assert value == value, f"{head}.{key} is NaN"  # NaN != NaN
            assert abs(value) != float("inf"), f"{head}.{key} is Inf"


def test_low_count_buffer_falls_back_to_global():
    # The fixture buffer has 4 steps << MIN_HEAD_NORM_COUNT, so even with the ablation
    # ON every head must fall back to the global scale (never renormalize on noise).
    assert MIN_HEAD_NORM_COUNT > 4
    stats = _run_update(per_head_advantage_norm=True)["head_advantage_norm_stats"]
    for head, s in stats.items():
        assert s["normalized"] == 0.0, f"{head} renormalized below the count guard"
        assert s["fellback"] == 1.0, f"{head} did not record the fallback"


def test_flag_off_emits_stats_without_fallback_semantics():
    # Observability-first: stats are emitted even with the ablation OFF, and the
    # fallback flag is meaningless there (nothing was ever attempted per-head).
    stats = _run_update(per_head_advantage_norm=False)["head_advantage_norm_stats"]
    assert set(stats.keys()) == set(HEAD_NAMES)
    for head, s in stats.items():
        assert s["normalized"] == 0.0
        assert s["fellback"] == 0.0
