from __future__ import annotations

from esper.simic.training.parallel_env_state import ParallelEnvState


class _MockModel:
    pass


class _MockOptimizer:
    pass


class _MockSignalTracker:
    def reset(self) -> None:
        pass


class _MockGovernor:
    def reset(self) -> None:
        pass


def _make_env_state() -> ParallelEnvState:
    return ParallelEnvState(
        model=_MockModel(),
        host_optimizer=_MockOptimizer(),
        signal_tracker=_MockSignalTracker(),
        governor=_MockGovernor(),
    )


def test_init_obs_v3_slot_tracking_clears_to_unknown() -> None:
    """A newly germinated seed must encode UNKNOWN (no measured evidence),
    NOT healthy/fresh. Tracking entries are therefore absent until a real
    reading arrives, and germinating into a recycled slot id clears any
    stale entry left by the prior occupant (TPD-003).
    """
    env_state = _make_env_state()
    # Stale values from a previous occupant of the same slot id.
    env_state.gradient_health_prev["r0c0"] = 0.2
    env_state.epochs_since_counterfactual["r0c0"] = 50

    env_state.init_obs_v3_slot_tracking("r0c0")

    # UNKNOWN == absence (NOT 1.0 healthy / 0 fresh).
    assert "r0c0" not in env_state.gradient_health_prev
    assert "r0c0" not in env_state.epochs_since_counterfactual


def test_init_obs_v3_slot_tracking_on_fresh_slot_leaves_unknown() -> None:
    """Germinating into a never-used slot leaves it UNKNOWN (no entry)."""
    env_state = _make_env_state()

    env_state.init_obs_v3_slot_tracking("r1c1")

    assert "r1c1" not in env_state.gradient_health_prev
    assert "r1c1" not in env_state.epochs_since_counterfactual


def test_clear_obs_v3_slot_tracking_removes_keys() -> None:
    env_state = _make_env_state()
    env_state.gradient_health_prev["r0c0"] = 0.9
    env_state.epochs_since_counterfactual["r0c0"] = 7

    env_state.clear_obs_v3_slot_tracking("r0c0")

    assert "r0c0" not in env_state.gradient_health_prev
    assert "r0c0" not in env_state.epochs_since_counterfactual
