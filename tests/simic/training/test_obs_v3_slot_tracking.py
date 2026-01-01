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


def test_init_obs_v3_slot_tracking_overwrites_stale_values() -> None:
    env_state = _make_env_state()
    env_state.gradient_health_prev["r0c0"] = 0.2
    env_state.epochs_since_counterfactual["r0c0"] = 50

    env_state.init_obs_v3_slot_tracking("r0c0")

    assert env_state.gradient_health_prev["r0c0"] == 1.0
    assert env_state.epochs_since_counterfactual["r0c0"] == 0


def test_clear_obs_v3_slot_tracking_removes_keys() -> None:
    env_state = _make_env_state()
    env_state.gradient_health_prev["r0c0"] = 0.9
    env_state.epochs_since_counterfactual["r0c0"] = 7

    env_state.clear_obs_v3_slot_tracking("r0c0")

    assert "r0c0" not in env_state.gradient_health_prev
    assert "r0c0" not in env_state.epochs_since_counterfactual
