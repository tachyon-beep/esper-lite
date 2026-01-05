"""Property-based tests for Simic ParallelEnvState invariants."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.leyline import LifecycleOp
from esper.leyline.stages import SeedStage
from esper.simic.training.parallel_env_state import ParallelEnvState

pytestmark = pytest.mark.property


@dataclass(slots=True)
class _StateStub:
    stage: SeedStage


@dataclass(slots=True)
class _SlotStub:
    seed: object | None = None
    state: _StateStub | None = None


@dataclass(slots=True)
class _ModelStub:
    seed_slots: dict[str, _SlotStub]


class _ResetCounter:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls += 1


def _make_env(slots: list[str]) -> tuple[ParallelEnvState, _ModelStub, _ResetCounter, _ResetCounter]:
    model = _ModelStub(seed_slots={slot_id: _SlotStub() for slot_id in slots})
    tracker = _ResetCounter()
    governor = _ResetCounter()
    env = ParallelEnvState(
        model=model,  # type: ignore[arg-type]
        host_optimizer=object(),  # type: ignore[arg-type]
        signal_tracker=tracker,  # type: ignore[arg-type]
        governor=governor,  # type: ignore[arg-type]
        env_device="cpu",
    )
    return env, model, tracker, governor


@given(n_slots=st.integers(min_value=1, max_value=4))
@settings(max_examples=50)
def test_reset_episode_state_is_idempotent_and_clears_obs_v3_tracking(n_slots: int) -> None:
    slots = [f"slot_{i}" for i in range(n_slots)]
    env, _, tracker, governor = _make_env(slots)

    env.reset_episode_state(slots)

    assert env.last_action_success is True
    assert env.last_action_op == LifecycleOp.WAIT.value
    assert env.gradient_health_prev == {}
    assert env.epochs_since_counterfactual == {}
    assert env.escrow_credit == {slot_id: 0.0 for slot_id in slots}
    assert tracker.calls == 1
    assert governor.calls == 1

    env.last_action_success = False
    env.last_action_op = LifecycleOp.PRUNE.value
    env.gradient_health_prev = {slots[0]: 0.5}
    env.epochs_since_counterfactual = {slots[0]: 3}

    assert env.train_loss_accum is not None
    env.train_loss_accum.fill_(3.0)
    env.cf_totals[slots[0]] = 7
    env.cf_correct_accums[slots[0]].fill_(2.0)

    env.reset_episode_state(slots)

    assert env.last_action_success is True
    assert env.last_action_op == LifecycleOp.WAIT.value
    assert env.gradient_health_prev == {}
    assert env.epochs_since_counterfactual == {}
    assert tracker.calls == 2
    assert governor.calls == 2

    assert env.train_loss_accum is not None
    assert env.train_loss_accum.item() == pytest.approx(0.0)
    assert all(total == 0 for total in env.cf_totals.values())
    assert env.cf_correct_accums[slots[0]].item() == pytest.approx(0.0)


@given(
    n_slots=st.integers(min_value=1, max_value=4),
    bad_idx=st.integers(min_value=0, max_value=3),
    set_seed=st.booleans(),
)
@settings(max_examples=40)
def test_reset_episode_state_rejects_non_dormant_slots(
    n_slots: int, bad_idx: int, set_seed: bool
) -> None:
    slots = [f"slot_{i}" for i in range(n_slots)]
    env, model, _, _ = _make_env(slots)

    bad_idx = min(bad_idx, n_slots - 1)
    bad_slot_id = slots[bad_idx]

    if set_seed:
        model.seed_slots[bad_slot_id].seed = object()
    else:
        model.seed_slots[bad_slot_id].state = _StateStub(stage=SeedStage.TRAINING)

    with pytest.raises(RuntimeError, match="Episode reset expected DORMANT slots"):
        env.reset_episode_state(slots)


@st.composite
def _slot_tracking_cases(draw: st.DrawFn):
    slots = draw(
        st.lists(st.sampled_from([f"slot_{i}" for i in range(4)]), min_size=1, max_size=4, unique=True)
    )
    init_slots = draw(st.lists(st.sampled_from(slots), min_size=0, max_size=len(slots), unique=True))
    clear_slots = draw(st.lists(st.sampled_from(slots), min_size=0, max_size=len(slots), unique=True))
    return slots, init_slots, clear_slots


@given(case=_slot_tracking_cases())
@settings(max_examples=60)
def test_obs_v3_slot_tracking_init_and_clear(case) -> None:
    slots, init_slots, clear_slots = case
    env, _, _, _ = _make_env(slots)

    for slot_id in init_slots:
        env.init_obs_v3_slot_tracking(slot_id)

    for slot_id in clear_slots:
        env.clear_obs_v3_slot_tracking(slot_id)

    for slot_id in slots:
        should_exist = slot_id in init_slots and slot_id not in clear_slots
        if should_exist:
            assert env.gradient_health_prev[slot_id] == pytest.approx(1.0)
            assert env.epochs_since_counterfactual[slot_id] == 0
        else:
            assert slot_id not in env.gradient_health_prev
            assert slot_id not in env.epochs_since_counterfactual

    assert set(env.gradient_health_prev.keys()).issubset(set(slots))
    assert set(env.epochs_since_counterfactual.keys()).issubset(set(slots))
