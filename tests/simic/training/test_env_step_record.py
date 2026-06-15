"""Unit tests for EnvStepRecord — type contract and reset discipline."""
from __future__ import annotations

from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
    EnvStepRecord,
    RewardSummaryAccumulator,
)


def _make_record(env_idx: int = 0) -> EnvStepRecord:
    from types import SimpleNamespace
    return EnvStepRecord(
        env_idx=env_idx,
        action_spec=ActionSpec(),
        action_outcome=ActionOutcome(),
        mask_flags=ActionMaskFlags(),
        reward_summary=RewardSummaryAccumulator(),
        contribution_reward_inputs=SimpleNamespace(),
        loss_reward_inputs=SimpleNamespace(),
    )


def test_env_step_record_has_slots() -> None:
    """EnvStepRecord must be a slots dataclass (no __dict__)."""
    r = _make_record()
    assert not hasattr(r, "__dict__"), "slots=True dataclass must not have __dict__"


def test_env_step_record_composed_objects_are_same_references() -> None:
    """Composition: the ActionSpec inside the record is the exact object passed in."""
    spec = ActionSpec()
    spec.slot_idx = 7
    from types import SimpleNamespace
    r = EnvStepRecord(
        env_idx=0,
        action_spec=spec,
        action_outcome=ActionOutcome(),
        mask_flags=ActionMaskFlags(),
        reward_summary=RewardSummaryAccumulator(),
        contribution_reward_inputs=SimpleNamespace(),
        loss_reward_inputs=SimpleNamespace(),
    )
    assert r.action_spec is spec, "action_spec must be the same object (no copy)"
    r.action_spec.slot_idx = 99
    assert spec.slot_idx == 99, "mutation through record must be visible on original object"


def test_env_step_record_reset_episode_zeroes_episode_fields() -> None:
    """reset_episode() must zero all episode-level accumulators."""
    r = _make_record()
    r.rollback_occurred = True
    r.env_final_acc = 0.87
    r.env_total_reward = 42.0
    r.reset_episode()
    assert r.rollback_occurred is False
    assert r.env_final_acc == 0.0
    assert r.env_total_reward == 0.0


def test_env_step_record_reset_episode_does_not_touch_composed_objects() -> None:
    """reset_episode() must NOT reset ActionSpec/ActionOutcome fields."""
    r = _make_record()
    r.action_spec.slot_idx = 5
    r.action_outcome.reward_raw = 3.14
    r.reset_episode()
    # Composed objects are managed separately (reset via in-place mutation in execute_actions)
    assert r.action_spec.slot_idx == 5
    assert r.action_outcome.reward_raw == 3.14
