"""Committed-Shapley retro-write delivery (WI-4).

The credit is written into ``buffer.rewards[env, t_f]`` at the pre-GAE seam
(GATE 2 mandate) via divide_by_std semantics: NO clip, NO normalizer-stat
update. F2 bounds (reviewer-mandated): normalized-space cap is the PRIMARY
bound; the divisor is floored at ``max(std, std_floor)``. Rollback-forfeited
envs are excluded (the exclusion set equals the forfeit set:
``mark_terminal_with_penalty`` is what increments ``buffer.rollback_count``).
"""

from __future__ import annotations

import pytest
import torch

from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
from esper.simic.control import RewardNormalizer
from esper.simic.rewards.committed_shapley import (
    CommittedShapleyEnvCredits,
    CommittedShapleyResult,
    SlotTopUp,
    compute_committed_shapley_topup,
)
from esper.simic.training.ppo_coordinator import apply_committed_shapley_credits


def _buffer(num_envs=2, steps=6):
    b = TamiyoRolloutBuffer(num_envs=num_envs, max_steps_per_env=steps, state_dim=8)
    for env in range(num_envs):
        b.step_counts[env] = steps
    return b


def _normalizer(samples=(2.0, 4.0, 6.0, 8.0)):
    n = RewardNormalizer(clip=10.0)
    for s in samples:
        n.update_and_normalize(s)
    return n


def _env_credits(env_idx, top_ups: dict[str, float], t_f: dict[str, int]):
    per_slot = {
        slot: SlotTopUp(phi=0.0, c_paid=0.0, gap=up, raw=up, top_up=up)
        for slot, up in top_ups.items()
    }
    result = CommittedShapleyResult(
        per_slot=per_slot,
        g=sum(top_ups.values()),
        sum_raw=sum(top_ups.values()),
        clamp_binding=False,
        k=len(top_ups),
    )
    return CommittedShapleyEnvCredits(
        env_idx=env_idx, result=result, t_f_by_slot=t_f, episode_idx=1
    )


def test_credit_written_at_t_f_divided_by_std():
    buffer = _buffer()
    normalizer = _normalizer()
    std = normalizer.current_std()
    credits = [_env_credits(0, {"r0c0": 2.0}, {"r0c0": 3})]

    payloads = apply_committed_shapley_credits(
        buffer=buffer,
        reward_normalizer=normalizer,
        env_credits=credits,
        std_floor=0.0,
        normalized_cap=100.0,
    )

    expected = 2.0 / std
    assert buffer.rewards[0, 3].item() == pytest.approx(expected)
    # Every other cell untouched.
    mask = torch.ones_like(buffer.rewards, dtype=torch.bool)
    mask[0, 3] = False
    assert torch.all(buffer.rewards[mask] == 0.0)
    # No normalizer stat update.
    assert normalizer.count == 4
    # Payload books balance.
    assert len(payloads) == 1
    assert payloads[0].credit_buf[0] == pytest.approx(expected)
    assert payloads[0].std_used == pytest.approx(std)
    assert payloads[0].dropped_no_std is False


def test_std_floor_bounds_small_std_amplification():
    """F2 (pytorch F1 / drl F-A): std < std_floor => divisor is std_floor."""
    buffer = _buffer()
    normalizer = _normalizer(samples=(1.0, 1.01, 1.02, 0.99))  # tiny std
    std = normalizer.current_std()
    std_floor = 0.5
    assert std < std_floor  # the scenario under test
    credits = [_env_credits(0, {"r0c0": 2.0}, {"r0c0": 1})]

    apply_committed_shapley_credits(
        buffer=buffer,
        reward_normalizer=normalizer,
        env_credits=credits,
        std_floor=std_floor,
        normalized_cap=100.0,
    )
    assert buffer.rewards[0, 1].item() == pytest.approx(2.0 / std_floor)


def test_normalized_cap_is_the_primary_bound():
    buffer = _buffer()
    normalizer = _normalizer(samples=(1.0, 1.01, 1.02, 0.99))  # tiny std
    credits = [_env_credits(0, {"r0c0": 2.0}, {"r0c0": 1})]

    payloads = apply_committed_shapley_credits(
        buffer=buffer,
        reward_normalizer=normalizer,
        env_credits=credits,
        std_floor=0.0,
        normalized_cap=5.0,
    )
    # 2.0 / tiny-std >> 5.0 -> clamped to the normalized cap.
    assert buffer.rewards[0, 1].item() == pytest.approx(5.0)
    assert payloads[0].credit_buf[0] == pytest.approx(5.0)


def test_rollback_env_excluded_entirely():
    buffer = _buffer(num_envs=2)
    buffer.rollback_count[1] = 1  # env 1 forfeited by mark_terminal_with_penalty
    normalizer = _normalizer()
    credits = [
        _env_credits(0, {"r0c0": 2.0}, {"r0c0": 3}),
        _env_credits(1, {"r0c0": 2.0}, {"r0c0": 3}),
    ]

    payloads = apply_committed_shapley_credits(
        buffer=buffer,
        reward_normalizer=normalizer,
        env_credits=credits,
        std_floor=0.0,
        normalized_cap=100.0,
    )
    assert buffer.rewards[0, 3].item() != 0.0
    assert buffer.rewards[1, 3].item() == 0.0
    assert len(payloads) == 1  # no event for the excluded env
    assert payloads[0].env_id == 0


def test_zero_top_up_slots_not_written():
    buffer = _buffer()
    normalizer = _normalizer()
    credits = [_env_credits(0, {"r0c0": 2.0, "r0c1": 0.0}, {"r0c0": 3, "r0c1": 4})]

    payloads = apply_committed_shapley_credits(
        buffer=buffer,
        reward_normalizer=normalizer,
        env_credits=credits,
        std_floor=0.0,
        normalized_cap=100.0,
    )
    assert buffer.rewards[0, 4].item() == 0.0
    # The event still carries the zero slot (books complete)...
    assert payloads[0].slot_ids == ("r0c0", "r0c1")
    assert payloads[0].credit_buf[1] == 0.0


def test_count_below_two_drops_credits_and_marks_payload():
    buffer = _buffer()
    normalizer = RewardNormalizer(clip=10.0)  # zero samples -> no std
    credits = [_env_credits(0, {"r0c0": 2.0}, {"r0c0": 3})]

    payloads = apply_committed_shapley_credits(
        buffer=buffer,
        reward_normalizer=normalizer,
        env_credits=credits,
        std_floor=0.5,
        normalized_cap=100.0,
    )
    assert torch.all(buffer.rewards == 0.0)  # nothing written
    assert payloads[0].dropped_no_std is True
    assert payloads[0].credit_buf == (0.0,)


def test_t_f_out_of_range_fails_loud():
    buffer = _buffer(steps=4)
    normalizer = _normalizer()
    credits = [_env_credits(0, {"r0c0": 2.0}, {"r0c0": 9})]

    with pytest.raises(ValueError, match="t_f"):
        apply_committed_shapley_credits(
            buffer=buffer,
            reward_normalizer=normalizer,
            env_credits=credits,
            std_floor=0.0,
            normalized_cap=100.0,
        )


def test_end_to_end_with_real_shapley_result():
    """Wire the actual WI-1 math through delivery (no hand-built SlotTopUp)."""
    accs = {
        frozenset(): 50.0,
        frozenset({"r0c0"}): 60.0,
        frozenset({"r0c1"}): 55.0,
        frozenset({"r0c0", "r0c1"}): 70.0,
    }
    result = compute_committed_shapley_topup(
        accs, ("r0c0", "r0c1"), scale=1.0, cap=10.0, tau=0.0
    )
    buffer = _buffer()
    normalizer = _normalizer()
    std = normalizer.current_std()
    credits = [
        CommittedShapleyEnvCredits(
            env_idx=0,
            result=result,
            t_f_by_slot={"r0c0": 2, "r0c1": 5},
            episode_idx=0,
        )
    ]
    payloads = apply_committed_shapley_credits(
        buffer=buffer,
        reward_normalizer=normalizer,
        env_credits=credits,
        std_floor=0.0,
        normalized_cap=100.0,
    )
    # top_up = 2.5 pp each (hand case from WI-1 tests).
    assert buffer.rewards[0, 2].item() == pytest.approx(2.5 / std)
    assert buffer.rewards[0, 5].item() == pytest.approx(2.5 / std)
    assert payloads[0].g == pytest.approx(20.0)
