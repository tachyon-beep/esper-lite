"""Tests for P0-3: governor-rollback credit assignment + telemetry.

Pins the deliberate P1-ROLLBACK-TAIL credit-assignment policy: a governor
rollback FORFEITS the entire episode prefix (intermediate rewards are zeroed to
the named ``ROLLBACK_FORFEIT_REWARD`` constant) because the host weights were
restored to the last-good snapshot and the "mostly-good" trajectory was
discarded by the environment. Crediting that prefix would teach
"germinate aggressively then panic" as net-positive (reward-hacking the safety
net); with gamma < 1 a long positive prefix can outweigh one terminal penalty.

Also pins the two observability counters (``rollback_count`` and
``rollback_steps_zeroed``) so frequent rollbacks cannot silently starve PPO of
learning signal without being visible in telemetry.
"""

from __future__ import annotations

import torch

from esper.leyline import ROLLBACK_FORFEIT_REWARD
from esper.simic.agent.rollout_buffer import (
    RollbackPenaltyResult,
    TamiyoRolloutBuffer,
)


def _add_steps(buffer: TamiyoRolloutBuffer, env_id: int, n_steps: int) -> None:
    """Add ``n_steps`` minimal transitions to ``env_id`` with reward 1.0 each."""
    num_slots = buffer.num_slots
    for step_idx in range(n_steps):
        buffer.add(
            env_id=env_id,
            state=torch.zeros(buffer.state_dim),
            blueprint_indices=torch.zeros(num_slots, dtype=torch.long),
            slot_action=0,
            blueprint_action=0,
            style_action=0,
            tempo_action=0,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=0,
            effective_op_action=0,
            slot_log_prob=0.0,
            blueprint_log_prob=0.0,
            style_log_prob=0.0,
            tempo_log_prob=0.0,
            alpha_target_log_prob=0.0,
            alpha_speed_log_prob=0.0,
            alpha_curve_log_prob=0.0,
            op_log_prob=0.0,
            value=0.0,
            reward=1.0,
            done=False,
            slot_mask=torch.ones(num_slots, dtype=torch.bool),
            blueprint_mask=torch.ones(buffer.num_blueprints, dtype=torch.bool),
            style_mask=torch.ones(buffer.num_styles, dtype=torch.bool),
            tempo_mask=torch.ones(buffer.num_tempo, dtype=torch.bool),
            alpha_target_mask=torch.ones(buffer.num_alpha_targets, dtype=torch.bool),
            alpha_speed_mask=torch.ones(buffer.num_alpha_speeds, dtype=torch.bool),
            alpha_curve_mask=torch.ones(buffer.num_alpha_curves, dtype=torch.bool),
            op_mask=torch.ones(buffer.num_ops, dtype=torch.bool),
            hidden_h=torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
            hidden_c=torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
            action_id=f"test-env{env_id}-step{step_idx}",
        )


def _make_buffer(num_envs: int = 2, max_steps: int = 60) -> TamiyoRolloutBuffer:
    return TamiyoRolloutBuffer(
        num_envs=num_envs,
        max_steps_per_env=max_steps,
        state_dim=16,
    )


def test_rollback_forfeit_reward_constant_lives_in_leyline() -> None:
    """ROLLBACK_FORFEIT_REWARD is the single source of truth for the forfeit fill.

    Both mutation sites (buffer + coordinator) must REFERENCE the named constant,
    not a literal 0.0. We assert the constant is importable and has the intended
    value, and that the buffer/coordinator source contains no literal 0.0 at the
    forfeit-fill line (guards against silent re-introduction of the magic number).
    """
    import inspect

    from esper.simic.agent import rollout_buffer as rb
    from esper.simic.training import ppo_coordinator as coord

    assert ROLLBACK_FORFEIT_REWARD == 0.0
    assert isinstance(ROLLBACK_FORFEIT_REWARD, float)

    buffer_src = inspect.getsource(rb.TamiyoRolloutBuffer.mark_terminal_with_penalty)
    assert "ROLLBACK_FORFEIT_REWARD" in buffer_src
    # The forfeit-fill assignment must not hard-code the literal.
    assert "] = 0.0" not in buffer_src

    coord_src = inspect.getsource(coord.PPOCoordinator.handle_rollbacks)
    assert "ROLLBACK_FORFEIT_REWARD" in coord_src
    assert "= 0.0" not in coord_src


def test_mark_terminal_with_penalty_zeroes_full_episode_prefix() -> None:
    """Long episode + late rollback forfeits the WHOLE prefix to the constant."""
    buffer = _make_buffer(max_steps=60)
    buffer.start_episode(0)
    _add_steps(buffer, 0, 50)
    buffer.end_episode(0)

    buffer.mark_terminal_with_penalty(0, penalty=-10.0)

    last = 49
    # Every intermediate reward forfeited to the named constant.
    assert torch.allclose(
        buffer.rewards[0, 0:last],
        torch.full((last,), float(ROLLBACK_FORFEIT_REWARD)),
    )
    # Terminal step carries the penalty (NOT the forfeit fill).
    assert buffer.rewards[0, last].item() == -10.0
    assert buffer.dones[0, last].item() is True


def test_mark_terminal_with_penalty_increments_rollback_count() -> None:
    """Two rollbacks on the same env accumulate rollback_count to 2."""
    buffer = _make_buffer()
    buffer.start_episode(0)
    _add_steps(buffer, 0, 5)
    buffer.end_episode(0)
    buffer.mark_terminal_with_penalty(0, penalty=-10.0)

    # Second episode on the same env, then a second rollback.
    buffer.start_episode(0)
    _add_steps(buffer, 0, 4)
    buffer.end_episode(0)
    buffer.mark_terminal_with_penalty(0, penalty=-10.0)

    assert buffer.rollback_count[0].item() == 2
    # The other env never rolled back.
    assert buffer.rollback_count[1].item() == 0


def test_mark_terminal_with_penalty_steps_zeroed_counter_correct() -> None:
    """Episode length L → rollback_steps_zeroed == L-1 and result.steps_zeroed == L-1."""
    buffer = _make_buffer()
    episode_len = 7
    buffer.start_episode(0)
    _add_steps(buffer, 0, episode_len)
    buffer.end_episode(0)

    result = buffer.mark_terminal_with_penalty(0, penalty=-10.0)

    assert isinstance(result, RollbackPenaltyResult)
    assert result.applied is True
    assert result.steps_zeroed == episode_len - 1
    assert buffer.rollback_steps_zeroed[0].item() == episode_len - 1


def test_mark_terminal_with_penalty_first_step_panic_no_zeroing() -> None:
    """step_count == 1 → nothing to zero, but the rollback ATTEMPT is still counted.

    A first-step panic forfeits no prefix (there is no prefix), but the rollback
    EVENT must still register so a high first-step-panic rate is observable.
    """
    buffer = _make_buffer()
    buffer.start_episode(0)
    _add_steps(buffer, 0, 1)
    buffer.end_episode(0)

    result = buffer.mark_terminal_with_penalty(0, penalty=-10.0)

    assert result.applied is True
    assert result.steps_zeroed == 0
    assert buffer.rollback_steps_zeroed[0].item() == 0
    # The rollback attempt is counted even though no prefix was forfeited.
    assert buffer.rollback_count[0].item() == 1
    # Terminal step still carries the penalty.
    assert buffer.rewards[0, 0].item() == -10.0


def test_rollback_counters_reset_across_rollouts() -> None:
    """reset() zeroes the cumulative rollback counters (per-rollout window)."""
    buffer = _make_buffer()
    buffer.start_episode(0)
    _add_steps(buffer, 0, 5)
    buffer.end_episode(0)
    buffer.mark_terminal_with_penalty(0, penalty=-10.0)

    assert buffer.rollback_count[0].item() == 1
    assert buffer.rollback_steps_zeroed[0].item() == 4

    buffer.reset()

    assert torch.all(buffer.rollback_count == 0)
    assert torch.all(buffer.rollback_steps_zeroed == 0)


def test_rollback_steps_zeroed_surfaced_in_snapshot() -> None:
    """Both counters appear in the get_batched_sequences snapshot dict."""
    buffer = _make_buffer()
    buffer.start_episode(0)
    _add_steps(buffer, 0, 6)
    buffer.end_episode(0)
    buffer.mark_terminal_with_penalty(0, penalty=-10.0)

    batch = buffer.get_batched_sequences()

    assert "rollback_count" in batch
    assert "rollback_steps_zeroed" in batch
    assert batch["rollback_count"][0].item() == 1
    assert batch["rollback_steps_zeroed"][0].item() == 5
