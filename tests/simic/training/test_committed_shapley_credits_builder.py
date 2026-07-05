"""Episode-identity contract for the Committed-Shapley credits builder (esper-lite-e780981fe7).

The trainer-side populator (extracted from ``_run_fused_val_pass`` as
``build_committed_shapley_env_credits`` so it is testable without the full GPU
trainer — same pattern as test_committed_shapley_pp_units.py: behavioral tests
on the extracted helper + source canary on the in-place wiring).

The bug: the populator stamped ``episode_idx=batch_idx`` (one flat per-batch
value) while the codebase convention for episode identity is
``episodes_completed + env_idx`` (set on every env's episode context and
emitter at batch start). With num_envs > 1, every COMMITTED_SHAPLEY_TOPUP
event after batch 0 carried the batch number instead of the episode id, so
per-episode joins misattached.

CPU-only, no CUDA requirement.
"""

import inspect

import pytest
import torch
import torch.nn as nn

from esper.simic.training.parallel_env_state import ParallelEnvState
from esper.simic.training.vectorized_trainer import (
    VectorizedPPOTrainer,
    build_committed_shapley_env_credits,
)

# Frozen F2 knobs (PDR-0020) — values irrelevant to the identity contract,
# realistic for the payment math.
_SCALE, _CAP, _TAU = 1.0, 5.0, 0.28


def _fossilized_slot(slot_id: str) -> "SeedSlot":
    """Real SeedSlot walked to FOSSILIZED (the kasmina test idiom, ADD blend)."""
    from esper.kasmina.slot import SeedSlot
    from esper.leyline import AlphaAlgorithm, SeedStage

    slot = SeedSlot(slot_id=slot_id, channels=8)
    slot.germinate(
        "norm", seed_id=f"s-{slot_id}", alpha_algorithm=AlphaAlgorithm.ADD
    )
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.state.transition(SeedStage.HOLDING)
    slot.state.stage = SeedStage.FOSSILIZED
    return slot


def _env_state(records: list[tuple[str, int]]) -> ParallelEnvState:
    """Minimal CPU ParallelEnvState; the builder reads fossilize_step_records
    and each committed slot's seed state (for the alpha-algorithm map)."""
    model = nn.Linear(2, 2)
    model.seed_slots = {sid: _fossilized_slot(sid) for sid, _ in records}
    state = ParallelEnvState(
        model=model,
        host_optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        signal_tracker=None,
        governor=None,
        env_device="cpu",
        stream=None,
    )
    state.fossilize_step_records.extend(records)
    return state


def _k1_table(slot: str) -> dict[frozenset[str], float]:
    """Complete 2^1 coalition table for a single committed slot."""
    return {frozenset(): 50.0, frozenset({slot}): 55.0}


def test_episode_idx_is_episodes_completed_plus_env_idx() -> None:
    """Each env's credits carry ITS episode id, not one flat per-batch value."""
    env_states = [
        _env_state([("r0c0", 3)]),
        _env_state([("r0c1", 7)]),
    ]
    credits = build_committed_shapley_env_credits(
        {0: _k1_table("r0c0"), 1: _k1_table("r0c1")},
        {0: ("r0c0",), 1: ("r0c1",)},
        env_states,
        scale=_SCALE,
        cap=_CAP,
        tau=_TAU,
        episodes_completed=8,
    )
    by_env = {c.env_idx: c for c in credits}
    assert set(by_env) == {0, 1}
    # The gate-stratification map is populated from real slot state.
    assert by_env[0].alpha_algorithm_by_slot == {"r0c0": "ADD"}
    assert by_env[0].episode_idx == 8
    assert by_env[1].episode_idx == 9, (
        "episode identity must be episodes_completed + env_idx (the batch-start "
        "convention) — a flat value here is the esper-lite-e780981fe7 regression"
    )


def test_t_f_mapping_filters_to_committed_set_and_carries_table() -> None:
    """t_f_by_slot keeps only committed slots; the raw v(S) table rides along."""
    table = _k1_table("r0c0")
    credits = build_committed_shapley_env_credits(
        {0: table},
        {0: ("r0c0",)},
        # A non-committed slot's record must not leak into t_f_by_slot.
        [_env_state([("r0c0", 3), ("r0c1", 5)])],
        scale=_SCALE,
        cap=_CAP,
        tau=_TAU,
        episodes_completed=0,
    )
    assert len(credits) == 1
    assert credits[0].t_f_by_slot == {"r0c0": 3}
    assert credits[0].coalition_accs is table


def test_missing_t_f_record_raises() -> None:
    """A committed slot without an execution-time t_f record is a bug, not skippable."""
    with pytest.raises(ValueError, match="r0c0"):
        build_committed_shapley_env_credits(
            {0: _k1_table("r0c0")},
            {0: ("r0c0",)},
            [_env_state([])],
            scale=_SCALE,
            cap=_CAP,
            tau=_TAU,
            episodes_completed=0,
        )


def test_trainer_wiring_threads_episodes_completed_not_batch_idx() -> None:
    """Source canary on the in-place wiring the behavioral tests cannot reach.

    Pins that ``_run_fused_val_pass`` delegates to the builder with the real
    episode counter, and that the flat ``episode_idx=batch_idx`` stamp is gone.
    Fails LOUDLY if the delegation is bypassed or the counter is un-threaded.
    """
    source = inspect.getsource(VectorizedPPOTrainer._run_fused_val_pass)
    collapsed = "".join(source.split())  # whitespace-insensitive

    assert "episode_idx=batch_idx" not in collapsed, (
        "the flat batch-index episode stamp is the esper-lite-e780981fe7 bug; "
        "episode identity must come from episodes_completed + env_idx"
    )
    assert "build_committed_shapley_env_credits(" in collapsed, (
        "the credits populator must delegate to the extracted builder so the "
        "episode-identity contract stays behaviorally tested"
    )
    assert "episodes_completed=episodes_completed" in collapsed, (
        "the builder must receive the real pre-batch episode counter"
    )
