"""Pin the driver's LIVE-convention sequencing against the real kasmina state machine.

SCOPE OF PROOF (review nit #1, 2026-07-14): this test establishes that GIVEN
the trainer's phase ordering, the real ``SeedState``/``SeedMetrics`` mechanics
(transition() previous_* recording, clock reset, record_accuracy ticking)
produce exactly the SeedInfo sequence the driver generates. The ordering
ITSELF is not proven here - the walker mirrors it, so driver and walker would
err together if it were wrong. The ordering rests on the code citations below,
independently re-verified in the Phase-1 review (drl-expert, APPROVE-WITH-NITS)
and confirmed empirically by the r9 population signature (0/4,151 commits pay
the forfeiture; the -0.012/+0.078 fossil histogram). A one-epoch real-trainer
integration test is the Phase-2 hardening option if a stronger pin is wanted.

Trainer ordering being mirrored (with the source of each step):
1. validation phase ticks metrics via ``record_accuracy`` for every slot with
   a non-None state - fossils included (vectorized_trainer.py:1870-1875);
2. the reward for the decision is computed on the PRE-action state
   (action_execution.py:226-230 capture, :973-977 SeedInfo, :1089 reward);
3. the lifecycle mutation executes AFTER the reward (action_execution.py
   dispatch phase, ~:1304), calling ``SeedState.transition`` which records
   previous_* and zeroes epochs_in_current_stage (slot.py:520-531);
4. ``step_epoch`` runs at the end of the action transaction
   (action_execution.py:1643).

Under this ordering the first post-transition reward row sees
``epochs_in_stage == 1`` - the PBRS ``eis == 0`` branch (contribution.py:1235)
is unreachable for policy-initiated transitions. That fact is load-bearing for
the Phase-1 findings, so it is pinned HERE against kasmina itself, not merely
assumed by the driver.
"""

from __future__ import annotations

from esper.kasmina.slot import SeedState
from esper.leyline import SeedStage
from esper.simic.rewards import SeedInfo

from .path_driver import PathScenario, replay_path


def _observed_sequence(scenario: PathScenario) -> list[tuple[int, int, int, int, int]]:
    """(epoch, stage, eis, prev_stage, prev_eis) as the reward path sees them."""
    state = SeedState(seed_id="fidelity", blueprint_id="conv", slot_id="r0c0")
    state.stage = SeedStage.BLENDING
    state.previous_stage = SeedStage.TRAINING
    state.previous_epochs_in_stage = scenario.prior_training_dwell
    # The seed trained before the path started: give the metrics a matching
    # history so epochs_total is nonzero (its exact value is not compared).
    for _ in range(scenario.prior_training_dwell):
        state.metrics.record_accuracy(scenario.val_acc)
    state.metrics.reset_stage_baseline()  # what transition() did at BLENDING entry

    observed: list[tuple[int, int, int, int, int]] = []
    commit_epoch = scenario.action_epoch
    for epoch in range(scenario.start_epoch, scenario.max_epochs + 1):
        # 1. validation phase ticks the stage clock (trainer :1875)
        state.metrics.record_accuracy(scenario.val_acc)

        # BLENDING -> HOLDING happens in step_epoch at the END of the previous
        # epoch (alpha reaches target); mirror it before the decision instead
        # of modeling alpha: the driver's scenario fixes the dwell.
        # (Handled below at step 4 - here the state is already correct.)

        # 2. the reward-visible snapshot (pre-action)
        info = SeedInfo.from_seed_state(
            state, 0, counterfactual_total_improvement=scenario.lifetime_counterfactual
        )
        assert info is not None
        observed.append(
            (
                epoch,
                info.stage,
                info.epochs_in_stage,
                info.previous_stage,
                info.previous_epochs_in_stage,
            )
        )

        # 3. mutation after the reward (policy FOSSILIZE at the commit row)
        if commit_epoch is not None and epoch == commit_epoch:
            assert state.transition(SeedStage.FOSSILIZED)

        # 4. step_epoch-equivalent: the BLENDING->HOLDING advance at the end
        # of the last BLENDING epoch (kasmina advances when alpha completes).
        if epoch == scenario.holding_entry_epoch - 1:
            assert state.transition(SeedStage.HOLDING)

    return observed


def test_live_convention_matches_kasmina_state_machine():
    scenario = PathScenario(
        kind="A",
        spot_contribution=3.0,
        lifetime_counterfactual=3.0,
        start_epoch=20,
        blending_dwell=3,
        prior_training_dwell=4,
        action_dwell=5,
        max_epochs=40,  # short horizon: the sequence shape is what matters
        germination_epoch=12,
    )
    observed = _observed_sequence(scenario)

    ledger = replay_path(scenario, convention="live")
    generated = [
        (
            row.epoch,
            row.stage,
            row.epochs_in_stage,
            row.previous_stage,
            row.previous_epochs_in_stage,
        )
        for row in ledger.rows
        if row.stage is not None
    ]

    assert generated == observed


def test_eis_zero_never_visible_to_reward_under_live_ordering():
    """The PBRS eis==0 branch is unreachable for policy transitions live."""
    scenario = PathScenario(
        kind="A",
        spot_contribution=3.0,
        lifetime_counterfactual=3.0,
        start_epoch=20,
        blending_dwell=3,
        prior_training_dwell=4,
        action_dwell=5,
        max_epochs=40,
        germination_epoch=12,
    )
    observed = _observed_sequence(scenario)
    assert all(eis >= 1 for (_, _, eis, _, _) in observed)
