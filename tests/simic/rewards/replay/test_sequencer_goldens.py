"""Per-row goldens for the replay path driver, both timing conventions.

Every expected value below is hand-derived from the audited shipping formulas
(docs/analysis/2026-07-14-fossilize-reward-path-audit.md section 1), NOT from
running the driver - these tests pin the driver's SeedInfo sequencing and the
production reward path's per-row outputs.

Conventions:
- "live": the trainer's actual ordering (vectorized_trainer.py:1870-1875
  record_accuracy ticks epochs_in_stage during the metrics phase, BEFORE the
  action transaction; mutations execute after the reward). The first reward
  row after a stage transition therefore sees epochs_in_stage == 1.
- "audit": the ordering the reward-path audit (and the PBRS eis==0 branch,
  contribution.py:1235-1246) assumed - the first post-transition reward row
  sees epochs_in_stage == 0 with previous_* describing the exited stage.

The divergence between the two IS a Phase-1 deliverable (the cross-stage PBRS
transition deltas - including audit F5's dwell-graded fossilize forfeiture -
pay only under the audit convention).
"""

from __future__ import annotations

import math

import pytest

from esper.leyline import LifecycleOp, SeedStage

from .path_driver import PathScenario, replay_path

GAMMA = 0.995
W = 0.3  # pbrs_weight


def std_scenario(kind: str, action_dwell: int | None = 5) -> PathScenario:
    return PathScenario(
        kind=kind,  # type: ignore[arg-type]
        spot_contribution=3.0,
        lifetime_counterfactual=3.0,
        start_epoch=20,
        blending_dwell=3,
        prior_training_dwell=4,
        action_dwell=None if kind == "S" else action_dwell,
        max_epochs=150,
        germination_epoch=12,
    )


def row_at(ledger, epoch):
    (row,) = [r for r in ledger.rows if r.epoch == epoch]
    return row


def test_constants_pin_for_goldens():
    """Every hand-derived golden in this package assumes these shipping
    constants. If this test fails, a constant legitimately changed - update
    the goldens knowingly rather than chasing per-test mismatches (review
    nit #4)."""
    from esper.simic.rewards.shaping import STAGE_POTENTIALS

    from .path_driver import make_config

    config = make_config()
    assert STAGE_POTENTIALS[SeedStage.TRAINING] == 2.0
    assert STAGE_POTENTIALS[SeedStage.BLENDING] == 3.5
    assert STAGE_POTENTIALS[SeedStage.HOLDING] == 5.5
    assert STAGE_POTENTIALS[SeedStage.FOSSILIZED] == 6.0
    assert STAGE_POTENTIALS[SeedStage.PRUNED] == 0.0
    assert config.pbrs_weight == W == 0.3
    assert config.epoch_progress_bonus == 0.3
    assert config.max_progress_bonus == 2.0
    assert config.gamma == GAMMA == 0.995
    assert config.fossilize_base_bonus == 0.5
    assert config.fossilize_contribution_scale == 0.1
    assert config.fossilize_terminal_scale == 3.0
    assert config.fossilize_quality_ceiling == 3.0
    assert config.fossilize_cost == -0.01
    assert config.fossilize_noncontributing_penalty == -0.2
    assert config.fossilized_maintenance_cost == 0.002
    assert config.terminal_acc_weight == 0.05
    assert config.contribution_weight == 1.0
    assert config.proxy_contribution_weight == pytest.approx(0.3)


class TestLiveConventionSequencing:
    def test_blending_entry_row_has_eis_1(self):
        ledger = replay_path(std_scenario("S"), convention="live")
        row = row_at(ledger, 20)
        assert row.stage == SeedStage.BLENDING.value
        assert row.epochs_in_stage == 1

    def test_holding_entry_row_pbrs_is_within_stage_only(self):
        # Live: first HOLDING reward row has eis=1; PBRS pays the within-stage
        # step from Phi(H,0)=5.5 to Phi(H,1)=5.8 - NOT the BLENDING->HOLDING
        # stage-entry delta.
        ledger = replay_path(std_scenario("S"), convention="live")
        row = row_at(ledger, 23)
        assert row.stage == SeedStage.HOLDING.value
        assert row.epochs_in_stage == 1
        expected = W * (GAMMA * 5.8 - 5.5)
        assert row.components.pbrs_bonus == pytest.approx(expected, abs=1e-12)

    def test_commit_row_pays_flat_prior_and_attribution(self):
        # Commit at HOLDING eis=5 (legitimacy = 1.0): action_shaping =
        # (0.5 + 0.1*3.0) + 3.0*tanh(1/3) - 0.01; attribution still pays c.
        ledger = replay_path(std_scenario("A"), convention="live")
        commit_epoch = 23 + 5 - 1  # eis==5 at epoch 27 under live convention
        row = row_at(ledger, commit_epoch)
        assert row.action == LifecycleOp.FOSSILIZE
        assert row.epochs_in_stage == 5
        expected_shaping = (0.5 + 0.1 * 3.0) + 3.0 * math.tanh(1.0 / 3.0) - 0.01
        assert row.components.action_shaping == pytest.approx(expected_shaping, abs=1e-12)
        assert row.components.bounded_attribution == pytest.approx(3.0, abs=1e-12)
        assert row.components.holding_warning == 0.0  # terminal actions exempt

    def test_first_fossil_row_pays_no_forfeiture(self):
        # THE live-semantics finding: the first FOSSILIZED reward row has
        # eis=1, so PBRS pays the within-stage step from Phi(F,0)=6.0 to
        # Phi(F,1)=6.3 (+0.0806) - audit F5's forfeiture (-0.309 at dwell 5)
        # never appears on the live path.
        ledger = replay_path(std_scenario("A"), convention="live")
        row = row_at(ledger, 28)
        assert row.stage == SeedStage.FOSSILIZED.value
        assert row.epochs_in_stage == 1
        expected = W * (GAMMA * 6.3 - 6.0)
        assert row.components.pbrs_bonus == pytest.approx(expected, abs=1e-12)
        assert row.components.bounded_attribution == 0.0  # H4: fossils pay nothing
        assert row.components.fossilized_rent == pytest.approx(0.002, abs=1e-15)

    def test_holding_warning_ramp(self):
        # WAIT in HOLDING with positive attribution: 0 at eis<2, then
        # -0.1, -0.15, -0.20 ... capped at -0.3 (contribution.py:706-717).
        ledger = replay_path(std_scenario("S"), convention="live")
        assert row_at(ledger, 23).components.holding_warning == 0.0  # eis=1
        assert row_at(ledger, 24).components.holding_warning == pytest.approx(-0.1)
        assert row_at(ledger, 25).components.holding_warning == pytest.approx(-0.15)
        assert row_at(ledger, 26).components.holding_warning == pytest.approx(-0.2)
        assert row_at(ledger, 30).components.holding_warning == pytest.approx(-0.3)

    def test_terminal_row_pays_accuracy_bonus(self):
        ledger = replay_path(std_scenario("S"), convention="live")
        row = row_at(ledger, 150)
        assert row.components.terminal_bonus == pytest.approx(60.0 * 0.05, abs=1e-12)


class TestAuditConventionSequencing:
    def test_holding_entry_row_pays_stage_delta(self):
        # Audit: first HOLDING row has eis=0 and PBRS pays the cross-stage
        # delta anchored at the LAST-PAID BLENDING potential. With eis 0-based
        # the three BLENDING rows pay up to Phi(B,2)=4.1, so previous_dwell
        # chains as blending_dwell-1 (anchor continuity is what lets the full
        # Ng identity hold, which test_pbrs_telescoping verifies end-to-end).
        ledger = replay_path(std_scenario("S"), convention="audit")
        row = row_at(ledger, 23)
        assert row.epochs_in_stage == 0
        expected = W * (GAMMA * 5.5 - (3.5 + 2 * 0.3))
        assert row.components.pbrs_bonus == pytest.approx(expected, abs=1e-12)

    def test_first_fossil_row_pays_f5_forfeiture(self):
        # Audit F5: commit at dwell 5 -> next row (F, eis=0, prev=(H,5)) pays
        # 0.3*(0.995*6.0 - 7.0) = -0.309.
        ledger = replay_path(std_scenario("A"), convention="audit")
        commit_epoch = 23 + 5  # eis==5 at epoch 28 under audit convention
        row = row_at(ledger, commit_epoch + 1)
        assert row.stage == SeedStage.FOSSILIZED.value
        assert row.epochs_in_stage == 0
        expected = W * (GAMMA * 6.0 - (5.5 + 5 * 0.3))
        assert row.components.pbrs_bonus == pytest.approx(expected, abs=1e-12)
        assert row.components.pbrs_bonus == pytest.approx(-0.309, abs=1e-9)


class TestNoncontributingAndPenaltyBranches:
    def test_below_threshold_commit_pays_noncontributing_penalty(self):
        # c=0.5 < DEFAULT_MIN_FOSSILIZE_CONTRIBUTION(1.0), lifetime cf >= 0:
        # shaping = -0.2 (noncontributing) - 0.01 (cost); no tanh bonus.
        scenario = PathScenario(
            kind="A",
            spot_contribution=0.5,
            lifetime_counterfactual=0.5,
            start_epoch=20,
            blending_dwell=3,
            prior_training_dwell=4,
            action_dwell=5,
            max_epochs=150,
            germination_epoch=12,
        )
        ledger = replay_path(scenario, convention="live")
        row = row_at(ledger, 27)
        assert row.components.action_shaping == pytest.approx(-0.2 - 0.01, abs=1e-12)

    def test_negative_counterfactual_commit_pays_graded_penalty(self):
        # lifetime cf < 0: -0.5 - min(|cf|*0.2, 1.0) (no ransomware term when
        # spot c <= 0.1), plus cost; attribution suppressed at the commit row.
        scenario = PathScenario(
            kind="A",
            spot_contribution=-0.5,
            lifetime_counterfactual=-1.0,
            start_epoch=20,
            blending_dwell=3,
            prior_training_dwell=4,
            action_dwell=5,
            max_epochs=150,
            germination_epoch=12,
        )
        ledger = replay_path(scenario, convention="live")
        row = row_at(ledger, 27)
        expected_shaping = (-0.5 - min(1.0 * 0.2, 1.0)) - 0.01
        assert row.components.action_shaping == pytest.approx(expected_shaping, abs=1e-12)
        assert row.components.bounded_attribution == 0.0
