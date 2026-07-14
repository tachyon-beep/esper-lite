"""G-PBRS: whole-path telescoping over the production per-step PBRS stream.

Audit A2 asked: does the discounted PBRS stream telescope end-to-end (clips?
terminal handling? stage-dependent exceptions)? Phase 1 answers it on the S/A
paths:

- Under the AUDIT convention (first post-transition reward row sees eis==0)
  the stream satisfies the full Ng telescoping identity
  ``PV == w * (gamma^N * Phi_last - Phi_entry)`` exactly.
- Under the LIVE convention (trainer ordering; first post-transition row sees
  eis==1) every cross-stage transition delta is SKIPPED: the stream telescopes
  only within stages. The unpaid deltas - including audit F5's dwell-graded
  fossilize forfeiture - are quantified here as the difference between the
  full identity and the within-stage closed form.
- The PRUNE-instant cull (paid in action_shaping at the action row) deviates
  from a clean next-step transition by one discount step; quantified exactly.
"""

from __future__ import annotations

import pytest

from esper.leyline import SeedStage

from .path_driver import (
    PathScenario,
    expected_pbrs_pv,
    potential,
    replay_path,
)

GAMMA = 0.995
W = 0.3


def scenario(kind: str, action_dwell: int | None, max_epochs: int = 150) -> PathScenario:
    return PathScenario(
        kind=kind,  # type: ignore[arg-type]
        spot_contribution=3.0,
        lifetime_counterfactual=3.0,
        start_epoch=20,
        blending_dwell=3,
        prior_training_dwell=4,
        action_dwell=action_dwell,
        max_epochs=max_epochs,
        germination_epoch=12,
    )


class TestWithinConventionIdentities:
    @pytest.mark.parametrize("kind,dwell", [("S", None), ("A", 1), ("A", 5), ("A", 7), ("A", 9)])
    def test_live_ledger_matches_its_closed_form(self, kind, dwell):
        ledger = replay_path(scenario(kind, dwell), convention="live")
        assert ledger.pv_pbrs() == pytest.approx(expected_pbrs_pv(ledger), abs=1e-9)

    @pytest.mark.parametrize("kind,dwell", [("S", None), ("A", 1), ("A", 5), ("A", 7), ("A", 9)])
    def test_audit_ledger_matches_its_closed_form(self, kind, dwell):
        ledger = replay_path(scenario(kind, dwell), convention="audit")
        assert ledger.pv_pbrs() == pytest.approx(expected_pbrs_pv(ledger), abs=1e-9)

    def test_audit_convention_satisfies_full_ng_identity(self):
        # With every transition visible at eis==0, the whole path telescopes
        # from the entry anchor Phi(TRAINING, prior_dwell) to the final state.
        config_scenario = scenario("A", 5)
        ledger = replay_path(config_scenario, convention="audit")
        seed_rows = [r for r in ledger.rows if r.stage is not None]
        n = len(seed_rows)
        last = seed_rows[-1]
        phi_last = potential(SeedStage(last.stage), last.epochs_in_stage, ledger.config)
        phi_entry = potential(
            SeedStage.TRAINING, config_scenario.prior_training_dwell, ledger.config
        )
        full_identity = W * (GAMMA**n * phi_last - phi_entry)
        assert ledger.pv_pbrs() == pytest.approx(full_identity, abs=1e-9)

    def test_live_convention_violates_full_ng_identity(self):
        # The same full identity does NOT hold live: the cross-stage deltas
        # (BLENDING->HOLDING and HOLDING->FOSSILIZED) go unpaid. This is the
        # Phase-1 PBRS finding in one assertion.
        config_scenario = scenario("A", 5)
        ledger = replay_path(config_scenario, convention="live")
        seed_rows = [r for r in ledger.rows if r.stage is not None]
        n = len(seed_rows)
        last = seed_rows[-1]
        phi_last = potential(SeedStage(last.stage), last.epochs_in_stage, ledger.config)
        phi_entry = potential(
            SeedStage.TRAINING, config_scenario.prior_training_dwell, ledger.config
        )
        full_identity = W * (GAMMA**n * phi_last - phi_entry)
        deviation = ledger.pv_pbrs() - full_identity
        assert abs(deviation) > 0.1  # material, not a float artifact


class TestF5ForfeitureRealization:
    def test_forfeiture_paid_only_under_audit_convention(self):
        live = replay_path(scenario("A", 5), convention="live")
        audit = replay_path(scenario("A", 5), convention="audit")

        f5 = W * (GAMMA * 6.0 - (5.5 + 5 * 0.3))  # -0.309 at dwell 5
        live_pbrs = [r.components.pbrs_bonus for r in live.rows if r.stage is not None]
        audit_pbrs = [r.components.pbrs_bonus for r in audit.rows if r.stage is not None]

        assert not any(p == pytest.approx(f5, abs=1e-9) for p in live_pbrs)
        assert any(p == pytest.approx(f5, abs=1e-9) for p in audit_pbrs)


class TestPruneCull:
    def test_prune_cull_deviation_from_clean_transition(self):
        # The cull pays w*(gamma*0 - Phi(H,d)) AT the action row (offset t).
        # A clean transition would pay the same delta one step later:
        # w*gamma^(t+1)*(0 - Phi(H,d)) discounted vs w*gamma^t*(-Phi(H,d)).
        # Deviation = -w * gamma^t * Phi * (1 - gamma): PRUNE is overcharged
        # by one discount step relative to clean PBRS.
        config_scenario = scenario("PRUNE", 6)
        ledger = replay_path(config_scenario, convention="live")

        action_row = next(r for r in ledger.rows if r.action.name == "PRUNE")
        t = action_row.epoch - config_scenario.start_epoch
        phi_at_prune = potential(SeedStage.HOLDING, action_row.epochs_in_stage, ledger.config)

        # Within-stage closed form for the seed-bearing rows PLUS a clean
        # transition-to-zero at t+1 would land at:
        clean = expected_pbrs_pv(ledger, exclude_embedded=True) + W * (
            GAMMA ** (t + 1)
        ) * (0.0 - phi_at_prune)
        actual = ledger.pv_pbrs()
        deviation = actual - clean
        expected_deviation = -W * GAMMA**t * phi_at_prune * (1 - GAMMA)
        assert deviation == pytest.approx(expected_deviation, abs=1e-9)


class TestTerminalEdge:
    def test_commit_at_terminal_epoch_skips_all_post_commit_accounting(self):
        # Committing AT max_epochs pays the full flat prior + attribution and
        # then the episode ends: no forfeiture (either convention), no fossil
        # maintenance, no re-accrual. "Sell at the bell" leaves the ledger at
        # the commit row.
        base = scenario("S", None, max_epochs=40)
        holding_entry = base.holding_entry_epoch  # 23
        dwell_at_terminal = 40 - holding_entry + 1  # eis at epoch 40, live
        config_scenario = PathScenario(
            kind="A",
            spot_contribution=3.0,
            lifetime_counterfactual=3.0,
            start_epoch=20,
            blending_dwell=3,
            prior_training_dwell=4,
            action_dwell=dwell_at_terminal,
            max_epochs=40,
            germination_epoch=12,
        )
        ledger = replay_path(config_scenario, convention="live")
        last = ledger.rows[-1]
        assert last.action.name == "FOSSILIZE"
        assert last.epoch == 40
        assert last.components.terminal_bonus == pytest.approx(3.0)
        assert all(r.stage != SeedStage.FOSSILIZED.value for r in ledger.rows)
