"""S vs A comparator ledgers: the shipping FOSSILIZE economics, measured.

The decision-relevant contrast is evaluated from the commit epoch onward
(identical prefixes cancel): per-component discounted PVs of A (commit now)
minus S (park to horizon), on the PRODUCTION reward path.

Named findings this file measures rather than asserts from hand arithmetic:
- the park-vs-fossil PBRS residual (PDR-0084 #4 recorded parking preferred by
  ~+0.31; the shipped code's post-commit potential RE-ACCRUAL - a fossil's
  epochs_in_stage keeps ticking, so Phi climbs 6.0 -> 8.0 - was missed by the
  audit and reverses the comparison at ordinary horizons);
- F8, the negative-carry escape hatch (commit a declining seed for a one-shot
  -0.2 and stop paying negative attribution) - LIVE in shipping code;
- H4: a fossil's attribution is exactly zero ever after (the -113-scale
  forfeiture's mechanism, here in per-component form).
"""

from __future__ import annotations

import pytest

from .path_driver import PathScenario, render_ledger_markdown, replay_path


def paired(kind: str, c: float, cf: float, dwell: int = 5, max_epochs: int = 150):
    return PathScenario(
        kind=kind,  # type: ignore[arg-type]
        spot_contribution=c,
        lifetime_counterfactual=cf,
        start_epoch=20,
        blending_dwell=3,
        prior_training_dwell=4,
        action_dwell=dwell if kind != "S" else None,
        max_epochs=max_epochs,
        germination_epoch=12,
    )


class TestPostCommitAccounting:
    def test_fossil_attribution_is_zero_forever(self):
        ledger = replay_path(paired("A", 3.0, 3.0), convention="live")
        commit = next(r for r in ledger.rows if r.action.name == "FOSSILIZE")
        post = [r for r in ledger.rows if r.epoch > commit.epoch]
        assert post, "commit must not be terminal in this scenario"
        assert all(r.components.bounded_attribution == 0.0 for r in post)

    def test_fossil_maintenance_starts_after_commit(self):
        ledger = replay_path(paired("A", 3.0, 3.0), convention="live")
        commit = next(r for r in ledger.rows if r.action.name == "FOSSILIZE")
        assert commit.components.fossilized_rent == 0.0
        post = [r for r in ledger.rows if r.epoch > commit.epoch]
        assert all(
            r.components.fossilized_rent == pytest.approx(0.002) for r in post
        )


class TestForfeitedAttributionStream:
    def test_commit_forfeits_the_provisional_stream(self):
        # The dominant term of the shipping FOSSILIZE ledger: S keeps earning
        # c per measured epoch, A earns 0 after the commit. Measured from the
        # commit epoch, in J-currency the forfeiture dwarfs the flat prior.
        s = replay_path(paired("S", 3.0, 3.0), convention="live")
        a = replay_path(paired("A", 3.0, 3.0), convention="live")
        t_c = next(r for r in a.rows if r.action.name == "FOSSILIZE").epoch

        forfeited = s.pv("bounded_attribution", from_epoch=t_c) - a.pv(
            "bounded_attribution", from_epoch=t_c
        )
        prior = a.pv("action_shaping", from_epoch=t_c) - s.pv(
            "action_shaping", from_epoch=t_c
        )
        assert forfeited > 100.0  # the -113-scale stream, per-path form
        assert prior < 2.0
        assert forfeited / prior > 50.0

    def test_holding_warning_relief_partially_offsets(self):
        # S pays the anti-park warning every eligible epoch; A stops at commit.
        s = replay_path(paired("S", 3.0, 3.0), convention="live")
        a = replay_path(paired("A", 3.0, 3.0), convention="live")
        t_c = next(r for r in a.rows if r.action.name == "FOSSILIZE").epoch
        relief = a.pv("holding_warning", from_epoch=t_c) - s.pv(
            "holding_warning", from_epoch=t_c
        )
        assert relief > 0  # committing escapes the indecision penalty


class TestParkVsFossilShapingResidual:
    def test_live_residual_favors_fossilizing_not_parking(self):
        # PDR-0084 #4 recorded the potential table paying ~+0.31 for PARKING.
        # On the shipped path the commit skips the forfeiture (live ordering)
        # AND the fossil re-accrues progress potential to 8.0 > 7.5: the PBRS
        # stream from the commit epoch favors FOSSILIZING. Sign is asserted;
        # the magnitude is reported by the ledger artifact.
        s = replay_path(paired("S", 3.0, 3.0), convention="live")
        a = replay_path(paired("A", 3.0, 3.0), convention="live")
        t_c = next(r for r in a.rows if r.action.name == "FOSSILIZE").epoch
        residual = a.pv_pbrs(from_epoch=t_c) - s.pv_pbrs(from_epoch=t_c)
        assert residual > 0.2

    def test_audit_convention_residual_also_not_park_favored(self):
        # Even paying the F5 forfeiture (audit convention), the re-accrual to
        # Phi(F, cap)=8.0 - which the A7 arithmetic missed entirely - keeps
        # the net residual from favoring parking by the recorded +0.31.
        s = replay_path(paired("S", 3.0, 3.0), convention="audit")
        a = replay_path(paired("A", 3.0, 3.0), convention="audit")
        t_c = next(r for r in a.rows if r.action.name == "FOSSILIZE").epoch
        residual = a.pv_pbrs(from_epoch=t_c) - s.pv_pbrs(from_epoch=t_c)
        assert residual > -0.31


class TestF8EscapeHatch:
    def test_committing_a_bleeding_seed_beats_holding_it(self):
        # Lifetime counterfactual >= 0, spot c < 0: staying bleeds
        # contribution_weight * c per epoch; committing costs a one-shot
        # -0.2 - 0.01 and stops the bleeding. The hatch is LIVE.
        s = replay_path(paired("S", -0.5, 2.0), convention="live")
        a = replay_path(paired("A", -0.5, 2.0), convention="live")
        t_c = next(r for r in a.rows if r.action.name == "FOSSILIZE").epoch
        advantage = a.pv_total(from_epoch=t_c) - s.pv_total(from_epoch=t_c)
        assert advantage > 5.0  # material, not a rounding artifact


class TestCaseMatrixAndReport:
    @pytest.mark.parametrize("c,cf", [(3.0, 3.0), (0.5, 0.5), (0.0, 0.0), (-0.5, 2.0), (-0.5, -1.0)])
    @pytest.mark.parametrize("dwell", [1, 5, 7, 9])
    @pytest.mark.parametrize("convention", ["live", "audit"])
    def test_matrix_ledgers_are_finite_and_renderable(self, c, cf, dwell, convention):
        ledger = replay_path(paired("A", c, cf, dwell=dwell), convention=convention)
        assert all(
            abs(r.reward_total) < 100 for r in ledger.rows
        ), "reward blow-up in the case matrix"
        report = render_ledger_markdown(ledger)
        assert "discounted PV" in report

    def test_stale_epochs_with_no_stage_signal_pay_nothing(self):
        # A6 status dimension, SCOPED (review nit #3): with BOTH signals absent
        # (contribution None AND stage_improvement None - the driver's
        # flat-val_acc world), unmeasured HOLDING epochs earn no attribution
        # and no warning. This is NOT a general production fact: live, a stale
        # epoch with positive stage-relative drift pays PROXY attribution and
        # the warning - covered by the proxy-path test below.
        base = paired("S", 3.0, 3.0)
        measured = frozenset(
            e for e in range(base.start_epoch, base.max_epochs + 1) if e % 2 == 0
        )
        scenario = PathScenario(
            kind="S",
            spot_contribution=3.0,
            lifetime_counterfactual=3.0,
            start_epoch=20,
            blending_dwell=3,
            prior_training_dwell=4,
            action_dwell=None,
            max_epochs=150,
            germination_epoch=12,
            measured_epochs=measured,
        )
        ledger = replay_path(scenario, convention="live")
        unmeasured_rows = [
            r for r in ledger.rows if r.epoch % 2 == 1 and r.stage is not None
        ]
        assert all(r.components.bounded_attribution == 0.0 for r in unmeasured_rows)
        assert all(r.components.holding_warning == 0.0 for r in unmeasured_rows)

    def test_unmeasured_epochs_with_positive_drift_pay_proxy_attribution(self):
        # The LIVE proxy path (contribution.py:602-642): a never-measured
        # HOLDING seed with positive stage-relative improvement pays the
        # heavily-discounted proxy (0.3 x improvement, no timing discount on
        # this branch) - and because attribution is then positive, the
        # holding_warning fires too. This is the branch the scoped test above
        # cannot reach.
        scenario = PathScenario(
            kind="S",
            spot_contribution=3.0,
            lifetime_counterfactual=None,
            start_epoch=20,
            blending_dwell=3,
            prior_training_dwell=4,
            action_dwell=None,
            max_epochs=40,
            germination_epoch=12,
            measured_epochs=frozenset(),  # never measured
            stage_improvement=2.0,
        )
        ledger = replay_path(scenario, convention="live")
        row = next(
            r for r in ledger.rows if r.stage == 6 and r.epochs_in_stage == 3
        )  # HOLDING, eis=3
        assert row.components.bounded_attribution == pytest.approx(0.3 * 2.0, abs=1e-12)
        assert row.components.holding_warning == pytest.approx(-0.15, abs=1e-12)
