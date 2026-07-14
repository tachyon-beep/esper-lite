"""Settlement increment 2: the committed-flag state machine on SeedState (Phase 2).

Spec: docs/plans/ready/2026-07-14-phase2-settlement-implementation.md §2/§5 +
pre-reg §2.1. PENDING is a FLAG on HOLDING (PDR-0088/0090): no stage change at
request, no stage-clock reset, the seed stays measurable; uncancellable BY
POLICY (safety always wins, §4); serialization is unconditional (no-legacy).
"""

from __future__ import annotations

import pytest
import torch

from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import SeedStage


def _holding_state(dwell: int = 5) -> SeedState:
    state = SeedState(seed_id="s1", blueprint_id="conv_light", slot_id="r0c0")
    state.stage = SeedStage.HOLDING
    state.previous_stage = SeedStage.BLENDING
    state.previous_epochs_in_stage = 3
    for _ in range(dwell):
        state.metrics.record_accuracy(60.0)
    return state


class TestRequestFossilization:
    def test_request_sets_the_flag_fields(self):
        state = _holding_state()
        state.request_fossilization(settlement_boundary_epoch=30, legitimacy=1.0)
        assert state.committed is True
        assert state.settlement_boundary_epoch == 30
        assert state.legitimacy_at_request == 1.0

    def test_request_requires_holding(self):
        state = SeedState(seed_id="s1", blueprint_id="conv_light", slot_id="r0c0")
        state.stage = SeedStage.BLENDING
        with pytest.raises(ValueError, match="HOLDING"):
            state.request_fossilization(settlement_boundary_epoch=30, legitimacy=1.0)

    def test_double_request_rejected(self):
        state = _holding_state()
        state.request_fossilization(settlement_boundary_epoch=30, legitimacy=1.0)
        with pytest.raises(ValueError, match="already committed"):
            state.request_fossilization(settlement_boundary_epoch=40, legitimacy=0.8)

    def test_no_stage_clock_reset_at_request(self):
        # Construction invariant (PDR-0088 #1 / plan §B): PENDING is a flag,
        # not a stage — no transition, no epochs_in_stage reset, no minted
        # PBRS climb.
        state = _holding_state(dwell=5)
        eis_before = state.metrics.epochs_in_current_stage
        stage_before = state.stage
        state.request_fossilization(settlement_boundary_epoch=30, legitimacy=1.0)
        assert state.stage == stage_before
        assert state.metrics.epochs_in_current_stage == eis_before
        assert state.previous_stage == SeedStage.BLENDING  # untouched

    def test_boundary_transition_still_legal_when_committed(self):
        # The settlement adapter executes the REAL fossilize at the boundary —
        # the certified A-path mechanics (exactly one stage-entry, at B).
        state = _holding_state()
        state.request_fossilization(settlement_boundary_epoch=30, legitimacy=1.0)
        assert state.transition(SeedStage.FOSSILIZED) is True
        assert state.stage == SeedStage.FOSSILIZED


class TestPolicyScopedUncancellability:
    def _committed_slot(self) -> SeedSlot:
        slot = SeedSlot(slot_id="r0c0", channels=8, device="cpu", on_telemetry=lambda e: None)
        slot.seed = torch.nn.Identity()
        slot.state = _holding_state()
        slot.state.request_fossilization(settlement_boundary_epoch=30, legitimacy=1.0)
        return slot

    def test_policy_prune_on_committed_raises(self):
        slot = self._committed_slot()
        with pytest.raises(RuntimeError, match="committed"):
            slot.prune(reason="policy_choice", initiator="policy")

    def test_non_policy_prune_proceeds(self):
        # Safety always wins (§4): governor/safety paths are NOT blocked by
        # commitment; ledger teardown is the settlement adapter's job.
        slot = self._committed_slot()
        assert slot.prune(reason="governor_panic", initiator="governor") is True


class TestSerialization:
    def test_mid_window_roundtrip(self):
        state = _holding_state()
        state.request_fossilization(settlement_boundary_epoch=30, legitimacy=0.8)
        restored = SeedState.from_dict(state.to_dict())
        assert restored.committed is True
        assert restored.settlement_boundary_epoch == 30
        assert restored.legitimacy_at_request == 0.8

    def test_uncommitted_roundtrip(self):
        state = _holding_state()
        restored = SeedState.from_dict(state.to_dict())
        assert restored.committed is False
        assert restored.settlement_boundary_epoch is None
        assert restored.legitimacy_at_request is None

    def test_from_dict_requires_the_new_keys(self):
        # No-legacy policy: required keys, no fallback defaults — pre-change
        # checkpoints are incompatible by design (plan §5, documented breakage).
        state = _holding_state()
        data = state.to_dict()
        del data["committed"]
        with pytest.raises(KeyError):
            SeedState.from_dict(data)
