"""Phase 4: prune/cooldown pipeline tests.

These tests lock the core behavioral guarantees before action-space rewiring:
- Prune completion physically removes the seed module immediately.
- The slot remains unavailable via PRUNED → EMBARGOED → RESETTING (anti-thrashing).
- Forward is identity in inactive stages (no compute paid after removal).
- Governor can force an emergency prune with explicit initiator/reason telemetry.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
import torch
import torch.nn as nn

from esper.kasmina.slot import SeedSlot
from esper.leyline import DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE, SeedStage, TelemetryEventType
from esper.leyline.alpha import AlphaMode


def _drain_cooldown(slot: SeedSlot) -> None:
    """Advance the PRUNED → EMBARGOED → RESETTING → DORMANT pipeline."""
    for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
        slot.step_epoch()


class TestPruneCooldownPipeline:
    def test_prune_pipeline_transitions_and_forward_identity(self):
        slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu", fast_mode=True)
        slot.germinate("noop", seed_id="seed0")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.set_alpha(0.5)

        x = torch.randn(2, 64, 8, 8)

        # Prune instant (currently: cull) removes seed but retains state.
        ok = slot.prune(reason="test_prune")
        assert ok is True
        assert slot.seed is None
        assert slot.is_active is False
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        torch.testing.assert_close(slot(x), x)

        # PRUNED → EMBARGOED on next tick.
        slot.step_epoch()
        assert slot.state is not None
        assert slot.state.stage == SeedStage.EMBARGOED
        assert slot.seed is None
        torch.testing.assert_close(slot(x), x)

        # Stay EMBARGOED for the dwell window.
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE - 1):
            slot.step_epoch()
            assert slot.state is not None
            assert slot.state.stage == SeedStage.EMBARGOED
            assert slot.seed is None
            torch.testing.assert_close(slot(x), x)

        # Final embargo tick transitions to RESETTING (state still exists, seed removed).
        slot.step_epoch()
        assert slot.state is not None
        assert slot.state.stage == SeedStage.RESETTING
        assert slot.seed is None
        torch.testing.assert_close(slot(x), x)

        # RESETTING tick clears state (slot returns to DORMANT/empty).
        slot.step_epoch()
        assert slot.state is None
        assert slot.seed is None
        torch.testing.assert_close(slot(x), x)

    def test_scheduled_prune_completes_when_alpha_hits_zero(self):
        slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu", fast_mode=True)
        slot.germinate("noop", seed_id="seed0")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        slot.set_alpha(1.0)
        slot.state.alpha_controller.alpha = 1.0
        slot.state.alpha_controller.alpha_target = 1.0
        slot.state.alpha_controller.alpha_mode = AlphaMode.HOLD

        ok = slot.schedule_prune(steps=2, reason="policy_prune")
        assert ok is True
        assert slot.state.stage == SeedStage.BLENDING
        assert slot.seed is not None

        slot.step_epoch()
        assert slot.state.stage == SeedStage.BLENDING
        assert slot.seed is not None

        slot.step_epoch()
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        assert slot.seed is None

    def test_embargo_blocks_germinate_until_dwell_completes(self):
        slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu", fast_mode=True)

        slot.germinate("noop", seed_id="seed0")
        slot.state.transition(SeedStage.TRAINING)
        slot.prune(reason="test_prune")

        with pytest.raises(RuntimeError, match="unavailable for germination"):
            slot.germinate("noop", seed_id="seed1")

        _drain_cooldown(slot)
        assert slot.state is None

        state = slot.germinate("noop", seed_id="seed1")
        assert state.seed_id == "seed1"
        assert state.stage == SeedStage.GERMINATED

    def test_seed_removed_but_state_persists_during_cooldown(self):
        slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu", fast_mode=True)
        slot.germinate("noop", seed_id="seed0")
        slot.state.transition(SeedStage.TRAINING)

        slot.prune(reason="test_prune")
        assert slot.seed is None
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED

        slot.step_epoch()
        assert slot.seed is None
        assert slot.state is not None
        assert slot.state.stage == SeedStage.EMBARGOED

        # Transition to RESETTING (seed remains None, state remains)
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE):
            slot.step_epoch()
        assert slot.seed is None
        assert slot.state is not None
        assert slot.state.stage == SeedStage.RESETTING


class TestGovernorEmergencyPrune:
    @pytest.mark.parametrize(
        "stage",
        [SeedStage.GERMINATED, SeedStage.TRAINING, SeedStage.BLENDING, SeedStage.HOLDING],
    )
    def test_governor_prune_emits_initiator_and_reason(self, stage: SeedStage):
        events_by_type: dict[str, list[dict]] = defaultdict(list)

        def _on_telemetry(event):
            events_by_type[event.event_type.name].append(event.data or {})

        slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu", on_telemetry=_on_telemetry, fast_mode=False)
        slot.germinate("noop", seed_id="seed0")
        slot.seed = nn.Identity()
        slot.state.stage = stage

        ok = slot.prune("governor_nan", initiator="governor")
        assert ok is True
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED

        pruned_events = events_by_type[TelemetryEventType.SEED_PRUNED.name]
        assert len(pruned_events) == 1
        payload = pruned_events[0]
        assert payload.get("reason") == "governor_nan"
        assert payload.get("initiator") == "governor"
