"""End-to-end tests for decision metrics (TELE-800 to TELE-899).

Verifies decision telemetry flows from VectorizedEmitter through to nissa.
The primary telemetry is AnalyticsSnapshotPayload with kind="last_action",
which captures the complete decision context for each Tamiyo action.

TELE-800: Recent Decisions
- Validates that decision snapshots include action details, confidence, value estimates
- Tests per-head telemetry (confidence/entropy for each action head)
- Verifies reward components and decision context fields

Reference:
    docs/telemetry/telemetry_needs/TELE-800_recent_decisions.md
"""

from __future__ import annotations

from esper.leyline import (
    AnalyticsSnapshotPayload,
    TelemetryEventType,
)
from esper.leyline.telemetry import HeadTelemetry
from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
from esper.simic.telemetry.emitters import VectorizedEmitter
from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
)

from .conftest import CaptureHubResult


def build_action_spec(action_indices: dict[str, int], slot_id: str) -> ActionSpec:
    return ActionSpec(
        slot_idx=action_indices["slot"],
        blueprint_idx=action_indices["blueprint"],
        style_idx=action_indices["style"],
        tempo_idx=action_indices["tempo"],
        alpha_target_idx=action_indices["alpha_target"],
        alpha_speed_idx=action_indices["alpha_speed"],
        alpha_curve_idx=action_indices["alpha_curve"],
        op_idx=action_indices["op"],
        target_slot=slot_id,
        slot_is_enabled=True,
    )


def build_mask_flags(masked: dict[str, bool] | None = None) -> ActionMaskFlags:
    flags = {
        "op": False,
        "slot": False,
        "blueprint": False,
        "style": False,
        "tempo": False,
        "alpha_target": False,
        "alpha_speed": False,
        "alpha_curve": False,
    }
    if masked is not None:
        for key, value in masked.items():
            flags[key] = value
    return ActionMaskFlags(
        op_masked=flags["op"],
        slot_masked=flags["slot"],
        blueprint_masked=flags["blueprint"],
        style_masked=flags["style"],
        tempo_masked=flags["tempo"],
        alpha_target_masked=flags["alpha_target"],
        alpha_speed_masked=flags["alpha_speed"],
        alpha_curve_masked=flags["alpha_curve"],
    )


def build_action_outcome(
    success: bool,
    *,
    total_reward: float = 0.0,
    reward_components: RewardComponentsTelemetry | None = None,
) -> ActionOutcome:
    return ActionOutcome(
        action_success=success,
        reward_raw=total_reward,
        reward_components=reward_components,
    )


class TestTELE800RecentDecisions:
    """TELE-800: Recent Decisions - verify last_action snapshot emission."""

    def test_last_action_emitted_with_correct_kind(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: last_action snapshot is emitted with kind='last_action'."""
        hub, backend = capture_hub

        # Create emitter with capture hub
        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        # Trigger last_action emission with minimal required fields
        action_indices = {
            "op": 0,  # WAIT
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }
        emitter.on_last_action(
            epoch=5,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
        )

        # Flush to ensure event is processed
        hub.flush(timeout=5.0)

        # Find ANALYTICS_SNAPSHOT events
        events = backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
        assert len(events) >= 1, "Expected at least one ANALYTICS_SNAPSHOT event"

        # Verify kind field
        last_action_events = [
            e for e in events
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(last_action_events) == 1, "Expected exactly one last_action snapshot"

    def test_action_name_populated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: action_name field contains the operation name."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        # Test GERMINATE action (op index 1)
        action_indices = {
            "op": 1,  # GERMINATE
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1
        assert events[0].data.action_name == "GERMINATE"

    def test_action_confidence_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: action_confidence field captures overall probability."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            action_confidence=0.85,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1
        assert events[0].data.action_confidence == 0.85

    def test_value_estimate_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: value_estimate field captures V(s) before action."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            value_estimate=2.5,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1
        assert events[0].data.value_estimate == 2.5

    def test_total_reward_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: total_reward field captures reward for the step."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True, total_reward=1.234),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1
        assert events[0].data.total_reward == 1.234


class TestTELE800HeadTelemetry:
    """TELE-800: Per-head confidence and entropy via HeadTelemetry."""

    def test_head_telemetry_confidence_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: head_telemetry contains per-head confidence values."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        head_telemetry = HeadTelemetry(
            op_confidence=0.75,
            slot_confidence=0.60,
            blueprint_confidence=0.85,
            style_confidence=0.70,
            tempo_confidence=0.55,
            alpha_target_confidence=0.90,
            alpha_speed_confidence=0.65,
            curve_confidence=0.80,
        )

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            head_telemetry=head_telemetry,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        assert payload.head_telemetry is not None
        assert payload.head_telemetry.op_confidence == 0.75
        assert payload.head_telemetry.slot_confidence == 0.60
        assert payload.head_telemetry.blueprint_confidence == 0.85
        assert payload.head_telemetry.style_confidence == 0.70
        assert payload.head_telemetry.tempo_confidence == 0.55
        assert payload.head_telemetry.alpha_target_confidence == 0.90
        assert payload.head_telemetry.alpha_speed_confidence == 0.65
        assert payload.head_telemetry.curve_confidence == 0.80

    def test_head_telemetry_entropy_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: head_telemetry contains per-head entropy values."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        head_telemetry = HeadTelemetry(
            op_entropy=1.2,
            slot_entropy=0.8,
            blueprint_entropy=1.5,
            style_entropy=0.6,
            tempo_entropy=0.4,
            alpha_target_entropy=1.1,
            alpha_speed_entropy=0.9,
            curve_entropy=0.7,
        )

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            head_telemetry=head_telemetry,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        assert payload.head_telemetry is not None
        assert payload.head_telemetry.op_entropy == 1.2
        assert payload.head_telemetry.slot_entropy == 0.8
        assert payload.head_telemetry.blueprint_entropy == 1.5
        assert payload.head_telemetry.style_entropy == 0.6
        assert payload.head_telemetry.tempo_entropy == 0.4
        assert payload.head_telemetry.alpha_target_entropy == 1.1
        assert payload.head_telemetry.alpha_speed_entropy == 0.9
        assert payload.head_telemetry.curve_entropy == 0.7


class TestTELE800DecisionContext:
    """TELE-800: Decision context fields (slot_states, alternatives, entropy)."""

    def test_slot_states_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: slot_states field captures slot ID to state mapping."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        slot_states = {
            "r0c0": "Training 12%",
            "r0c1": "Empty",
            "r1c0": "Blending 45%",
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            slot_states=slot_states,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        assert payload.slot_states == {
            "r0c0": "Training 12%",
            "r0c1": "Empty",
            "r1c0": "Blending 45%",
        }

    def test_alternatives_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: alternatives field captures top-2 alternative actions."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,  # WAIT
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        alternatives = [("GERMINATE", 0.25), ("ADVANCE", 0.15)]

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            alternatives=alternatives,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        assert payload.alternatives is not None
        assert len(payload.alternatives) == 2
        assert payload.alternatives[0] == ("GERMINATE", 0.25)
        assert payload.alternatives[1] == ("ADVANCE", 0.15)

    def test_decision_entropy_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: decision_entropy field captures action distribution entropy."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            decision_entropy=1.789,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1
        assert events[0].data.decision_entropy == 1.789


class TestTELE800RewardComponents:
    """TELE-800: Reward component breakdown via RewardComponentsTelemetry."""

    def test_reward_components_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: reward_components dataclass is included in snapshot."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 1,  # GERMINATE
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        reward_components = RewardComponentsTelemetry(
            base_acc_delta=0.5,
            bounded_attribution=0.3,
            compute_rent=-0.1,
            stage_bonus=0.2,
            total_reward=0.9,
            action_name="GERMINATE",
            action_success=True,
        )

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(
                success=True,
                reward_components=reward_components,
            ),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        assert payload.reward_components is not None
        assert payload.reward_components.base_acc_delta == 0.5
        assert payload.reward_components.bounded_attribution == 0.3
        assert payload.reward_components.compute_rent == -0.1
        assert payload.reward_components.stage_bonus == 0.2
        assert payload.reward_components.total_reward == 0.9
        assert payload.reward_components.action_name == "GERMINATE"
        assert payload.reward_components.action_success is True

    def test_reward_components_all_fields(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: All reward component fields are preserved in telemetry."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        reward_components = RewardComponentsTelemetry(
            base_acc_delta=0.1,
            seed_contribution=0.25,
            bounded_attribution=0.15,
            progress_since_germination=0.05,
            attribution_discount=0.9,
            ratio_penalty=-0.02,
            compute_rent=-0.05,
            alpha_shock=-0.01,
            blending_warning=-0.03,
            holding_warning=0.0,
            stage_bonus=0.1,
            pbrs_bonus=0.08,
            synergy_bonus=0.05,
            action_shaping=0.02,
            terminal_bonus=0.0,
            fossilize_terminal_bonus=0.0,
            hindsight_credit=0.0,
            num_fossilized_seeds=2,
            num_contributing_fossilized=1,
            action_name="WAIT",
            action_success=True,
            seed_stage=3,
            epoch=10,
            val_acc=72.5,
            acc_at_germination=68.0,
            host_baseline_acc=65.0,
            growth_ratio=1.05,
            total_reward=0.67,
        )

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(
                success=True,
                reward_components=reward_components,
            ),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        rc = payload.reward_components
        assert rc is not None

        # Verify all fields are preserved
        assert rc.base_acc_delta == 0.1
        assert rc.seed_contribution == 0.25
        assert rc.bounded_attribution == 0.15
        assert rc.progress_since_germination == 0.05
        assert rc.attribution_discount == 0.9
        assert rc.ratio_penalty == -0.02
        assert rc.compute_rent == -0.05
        assert rc.alpha_shock == -0.01
        assert rc.blending_warning == -0.03
        assert rc.holding_warning == 0.0
        assert rc.stage_bonus == 0.1
        assert rc.pbrs_bonus == 0.08
        assert rc.synergy_bonus == 0.05
        assert rc.action_shaping == 0.02
        assert rc.terminal_bonus == 0.0
        assert rc.fossilize_terminal_bonus == 0.0
        assert rc.hindsight_credit == 0.0
        assert rc.num_fossilized_seeds == 2
        assert rc.num_contributing_fossilized == 1
        assert rc.seed_stage == 3
        assert rc.epoch == 10
        assert rc.val_acc == 72.5
        assert rc.acc_at_germination == 68.0
        assert rc.host_baseline_acc == 65.0
        assert rc.growth_ratio == 1.05


class TestTELE800HeadChoices:
    """TELE-800: Per-head choice fields (slot_id, blueprint_id, style, etc.)."""

    def test_slot_and_blueprint_captured(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: slot_id and blueprint_id reflect action indices."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 1,  # GERMINATE
            "slot": 2,
            "blueprint": 1,
            "style": 0,
            "tempo": 1,
            "alpha_target": 2,
            "alpha_speed": 1,
            "alpha_curve": 0,
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c2"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        assert payload.slot_id == "r0c2"
        # Blueprint IDs are looked up from BLUEPRINT_IDS constant
        assert payload.blueprint_id is not None

    def test_mask_flags_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: Per-head mask flags indicate which heads were forced."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        masked = {
            "op": True,
            "slot": False,
            "blueprint": True,
            "style": False,
            "tempo": True,
            "alpha_target": False,
            "alpha_speed": True,
            "alpha_curve": False,
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(masked),
            outcome=build_action_outcome(success=True),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        assert payload.op_masked is True
        assert payload.slot_masked is False
        assert payload.blueprint_masked is True
        assert payload.style_masked is False
        assert payload.tempo_masked is True
        assert payload.alpha_target_masked is False
        assert payload.alpha_speed_masked is True
        assert payload.alpha_curve_masked is False

    def test_action_success_propagated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: action_success indicates if action executed successfully."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 1,  # GERMINATE
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        # Test success=False (e.g., germination failed due to slot already occupied)
        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=False),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1
        assert events[0].data.action_success is False


class TestTELE800AlphaFields:
    """TELE-800: Alpha-related fields (target, speed, curve, algorithm)."""

    def test_alpha_fields_populated(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: Alpha target, speed, curve, and algorithm fields captured."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            # SET_ALPHA_TARGET is the op where alpha schedule fields are meaningful.
            "op": 2,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 1,  # HALF=0.5, SEVENTY=0.7, FULL=1.0 depending on constant
            "alpha_speed": 1,  # FAST, MEDIUM, SLOW depending on constant
            "alpha_curve": 1,  # LINEAR, COSINE, etc. depending on constant
        }

        emitter.on_last_action(
            epoch=10,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
            active_alpha_algorithm="GATED_GATE",
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        # These are looked up from constants, so we just verify they're populated
        assert payload.alpha_target is not None
        assert payload.alpha_speed is not None
        assert payload.alpha_curve is not None
        assert payload.alpha_algorithm == "GATED_GATE"


class TestTELE800EnvContext:
    """TELE-800: Environment context (env_id, inner_epoch)."""

    def test_env_id_captured(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: env_id identifies which environment made the decision."""
        hub, backend = capture_hub

        # Test with different env_ids
        for env_id in [0, 1, 3]:
            backend.clear()
            emitter = VectorizedEmitter(env_id=env_id, device="cpu", group_id="test", hub=hub)

            action_indices = {
                "op": 0,
                "slot": 0,
                "blueprint": 0,
                "style": 0,
                "tempo": 0,
                "alpha_target": 0,
                "alpha_speed": 0,
                "alpha_curve": 0,
            }

            emitter.on_last_action(
                epoch=5,
                action_spec=build_action_spec(action_indices, slot_id="r0c0"),
                masked=build_mask_flags(),
                outcome=build_action_outcome(success=True),
            )

            hub.flush(timeout=5.0)

            events = [
                e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
                if isinstance(e.data, AnalyticsSnapshotPayload)
                and e.data.kind == "last_action"
            ]
            assert len(events) == 1
            assert events[0].data.env_id == env_id

    def test_inner_epoch_captured(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: inner_epoch tracks epoch when decision was made."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="test", hub=hub)

        action_indices = {
            "op": 0,
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 0,
            "alpha_target": 0,
            "alpha_speed": 0,
            "alpha_curve": 0,
        }

        emitter.on_last_action(
            epoch=42,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags(),
            outcome=build_action_outcome(success=True),
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1
        assert events[0].data.inner_epoch == 42


class TestTELE800FullSnapshot:
    """TELE-800: Complete snapshot with all fields populated."""

    def test_full_decision_snapshot(self, capture_hub: CaptureHubResult) -> None:
        """TELE-800: Complete snapshot includes all decision context fields."""
        hub, backend = capture_hub

        emitter = VectorizedEmitter(env_id=2, device="cuda:0", group_id="test", hub=hub)

        action_indices = {
            "op": 1,  # GERMINATE
            "slot": 0,
            "blueprint": 0,
            "style": 0,
            "tempo": 1,
            "alpha_target": 2,
            "alpha_speed": 1,
            "alpha_curve": 0,
        }

        head_telemetry = HeadTelemetry(
            op_confidence=0.80,
            slot_confidence=0.65,
            blueprint_confidence=0.90,
            style_confidence=0.75,
            tempo_confidence=0.60,
            alpha_target_confidence=0.85,
            alpha_speed_confidence=0.70,
            curve_confidence=0.78,
            op_entropy=0.8,
            slot_entropy=1.2,
            blueprint_entropy=0.5,
            style_entropy=0.9,
            tempo_entropy=1.1,
            alpha_target_entropy=0.6,
            alpha_speed_entropy=1.0,
            curve_entropy=0.7,
        )

        reward_components = RewardComponentsTelemetry(
            base_acc_delta=0.3,
            bounded_attribution=0.2,
            compute_rent=-0.08,
            stage_bonus=0.15,
            total_reward=0.57,
            action_name="GERMINATE",
            action_success=True,
            epoch=25,
            val_acc=71.5,
        )

        slot_states = {
            "r0c0": "Empty",
            "r0c1": "Training 45%",
            "r1c0": "Fossilized",
        }

        alternatives = [("ADVANCE", 0.12), ("WAIT", 0.08)]

        emitter.on_last_action(
            epoch=25,
            action_spec=build_action_spec(action_indices, slot_id="r0c0"),
            masked=build_mask_flags({"op": False, "blueprint": True}),
            outcome=build_action_outcome(
                success=True,
                total_reward=0.57,
                reward_components=reward_components,
            ),
            value_estimate=1.8,
            host_accuracy=71.5,
            slot_states=slot_states,
            action_confidence=0.80,
            alternatives=alternatives,
            decision_entropy=1.45,
            head_telemetry=head_telemetry,
        )

        hub.flush(timeout=5.0)

        events = [
            e for e in backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
            if isinstance(e.data, AnalyticsSnapshotPayload)
            and e.data.kind == "last_action"
        ]
        assert len(events) == 1

        payload = events[0].data
        # Core decision fields
        assert payload.kind == "last_action"
        assert payload.env_id == 2
        assert payload.inner_epoch == 25
        assert payload.action_name == "GERMINATE"
        assert payload.action_success is True

        # Value and reward
        assert payload.total_reward == 0.57
        assert payload.value_estimate == 1.8
        assert payload.action_confidence == 0.80
        assert payload.decision_entropy == 1.45

        # Decision context
        assert payload.slot_states == slot_states
        assert payload.alternatives == alternatives

        # Head telemetry
        assert payload.head_telemetry is not None
        assert payload.head_telemetry.op_confidence == 0.80
        assert payload.head_telemetry.op_entropy == 0.8

        # Reward components
        assert payload.reward_components is not None
        assert payload.reward_components.base_acc_delta == 0.3
        assert payload.reward_components.total_reward == 0.57

        # Mask flags
        assert payload.op_masked is False
        assert payload.blueprint_masked is True
