"""Tests for MORPHOLOGY_CAUSAL_LOG handling in SanctumAggregator (UI-001).

Processing a MORPHOLOGY_CAUSAL_LOG event must add structured snapshot state
carrying the full causal/identity chain: action/proposal/verdict/mutation IDs,
phase, watch-window evidence, and the linked terminal event.
"""

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.schema import MorphologyCausalLogEntry
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import MorphologyCausalLogPayload


def _make_payload(**overrides) -> MorphologyCausalLogPayload:
    base = dict(
        phase="verdict",
        env_id=2,
        slot_id="r0c0",
        operation="GERMINATE",
        action_id="act-123",
        proposal_id="prop-456",
        verdict_id="ver-789",
        mutation_id="mut-abc",
        observation_hash="obs-deadbeef",
        rng_stream="kasmina:germinate",
        rng_seed=4242,
        topology="conv_heavy@r0c0",
        blueprint_id="conv_heavy",
        governor_approved=True,
        governor_reason="watch_evidence_sufficient",
        governor_blocked_factor=None,
        watch_window_evidence=0.87,
        linked_event_id="evt-seed-germinated-1",
    )
    base.update(overrides)
    return MorphologyCausalLogPayload(**base)


def test_morphology_causal_log_populates_snapshot_fields():
    """A causal-log event must add an entry with the exact causal/identity fields."""
    agg = SanctumAggregator(num_envs=4)

    payload = _make_payload()
    agg.process_event(
        TelemetryEvent(
            event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
            data=payload,
        )
    )

    snapshot = agg.get_snapshot()

    assert len(snapshot.morphology_causal_log) == 1
    entry = snapshot.morphology_causal_log[0]
    assert isinstance(entry, MorphologyCausalLogEntry)

    # Identity chain - exact values preserved
    assert entry.action_id == "act-123"
    assert entry.proposal_id == "prop-456"
    assert entry.verdict_id == "ver-789"
    assert entry.mutation_id == "mut-abc"

    # Phase
    assert entry.phase == "verdict"

    # Watch evidence
    assert entry.watch_window_evidence == 0.87

    # Linked terminal event
    assert entry.linked_event_id == "evt-seed-germinated-1"

    # Other carried causal/identity fields
    assert entry.env_id == 2
    assert entry.slot_id == "r0c0"
    assert entry.operation == "GERMINATE"
    assert entry.observation_hash == "obs-deadbeef"
    assert entry.rng_stream == "kasmina:germinate"
    assert entry.rng_seed == 4242
    assert entry.topology == "conv_heavy@r0c0"
    assert entry.blueprint_id == "conv_heavy"
    assert entry.governor_approved is True
    assert entry.governor_reason == "watch_evidence_sufficient"
    assert entry.governor_blocked_factor is None


def test_morphology_causal_log_preserves_phase_chain_order():
    """Multiple phases for one action join in arrival order (most recent last)."""
    agg = SanctumAggregator(num_envs=4)

    for phase, mut in (("proposal", ""), ("verdict", ""), ("mutation", "mut-1"), ("commit", "mut-1")):
        agg.process_event(
            TelemetryEvent(
                event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                data=_make_payload(phase=phase, mutation_id=mut, action_id="act-shared"),
            )
        )

    snapshot = agg.get_snapshot()
    entries = snapshot.morphology_causal_log
    assert [e.phase for e in entries] == ["proposal", "verdict", "mutation", "commit"]
    # All rows share the same action_id so they can be joined.
    assert {e.action_id for e in entries} == {"act-shared"}


def test_morphology_causal_log_snapshot_is_isolated_copy():
    """Snapshot entries must be a structural copy, not the live aggregator buffer."""
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(
        TelemetryEvent(
            event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
            data=_make_payload(),
        )
    )

    snap1 = agg.get_snapshot()

    # Add a second event; the first snapshot must not change length.
    agg.process_event(
        TelemetryEvent(
            event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
            data=_make_payload(phase="commit"),
        )
    )

    assert len(snap1.morphology_causal_log) == 1
    assert len(agg.get_snapshot().morphology_causal_log) == 2


def test_morphology_causal_log_rejects_wrong_payload():
    """Wrong payload type must fail fast (no defensive swallow)."""
    agg = SanctumAggregator(num_envs=4)
    import pytest

    with pytest.raises(TypeError):
        agg.process_event(
            TelemetryEvent(
                event_type=TelemetryEventType.MORPHOLOGY_CAUSAL_LOG,
                data="not-a-payload",
            )
        )
