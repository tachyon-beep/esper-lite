"""GovernorState: rollback ledger + counters + defect fix (371871)."""
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import GovernorRollbackPayload, PPOUpdatePayload


def _rollback_event(**overrides: object) -> TelemetryEvent:
    kwargs: dict[str, object] = dict(env_id=0, device="cuda:0", reason="Structural Collapse")
    kwargs.update(overrides)
    return TelemetryEvent(
        event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
        data=GovernorRollbackPayload(**kwargs),  # type: ignore[arg-type]
    )


def test_rollback_builds_durable_ledger_record():
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(
        _rollback_event(
            env_id=2, panic_reason="governor_divergence",
            loss_at_panic=12.5, loss_threshold=8.0, consecutive_panics=3,
            triggering_action_id="act-99", rollback_severity=50.0,
        )
    )
    gov = agg.get_snapshot().governor
    assert gov.total_rollbacks == 1
    assert gov.rollbacks_by_reason["governor_divergence"] == 1
    rec = gov.rollback_ledger[-1]
    assert rec.env_id == 2
    assert rec.panic_reason == "governor_divergence"
    assert rec.loss_at_panic == 12.5
    assert rec.loss_threshold == 8.0
    assert rec.consecutive_panics == 3
    assert rec.triggering_action_id == "act-99"
    assert rec.attributed is True  # triggering_action_id present
    assert rec.rollback_severity == 50.0


def test_rollback_reason_defect_fixed():
    """env.rollback_reason must be the rich panic_reason, not the banner string."""
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(_rollback_event(env_id=1, panic_reason="governor_nan"))
    snap = agg.get_snapshot()
    assert snap.envs[1].rollback_reason == "governor_nan"  # not "Structural Collapse"
    assert snap.envs[1].rolled_back is True


def test_rollback_reason_falls_back_when_panic_reason_absent():
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(_rollback_event(env_id=0, panic_reason=None))
    assert agg.get_snapshot().envs[0].rollback_reason == "Structural Collapse"


def test_unattributed_rollback_flag():
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(_rollback_event(env_id=0, triggering_action_id=None))
    assert agg.get_snapshot().governor.rollback_ledger[-1].attributed is False


def test_counters_live_on_governor_not_policy():
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(
        TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=PPOUpdatePayload(
                policy_loss=0.1, value_loss=0.2, entropy=1.0, grad_norm=0.5,
                kl_divergence=0.01, clip_fraction=0.1, nan_grad_count=0,
                rollback_attempt_count=4, rollback_unattributed_count=2,
            ),  # type: ignore[arg-type]
        )
    )
    gov = agg.get_snapshot().governor
    assert gov.rollback_attempt_count == 4
    assert gov.rollback_unattributed_count == 2
    # The policy object no longer carries these (moved off TamiyoState).
    assert not hasattr(agg.get_snapshot().tamiyo, "rollback_attempt_count")


def test_ledger_survives_snapshot_copy_independently():
    """The copied snapshot's ledger must not alias the live aggregator's."""
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(_rollback_event(env_id=0, panic_reason="governor_nan"))
    snap = agg.get_snapshot()
    agg.process_event(_rollback_event(env_id=1, panic_reason="governor_divergence"))
    # The earlier snapshot copy is frozen at 1 record; the live state has 2.
    assert len(snap.governor.rollback_ledger) == 1
    assert agg.get_snapshot().governor.total_rollbacks == 2
