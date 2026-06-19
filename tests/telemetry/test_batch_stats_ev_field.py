"""SLICE D — batch_stats ANALYTICS_SNAPSHOT explained_variance mislabel fix (Step 8).

Locks the contract that the batch_stats analytics snapshot carries pre-update
explained_variance under its honest field name `explained_variance` (NOT the
historical mislabel `value_variance`), and that when the PPO metrics dict has no
explained_variance (skipped / empty update) the field is emitted as None — an
honest "no EV this batch" — rather than a fabricated 0.0.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from esper.leyline import AnalyticsSnapshotPayload, TelemetryEventType
from esper.simic.telemetry.emitters import VectorizedEmitter


def _env_state() -> SimpleNamespace:
    return SimpleNamespace(
        seeds_created=0,
        seeds_fossilized=0,
        action_counts={},
        successful_action_counts={},
    )


def _call_on_batch_completed(emitter: VectorizedEmitter, metrics: dict) -> None:
    emitter.on_batch_completed(
        batch_idx=0,
        episodes_completed=4,
        rolling_avg_acc=0.5,
        avg_acc=0.5,
        metrics=metrics,
        env_states=[_env_state()],
        update_skipped=False,
        plateau_threshold=0.001,
        improvement_threshold=0.01,
        prev_rolling_avg_acc=None,
        total_episodes=100,
        start_episode=0,
        n_episodes=100,
        env_final_accs=[0.5],
        avg_reward=0.1,
        train_losses=[0.5],
        train_corrects=[5],
        train_totals=[10],
        val_losses=[0.5],
        val_corrects=[5],
        val_totals=[10],
        num_train_batches=1,
        num_test_batches=1,
    )


def _batch_stats_payload(backend) -> AnalyticsSnapshotPayload:
    events = backend.find_events(TelemetryEventType.ANALYTICS_SNAPSHOT)
    batch_stats = [
        e.data
        for e in events
        if isinstance(e.data, AnalyticsSnapshotPayload) and e.data.kind == "batch_stats"
    ]
    assert len(batch_stats) == 1, f"expected one batch_stats snapshot, got {len(batch_stats)}"
    return batch_stats[0]


def test_field_is_renamed_to_explained_variance() -> None:
    """The mislabeled `value_variance` field must be gone; EV lives under its real name."""
    field_names = {f.name for f in dataclasses.fields(AnalyticsSnapshotPayload)}
    assert "explained_variance" in field_names
    assert "value_variance" not in field_names, (
        "value_variance is a mislabel for explained_variance; it must be deleted, not kept"
    )


def test_batch_stats_carries_explained_variance(capture_hub) -> None:
    hub, backend = capture_hub
    emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="A", hub=hub)

    _call_on_batch_completed(emitter, {"explained_variance": 0.42})
    hub.flush()

    payload = _batch_stats_payload(backend)
    assert payload.explained_variance == 0.42


def test_batch_stats_explained_variance_is_none_when_absent(capture_hub) -> None:
    """A skipped/empty update has no EV; emit None, not a fabricated 0.0 (no bug-hiding)."""
    hub, backend = capture_hub
    emitter = VectorizedEmitter(env_id=0, device="cpu", group_id="A", hub=hub)

    _call_on_batch_completed(emitter, {})
    hub.flush()

    payload = _batch_stats_payload(backend)
    assert payload.explained_variance is None
