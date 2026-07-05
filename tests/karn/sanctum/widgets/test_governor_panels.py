"""Tests for the GovernorStatusPanel and GovernorLedgerPanel render surfaces."""
from datetime import datetime, timezone

from rich.console import Console

from esper.karn.sanctum.schema import (
    GovernorRollbackRecord,
    GovernorState,
    SanctumSnapshot,
)
from esper.karn.sanctum.widgets.governor_ledger_panel import GovernorLedgerPanel
from esper.karn.sanctum.widgets.governor_status_panel import GovernorStatusPanel


def _plain(renderable) -> str:
    console = Console(width=120)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


# --- status panel -----------------------------------------------------------


def test_status_armed_when_all_envs_past_warmup():
    snap = SanctumSnapshot()
    snap.governor = GovernorState(total_env_count=6, armed_env_count=6, warming_env_count=0)
    panel = GovernorStatusPanel()
    panel.update_snapshot(snap)
    text = _plain(panel.render())
    assert "ARMED" in text
    assert "watching 6 envs" in text
    assert "independent of policy" in text  # invariant made visible


def test_status_warming_is_not_overclaimed():
    snap = SanctumSnapshot()
    snap.governor = GovernorState(total_env_count=6, armed_env_count=2, warming_env_count=4)
    panel = GovernorStatusPanel()
    panel.update_snapshot(snap)
    text = _plain(panel.render())
    assert "warming 4/6" in text
    assert "ARMED" not in text  # must not claim all rules live during warmup


def test_status_shows_rollback_counts_by_reason():
    snap = SanctumSnapshot()
    snap.governor = GovernorState(
        total_env_count=4, armed_env_count=4,
        total_rollbacks=3, rollbacks_by_reason={"governor_nan": 2, "governor_divergence": 1},
    )
    panel = GovernorStatusPanel()
    panel.update_snapshot(snap)
    text = _plain(panel.render())
    assert "governor_nan:2" in text
    assert "governor_divergence:1" in text


# --- ledger panel -----------------------------------------------------------


def test_ledger_empty_reads_positively():
    panel = GovernorLedgerPanel()
    panel.update_snapshot(SanctumSnapshot())
    assert "gate has not fired" in _plain(panel.render())


def test_ledger_renders_rich_fields_newest_first():
    snap = SanctumSnapshot()
    snap.governor.rollback_ledger.append(
        GovernorRollbackRecord(
            env_id=1, epoch=12, timestamp=datetime.now(timezone.utc),
            panic_reason="governor_nan", loss_at_panic=float("nan"),
            consecutive_panics=1, attributed=False,
        )
    )
    snap.governor.rollback_ledger.append(
        GovernorRollbackRecord(
            env_id=2, epoch=40, timestamp=datetime.now(timezone.utc),
            panic_reason="governor_divergence", loss_at_panic=12.5,
            loss_threshold=8.0, consecutive_panics=3,
            triggering_action_id="act-99", attributed=True,
        )
    )
    panel = GovernorLedgerPanel()
    panel.update_snapshot(snap)
    text = _plain(panel.render())
    assert "divergence" in text and "nan" in text  # "governor_" prefix stripped
    assert "12.50" in text          # loss_at_panic for the divergence panic
    assert "8.00" in text           # threshold shown for divergence
    assert "act-99" in text         # attributed action
    assert "(unattr)" in text       # the nan panic was unattributed
    # newest-first: the epoch-40 divergence row precedes the epoch-12 nan row
    assert text.index("divergence") < text.index("nan")


def test_ledger_threshold_hidden_for_non_divergence():
    snap = SanctumSnapshot()
    snap.governor.rollback_ledger.append(
        GovernorRollbackRecord(
            env_id=0, epoch=5, timestamp=None, panic_reason="governor_nan",
            loss_threshold=8.0,  # computed, but was NOT the trip line for a nan panic
        )
    )
    panel = GovernorLedgerPanel()
    panel.update_snapshot(snap)
    # threshold is not rendered for a nan panic (only divergence uses it)
    assert "8.00" not in _plain(panel.render())
