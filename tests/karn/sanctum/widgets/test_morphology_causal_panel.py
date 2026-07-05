"""Tests for MorphologyCausalPanel (the growth causal-chain render)."""
from esper.karn.sanctum.schema import MorphologyCausalLogEntry, SanctumSnapshot
from esper.karn.sanctum.widgets.morphology_causal_panel import MorphologyCausalPanel


def _entry(phase: str, *, env_id: int = 0, slot_id: str = "r0c0",
           operation: str = "GERMINATE", approved: bool | None = None,
           blocked: str | None = None, evidence: float | None = None
           ) -> MorphologyCausalLogEntry:
    return MorphologyCausalLogEntry(
        phase=phase, env_id=env_id, slot_id=slot_id, operation=operation,
        action_id="a1", proposal_id="p1", verdict_id="v1", mutation_id="m1",
        observation_hash="h", rng_stream="s", rng_seed=1, topology="t",
        governor_approved=approved, governor_blocked_factor=blocked,
        watch_window_evidence=evidence,
    )


def test_empty_log_shows_hint_not_dashes():
    panel = MorphologyCausalPanel()
    panel.update_snapshot(SanctumSnapshot())
    out = panel.render()
    assert "No morphology events yet" in out.plain


def test_renders_recent_entries_newest_first():
    snap = SanctumSnapshot()
    snap.morphology_causal_log = [
        _entry("proposal"),
        _entry("verdict", approved=True),
        _entry("commit", operation="FOSSILIZE"),
    ]
    panel = MorphologyCausalPanel()
    panel.update_snapshot(snap)
    table = panel.render()
    # A Rich Table is returned (not the empty-hint Text).
    assert table.__class__.__name__ == "Table"
    assert table.row_count == 3


def test_governor_blocked_is_surfaced():
    snap = SanctumSnapshot()
    snap.morphology_causal_log = [
        _entry("verdict", approved=False, blocked="param_budget"),
    ]
    panel = MorphologyCausalPanel()
    panel.update_snapshot(snap)
    # Render to a plain string via the console to assert the blocked factor shows.
    from rich.console import Console
    console = Console(width=120)
    with console.capture() as cap:
        console.print(panel.render())
    text = cap.get()
    assert "param_budget" in text
    assert "✗" in text


def test_caps_at_max_rows():
    snap = SanctumSnapshot()
    snap.morphology_causal_log = [_entry("proposal") for _ in range(30)]
    panel = MorphologyCausalPanel()
    panel.update_snapshot(snap)
    table = panel.render()
    assert table.row_count == 12  # _MAX_ROWS
