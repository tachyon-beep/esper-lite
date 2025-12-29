"""Test keyboard navigation for decision cards."""

import pytest
from datetime import datetime, timezone

from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo_brain_v2.decisions_column import DecisionCard, DecisionsColumn


def _make_decision(decision_id: str) -> DecisionSnapshot:
    """Create a test decision snapshot."""
    return DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={},
        host_accuracy=75.0,
        chosen_action="GERMINATE",
        chosen_slot="r0c0",
        confidence=0.8,
        expected_value=0.5,
        actual_reward=None,
        alternatives=[],
        decision_id=decision_id,
        pinned=False,
    )


class TestDecisionCardKeyboard:
    """Test keyboard navigation for decision cards."""

    @pytest.mark.asyncio
    async def test_p_key_toggles_pin(self):
        """Pressing 'p' on focused card should toggle pin."""
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DecisionsColumn(id="decisions")

        app = TestApp()
        async with app.run_test() as pilot:
            col = app.query_one(DecisionsColumn)

            decision = _make_decision("test-1")
            snapshot = SanctumSnapshot(
                tamiyo=TamiyoState(recent_decisions=[decision]),
                current_batch=60,
            )
            col.update_snapshot(snapshot)
            await pilot.pause()

            cards = list(col.query(DecisionCard))
            assert len(cards) == 1
            cards[0].focus()
            await pilot.pause()

            await pilot.press("p")
            await pilot.pause()

            # Verify pin message was posted (card should now be pinned)
            # The on_decision_card_pinned handler toggles the pin state
            assert col._displayed_decisions[0].pinned is True

    @pytest.mark.asyncio
    async def test_j_k_navigation(self):
        """j/k should navigate between cards."""
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DecisionsColumn(id="decisions")

        app = TestApp()
        async with app.run_test() as pilot:
            col = app.query_one(DecisionsColumn)

            # Directly populate displayed_decisions to bypass throttling
            decisions = [_make_decision(f"test-{i}") for i in range(3)]
            col._displayed_decisions = decisions
            col._render_cards()
            await pilot.pause()

            cards = list(col.query(DecisionCard))
            assert len(cards) == 3

            # Focus first card
            cards[0].focus()
            await pilot.pause()
            assert app.focused == cards[0]

            # j moves to next
            await pilot.press("j")
            await pilot.pause()
            assert app.focused == cards[1]

            # k moves back
            await pilot.press("k")
            await pilot.pause()
            assert app.focused == cards[0]

    @pytest.mark.asyncio
    async def test_j_k_wraps_around(self):
        """j/k should wrap around at list boundaries."""
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DecisionsColumn(id="decisions")

        app = TestApp()
        async with app.run_test() as pilot:
            col = app.query_one(DecisionsColumn)

            # Directly populate displayed_decisions to bypass throttling
            decisions = [_make_decision(f"test-{i}") for i in range(3)]
            col._displayed_decisions = decisions
            col._render_cards()
            await pilot.pause()

            cards = list(col.query(DecisionCard))
            assert len(cards) == 3

            # Focus last card
            cards[2].focus()
            await pilot.pause()
            assert app.focused == cards[2]

            # j wraps to first
            await pilot.press("j")
            await pilot.pause()
            assert app.focused == cards[0]

            # k wraps to last
            await pilot.press("k")
            await pilot.pause()
            assert app.focused == cards[2]

    @pytest.mark.asyncio
    async def test_decision_card_can_focus(self):
        """DecisionCard should be focusable."""
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DecisionsColumn(id="decisions")

        app = TestApp()
        async with app.run_test() as pilot:
            col = app.query_one(DecisionsColumn)

            decision = _make_decision("test-1")
            snapshot = SanctumSnapshot(
                tamiyo=TamiyoState(recent_decisions=[decision]),
                current_batch=60,
            )
            col.update_snapshot(snapshot)
            await pilot.pause()

            cards = list(col.query(DecisionCard))
            assert len(cards) == 1

            # Card should be focusable
            assert cards[0].can_focus is True
            cards[0].focus()
            await pilot.pause()
            assert app.focused == cards[0]
