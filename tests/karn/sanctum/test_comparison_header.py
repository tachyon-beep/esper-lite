"""Tests for ComparisonHeader widget."""

import pytest
from esper.karn.sanctum.widgets.comparison_header import ComparisonHeader


@pytest.mark.asyncio
async def test_comparison_header_shows_delta():
    """Comparison header should show accuracy delta between policies."""
    from textual.app import App, ComposeResult

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield ComparisonHeader()

    app = TestApp()
    async with app.run_test():
        header = app.query_one(ComparisonHeader)

        header.update_comparison(
            group_a_accuracy=75.0,
            group_b_accuracy=68.0,
            group_a_reward=12.5,
            group_b_reward=10.2,
        )

        rendered = header.render()
        # Rich Text objects have .plain property for plain text version
        plain = rendered.plain

        # Should show delta (75.0 - 68.0 = +7.0%)
        assert "+7.0%" in plain


def test_comparison_header_winner_indication():
    """Header should indicate which policy is leading."""
    header = ComparisonHeader()

    header.update_comparison(
        group_a_accuracy=75.0,
        group_b_accuracy=68.0,
        group_a_reward=12.5,
        group_b_reward=10.2,
    )

    # A has higher reward AND accuracy, should be leader
    assert header.leader == "A"


def test_comparison_header_reward_decisive():
    """Reward should be decisive when significantly different."""
    header = ComparisonHeader()

    # B has lower accuracy but significantly higher reward
    header.update_comparison(
        group_a_accuracy=75.0,
        group_b_accuracy=68.0,
        group_a_reward=10.0,
        group_b_reward=15.0,  # B has 50% higher reward
    )

    # B should lead because reward is the RL objective
    assert header.leader == "B"


def test_comparison_header_tied():
    """Header should show Tied when metrics are equal."""
    header = ComparisonHeader()

    header.update_comparison(
        group_a_accuracy=70.0,
        group_b_accuracy=70.0,
        group_a_reward=10.0,
        group_b_reward=10.0,
    )

    assert header.leader is None
