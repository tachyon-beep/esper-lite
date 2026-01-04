"""Tests for shared Tamiyo trend indicator semantics."""

from esper.karn.sanctum.widgets.tamiyo_brain.trends import trend_arrow_for_history


def test_loss_improving_is_up_green() -> None:
    history = [10.0] * 10 + [5.0] * 10
    arrow, style = trend_arrow_for_history(history, metric_name="policy_loss", metric_type="loss")
    assert arrow == "↑"
    assert style == "green"


def test_loss_worsening_is_down_red() -> None:
    history = [5.0] * 10 + [10.0] * 10
    arrow, style = trend_arrow_for_history(history, metric_name="policy_loss", metric_type="loss")
    assert arrow == "↓"
    assert style == "red"


def test_accuracy_improving_is_up_green() -> None:
    history = [10.0] * 10 + [15.0] * 10
    arrow, style = trend_arrow_for_history(history, metric_name="episode_return", metric_type="accuracy")
    assert arrow == "↑"
    assert style == "green"


def test_small_change_is_stable_dim() -> None:
    history = [10.0] * 10 + [10.5] * 10
    arrow, style = trend_arrow_for_history(history, metric_name="episode_return", metric_type="accuracy")
    assert arrow == "→"
    assert style == "dim"


def test_volatility_is_tilde_yellow() -> None:
    older = [1.0, 1.1] * 5
    recent = [1.0, 3.0] * 5
    history = older + recent
    arrow, style = trend_arrow_for_history(history, metric_name="policy_loss", metric_type="loss")
    assert arrow == "~"
    assert style == "yellow"


def test_short_history_returns_no_arrow() -> None:
    arrow, style = trend_arrow_for_history([1.0, 2.0, 3.0], metric_name="policy_loss", metric_type="loss")
    assert arrow == ""
    assert style == "dim"

