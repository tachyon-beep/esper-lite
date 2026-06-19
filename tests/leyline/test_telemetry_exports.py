"""Guard tests: dead telemetry budget symbols must not be exported."""

from __future__ import annotations

import esper.leyline


def test_performance_budgets_not_exported() -> None:
    assert "PerformanceBudgets" not in dir(esper.leyline)
    assert "DEFAULT_BUDGETS" not in dir(esper.leyline)


def test_performance_budgets_not_in_all() -> None:
    assert "PerformanceBudgets" not in esper.leyline.__all__
    assert "DEFAULT_BUDGETS" not in esper.leyline.__all__
