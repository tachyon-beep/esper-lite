"""Tests for Nissa Blueprint Analytics."""

import pytest
from esper.nissa.analytics import BlueprintStats, SeedScoreboard


class TestBlueprintStats:
    """Tests for BlueprintStats dataclass."""

    def test_initial_values(self):
        """Stats start at zero."""
        stats = BlueprintStats()
        assert stats.germinated == 0
        assert stats.fossilized == 0
        assert stats.culled == 0
        assert stats.acc_deltas == []
        assert stats.churns == []

    def test_mean_acc_delta_empty(self):
        """Empty acc_deltas returns 0."""
        stats = BlueprintStats()
        assert stats.mean_acc_delta == 0.0

    def test_mean_acc_delta_with_values(self):
        """Mean accuracy delta calculated correctly."""
        stats = BlueprintStats(acc_deltas=[1.0, 2.0, 3.0])
        assert stats.mean_acc_delta == 2.0

    def test_fossilization_rate_no_terminal(self):
        """Rate is 0% when no seeds reached terminal state."""
        stats = BlueprintStats(germinated=5)
        assert stats.fossilization_rate == 0.0

    def test_fossilization_rate_all_fossilized(self):
        """Rate is 100% when all seeds fossilized."""
        stats = BlueprintStats(germinated=5, fossilized=5, culled=0)
        assert stats.fossilization_rate == 100.0

    def test_fossilization_rate_mixed(self):
        """Rate calculated correctly for mixed outcomes."""
        stats = BlueprintStats(germinated=10, fossilized=3, culled=7)
        assert stats.fossilization_rate == 30.0


class TestSeedScoreboard:
    """Tests for SeedScoreboard dataclass."""

    def test_initial_values(self):
        """Scoreboard starts empty."""
        sb = SeedScoreboard()
        assert sb.total_germinated == 0
        assert sb.total_fossilized == 0
        assert sb.params_added == 0
        assert sb.live_blueprint is None

    def test_compute_cost_empty(self):
        """Empty scoreboard has 1.0x cost."""
        sb = SeedScoreboard()
        assert sb.compute_cost == 1.0

    def test_compute_cost_with_fossilized(self):
        """Compute cost accumulates from fossilized seeds."""
        sb = SeedScoreboard()
        sb.fossilized_by_blueprint["depthwise"] = 2  # 2 * 0.08 = 0.16 extra
        sb.fossilized_by_blueprint["attention"] = 1  # 1 * 0.35 = 0.35 extra
        # Total: 1.0 + 0.16 + 0.35 = 1.51
        assert abs(sb.compute_cost - 1.51) < 0.01

    def test_params_percentage(self):
        """Params percentage calculated correctly."""
        sb = SeedScoreboard(params_added=10000, host_params=100000)
        assert sb.params_percentage == 10.0
