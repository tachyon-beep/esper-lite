"""Tests for StatusBanner widget NaN/Inf display."""

import pytest

from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo_brain_v2.status_banner import StatusBanner


# =============================================================================
# NaN/Inf DISPLAY TESTS
# =============================================================================


class TestNaNInfDisplay:
    """Test NaN/Inf gradient count display in status banner."""

    @pytest.fixture
    def banner(self):
        """Create a StatusBanner widget."""
        return StatusBanner()

    @pytest.fixture
    def healthy_snapshot(self):
        """Snapshot with healthy PPO metrics (past warmup period)."""
        tamiyo = TamiyoState(
            ppo_data_received=True,
            entropy=1.0,
            explained_variance=0.5,
            clip_fraction=0.1,
            kl_divergence=0.01,
            advantage_std=1.0,
            grad_norm=0.5,
            nan_grad_count=0,
            inf_grad_count=0,
        )
        return SanctumSnapshot(
            tamiyo=tamiyo,
            current_batch=100,  # Past warmup period (>50)
        )

    def test_no_nan_shows_nothing(self, banner, healthy_snapshot):
        """When nan_grad_count=0 and inf_grad_count=0, no NaN/Inf indicator should appear."""
        # Directly set snapshot and call render without update()
        banner._snapshot = healthy_snapshot
        content = banner._render_content()

        # Convert to plain text for assertion
        plain = content.plain

        assert "NaN" not in plain
        assert "Inf" not in plain

    def test_nan_detected_shows_indicator(self, banner, healthy_snapshot):
        """When nan_grad_count>0, show 'NaN:N' in content."""
        healthy_snapshot.tamiyo.nan_grad_count = 3
        banner._snapshot = healthy_snapshot
        content = banner._render_content()

        plain = content.plain

        assert "NaN:3" in plain

    def test_both_nan_and_inf_shows_both(self, banner, healthy_snapshot):
        """When both NaN and Inf present, show both 'NaN:N' and 'Inf:M'."""
        healthy_snapshot.tamiyo.nan_grad_count = 2
        healthy_snapshot.tamiyo.inf_grad_count = 5
        banner._snapshot = healthy_snapshot
        content = banner._render_content()

        plain = content.plain

        assert "NaN:2" in plain
        assert "Inf:5" in plain

    def test_nan_triggers_critical_status(self, banner, healthy_snapshot):
        """Any NaN should trigger critical status with 'NaN' in label."""
        healthy_snapshot.tamiyo.nan_grad_count = 1
        banner._snapshot = healthy_snapshot
        status, label, style = banner._get_overall_status()

        assert status == "critical"
        assert "NaN" in label
        assert "red" in style

    def test_inf_triggers_critical_status(self, banner, healthy_snapshot):
        """Any Inf should trigger critical status with 'Inf' in label."""
        healthy_snapshot.tamiyo.inf_grad_count = 1
        banner._snapshot = healthy_snapshot
        status, label, style = banner._get_overall_status()

        assert status == "critical"
        assert "Inf" in label
        assert "red" in style

    def test_nan_priority_over_other_critical(self, banner, healthy_snapshot):
        """NaN should override other critical conditions (highest priority)."""
        # Set up another critical condition (entropy collapse)
        healthy_snapshot.tamiyo.entropy = 0.01  # Below ENTROPY_CRITICAL
        healthy_snapshot.tamiyo.nan_grad_count = 1
        banner._snapshot = healthy_snapshot
        status, label, style = banner._get_overall_status()

        # NaN should take priority
        assert status == "critical"
        assert "NaN" in label

    def test_nan_priority_over_inf(self, banner, healthy_snapshot):
        """When both NaN and Inf present, NaN takes priority in status label."""
        healthy_snapshot.tamiyo.nan_grad_count = 1
        healthy_snapshot.tamiyo.inf_grad_count = 1
        banner._snapshot = healthy_snapshot
        status, label, style = banner._get_overall_status()

        # NaN should take priority
        assert status == "critical"
        assert "NaN" in label

    def test_high_nan_count_triggers_reverse_style(self, banner, healthy_snapshot):
        """NaN count >5 should trigger reverse video style for maximum visibility."""
        healthy_snapshot.tamiyo.nan_grad_count = 10
        banner._snapshot = healthy_snapshot
        content = banner._render_content()

        # Check that the NaN span has reverse style
        # We need to inspect the spans, not just plain text
        found_reverse = False
        for span in content.spans:
            span_text = content.plain[span.start:span.end]
            if "NaN" in span_text and "reverse" in str(span.style):
                found_reverse = True
                break

        assert found_reverse, "NaN count >5 should use reverse video style"

    def test_high_inf_count_triggers_reverse_style(self, banner, healthy_snapshot):
        """Inf count >5 should trigger reverse video style for maximum visibility."""
        healthy_snapshot.tamiyo.inf_grad_count = 8
        banner._snapshot = healthy_snapshot
        content = banner._render_content()

        # Check that the Inf span has reverse style
        found_reverse = False
        for span in content.spans:
            span_text = content.plain[span.start:span.end]
            if "Inf" in span_text and "reverse" in str(span.style):
                found_reverse = True
                break

        assert found_reverse, "Inf count >5 should use reverse video style"

    def test_nan_inf_appears_before_status_icon(self, banner, healthy_snapshot):
        """NaN/Inf indicator should appear FIRST (leftmost) for F-pattern visibility."""
        healthy_snapshot.tamiyo.nan_grad_count = 3
        banner._snapshot = healthy_snapshot
        content = banner._render_content()

        plain = content.plain

        # NaN should appear before the status icon (which uses brackets like [x])
        nan_pos = plain.find("NaN")
        bracket_pos = plain.find("[")

        assert nan_pos != -1, "NaN indicator should be present"
        assert bracket_pos != -1, "Status icon should be present"
        assert nan_pos < bracket_pos, "NaN should appear before status icon"
