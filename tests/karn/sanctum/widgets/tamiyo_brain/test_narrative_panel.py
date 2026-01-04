"""Tests for NarrativePanel NOW/WHY/NEXT display."""

import pytest

from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo_brain.narrative_panel import NarrativePanel


# =============================================================================
# NaN/Inf DISPLAY TESTS
# =============================================================================


class TestNaNInfDisplay:
    """Test NaN/Inf gradient count display in narrative NOW section."""

    @pytest.fixture
    def panel(self):
        """Create a NarrativePanel widget."""
        return NarrativePanel()

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

    def test_no_nan_shows_nothing(self, panel, healthy_snapshot):
        """When nan_grad_count=0 and inf_grad_count=0, no NaN/Inf indicator should appear."""
        # Directly set snapshot and call render without update()
        panel._snapshot = healthy_snapshot
        content = panel.render()

        # Convert to plain text for assertion
        plain = content.plain

        assert "NaN" not in plain
        assert "Inf" not in plain

    def test_nan_detected_shows_indicator(self, panel, healthy_snapshot):
        """When nan_grad_count>0, show 'NaN:N' in content."""
        healthy_snapshot.tamiyo.nan_grad_count = 3
        panel._snapshot = healthy_snapshot
        content = panel.render()

        plain = content.plain

        assert "NaN:3" in plain

    def test_both_nan_and_inf_shows_both(self, panel, healthy_snapshot):
        """When both NaN and Inf present, show both 'NaN:N' and 'Inf:M'."""
        healthy_snapshot.tamiyo.nan_grad_count = 2
        healthy_snapshot.tamiyo.inf_grad_count = 5
        panel._snapshot = healthy_snapshot
        content = panel.render()

        plain = content.plain

        assert "NaN:2" in plain
        assert "Inf:5" in plain

    def test_nan_triggers_critical_status(self, panel, healthy_snapshot):
        """Any NaN should trigger critical status with 'NaN' in label."""
        healthy_snapshot.tamiyo.nan_grad_count = 1
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "critical"
        assert "NaN" in label
        assert "red" in style

    def test_inf_triggers_critical_status(self, panel, healthy_snapshot):
        """Any Inf should trigger critical status with 'Inf' in label."""
        healthy_snapshot.tamiyo.inf_grad_count = 1
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "critical"
        assert "Inf" in label
        assert "red" in style

    def test_nan_priority_over_other_critical(self, panel, healthy_snapshot):
        """NaN should override other critical conditions (highest priority)."""
        # Set up another critical condition (entropy collapse)
        healthy_snapshot.tamiyo.entropy = 0.01  # Below ENTROPY_CRITICAL
        healthy_snapshot.tamiyo.nan_grad_count = 1
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        # NaN should take priority
        assert status == "critical"
        assert "NaN" in label

    def test_nan_priority_over_inf(self, panel, healthy_snapshot):
        """When both NaN and Inf present, NaN takes priority in status label."""
        healthy_snapshot.tamiyo.nan_grad_count = 1
        healthy_snapshot.tamiyo.inf_grad_count = 1
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        # NaN should take priority
        assert status == "critical"
        assert "NaN" in label

    def test_high_nan_count_triggers_reverse_style(self, panel, healthy_snapshot):
        """NaN count >5 should trigger reverse video style for maximum visibility."""
        healthy_snapshot.tamiyo.nan_grad_count = 10
        panel._snapshot = healthy_snapshot
        content = panel.render()

        # Check that the NaN span has reverse style
        # We need to inspect the spans, not just plain text
        found_reverse = False
        for span in content.spans:
            span_text = content.plain[span.start:span.end]
            if "NaN" in span_text and "reverse" in str(span.style):
                found_reverse = True
                break

        assert found_reverse, "NaN count >5 should use reverse video style"

    def test_high_inf_count_triggers_reverse_style(self, panel, healthy_snapshot):
        """Inf count >5 should trigger reverse video style for maximum visibility."""
        healthy_snapshot.tamiyo.inf_grad_count = 8
        panel._snapshot = healthy_snapshot
        content = panel.render()

        # Check that the Inf span has reverse style
        found_reverse = False
        for span in content.spans:
            span_text = content.plain[span.start:span.end]
            if "Inf" in span_text and "reverse" in str(span.style):
                found_reverse = True
                break

        assert found_reverse, "Inf count >5 should use reverse video style"

    def test_nan_inf_appears_before_status_icon(self, panel, healthy_snapshot):
        """NaN/Inf indicator should appear FIRST (leftmost) for F-pattern visibility."""
        healthy_snapshot.tamiyo.nan_grad_count = 3
        panel._snapshot = healthy_snapshot
        content = panel.render()

        plain = content.plain

        # NaN should appear before the status icon (which uses brackets like [x])
        nan_pos = plain.find("NaN")
        bracket_pos = plain.find("[")

        assert nan_pos != -1, "NaN indicator should be present"
        assert bracket_pos != -1, "Status icon should be present"
        assert nan_pos < bracket_pos, "NaN should appear before status icon"


# =============================================================================
# TRIGGERING CONDITION DISPLAY TESTS
# =============================================================================


class TestTriggeringCondition:
    """Test that narrative NOW section shows trigger in status label."""

    @pytest.fixture
    def panel(self):
        """Create a NarrativePanel widget."""
        return NarrativePanel()

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

    def test_entropy_warning_shows_reason(self, panel, healthy_snapshot):
        """Low entropy (warning level) should show 'Entropy' in label."""
        # Set entropy below WARNING threshold (0.3) but above CRITICAL (0.1)
        healthy_snapshot.tamiyo.entropy = 0.2
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "warning"
        assert "Entropy" in label
        assert "WARN" in label

    def test_clip_critical_shows_reason(self, panel, healthy_snapshot):
        """High clip fraction (critical level) should show 'Clip' in label."""
        # Set clip above CRITICAL threshold (0.3)
        healthy_snapshot.tamiyo.clip_fraction = 0.35
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "critical"
        assert "Clip" in label
        assert "FAIL" in label

    def test_multiple_issues_shows_count(self, panel, healthy_snapshot):
        """Multiple critical issues should show count like 'FAIL:Entropy (+2)'."""
        # Set multiple critical conditions:
        # 1. Entropy critical (<0.1)
        # 2. Clip critical (>0.3)
        # 3. KL critical (>0.03)
        healthy_snapshot.tamiyo.entropy = 0.05  # Below ENTROPY_CRITICAL
        healthy_snapshot.tamiyo.clip_fraction = 0.35  # Above CLIP_CRITICAL
        healthy_snapshot.tamiyo.kl_divergence = 0.04  # Above KL_CRITICAL
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "critical"
        assert "FAIL" in label
        # First issue should be named, +2 indicates 2 additional issues
        assert "(+2)" in label

    def test_single_critical_no_count(self, panel, healthy_snapshot):
        """Single critical issue should not show count."""
        healthy_snapshot.tamiyo.entropy = 0.05  # Below ENTROPY_CRITICAL
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "critical"
        assert "FAIL:Entropy" in label
        assert "(+" not in label

    def test_single_warning_no_count(self, panel, healthy_snapshot):
        """Single warning should not show count."""
        healthy_snapshot.tamiyo.entropy = 0.2  # Below WARNING (0.3), above CRITICAL (0.1)
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "warning"
        assert "WARN:Entropy" in label
        assert "(+" not in label

    def test_multiple_warnings_shows_count(self, panel, healthy_snapshot):
        """Multiple warnings should show count."""
        # Set multiple warning conditions (but not critical):
        # 1. Entropy warning (<0.3)
        # 2. Clip warning (>0.25)
        healthy_snapshot.tamiyo.entropy = 0.2  # Below ENTROPY_WARNING (0.3)
        healthy_snapshot.tamiyo.clip_fraction = 0.27  # Above CLIP_WARNING (0.25)
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "warning"
        assert "WARN" in label
        assert "(+1)" in label

    def test_healthy_shows_learning(self, panel, healthy_snapshot):
        """Healthy metrics should show 'LEARNING' with no condition name."""
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "ok"
        assert label == "LEARNING"
        assert "FAIL" not in label
        assert "WARN" not in label

    def test_nan_still_takes_priority(self, panel, healthy_snapshot):
        """NaN/Inf should still take priority over other conditions."""
        # Set both NaN and other critical conditions
        healthy_snapshot.tamiyo.nan_grad_count = 1
        healthy_snapshot.tamiyo.entropy = 0.05
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        # NaN should still win
        assert status == "critical"
        assert "NaN" in label
        # The new format should not apply to NaN/Inf
        assert "FAIL:" not in label

    def test_value_critical_shows_reason(self, panel, healthy_snapshot):
        """Explained variance critical should show 'Value' in label."""
        healthy_snapshot.tamiyo.explained_variance = -0.1  # Below CRITICAL (0.0)
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "critical"
        assert "Value" in label
        assert "FAIL" in label

    def test_grad_warning_shows_reason(self, panel, healthy_snapshot):
        """High grad norm warning should show 'Grad' in label."""
        healthy_snapshot.tamiyo.grad_norm = 6.0  # Above WARNING (5.0), below CRITICAL (10.0)
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "warning"
        assert "Grad" in label
        assert "WARN" in label

    def test_adv_low_warning_shows_reason(self, panel, healthy_snapshot):
        """Low advantage std warning should show 'AdvLow' in label."""
        healthy_snapshot.tamiyo.advantage_std = 0.3  # Below LOW_WARNING (0.5), above COLLAPSED (0.1)
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "warning"
        assert "AdvLow" in label
        assert "WARN" in label

    def test_adv_high_warning_shows_reason(self, panel, healthy_snapshot):
        """High advantage std warning should show 'AdvHigh' in label."""
        healthy_snapshot.tamiyo.advantage_std = 2.5  # Above WARNING (2.0), below CRITICAL (3.0)
        panel._snapshot = healthy_snapshot
        status, label, style = panel._get_overall_status()

        assert status == "warning"
        assert "AdvHigh" in label
        assert "WARN" in label


# =============================================================================
# MEMORY PERCENTAGE DISPLAY TESTS
# =============================================================================


class TestMemoryPercentageDisplay:
    """Test memory percentage display in narrative NOW section."""

    @pytest.fixture
    def panel(self):
        """Create a NarrativePanel widget."""
        return NarrativePanel()

    def test_shows_memory_percentage(self, panel):
        """Status banner should display memory as percentage, not absolute."""
        from esper.karn.sanctum.schema import InfrastructureMetrics

        snapshot = SanctumSnapshot()
        snapshot.tamiyo = TamiyoState(ppo_data_received=True)
        snapshot.tamiyo.infrastructure = InfrastructureMetrics(
            cuda_memory_allocated_gb=4.2,
            cuda_memory_reserved_gb=8.0,
        )
        snapshot.current_batch = 60

        panel._snapshot = snapshot
        content = panel.render()

        content_str = content.plain
        # Should show percentage (52%), not absolute values
        assert "52%" in content_str or "53%" in content_str  # Allow rounding
        # Should NOT show absolute GB values in banner
        assert "4.2/8.0" not in content_str


# =============================================================================
# COMPILE IN TITLE TESTS
# =============================================================================


class TestCompileInTitle:
    """Test compile indicator in narrative panel title (per UX review)."""

    @pytest.fixture
    def panel(self):
        """Create a NarrativePanel widget."""
        return NarrativePanel()

    def test_compile_in_title(self, panel):
        """Compile indicator should be in border title."""
        from esper.karn.sanctum.schema import InfrastructureMetrics

        snapshot = SanctumSnapshot()
        snapshot.tamiyo = TamiyoState(ppo_data_received=True)
        snapshot.tamiyo.infrastructure = InfrastructureMetrics(compile_enabled=True)
        snapshot.current_batch = 60

        panel.update_snapshot(snapshot)

        # Compile indicator should be in border_title
        # Note: Textual escapes [ as [[ in markup, so border_title contains [[compiled]]
        assert "compiled" in panel.border_title.lower()

    def test_no_compile_in_title_when_disabled(self, panel):
        """No compile indicator when compile is disabled."""
        from esper.karn.sanctum.schema import InfrastructureMetrics

        snapshot = SanctumSnapshot()
        snapshot.tamiyo = TamiyoState(ppo_data_received=True)
        snapshot.tamiyo.infrastructure = InfrastructureMetrics(compile_enabled=False)
        snapshot.current_batch = 60

        panel.update_snapshot(snapshot)

        # Should not have compile indicator
        assert "compiled" not in panel.border_title.lower()
        assert "narrative" in panel.border_title.lower()
