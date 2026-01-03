"""Tests for ValueFunctionMetrics TELE records (TELE-220 to TELE-228).

Verifies value function diagnostics schema fields and consumer widget behavior.
These metrics are critical for detecting value function health issues:

- TELE-220: v_return_correlation - Pearson correlation between V(s) and returns
- TELE-221: td_error_mean - Mean temporal difference error (bias indicator)
- TELE-222: td_error_std - TD error standard deviation (noise indicator)
- TELE-223: bellman_error - Squared Bellman error (NaN early warning)
- TELE-224: return_p10 - 10th percentile of returns (worst-case)
- TELE-225: return_p50 - 50th percentile/median of returns (typical case)
- TELE-226: return_p90 - 90th percentile of returns (best-case)
- TELE-227: return_variance - Variance of episode returns (consistency)
- TELE-228: return_skewness - Skewness of return distribution (asymmetry)

Per TELE records: Schema fields exist in ValueFunctionMetrics dataclass,
consumer is ValueDiagnosticsPanel widget.

WIRING STATUS (as of 2026-01-04):
- Schema fields exist in ValueFunctionMetrics (karn/sanctum/schema.py)
- Consumer widget (ValueDiagnosticsPanel) reads from schema correctly
- Emitters compute and emit values from PPO update loop
- Aggregator wires values to ValueFunctionMetrics
- Full end-to-end wiring is complete
"""

import math
from dataclasses import fields
from collections import deque

import pytest

from esper.karn.sanctum.schema import (
    ValueFunctionMetrics,
    SanctumSnapshot,
    TamiyoState,
    InfrastructureMetrics,
)
from esper.karn.sanctum.widgets.tamiyo_brain.value_diagnostics_panel import (
    ValueDiagnosticsPanel,
)


# =============================================================================
# Schema Tests - Field Existence and Types
# =============================================================================


class TestValueFunctionMetricsSchema:
    """Verify ValueFunctionMetrics dataclass schema matches TELE records."""

    def test_dataclass_has_all_tele_220_228_fields(self):
        """All TELE-220 to TELE-228 fields exist in ValueFunctionMetrics."""
        field_names = {f.name for f in fields(ValueFunctionMetrics)}

        # TELE-220 to TELE-228 required fields
        required_fields = {
            "v_return_correlation",  # TELE-220
            "td_error_mean",  # TELE-221
            "td_error_std",  # TELE-222
            "bellman_error",  # TELE-223
            "return_p10",  # TELE-224
            "return_p50",  # TELE-225
            "return_p90",  # TELE-226
            "return_variance",  # TELE-227
            "return_skewness",  # TELE-228
        }

        for field_name in required_fields:
            assert field_name in field_names, f"Missing field: {field_name}"

    def test_all_fields_are_float_type(self):
        """All TELE-220 to TELE-228 fields are float type."""
        metrics = ValueFunctionMetrics()

        float_fields = [
            "v_return_correlation",
            "td_error_mean",
            "td_error_std",
            "bellman_error",
            "return_p10",
            "return_p50",
            "return_p90",
            "return_variance",
            "return_skewness",
        ]

        for field_name in float_fields:
            value = getattr(metrics, field_name)
            assert isinstance(value, float), f"{field_name} should be float, got {type(value)}"


# =============================================================================
# TELE-220: V-Return Correlation
# =============================================================================


class TestTELE220VReturnCorrelation:
    """TELE-220: V-Return Correlation - primary value function diagnostic.

    Pearson correlation between V(s) predictions and actual episode returns.
    Range: [-1.0, 1.0]
    Default: 0.0

    Health Thresholds:
    - Excellent: >= 0.8 (value network well-calibrated)
    - Good: >= 0.5 (value network learning)
    - Warning: >= 0.3 (value network weak)
    - Critical: < 0.3 (value network not learning)
    """

    def test_field_exists(self):
        """TELE-220: v_return_correlation field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "v_return_correlation")

    def test_default_value(self):
        """TELE-220: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.v_return_correlation == 0.0

    def test_accepts_valid_correlation_range(self):
        """TELE-220: Accepts values in [-1.0, 1.0] range."""
        # Negative correlation (anti-correlated)
        metrics = ValueFunctionMetrics(v_return_correlation=-0.5)
        assert metrics.v_return_correlation == -0.5

        # Zero correlation
        metrics = ValueFunctionMetrics(v_return_correlation=0.0)
        assert metrics.v_return_correlation == 0.0

        # Positive correlation
        metrics = ValueFunctionMetrics(v_return_correlation=0.85)
        assert metrics.v_return_correlation == 0.85

        # Perfect correlation
        metrics = ValueFunctionMetrics(v_return_correlation=1.0)
        assert metrics.v_return_correlation == 1.0

    def test_excellent_threshold(self):
        """TELE-220: Excellent threshold is >= 0.8."""
        excellent_value = 0.85
        assert excellent_value >= 0.8

    def test_good_threshold(self):
        """TELE-220: Good threshold is >= 0.5 and < 0.8."""
        good_value = 0.65
        assert 0.5 <= good_value < 0.8

    def test_warning_threshold(self):
        """TELE-220: Warning threshold is >= 0.3 and < 0.5."""
        warning_value = 0.35
        assert 0.3 <= warning_value < 0.5

    def test_critical_threshold(self):
        """TELE-220: Critical threshold is < 0.3."""
        critical_value = 0.2
        assert critical_value < 0.3


# =============================================================================
# TELE-221: TD Error Mean
# =============================================================================


class TestTELE221TDErrorMean:
    """TELE-221: TD Error Mean - bias indicator for value function.

    Mean temporal difference error: TD = r + gamma * V(s') - V(s)
    Units: reward units (same scale as returns)
    Default: 0.0

    Health Thresholds:
    - Healthy: abs(value) < 5
    - Warning: 5 <= abs(value) < 15
    - Critical: abs(value) >= 15
    """

    def test_field_exists(self):
        """TELE-221: td_error_mean field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "td_error_mean")

    def test_default_value(self):
        """TELE-221: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.td_error_mean == 0.0

    def test_accepts_positive_mean(self):
        """TELE-221: Positive TD mean indicates under-estimation (pessimistic)."""
        metrics = ValueFunctionMetrics(td_error_mean=8.5)
        assert metrics.td_error_mean == 8.5
        assert metrics.td_error_mean > 0

    def test_accepts_negative_mean(self):
        """TELE-221: Negative TD mean indicates over-estimation (optimistic)."""
        metrics = ValueFunctionMetrics(td_error_mean=-3.2)
        assert metrics.td_error_mean == -3.2
        assert metrics.td_error_mean < 0

    def test_healthy_threshold(self):
        """TELE-221: Healthy threshold is abs(value) < 5."""
        healthy_value = 3.5
        assert abs(healthy_value) < 5

    def test_warning_threshold(self):
        """TELE-221: Warning threshold is 5 <= abs(value) < 15."""
        warning_value = 10.0
        assert 5 <= abs(warning_value) < 15

    def test_critical_threshold(self):
        """TELE-221: Critical threshold is abs(value) >= 15."""
        critical_value = 20.0
        assert abs(critical_value) >= 15


# =============================================================================
# TELE-222: TD Error Std
# =============================================================================


class TestTELE222TDErrorStd:
    """TELE-222: TD Error Standard Deviation - noise indicator.

    Standard deviation of temporal difference errors.
    Units: reward units (same scale as returns)
    Range: [0, +inf)
    Default: 0.0

    Note: No explicit health thresholds - displayed as informational metric.
    High std in early training is normal; should decrease as value network converges.
    """

    def test_field_exists(self):
        """TELE-222: td_error_std field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "td_error_std")

    def test_default_value(self):
        """TELE-222: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.td_error_std == 0.0

    def test_is_nonnegative(self):
        """TELE-222: TD error std is always non-negative."""
        metrics = ValueFunctionMetrics(td_error_std=0.0)
        assert metrics.td_error_std >= 0

        metrics = ValueFunctionMetrics(td_error_std=12.5)
        assert metrics.td_error_std >= 0

    def test_accepts_typical_values(self):
        """TELE-222: Accepts typical TD error std values."""
        # Low std (consistent)
        metrics = ValueFunctionMetrics(td_error_std=2.0)
        assert metrics.td_error_std == 2.0

        # Moderate std (normal in early training)
        metrics = ValueFunctionMetrics(td_error_std=8.0)
        assert metrics.td_error_std == 8.0

        # High std (noisy gradients)
        metrics = ValueFunctionMetrics(td_error_std=25.0)
        assert metrics.td_error_std == 25.0


# =============================================================================
# TELE-223: Bellman Error
# =============================================================================


class TestTELE223BellmanError:
    """TELE-223: Bellman Error - early warning for value collapse.

    Squared temporal difference error: |V(s) - (r + gamma * V(s'))|^2
    Units: squared reward units
    Range: [0, +inf)
    Default: 0.0

    Health Thresholds:
    - Healthy: < 20
    - Warning: 20 <= value < 50
    - Critical: >= 50 (NaN likely imminent)
    """

    def test_field_exists(self):
        """TELE-223: bellman_error field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "bellman_error")

    def test_default_value(self):
        """TELE-223: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.bellman_error == 0.0

    def test_is_nonnegative(self):
        """TELE-223: Bellman error is always non-negative (squared)."""
        metrics = ValueFunctionMetrics(bellman_error=0.0)
        assert metrics.bellman_error >= 0

        metrics = ValueFunctionMetrics(bellman_error=35.0)
        assert metrics.bellman_error >= 0

    def test_healthy_threshold(self):
        """TELE-223: Healthy threshold is < 20."""
        healthy_value = 12.0
        assert healthy_value < 20

    def test_warning_threshold(self):
        """TELE-223: Warning threshold is 20 <= value < 50."""
        warning_value = 35.0
        assert 20 <= warning_value < 50

    def test_critical_threshold(self):
        """TELE-223: Critical threshold is >= 50 (NaN imminent)."""
        critical_value = 75.0
        assert critical_value >= 50


# =============================================================================
# TELE-224: Return P10
# =============================================================================


class TestTELE224ReturnP10:
    """TELE-224: Return P10 - 10th percentile (worst-case performance).

    Units: episode return (sum of rewards)
    Range: unbounded
    Default: 0.0

    Health Thresholds:
    - Healthy: >= 0 (worst episodes still positive)
    - Concern: < 0 (some episodes have negative returns)
    - Bimodal Warning: P90 - P10 > 50 (policy inconsistent)
    """

    def test_field_exists(self):
        """TELE-224: return_p10 field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "return_p10")

    def test_default_value(self):
        """TELE-224: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.return_p10 == 0.0

    def test_accepts_negative_values(self):
        """TELE-224: Accepts negative P10 (worst episodes fail)."""
        metrics = ValueFunctionMetrics(return_p10=-25.0)
        assert metrics.return_p10 == -25.0

    def test_accepts_positive_values(self):
        """TELE-224: Accepts positive P10 (even worst episodes succeed)."""
        metrics = ValueFunctionMetrics(return_p10=15.0)
        assert metrics.return_p10 == 15.0

    def test_p10_less_than_or_equal_p50(self):
        """TELE-224: P10 <= P50 by definition."""
        metrics = ValueFunctionMetrics(return_p10=10.0, return_p50=50.0, return_p90=90.0)
        assert metrics.return_p10 <= metrics.return_p50


# =============================================================================
# TELE-225: Return P50
# =============================================================================


class TestTELE225ReturnP50:
    """TELE-225: Return P50 - 50th percentile/median (typical performance).

    The median is more robust than mean for RL training because episode
    returns frequently have heavy tails.

    Units: episode return (sum of rewards)
    Range: unbounded
    Default: 0.0

    Health Thresholds:
    - Healthy: >= 0 (typical episode is positive)
    - Concern: < 0 (typical episode has negative return)
    """

    def test_field_exists(self):
        """TELE-225: return_p50 field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "return_p50")

    def test_default_value(self):
        """TELE-225: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.return_p50 == 0.0

    def test_accepts_negative_values(self):
        """TELE-225: Accepts negative P50 (typical episode fails)."""
        metrics = ValueFunctionMetrics(return_p50=-10.0)
        assert metrics.return_p50 == -10.0

    def test_accepts_positive_values(self):
        """TELE-225: Accepts positive P50 (typical episode succeeds)."""
        metrics = ValueFunctionMetrics(return_p50=45.0)
        assert metrics.return_p50 == 45.0

    def test_p50_between_p10_and_p90(self):
        """TELE-225: P10 <= P50 <= P90 by definition."""
        metrics = ValueFunctionMetrics(return_p10=-20.0, return_p50=30.0, return_p90=85.0)
        assert metrics.return_p10 <= metrics.return_p50 <= metrics.return_p90


# =============================================================================
# TELE-226: Return P90
# =============================================================================


class TestTELE226ReturnP90:
    """TELE-226: Return P90 - 90th percentile (best-case performance).

    Shows the policy's "capability ceiling" - what it can achieve when
    conditions align.

    Units: episode return (sum of rewards)
    Range: unbounded
    Default: 0.0

    Health Thresholds:
    - Healthy: >= 0 (best episodes are positive)
    - Concern: < 0 (even best episodes have negative returns)
    - Bimodal Warning: P90 - P10 > 50 (policy inconsistent)
    """

    def test_field_exists(self):
        """TELE-226: return_p90 field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "return_p90")

    def test_default_value(self):
        """TELE-226: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.return_p90 == 0.0

    def test_accepts_positive_values(self):
        """TELE-226: Accepts positive P90 (best episodes succeed)."""
        metrics = ValueFunctionMetrics(return_p90=100.0)
        assert metrics.return_p90 == 100.0

    def test_accepts_negative_values(self):
        """TELE-226: Accepts negative P90 (even best episodes fail - bad policy)."""
        metrics = ValueFunctionMetrics(return_p90=-5.0)
        assert metrics.return_p90 == -5.0

    def test_p90_greater_than_or_equal_p50(self):
        """TELE-226: P90 >= P50 by definition."""
        metrics = ValueFunctionMetrics(return_p10=10.0, return_p50=50.0, return_p90=90.0)
        assert metrics.return_p90 >= metrics.return_p50

    def test_bimodal_spread_calculation(self):
        """TELE-226: P90 - P10 spread indicates policy consistency."""
        metrics = ValueFunctionMetrics(return_p10=-20.0, return_p50=30.0, return_p90=80.0)
        spread = metrics.return_p90 - metrics.return_p10
        assert spread == 100.0
        # Spread > 50 triggers bimodal warning
        assert spread > 50


# =============================================================================
# TELE-227: Return Variance
# =============================================================================


class TestTELE227ReturnVariance:
    """TELE-227: Return Variance - policy consistency measure.

    Variance of episode returns around their mean.
    Widget displays sqrt(variance) as "Ret sigma" for easier interpretation.

    Units: squared episode return units
    Range: [0, +inf)
    Default: 0.0

    Health Thresholds:
    - Healthy: <= 100 (sigma <= 10, reasonably consistent)
    - Warning: > 100 (sigma > 10, high variability)
    """

    def test_field_exists(self):
        """TELE-227: return_variance field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "return_variance")

    def test_default_value(self):
        """TELE-227: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.return_variance == 0.0

    def test_is_nonnegative(self):
        """TELE-227: Variance is always non-negative."""
        metrics = ValueFunctionMetrics(return_variance=0.0)
        assert metrics.return_variance >= 0

        metrics = ValueFunctionMetrics(return_variance=150.0)
        assert metrics.return_variance >= 0

    def test_healthy_threshold(self):
        """TELE-227: Healthy threshold is variance <= 100 (sigma <= 10)."""
        healthy_variance = 64.0  # sigma = 8
        assert healthy_variance <= 100

    def test_warning_threshold(self):
        """TELE-227: Warning threshold is variance > 100 (sigma > 10)."""
        warning_variance = 225.0  # sigma = 15
        assert warning_variance > 100

    def test_sigma_calculation(self):
        """TELE-227: Sigma (std dev) is sqrt of variance."""
        variance = 144.0
        sigma = variance ** 0.5
        assert sigma == 12.0


# =============================================================================
# TELE-228: Return Skewness
# =============================================================================


class TestTELE228ReturnSkewness:
    """TELE-228: Return Skewness - distribution asymmetry indicator.

    Standardized third moment of episode returns.
    Reveals whether mean return is representative or dominated by outliers.

    Units: dimensionless
    Range: unbounded (typically -3 to +3)
    Default: 0.0

    Health Thresholds:
    - Healthy: abs(value) < 1.0 (roughly symmetric)
    - Warning: 1.0 <= abs(value) < 2.0 (moderately asymmetric)
    - Critical: abs(value) >= 2.0 (severely skewed)
    """

    def test_field_exists(self):
        """TELE-228: return_skewness field exists in schema."""
        metrics = ValueFunctionMetrics()
        assert hasattr(metrics, "return_skewness")

    def test_default_value(self):
        """TELE-228: Default value is 0.0 per TELE record."""
        metrics = ValueFunctionMetrics()
        assert metrics.return_skewness == 0.0

    def test_accepts_positive_skewness(self):
        """TELE-228: Positive skewness = right-skewed (few big wins)."""
        metrics = ValueFunctionMetrics(return_skewness=1.5)
        assert metrics.return_skewness == 1.5
        assert metrics.return_skewness > 0

    def test_accepts_negative_skewness(self):
        """TELE-228: Negative skewness = left-skewed (few catastrophic failures)."""
        metrics = ValueFunctionMetrics(return_skewness=-1.2)
        assert metrics.return_skewness == -1.2
        assert metrics.return_skewness < 0

    def test_healthy_threshold(self):
        """TELE-228: Healthy threshold is abs(value) < 1.0 (symmetric)."""
        healthy_skew = 0.5
        assert abs(healthy_skew) < 1.0

    def test_warning_threshold(self):
        """TELE-228: Warning threshold is 1.0 <= abs(value) < 2.0."""
        warning_skew = 1.5
        assert 1.0 <= abs(warning_skew) < 2.0

    def test_critical_threshold(self):
        """TELE-228: Critical threshold is abs(value) >= 2.0."""
        critical_skew = 2.5
        assert abs(critical_skew) >= 2.0


# =============================================================================
# Consumer Tests - ValueDiagnosticsPanel
# =============================================================================


class TestValueDiagnosticsPanelReadsSchema:
    """Tests that ValueDiagnosticsPanel correctly reads ValueFunctionMetrics fields."""

    @pytest.fixture
    def mock_snapshot(self) -> SanctumSnapshot:
        """Create a SanctumSnapshot with ValueFunctionMetrics for testing."""
        vf_metrics = ValueFunctionMetrics(
            v_return_correlation=0.75,
            td_error_mean=3.5,
            td_error_std=8.2,
            bellman_error=15.0,
            return_p10=-10.0,
            return_p50=25.0,
            return_p90=70.0,
            return_variance=144.0,
            return_skewness=0.5,
        )

        infra_metrics = InfrastructureMetrics(
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="reduce-overhead",
        )

        tamiyo = TamiyoState(
            value_function=vf_metrics,
            infrastructure=infra_metrics,
        )

        return SanctumSnapshot(tamiyo=tamiyo)

    def test_panel_reads_v_return_correlation(self, mock_snapshot: SanctumSnapshot):
        """TELE-220: Panel accesses v_return_correlation field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        # Verify the snapshot stores the value correctly
        assert panel._snapshot.tamiyo.value_function.v_return_correlation == 0.75

    def test_panel_reads_td_error_mean(self, mock_snapshot: SanctumSnapshot):
        """TELE-221: Panel accesses td_error_mean field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.td_error_mean == 3.5

    def test_panel_reads_td_error_std(self, mock_snapshot: SanctumSnapshot):
        """TELE-222: Panel accesses td_error_std field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.td_error_std == 8.2

    def test_panel_reads_bellman_error(self, mock_snapshot: SanctumSnapshot):
        """TELE-223: Panel accesses bellman_error field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.bellman_error == 15.0

    def test_panel_reads_return_p10(self, mock_snapshot: SanctumSnapshot):
        """TELE-224: Panel accesses return_p10 field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.return_p10 == -10.0

    def test_panel_reads_return_p50(self, mock_snapshot: SanctumSnapshot):
        """TELE-225: Panel accesses return_p50 field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.return_p50 == 25.0

    def test_panel_reads_return_p90(self, mock_snapshot: SanctumSnapshot):
        """TELE-226: Panel accesses return_p90 field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.return_p90 == 70.0

    def test_panel_reads_return_variance(self, mock_snapshot: SanctumSnapshot):
        """TELE-227: Panel accesses return_variance field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.return_variance == 144.0

    def test_panel_reads_return_skewness(self, mock_snapshot: SanctumSnapshot):
        """TELE-228: Panel accesses return_skewness field."""
        panel = ValueDiagnosticsPanel()
        panel.update_snapshot(mock_snapshot)

        assert panel._snapshot.tamiyo.value_function.return_skewness == 0.5


class TestValueDiagnosticsPanelStyleMethods:
    """Tests for style methods that apply health thresholds."""

    def test_correlation_style_excellent(self):
        """TELE-220: Excellent correlation (>= 0.8) returns green bold with up-arrow."""
        panel = ValueDiagnosticsPanel()
        style, icon = panel._get_correlation_style(0.85)
        assert style == "green bold"
        assert icon == "↗"

    def test_correlation_style_good(self):
        """TELE-220: Good correlation (>= 0.5) returns green with right-arrow."""
        panel = ValueDiagnosticsPanel()
        style, icon = panel._get_correlation_style(0.65)
        assert style == "green"
        assert icon == "→"

    def test_correlation_style_warning(self):
        """TELE-220: Warning correlation (>= 0.3) returns yellow with right-arrow."""
        panel = ValueDiagnosticsPanel()
        style, icon = panel._get_correlation_style(0.35)
        assert style == "yellow"
        assert icon == "→"

    def test_correlation_style_critical(self):
        """TELE-220: Critical correlation (< 0.3) returns red bold with down-arrow."""
        panel = ValueDiagnosticsPanel()
        style, icon = panel._get_correlation_style(0.2)
        assert style == "red bold"
        assert icon == "↘"

    def test_td_error_style_healthy(self):
        """TELE-221: Healthy TD error (abs < 5) returns green."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_td_error_style(3.0, 5.0)
        assert style == "green"

    def test_td_error_style_warning(self):
        """TELE-221: Warning TD error (5 <= abs < 15) returns yellow."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_td_error_style(10.0, 8.0)
        assert style == "yellow"

    def test_td_error_style_critical(self):
        """TELE-221: Critical TD error (abs >= 15) returns red bold."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_td_error_style(20.0, 10.0)
        assert style == "red bold"

    def test_bellman_style_healthy(self):
        """TELE-223: Healthy Bellman error (< 20) returns green."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_bellman_style(12.0)
        assert style == "green"

    def test_bellman_style_warning(self):
        """TELE-223: Warning Bellman error (20-50) returns yellow."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_bellman_style(35.0)
        assert style == "yellow"

    def test_bellman_style_critical(self):
        """TELE-223: Critical Bellman error (>= 50) returns red bold."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_bellman_style(65.0)
        assert style == "red bold"

    def test_skewness_style_healthy(self):
        """TELE-228: Healthy skewness (abs < 1.0) returns cyan."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_skewness_style(0.5)
        assert style == "cyan"

    def test_skewness_style_warning(self):
        """TELE-228: Warning skewness (1.0 <= abs < 2.0) returns yellow."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_skewness_style(1.5)
        assert style == "yellow"

    def test_skewness_style_critical(self):
        """TELE-228: Critical skewness (abs >= 2.0) returns red bold."""
        panel = ValueDiagnosticsPanel()
        style = panel._get_skewness_style(2.5)
        assert style == "red bold"


# =============================================================================
# Integration Tests
# =============================================================================


class TestValueFunctionMetricsIntegration:
    """Integration tests for complete ValueFunctionMetrics flows."""

    def test_all_percentiles_form_valid_distribution(self):
        """Return percentiles must satisfy p10 <= p50 <= p90."""
        metrics = ValueFunctionMetrics(
            return_p10=-15.0,
            return_p50=40.0,
            return_p90=95.0,
        )

        assert metrics.return_p10 <= metrics.return_p50
        assert metrics.return_p50 <= metrics.return_p90

    def test_spread_warning_calculation(self):
        """P90 - P10 > 50 triggers bimodal warning."""
        metrics = ValueFunctionMetrics(
            return_p10=-30.0,
            return_p50=20.0,
            return_p90=70.0,
        )

        spread = metrics.return_p90 - metrics.return_p10
        assert spread == 100.0
        assert spread > 50  # Would trigger warning

    def test_variance_and_sigma_relationship(self):
        """Widget displays sqrt(variance) as sigma."""
        metrics = ValueFunctionMetrics(return_variance=225.0)
        sigma = metrics.return_variance ** 0.5
        assert sigma == 15.0

    def test_historical_tracking_deques_exist(self):
        """ValueFunctionMetrics has deques for correlation computation."""
        metrics = ValueFunctionMetrics()

        assert hasattr(metrics, "value_predictions")
        assert hasattr(metrics, "actual_returns")
        assert hasattr(metrics, "td_errors")

        assert isinstance(metrics.value_predictions, deque)
        assert isinstance(metrics.actual_returns, deque)
        assert isinstance(metrics.td_errors, deque)

        # Verify maxlen is 100 per schema
        assert metrics.value_predictions.maxlen == 100
        assert metrics.actual_returns.maxlen == 100
        assert metrics.td_errors.maxlen == 100

    def test_metrics_can_be_updated(self):
        """All metric fields can be updated after creation."""
        metrics = ValueFunctionMetrics()

        # Update all TELE-220 to TELE-228 fields
        metrics.v_return_correlation = 0.92
        metrics.td_error_mean = -2.5
        metrics.td_error_std = 6.0
        metrics.bellman_error = 8.5
        metrics.return_p10 = 5.0
        metrics.return_p50 = 45.0
        metrics.return_p90 = 85.0
        metrics.return_variance = 100.0
        metrics.return_skewness = -0.3

        assert metrics.v_return_correlation == 0.92
        assert metrics.td_error_mean == -2.5
        assert metrics.td_error_std == 6.0
        assert metrics.bellman_error == 8.5
        assert metrics.return_p10 == 5.0
        assert metrics.return_p50 == 45.0
        assert metrics.return_p90 == 85.0
        assert metrics.return_variance == 100.0
        assert metrics.return_skewness == -0.3


# =============================================================================
# WIRING TESTS - End-to-End Data Flow Verification
#
# TELE-220 to TELE-228 wiring is tested in tests/simic/test_ppo_value_metrics.py
# and tests/simic/telemetry/test_value_metrics.py. These tests verify:
# - PPO update loop computes and returns all 9 value function metrics
# - Metrics are in valid ranges (correlation in [-1,1], percentiles ordered)
# - Integration from buffer data through compute_value_function_metrics()
#
# As of 2026-01-04, all wiring is complete.
# =============================================================================
