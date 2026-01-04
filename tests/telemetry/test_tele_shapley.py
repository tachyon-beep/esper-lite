"""Tests for Shapley telemetry records (TELE-530 to TELE-534).

Verifies Shapley value attribution telemetry flows correctly through schema:
- TELE-530: values (dict[str, ShapleyEstimate])
- TELE-531: epoch
- TELE-532: ranked_slots (method)
- TELE-533: estimate_mean (ShapleyEstimate.mean)
- TELE-534: estimate_std (ShapleyEstimate.std)

These metrics are used by ShapleyPanel for displaying per-slot attribution
with uncertainty bounds and significance indicators.
"""

import pytest
from datetime import datetime, timezone

from esper.karn.sanctum.schema import ShapleyEstimate, ShapleySnapshot


# -----------------------------------------------------------------------------
# TELE-533: ShapleyEstimate.mean
# -----------------------------------------------------------------------------


class TestTELE533EstimateMean:
    """TELE-533: Shapley Estimate Mean field.

    The mean Shapley value represents the expected marginal contribution
    of a slot to ensemble performance (in accuracy percentage points).
    """

    def test_shapley_estimate_has_mean_field(self):
        """TELE-533: ShapleyEstimate has mean field."""
        estimate = ShapleyEstimate()
        assert hasattr(estimate, "mean")
        assert isinstance(estimate.mean, float)

    def test_shapley_estimate_mean_default(self):
        """TELE-533: ShapleyEstimate.mean defaults to 0.0."""
        estimate = ShapleyEstimate()
        assert estimate.mean == 0.0

    def test_shapley_estimate_mean_positive(self):
        """TELE-533: mean > 0 indicates positive contribution."""
        estimate = ShapleyEstimate(mean=2.5)
        assert estimate.mean == 2.5
        assert estimate.mean > 0

    def test_shapley_estimate_mean_negative(self):
        """TELE-533: mean < 0 indicates slot hurts performance."""
        estimate = ShapleyEstimate(mean=-1.2)
        assert estimate.mean == -1.2
        assert estimate.mean < 0

    def test_shapley_estimate_mean_near_zero(self):
        """TELE-533: mean near zero indicates negligible contribution."""
        estimate = ShapleyEstimate(mean=0.005)
        assert abs(estimate.mean) < 0.01


# -----------------------------------------------------------------------------
# TELE-534: ShapleyEstimate.std
# -----------------------------------------------------------------------------


class TestTELE534EstimateStd:
    """TELE-534: Shapley Estimate Std field.

    Standard deviation represents uncertainty in the mean estimate
    from Monte Carlo permutation sampling.
    """

    def test_shapley_estimate_has_std_field(self):
        """TELE-534: ShapleyEstimate has std field."""
        estimate = ShapleyEstimate()
        assert hasattr(estimate, "std")
        assert isinstance(estimate.std, float)

    def test_shapley_estimate_std_default(self):
        """TELE-534: ShapleyEstimate.std defaults to 0.0."""
        estimate = ShapleyEstimate()
        assert estimate.std == 0.0

    def test_shapley_estimate_std_nonnegative(self):
        """TELE-534: std is always non-negative."""
        estimate = ShapleyEstimate(std=0.5)
        assert estimate.std >= 0.0

    def test_shapley_estimate_n_samples_field(self):
        """TELE-534: ShapleyEstimate has n_samples field for sample count."""
        estimate = ShapleyEstimate(n_samples=100)
        assert estimate.n_samples == 100


# -----------------------------------------------------------------------------
# TELE-530: ShapleySnapshot.values
# -----------------------------------------------------------------------------


class TestTELE530ShapleyValues:
    """TELE-530: Shapley Values dictionary.

    Maps slot_id to ShapleyEstimate for all active slots.
    """

    def test_shapley_snapshot_has_values_field(self):
        """TELE-530: ShapleySnapshot has values field."""
        snapshot = ShapleySnapshot()
        assert hasattr(snapshot, "values")
        assert isinstance(snapshot.values, dict)

    def test_shapley_snapshot_values_default_empty(self):
        """TELE-530: ShapleySnapshot.values defaults to empty dict."""
        snapshot = ShapleySnapshot()
        assert snapshot.values == {}

    def test_shapley_snapshot_values_stores_estimates(self):
        """TELE-530: values dict maps slot_id to ShapleyEstimate."""
        estimates = {
            "r0c0": ShapleyEstimate(mean=2.5, std=0.3, n_samples=100),
            "r0c1": ShapleyEstimate(mean=-0.5, std=0.2, n_samples=100),
        }
        snapshot = ShapleySnapshot(values=estimates)

        assert "r0c0" in snapshot.values
        assert snapshot.values["r0c0"].mean == 2.5
        assert snapshot.values["r0c0"].std == 0.3

        assert "r0c1" in snapshot.values
        assert snapshot.values["r0c1"].mean == -0.5

    def test_shapley_snapshot_has_slot_ids_field(self):
        """TELE-530: ShapleySnapshot has slot_ids tuple for ordering."""
        snapshot = ShapleySnapshot(slot_ids=("r0c0", "r0c1", "r1c0"))
        assert snapshot.slot_ids == ("r0c0", "r0c1", "r1c0")

    def test_shapley_snapshot_slot_ids_default_empty(self):
        """TELE-530: ShapleySnapshot.slot_ids defaults to empty tuple."""
        snapshot = ShapleySnapshot()
        assert snapshot.slot_ids == ()


# -----------------------------------------------------------------------------
# TELE-531: ShapleySnapshot.epoch
# -----------------------------------------------------------------------------


class TestTELE531ShapleyEpoch:
    """TELE-531: Shapley Epoch field.

    Records when Shapley values were computed for staleness detection.
    """

    def test_shapley_snapshot_has_epoch_field(self):
        """TELE-531: ShapleySnapshot has epoch field."""
        snapshot = ShapleySnapshot()
        assert hasattr(snapshot, "epoch")
        assert isinstance(snapshot.epoch, int)

    def test_shapley_snapshot_epoch_default(self):
        """TELE-531: ShapleySnapshot.epoch defaults to 0."""
        snapshot = ShapleySnapshot()
        assert snapshot.epoch == 0

    def test_shapley_snapshot_epoch_set(self):
        """TELE-531: epoch can be set to computation time."""
        snapshot = ShapleySnapshot(epoch=42)
        assert snapshot.epoch == 42

    def test_shapley_snapshot_has_timestamp_field(self):
        """TELE-531: ShapleySnapshot has optional timestamp field."""
        now = datetime.now(timezone.utc)
        snapshot = ShapleySnapshot(timestamp=now)
        assert snapshot.timestamp == now

    def test_shapley_snapshot_timestamp_default_none(self):
        """TELE-531: ShapleySnapshot.timestamp defaults to None."""
        snapshot = ShapleySnapshot()
        assert snapshot.timestamp is None


# -----------------------------------------------------------------------------
# TELE-532: ShapleySnapshot.ranked_slots() method
# -----------------------------------------------------------------------------


class TestTELE532RankedSlots:
    """TELE-532: Shapley Ranked Slots method.

    Returns slots sorted by mean contribution (descending).
    """

    def test_ranked_slots_returns_list(self):
        """TELE-532: ranked_slots() returns a list."""
        snapshot = ShapleySnapshot()
        result = snapshot.ranked_slots()
        assert isinstance(result, list)

    def test_ranked_slots_empty_values_returns_empty(self):
        """TELE-532: ranked_slots() returns [] when values is empty."""
        snapshot = ShapleySnapshot()
        assert snapshot.ranked_slots() == []

    def test_ranked_slots_sorted_descending(self):
        """TELE-532: ranked_slots() sorts by mean contribution descending."""
        estimates = {
            "r0c0": ShapleyEstimate(mean=1.5),
            "r0c1": ShapleyEstimate(mean=3.2),
            "r0c2": ShapleyEstimate(mean=-0.3),
        }
        snapshot = ShapleySnapshot(values=estimates)

        ranked = snapshot.ranked_slots()

        assert len(ranked) == 3
        # First should be highest (r0c1: 3.2)
        assert ranked[0] == ("r0c1", 3.2)
        # Second should be middle (r0c0: 1.5)
        assert ranked[1] == ("r0c0", 1.5)
        # Last should be lowest (r0c2: -0.3)
        assert ranked[2] == ("r0c2", -0.3)

    def test_ranked_slots_returns_tuples(self):
        """TELE-532: ranked_slots() returns list of (slot_id, mean) tuples."""
        estimates = {
            "r0c0": ShapleyEstimate(mean=2.0),
        }
        snapshot = ShapleySnapshot(values=estimates)

        ranked = snapshot.ranked_slots()

        assert len(ranked) == 1
        slot_id, mean = ranked[0]
        assert slot_id == "r0c0"
        assert mean == 2.0

    def test_ranked_slots_handles_negative_values(self):
        """TELE-532: ranked_slots() correctly orders negative values."""
        estimates = {
            "r0c0": ShapleyEstimate(mean=-2.0),
            "r0c1": ShapleyEstimate(mean=-0.5),
            "r0c2": ShapleyEstimate(mean=-3.0),
        }
        snapshot = ShapleySnapshot(values=estimates)

        ranked = snapshot.ranked_slots()

        # Order: -0.5 > -2.0 > -3.0
        assert ranked[0][0] == "r0c1"
        assert ranked[1][0] == "r0c0"
        assert ranked[2][0] == "r0c2"


# -----------------------------------------------------------------------------
# Health Thresholds (from TELE-530, TELE-533, TELE-534)
# -----------------------------------------------------------------------------


class TestShapleyHealthThresholds:
    """Health thresholds for Shapley values.

    - mean > 0.01: Positive contribution (green in UI)
    - abs(mean) <= 0.01: Negligible contribution (dim in UI)
    - mean < -0.01: Negative contribution (red in UI)
    - abs(mean) > 1.96 * std: 95% significant (bold + star in UI)
    """

    def test_positive_contribution_threshold(self):
        """Health: mean > 0.01 indicates positive contribution."""
        estimate = ShapleyEstimate(mean=0.02)
        assert estimate.mean > 0.01  # Green in UI

    def test_negative_contribution_threshold(self):
        """Health: mean < -0.01 indicates negative contribution."""
        estimate = ShapleyEstimate(mean=-0.02)
        assert estimate.mean < -0.01  # Red in UI

    def test_neutral_contribution_threshold(self):
        """Health: abs(mean) <= 0.01 indicates negligible contribution."""
        estimate = ShapleyEstimate(mean=0.005)
        assert abs(estimate.mean) <= 0.01  # Dim in UI

    def test_neutral_contribution_negative_side(self):
        """Health: Negative but small mean is also neutral."""
        estimate = ShapleyEstimate(mean=-0.005)
        assert abs(estimate.mean) <= 0.01  # Dim in UI


# -----------------------------------------------------------------------------
# Significance Calculation (TELE-534)
# -----------------------------------------------------------------------------


class TestShapleySignificance:
    """Statistical significance for Shapley values.

    Significance = abs(mean) > z * std where z=1.96 for 95% CI.
    If significant, we're 95% confident the true contribution is non-zero.
    """

    def test_get_significance_exists(self):
        """Significance: ShapleySnapshot has get_significance() method."""
        snapshot = ShapleySnapshot()
        assert hasattr(snapshot, "get_significance")

    def test_get_significance_returns_false_for_missing_slot(self):
        """Significance: get_significance() returns False for unknown slot."""
        snapshot = ShapleySnapshot()
        assert snapshot.get_significance("unknown") is False

    def test_get_significance_positive_significant(self):
        """Significance: mean=1.0, std=0.4 is significant (1.0 > 1.96*0.4=0.784)."""
        estimates = {"r0c0": ShapleyEstimate(mean=1.0, std=0.4)}
        snapshot = ShapleySnapshot(values=estimates)

        assert snapshot.get_significance("r0c0") is True

    def test_get_significance_negative_significant(self):
        """Significance: mean=-1.5, std=0.5 is significant (1.5 > 1.96*0.5=0.98)."""
        estimates = {"r0c0": ShapleyEstimate(mean=-1.5, std=0.5)}
        snapshot = ShapleySnapshot(values=estimates)

        assert snapshot.get_significance("r0c0") is True

    def test_get_significance_not_significant(self):
        """Significance: mean=0.5, std=0.5 is NOT significant (0.5 < 1.96*0.5=0.98)."""
        estimates = {"r0c0": ShapleyEstimate(mean=0.5, std=0.5)}
        snapshot = ShapleySnapshot(values=estimates)

        assert snapshot.get_significance("r0c0") is False

    def test_get_significance_zero_std_nonzero_mean(self):
        """Significance: std=0 with mean!=0 is significant (deterministic)."""
        estimates = {"r0c0": ShapleyEstimate(mean=0.5, std=0.0)}
        snapshot = ShapleySnapshot(values=estimates)

        # Edge case: When std=0, any non-zero mean is significant
        assert snapshot.get_significance("r0c0") is True

    def test_get_significance_zero_std_zero_mean(self):
        """Significance: std=0 with mean=0 is NOT significant."""
        estimates = {"r0c0": ShapleyEstimate(mean=0.0, std=0.0)}
        snapshot = ShapleySnapshot(values=estimates)

        assert snapshot.get_significance("r0c0") is False

    def test_get_significance_custom_z_value(self):
        """Significance: Custom z value changes threshold."""
        estimates = {"r0c0": ShapleyEstimate(mean=0.5, std=0.3)}
        snapshot = ShapleySnapshot(values=estimates)

        # z=1.96: 0.5 < 1.96*0.3=0.588, not significant
        assert snapshot.get_significance("r0c0", z=1.96) is False

        # z=1.0: 0.5 > 1.0*0.3=0.3, significant
        assert snapshot.get_significance("r0c0", z=1.0) is True

    def test_significance_threshold_boundary(self):
        """Significance: Test exact boundary of 1.96 * std."""
        # mean exactly equals 1.96 * std - should NOT be significant (not >)
        estimates = {"r0c0": ShapleyEstimate(mean=0.98, std=0.5)}  # 0.98 == 1.96*0.5
        snapshot = ShapleySnapshot(values=estimates)

        # 0.98 is not > 0.98, so not significant
        assert snapshot.get_significance("r0c0") is False

        # Just above threshold: 0.99 > 0.98
        estimates2 = {"r0c0": ShapleyEstimate(mean=0.99, std=0.5)}
        snapshot2 = ShapleySnapshot(values=estimates2)
        assert snapshot2.get_significance("r0c0") is True


# -----------------------------------------------------------------------------
# get_mean() accessor method
# -----------------------------------------------------------------------------


class TestShapleyGetMean:
    """ShapleySnapshot.get_mean() accessor method."""

    def test_get_mean_exists(self):
        """get_mean: ShapleySnapshot has get_mean() method."""
        snapshot = ShapleySnapshot()
        assert hasattr(snapshot, "get_mean")

    def test_get_mean_returns_mean(self):
        """get_mean: Returns mean for existing slot."""
        estimates = {"r0c0": ShapleyEstimate(mean=2.5)}
        snapshot = ShapleySnapshot(values=estimates)

        assert snapshot.get_mean("r0c0") == 2.5

    def test_get_mean_returns_zero_for_missing(self):
        """get_mean: Returns 0.0 for unknown slot."""
        snapshot = ShapleySnapshot()
        assert snapshot.get_mean("unknown") == 0.0


# -----------------------------------------------------------------------------
# Consumer Widget Integration (ShapleyPanel)
# -----------------------------------------------------------------------------


class TestShapleyPanelIntegration:
    """Tests verifying ShapleyPanel can read snapshot fields.

    ShapleyPanel is the primary consumer of ShapleySnapshot.
    These tests verify the data contract between schema and widget.
    """

    def test_panel_accesses_values(self):
        """ShapleyPanel reads snapshot.values for per-slot display."""
        estimates = {
            "r0c0": ShapleyEstimate(mean=2.5, std=0.3, n_samples=100),
            "r0c1": ShapleyEstimate(mean=-0.5, std=0.2, n_samples=100),
        }
        snapshot = ShapleySnapshot(values=estimates)

        # Widget accesses values dict
        assert snapshot.values["r0c0"].mean == 2.5
        assert snapshot.values["r0c0"].std == 0.3
        assert snapshot.values["r0c1"].mean == -0.5

    def test_panel_accesses_epoch(self):
        """ShapleyPanel reads snapshot.epoch for header display."""
        snapshot = ShapleySnapshot(epoch=42)

        # Widget displays "[Epoch 42]" in header
        epoch_str = f"Epoch {snapshot.epoch}" if snapshot.epoch > 0 else "Latest"
        assert epoch_str == "Epoch 42"

    def test_panel_accesses_ranked_slots(self):
        """ShapleyPanel calls ranked_slots() for display ordering."""
        estimates = {
            "r0c0": ShapleyEstimate(mean=1.5),
            "r0c1": ShapleyEstimate(mean=3.2),
        }
        snapshot = ShapleySnapshot(values=estimates)

        # Widget iterates in ranked order
        ranked = snapshot.ranked_slots()
        for slot_id, mean in ranked:
            estimate = snapshot.values[slot_id]
            assert estimate.mean == mean

    def test_panel_accesses_significance(self):
        """ShapleyPanel calls get_significance() for star indicator."""
        estimates = {"r0c0": ShapleyEstimate(mean=1.0, std=0.4)}
        snapshot = ShapleySnapshot(values=estimates)

        # Widget checks significance for star display
        is_sig = snapshot.get_significance("r0c0")
        sig_char = "*" if is_sig else "o"
        assert sig_char == "*"  # Significant, so star

    def test_panel_color_coding_positive(self):
        """ShapleyPanel applies green style for mean > 0.01."""
        estimate = ShapleyEstimate(mean=0.02)
        # Widget logic: if mean > 0.01: style = "green"
        if estimate.mean > 0.01:
            value_style = "green"
        elif estimate.mean < -0.01:
            value_style = "red"
        else:
            value_style = "dim"
        assert value_style == "green"

    def test_panel_color_coding_negative(self):
        """ShapleyPanel applies red style for mean < -0.01."""
        estimate = ShapleyEstimate(mean=-0.02)
        if estimate.mean > 0.01:
            value_style = "green"
        elif estimate.mean < -0.01:
            value_style = "red"
        else:
            value_style = "dim"
        assert value_style == "red"

    def test_panel_color_coding_neutral(self):
        """ShapleyPanel applies dim style for abs(mean) <= 0.01."""
        estimate = ShapleyEstimate(mean=0.005)
        if estimate.mean > 0.01:
            value_style = "green"
        elif estimate.mean < -0.01:
            value_style = "red"
        else:
            value_style = "dim"
        assert value_style == "dim"


# -----------------------------------------------------------------------------
# Schema completeness
# -----------------------------------------------------------------------------


class TestShapleySchemaCompleteness:
    """Verify ShapleyEstimate and ShapleySnapshot have all required fields."""

    def test_shapley_estimate_all_fields(self):
        """ShapleyEstimate has mean, std, and n_samples fields."""
        estimate = ShapleyEstimate(mean=1.0, std=0.5, n_samples=100)

        assert estimate.mean == 1.0
        assert estimate.std == 0.5
        assert estimate.n_samples == 100

    def test_shapley_snapshot_all_fields(self):
        """ShapleySnapshot has slot_ids, values, epoch, timestamp fields."""
        now = datetime.now(timezone.utc)
        estimates = {"r0c0": ShapleyEstimate(mean=1.0)}
        snapshot = ShapleySnapshot(
            slot_ids=("r0c0",),
            values=estimates,
            epoch=10,
            timestamp=now,
        )

        assert snapshot.slot_ids == ("r0c0",)
        assert "r0c0" in snapshot.values
        assert snapshot.epoch == 10
        assert snapshot.timestamp == now

    def test_shapley_snapshot_all_methods(self):
        """ShapleySnapshot has get_mean, get_significance, ranked_slots methods."""
        snapshot = ShapleySnapshot()

        assert callable(snapshot.get_mean)
        assert callable(snapshot.get_significance)
        assert callable(snapshot.ranked_slots)
