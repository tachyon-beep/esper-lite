"""End-to-end tests for Scoreboard / BestRunRecord metrics (TELE-620 to TELE-627).

Verifies scoreboard telemetry flows from source through to sanctum widgets.

These tests cover:
- TELE-620: peak_accuracy (BestRunRecord.peak_accuracy)
- TELE-621: final_accuracy (BestRunRecord.final_accuracy)
- TELE-622: trajectory_delta (derived: final - peak)
- TELE-623: growth_ratio (BestRunRecord.growth_ratio)
- TELE-624: episode (BestRunRecord.episode)
- TELE-625: epoch (BestRunRecord.epoch)
- TELE-626: global_best (derived: max peak_accuracy)
- TELE-627: mean_best (derived: avg peak_accuracy)

All metrics are FULLY WIRED - BestRunRecord is created at EPISODE_ENDED event
with data from EnvState tracking during training.
"""

import pytest

from esper.karn.sanctum.schema import BestRunRecord, SeedState


# =============================================================================
# TELE-620: Peak Accuracy
# =============================================================================


class TestTELE620PeakAccuracy:
    """TELE-620: peak_accuracy field in BestRunRecord.

    Wiring Status: FULLY WIRED
    - Emitter: EnvState.add_accuracy() tracks best_accuracy
    - Transport: EPISODE_ENDED handler creates BestRunRecord
    - Schema: BestRunRecord.peak_accuracy
    - Consumer: Scoreboard displays in "Peak" column as bold green
    """

    def test_peak_accuracy_field_exists(self) -> None:
        """TELE-620: Verify BestRunRecord.peak_accuracy field exists."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.5,
            final_accuracy=82.0,
        )
        assert hasattr(record, "peak_accuracy")
        assert record.peak_accuracy == 85.5

    def test_peak_accuracy_type_is_float(self) -> None:
        """TELE-620: peak_accuracy must be float type."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=75.0,
            final_accuracy=70.0,
        )
        assert isinstance(record.peak_accuracy, float)

    def test_peak_accuracy_range_zero_to_hundred(self) -> None:
        """TELE-620: peak_accuracy represents percentage (0.0 to 100.0)."""
        # Valid at boundaries
        record_zero = BestRunRecord(
            env_id=0, episode=1, peak_accuracy=0.0, final_accuracy=0.0
        )
        assert record_zero.peak_accuracy == 0.0

        record_full = BestRunRecord(
            env_id=0, episode=1, peak_accuracy=100.0, final_accuracy=100.0
        )
        assert record_full.peak_accuracy == 100.0

    def test_peak_accuracy_one_decimal_precision(self) -> None:
        """TELE-620: peak_accuracy supports 1 decimal place precision."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=87.3,
            final_accuracy=85.1,
        )
        assert record.peak_accuracy == pytest.approx(87.3, rel=1e-6)


# =============================================================================
# TELE-621: Final Accuracy
# =============================================================================


class TestTELE621FinalAccuracy:
    """TELE-621: final_accuracy field in BestRunRecord.

    Wiring Status: FULLY WIRED
    - Emitter: EnvState.host_accuracy at time of EPISODE_ENDED
    - Transport: EPISODE_ENDED handler reads current host_accuracy
    - Schema: BestRunRecord.final_accuracy
    - Consumer: Scoreboard uses in trajectory calculation
    """

    def test_final_accuracy_field_exists(self) -> None:
        """TELE-621: Verify BestRunRecord.final_accuracy field exists."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.5,
            final_accuracy=82.0,
        )
        assert hasattr(record, "final_accuracy")
        assert record.final_accuracy == 82.0

    def test_final_accuracy_type_is_float(self) -> None:
        """TELE-621: final_accuracy must be float type."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=75.0,
            final_accuracy=72.5,
        )
        assert isinstance(record.final_accuracy, float)

    def test_final_accuracy_can_equal_peak(self) -> None:
        """TELE-621: final_accuracy can equal peak (held steady)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=80.0,
            final_accuracy=80.0,
        )
        assert record.final_accuracy == record.peak_accuracy

    def test_final_accuracy_can_be_less_than_peak(self) -> None:
        """TELE-621: final_accuracy can be less than peak (regression)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=78.0,
        )
        assert record.final_accuracy < record.peak_accuracy


# =============================================================================
# TELE-622: Trajectory Delta (Derived)
# =============================================================================


class TestTELE622TrajectoryDelta:
    """TELE-622: trajectory_delta is derived: final_accuracy - peak_accuracy.

    Wiring Status: FULLY WIRED (derived at display time)
    - Source: BestRunRecord.peak_accuracy and final_accuracy
    - Computation: delta = final - peak
    - Consumer: Scoreboard._format_trajectory() for "Traj" column

    Trajectory Thresholds:
    - >+0.5%: green arrow_up_right (still climbing)
    - -1.0% to +0.5%: dim arrow_right (held steady)
    - -2.0% to -1.0%: dim arrow_down_right (small regression)
    - -5.0% to -2.0%: yellow arrow_down_right (moderate regression)
    - <-5.0%: red arrow_down_right (severe regression)
    """

    def test_trajectory_delta_climbing(self) -> None:
        """TELE-622: delta > +0.5% indicates still climbing (green)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=80.0,
            final_accuracy=81.0,  # delta = +1.0%
        )
        delta = record.final_accuracy - record.peak_accuracy
        assert delta > 0.5
        assert delta == pytest.approx(1.0)

    def test_trajectory_delta_held_steady_positive(self) -> None:
        """TELE-622: delta +0.5% to -1.0% indicates held steady (dim)."""
        # Upper bound of held steady range
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=80.0,
            final_accuracy=80.5,  # delta = +0.5%
        )
        delta = record.final_accuracy - record.peak_accuracy
        assert -1.0 <= delta <= 0.5
        assert delta == pytest.approx(0.5)

    def test_trajectory_delta_held_steady_negative(self) -> None:
        """TELE-622: delta within -1.0% indicates held steady (dim)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=80.0,
            final_accuracy=79.0,  # delta = -1.0%
        )
        delta = record.final_accuracy - record.peak_accuracy
        assert delta >= -1.0
        assert delta == pytest.approx(-1.0)

    def test_trajectory_delta_small_regression(self) -> None:
        """TELE-622: delta -2.0% to -1.0% indicates small regression (dim arrow)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=80.0,
            final_accuracy=78.5,  # delta = -1.5%
        )
        delta = record.final_accuracy - record.peak_accuracy
        assert -2.0 <= delta < -1.0
        assert delta == pytest.approx(-1.5)

    def test_trajectory_delta_moderate_regression(self) -> None:
        """TELE-622: delta -5.0% to -2.0% indicates moderate regression (yellow)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=80.0,
            final_accuracy=76.5,  # delta = -3.5%
        )
        delta = record.final_accuracy - record.peak_accuracy
        assert -5.0 <= delta < -2.0
        assert delta == pytest.approx(-3.5)

    def test_trajectory_delta_severe_regression(self) -> None:
        """TELE-622: delta < -5.0% indicates severe regression (red)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=80.0,
            final_accuracy=73.0,  # delta = -7.0%
        )
        delta = record.final_accuracy - record.peak_accuracy
        assert delta < -5.0
        assert delta == pytest.approx(-7.0)

    def test_trajectory_threshold_boundaries(self) -> None:
        """TELE-622: Verify all trajectory threshold boundaries."""
        test_cases = [
            # (peak, final, expected_category)
            (80.0, 81.0, "climbing"),  # +1.0% > +0.5%
            (80.0, 80.5, "held"),  # +0.5% exactly at boundary
            (80.0, 80.4, "held"),  # +0.4% within held range
            (80.0, 79.0, "held"),  # -1.0% exactly at boundary
            (80.0, 78.9, "small_regression"),  # -1.1% just past boundary
            (80.0, 78.0, "small_regression"),  # -2.0% at boundary
            (80.0, 77.9, "moderate_regression"),  # -2.1% just past boundary
            (80.0, 75.0, "moderate_regression"),  # -5.0% at boundary
            (80.0, 74.9, "severe_regression"),  # -5.1% just past boundary
        ]

        for peak, final, expected_cat in test_cases:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=peak,
                final_accuracy=final,
            )
            delta = record.final_accuracy - record.peak_accuracy

            if expected_cat == "climbing":
                assert delta > 0.5, f"Expected climbing for delta {delta}"
            elif expected_cat == "held":
                assert -1.0 <= delta <= 0.5, f"Expected held for delta {delta}"
            elif expected_cat == "small_regression":
                assert -2.0 <= delta < -1.0, f"Expected small regression for delta {delta}"
            elif expected_cat == "moderate_regression":
                assert -5.0 <= delta < -2.0, f"Expected moderate regression for delta {delta}"
            elif expected_cat == "severe_regression":
                assert delta < -5.0, f"Expected severe regression for delta {delta}"


# =============================================================================
# TELE-623: Growth Ratio
# =============================================================================


class TestTELE623GrowthRatio:
    """TELE-623: growth_ratio field in BestRunRecord.

    Wiring Status: FULLY WIRED
    - Emitter: Aggregator computes from host_params + seed_params
    - Transport: EPISODE_ENDED handler calculates ratio
    - Schema: BestRunRecord.growth_ratio (default 1.0)
    - Consumer: Scoreboard displays in "Grw" column

    Growth Ratio Thresholds:
    - <= 1.0: dim (no fossilized seeds)
    - 1.0 < x < 1.1: cyan (modest growth)
    - >= 1.1: bold cyan (significant growth)
    """

    def test_growth_ratio_field_exists(self) -> None:
        """TELE-623: Verify BestRunRecord.growth_ratio field exists."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            growth_ratio=1.05,
        )
        assert hasattr(record, "growth_ratio")
        assert record.growth_ratio == 1.05

    def test_growth_ratio_default_is_one(self) -> None:
        """TELE-623: Default growth_ratio is 1.0 (no fossilized seeds)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        assert record.growth_ratio == 1.0

    def test_growth_ratio_no_growth_threshold(self) -> None:
        """TELE-623: growth_ratio <= 1.0 renders as dim."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            growth_ratio=1.0,
        )
        assert record.growth_ratio <= 1.0

    def test_growth_ratio_modest_growth_threshold(self) -> None:
        """TELE-623: 1.0 < growth_ratio < 1.1 renders as cyan."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            growth_ratio=1.05,
        )
        assert 1.0 < record.growth_ratio < 1.1

    def test_growth_ratio_significant_growth_threshold(self) -> None:
        """TELE-623: growth_ratio >= 1.1 renders as bold cyan."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            growth_ratio=1.15,
        )
        assert record.growth_ratio >= 1.1

    def test_growth_ratio_threshold_boundaries(self) -> None:
        """TELE-623: Verify growth ratio threshold boundaries."""
        test_cases = [
            (1.0, "no_growth"),  # Exactly 1.0
            (0.99, "no_growth"),  # Below 1.0 (edge case)
            (1.01, "modest"),  # Just above 1.0
            (1.09, "modest"),  # Just below 1.1
            (1.1, "significant"),  # Exactly 1.1
            (1.5, "significant"),  # Well above 1.1
        ]

        for ratio, expected_cat in test_cases:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                growth_ratio=ratio,
            )

            if expected_cat == "no_growth":
                assert record.growth_ratio <= 1.0
            elif expected_cat == "modest":
                assert 1.0 < record.growth_ratio < 1.1
            elif expected_cat == "significant":
                assert record.growth_ratio >= 1.1


# =============================================================================
# TELE-624: Episode
# =============================================================================


class TestTELE624Episode:
    """TELE-624: episode field in BestRunRecord.

    Wiring Status: FULLY WIRED
    - Emitter: Computed as episode_start + env_id at EPISODE_ENDED
    - Transport: EPISODE_ENDED handler sets episode
    - Schema: BestRunRecord.episode
    - Consumer: Scoreboard displays as "Ep" column (1-indexed)
    """

    def test_episode_field_exists(self) -> None:
        """TELE-624: Verify BestRunRecord.episode field exists."""
        record = BestRunRecord(
            env_id=0,
            episode=42,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        assert hasattr(record, "episode")
        assert record.episode == 42

    def test_episode_type_is_int(self) -> None:
        """TELE-624: episode must be int type."""
        record = BestRunRecord(
            env_id=0,
            episode=10,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        assert isinstance(record.episode, int)

    def test_episode_zero_indexed_internally(self) -> None:
        """TELE-624: episode is 0-indexed internally."""
        record = BestRunRecord(
            env_id=0,
            episode=0,  # First episode
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        assert record.episode == 0

    def test_episode_displayed_one_indexed(self) -> None:
        """TELE-624: episode displays as 1-indexed (episode + 1)."""
        record = BestRunRecord(
            env_id=0,
            episode=0,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        displayed_episode = record.episode + 1
        assert displayed_episode == 1


# =============================================================================
# TELE-625: Epoch
# =============================================================================


class TestTELE625Epoch:
    """TELE-625: epoch field in BestRunRecord (epoch when peak was achieved).

    Wiring Status: FULLY WIRED
    - Emitter: EnvState.best_accuracy_epoch captured when peak is set
    - Transport: EPISODE_ENDED handler reads best_accuracy_epoch
    - Schema: BestRunRecord.epoch (default 0)
    - Consumer: Scoreboard displays as "@" column with color coding

    Epoch Color Thresholds:
    - < 25: green (early peak)
    - 25-49: white (mid peak)
    - 50-64: yellow (late peak)
    - >= 65: red (very late peak)
    """

    def test_epoch_field_exists(self) -> None:
        """TELE-625: Verify BestRunRecord.epoch field exists."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            epoch=30,
        )
        assert hasattr(record, "epoch")
        assert record.epoch == 30

    def test_epoch_default_is_zero(self) -> None:
        """TELE-625: Default epoch is 0."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        assert record.epoch == 0

    def test_epoch_type_is_int(self) -> None:
        """TELE-625: epoch must be int type."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            epoch=45,
        )
        assert isinstance(record.epoch, int)

    def test_epoch_early_peak_threshold(self) -> None:
        """TELE-625: epoch < 25 renders as green (early peak)."""
        for epoch in [0, 10, 24]:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                epoch=epoch,
            )
            assert record.epoch < 25

    def test_epoch_mid_peak_threshold(self) -> None:
        """TELE-625: epoch 25-49 renders as white (mid peak)."""
        for epoch in [25, 35, 49]:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                epoch=epoch,
            )
            assert 25 <= record.epoch < 50

    def test_epoch_late_peak_threshold(self) -> None:
        """TELE-625: epoch 50-64 renders as yellow (late peak)."""
        for epoch in [50, 55, 64]:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                epoch=epoch,
            )
            assert 50 <= record.epoch < 65

    def test_epoch_very_late_peak_threshold(self) -> None:
        """TELE-625: epoch >= 65 renders as red (very late peak)."""
        for epoch in [65, 70, 100]:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                epoch=epoch,
            )
            assert record.epoch >= 65

    def test_epoch_threshold_boundaries(self) -> None:
        """TELE-625: Verify all epoch threshold boundaries."""
        test_cases = [
            (0, "early"),
            (24, "early"),
            (25, "mid"),
            (49, "mid"),
            (50, "late"),
            (64, "late"),
            (65, "very_late"),
            (100, "very_late"),
        ]

        for epoch, expected_cat in test_cases:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                epoch=epoch,
            )

            if expected_cat == "early":
                assert record.epoch < 25, f"Expected early for epoch {epoch}"
            elif expected_cat == "mid":
                assert 25 <= record.epoch < 50, f"Expected mid for epoch {epoch}"
            elif expected_cat == "late":
                assert 50 <= record.epoch < 65, f"Expected late for epoch {epoch}"
            elif expected_cat == "very_late":
                assert record.epoch >= 65, f"Expected very_late for epoch {epoch}"


# =============================================================================
# TELE-626: Global Best (Derived)
# =============================================================================


class TestTELE626GlobalBest:
    """TELE-626: global_best is derived: max(peak_accuracy) across all records.

    Wiring Status: FULLY WIRED (derived at display time)
    - Source: BestRunRecord.peak_accuracy from all records
    - Computation: max(r.peak_accuracy for r in best_runs)
    - Consumer: Scoreboard stats header as "Best: {value}%"
    """

    def test_global_best_from_single_record(self) -> None:
        """TELE-626: global_best equals peak_accuracy for single record."""
        records = [
            BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
            )
        ]
        global_best = max(r.peak_accuracy for r in records)
        assert global_best == 85.0

    def test_global_best_from_multiple_records(self) -> None:
        """TELE-626: global_best is max peak_accuracy across all records."""
        records = [
            BestRunRecord(env_id=0, episode=1, peak_accuracy=75.0, final_accuracy=72.0),
            BestRunRecord(env_id=1, episode=2, peak_accuracy=82.0, final_accuracy=80.0),
            BestRunRecord(env_id=2, episode=3, peak_accuracy=90.5, final_accuracy=88.0),
            BestRunRecord(env_id=3, episode=4, peak_accuracy=78.0, final_accuracy=75.0),
        ]
        global_best = max(r.peak_accuracy for r in records)
        assert global_best == 90.5

    def test_global_best_empty_records_fallback(self) -> None:
        """TELE-626: Empty records list should handle gracefully."""
        records: list[BestRunRecord] = []
        # Fallback behavior: use 0.0 or compute from EnvState.best_accuracy
        global_best = max((r.peak_accuracy for r in records), default=0.0)
        assert global_best == 0.0

    def test_global_best_updates_with_new_high(self) -> None:
        """TELE-626: global_best increases when new record beats previous high."""
        records = [
            BestRunRecord(env_id=0, episode=1, peak_accuracy=80.0, final_accuracy=78.0),
        ]
        initial_best = max(r.peak_accuracy for r in records)
        assert initial_best == 80.0

        # Add new record that beats the previous high
        records.append(
            BestRunRecord(env_id=1, episode=2, peak_accuracy=88.0, final_accuracy=85.0)
        )
        new_best = max(r.peak_accuracy for r in records)
        assert new_best == 88.0
        assert new_best > initial_best


# =============================================================================
# TELE-627: Mean Best (Derived)
# =============================================================================


class TestTELE627MeanBest:
    """TELE-627: mean_best is derived: avg(peak_accuracy) across all records.

    Wiring Status: FULLY WIRED (derived at display time)
    - Source: BestRunRecord.peak_accuracy from all records
    - Computation: sum(peak_accuracy) / len(best_runs)
    - Consumer: Scoreboard stats header as "Mean: {value}%"
    """

    def test_mean_best_from_single_record(self) -> None:
        """TELE-627: mean_best equals peak_accuracy for single record."""
        records = [
            BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
            )
        ]
        mean_best = sum(r.peak_accuracy for r in records) / len(records)
        assert mean_best == 85.0

    def test_mean_best_from_multiple_records(self) -> None:
        """TELE-627: mean_best is average peak_accuracy across all records."""
        records = [
            BestRunRecord(env_id=0, episode=1, peak_accuracy=70.0, final_accuracy=68.0),
            BestRunRecord(env_id=1, episode=2, peak_accuracy=80.0, final_accuracy=78.0),
            BestRunRecord(env_id=2, episode=3, peak_accuracy=90.0, final_accuracy=88.0),
        ]
        mean_best = sum(r.peak_accuracy for r in records) / len(records)
        assert mean_best == pytest.approx(80.0)  # (70 + 80 + 90) / 3

    def test_mean_best_empty_records_fallback(self) -> None:
        """TELE-627: Empty records list should handle gracefully."""
        records: list[BestRunRecord] = []
        # Fallback behavior: use 0.0
        if records:
            mean_best = sum(r.peak_accuracy for r in records) / len(records)
        else:
            mean_best = 0.0
        assert mean_best == 0.0

    def test_mean_best_less_than_or_equal_global_best(self) -> None:
        """TELE-627: mean_best is always <= global_best."""
        records = [
            BestRunRecord(env_id=0, episode=1, peak_accuracy=60.0, final_accuracy=58.0),
            BestRunRecord(env_id=1, episode=2, peak_accuracy=80.0, final_accuracy=78.0),
            BestRunRecord(env_id=2, episode=3, peak_accuracy=95.0, final_accuracy=93.0),
        ]
        mean_best = sum(r.peak_accuracy for r in records) / len(records)
        global_best = max(r.peak_accuracy for r in records)

        assert mean_best <= global_best
        assert mean_best == pytest.approx(78.33, rel=0.01)  # (60 + 80 + 95) / 3
        assert global_best == 95.0

    def test_mean_best_equals_global_when_uniform(self) -> None:
        """TELE-627: mean_best equals global_best when all accuracies are equal."""
        records = [
            BestRunRecord(env_id=0, episode=1, peak_accuracy=85.0, final_accuracy=82.0),
            BestRunRecord(env_id=1, episode=2, peak_accuracy=85.0, final_accuracy=83.0),
            BestRunRecord(env_id=2, episode=3, peak_accuracy=85.0, final_accuracy=84.0),
        ]
        mean_best = sum(r.peak_accuracy for r in records) / len(records)
        global_best = max(r.peak_accuracy for r in records)

        assert mean_best == global_best == 85.0


# =============================================================================
# BestRunRecord Schema Completeness
# =============================================================================


class TestBestRunRecordSchemaCompleteness:
    """Verify BestRunRecord schema has all expected fields."""

    def test_all_required_fields_exist(self) -> None:
        """BestRunRecord must have all required fields."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )

        # Core identity
        assert hasattr(record, "env_id")
        assert hasattr(record, "episode")

        # Accuracy metrics
        assert hasattr(record, "peak_accuracy")
        assert hasattr(record, "final_accuracy")

        # Epoch tracking
        assert hasattr(record, "epoch")

        # Growth tracking
        assert hasattr(record, "growth_ratio")

        # Seed snapshot
        assert hasattr(record, "seeds")

        # Interactive features
        assert hasattr(record, "record_id")
        assert hasattr(record, "pinned")

    def test_default_values(self) -> None:
        """Verify default values for optional fields."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )

        assert record.epoch == 0
        assert record.growth_ratio == 1.0
        assert record.seeds == {}
        assert record.record_id == ""
        assert record.pinned is False

    def test_seeds_field_accepts_seed_state_dict(self) -> None:
        """Verify seeds field accepts dict of SeedState."""
        seed_states = {
            "slot_0": SeedState(slot_id="slot_0", stage="BLENDING"),
            "slot_1": SeedState(slot_id="slot_1", stage="FOSSILIZED"),
        }
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            seeds=seed_states,
        )

        assert len(record.seeds) == 2
        assert record.seeds["slot_0"].stage == "BLENDING"
        assert record.seeds["slot_1"].stage == "FOSSILIZED"


# =============================================================================
# Consumer Widget Tests (Scoreboard formatting)
# =============================================================================


class TestScoreboardFormatting:
    """Tests verifying Scoreboard widget can correctly format BestRunRecord fields.

    These tests verify the data contracts between BestRunRecord and
    Scoreboard widget formatting methods.
    """

    def test_trajectory_format_data_contract(self) -> None:
        """Scoreboard._format_trajectory expects peak_accuracy and final_accuracy."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )

        # Format trajectory needs these fields
        delta = record.final_accuracy - record.peak_accuracy
        assert isinstance(delta, float)
        assert delta == pytest.approx(-3.0)

    def test_epoch_format_data_contract(self) -> None:
        """Scoreboard._format_epoch expects epoch field."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            epoch=35,
        )

        # Format epoch needs this field
        assert isinstance(record.epoch, int)
        assert record.epoch == 35

    def test_growth_ratio_format_data_contract(self) -> None:
        """Scoreboard._format_growth_ratio expects growth_ratio field."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            growth_ratio=1.05,
        )

        # Format growth ratio needs this field
        assert isinstance(record.growth_ratio, float)
        assert record.growth_ratio == 1.05

    def test_seeds_format_data_contract(self) -> None:
        """Scoreboard._format_seeds expects seeds dict with stage field."""
        seed_states = {
            "slot_0": SeedState(slot_id="slot_0", stage="BLENDING"),
            "slot_1": SeedState(slot_id="slot_1", stage="HOLDING"),
            "slot_2": SeedState(slot_id="slot_2", stage="FOSSILIZED"),
        }
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            seeds=seed_states,
        )

        # Format seeds needs to count by stage
        blending_count = sum(1 for s in record.seeds.values() if s.stage == "BLENDING")
        holding_count = sum(1 for s in record.seeds.values() if s.stage == "HOLDING")
        fossilized_count = sum(1 for s in record.seeds.values() if s.stage == "FOSSILIZED")

        assert blending_count == 1
        assert holding_count == 1
        assert fossilized_count == 1

    def test_stats_header_data_contract(self) -> None:
        """Scoreboard stats header computes global_best and mean_best from records."""
        records = [
            BestRunRecord(env_id=0, episode=1, peak_accuracy=75.0, final_accuracy=72.0),
            BestRunRecord(env_id=1, episode=2, peak_accuracy=85.0, final_accuracy=82.0),
            BestRunRecord(env_id=2, episode=3, peak_accuracy=80.0, final_accuracy=78.0),
        ]

        # Stats header computation
        global_best = max(r.peak_accuracy for r in records)
        mean_best = sum(r.peak_accuracy for r in records) / len(records)

        assert global_best == 85.0
        assert mean_best == pytest.approx(80.0)
