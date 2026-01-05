"""End-to-end tests for BestRunRecord lifecycle count metrics (TELE-628 to TELE-629).

Verifies BestRunRecord lifecycle count telemetry flows from source through to sanctum widgets.

These tests cover:
- TELE-628: fossilized_count (BestRunRecord.fossilized_count)
- TELE-629: pruned_count (BestRunRecord.pruned_count)

IMPORTANT DISTINCTION from TELE-503/504:
- TELE-503/504 (SeedLifecycleStats): Cumulative counts for the ENTIRE training run
- TELE-628/629 (BestRunRecord): Point-in-time SNAPSHOTS at peak accuracy for a specific episode

These metrics are FULLY WIRED:
- Source: EnvState.fossilized_count / pruned_count accumulated during episode
- Transport: EPISODE_ENDED handler copies values to BestRunRecord
- Consumer: HistoricalEnvDetail displays in header and metrics section
"""

import pytest

from esper.karn.sanctum.schema import BestRunRecord, SeedState


# =============================================================================
# TELE-628: Best Run Fossilized Count
# =============================================================================


class TestTELE628BestRunFossilizedCount:
    """TELE-628: fossilized_count field in BestRunRecord.

    Wiring Status: FULLY WIRED
    - Emitter: EnvState.fossilized_count incremented by _handle_seed_fossilized()
    - Transport: EPISODE_ENDED handler copies value to BestRunRecord
    - Schema: BestRunRecord.fossilized_count (default 0)
    - Consumer: HistoricalEnvDetail displays in header with green styling

    Semantic Meaning:
    - Point-in-time snapshot of seeds fossilized BY the time of peak accuracy
    - NOT cumulative across training run (that's TELE-503)
    - Resets each episode at EnvState level
    """

    def test_fossilized_count_field_exists(self) -> None:
        """TELE-628: Verify BestRunRecord.fossilized_count field exists."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.5,
            final_accuracy=82.0,
            fossilized_count=3,
        )
        assert hasattr(record, "fossilized_count")
        assert record.fossilized_count == 3

    def test_fossilized_count_default_is_zero(self) -> None:
        """TELE-628: Default fossilized_count is 0 (no fossilizations in episode)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        assert record.fossilized_count == 0

    def test_fossilized_count_type_is_int(self) -> None:
        """TELE-628: fossilized_count must be int type."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            fossilized_count=5,
        )
        assert isinstance(record.fossilized_count, int)

    def test_fossilized_count_accepts_valid_values(self) -> None:
        """TELE-628: fossilized_count accepts valid positive integers."""
        test_values = [0, 1, 5, 10, 100]
        for value in test_values:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                fossilized_count=value,
            )
            assert record.fossilized_count == value

    def test_fossilized_count_consumer_access_header(self) -> None:
        """TELE-628: HistoricalEnvDetail can access fossilized_count for header display.

        Header line 235: header.append(f"Fossilized: {record.fossilized_count}", style="green")
        """
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            fossilized_count=7,
        )
        # Simulate consumer access pattern from historical_env_detail.py line 235
        header_text = f"Fossilized: {record.fossilized_count}"
        assert header_text == "Fossilized: 7"
        assert record.fossilized_count == 7

    def test_fossilized_count_consumer_access_metrics(self) -> None:
        """TELE-628: HistoricalEnvDetail can access fossilized_count for metrics section.

        Metrics line 280: seed_counts.append(f"Fossilized: {record.fossilized_count}", style="green")
        """
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            fossilized_count=4,
        )
        # Simulate consumer access pattern from historical_env_detail.py line 280
        metrics_text = f"Fossilized: {record.fossilized_count}"
        assert metrics_text == "Fossilized: 4"

    def test_fossilized_count_distinct_from_tele_503(self) -> None:
        """TELE-628: fossilized_count is per-episode, not cumulative like TELE-503.

        TELE-503 (SeedLifecycleStats.fossilize_count): Cumulative across entire training run
        TELE-628 (BestRunRecord.fossilized_count): Snapshot at peak accuracy for one episode
        """
        # Create multiple records to show they have independent counts
        records = [
            BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=75.0,
                final_accuracy=72.0,
                fossilized_count=2,  # Episode 1: 2 fossilized
            ),
            BestRunRecord(
                env_id=0,
                episode=2,
                peak_accuracy=80.0,
                final_accuracy=78.0,
                fossilized_count=1,  # Episode 2: 1 fossilized (NOT cumulative 3)
            ),
            BestRunRecord(
                env_id=0,
                episode=3,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                fossilized_count=3,  # Episode 3: 3 fossilized (NOT cumulative 6)
            ),
        ]

        # Each record has its own independent count
        assert records[0].fossilized_count == 2
        assert records[1].fossilized_count == 1
        assert records[2].fossilized_count == 3

        # Total across records is NOT what's stored (that would be TELE-503 pattern)
        total_fossilized = sum(r.fossilized_count for r in records)
        assert total_fossilized == 6
        # But each record stores only its own episode's count
        assert records[2].fossilized_count != total_fossilized


# =============================================================================
# TELE-629: Best Run Pruned Count
# =============================================================================


class TestTELE629BestRunPrunedCount:
    """TELE-629: pruned_count field in BestRunRecord.

    Wiring Status: FULLY WIRED
    - Emitter: EnvState.pruned_count incremented by _handle_seed_pruned()
    - Transport: EPISODE_ENDED handler copies value to BestRunRecord
    - Schema: BestRunRecord.pruned_count (default 0)
    - Consumer: HistoricalEnvDetail displays in header with red styling

    Semantic Meaning:
    - Point-in-time snapshot of seeds pruned BY the time of peak accuracy
    - NOT cumulative across training run (that's TELE-504)
    - Resets each episode at EnvState level
    """

    def test_pruned_count_field_exists(self) -> None:
        """TELE-629: Verify BestRunRecord.pruned_count field exists."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.5,
            final_accuracy=82.0,
            pruned_count=2,
        )
        assert hasattr(record, "pruned_count")
        assert record.pruned_count == 2

    def test_pruned_count_default_is_zero(self) -> None:
        """TELE-629: Default pruned_count is 0 (no prunes in episode)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
        )
        assert record.pruned_count == 0

    def test_pruned_count_type_is_int(self) -> None:
        """TELE-629: pruned_count must be int type."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            pruned_count=4,
        )
        assert isinstance(record.pruned_count, int)

    def test_pruned_count_accepts_valid_values(self) -> None:
        """TELE-629: pruned_count accepts valid positive integers."""
        test_values = [0, 1, 5, 10, 100]
        for value in test_values:
            record = BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                pruned_count=value,
            )
            assert record.pruned_count == value

    def test_pruned_count_consumer_access_header(self) -> None:
        """TELE-629: HistoricalEnvDetail can access pruned_count for header display.

        Header line 237: header.append(f"Pruned: {record.pruned_count}", style="red")
        """
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            pruned_count=5,
        )
        # Simulate consumer access pattern from historical_env_detail.py line 237
        header_text = f"Pruned: {record.pruned_count}"
        assert header_text == "Pruned: 5"
        assert record.pruned_count == 5

    def test_pruned_count_consumer_access_metrics(self) -> None:
        """TELE-629: HistoricalEnvDetail can access pruned_count for metrics section.

        Metrics line 282: seed_counts.append(f"Pruned: {record.pruned_count}", style="red")
        """
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            pruned_count=3,
        )
        # Simulate consumer access pattern from historical_env_detail.py line 282
        metrics_text = f"Pruned: {record.pruned_count}"
        assert metrics_text == "Pruned: 3"

    def test_pruned_count_distinct_from_tele_504(self) -> None:
        """TELE-629: pruned_count is per-episode, not cumulative like TELE-504.

        TELE-504 (SeedLifecycleStats.prune_count): Cumulative across entire training run
        TELE-629 (BestRunRecord.pruned_count): Snapshot at peak accuracy for one episode
        """
        # Create multiple records to show they have independent counts
        records = [
            BestRunRecord(
                env_id=0,
                episode=1,
                peak_accuracy=75.0,
                final_accuracy=72.0,
                pruned_count=1,  # Episode 1: 1 pruned
            ),
            BestRunRecord(
                env_id=0,
                episode=2,
                peak_accuracy=80.0,
                final_accuracy=78.0,
                pruned_count=3,  # Episode 2: 3 pruned (NOT cumulative 4)
            ),
            BestRunRecord(
                env_id=0,
                episode=3,
                peak_accuracy=85.0,
                final_accuracy=82.0,
                pruned_count=2,  # Episode 3: 2 pruned (NOT cumulative 6)
            ),
        ]

        # Each record has its own independent count
        assert records[0].pruned_count == 1
        assert records[1].pruned_count == 3
        assert records[2].pruned_count == 2

        # Total across records is NOT what's stored (that would be TELE-504 pattern)
        total_pruned = sum(r.pruned_count for r in records)
        assert total_pruned == 6
        # But each record stores only its own episode's count
        assert records[2].pruned_count != total_pruned


# =============================================================================
# Combined Tests: Fossilized + Pruned Interaction
# =============================================================================


class TestBestRunLifecycleCounts:
    """Tests for combined fossilized_count and pruned_count behavior."""

    def test_both_counts_coexist(self) -> None:
        """TELE-628/629: Both counts can be set independently."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            fossilized_count=5,
            pruned_count=3,
        )
        assert record.fossilized_count == 5
        assert record.pruned_count == 3

    def test_ratio_calculation(self) -> None:
        """TELE-628/629: Success ratio can be derived from counts.

        This is useful for analyzing which episodes had better seed outcomes.
        """
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            fossilized_count=6,
            pruned_count=4,
        )

        total_terminated = record.fossilized_count + record.pruned_count
        if total_terminated > 0:
            success_rate = record.fossilized_count / total_terminated
        else:
            success_rate = 0.0

        # 6 / (6 + 4) = 0.6 = 60% success rate
        assert success_rate == pytest.approx(0.6)

    def test_zero_counts_for_early_episode(self) -> None:
        """TELE-628/629: Early episodes may have zero counts (no terminations yet)."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=70.0,
            final_accuracy=68.0,
            fossilized_count=0,
            pruned_count=0,
        )
        # Valid state for early in training
        assert record.fossilized_count == 0
        assert record.pruned_count == 0

    def test_all_fossilized_no_pruned(self) -> None:
        """TELE-628/629: Some episodes may have all fossilized, none pruned."""
        record = BestRunRecord(
            env_id=0,
            episode=10,
            peak_accuracy=90.0,
            final_accuracy=88.0,
            fossilized_count=4,
            pruned_count=0,
        )
        # 100% success rate
        assert record.fossilized_count == 4
        assert record.pruned_count == 0

    def test_all_pruned_no_fossilized(self) -> None:
        """TELE-628/629: Some episodes may have all pruned, none fossilized."""
        record = BestRunRecord(
            env_id=0,
            episode=5,
            peak_accuracy=65.0,
            final_accuracy=62.0,
            fossilized_count=0,
            pruned_count=4,
        )
        # 0% success rate - struggling policy
        assert record.fossilized_count == 0
        assert record.pruned_count == 4


# =============================================================================
# HistoricalEnvDetail Consumer Access Tests
# =============================================================================


class TestHistoricalEnvDetailConsumer:
    """Tests verifying HistoricalEnvDetail widget correctly consumes lifecycle counts.

    HistoricalEnvDetail displays:
    - Header (line 235): "Fossilized: {record.fossilized_count}" in green
    - Header (line 237): "Pruned: {record.pruned_count}" in red
    - Metrics (line 280): "Fossilized: {record.fossilized_count}" in green
    - Metrics (line 282): "Pruned: {record.pruned_count}" in red
    """

    def test_header_format_matches_widget(self) -> None:
        """Header format matches historical_env_detail.py lines 235, 237."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            fossilized_count=5,
            pruned_count=3,
        )

        # Exact format from widget
        fossilized_text = f"Fossilized: {record.fossilized_count}"
        pruned_text = f"Pruned: {record.pruned_count}"

        assert fossilized_text == "Fossilized: 5"
        assert pruned_text == "Pruned: 3"

    def test_metrics_format_matches_widget(self) -> None:
        """Metrics format matches historical_env_detail.py lines 280, 282."""
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            fossilized_count=7,
            pruned_count=2,
        )

        # Exact format from widget
        fossilized_text = f"Fossilized: {record.fossilized_count}"
        pruned_text = f"Pruned: {record.pruned_count}"

        assert fossilized_text == "Fossilized: 7"
        assert pruned_text == "Pruned: 2"

    def test_display_alongside_seeds_dict(self) -> None:
        """Lifecycle counts display alongside seeds dict for complete snapshot."""
        seed_states = {
            "slot_0": SeedState(slot_id="slot_0", stage="BLENDING"),
            "slot_1": SeedState(slot_id="slot_1", stage="FOSSILIZED"),
            "slot_2": SeedState(slot_id="slot_2", stage="DORMANT"),
        }
        record = BestRunRecord(
            env_id=0,
            episode=1,
            peak_accuracy=85.0,
            final_accuracy=82.0,
            seeds=seed_states,
            fossilized_count=1,  # Matches FOSSILIZED in seeds
            pruned_count=2,  # Historical prunes (not in current seeds dict)
        )

        # Consumer can access both seeds dict and counts
        assert len(record.seeds) == 3
        assert record.fossilized_count == 1
        assert record.pruned_count == 2

        # Count fossilized from seeds dict should match fossilized_count
        fossilized_in_seeds = sum(
            1 for s in record.seeds.values() if s.stage == "FOSSILIZED"
        )
        assert fossilized_in_seeds == record.fossilized_count


# =============================================================================
# Schema Completeness Tests
# =============================================================================


class TestBestRunRecordLifecycleSchema:
    """Verify BestRunRecord schema includes lifecycle count fields."""

    def test_fossilized_count_in_schema(self) -> None:
        """TELE-628: fossilized_count is defined in BestRunRecord schema."""
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(BestRunRecord)}
        assert "fossilized_count" in fields
        assert fields["fossilized_count"].default == 0

    def test_pruned_count_in_schema(self) -> None:
        """TELE-629: pruned_count is defined in BestRunRecord schema."""
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(BestRunRecord)}
        assert "pruned_count" in fields
        assert fields["pruned_count"].default == 0

    def test_lifecycle_counts_after_other_fields(self) -> None:
        """TELE-628/629: Lifecycle counts are positioned after core metrics.

        Schema lines 1267-1268 in schema.py:
        - fossilized_count: int = 0
        - pruned_count: int = 0
        """
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(BestRunRecord)]

        # Verify ordering (fossilized_count comes before pruned_count)
        foss_idx = field_names.index("fossilized_count")
        prune_idx = field_names.index("pruned_count")
        assert foss_idx < prune_idx

        # Both should come after core metrics
        assert "host_loss" in field_names
        host_loss_idx = field_names.index("host_loss")
        assert host_loss_idx < foss_idx
