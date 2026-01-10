"""Integration test for reward_components telemetry flow.

Verifies end-to-end: RewardComponentsTelemetry creation -> AnalyticsSnapshotPayload
-> TelemetryEvent serialization -> JSONL file -> MCP DuckDB query.

This tests the entire refactored pipeline to ensure all fields flow through correctly.
"""

import duckdb

from datetime import datetime, timezone
from typing import cast

from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
from esper.leyline import SeedStage
from esper.leyline.telemetry import (
    TelemetryEvent,
    AnalyticsSnapshotPayload,
    TelemetryEventType,
)
from esper.karn.contracts import TelemetryEventLike
from esper.karn.mcp.views import create_views
from esper.karn.serialization import serialize_event


def test_reward_components_flow_end_to_end(tmp_path):
    """Verify reward_components flows from dataclass through to MCP query."""
    # 1. Create reward components (simulating compute_reward output)
    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        compute_rent=-0.1,
        seed_stage=SeedStage.GERMINATED.value,
        action_shaping=0.05,
        terminal_bonus=0.0,
        total_reward=0.45,
        val_acc=0.78,
        num_fossilized_seeds=1,
    )

    # 2. Create payload with typed dataclass
    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        total_reward=0.45,
        action_name="WAIT",
        reward_components=rc,
    )

    # 3. Create event and serialize to JSONL
    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    # Write to JSONL (using actual serialization)
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    events_file.write_text(serialize_event(cast(TelemetryEventLike, event)) + "\n")

    # 4. Query via MCP view
    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    result = conn.execute("""
        SELECT seed_stage, action_shaping, bounded_attribution, val_acc
        FROM rewards
    """).fetchone()

    # 5. Verify all fields came through
    assert result is not None
    assert result[0] == SeedStage.GERMINATED.value  # seed_stage - WAS MISSING BEFORE
    assert result[1] == 0.05  # action_shaping - WAS MISSING BEFORE
    assert result[2] == 0.5  # bounded_attribution
    assert result[3] == 0.78  # val_acc - WAS MISSING BEFORE


def test_reward_components_all_fields_serialized(tmp_path):
    """Verify all RewardComponentsTelemetry fields are accessible via MCP query."""
    # Create reward components with all fields populated
    rc = RewardComponentsTelemetry(
        base_acc_delta=0.02,
        seed_contribution=0.15,
        bounded_attribution=0.5,
        progress_since_germination=0.12,
        attribution_discount=0.95,
        ratio_penalty=-0.03,
        compute_rent=-0.1,
        alpha_shock=-0.01,
        blending_warning=-0.02,
        holding_warning=0.0,
        stage_bonus=0.1,
        pbrs_bonus=0.05,
        synergy_bonus=0.03,
        action_shaping=0.05,
        terminal_bonus=0.2,
        fossilize_terminal_bonus=0.15,
        hindsight_credit=0.08,
        num_fossilized_seeds=2,
        num_contributing_fossilized=1,
        action_name="GERMINATE",
        action_success=True,
        seed_stage=SeedStage.TRAINING.value,
        epoch=15,
        val_acc=0.82,
        acc_at_germination=0.65,
        host_baseline_acc=0.70,
        growth_ratio=0.05,
        total_reward=0.97,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        total_reward=0.97,
        action_name="GERMINATE",
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=15,
    )

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    events_file.write_text(serialize_event(cast(TelemetryEventLike, event)) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    # Query all reward component fields defined in the MCP view
    result = conn.execute("""
        SELECT
            base_acc_delta,
            seed_contribution,
            bounded_attribution,
            progress_since_germination,
            attribution_discount,
            ratio_penalty,
            compute_rent,
            alpha_shock,
            blending_warning,
            holding_warning,
            stage_bonus,
            pbrs_bonus,
            synergy_bonus,
            action_shaping,
            terminal_bonus,
            fossilize_terminal_bonus,
            hindsight_credit,
            num_fossilized_seeds,
            num_contributing_fossilized,
            seed_stage,
            val_acc,
            acc_at_germination,
            host_baseline_acc,
            growth_ratio
        FROM rewards
    """).fetchone()

    assert result is not None

    # Verify all fields match expected values
    assert abs(result[0] - 0.02) < 1e-6  # base_acc_delta
    assert abs(result[1] - 0.15) < 1e-6  # seed_contribution
    assert abs(result[2] - 0.5) < 1e-6   # bounded_attribution
    assert abs(result[3] - 0.12) < 1e-6  # progress_since_germination
    assert abs(result[4] - 0.95) < 1e-6  # attribution_discount
    assert abs(result[5] - (-0.03)) < 1e-6  # ratio_penalty
    assert abs(result[6] - (-0.1)) < 1e-6   # compute_rent
    assert abs(result[7] - (-0.01)) < 1e-6  # alpha_shock
    assert abs(result[8] - (-0.02)) < 1e-6  # blending_warning
    assert abs(result[9] - 0.0) < 1e-6   # holding_warning
    assert abs(result[10] - 0.1) < 1e-6  # stage_bonus
    assert abs(result[11] - 0.05) < 1e-6 # pbrs_bonus
    assert abs(result[12] - 0.03) < 1e-6 # synergy_bonus
    assert abs(result[13] - 0.05) < 1e-6 # action_shaping
    assert abs(result[14] - 0.2) < 1e-6  # terminal_bonus
    assert abs(result[15] - 0.15) < 1e-6 # fossilize_terminal_bonus
    assert abs(result[16] - 0.08) < 1e-6 # hindsight_credit
    assert result[17] == 2  # num_fossilized_seeds
    assert result[18] == 1  # num_contributing_fossilized
    assert result[19] == 3  # seed_stage
    assert abs(result[20] - 0.82) < 1e-6 # val_acc
    assert abs(result[21] - 0.65) < 1e-6 # acc_at_germination
    assert abs(result[22] - 0.70) < 1e-6 # host_baseline_acc
    assert abs(result[23] - 0.05) < 1e-6 # growth_ratio


def test_reward_components_shaped_reward_ratio_computed(tmp_path):
    """Verify shaped_reward_ratio is computed property that flows through serialization."""
    # Create reward components where shaping dominates
    rc = RewardComponentsTelemetry(
        # Low primary signal
        bounded_attribution=0.1,
        # High shaping terms
        stage_bonus=0.3,
        action_shaping=0.2,
        terminal_bonus=0.1,
        total_reward=0.7,  # Most is shaping
    )

    # Verify the property computes correctly before serialization
    expected_ratio = rc.shaped_reward_ratio
    assert expected_ratio > 0.5  # Shaping-dominated

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        total_reward=0.7,
        action_name="WAIT",
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=5,
    )

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    serialized = serialize_event(cast(TelemetryEventLike, event))
    events_file.write_text(serialized + "\n")

    # Verify shaped_reward_ratio is in serialized JSON
    import json
    parsed = json.loads(serialized)
    assert "shaped_reward_ratio" in parsed["data"]["reward_components"]
    assert abs(parsed["data"]["reward_components"]["shaped_reward_ratio"] - expected_ratio) < 1e-6


def test_reward_components_null_optional_fields(tmp_path):
    """Verify optional fields that are None serialize correctly as NULL."""
    # Create minimal reward components - optional fields remain None
    rc = RewardComponentsTelemetry(
        total_reward=0.5,
        # seed_contribution, bounded_attribution, etc. default to None
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        total_reward=0.5,
        action_name="WAIT",
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=1,
    )

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    events_file.write_text(serialize_event(cast(TelemetryEventLike, event)) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    result = conn.execute("""
        SELECT seed_contribution, bounded_attribution, seed_stage
        FROM rewards
    """).fetchone()

    assert result is not None
    # These should be NULL (None in Python) since they weren't set
    assert result[0] is None  # seed_contribution
    assert result[1] is None  # bounded_attribution
    assert result[2] is None  # seed_stage (defaults to None in dataclass)
