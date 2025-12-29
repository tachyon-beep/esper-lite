"""Tests for DuckDB view definitions."""
import tempfile
import json
from pathlib import Path

import duckdb

from esper.karn.mcp.views import create_views, VIEW_DEFINITIONS


def test_view_definitions_exist():
    """All expected views are defined."""
    expected = {"raw_events", "runs", "epochs", "ppo_updates", "seed_lifecycle", "rewards", "anomalies", "episode_outcomes"}
    assert set(VIEW_DEFINITIONS.keys()) == expected


def test_create_views_on_empty_dir():
    """Views can be created even with no telemetry files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = duckdb.connect(":memory:")
        create_views(conn, tmpdir)
        # Should not raise - views exist but return empty
        result = conn.execute("SELECT * FROM runs LIMIT 1").fetchall()
        assert result == []


def test_epochs_view_parses_jsonl():
    """Epochs view correctly extracts fields from JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure matching real telemetry layout
        run_dir = Path(tmpdir) / "telemetry_2025-01-01_000000"
        run_dir.mkdir()
        events_file = run_dir / "events.jsonl"

        event = {
            "event_id": "test-event-id",
            "event_type": "EPOCH_COMPLETED",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "seed_id": None,
            "slot_id": None,
            "epoch": 5,
            "message": "",
            "data": {
                "env_id": 0,
                "inner_epoch": 10,
                "val_accuracy": 75.5,
                "val_loss": 0.25,
                "train_accuracy": 80.0,
                "train_loss": 0.20
            },
            "severity": "info"
        }
        events_file.write_text(json.dumps(event) + "\n")

        conn = duckdb.connect(":memory:")
        create_views(conn, tmpdir)

        result = conn.execute("SELECT env_id, val_accuracy FROM epochs").fetchone()
        assert result == (0, 75.5)


def test_rewards_view_extracts_all_fields(tmp_path):
    """rewards view should extract all 28 fields from RewardComponentsTelemetry."""
    from datetime import datetime

    # Create test telemetry with all fields from RewardComponentsTelemetry
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "test-1",
        "event_type": "ANALYTICS_SNAPSHOT",
        "timestamp": datetime.now().isoformat(),
        "epoch": 10,
        "data": {
            "kind": "last_action",
            "env_id": 0,
            "episode": 5,
            "ab_group": "treatment",
            "action_name": "GERMINATE",
            "action_success": True,
            "total_reward": 0.5,
            "reward_components": {
                # Base signal
                "base_acc_delta": 0.02,
                # Contribution-primary signal
                "seed_contribution": 0.15,
                "bounded_attribution": 0.3,
                "progress_since_germination": 0.08,
                "attribution_discount": 0.95,
                "ratio_penalty": 0.0,
                # Penalties
                "compute_rent": -0.1,
                "alpha_shock": -0.01,
                "blending_warning": -0.05,
                "holding_warning": -0.02,
                # Bonuses
                "stage_bonus": 0.1,
                "pbrs_bonus": 0.03,
                "synergy_bonus": 0.02,
                "action_shaping": 0.05,
                "terminal_bonus": 0.0,
                "fossilize_terminal_bonus": 0.0,
                "hindsight_credit": 0.0,
                "num_fossilized_seeds": 1,
                "num_contributing_fossilized": 1,
                # Context fields
                "seed_stage": 2,
                "val_acc": 0.75,
                "acc_at_germination": 0.60,
                "host_baseline_acc": 0.55,
                "growth_ratio": 0.12,
                # Computed property
                "shaped_reward_ratio": 0.25,
            }
        }
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    # Fetch all columns to verify comprehensive coverage
    result = conn.execute("""
        SELECT
            -- Top-level fields
            epoch, env_id, episode, ab_group, action_name, action_success, total_reward,
            -- Base signal
            base_acc_delta,
            -- Contribution-primary signal
            seed_contribution, bounded_attribution, progress_since_germination,
            attribution_discount, ratio_penalty,
            -- Penalties
            compute_rent, alpha_shock, blending_warning, holding_warning,
            -- Bonuses
            stage_bonus, pbrs_bonus, synergy_bonus, action_shaping,
            terminal_bonus, fossilize_terminal_bonus, hindsight_credit,
            num_fossilized_seeds, num_contributing_fossilized,
            -- Context fields
            seed_stage, val_acc, acc_at_germination, host_baseline_acc, growth_ratio,
            -- Computed property
            shaped_reward_ratio
        FROM rewards
    """).fetchone()

    assert result is not None

    # Top-level fields
    assert result[0] == 10  # epoch
    assert result[1] == 0  # env_id
    assert result[2] == 5  # episode
    assert result[3] == "treatment"  # ab_group
    assert result[4] == "GERMINATE"  # action_name
    assert result[5] is True  # action_success
    assert result[6] == 0.5  # total_reward

    # Base signal
    assert result[7] == 0.02  # base_acc_delta

    # Contribution-primary signal (key fields per task)
    assert result[8] == 0.15  # seed_contribution
    assert result[9] == 0.3  # bounded_attribution
    assert result[10] == 0.08  # progress_since_germination
    assert result[11] == 0.95  # attribution_discount
    assert result[12] == 0.0  # ratio_penalty

    # Penalties
    assert result[13] == -0.1  # compute_rent
    assert result[14] == -0.01  # alpha_shock
    assert result[15] == -0.05  # blending_warning (key field per task)
    assert result[16] == -0.02  # holding_warning

    # Bonuses
    assert result[17] == 0.1  # stage_bonus
    assert result[18] == 0.03  # pbrs_bonus (key field per task)
    assert result[19] == 0.02  # synergy_bonus
    assert result[20] == 0.05  # action_shaping
    assert result[21] == 0.0  # terminal_bonus
    assert result[22] == 0.0  # fossilize_terminal_bonus
    assert result[23] == 0.0  # hindsight_credit
    assert result[24] == 1  # num_fossilized_seeds
    assert result[25] == 1  # num_contributing_fossilized

    # Context fields
    assert result[26] == 2  # seed_stage
    assert result[27] == 0.75  # val_acc
    assert result[28] == 0.60  # acc_at_germination
    assert result[29] == 0.55  # host_baseline_acc
    assert result[30] == 0.12  # growth_ratio

    # Computed property
    assert result[31] == 0.25  # shaped_reward_ratio
