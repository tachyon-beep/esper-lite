"""Tests for DuckDB view definitions."""
import tempfile
import json
from pathlib import Path

import duckdb

from esper.karn.mcp.views import create_views, VIEW_DEFINITIONS


def test_view_definitions_exist():
    """All expected views are defined."""
    expected = {
        "raw_events",
        "runs",
        "epochs",
        "ppo_updates",
        "batch_epochs",
        "trends",
        "seed_lifecycle",
        "decisions",
        "rewards",
        "batch_stats",
        "anomalies",
        "episode_outcomes",
        "shapley_computed",
    }
    assert set(VIEW_DEFINITIONS.keys()) == expected


def test_create_views_on_empty_dir():
    """Views can be created even with no telemetry files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        conn = duckdb.connect(":memory:")
        create_views(conn, tmpdir)
        # Should not raise - views exist but return empty
        result = conn.execute("SELECT * FROM runs LIMIT 1").fetchall()
        assert result == []


def test_raw_events_includes_filename_and_run_dir(tmp_path):
    """raw_events should include provenance columns for run scoping."""
    run_dir = tmp_path / "telemetry_2025-01-01_000000"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "test-event-id",
        "event_type": "TRAINING_STARTED",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "seed_id": None,
        "slot_id": None,
        "epoch": None,
        "group_id": "default",
        "message": "",
        "data": {"episode_id": "test_run"},
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    filename, extracted_run_dir = conn.execute(
        "SELECT filename, run_dir FROM raw_events"
    ).fetchone()
    assert extracted_run_dir == "telemetry_2025-01-01_000000"
    assert filename.endswith("/telemetry_2025-01-01_000000/events.jsonl") or filename.endswith(
        "\\telemetry_2025-01-01_000000\\events.jsonl"
    )


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
            "group_id": "default",
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
        "group_id": "treatment",
        "data": {
            "kind": "last_action",
            "env_id": 0,
            "inner_epoch": 10,
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
                # Computed diagnostic (from to_dict() serialization)
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
            group_id, env_id, inner_epoch, action_name, action_success, total_reward,
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
            -- Computed diagnostic
            shaped_reward_ratio
        FROM rewards
    """).fetchone()

    assert result is not None

    # Top-level fields
    assert result[0] == "treatment"  # group_id
    assert result[1] == 0  # env_id
    assert result[2] == 10  # inner_epoch
    assert result[3] == "GERMINATE"  # action_name
    assert result[4] is True  # action_success
    assert result[5] == 0.5  # total_reward

    # Base signal
    assert result[6] == 0.02  # base_acc_delta

    # Contribution-primary signal (key fields per task)
    assert result[7] == 0.15  # seed_contribution
    assert result[8] == 0.3  # bounded_attribution
    assert result[9] == 0.08  # progress_since_germination
    assert result[10] == 0.95  # attribution_discount
    assert result[11] == 0.0  # ratio_penalty

    # Penalties
    assert result[12] == -0.1  # compute_rent
    assert result[13] == -0.01  # alpha_shock
    assert result[14] == -0.05  # blending_warning (key field per task)
    assert result[15] == -0.02  # holding_warning

    # Bonuses
    assert result[16] == 0.1  # stage_bonus
    assert result[17] == 0.03  # pbrs_bonus (key field per task)
    assert result[18] == 0.02  # synergy_bonus
    assert result[19] == 0.05  # action_shaping
    assert result[20] == 0.0  # terminal_bonus
    assert result[21] == 0.0  # fossilize_terminal_bonus
    assert result[22] == 0.0  # hindsight_credit
    assert result[23] == 1  # num_fossilized_seeds
    assert result[24] == 1  # num_contributing_fossilized

    # Context fields
    assert result[25] == 2  # seed_stage
    assert result[26] == 0.75  # val_acc
    assert result[27] == 0.60  # acc_at_germination
    assert result[28] == 0.55  # host_baseline_acc
    assert result[29] == 0.12  # growth_ratio

    # Computed diagnostic
    assert result[30] == 0.25  # shaped_reward_ratio


def test_ppo_updates_view_extracts_head_entropy_fields(tmp_path):
    """ppo_updates view should expose head_* entropy and grad-norm fields."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "ppo-1",
        "event_type": "PPO_UPDATE_COMPLETED",
        "timestamp": datetime.now().isoformat(),
        "seed_id": None,
        "slot_id": None,
        "epoch": 42,  # episodes_completed in this event type
        "group_id": "A",
        "message": "",
        "data": {
            "inner_epoch": 7,
            "batch": 3,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 1.5,
            "kl_divergence": 0.01,
            "clip_fraction": 0.15,
            "explained_variance": 0.8,
            "grad_norm": 0.5,
            "lr": 0.0003,
            "entropy_coef": 0.01,
            "entropy_collapsed": False,
            "head_slot_entropy": 0.9,
            "head_blueprint_entropy": 0.8,
            "head_style_entropy": 0.7,
            "head_tempo_entropy": 0.6,
            "head_alpha_target_entropy": 0.5,
            "head_alpha_speed_entropy": 0.4,
            "head_alpha_curve_entropy": 0.3,
            "head_op_entropy": 0.2,
            "head_slot_grad_norm": 1.1,
            "head_blueprint_grad_norm": 1.2,
            "head_style_grad_norm": 1.3,
            "head_tempo_grad_norm": 1.4,
            "head_alpha_target_grad_norm": 1.5,
            "head_alpha_speed_grad_norm": 1.6,
            "head_alpha_curve_grad_norm": 1.7,
            "head_op_grad_norm": 1.8,
        },
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            episodes_completed,
            group_id,
            inner_epoch,
            batch,
            policy_loss,
            head_slot_entropy,
            head_op_grad_norm
        FROM ppo_updates
        """
    ).fetchone()

    assert row == (42, "A", 7, 3, 0.1, 0.9, 1.8)


def test_seed_lifecycle_view_parses_seed_payload_fields(tmp_path):
    """seed_lifecycle view should align with typed seed lifecycle payload schemas."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    now = datetime.now().isoformat()
    events = [
        {
            "event_id": "seed-1",
            "event_type": "SEED_GERMINATED",
            "timestamp": now,
            "seed_id": "seed-abc",
            "slot_id": "r0c0",
            "epoch": None,
            "group_id": "default",
            "message": "",
            "data": {
                "slot_id": "r0c0",
                "env_id": 0,
                "blueprint_id": "conv_l",
                "params": 1024,
                "alpha": 0.0,
                "grad_ratio": 0.75,
                "has_vanishing": False,
                "has_exploding": False,
                "epochs_in_stage": 0,
                "blend_tempo_epochs": 5,
                "alpha_curve": "LINEAR",
            },
            "severity": "info",
        },
        {
            "event_id": "seed-2",
            "event_type": "SEED_STAGE_CHANGED",
            "timestamp": now,
            "seed_id": "seed-abc",
            "slot_id": "r0c0",
            "epoch": None,
            "group_id": "default",
            "message": "",
            "data": {
                "slot_id": "r0c0",
                "env_id": 0,
                "from_stage": "TRAINING",
                "to_stage": "BLENDING",
                "alpha": 0.5,
                "accuracy_delta": 1.2,
                "epochs_in_stage": 1,
                "grad_ratio": 0.7,
                "has_vanishing": False,
                "has_exploding": False,
                "alpha_curve": "LINEAR",
            },
            "severity": "info",
        },
        {
            "event_id": "seed-3",
            "event_type": "SEED_PRUNED",
            "timestamp": now,
            "seed_id": "seed-abc",
            "slot_id": "r0c0",
            "epoch": None,
            "group_id": "default",
            "message": "",
            "data": {
                "slot_id": "r0c0",
                "env_id": 0,
                "reason": "policy_prune",
                "blueprint_id": "conv_l",
                "improvement": 0.0,
                "auto_pruned": False,
                "epochs_total": 1,
                "counterfactual": None,
                "blending_delta": None,
                "initiator": "policy",
            },
            "severity": "info",
        },
    ]
    events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    rows = conn.execute(
        """
        SELECT event_type, env_id, blueprint_id, params, grad_ratio, from_stage, to_stage, reason, initiator
        FROM seed_lifecycle
        ORDER BY event_type
        """
    ).fetchall()

    assert rows == [
        ("SEED_GERMINATED", 0, "conv_l", 1024, 0.75, None, None, None, None),
        ("SEED_PRUNED", 0, "conv_l", None, None, None, None, "policy_prune", "policy"),
        ("SEED_STAGE_CHANGED", 0, None, None, 0.7, "TRAINING", "BLENDING", None, None),
    ]


def test_decisions_view_extracts_masks_and_head_telemetry(tmp_path):
    """decisions view should parse last_action analytics snapshots."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "decision-1",
        "event_type": "ANALYTICS_SNAPSHOT",
        "timestamp": datetime.now().isoformat(),
        "group_id": "A",
        "message": "",
        "data": {
            "kind": "last_action",
            "env_id": 0,
            "inner_epoch": 5,
            "action_name": "WAIT",
            "action_success": True,
            "total_reward": 0.1,
            "value_estimate": 0.2,
            "action_confidence": 0.3,
            "decision_entropy": 1.0,
            "op_masked": False,
            "slot_masked": True,
            "slot_states": [{"slot_id": "r0c0", "stage": "DORMANT"}],
            "alternatives": [{"action_name": "GERMINATE"}],
            "head_telemetry": {"op_confidence": 0.9, "curve_entropy": 0.8},
        },
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            group_id,
            env_id,
            inner_epoch,
            action_name,
            action_success,
            op_masked,
            slot_masked,
            head_op_confidence,
            head_alpha_curve_entropy
        FROM decisions
        """
    ).fetchone()
    assert row == ("A", 0, 5, "WAIT", True, False, True, 0.9, 0.8)


def test_batch_stats_view_parses_payload(tmp_path):
    """batch_stats view should parse batch_stats analytics snapshots."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "batch-1",
        "event_type": "ANALYTICS_SNAPSHOT",
        "timestamp": datetime.now().isoformat(),
        "group_id": "A",
        "message": "",
        "data": {
            "kind": "batch_stats",
            "episodes_completed": 12,
            "batch": 3,
            "inner_epoch": 7,
            "accuracy": 0.55,
            "host_accuracy": 0.50,
            "entropy": 1.2,
            "kl_divergence": 0.01,
            "value_variance": 0.02,
            "seeds_created": 5,
            "seeds_fossilized": 2,
            "skipped_update": False,
        },
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            episodes_completed,
            batch,
            inner_epoch,
            accuracy,
            host_accuracy,
            entropy,
            seeds_created,
            seeds_fossilized,
            skipped_update
        FROM batch_stats
        """
    ).fetchone()
    assert row == (12, 3, 7, 0.55, 0.5, 1.2, 5, 2, False)


def test_batch_epochs_and_trends_views_parse_payloads(tmp_path):
    """batch_epochs and trends views should parse batch/trend payloads."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    events = [
        {
            "event_id": "batch-epoch-1",
            "event_type": "BATCH_EPOCH_COMPLETED",
            "timestamp": datetime.now().isoformat(),
            "group_id": "A",
            "message": "",
            "data": {
                "episodes_completed": 16,
                "batch_idx": 4,
                "avg_accuracy": 0.6,
                "avg_reward": 0.2,
                "total_episodes": 100,
                "n_envs": 4,
                "start_episode": 12,
                "requested_episodes": 4,
                "rolling_accuracy": 0.58,
                "env_accuracies": [0.6, 0.59, 0.61, 0.6],
            },
            "severity": "info",
        },
        {
            "event_id": "trend-1",
            "event_type": "PLATEAU_DETECTED",
            "timestamp": datetime.now().isoformat(),
            "group_id": "A",
            "message": "",
            "data": {
                "batch_idx": 4,
                "episodes_completed": 16,
                "rolling_delta": 0.001,
                "rolling_avg_accuracy": 0.58,
                "prev_rolling_avg_accuracy": 0.579,
            },
            "severity": "warn",
        },
    ]
    events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    batch_row = conn.execute(
        "SELECT batch_idx, episodes_completed, avg_accuracy FROM batch_epochs"
    ).fetchone()
    assert batch_row == (4, 16, 0.6)

    trend_row = conn.execute(
        "SELECT event_type, batch_idx, rolling_delta FROM trends"
    ).fetchone()
    assert trend_row == ("PLATEAU_DETECTED", 4, 0.001)
