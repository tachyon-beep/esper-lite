"""Tests for DuckDB view definitions."""
import tempfile
import json
from pathlib import Path

import duckdb
import pytest

from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
)
from esper.karn.mcp.views import (
    create_views,
    scan_ingestion_integrity,
    VIEW_DEFINITIONS,
)


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
        "action_distribution",
        "rewards",
        "reward_calibration",
        "batch_stats",
        "anomalies",
        "run_confounders",
        "episode_outcomes",
        "shapley_computed",
        "morphology_causal_log",
        "topology_manifests",
        "phase_occupancy",
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


def test_runs_view_exposes_proof_baseline_lifecycle_policy(tmp_path):
    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "start-1",
        "event_type": "TRAINING_STARTED",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "seed_id": None,
        "slot_id": None,
        "epoch": None,
        "group_id": "proof",
        "message": "",
        "data": {
            "episode_id": "proof",
            "task": "cifar_impaired",
            "reward_mode": "shaped",
            "seed": 12345,
            "proof_baseline_mode": "fixed_schedule",
            "proof_baseline_pair_id": "blueprint-health-proof",
            "proof_baseline_lifecycle_policy": "apply_declared_lifecycle_schedule",
            "proof_baseline_schedule_id": FIXED_SCHEDULE_GERMINATE_R0C0_V1,
            "proof_baseline_schedule_hash": FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
            "proof_baseline_schedule_version": FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
            "proof_baseline_schedule_action_count": (
                FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
            ),
        },
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    (
        mode,
        pair_id,
        lifecycle_policy,
        seed,
        schedule_id,
        schedule_hash,
        schedule_version,
        schedule_action_count,
    ) = conn.execute(
        """
        SELECT
            proof_baseline_mode,
            proof_baseline_pair_id,
            proof_baseline_lifecycle_policy,
            seed,
            proof_baseline_schedule_id,
            proof_baseline_schedule_hash,
            proof_baseline_schedule_version,
            proof_baseline_schedule_action_count
        FROM runs
        """
    ).fetchone()

    assert mode == "fixed_schedule"
    assert pair_id == "blueprint-health-proof"
    assert lifecycle_policy == "apply_declared_lifecycle_schedule"
    assert seed == 12345
    assert schedule_id == FIXED_SCHEDULE_GERMINATE_R0C0_V1
    assert schedule_hash == FIXED_SCHEDULE_GERMINATE_R0C0_HASH
    assert schedule_version == FIXED_SCHEDULE_GERMINATE_R0C0_VERSION
    assert schedule_action_count == FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT


def test_phase_occupancy_view_expands_phase_timings(tmp_path):
    """phase_occupancy expands the per-epoch phases dict into one row per phase."""
    run_dir = tmp_path / "phase_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "phase-1",
        "event_type": "PHASE_PROFILE_COMPLETED",
        "timestamp": "2026-06-16T00:00:00+00:00",
        "seed_id": None,
        "slot_id": None,
        "epoch": 3,
        "group_id": "default",
        "message": None,
        "data": {
            "phases": {
                "train": {
                    "wall_ms": 12.5,
                    "python_cpu_ms": 11.0,
                    "python_cpu_ratio": 0.88,
                },
                "rollout": {
                    "wall_ms": 4.0,
                    "python_cpu_ms": 3.8,
                    "python_cpu_ratio": 0.95,
                },
            },
            "epoch": 3,
            "batch_idx": 2,
        },
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    rows = conn.execute(
        """
        SELECT run_dir, epoch, batch_idx, phase_name, wall_ms,
               python_cpu_ms, python_cpu_ratio
        FROM phase_occupancy
        ORDER BY phase_name
        """
    ).fetchall()

    assert len(rows) == 2  # one row per phase
    by_name = {r[3]: r for r in rows}
    assert set(by_name) == {"train", "rollout"}

    run_dir_val, epoch, batch_idx, _name, wall_ms, cpu_ms, ratio = by_name["train"]
    assert run_dir_val == "phase_run"
    assert epoch == 3
    assert batch_idx == 2
    assert wall_ms == pytest.approx(12.5)
    assert cpu_ms == pytest.approx(11.0)
    assert ratio == pytest.approx(0.88)

    assert by_name["rollout"][4] == pytest.approx(4.0)


def test_morphology_causal_log_view_extracts_joinable_fields(tmp_path):
    """morphology_causal_log exposes stable action identity and proof evidence."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "causal-1",
        "event_type": "MORPHOLOGY_CAUSAL_LOG",
        "timestamp": "2026-06-13T00:00:00+00:00",
        "seed_id": None,
        "slot_id": None,
        "epoch": 7,
        "group_id": "default",
        "message": "Morphology audit evidence",
        "data": {
            "phase": "audit",
            "env_id": 0,
            "slot_id": "r0c0",
            "operation": "GERMINATE",
            "action_id": "morph-b1-e7-env0-r0c0-op1",
            "proposal_id": "morph-b1-e7-env0-r0c0-op1-proposal",
            "verdict_id": "morph-b1-e7-env0-r0c0-op1-verdict",
            "mutation_id": "morph-b1-e7-env0-r0c0-op1-mutation",
            "observation_hash": "obs-abc123",
            "rng_stream": "simic.lifecycle.env0",
            "rng_seed": 12345,
            "topology": "cnn",
            "blueprint_id": "conv_l",
            "governor_approved": True,
            "governor_reason": "approved",
            "governor_blocked_factor": None,
            "watch_window_evidence": 1.25,
            "linked_event_id": "morph-b1-e7-env0-r0c0-op1-mutation",
        },
        "severity": "debug",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            run_dir,
            epoch,
            phase,
            env_id,
            slot_id,
            operation,
            action_id,
            proposal_id,
            verdict_id,
            mutation_id,
            observation_hash,
            rng_stream,
            rng_seed,
            topology,
            blueprint_id,
            governor_approved,
            governor_reason,
            governor_blocked_factor,
            watch_window_evidence,
            linked_event_id
        FROM morphology_causal_log
        """
    ).fetchone()

    assert row == (
        "test_run",
        7,
        "audit",
        0,
        "r0c0",
        "GERMINATE",
        "morph-b1-e7-env0-r0c0-op1",
        "morph-b1-e7-env0-r0c0-op1-proposal",
        "morph-b1-e7-env0-r0c0-op1-verdict",
        "morph-b1-e7-env0-r0c0-op1-mutation",
        "obs-abc123",
        "simic.lifecycle.env0",
        12345,
        "cnn",
        "conv_l",
        True,
        "approved",
        None,
        1.25,
        "morph-b1-e7-env0-r0c0-op1-mutation",
    )


def test_morphology_causal_log_accepts_uint64_rng_seed(tmp_path):
    """Morphology evidence uses uint64 RNG seeds from lifecycle streams."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"
    uint64_seed = 14_576_210_837_687_595_958

    event = {
        "event_id": "causal-uint64",
        "event_type": "MORPHOLOGY_CAUSAL_LOG",
        "timestamp": "2026-06-13T00:00:00+00:00",
        "seed_id": None,
        "slot_id": None,
        "epoch": 1,
        "group_id": "default",
        "message": "Morphology audit evidence",
        "data": {
            "phase": "commit",
            "env_id": 0,
            "slot_id": "r0c0",
            "operation": "GERMINATE",
            "action_id": "morph-b0-e1-env0-r0c0-op1",
            "proposal_id": "morph-b0-e1-env0-r0c0-op1-proposal",
            "verdict_id": "morph-b0-e1-env0-r0c0-op1-verdict",
            "mutation_id": "morph-b0-e1-env0-r0c0-op1-mutation",
            "observation_hash": "obs-abc123",
            "rng_stream": "simic.lifecycle.env0",
            "rng_seed": uint64_seed,
            "topology": "cnn",
            "blueprint_id": "conv_l",
            "governor_approved": True,
            "governor_reason": "approved",
            "governor_blocked_factor": None,
            "watch_window_evidence": None,
            "linked_event_id": "morph-b0-e1-env0-r0c0-op1-mutation",
        },
        "severity": "debug",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    (rng_seed,) = conn.execute(
        "SELECT rng_seed FROM morphology_causal_log"
    ).fetchone()

    assert rng_seed == uint64_seed


def test_morphology_causal_log_derives_epoch_from_action_id(tmp_path):
    """Morphology evidence remains epoch-joinable when event.epoch is absent."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "causal-derived-epoch",
        "event_type": "MORPHOLOGY_CAUSAL_LOG",
        "timestamp": "2026-06-13T00:00:00+00:00",
        "seed_id": None,
        "slot_id": None,
        "epoch": None,
        "group_id": "default",
        "message": "Morphology audit evidence",
        "data": {
            "phase": "commit",
            "env_id": 0,
            "slot_id": "r0c0",
            "operation": "GERMINATE",
            "action_id": "morph-b1-e7-env0-r0c0-op1",
            "proposal_id": "morph-b1-e7-env0-r0c0-op1-proposal",
            "verdict_id": "morph-b1-e7-env0-r0c0-op1-verdict",
            "mutation_id": "morph-b1-e7-env0-r0c0-op1-mutation",
            "observation_hash": "obs-abc123",
            "rng_stream": "simic.lifecycle.env0",
            "rng_seed": 12345,
            "topology": "cnn",
            "blueprint_id": "conv_l",
            "governor_approved": True,
            "governor_reason": "approved",
            "governor_blocked_factor": None,
            "watch_window_evidence": None,
            "linked_event_id": "morph-b1-e7-env0-r0c0-op1-mutation",
        },
        "severity": "debug",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    (epoch,) = conn.execute("SELECT epoch FROM morphology_causal_log").fetchone()

    assert epoch == 7


def test_topology_manifests_view_extracts_static_final_replay_evidence(tmp_path):
    """topology_manifests exposes source/replay join fields for static-final proof."""
    run_dir = tmp_path / "static_final_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "replay-manifest-1",
        "event_type": "TOPOLOGY_MANIFEST_RECORDED",
        "timestamp": "2026-06-15T00:00:00+00:00",
        "seed_id": None,
        "slot_id": None,
        "epoch": 0,
        "group_id": "final",
        "message": "Static-final replay manifest",
        "data": {
            "manifest_role": "static_final_replay",
            "proof_baseline_pair_id": "blueprint-health-proof",
            "topology_manifest_version": 1,
            "topology_manifest_hash": "topo-hash",
            "topology_manifest_json": '{"slots":["r0c0"]}',
            "task": "cifar_impaired",
            "host_topology": "cnn",
            "slot_config_hash": "slots-hash",
            "slot_count": 1,
            "fossilized_seed_count": 1,
            "topology_delta_count": 1,
            "source_run_dir": "source_run",
            "source_group_id": "dynamic",
            "source_episode_idx": 0,
            "source_event_id": "source-manifest-1",
            "source_topology_manifest_hash": "topo-hash",
            "replay_weight_policy": "topology_only",
            "replay_env_id": 3,
            "replay_episode_idx": 9,
            "replayed_topology_manifest_hash": "topo-hash",
            "manifest_match": True,
        },
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            run_dir,
            group_id,
            epoch,
            manifest_role,
            proof_baseline_pair_id,
            topology_manifest_version,
            topology_manifest_hash,
            topology_manifest_json,
            task,
            host_topology,
            slot_config_hash,
            slot_count,
            fossilized_seed_count,
            topology_delta_count,
            source_run_dir,
            source_group_id,
            source_episode_idx,
            source_event_id,
            source_topology_manifest_hash,
            replay_weight_policy,
            replay_env_id,
            replay_episode_idx,
            replayed_topology_manifest_hash,
            manifest_match
        FROM topology_manifests
        """
    ).fetchone()

    assert row == (
        "static_final_run",
        "final",
        0,
        "static_final_replay",
        "blueprint-health-proof",
        1,
        "topo-hash",
        '{"slots":["r0c0"]}',
        "cifar_impaired",
        "cnn",
        "slots-hash",
        1,
        1,
        1,
        "source_run",
        "dynamic",
        0,
        "source-manifest-1",
        "topo-hash",
        "topology_only",
        3,
        9,
        "topo-hash",
        True,
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
    """rewards view should extract all RewardComponentsTelemetry fields it exposes."""
    from datetime import datetime

    # Create test telemetry with all RewardComponentsTelemetry fields exposed in the view.
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
                "stable_val_acc": 0.74,
                "attribution_discount": 0.95,
                "ratio_penalty": 0.0,
                # Escrow attribution (RewardMode.ESCROW)
                "escrow_credit_prev": 0.10,
                "escrow_credit_target": 0.25,
                "escrow_delta": 0.15,
                "escrow_credit_next": 0.25,
                "escrow_forfeit": -0.07,
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
            seed_contribution, bounded_attribution, progress_since_germination, stable_val_acc,
            attribution_discount, ratio_penalty,
            -- Escrow attribution
            escrow_credit_prev, escrow_credit_target, escrow_delta, escrow_credit_next, escrow_forfeit,
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
    assert result[10] == 0.74  # stable_val_acc
    assert result[11] == 0.95  # attribution_discount
    assert result[12] == 0.0  # ratio_penalty

    # Escrow attribution
    assert result[13] == 0.10  # escrow_credit_prev
    assert result[14] == 0.25  # escrow_credit_target
    assert result[15] == 0.15  # escrow_delta
    assert result[16] == 0.25  # escrow_credit_next
    assert result[17] == -0.07  # escrow_forfeit

    # Penalties
    assert result[18] == -0.1  # compute_rent
    assert result[19] == -0.01  # alpha_shock
    assert result[20] == -0.05  # blending_warning (key field per task)
    assert result[21] == -0.02  # holding_warning

    # Bonuses
    assert result[22] == 0.1  # stage_bonus
    assert result[23] == 0.03  # pbrs_bonus (key field per task)
    assert result[24] == 0.02  # synergy_bonus
    assert result[25] == 0.05  # action_shaping
    assert result[26] == 0.0  # terminal_bonus
    assert result[27] == 0.0  # fossilize_terminal_bonus
    assert result[28] == 0.0  # hindsight_credit
    assert result[29] == 1  # num_fossilized_seeds
    assert result[30] == 1  # num_contributing_fossilized

    # Context fields
    assert result[31] == 2  # seed_stage
    assert result[32] == 0.75  # val_acc
    assert result[33] == 0.60  # acc_at_germination
    assert result[34] == 0.55  # host_baseline_acc
    assert result[35] == 0.12  # growth_ratio

    # Computed diagnostic
    assert result[36] == 0.25  # shaped_reward_ratio


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
            "forced_step_ratio": 0.75,
            "usable_actor_timesteps": 5,
            "decision_density": 0.25,
            "advantage_std_floored": True,
            "d5_pre_norm_advantage_std": 0.125,
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
            "head_value_grad_norm": 1.9,
            "head_slot_learnable_fraction": 0.25,
            "head_blueprint_learnable_fraction": 0.0,
            "head_style_learnable_fraction": 0.5,
            "head_tempo_learnable_fraction": 0.0,
            "head_alpha_target_learnable_fraction": 0.5,
            "head_alpha_speed_learnable_fraction": 0.25,
            "head_alpha_curve_learnable_fraction": 0.25,
            "head_op_learnable_fraction": 1.0,
            "head_slot_gradient_state": "finite",
            "head_blueprint_gradient_state": "not_learnable",
            "head_style_gradient_state": "finite",
            "head_tempo_gradient_state": "not_learnable",
            "head_alpha_target_gradient_state": "finite",
            "head_alpha_speed_gradient_state": "finite",
            "head_alpha_curve_gradient_state": "finite",
            "head_op_gradient_state": "finite",
            "head_value_gradient_state": "finite",
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
            head_op_grad_norm,
            head_value_grad_norm,
            head_blueprint_learnable_fraction,
            head_blueprint_gradient_state,
            head_value_gradient_state,
            forced_step_ratio,
            usable_actor_timesteps,
            decision_density,
            advantage_std_floored,
            d5_pre_norm_advantage_std
        FROM ppo_updates
        """
    ).fetchone()

    assert row == (
        42,
        "A",
        7,
        3,
        0.1,
        0.9,
        1.8,
        1.9,
        0.0,
        "not_learnable",
        "finite",
        0.75,
        5,
        0.25,
        True,
        0.125,
    )


def test_ppo_updates_exposes_ev_robustness_columns(tmp_path):
    """ppo_updates view should expose value_nrmse / ev_low_return_variance / ev_return_variance."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "ppo-ev-1",
        "event_type": "PPO_UPDATE_COMPLETED",
        "timestamp": datetime.now().isoformat(),
        "seed_id": None,
        "slot_id": None,
        "epoch": 11,
        "group_id": "A",
        "message": "",
        "data": {
            "explained_variance": -3.5,
            "value_nrmse": 0.42,
            "ev_low_return_variance": True,
            "ev_return_variance": 0.7,
        },
        "severity": "info",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            explained_variance,
            value_nrmse,
            ev_low_return_variance,
            ev_return_variance
        FROM ppo_updates
        """
    ).fetchone()

    assert row == (-3.5, 0.42, True, 0.7)


def test_run_confounders_view_surfaces_numerical_instability(tmp_path):
    """run_confounders should expose proof-blocking anomaly payload facts."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    event = {
        "event_id": "anomaly-1",
        "event_type": "NUMERICAL_INSTABILITY_DETECTED",
        "timestamp": datetime.now().isoformat(),
        "seed_id": None,
        "slot_id": None,
        "epoch": 12,
        "group_id": "treatment",
        "message": "ignored by ledger",
        "data": {
            "anomaly_type": "numerical_instability",
            "env_id": 2,
            "episode": 12,
            "batch": 3,
            "inner_epoch": 7,
            "total_episodes": 100,
            "detail": "nonfinite policy loss",
        },
        "severity": "error",
    }
    events_file.write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            run_dir,
            group_id,
            env_id,
            event_type,
            anomaly_type,
            episode,
            batch,
            inner_epoch,
            total_episodes,
            detail,
            proof_blocking
        FROM run_confounders
        """
    ).fetchone()

    assert row == (
        "test_run",
        "treatment",
        2,
        "NUMERICAL_INSTABILITY_DETECTED",
        "numerical_instability",
        12,
        3,
        7,
        100,
        "nonfinite policy loss",
        True,
    )


def test_run_confounders_view_surfaces_integrity_confounders(tmp_path):
    """run_confounders must surface rollback, reward-hacking, and degradation
    as proof-blocking confounders (KARN-PROOF-002)."""
    from datetime import datetime

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    events_file = run_dir / "events.jsonl"

    events = [
        {
            "event_id": "rollback-1",
            "event_type": "GOVERNOR_ROLLBACK",
            "timestamp": datetime.now().isoformat(),
            "epoch": 5,
            "group_id": "treatment",
            "data": {
                "env_id": 1,
                "device": "cpu",
                "reason": "governor_nan",
                "episode_idx": 5,
            },
            "severity": "error",
        },
        {
            "event_id": "hack-1",
            "event_type": "REWARD_HACKING_SUSPECTED",
            "timestamp": datetime.now().isoformat(),
            "epoch": 6,
            "group_id": "treatment",
            "data": {
                "pattern": "attribution_ratio",
                "slot_id": "r0c0",
                "seed_id": "seed-1",
                "seed_contribution": 0.9,
                "total_improvement": 0.1,
                "ratio": 9.0,
                "threshold": 2.0,
            },
            "severity": "warning",
        },
        {
            "event_id": "degrade-1",
            "event_type": "PERFORMANCE_DEGRADATION",
            "timestamp": datetime.now().isoformat(),
            "epoch": 7,
            "group_id": "treatment",
            "data": {
                "env_id": 2,
                "current_acc": 40.0,
                "rolling_avg_acc": 80.0,
                "drop_percent": 50.0,
                "threshold_percent": 20.0,
                "episode_idx": 7,
            },
            "severity": "warning",
        },
    ]
    events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    rows = conn.execute(
        """
        SELECT event_type, proof_blocking, detail, episode
        FROM run_confounders
        ORDER BY event_type
        """
    ).fetchall()

    by_type = {row[0]: row for row in rows}
    assert set(by_type) == {
        "GOVERNOR_ROLLBACK",
        "REWARD_HACKING_SUSPECTED",
        "PERFORMANCE_DEGRADATION",
    }
    # All three must be proof-blocking.
    for row in rows:
        assert row[1] is True, f"{row[0]} should be proof_blocking"

    # Each surfaces a human-readable detail (never a bare NULL).
    assert "governor_nan" in by_type["GOVERNOR_ROLLBACK"][2]
    assert "attribution_ratio" in by_type["REWARD_HACKING_SUSPECTED"][2]
    assert "performance degradation" in by_type["PERFORMANCE_DEGRADATION"][2]

    # episode is coalesced from $.episode_idx for rollback/degradation.
    assert by_type["GOVERNOR_ROLLBACK"][3] == 5
    assert by_type["PERFORMANCE_DEGRADATION"][3] == 7


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
                "morphology_proposal_id": "morph-b1-e2-env0-r0c0-op1-proposal",
                "morphology_verdict_id": "morph-b1-e2-env0-r0c0-op1-verdict",
                "morphology_mutation_id": "morph-b1-e2-env0-r0c0-op1-mutation",
                "rng_stream": "simic.lifecycle.env0",
                "rng_seed": 42000,
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
                "morphology_proposal_id": "morph-b1-e3-env0-r0c0-op5-proposal",
                "morphology_verdict_id": "morph-b1-e3-env0-r0c0-op5-verdict",
                "morphology_mutation_id": "morph-b1-e3-env0-r0c0-op5-mutation",
                "rng_stream": "simic.lifecycle.env0",
                "rng_seed": 42001,
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
                "morphology_proposal_id": "morph-b1-e4-env0-r0c0-op3-proposal",
                "morphology_verdict_id": "morph-b1-e4-env0-r0c0-op3-verdict",
                "morphology_mutation_id": "morph-b1-e4-env0-r0c0-op3-mutation",
                "rng_stream": "simic.lifecycle.env0",
                "rng_seed": 42002,
            },
            "severity": "info",
        },
    ]
    events_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    rows = conn.execute(
        """
        SELECT
            event_type, env_id, blueprint_id, params, grad_ratio, from_stage, to_stage,
            reason, initiator, morphology_proposal_id, morphology_verdict_id,
            morphology_mutation_id, rng_stream, rng_seed
        FROM seed_lifecycle
        ORDER BY event_type
        """
    ).fetchall()

    assert rows == [
        (
            "SEED_GERMINATED", 0, "conv_l", 1024, 0.75, None, None, None, None,
            "morph-b1-e2-env0-r0c0-op1-proposal", "morph-b1-e2-env0-r0c0-op1-verdict",
            "morph-b1-e2-env0-r0c0-op1-mutation", "simic.lifecycle.env0", 42000,
        ),
        (
            "SEED_PRUNED", 0, "conv_l", None, None, None, None, "policy_prune", "policy",
            "morph-b1-e4-env0-r0c0-op3-proposal", "morph-b1-e4-env0-r0c0-op3-verdict",
            "morph-b1-e4-env0-r0c0-op3-mutation", "simic.lifecycle.env0", 42002,
        ),
        (
            "SEED_STAGE_CHANGED", 0, None, None, 0.7, "TRAINING", "BLENDING", None, None,
            "morph-b1-e3-env0-r0c0-op5-proposal", "morph-b1-e3-env0-r0c0-op5-verdict",
            "morph-b1-e3-env0-r0c0-op5-mutation", "simic.lifecycle.env0", 42001,
        ),
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
            "explained_variance": 0.02,
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


def test_create_views_escapes_telemetry_dir_sql_literal(tmp_path):
    injected_table = "injected_by_path"
    telemetry_dir = tmp_path / f"telemetry'; CREATE TABLE {injected_table} AS SELECT 1; --"
    run_dir = telemetry_dir / "run-a"
    run_dir.mkdir(parents=True)
    events_file = run_dir / "events.jsonl"
    events_file.write_text('{"event_id": "evt-1", "event_type": "TEST"}\n')

    conn = duckdb.connect(":memory:")
    create_views(conn, str(telemetry_dir))

    assert conn.execute("SELECT event_id FROM raw_events").fetchone() == ("evt-1",)
    with pytest.raises(duckdb.CatalogException):
        conn.execute(f"SELECT * FROM {injected_table}")


# ---------------------------------------------------------------------------
# KARN-PROOF-003: scan_ingestion_integrity re-reads events.jsonl independently
# of DuckDB's ignore_errors and surfaces malformed lines so the proof FAILS
# CLOSED on corruption. These pin the scanner the proof packet depends on.
# ---------------------------------------------------------------------------


def test_scan_ingestion_integrity_clean_on_empty_dir(tmp_path):
    """No events.jsonl files at all is structurally clean (no corruption)."""
    result = scan_ingestion_integrity(str(tmp_path))

    assert result.is_clean is True
    assert result.malformed_count == 0
    assert result.malformed_lines == []


def test_scan_ingestion_integrity_clean_on_valid_jsonl(tmp_path):
    """Every line valid JSON -> clean, never a false positive."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        {"event_id": "a", "event_type": "TRAINING_STARTED"},
        {"event_id": "b", "event_type": "EPISODE_OUTCOME"},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )

    result = scan_ingestion_integrity(str(tmp_path))

    assert result.is_clean is True
    assert result.malformed_count == 0


def test_scan_ingestion_integrity_ignores_blank_lines(tmp_path):
    """Whitespace-only lines are not data and must not be flagged as corrupt."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        json.dumps({"event_id": "a", "event_type": "X"}) + "\n\n   \n"
    )

    result = scan_ingestion_integrity(str(tmp_path))

    assert result.is_clean is True


def test_scan_ingestion_integrity_blocks_on_malformed_jsonl(tmp_path):
    """A corrupt JSONL line that DuckDB would silently drop is surfaced.

    This is the independent re-read that makes the proof packet FAIL CLOSED:
    the malformed line must be recorded with its run_dir, file, 1-based line
    number, and a triage snippet -- it must NOT vanish the way DuckDB's
    ignore_errors ingestion would let it.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    good = json.dumps({"event_id": "a", "event_type": "EPISODE_OUTCOME"})
    # Truncated JSON: DuckDB's ignore_errors reader would silently skip this.
    corrupt = '{"event_id": "bad", "event_type": "EPISODE_OUTCOME", "data": {'
    (run_dir / "events.jsonl").write_text(good + "\n" + corrupt + "\n")

    result = scan_ingestion_integrity(str(tmp_path))

    assert result.is_clean is False
    assert result.malformed_count == 1
    bad_line = result.malformed_lines[0]
    assert bad_line.run_dir == "run"
    assert bad_line.line_number == 2
    assert bad_line.file.endswith("events.jsonl")
    assert "bad" in bad_line.snippet


def test_scan_ingestion_integrity_reports_every_malformed_line(tmp_path):
    """Multiple corrupt lines across files are each recorded, none coalesced."""
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    run_b = tmp_path / "run_b"
    run_b.mkdir()
    (run_a / "events.jsonl").write_text(
        json.dumps({"event_id": "ok"}) + "\n" + "{not json\n"
    )
    (run_b / "events.jsonl").write_text("also bad}\n")

    result = scan_ingestion_integrity(str(tmp_path))

    assert result.is_clean is False
    assert result.malformed_count == 2
    run_dirs = {line.run_dir for line in result.malformed_lines}
    assert run_dirs == {"run_a", "run_b"}


def test_run_confounders_view_empty_on_clean_run(tmp_path):
    """A run with only benign events surfaces NO confounders (no false block)."""
    from datetime import datetime

    run_dir = tmp_path / "clean_run"
    run_dir.mkdir()
    events = [
        {
            "event_id": "start-1",
            "event_type": "TRAINING_STARTED",
            "timestamp": datetime.now().isoformat(),
            "group_id": "A",
            "data": {"episode_id": "proof"},
            "severity": "info",
        },
        {
            "event_id": "eo-1",
            "event_type": "EPISODE_OUTCOME",
            "timestamp": datetime.now().isoformat(),
            "group_id": "A",
            "data": {"env_id": 0, "final_accuracy": 80.0},
            "severity": "info",
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    rows = conn.execute("SELECT * FROM run_confounders").fetchall()
    assert rows == []


def test_episode_outcomes_view_extracts_required_proof_fields(tmp_path):
    """episode_outcomes exposes every required proof ROI contract field."""
    from datetime import datetime

    run_dir = tmp_path / "proof_run"
    run_dir.mkdir()
    event = {
        "event_id": "outcome-1",
        "event_type": "EPISODE_OUTCOME",
        "timestamp": datetime.now().isoformat(),
        "epoch": 8,
        "group_id": "A",
        "data": {
            "env_id": 2,
            "episode_idx": 8,
            "final_accuracy": 91.5,
            "param_ratio": 1.2,
            "num_fossilized": 3,
            "num_contributing_fossilized": 2,
            "episode_reward": 4.5,
            "stability_score": 0.8,
            "reward_mode": "simplified",
            "episode_length": 25,
            "outcome_type": "success",
            "germinate_count": 4,
            "prune_count": 1,
            "fossilize_count": 3,
        },
        "severity": "info",
    }
    (run_dir / "events.jsonl").write_text(json.dumps(event) + "\n")

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    row = conn.execute(
        """
        SELECT
            event_id,
            run_dir,
            group_id,
            env_id,
            episode_idx,
            final_accuracy,
            param_ratio,
            num_fossilized,
            num_contributing_fossilized,
            episode_reward,
            stability_score,
            reward_mode,
            episode_length,
            outcome_type,
            germinate_count,
            prune_count,
            fossilize_count
        FROM episode_outcomes
        """
    ).fetchone()

    assert row == (
        "outcome-1",
        "proof_run",
        "A",
        2,
        8,
        91.5,
        1.2,
        3,
        2,
        4.5,
        0.8,
        "simplified",
        25,
        "success",
        4,
        1,
        3,
    )
