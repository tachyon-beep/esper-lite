"""DuckDB view definitions for telemetry data.

NOTE: Views depend on esper.leyline.telemetry.TelemetryEvent schema.
Breaking changes to TelemetryEvent.data fields may require view updates.
"""
from __future__ import annotations

import duckdb
from pathlib import Path
from typing import TYPE_CHECKING

VIEW_DEFINITIONS: dict[str, str] = {
    "raw_events": """
        CREATE OR REPLACE VIEW raw_events AS
        SELECT * FROM read_json_auto(
            '{telemetry_dir}/*/events.jsonl',
            ignore_errors=true,
            maximum_object_size=16777216,
            union_by_name=true
        )
    """,
    "runs": """
        CREATE OR REPLACE VIEW runs AS
        SELECT
            json_extract_string(data, '$.episode_id') as run_id,
            timestamp as started_at,
            json_extract_string(data, '$.task') as task,
            json_extract_string(data, '$.topology') as topology,
            json_extract_string(data, '$.reward_mode') as reward_mode,
            json_extract(data, '$.n_envs')::INTEGER as n_envs,
            json_extract(data, '$.n_episodes')::INTEGER as n_episodes,
            json_extract(data, '$.max_epochs')::INTEGER as max_epochs,
            json_extract(data, '$.lr')::DOUBLE as learning_rate,
            json_extract(data, '$.entropy_coef')::DOUBLE as entropy_coef,
            json_extract(data, '$.clip_ratio')::DOUBLE as clip_ratio,
            json_extract(data, '$.param_budget')::INTEGER as param_budget,
            json_extract_string(data, '$.policy_device') as policy_device,
            json_extract(data, '$.host_params')::INTEGER as host_params
        FROM raw_events
        WHERE event_type = 'TRAINING_STARTED'
    """,
    "epochs": """
        CREATE OR REPLACE VIEW epochs AS
        SELECT
            timestamp,
            epoch as global_epoch,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.val_accuracy')::DOUBLE as val_accuracy,
            json_extract(data, '$.val_loss')::DOUBLE as val_loss,
            json_extract(data, '$.train_accuracy')::DOUBLE as train_accuracy,
            json_extract(data, '$.train_loss')::DOUBLE as train_loss
        FROM raw_events
        WHERE event_type = 'EPOCH_COMPLETED'
    """,
    "ppo_updates": """
        CREATE OR REPLACE VIEW ppo_updates AS
        SELECT
            timestamp,
            epoch as episodes_completed,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.policy_loss')::DOUBLE as policy_loss,
            json_extract(data, '$.value_loss')::DOUBLE as value_loss,
            json_extract(data, '$.entropy')::DOUBLE as entropy,
            json_extract(data, '$.kl_divergence')::DOUBLE as kl_divergence,
            json_extract(data, '$.clip_fraction')::DOUBLE as clip_fraction,
            json_extract(data, '$.explained_variance')::DOUBLE as explained_variance,
            json_extract(data, '$.avg_accuracy')::DOUBLE as avg_accuracy,
            json_extract(data, '$.avg_reward')::DOUBLE as avg_reward,
            json_extract(data, '$.grad_norm')::DOUBLE as grad_norm,
            json_extract(data, '$.entropy_collapsed')::BOOLEAN as entropy_collapsed,
            json_extract(data, '$.slot_entropy')::DOUBLE as slot_entropy,
            json_extract(data, '$.blueprint_entropy')::DOUBLE as blueprint_entropy,
            json_extract(data, '$.blend_entropy')::DOUBLE as blend_entropy,
            json_extract(data, '$.tempo_entropy')::DOUBLE as tempo_entropy,
            json_extract(data, '$.op_entropy')::DOUBLE as op_entropy
        FROM raw_events
        WHERE event_type = 'PPO_UPDATE_COMPLETED'
    """,
    "seed_lifecycle": """
        CREATE OR REPLACE VIEW seed_lifecycle AS
        SELECT
            timestamp,
            event_type,
            seed_id,
            slot_id,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract_string(data, '$.blueprint_id') as blueprint_id,
            json_extract(data, '$.params')::INTEGER as params,
            json_extract(data, '$.alpha')::DOUBLE as alpha,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.global_epoch')::INTEGER as global_epoch,
            json_extract_string(data, '$.from') as from_stage,
            json_extract_string(data, '$.to') as to_stage,
            json_extract(data, '$.improvement')::DOUBLE as improvement,
            json_extract(data, '$.counterfactual')::DOUBLE as counterfactual,
            json_extract(data, '$.epochs_total')::INTEGER as epochs_total,
            json_extract(data, '$.gradient_health')::DOUBLE as gradient_health,
            json_extract_string(data, '$.reason') as cull_reason,
            json_extract(data, '$.auto_culled')::BOOLEAN as auto_culled
        FROM raw_events
        WHERE event_type IN (
            'SEED_GERMINATED',
            'SEED_STAGE_CHANGED',
            'SEED_FOSSILIZED',
            'SEED_CULLED'
        )
    """,
    "rewards": """
        CREATE OR REPLACE VIEW rewards AS
        SELECT
            timestamp,
            epoch,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.episode')::INTEGER as episode,
            json_extract_string(data, '$.ab_group') as ab_group,
            json_extract_string(data, '$.action_name') as action_name,
            json_extract(data, '$.action_success')::BOOLEAN as action_success,
            json_extract_string(data, '$.seed_stage') as seed_stage,
            json_extract(data, '$.total_reward')::DOUBLE as total_reward,
            json_extract(data, '$.base_acc_delta')::DOUBLE as base_acc_delta,
            json_extract(data, '$.bounded_attribution')::DOUBLE as bounded_attribution,
            json_extract(data, '$.ratio_penalty')::DOUBLE as ratio_penalty,
            json_extract(data, '$.compute_rent')::DOUBLE as compute_rent,
            json_extract(data, '$.stage_bonus')::DOUBLE as stage_bonus,
            json_extract(data, '$.action_shaping')::DOUBLE as action_shaping,
            json_extract(data, '$.terminal_bonus')::DOUBLE as terminal_bonus,
            json_extract(data, '$.val_acc')::DOUBLE as val_acc,
            json_extract(data, '$.num_fossilized_seeds')::INTEGER as num_fossilized_seeds
        FROM raw_events
        WHERE event_type = 'REWARD_COMPUTED'
    """,
    "anomalies": """
        CREATE OR REPLACE VIEW anomalies AS
        SELECT
            timestamp,
            event_type,
            message,
            data
        FROM raw_events
        WHERE event_type IN (
            'VALUE_COLLAPSE_DETECTED',
            'RATIO_EXPLOSION_DETECTED',
            'RATIO_COLLAPSE_DETECTED',
            'GRADIENT_PATHOLOGY_DETECTED',
            'NUMERICAL_INSTABILITY_DETECTED',
            'GOVERNOR_PANIC',
            'GOVERNOR_ROLLBACK',
            'PLATEAU_DETECTED'
        )
    """,
}


def create_views(conn: duckdb.DuckDBPyConnection, telemetry_dir: str) -> None:
    """Create all telemetry views on the given connection.

    Args:
        conn: DuckDB connection (in-memory or file-based)
        telemetry_dir: Path to telemetry directory containing run subdirectories
    """
    conn.execute("PRAGMA threads=4")

    # Check if any telemetry files exist
    telemetry_path = Path(telemetry_dir)
    has_files = any(telemetry_path.glob("*/events.jsonl"))

    for view_name, view_sql in VIEW_DEFINITIONS.items():
        sql = view_sql.format(telemetry_dir=telemetry_dir)
        try:
            conn.execute(sql)
        except duckdb.IOException as e:
            # If no files exist, create empty views with proper schema
            if "No files found" in str(e) and view_name == "raw_events":
                # Create a stub view that returns empty results with the expected columns
                conn.execute("""
                    CREATE OR REPLACE VIEW raw_events AS
                    SELECT
                        CAST(NULL AS VARCHAR) as event_id,
                        CAST(NULL AS VARCHAR) as event_type,
                        CAST(NULL AS TIMESTAMP) as timestamp,
                        CAST(NULL AS VARCHAR) as seed_id,
                        CAST(NULL AS VARCHAR) as slot_id,
                        CAST(NULL AS BIGINT) as epoch,
                        CAST(NULL AS VARCHAR) as message,
                        CAST(NULL AS JSON) as data,
                        CAST(NULL AS VARCHAR) as severity
                    WHERE false
                """)
            else:
                raise
