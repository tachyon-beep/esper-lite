"""DuckDB view definitions for telemetry data.

NOTE: Views depend on esper.leyline.telemetry.TelemetryEvent schema.
Breaking changes to TelemetryEvent.data fields may require view updates.
"""
from __future__ import annotations

import duckdb
from pathlib import Path

VIEW_DEFINITIONS: dict[str, str] = {
    "raw_events": """
        CREATE OR REPLACE VIEW raw_events AS
        SELECT
            event_id,
            event_type,
            timestamp,
            seed_id,
            slot_id,
            epoch,
            group_id,
            message,
            data,
            severity,
            filename,
            regexp_extract(filename, '(?:^|[\\\\/])([^\\\\/]+)[\\\\/]events\\.jsonl$', 1) as run_dir
        FROM read_json_auto(
            '{telemetry_dir}/*/events.jsonl',
            format='newline_delimited',
            auto_detect=false,
            columns={{
                'event_id': 'VARCHAR',
                'event_type': 'VARCHAR',
                'timestamp': 'TIMESTAMP',
                'seed_id': 'VARCHAR',
                'slot_id': 'VARCHAR',
                'epoch': 'BIGINT',
                'group_id': 'VARCHAR',
                'message': 'VARCHAR',
                'data': 'JSON',
                'severity': 'VARCHAR'
            }},
            filename=true,
            ignore_errors=true,
            maximum_object_size=16777216,
            union_by_name=true
        )
    """,
    "runs": """
        CREATE OR REPLACE VIEW runs AS
        SELECT
            event_id,
            run_dir,
            group_id,
            json_extract_string(data, '$.episode_id') as episode_id,
            timestamp as started_at,
            json_extract_string(data, '$.task') as task,
            json_extract_string(data, '$.reward_mode') as reward_mode,
            json_extract(data, '$.n_envs')::INTEGER as n_envs,
            json_extract(data, '$.n_episodes')::INTEGER as n_episodes,
            json_extract(data, '$.max_epochs')::INTEGER as max_epochs,
            json_extract(data, '$.lr')::DOUBLE as lr,
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
            event_id,
            timestamp,
            run_dir,
            group_id,
            epoch,
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
            event_id,
            timestamp,
            run_dir,
            epoch as episodes_completed,
            group_id,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.batch')::INTEGER as batch,
            json_extract(data, '$.policy_loss')::DOUBLE as policy_loss,
            json_extract(data, '$.value_loss')::DOUBLE as value_loss,
            json_extract(data, '$.entropy')::DOUBLE as entropy,
            json_extract(data, '$.kl_divergence')::DOUBLE as kl_divergence,
            json_extract(data, '$.clip_fraction')::DOUBLE as clip_fraction,
            json_extract(data, '$.explained_variance')::DOUBLE as explained_variance,
            json_extract(data, '$.grad_norm')::DOUBLE as grad_norm,
            json_extract(data, '$.lr')::DOUBLE as lr,
            json_extract(data, '$.entropy_coef')::DOUBLE as entropy_coef,
            json_extract(data, '$.entropy_collapsed')::BOOLEAN as entropy_collapsed,
            json_extract(data, '$.head_slot_entropy')::DOUBLE as head_slot_entropy,
            json_extract(data, '$.head_blueprint_entropy')::DOUBLE as head_blueprint_entropy,
            json_extract(data, '$.head_style_entropy')::DOUBLE as head_style_entropy,
            json_extract(data, '$.head_tempo_entropy')::DOUBLE as head_tempo_entropy,
            json_extract(data, '$.head_alpha_target_entropy')::DOUBLE as head_alpha_target_entropy,
            json_extract(data, '$.head_alpha_speed_entropy')::DOUBLE as head_alpha_speed_entropy,
            json_extract(data, '$.head_alpha_curve_entropy')::DOUBLE as head_alpha_curve_entropy,
            json_extract(data, '$.head_op_entropy')::DOUBLE as head_op_entropy,
            json_extract(data, '$.head_slot_grad_norm')::DOUBLE as head_slot_grad_norm,
            json_extract(data, '$.head_blueprint_grad_norm')::DOUBLE as head_blueprint_grad_norm,
            json_extract(data, '$.head_style_grad_norm')::DOUBLE as head_style_grad_norm,
            json_extract(data, '$.head_tempo_grad_norm')::DOUBLE as head_tempo_grad_norm,
            json_extract(data, '$.head_alpha_target_grad_norm')::DOUBLE as head_alpha_target_grad_norm,
            json_extract(data, '$.head_alpha_speed_grad_norm')::DOUBLE as head_alpha_speed_grad_norm,
            json_extract(data, '$.head_alpha_curve_grad_norm')::DOUBLE as head_alpha_curve_grad_norm,
            json_extract(data, '$.head_op_grad_norm')::DOUBLE as head_op_grad_norm
        FROM raw_events
        WHERE event_type = 'PPO_UPDATE_COMPLETED'
    """,
    "batch_epochs": """
        CREATE OR REPLACE VIEW batch_epochs AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            json_extract(data, '$.episodes_completed')::INTEGER as episodes_completed,
            json_extract(data, '$.batch_idx')::INTEGER as batch_idx,
            json_extract(data, '$.avg_accuracy')::DOUBLE as avg_accuracy,
            json_extract(data, '$.avg_reward')::DOUBLE as avg_reward,
            json_extract(data, '$.total_episodes')::INTEGER as total_episodes,
            json_extract(data, '$.n_envs')::INTEGER as n_envs,
            json_extract(data, '$.start_episode')::INTEGER as start_episode,
            json_extract(data, '$.requested_episodes')::INTEGER as requested_episodes,
            json_extract(data, '$.rolling_accuracy')::DOUBLE as rolling_accuracy,
            json_extract(data, '$.env_accuracies') as env_accuracies
        FROM raw_events
        WHERE event_type = 'BATCH_EPOCH_COMPLETED'
    """,
    "trends": """
        CREATE OR REPLACE VIEW trends AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            event_type,
            json_extract(data, '$.batch_idx')::INTEGER as batch_idx,
            json_extract(data, '$.episodes_completed')::INTEGER as episodes_completed,
            json_extract(data, '$.rolling_delta')::DOUBLE as rolling_delta,
            json_extract(data, '$.rolling_avg_accuracy')::DOUBLE as rolling_avg_accuracy,
            json_extract(data, '$.prev_rolling_avg_accuracy')::DOUBLE as prev_rolling_avg_accuracy
        FROM raw_events
        WHERE event_type IN (
            'PLATEAU_DETECTED',
            'DEGRADATION_DETECTED',
            'IMPROVEMENT_DETECTED'
        )
    """,
    "seed_lifecycle": """
        CREATE OR REPLACE VIEW seed_lifecycle AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            event_type,
            seed_id,
            slot_id,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract_string(data, '$.blueprint_id') as blueprint_id,
            json_extract(data, '$.params')::INTEGER as params,
            json_extract(data, '$.params_added')::INTEGER as params_added,
            json_extract(data, '$.alpha')::DOUBLE as alpha,
            json_extract(data, '$.grad_ratio')::DOUBLE as grad_ratio,
            json_extract(data, '$.has_vanishing')::BOOLEAN as has_vanishing,
            json_extract(data, '$.has_exploding')::BOOLEAN as has_exploding,
            json_extract(data, '$.epochs_in_stage')::INTEGER as epochs_in_stage,
            json_extract(data, '$.blend_tempo_epochs')::INTEGER as blend_tempo_epochs,
            json_extract_string(data, '$.alpha_curve') as alpha_curve,
            json_extract_string(data, '$.from_stage') as from_stage,
            json_extract_string(data, '$.to_stage') as to_stage,
            json_extract(data, '$.accuracy_delta')::DOUBLE as accuracy_delta,
            json_extract(data, '$.improvement')::DOUBLE as improvement,
            json_extract(data, '$.counterfactual')::DOUBLE as counterfactual,
            json_extract(data, '$.blending_delta')::DOUBLE as blending_delta,
            json_extract(data, '$.epochs_total')::INTEGER as epochs_total,
            json_extract_string(data, '$.reason') as reason,
            json_extract_string(data, '$.initiator') as initiator,
            json_extract(data, '$.auto_pruned')::BOOLEAN as auto_pruned
        FROM raw_events
        WHERE event_type IN (
            'SEED_GERMINATED',
            'SEED_STAGE_CHANGED',
            'SEED_FOSSILIZED',
            'SEED_PRUNED'
        )
    """,
    "decisions": """
        CREATE OR REPLACE VIEW decisions AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract_string(data, '$.action_name') as action_name,
            json_extract(data, '$.action_success')::BOOLEAN as action_success,
            json_extract(data, '$.total_reward')::DOUBLE as total_reward,
            json_extract(data, '$.value_estimate')::DOUBLE as value_estimate,
            json_extract(data, '$.action_confidence')::DOUBLE as action_confidence,
            json_extract(data, '$.decision_entropy')::DOUBLE as decision_entropy,
            json_extract_string(data, '$.slot_id') as slot_id,
            json_extract_string(data, '$.blueprint_id') as blueprint_id,
            json_extract_string(data, '$.style') as style,
            json_extract_string(data, '$.blend_id') as blend_id,
            json_extract(data, '$.tempo_idx')::INTEGER as tempo_idx,
            json_extract(data, '$.alpha_target')::DOUBLE as alpha_target,
            json_extract_string(data, '$.alpha_speed') as alpha_speed,
            json_extract_string(data, '$.alpha_curve') as alpha_curve,
            json_extract_string(data, '$.alpha_algorithm') as alpha_algorithm,
            json_extract_string(data, '$.alpha_algorithm_selected') as alpha_algorithm_selected,
            json_extract(data, '$.op_masked')::BOOLEAN as op_masked,
            json_extract(data, '$.slot_masked')::BOOLEAN as slot_masked,
            json_extract(data, '$.blueprint_masked')::BOOLEAN as blueprint_masked,
            json_extract(data, '$.style_masked')::BOOLEAN as style_masked,
            json_extract(data, '$.tempo_masked')::BOOLEAN as tempo_masked,
            json_extract(data, '$.alpha_target_masked')::BOOLEAN as alpha_target_masked,
            json_extract(data, '$.alpha_speed_masked')::BOOLEAN as alpha_speed_masked,
            json_extract(data, '$.alpha_curve_masked')::BOOLEAN as alpha_curve_masked,
            json_extract(data, '$.slot_states') as slot_states,
            json_extract(data, '$.alternatives') as alternatives,
            json_extract(data, '$.head_telemetry.op_confidence')::DOUBLE as head_op_confidence,
            json_extract(data, '$.head_telemetry.slot_confidence')::DOUBLE as head_slot_confidence,
            json_extract(data, '$.head_telemetry.blueprint_confidence')::DOUBLE as head_blueprint_confidence,
            json_extract(data, '$.head_telemetry.style_confidence')::DOUBLE as head_style_confidence,
            json_extract(data, '$.head_telemetry.tempo_confidence')::DOUBLE as head_tempo_confidence,
            json_extract(data, '$.head_telemetry.alpha_target_confidence')::DOUBLE as head_alpha_target_confidence,
            json_extract(data, '$.head_telemetry.alpha_speed_confidence')::DOUBLE as head_alpha_speed_confidence,
            json_extract(data, '$.head_telemetry.curve_confidence')::DOUBLE as head_alpha_curve_confidence,
            json_extract(data, '$.head_telemetry.op_entropy')::DOUBLE as head_op_entropy,
            json_extract(data, '$.head_telemetry.slot_entropy')::DOUBLE as head_slot_entropy,
            json_extract(data, '$.head_telemetry.blueprint_entropy')::DOUBLE as head_blueprint_entropy,
            json_extract(data, '$.head_telemetry.style_entropy')::DOUBLE as head_style_entropy,
            json_extract(data, '$.head_telemetry.tempo_entropy')::DOUBLE as head_tempo_entropy,
            json_extract(data, '$.head_telemetry.alpha_target_entropy')::DOUBLE as head_alpha_target_entropy,
            json_extract(data, '$.head_telemetry.alpha_speed_entropy')::DOUBLE as head_alpha_speed_entropy,
            json_extract(data, '$.head_telemetry.curve_entropy')::DOUBLE as head_alpha_curve_entropy
        FROM raw_events
        WHERE
            event_type = 'ANALYTICS_SNAPSHOT'
            AND json_extract_string(data, '$.kind') = 'last_action'
    """,
    "rewards": """
        CREATE OR REPLACE VIEW rewards AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract_string(data, '$.action_name') as action_name,
            json_extract(data, '$.action_success')::BOOLEAN as action_success,
            json_extract(data, '$.total_reward')::DOUBLE as total_reward,
            -- All 28 fields from RewardComponentsTelemetry (nested under reward_components)
            -- Base signal
            json_extract(data, '$.reward_components.base_acc_delta')::DOUBLE as base_acc_delta,
            -- Contribution-primary signal
            json_extract(data, '$.reward_components.seed_contribution')::DOUBLE as seed_contribution,
            json_extract(data, '$.reward_components.bounded_attribution')::DOUBLE as bounded_attribution,
            json_extract(data, '$.reward_components.progress_since_germination')::DOUBLE as progress_since_germination,
            json_extract(data, '$.reward_components.attribution_discount')::DOUBLE as attribution_discount,
            json_extract(data, '$.reward_components.ratio_penalty')::DOUBLE as ratio_penalty,
            -- Penalties
            json_extract(data, '$.reward_components.compute_rent')::DOUBLE as compute_rent,
            json_extract(data, '$.reward_components.alpha_shock')::DOUBLE as alpha_shock,
            json_extract(data, '$.reward_components.blending_warning')::DOUBLE as blending_warning,
            json_extract(data, '$.reward_components.holding_warning')::DOUBLE as holding_warning,
            -- Bonuses
            json_extract(data, '$.reward_components.stage_bonus')::DOUBLE as stage_bonus,
            json_extract(data, '$.reward_components.pbrs_bonus')::DOUBLE as pbrs_bonus,
            json_extract(data, '$.reward_components.synergy_bonus')::DOUBLE as synergy_bonus,
            json_extract(data, '$.reward_components.action_shaping')::DOUBLE as action_shaping,
            json_extract(data, '$.reward_components.terminal_bonus')::DOUBLE as terminal_bonus,
            json_extract(data, '$.reward_components.fossilize_terminal_bonus')::DOUBLE as fossilize_terminal_bonus,
            json_extract(data, '$.reward_components.hindsight_credit')::DOUBLE as hindsight_credit,
            json_extract(data, '$.reward_components.num_fossilized_seeds')::INTEGER as num_fossilized_seeds,
            json_extract(data, '$.reward_components.num_contributing_fossilized')::INTEGER as num_contributing_fossilized,
            -- Context fields
            json_extract(data, '$.reward_components.seed_stage')::INTEGER as seed_stage,
            json_extract(data, '$.reward_components.val_acc')::DOUBLE as val_acc,
            json_extract(data, '$.reward_components.acc_at_germination')::DOUBLE as acc_at_germination,
            json_extract(data, '$.reward_components.host_baseline_acc')::DOUBLE as host_baseline_acc,
            json_extract(data, '$.reward_components.growth_ratio')::DOUBLE as growth_ratio,
            -- Computed diagnostic (available via to_dict() serialization)
            json_extract(data, '$.reward_components.shaped_reward_ratio')::DOUBLE as shaped_reward_ratio
        FROM raw_events
        WHERE
            event_type = 'ANALYTICS_SNAPSHOT'
            AND json_extract_string(data, '$.kind') = 'last_action'
    """,
    "batch_stats": """
        CREATE OR REPLACE VIEW batch_stats AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            json_extract(data, '$.episodes_completed')::INTEGER as episodes_completed,
            json_extract(data, '$.batch')::INTEGER as batch,
            json_extract(data, '$.inner_epoch')::INTEGER as inner_epoch,
            json_extract(data, '$.accuracy')::DOUBLE as accuracy,
            json_extract(data, '$.host_accuracy')::DOUBLE as host_accuracy,
            json_extract(data, '$.entropy')::DOUBLE as entropy,
            json_extract(data, '$.kl_divergence')::DOUBLE as kl_divergence,
            json_extract(data, '$.value_variance')::DOUBLE as value_variance,
            json_extract(data, '$.seeds_created')::INTEGER as seeds_created,
            json_extract(data, '$.seeds_fossilized')::INTEGER as seeds_fossilized,
            json_extract(data, '$.skipped_update')::BOOLEAN as skipped_update
        FROM raw_events
        WHERE
            event_type = 'ANALYTICS_SNAPSHOT'
            AND json_extract_string(data, '$.kind') = 'batch_stats'
    """,
    "anomalies": """
        CREATE OR REPLACE VIEW anomalies AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            epoch,
            event_type,
            message,
            data
        FROM raw_events
        WHERE event_type IN (
            'VALUE_COLLAPSE_DETECTED',
            'RATIO_EXPLOSION_DETECTED',
            'RATIO_COLLAPSE_DETECTED',
            'GRADIENT_ANOMALY',
            'GRADIENT_PATHOLOGY_DETECTED',
            'NUMERICAL_INSTABILITY_DETECTED',
            'GOVERNOR_PANIC',
            'GOVERNOR_ROLLBACK',
            'PLATEAU_DETECTED'
        )
    """,
    "episode_outcomes": """
        CREATE OR REPLACE VIEW episode_outcomes AS
        SELECT
            event_id,
            timestamp,
            run_dir,
            group_id,
            json_extract(data, '$.env_id')::INTEGER as env_id,
            json_extract(data, '$.episode_idx')::INTEGER as episode_idx,
            json_extract(data, '$.final_accuracy')::DOUBLE as final_accuracy,
            json_extract(data, '$.param_ratio')::DOUBLE as param_ratio,
            json_extract(data, '$.num_fossilized')::INTEGER as num_fossilized,
            json_extract(data, '$.num_contributing_fossilized')::INTEGER as num_contributing_fossilized,
            json_extract(data, '$.episode_reward')::DOUBLE as episode_reward,
            json_extract(data, '$.stability_score')::DOUBLE as stability_score,
            json_extract_string(data, '$.reward_mode') as reward_mode
        FROM raw_events
        WHERE event_type = 'EPISODE_OUTCOME'
    """,
}


def telemetry_has_event_files(telemetry_dir: str) -> bool:
    """Return True if the telemetry directory contains any events.jsonl files."""
    telemetry_path = Path(telemetry_dir)
    return any(telemetry_path.glob("*/events.jsonl"))


def _create_empty_raw_events_view(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE OR REPLACE VIEW raw_events AS
        SELECT
            CAST(NULL AS VARCHAR) as event_id,
            CAST(NULL AS VARCHAR) as event_type,
            CAST(NULL AS TIMESTAMP) as timestamp,
            CAST(NULL AS VARCHAR) as seed_id,
            CAST(NULL AS VARCHAR) as slot_id,
            CAST(NULL AS BIGINT) as epoch,
            CAST(NULL AS VARCHAR) as group_id,
            CAST(NULL AS VARCHAR) as message,
            CAST(NULL AS JSON) as data,
            CAST(NULL AS VARCHAR) as severity,
            CAST(NULL AS VARCHAR) as filename,
            CAST(NULL AS VARCHAR) as run_dir
        WHERE false
    """)


def create_views(conn: duckdb.DuckDBPyConnection, telemetry_dir: str) -> None:
    """Create all telemetry views on the given connection.

    Args:
        conn: DuckDB connection (in-memory or file-based)
        telemetry_dir: Path to telemetry directory containing run subdirectories
    """
    conn.execute("PRAGMA threads=4")

    has_files = telemetry_has_event_files(telemetry_dir)

    for view_name, view_sql in VIEW_DEFINITIONS.items():
        if view_name == "raw_events" and not has_files:
            _create_empty_raw_events_view(conn)
            continue

        sql = view_sql.format(telemetry_dir=telemetry_dir)
        try:
            conn.execute(sql)
        except duckdb.IOException as e:
            # If no files exist, create empty views with proper schema
            if "No files found" in str(e) and view_name == "raw_events":
                _create_empty_raw_events_view(conn)
            else:
                raise
