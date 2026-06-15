#!/usr/bin/env python3
"""Generate a reward-efficiency proof packet from Karn telemetry views."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import duckdb

from esper.karn.mcp.views import create_views, scan_ingestion_integrity
from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_STEPS,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
    STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
    STATIC_FINAL_SOURCE_MODE,
    STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT,
    STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
    STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
    STATIC_FINAL_SOURCE_TOPOLOGY_V1,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.rewards import ContributionRewardConfig, RewardMode
from esper.simic.training.proof_baselines import missing_required_baseline_modes


LEARNABILITY_COLUMNS: tuple[str, ...] = (
    "head_slot_learnable_fraction",
    "head_blueprint_learnable_fraction",
    "head_style_learnable_fraction",
    "head_tempo_learnable_fraction",
    "head_alpha_target_learnable_fraction",
    "head_alpha_speed_learnable_fraction",
    "head_alpha_curve_learnable_fraction",
    "head_op_learnable_fraction",
    "head_slot_gradient_state",
    "head_blueprint_gradient_state",
    "head_style_gradient_state",
    "head_tempo_gradient_state",
    "head_alpha_target_gradient_state",
    "head_alpha_speed_gradient_state",
    "head_alpha_curve_gradient_state",
    "head_op_gradient_state",
    "head_value_grad_norm",
    "head_value_gradient_state",
)

VALID_OUTCOME_PREDICATE = """
    env_id IS NOT NULL
    AND env_id >= 0
    AND episode_idx IS NOT NULL
    AND episode_idx >= 0
    AND final_accuracy IS NOT NULL
    AND final_accuracy BETWEEN 0.0 AND 100.0
    AND param_ratio IS NOT NULL
    AND param_ratio >= 1.0
    AND num_fossilized IS NOT NULL
    AND num_fossilized >= 0
    AND num_contributing_fossilized IS NOT NULL
    AND num_contributing_fossilized >= 0
    AND num_contributing_fossilized <= num_fossilized
    AND episode_reward IS NOT NULL
    AND stability_score IS NOT NULL
    AND stability_score BETWEEN 0.0 AND 1.0
    AND reward_mode IS NOT NULL
    AND reward_mode <> ''
    AND episode_length IS NOT NULL
    AND episode_length > 0
    AND outcome_type IS NOT NULL
    AND outcome_type <> ''
    AND germinate_count IS NOT NULL
    AND germinate_count >= 0
    AND prune_count IS NOT NULL
    AND prune_count >= 0
    AND fossilize_count IS NOT NULL
    AND fossilize_count >= 0
"""

BLOCKED_INSTRUMENTATION = "BLOCKED_INSTRUMENTATION"
BLOCKED_PRECISION = "BLOCKED_PRECISION"
BLOCKED_MATH = "BLOCKED_MATH"
BLOCKED_MECHANICS = "BLOCKED_MECHANICS"
REVISE_ALGORITHM = "REVISE_ALGORITHM"
STOP_THEORY = "STOP_THEORY"
CONTINUE = "CONTINUE"

ProofProfile = Literal["generic", "reward-efficiency"]
GENERIC_PROOF_PROFILE: ProofProfile = "generic"
REWARD_EFFICIENCY_PROOF_PROFILE: ProofProfile = "reward-efficiency"
PROOF_PROFILES: tuple[ProofProfile, ...] = (
    GENERIC_PROOF_PROFILE,
    REWARD_EFFICIENCY_PROOF_PROFILE,
)
BLUEPRINT_ECONOMY_REWARD_MODES: frozenset[str] = frozenset(
    mode.value
    for mode in RewardMode
    if ContributionRewardConfig(reward_mode=mode).supports_blueprint_economy_evidence
)
ALL_REWARD_MODE_VALUES: frozenset[str] = frozenset(mode.value for mode in RewardMode)


def _rows(conn: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    result = conn.execute(query)
    columns = [col[0] for col in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def _blueprint_economy_roi_rows(roi_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in roi_rows
        if row["reward_mode"] in BLUEPRINT_ECONOMY_REWARD_MODES
    ]


def _unsupported_blueprint_economy_roi_rows(
    roi_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in roi_rows
        if row["reward_mode"] not in BLUEPRINT_ECONOMY_REWARD_MODES
    ]


def _reward_mode_evidence_reason(reward_mode: str) -> str:
    if reward_mode in ALL_REWARD_MODE_VALUES:
        return "diagnostic-only"
    return "unsupported"


def _missing_learnability_query() -> str:
    missing_predicate = " OR ".join(f"{column} IS NULL" for column in LEARNABILITY_COLUMNS)
    return f"""
        SELECT
            run_dir,
            group_id,
            COUNT(*) AS missing_update_count
        FROM ppo_updates
        WHERE {missing_predicate}
        GROUP BY run_dir, group_id
        ORDER BY run_dir, group_id
    """


def _invalid_outcomes_query() -> str:
    return """
        WITH checked AS (
            SELECT
                event_id,
                run_dir,
                group_id,
                env_id,
                episode_idx,
                reward_mode,
                rtrim(concat(
                    CASE WHEN env_id IS NULL THEN 'missing env_id, ' ELSE '' END,
                    CASE WHEN env_id < 0 THEN 'env_id < 0, ' ELSE '' END,
                    CASE WHEN episode_idx IS NULL THEN 'missing episode_idx, ' ELSE '' END,
                    CASE WHEN episode_idx < 0 THEN 'episode_idx < 0, ' ELSE '' END,
                    CASE WHEN final_accuracy IS NULL THEN 'missing final_accuracy, ' ELSE '' END,
                    CASE
                        WHEN final_accuracy < 0.0 OR final_accuracy > 100.0
                        THEN 'final_accuracy outside 0..100, '
                        ELSE ''
                    END,
                    CASE WHEN param_ratio IS NULL THEN 'missing param_ratio, ' ELSE '' END,
                    CASE WHEN param_ratio < 1.0 THEN 'param_ratio < 1.0, ' ELSE '' END,
                    CASE WHEN num_fossilized IS NULL THEN 'missing num_fossilized, ' ELSE '' END,
                    CASE WHEN num_fossilized < 0 THEN 'num_fossilized < 0, ' ELSE '' END,
                    CASE
                        WHEN num_contributing_fossilized IS NULL
                        THEN 'missing num_contributing_fossilized, '
                        ELSE ''
                    END,
                    CASE
                        WHEN num_contributing_fossilized < 0
                        THEN 'num_contributing_fossilized < 0, '
                        ELSE ''
                    END,
                    CASE
                        WHEN num_contributing_fossilized > num_fossilized
                        THEN 'num_contributing_fossilized > num_fossilized, '
                        ELSE ''
                    END,
                    CASE WHEN episode_reward IS NULL THEN 'missing episode_reward, ' ELSE '' END,
                    CASE WHEN stability_score IS NULL THEN 'missing stability_score, ' ELSE '' END,
                    CASE
                        WHEN stability_score < 0.0 OR stability_score > 1.0
                        THEN 'stability_score outside 0..1, '
                        ELSE ''
                    END,
                    CASE WHEN reward_mode IS NULL THEN 'missing reward_mode, ' ELSE '' END,
                    CASE WHEN reward_mode = '' THEN 'empty reward_mode, ' ELSE '' END,
                    CASE WHEN episode_length IS NULL THEN 'missing episode_length, ' ELSE '' END,
                    CASE WHEN episode_length <= 0 THEN 'episode_length <= 0, ' ELSE '' END,
                    CASE WHEN outcome_type IS NULL THEN 'missing outcome_type, ' ELSE '' END,
                    CASE WHEN outcome_type = '' THEN 'empty outcome_type, ' ELSE '' END,
                    CASE WHEN germinate_count IS NULL THEN 'missing germinate_count, ' ELSE '' END,
                    CASE WHEN germinate_count < 0 THEN 'germinate_count < 0, ' ELSE '' END,
                    CASE WHEN prune_count IS NULL THEN 'missing prune_count, ' ELSE '' END,
                    CASE WHEN prune_count < 0 THEN 'prune_count < 0, ' ELSE '' END,
                    CASE WHEN fossilize_count IS NULL THEN 'missing fossilize_count, ' ELSE '' END,
                    CASE WHEN fossilize_count < 0 THEN 'fossilize_count < 0, ' ELSE '' END
                ), ', ') AS violations
            FROM episode_outcomes
        )
        SELECT
            event_id,
            run_dir,
            group_id,
            env_id,
            episode_idx,
            reward_mode,
            violations
        FROM checked
        WHERE violations <> ''
        ORDER BY run_dir, group_id, event_id
    """


def _baseline_evidence_rows_query() -> str:
    return f"""
        WITH valid_outcomes AS (
            SELECT
                run_dir,
                group_id,
                reward_mode,
                COUNT(*) AS valid_outcome_count
            FROM episode_outcomes
            WHERE {VALID_OUTCOME_PREDICATE}
            GROUP BY run_dir, group_id, reward_mode
        )
        SELECT
            runs.proof_baseline_mode,
            runs.proof_baseline_pair_id,
            runs.proof_baseline_lifecycle_policy,
            runs.proof_baseline_schedule_id,
            runs.proof_baseline_schedule_hash,
            runs.proof_baseline_schedule_version,
            runs.proof_baseline_schedule_action_count,
            COUNT(*) AS run_count,
            SUM(valid_outcomes.valid_outcome_count) AS valid_outcome_count
        FROM runs
        JOIN valid_outcomes
          ON valid_outcomes.run_dir = runs.run_dir
         AND COALESCE(valid_outcomes.group_id, '') = COALESCE(runs.group_id, '')
         AND valid_outcomes.reward_mode = runs.reward_mode
        WHERE runs.proof_baseline_mode IS NOT NULL
          AND runs.proof_baseline_mode <> ''
          AND runs.proof_baseline_mode <> '{STATIC_FINAL_SOURCE_MODE}'
          AND runs.proof_baseline_pair_id IS NOT NULL
          AND runs.proof_baseline_pair_id <> ''
          AND runs.proof_baseline_lifecycle_policy IS NOT NULL
          AND runs.proof_baseline_lifecycle_policy <> ''
        GROUP BY
            runs.proof_baseline_mode,
            runs.proof_baseline_pair_id,
            runs.proof_baseline_lifecycle_policy,
            runs.proof_baseline_schedule_id,
            runs.proof_baseline_schedule_hash,
            runs.proof_baseline_schedule_version,
            runs.proof_baseline_schedule_action_count
        ORDER BY
            runs.proof_baseline_mode,
            runs.proof_baseline_pair_id,
            runs.proof_baseline_lifecycle_policy,
            runs.proof_baseline_schedule_id,
            runs.proof_baseline_schedule_hash,
            runs.proof_baseline_schedule_version,
            runs.proof_baseline_schedule_action_count
    """


def _baseline_evidence_blockers_query() -> str:
    return f"""
        WITH valid_outcomes AS (
            SELECT
                run_dir,
                group_id,
                reward_mode,
                COUNT(*) AS valid_outcome_count
            FROM episode_outcomes
            WHERE {VALID_OUTCOME_PREDICATE}
            GROUP BY run_dir, group_id, reward_mode
        ),
        checked AS (
            SELECT
                runs.run_dir,
                runs.group_id,
                runs.reward_mode,
                runs.proof_baseline_mode,
                runs.proof_baseline_pair_id,
                runs.proof_baseline_lifecycle_policy,
                runs.proof_baseline_schedule_id,
                runs.proof_baseline_schedule_hash,
                runs.proof_baseline_schedule_version,
                runs.proof_baseline_schedule_action_count,
                valid_outcomes.valid_outcome_count,
                CASE
                    WHEN valid_outcomes.valid_outcome_count IS NULL
                    THEN 'missing valid EPISODE_OUTCOME for baseline cohort reward_mode'
                    ELSE ''
                END AS violations
            FROM runs
            LEFT JOIN valid_outcomes
              ON valid_outcomes.run_dir = runs.run_dir
             AND COALESCE(valid_outcomes.group_id, '') = COALESCE(runs.group_id, '')
             AND valid_outcomes.reward_mode = runs.reward_mode
            WHERE runs.proof_baseline_mode IS NOT NULL
              AND runs.proof_baseline_mode <> ''
              AND runs.proof_baseline_mode <> '{STATIC_FINAL_SOURCE_MODE}'
        )
        SELECT
            run_dir,
            group_id,
            reward_mode,
            proof_baseline_mode,
            proof_baseline_pair_id,
            proof_baseline_lifecycle_policy,
            proof_baseline_schedule_id,
            proof_baseline_schedule_hash,
            proof_baseline_schedule_version,
            proof_baseline_schedule_action_count,
            violations
        FROM checked
        WHERE violations <> ''
        ORDER BY run_dir, group_id
    """


def _lockstep_baseline_blockers_query() -> str:
    return f"""
        WITH valid_outcomes AS (
            SELECT
                run_dir,
                group_id,
                reward_mode,
                COUNT(*) AS valid_outcome_count
            FROM episode_outcomes
            WHERE {VALID_OUTCOME_PREDICATE}
            GROUP BY run_dir, group_id, reward_mode
        ),
        valid_lockstep_runs AS (
            SELECT
                runs.run_dir,
                runs.group_id,
                runs.reward_mode,
                runs.seed,
                runs.proof_baseline_pair_id
            FROM runs
            JOIN valid_outcomes
              ON valid_outcomes.run_dir = runs.run_dir
             AND COALESCE(valid_outcomes.group_id, '') = COALESCE(runs.group_id, '')
             AND valid_outcomes.reward_mode = runs.reward_mode
            WHERE runs.proof_baseline_mode = 'lockstep_reward_ab'
              AND runs.proof_baseline_pair_id IS NOT NULL
              AND runs.proof_baseline_pair_id <> ''
              AND runs.proof_baseline_lifecycle_policy IS NOT NULL
              AND runs.proof_baseline_lifecycle_policy <> ''
        ),
        pair_stats AS (
            SELECT
                proof_baseline_pair_id,
                COUNT(DISTINCT CONCAT(run_dir, ':', COALESCE(group_id, ''))) AS run_count,
                COUNT(DISTINCT reward_mode) AS reward_mode_count,
                COUNT(seed) AS seeded_run_count,
                COUNT(DISTINCT seed) AS seed_count,
                STRING_AGG(DISTINCT reward_mode, ', ' ORDER BY reward_mode) AS reward_modes,
                rtrim(concat(
                    CASE
                        WHEN COUNT(DISTINCT CONCAT(run_dir, ':', COALESCE(group_id, ''))) < 2
                        THEN 'requires at least two outcome-bearing runs, '
                        ELSE ''
                    END,
                    CASE
                        WHEN COUNT(DISTINCT reward_mode) < 2
                        THEN 'requires distinct reward modes, '
                        ELSE ''
                    END,
                    CASE
                        WHEN COUNT(seed) <> COUNT(*)
                        THEN 'missing seed provenance, '
                        ELSE ''
                    END,
                    CASE
                        WHEN COUNT(seed) = COUNT(*) AND COUNT(DISTINCT seed) <> 1
                        THEN 'requires matching seed, '
                        ELSE ''
                    END
                ), ', ') AS violations
            FROM valid_lockstep_runs
            GROUP BY proof_baseline_pair_id
        )
        SELECT
            proof_baseline_pair_id,
            run_count,
            reward_mode_count,
            seeded_run_count,
            seed_count,
            reward_modes,
            violations
        FROM pair_stats
        WHERE violations <> ''
        ORDER BY proof_baseline_pair_id
    """


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fixed_schedule_expected_trace_values_sql() -> str:
    slot_config = SlotConfig.default()
    rows = []
    for step in FIXED_SCHEDULE_GERMINATE_R0C0_STEPS:
        if step.action.op.name == "WAIT":
            continue
        rows.append(
            "("
            f"{step.epoch}, "
            f"{_sql_string(slot_config.slot_id_for_index(step.action.slot_idx))}, "
            f"{_sql_string(step.action.op.name)}, "
            f"{_sql_string(step.action.blueprint.to_blueprint_id())}"
            ")"
        )
    return ",\n".join(rows)


def _fixed_schedule_trace_blockers_query() -> str:
    expected_values = _fixed_schedule_expected_trace_values_sql()
    return f"""
        WITH expected_schedule AS (
            SELECT
                expected_epoch::INTEGER AS expected_epoch,
                expected_slot_id::VARCHAR AS expected_slot_id,
                expected_operation::VARCHAR AS expected_operation,
                expected_blueprint_id::VARCHAR AS expected_blueprint_id
            FROM (
                VALUES {expected_values}
            ) AS expected(
                expected_epoch,
                expected_slot_id,
                expected_operation,
                expected_blueprint_id
            )
        ),
        fixed_schedule_runs AS (
            SELECT
                run_dir,
                group_id,
                proof_baseline_schedule_id
            FROM runs
            WHERE proof_baseline_mode = 'fixed_schedule'
              AND proof_baseline_schedule_id = '{FIXED_SCHEDULE_GERMINATE_R0C0_V1}'
              AND proof_baseline_schedule_hash = '{FIXED_SCHEDULE_GERMINATE_R0C0_HASH}'
              AND proof_baseline_schedule_version = {FIXED_SCHEDULE_GERMINATE_R0C0_VERSION}
              AND proof_baseline_schedule_action_count = {FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT}
        ),
        expected_by_run AS (
            SELECT
                fixed_schedule_runs.run_dir,
                fixed_schedule_runs.group_id,
                fixed_schedule_runs.proof_baseline_schedule_id,
                expected_schedule.expected_epoch,
                expected_schedule.expected_slot_id,
                expected_schedule.expected_operation,
                expected_schedule.expected_blueprint_id
            FROM fixed_schedule_runs
            CROSS JOIN expected_schedule
        ),
        missing AS (
            SELECT
                expected_by_run.run_dir,
                expected_by_run.group_id,
                expected_by_run.proof_baseline_schedule_id,
                expected_by_run.expected_epoch,
                expected_by_run.expected_slot_id,
                expected_by_run.expected_operation,
                expected_by_run.expected_blueprint_id,
                NULL::VARCHAR AS observed_event_id,
                NULL::VARCHAR AS observed_phase,
                NULL::VARCHAR AS observed_operation,
                NULL::VARCHAR AS observed_blueprint_id,
                NULL::BOOLEAN AS observed_governor_approved,
                'missing declared fixed-schedule action trace' AS violations
            FROM expected_by_run
            WHERE NOT EXISTS (
                SELECT 1
                FROM morphology_causal_log
                WHERE morphology_causal_log.run_dir = expected_by_run.run_dir
                  AND COALESCE(morphology_causal_log.group_id, '') = COALESCE(expected_by_run.group_id, '')
                  AND morphology_causal_log.epoch = expected_by_run.expected_epoch
                  AND morphology_causal_log.slot_id = expected_by_run.expected_slot_id
                  AND morphology_causal_log.phase IN ('commit', 'fossilization')
            )
        ),
        mismatched AS (
            SELECT
                expected_by_run.run_dir,
                expected_by_run.group_id,
                expected_by_run.proof_baseline_schedule_id,
                expected_by_run.expected_epoch,
                expected_by_run.expected_slot_id,
                expected_by_run.expected_operation,
                expected_by_run.expected_blueprint_id,
                morphology_causal_log.event_id AS observed_event_id,
                morphology_causal_log.phase AS observed_phase,
                morphology_causal_log.operation AS observed_operation,
                morphology_causal_log.blueprint_id AS observed_blueprint_id,
                morphology_causal_log.governor_approved AS observed_governor_approved,
                'mismatched declared fixed-schedule action trace' AS violations
            FROM expected_by_run
            JOIN morphology_causal_log
              ON morphology_causal_log.run_dir = expected_by_run.run_dir
             AND COALESCE(morphology_causal_log.group_id, '') = COALESCE(expected_by_run.group_id, '')
             AND morphology_causal_log.epoch = expected_by_run.expected_epoch
             AND morphology_causal_log.slot_id = expected_by_run.expected_slot_id
             AND morphology_causal_log.phase IN ('commit', 'fossilization')
            WHERE morphology_causal_log.operation IS DISTINCT FROM expected_by_run.expected_operation
               OR morphology_causal_log.blueprint_id IS DISTINCT FROM expected_by_run.expected_blueprint_id
               OR morphology_causal_log.governor_approved IS DISTINCT FROM TRUE
        ),
        unexpected AS (
            SELECT
                fixed_schedule_runs.run_dir,
                fixed_schedule_runs.group_id,
                fixed_schedule_runs.proof_baseline_schedule_id,
                morphology_causal_log.epoch AS expected_epoch,
                morphology_causal_log.slot_id AS expected_slot_id,
                NULL::VARCHAR AS expected_operation,
                NULL::VARCHAR AS expected_blueprint_id,
                morphology_causal_log.event_id AS observed_event_id,
                morphology_causal_log.phase AS observed_phase,
                morphology_causal_log.operation AS observed_operation,
                morphology_causal_log.blueprint_id AS observed_blueprint_id,
                morphology_causal_log.governor_approved AS observed_governor_approved,
                'unexpected fixed-schedule action trace' AS violations
            FROM fixed_schedule_runs
            JOIN morphology_causal_log
              ON morphology_causal_log.run_dir = fixed_schedule_runs.run_dir
             AND COALESCE(morphology_causal_log.group_id, '') = COALESCE(fixed_schedule_runs.group_id, '')
             AND morphology_causal_log.phase IN ('commit', 'fossilization')
            WHERE NOT EXISTS (
                SELECT 1
                FROM expected_schedule
                WHERE expected_schedule.expected_epoch = morphology_causal_log.epoch
                  AND expected_schedule.expected_slot_id = morphology_causal_log.slot_id
            )
        )
        SELECT *
        FROM missing
        UNION ALL
        SELECT *
        FROM mismatched
        UNION ALL
        SELECT *
        FROM unexpected
        ORDER BY run_dir, group_id, expected_epoch, expected_slot_id, observed_event_id
    """


def _static_final_replay_blockers_query() -> str:
    return f"""
        WITH static_final_runs AS (
            SELECT
                run_dir,
                group_id,
                reward_mode,
                proof_baseline_pair_id,
                proof_baseline_lifecycle_policy
            FROM runs
            WHERE proof_baseline_mode = 'static_final'
        ),
        valid_episode_outcomes AS (
            SELECT *
            FROM episode_outcomes
            WHERE {VALID_OUTCOME_PREDICATE}
        ),
        valid_static_final_outcomes AS (
            SELECT
                static_final_runs.run_dir,
                static_final_runs.group_id,
                static_final_runs.reward_mode,
                static_final_runs.proof_baseline_pair_id,
                static_final_runs.proof_baseline_lifecycle_policy,
                valid_episode_outcomes.event_id AS outcome_event_id,
                valid_episode_outcomes.env_id AS target_env_id,
                valid_episode_outcomes.episode_idx AS target_episode_idx
            FROM static_final_runs
            JOIN valid_episode_outcomes
              ON valid_episode_outcomes.run_dir = static_final_runs.run_dir
             AND COALESCE(valid_episode_outcomes.group_id, '') = COALESCE(static_final_runs.group_id, '')
             AND valid_episode_outcomes.reward_mode = static_final_runs.reward_mode
        ),
        replay_rows AS (
            SELECT *
            FROM topology_manifests
            WHERE manifest_role = 'static_final_replay'
        ),
        source_rows AS (
            SELECT *
            FROM topology_manifests
            WHERE manifest_role = 'source_final'
        ),
        source_runs AS (
            SELECT
                run_dir AS source_run_run_dir,
                group_id AS source_run_group_id,
                proof_baseline_mode AS source_run_proof_baseline_mode,
                proof_baseline_lifecycle_policy AS source_run_lifecycle_policy,
                proof_baseline_schedule_id AS source_run_schedule_id,
                proof_baseline_schedule_hash AS source_run_schedule_hash,
                proof_baseline_schedule_version AS source_run_schedule_version,
                proof_baseline_schedule_action_count AS source_run_schedule_action_count
            FROM runs
            WHERE proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
        ),
        matched_replay AS (
            SELECT
                valid_static_final_outcomes.run_dir,
                valid_static_final_outcomes.group_id,
                valid_static_final_outcomes.proof_baseline_pair_id,
                valid_static_final_outcomes.proof_baseline_lifecycle_policy,
                valid_static_final_outcomes.outcome_event_id,
                valid_static_final_outcomes.target_env_id,
                valid_static_final_outcomes.target_episode_idx,
                replay_rows.event_id AS replay_event_id,
                replay_rows.epoch AS replay_epoch,
                replay_rows.proof_baseline_pair_id AS replay_pair_id,
                replay_rows.topology_manifest_version AS replay_manifest_version,
                replay_rows.topology_manifest_hash AS replay_manifest_hash,
                replay_rows.topology_manifest_json AS replay_manifest_json,
                replay_rows.task AS replay_task,
                replay_rows.host_topology AS replay_host_topology,
                replay_rows.slot_config_hash AS replay_slot_config_hash,
                replay_rows.slot_count AS replay_slot_count,
                replay_rows.fossilized_seed_count AS replay_fossilized_seed_count,
                replay_rows.topology_delta_count AS replay_topology_delta_count,
                replay_rows.source_run_dir,
                replay_rows.source_group_id,
                replay_rows.source_episode_idx,
                replay_rows.source_event_id,
                replay_rows.source_topology_manifest_hash,
                replay_rows.replay_weight_policy,
                replay_rows.replay_env_id,
                replay_rows.replay_episode_idx,
                replay_rows.replayed_topology_manifest_hash,
                replay_rows.manifest_match
            FROM valid_static_final_outcomes
            LEFT JOIN replay_rows
              ON replay_rows.run_dir = valid_static_final_outcomes.run_dir
             AND COALESCE(replay_rows.group_id, '') = COALESCE(valid_static_final_outcomes.group_id, '')
             AND replay_rows.proof_baseline_pair_id = valid_static_final_outcomes.proof_baseline_pair_id
             AND replay_rows.replay_env_id = valid_static_final_outcomes.target_env_id
             AND replay_rows.replay_episode_idx = valid_static_final_outcomes.target_episode_idx
        ),
        matched AS (
            SELECT
                matched_replay.*,
                source_rows.event_id AS matched_source_event_id,
                source_rows.topology_manifest_hash AS matched_source_manifest_hash,
                source_rows.topology_manifest_version AS source_manifest_version,
                source_rows.task AS source_task,
                source_rows.host_topology AS source_host_topology,
                source_rows.slot_config_hash AS source_slot_config_hash,
                source_rows.slot_count AS source_slot_count,
                source_rows.fossilized_seed_count AS source_fossilized_seed_count,
                source_rows.topology_delta_count AS source_topology_delta_count,
                source_runs.source_run_proof_baseline_mode,
                source_runs.source_run_lifecycle_policy,
                source_runs.source_run_schedule_id,
                source_runs.source_run_schedule_hash,
                source_runs.source_run_schedule_version,
                source_runs.source_run_schedule_action_count,
                (
                    SELECT COUNT(*)
                    FROM morphology_causal_log
                    WHERE morphology_causal_log.run_dir = matched_replay.run_dir
                      AND COALESCE(morphology_causal_log.group_id, '') = COALESCE(matched_replay.group_id, '')
                      AND morphology_causal_log.phase IN ('commit', 'fossilization')
                ) AS static_final_mutation_count
            FROM matched_replay
            LEFT JOIN source_rows
              ON source_rows.run_dir = matched_replay.source_run_dir
             AND COALESCE(source_rows.group_id, '') = COALESCE(matched_replay.source_group_id, '')
             AND source_rows.event_id = matched_replay.source_event_id
             AND source_rows.topology_manifest_hash = matched_replay.source_topology_manifest_hash
            LEFT JOIN source_runs
              ON source_runs.source_run_run_dir = source_rows.run_dir
             AND COALESCE(source_runs.source_run_group_id, '') = COALESCE(source_rows.group_id, '')
        ),
        checked AS (
            SELECT
                *,
                rtrim(concat(
                    CASE
                        WHEN replay_event_id IS NULL
                        THEN 'missing static-final topology replay evidence, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND proof_baseline_lifecycle_policy IS DISTINCT FROM 'freeze_replayed_final_topology'
                        THEN 'unexpected proof_baseline_lifecycle_policy, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_epoch IS NULL
                        THEN 'missing replay epoch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_pair_id IS DISTINCT FROM proof_baseline_pair_id
                        THEN 'replay proof_baseline_pair_id mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_manifest_version IS NULL
                        THEN 'missing topology_manifest_version, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_manifest_version < 1
                        THEN 'topology_manifest_version < 1, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (replay_manifest_hash IS NULL OR replay_manifest_hash = '')
                        THEN 'missing topology_manifest_hash, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (replay_manifest_json IS NULL OR replay_manifest_json = '')
                        THEN 'missing topology_manifest_json, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (replay_task IS NULL OR replay_task = '')
                        THEN 'missing task, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (replay_host_topology IS NULL OR replay_host_topology = '')
                        THEN 'missing host_topology, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (replay_slot_config_hash IS NULL OR replay_slot_config_hash = '')
                        THEN 'missing slot_config_hash, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_slot_count IS NULL
                        THEN 'missing slot_count, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_slot_count <= 0
                        THEN 'slot_count <= 0, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_fossilized_seed_count IS NULL
                        THEN 'missing fossilized_seed_count, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_fossilized_seed_count <= 0
                        THEN 'fossilized_seed_count <= 0, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_topology_delta_count IS NULL
                        THEN 'missing topology_delta_count, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_topology_delta_count <= 0
                        THEN 'topology_delta_count <= 0, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (source_run_dir IS NULL OR source_run_dir = '')
                        THEN 'missing source_run_dir, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (source_event_id IS NULL OR source_event_id = '')
                        THEN 'missing source_event_id, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_episode_idx IS NULL
                        THEN 'missing source_episode_idx, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (source_topology_manifest_hash IS NULL OR source_topology_manifest_hash = '')
                        THEN 'missing source_topology_manifest_hash, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NULL
                        THEN 'missing matching source-final topology manifest, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_proof_baseline_mode IS NULL
                        THEN 'missing static-final source run provenance, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_lifecycle_policy IS DISTINCT FROM '{STATIC_FINAL_SOURCE_LIFECYCLE_POLICY}'
                        THEN 'unexpected static-final source lifecycle policy, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND (
                            source_run_schedule_id IS NULL
                            OR source_run_schedule_id = ''
                         )
                        THEN 'missing static-final source schedule_id, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_schedule_id IS NOT NULL
                         AND source_run_schedule_id <> ''
                         AND source_run_schedule_id <> '{STATIC_FINAL_SOURCE_TOPOLOGY_V1}'
                        THEN 'unexpected static-final source schedule_id, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND (
                            source_run_schedule_hash IS NULL
                            OR source_run_schedule_hash = ''
                         )
                        THEN 'missing static-final source schedule_hash, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_schedule_hash IS NOT NULL
                         AND source_run_schedule_hash <> ''
                         AND source_run_schedule_hash <> '{STATIC_FINAL_SOURCE_TOPOLOGY_HASH}'
                        THEN 'unexpected static-final source schedule_hash, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_schedule_version IS NULL
                        THEN 'missing static-final source schedule_version, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_schedule_version IS NOT NULL
                         AND source_run_schedule_version <> {STATIC_FINAL_SOURCE_TOPOLOGY_VERSION}
                        THEN 'unexpected static-final source schedule_version, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_schedule_action_count IS NULL
                        THEN 'missing static-final source schedule_action_count, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_event_id IS NOT NULL
                         AND source_run_schedule_action_count IS NOT NULL
                         AND source_run_schedule_action_count <> {STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT}
                        THEN 'unexpected static-final source schedule_action_count, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (replayed_topology_manifest_hash IS NULL OR replayed_topology_manifest_hash = '')
                        THEN 'missing replayed_topology_manifest_hash, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_topology_manifest_hash IS NOT NULL
                         AND source_topology_manifest_hash <> ''
                         AND replayed_topology_manifest_hash IS NOT NULL
                         AND replayed_topology_manifest_hash <> ''
                         AND source_topology_manifest_hash <> replayed_topology_manifest_hash
                        THEN 'source/replayed topology manifest hash mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND matched_source_manifest_hash IS NOT NULL
                         AND matched_source_manifest_hash IS DISTINCT FROM replayed_topology_manifest_hash
                        THEN 'matched source manifest hash mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_manifest_hash IS NOT NULL
                         AND replayed_topology_manifest_hash IS NOT NULL
                         AND replay_manifest_hash IS DISTINCT FROM replayed_topology_manifest_hash
                        THEN 'replay manifest hash mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND replay_weight_policy = 'state_dict_exact'
                        THEN 'state_dict_exact replay_weight_policy lacks source/replay state-dict hash evidence, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND (replay_weight_policy IS NULL OR replay_weight_policy <> 'topology_only')
                        THEN 'missing or unexpected replay_weight_policy, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND manifest_match IS DISTINCT FROM TRUE
                        THEN 'manifest_match is not true, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_manifest_version IS NOT NULL
                         AND source_manifest_version IS DISTINCT FROM replay_manifest_version
                        THEN 'source/replay manifest version mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_task IS NOT NULL
                         AND source_task IS DISTINCT FROM replay_task
                        THEN 'source/replay task mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_host_topology IS NOT NULL
                         AND source_host_topology IS DISTINCT FROM replay_host_topology
                        THEN 'source/replay host_topology mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_slot_config_hash IS NOT NULL
                         AND source_slot_config_hash IS DISTINCT FROM replay_slot_config_hash
                        THEN 'source/replay slot_config_hash mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_slot_count IS NOT NULL
                         AND source_slot_count IS DISTINCT FROM replay_slot_count
                        THEN 'source/replay slot_count mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_fossilized_seed_count IS NOT NULL
                         AND source_fossilized_seed_count IS DISTINCT FROM replay_fossilized_seed_count
                        THEN 'source/replay fossilized_seed_count mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND source_topology_delta_count IS NOT NULL
                         AND source_topology_delta_count IS DISTINCT FROM replay_topology_delta_count
                        THEN 'source/replay topology_delta_count mismatch, '
                        ELSE ''
                    END,
                    CASE
                        WHEN replay_event_id IS NOT NULL
                         AND static_final_mutation_count > 0
                        THEN 'static-final run mutated topology, '
                        ELSE ''
                    END
                ), ', ') AS violations
            FROM matched
        )
        SELECT
            run_dir,
            group_id,
            proof_baseline_pair_id,
            proof_baseline_lifecycle_policy,
            outcome_event_id,
            target_env_id,
            target_episode_idx,
            replay_event_id,
            replay_epoch,
            source_run_dir,
            source_group_id,
            source_episode_idx,
            source_event_id,
            replay_manifest_hash,
            source_topology_manifest_hash,
            replayed_topology_manifest_hash,
            matched_source_event_id,
            source_run_schedule_id,
            source_run_schedule_hash,
            source_run_schedule_version,
            source_run_schedule_action_count,
            replay_weight_policy,
            replay_env_id,
            replay_episode_idx,
            manifest_match,
            replay_fossilized_seed_count,
            replay_topology_delta_count,
            static_final_mutation_count,
            violations
        FROM checked
        WHERE violations <> ''
        ORDER BY run_dir, group_id, target_episode_idx, target_env_id, replay_event_id
    """


def _classify_verdict(
    *,
    blocking_confounders: list[dict[str, Any]],
    invalid_outcomes: list[dict[str, Any]],
    missing_learnability: list[dict[str, Any]],
    missing_baselines: tuple[str, ...],
    baseline_provenance_blockers: list[dict[str, Any]],
    baseline_evidence_blockers: list[dict[str, Any]],
    lockstep_baseline_blockers: list[dict[str, Any]],
    fixed_schedule_trace_blockers: list[dict[str, Any]],
    static_final_replay_blockers: list[dict[str, Any]],
    precision_blockers: list[dict[str, Any]],
    economy_evidence_blocking: bool,
    ingestion_blocking: bool,
    missing_runs: bool,
    missing_outcomes: bool,
    roi_rows: list[dict[str, Any]],
    min_mean_accuracy_roi: float,
) -> str:
    if ingestion_blocking or missing_runs or missing_outcomes or missing_learnability:
        return BLOCKED_INSTRUMENTATION
    if precision_blockers:
        return BLOCKED_PRECISION
    if (
        invalid_outcomes
        or missing_baselines
        or baseline_provenance_blockers
        or baseline_evidence_blockers
        or lockstep_baseline_blockers
        or fixed_schedule_trace_blockers
        or static_final_replay_blockers
        or economy_evidence_blocking
    ):
        return BLOCKED_MATH
    if blocking_confounders:
        return BLOCKED_MECHANICS

    roi_values = [
        float(row["mean_accuracy_roi"])
        for row in roi_rows
        if row["mean_accuracy_roi"] is not None
    ]
    if not roi_values:
        return BLOCKED_INSTRUMENTATION

    best_roi = max(roi_values)
    if best_roi <= 0.0:
        return STOP_THEORY
    if best_roi < min_mean_accuracy_roi:
        return REVISE_ALGORITHM
    return CONTINUE


def build_proof_packet(
    telemetry_dir: str,
    *,
    proof_profile: ProofProfile = REWARD_EFFICIENCY_PROOF_PROFILE,
    require_blueprint_health_baselines: bool | None = None,
    require_precision_provenance: bool | None = None,
    min_mean_accuracy_roi: float = 1.0,
) -> str:
    """Build a markdown proof packet from telemetry."""
    if proof_profile not in PROOF_PROFILES:
        raise ValueError(f"Unknown proof profile: {proof_profile}")
    if proof_profile == REWARD_EFFICIENCY_PROOF_PROFILE:
        if require_blueprint_health_baselines is False:
            raise ValueError(
                "proof_profile='reward-efficiency' requires blueprint-health baselines"
            )
        if require_precision_provenance is False:
            raise ValueError(
                "proof_profile='reward-efficiency' requires precision provenance"
            )
        resolved_require_blueprint_health_baselines = True
        resolved_require_precision_provenance = True
    else:
        resolved_require_blueprint_health_baselines = (
            False
            if require_blueprint_health_baselines is None
            else require_blueprint_health_baselines
        )
        resolved_require_precision_provenance = (
            True
            if require_precision_provenance is None
            else require_precision_provenance
        )

    # FAIL CLOSED on ingestion corruption: the raw_events view ingests with
    # ignore_errors=true, silently dropping malformed JSONL. Scan the files
    # independently first so a corrupt line cannot vanish from the proof.
    ingestion = scan_ingestion_integrity(telemetry_dir)

    conn = duckdb.connect(":memory:")
    try:
        create_views(conn, telemetry_dir)
        runs = _rows(
            conn,
            """
            SELECT
                run_dir,
                group_id,
                episode_id,
                task,
                reward_mode,
                seed,
                n_envs,
                n_episodes,
                max_epochs,
                max_batches,
                amp_enabled,
                amp_dtype,
                proof_baseline_mode,
                proof_baseline_pair_id,
                proof_baseline_lifecycle_policy,
                proof_baseline_schedule_id,
                proof_baseline_schedule_hash,
                proof_baseline_schedule_version,
                proof_baseline_schedule_action_count
            FROM runs
            ORDER BY started_at, run_dir, group_id
            """,
        )
        precision_blockers = (
            _rows(
                conn,
                """
                SELECT
                    run_dir,
                    group_id,
                    amp_enabled,
                    amp_dtype,
                    CASE
                        WHEN amp_enabled IS NULL THEN 'missing amp_enabled'
                        WHEN amp_enabled AND (amp_dtype IS NULL OR amp_dtype = '')
                            THEN 'missing amp_dtype for AMP run'
                    END AS violation
                FROM runs
                WHERE amp_enabled IS NULL
                   OR (amp_enabled AND (amp_dtype IS NULL OR amp_dtype = ''))
                ORDER BY run_dir, group_id
                """,
            )
            if resolved_require_precision_provenance
            else []
        )
        baseline_rows = _rows(
            conn,
            """
            SELECT
                proof_baseline_mode,
                proof_baseline_pair_id,
                proof_baseline_lifecycle_policy,
                proof_baseline_schedule_id,
                proof_baseline_schedule_hash,
                proof_baseline_schedule_version,
                proof_baseline_schedule_action_count,
                COUNT(*) AS run_count
            FROM runs
            WHERE proof_baseline_mode IS NOT NULL
              AND proof_baseline_mode <> ''
            GROUP BY
                proof_baseline_mode,
                proof_baseline_pair_id,
                proof_baseline_lifecycle_policy,
                proof_baseline_schedule_id,
                proof_baseline_schedule_hash,
                proof_baseline_schedule_version,
                proof_baseline_schedule_action_count
            ORDER BY
                proof_baseline_mode,
                proof_baseline_pair_id,
                proof_baseline_lifecycle_policy,
                proof_baseline_schedule_id,
                proof_baseline_schedule_hash,
                proof_baseline_schedule_version,
                proof_baseline_schedule_action_count
            """,
        )
        baseline_evidence_rows = _rows(conn, _baseline_evidence_rows_query())
        baseline_evidence_blockers = _rows(conn, _baseline_evidence_blockers_query())
        lockstep_baseline_blockers = _rows(conn, _lockstep_baseline_blockers_query())
        fixed_schedule_trace_blockers = _rows(
            conn,
            _fixed_schedule_trace_blockers_query(),
        )
        static_final_replay_blockers = _rows(
            conn,
            _static_final_replay_blockers_query(),
        )
        baseline_provenance_blockers = _rows(
            conn,
            f"""
            WITH checked AS (
                SELECT
                    run_dir,
                    group_id,
                    proof_baseline_mode,
                    proof_baseline_pair_id,
                    proof_baseline_lifecycle_policy,
                    proof_baseline_schedule_id,
                    proof_baseline_schedule_hash,
                    proof_baseline_schedule_version,
                    proof_baseline_schedule_action_count,
                    CONCAT(
                        CASE
                            WHEN proof_baseline_pair_id IS NULL
                              OR proof_baseline_pair_id = ''
                            THEN 'missing proof_baseline_pair_id, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_lifecycle_policy IS NULL
                              OR proof_baseline_lifecycle_policy = ''
                            THEN 'missing proof_baseline_lifecycle_policy, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND (
                                proof_baseline_schedule_id IS NULL
                                OR proof_baseline_schedule_id = ''
                              )
                            THEN 'missing proof_baseline_schedule_id, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND proof_baseline_schedule_id IS NOT NULL
                              AND proof_baseline_schedule_id <> ''
                              AND proof_baseline_schedule_id <> '{FIXED_SCHEDULE_GERMINATE_R0C0_V1}'
                            THEN 'unexpected proof_baseline_schedule_id, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND (
                                proof_baseline_schedule_hash IS NULL
                                OR proof_baseline_schedule_hash = ''
                              )
                            THEN 'missing proof_baseline_schedule_hash, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND proof_baseline_schedule_hash IS NOT NULL
                              AND proof_baseline_schedule_hash <> ''
                              AND proof_baseline_schedule_hash <> '{FIXED_SCHEDULE_GERMINATE_R0C0_HASH}'
                            THEN 'unexpected proof_baseline_schedule_hash, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND proof_baseline_schedule_version IS NULL
                            THEN 'missing proof_baseline_schedule_version, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND proof_baseline_schedule_version IS NOT NULL
                              AND proof_baseline_schedule_version <> {FIXED_SCHEDULE_GERMINATE_R0C0_VERSION}
                            THEN 'unexpected proof_baseline_schedule_version, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND proof_baseline_schedule_action_count IS NULL
                            THEN 'missing proof_baseline_schedule_action_count, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = 'fixed_schedule'
                              AND proof_baseline_schedule_action_count IS NOT NULL
                              AND proof_baseline_schedule_action_count <> {FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT}
                            THEN 'unexpected proof_baseline_schedule_action_count, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND (
                                proof_baseline_schedule_id IS NULL
                                OR proof_baseline_schedule_id = ''
                              )
                            THEN 'missing static-final source schedule_id, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND proof_baseline_schedule_id IS NOT NULL
                              AND proof_baseline_schedule_id <> ''
                              AND proof_baseline_schedule_id <> '{STATIC_FINAL_SOURCE_TOPOLOGY_V1}'
                            THEN 'unexpected static-final source schedule_id, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND (
                                proof_baseline_schedule_hash IS NULL
                                OR proof_baseline_schedule_hash = ''
                              )
                            THEN 'missing static-final source schedule_hash, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND proof_baseline_schedule_hash IS NOT NULL
                              AND proof_baseline_schedule_hash <> ''
                              AND proof_baseline_schedule_hash <> '{STATIC_FINAL_SOURCE_TOPOLOGY_HASH}'
                            THEN 'unexpected static-final source schedule_hash, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND proof_baseline_schedule_version IS NULL
                            THEN 'missing static-final source schedule_version, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND proof_baseline_schedule_version IS NOT NULL
                              AND proof_baseline_schedule_version <> {STATIC_FINAL_SOURCE_TOPOLOGY_VERSION}
                            THEN 'unexpected static-final source schedule_version, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND proof_baseline_schedule_action_count IS NULL
                            THEN 'missing static-final source schedule_action_count, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode = '{STATIC_FINAL_SOURCE_MODE}'
                              AND proof_baseline_schedule_action_count IS NOT NULL
                              AND proof_baseline_schedule_action_count <> {STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT}
                            THEN 'unexpected static-final source schedule_action_count, '
                            ELSE ''
                        END,
                        CASE
                            WHEN proof_baseline_mode NOT IN ('fixed_schedule', '{STATIC_FINAL_SOURCE_MODE}')
                              AND (
                                proof_baseline_schedule_id IS NOT NULL
                                OR proof_baseline_schedule_hash IS NOT NULL
                                OR proof_baseline_schedule_version IS NOT NULL
                                OR proof_baseline_schedule_action_count IS NOT NULL
                              )
                            THEN 'unexpected fixed-schedule provenance on non-schedule baseline, '
                            ELSE ''
                        END
                    ) AS violations
                FROM runs
                WHERE proof_baseline_mode IS NOT NULL
                  AND proof_baseline_mode <> ''
            )
            SELECT
                run_dir,
                group_id,
                proof_baseline_mode,
                proof_baseline_pair_id,
                proof_baseline_lifecycle_policy,
                proof_baseline_schedule_id,
                proof_baseline_schedule_hash,
                proof_baseline_schedule_version,
                proof_baseline_schedule_action_count,
                TRIM(TRAILING ', ' FROM violations) AS violations
            FROM checked
            WHERE violations <> ''
            ORDER BY run_dir, group_id
            """,
        )
        lifecycle_rows = _rows(
            conn,
            """
            WITH lifecycle AS (
                SELECT
                    run_dir,
                    group_id,
                    COUNT(*) FILTER (WHERE event_type = 'SEED_GERMINATED') AS germinated,
                    COUNT(*) FILTER (WHERE event_type = 'SEED_FOSSILIZED') AS fossilized,
                    COUNT(*) FILTER (WHERE event_type = 'SEED_PRUNED') AS pruned,
                    COUNT(*) FILTER (WHERE event_type = 'SEED_STAGE_CHANGED') AS stage_changes
                FROM seed_lifecycle
                GROUP BY run_dir, group_id
            )
            SELECT
                run_dir,
                group_id,
                germinated,
                fossilized,
                pruned,
                stage_changes,
                fossilized::DOUBLE / NULLIF(germinated, 0) AS fossilize_rate,
                pruned::DOUBLE / NULLIF(germinated, 0) AS prune_rate
            FROM lifecycle
            ORDER BY run_dir, group_id
            """,
        )
        blocking_confounders = _rows(
            conn,
            """
            SELECT run_dir, group_id, env_id, event_type, anomaly_type, episode, batch, detail
            FROM run_confounders
            WHERE proof_blocking
            ORDER BY timestamp, run_dir, group_id, env_id
            """,
        )
        missing_learnability = _rows(conn, _missing_learnability_query())
        invalid_outcomes = _rows(conn, _invalid_outcomes_query())
        outcome_count_rows = _rows(
            conn,
            """
            SELECT COUNT(*) AS outcome_count
            FROM episode_outcomes
            """,
        )
        roi_rows = _rows(
            conn,
            f"""
            WITH valid_episode_outcomes AS (
                SELECT *
                FROM episode_outcomes
                WHERE {VALID_OUTCOME_PREDICATE}
            )
            SELECT
                valid_episode_outcomes.run_dir,
                valid_episode_outcomes.group_id,
                valid_episode_outcomes.reward_mode,
                COUNT(*) AS episodes,
                AVG(final_accuracy) AS mean_final_accuracy,
                AVG(param_ratio) AS mean_param_ratio,
                AVG(final_accuracy / NULLIF(param_ratio, 0.0)) AS mean_accuracy_roi
            FROM valid_episode_outcomes
            LEFT JOIN runs
              ON runs.run_dir = valid_episode_outcomes.run_dir
             AND COALESCE(runs.group_id, '') = COALESCE(valid_episode_outcomes.group_id, '')
             AND runs.reward_mode = valid_episode_outcomes.reward_mode
            WHERE (
                runs.proof_baseline_mode IS NULL
                OR runs.proof_baseline_mode <> '{STATIC_FINAL_SOURCE_MODE}'
              )
            GROUP BY
                valid_episode_outcomes.run_dir,
                valid_episode_outcomes.group_id,
                valid_episode_outcomes.reward_mode
            ORDER BY
                valid_episode_outcomes.run_dir,
                valid_episode_outcomes.group_id,
                valid_episode_outcomes.reward_mode
            """,
        )
    finally:
        conn.close()

    observed_baseline_modes = tuple(
        row["proof_baseline_mode"] for row in baseline_evidence_rows
    )
    missing_baselines = (
        missing_required_baseline_modes(observed_baseline_modes)
        if resolved_require_blueprint_health_baselines
        else ()
    )

    # FAIL CLOSED gates: a proof packet must not assign a continue/revise/stop
    # verdict when the proof-critical telemetry is absent, corrupt, precision-
    # ambiguous, control-invalid, or mechanically unstable.
    outcome_count = int(outcome_count_rows[0]["outcome_count"])
    has_event_files = bool(runs) or outcome_count > 0 or not ingestion.is_clean
    missing_runs = not runs
    missing_outcomes = outcome_count == 0
    ingestion_blocking = not ingestion.is_clean
    if proof_profile == REWARD_EFFICIENCY_PROOF_PROFILE:
        roi_rows_for_verdict = _blueprint_economy_roi_rows(roi_rows)
        unsupported_economy_roi_rows = _unsupported_blueprint_economy_roi_rows(
            roi_rows
        )
    else:
        roi_rows_for_verdict = roi_rows
        unsupported_economy_roi_rows = []
    economy_evidence_blocking = (
        proof_profile == REWARD_EFFICIENCY_PROOF_PROFILE
        and bool(roi_rows)
        and not roi_rows_for_verdict
    )

    verdict = _classify_verdict(
        blocking_confounders=blocking_confounders,
        invalid_outcomes=invalid_outcomes,
        missing_learnability=missing_learnability,
        missing_baselines=missing_baselines,
        baseline_provenance_blockers=baseline_provenance_blockers,
        baseline_evidence_blockers=baseline_evidence_blockers,
        lockstep_baseline_blockers=lockstep_baseline_blockers,
        fixed_schedule_trace_blockers=fixed_schedule_trace_blockers,
        static_final_replay_blockers=static_final_replay_blockers,
        precision_blockers=precision_blockers,
        economy_evidence_blocking=economy_evidence_blocking,
        ingestion_blocking=ingestion_blocking,
        missing_runs=missing_runs,
        missing_outcomes=missing_outcomes,
        roi_rows=roi_rows_for_verdict,
        min_mean_accuracy_roi=min_mean_accuracy_roi,
    )
    lines = [
        "# Reward-Efficiency Proof Packet",
        "",
        f"- Telemetry dir: `{telemetry_dir}`",
        f"- Proof profile: `{proof_profile}`",
        f"- Verdict: `{verdict}`",
        f"- Minimum mean accuracy ROI for `CONTINUE`: `{min_mean_accuracy_roi}`",
        "",
        "## Ingestion Integrity",
        "",
    ]

    if not has_event_files:
        lines.append(
            "- BLOCKING empty telemetry: no `events.jsonl` files were found, so "
            "there is no evidence to support a proof verdict."
        )
    elif ingestion_blocking:
        lines.append(
            "- BLOCKING malformed telemetry: "
            f"{ingestion.malformed_count} JSONL line(s) failed to parse and were "
            "silently dropped by ingestion. The proof cannot be trusted while "
            "telemetry is corrupt."
        )
        for bad in ingestion.malformed_lines:
            lines.append(
                "  - "
                f"run `{bad.run_dir}` file `{bad.file}` line {bad.line_number}: "
                f"`{bad.snippet}`"
            )
    else:
        lines.append("- All telemetry JSONL lines parsed successfully.")

    lines.extend(["", "## Cohorts", ""])

    if runs:
        for row in runs:
            lines.append(
                    "- "
                    f"`{row['run_dir']}` group `{row['group_id']}`: "
                    f"task={row['task']}, reward_mode={row['reward_mode']}, "
                    f"seed={row['seed']}, "
                    f"envs={row['n_envs']}, env_episodes={row['n_episodes']}, "
                    f"episode_length={row['max_epochs']}"
                )
    else:
        lines.append(
            "- BLOCKING no `TRAINING_STARTED` events found: without run metadata "
            "the proof cannot identify cohorts or reproduce the experiment."
        )

    lines.extend(["", "## Precision Provenance", ""])
    if resolved_require_precision_provenance:
        if runs:
            for row in runs:
                lines.append(
                    "- "
                    f"`{row['run_dir']}` group `{row['group_id']}`: "
                    f"amp_enabled={row['amp_enabled']}, amp_dtype={row['amp_dtype']}"
                )
        else:
            lines.append("- No run precision metadata available because run metadata is missing.")
        if precision_blockers:
            for row in precision_blockers:
                lines.append(
                    "- BLOCKING precision provenance "
                    f"run `{row['run_dir']}` group `{row['group_id']}`: "
                    f"{row['violation']}"
                )
        else:
            lines.append("- All runs carry proof-grade precision provenance.")
    else:
        lines.append("- Precision provenance gate was not requested.")

    lines.extend(["", "## Reproduction Commands", ""])
    if runs:
        reward_modes = {row["reward_mode"] for row in runs}
        representative = runs[0]
        rounds = representative["max_batches"]
        if reward_modes == {"shaped", "simplified"}:
            command = (
                "PYTHONPATH=src uv run python -m esper.scripts.train ppo "
                f"--task {representative['task']} "
                "--dual-ab shaped-vs-simplified "
                f"--rounds {rounds} "
                f"--envs {representative['n_envs']} "
                f"--episode-length {representative['max_epochs']} "
                f"--telemetry-dir {telemetry_dir}"
            )
            lines.append(f"- Equivalent training command: `{command}`")
        else:
            for row in runs:
                command = (
                    "PYTHONPATH=src uv run python -m esper.scripts.train ppo "
                    f"--task {row['task']} "
                    f"--rounds {row['max_batches']} "
                    f"--envs {row['n_envs']} "
                    f"--episode-length {row['max_epochs']} "
                    f"--telemetry-dir {telemetry_dir}"
                )
                lines.append(
                    "- Equivalent training command "
                    f"for `{row['run_dir']}` group `{row['group_id']}`: `{command}`"
                )
    else:
        lines.append("- Training command could not be derived because run metadata is missing.")
    packet_command_parts = [
        "PYTHONPATH=src uv run python scripts/proof_packet.py",
        f"--telemetry-dir {telemetry_dir}",
        "--output <packet.md>",
        f"--proof-profile {proof_profile}",
    ]
    if (
        proof_profile == GENERIC_PROOF_PROFILE
        and resolved_require_blueprint_health_baselines
    ):
        packet_command_parts.append("--require-blueprint-health-baselines")
    if not resolved_require_precision_provenance:
        packet_command_parts.append("--no-require-precision-provenance")
    packet_command_parts.append(f"--min-mean-accuracy-roi {min_mean_accuracy_roi}")
    lines.append(
        "- Packet command: "
        f"`{' '.join(packet_command_parts)}`"
    )

    lines.extend(["", "## Confounder Ledger", ""])
    if blocking_confounders:
        for row in blocking_confounders:
            lines.append(
                "- BLOCKING "
                f"`{row['event_type']}` run `{row['run_dir']}` group `{row['group_id']}` "
                f"env={row['env_id']} episode={row['episode']} batch={row['batch']}: "
                f"{row['detail']}"
            )
    else:
        lines.append("- No proof-blocking anomaly events found.")

    lines.extend(["", "## Outcome Integrity", ""])
    if invalid_outcomes:
        for row in invalid_outcomes:
            lines.append(
                "- BLOCKING invalid `EPISODE_OUTCOME` "
                f"event `{row['event_id']}` run `{row['run_dir']}` "
                f"group `{row['group_id']}` env={row['env_id']} "
                f"episode={row['episode_idx']} reward={row['reward_mode']}: "
                f"{row['violations']}"
            )
    else:
        lines.append("- All episode outcome rows satisfy the proof ROI contract.")

    lines.extend(["", "## Reward-Mode Economy Evidence", ""])
    if proof_profile == REWARD_EFFICIENCY_PROOF_PROFILE:
        if unsupported_economy_roi_rows:
            for row in unsupported_economy_roi_rows:
                lines.append(
                    "- EXCLUDED "
                    f"`{row['run_dir']}` group `{row['group_id']}` "
                    f"reward_mode={row['reward_mode']}: "
                    f"{_reward_mode_evidence_reason(row['reward_mode'])} "
                    "reward mode is not blueprint-economy evidence and is "
                    "excluded from reward-efficiency verdict ROI."
                )
        else:
            lines.append(
                "- All outcome-bearing reward modes support blueprint-economy evidence."
            )
        if economy_evidence_blocking:
            lines.append(
                "- BLOCKING reward-efficiency evidence: valid outcome rows exist, "
                "but none use a reward mode that supports blueprint-economy evidence."
            )
        elif roi_rows_for_verdict:
            supported_modes = sorted(
                {row["reward_mode"] for row in roi_rows_for_verdict}
            )
            modes_text = ", ".join(f"`{mode}`" for mode in supported_modes)
            lines.append(
                "- Proof-grade economy reward modes included in verdict ROI: "
                f"{modes_text}."
            )
        else:
            lines.append(
                "- No valid ROI rows are available for reward-mode economy evidence "
                "classification."
            )
    else:
        lines.append(
            "- Reward-mode economy evidence gate applies only to "
            "`reward-efficiency` profile."
        )

    lines.extend(["", "## Learnability Gate", ""])
    if missing_learnability:
        for row in missing_learnability:
            lines.append(
                "- BLOCKING missing learnability telemetry "
                f"run `{row['run_dir']}` group `{row['group_id']}` "
                f"updates={row['missing_update_count']}"
            )
    else:
        lines.append("- PPO updates include per-head learnability telemetry.")

    lines.extend(["", "## Blueprint-Health Baselines", ""])
    if resolved_require_blueprint_health_baselines:
        if baseline_rows:
            for row in baseline_rows:
                schedule_text = ""
                if row["proof_baseline_schedule_id"] is not None:
                    schedule_text = (
                        f", schedule_id=`{row['proof_baseline_schedule_id']}`, "
                        f"schedule_hash=`{row['proof_baseline_schedule_hash']}`, "
                        f"schedule_version={row['proof_baseline_schedule_version']}, "
                        f"schedule_actions={row['proof_baseline_schedule_action_count']}"
                    )
                lines.append(
                    "- "
                    f"`{row['proof_baseline_mode']}`: runs={row['run_count']}, "
                    f"pair_id=`{row['proof_baseline_pair_id']}`, "
                    f"lifecycle_policy=`{row['proof_baseline_lifecycle_policy']}`"
                    f"{schedule_text}"
                )
        else:
            lines.append("- No proof baseline cohort identity found in run metadata.")
        if baseline_evidence_rows:
            for row in baseline_evidence_rows:
                schedule_text = ""
                if row["proof_baseline_schedule_id"] is not None:
                    schedule_text = (
                        f", schedule_id=`{row['proof_baseline_schedule_id']}`, "
                        f"schedule_hash=`{row['proof_baseline_schedule_hash']}`, "
                        f"schedule_version={row['proof_baseline_schedule_version']}, "
                        f"schedule_actions={row['proof_baseline_schedule_action_count']}"
                    )
                lines.append(
                    "- "
                    f"outcome evidence `{row['proof_baseline_mode']}`: "
                    f"runs={row['run_count']}, "
                    f"valid_outcomes={row['valid_outcome_count']}, "
                    f"pair_id=`{row['proof_baseline_pair_id']}`, "
                    f"lifecycle_policy=`{row['proof_baseline_lifecycle_policy']}`"
                    f"{schedule_text}"
                )
        else:
            lines.append("- No outcome-bearing proof baseline cohorts found.")
        if baseline_provenance_blockers:
            for row in baseline_provenance_blockers:
                lines.append(
                    "- BLOCKING incomplete proof baseline provenance "
                    f"run `{row['run_dir']}` group `{row['group_id']}` "
                    f"mode `{row['proof_baseline_mode']}`: "
                    f"{row['violations']}"
                )
        if baseline_evidence_blockers:
            for row in baseline_evidence_blockers:
                lines.append(
                    "- BLOCKING missing proof baseline outcome evidence "
                    f"run `{row['run_dir']}` group `{row['group_id']}` "
                    f"mode `{row['proof_baseline_mode']}` "
                    f"reward={row['reward_mode']}: {row['violations']}"
                )
        if lockstep_baseline_blockers:
            for row in lockstep_baseline_blockers:
                lines.append(
                    "- BLOCKING incomplete lockstep reward A/B baseline "
                    f"pair `{row['proof_baseline_pair_id']}`: "
                    f"{row['violations']} "
                    f"(runs={row['run_count']}, reward_modes={row['reward_modes']})"
                )
        if missing_baselines:
            for mode in missing_baselines:
                lines.append(f"- BLOCKING missing required proof baseline `{mode}`")
        elif (
            not baseline_provenance_blockers
            and not baseline_evidence_blockers
            and not lockstep_baseline_blockers
        ):
            lines.append(
                "- All required blueprint-health proof baselines have outcome-bearing evidence."
            )
    else:
        lines.append("- Blueprint-health baseline gate was not requested.")

    lines.extend(["", "## Fixed-Schedule Trace", ""])
    has_fixed_schedule_baseline = any(
        row["proof_baseline_mode"] == "fixed_schedule" for row in baseline_rows
    )
    if fixed_schedule_trace_blockers:
        for row in fixed_schedule_trace_blockers:
            observed_text = ""
            if row["observed_event_id"] is not None:
                observed_text = (
                    f"; observed event `{row['observed_event_id']}` "
                    f"phase=`{row['observed_phase']}` "
                    f"op=`{row['observed_operation']}` "
                    f"blueprint=`{row['observed_blueprint_id']}` "
                    f"governor_approved={row['observed_governor_approved']}"
                )
            lines.append(
                "- BLOCKING fixed-schedule trace "
                f"run `{row['run_dir']}` group `{row['group_id']}` "
                f"schedule `{row['proof_baseline_schedule_id']}`: "
                f"expected epoch={row['expected_epoch']} "
                f"slot=`{row['expected_slot_id']}` "
                f"op=`{row['expected_operation']}` "
                f"blueprint=`{row['expected_blueprint_id']}`"
                f"{observed_text}: {row['violations']}"
            )
    elif has_fixed_schedule_baseline:
        lines.append(
            "- Fixed-schedule committed morphology trace matches the declared schedule."
        )
    else:
        lines.append("- No fixed-schedule baseline cohorts found.")

    lines.extend(["", "## Static-Final Replay", ""])
    has_static_final_baseline = any(
        row["proof_baseline_mode"] == "static_final" for row in baseline_rows
    )
    if static_final_replay_blockers:
        for row in static_final_replay_blockers:
            lines.append(
                "- BLOCKING static-final replay "
                f"run `{row['run_dir']}` group `{row['group_id']}` "
                f"pair_id=`{row['proof_baseline_pair_id']}` "
                f"outcome_event=`{row['outcome_event_id']}` "
                f"env_id={row['target_env_id']} "
                f"episode_idx={row['target_episode_idx']} "
                f"lifecycle_policy=`{row['proof_baseline_lifecycle_policy']}`: "
                f"{row['violations']}; `freeze_replayed_final_topology` "
                "needs replayed topology identity before this control is proof-grade."
            )
    elif has_static_final_baseline:
        lines.append("- Static-final topology replay evidence is present.")
    else:
        lines.append("- No static-final baseline cohorts found.")

    lines.extend(["", "## Lifecycle Efficiency", ""])
    if lifecycle_rows:
        for row in lifecycle_rows:
            lines.append(
                "- "
                f"`{row['run_dir']}` group `{row['group_id']}`: "
                f"germinated={row['germinated']}, fossilized={row['fossilized']}, "
                f"pruned={row['pruned']}, stage_changes={row['stage_changes']}, "
                f"fossilize_rate={row['fossilize_rate']}, prune_rate={row['prune_rate']}"
            )
    else:
        lines.append("- No seed lifecycle events found.")

    lines.extend(["", "## Accuracy ROI", ""])
    if roi_rows:
        for row in roi_rows:
            lines.append(
                "- "
                f"`{row['run_dir']}` group `{row['group_id']}` reward={row['reward_mode']}: "
                f"episodes={row['episodes']}, mean_final_accuracy={row['mean_final_accuracy']}, "
                f"mean_param_ratio={row['mean_param_ratio']}, mean_accuracy_roi={row['mean_accuracy_roi']}"
            )
    elif invalid_outcomes:
        lines.append(
            "- BLOCKING no valid `EPISODE_OUTCOME` rows available for ROI aggregation: "
            "all observed outcome rows failed integrity checks."
        )
    else:
        lines.append(
            "- BLOCKING no `EPISODE_OUTCOME` events found: there are no reward / "
            "accuracy outcomes to compare, so no ROI verdict can be drawn."
        )

    lines.extend(["", "## Decision", ""])
    if verdict == BLOCKED_INSTRUMENTATION:
        lines.append(
            "The proof run cannot distinguish theory from implementation because required evidence is missing, malformed, or incomplete."
        )
    elif verdict == BLOCKED_PRECISION:
        lines.append(
            "The proof run cannot distinguish theory from precision artifacts because at least one run lacks proof-grade precision provenance."
        )
    elif verdict == BLOCKED_MATH:
        lines.append(
            "The proof run cannot distinguish theory from comparison math because outcome contracts or required controls are invalid."
        )
    elif verdict == BLOCKED_MECHANICS:
        lines.append(
            "The proof run cannot distinguish theory from training mechanics because value, gradient, rollback, reward-hacking, or degradation confounders are present."
        )
    elif verdict == CONTINUE:
        lines.append(
            "The clean run meets the configured ROI threshold. Continue scaling the experiment and preserve the same proof gates."
        )
    elif verdict == REVISE_ALGORITHM:
        lines.append(
            "The clean run has a measurable signal but does not meet the configured ROI threshold. Revise the algorithm before treating this as theory failure."
        )
    elif verdict == STOP_THEORY:
        lines.append(
            "The clean run shows no positive ROI signal under the configured proof gates. Stop or replace the underlying theory for this setup."
        )
    else:
        lines.append(
            f"Unhandled proof verdict `{verdict}`."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Esper proof packet markdown")
    parser.add_argument("--telemetry-dir", required=True, help="Telemetry root containing */events.jsonl")
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument(
        "--proof-profile",
        choices=PROOF_PROFILES,
        default=REWARD_EFFICIENCY_PROOF_PROFILE,
        help=(
            "Proof gate profile. Defaults to `reward-efficiency`, which automatically requires "
            "precision provenance and blueprint-health baseline controls."
        ),
    )
    parser.add_argument(
        "--require-blueprint-health-baselines",
        action="store_true",
        help="Block proof verdicts unless all blueprint-health baseline cohorts are present",
    )
    parser.add_argument(
        "--no-require-precision-provenance",
        action="store_true",
        help="Do not block proof verdicts on missing TRAINING_STARTED precision metadata",
    )
    parser.add_argument(
        "--min-mean-accuracy-roi",
        type=float,
        default=1.0,
        help="Minimum cohort mean accuracy ROI required for CONTINUE",
    )
    args = parser.parse_args()
    if (
        args.proof_profile == REWARD_EFFICIENCY_PROOF_PROFILE
        and args.no_require_precision_provenance
    ):
        parser.error(
            "--proof-profile reward-efficiency requires precision provenance; "
            "remove --no-require-precision-provenance"
        )

    packet = build_proof_packet(
        args.telemetry_dir,
        proof_profile=args.proof_profile,
        require_blueprint_health_baselines=(
            True if args.require_blueprint_health_baselines else None
        ),
        require_precision_provenance=(
            False if args.no_require_precision_provenance else None
        ),
        min_mean_accuracy_roi=args.min_mean_accuracy_roi,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet)


if __name__ == "__main__":
    main()
