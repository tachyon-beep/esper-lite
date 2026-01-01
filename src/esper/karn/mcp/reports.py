"""Canned analytic reports for Karn MCP telemetry server."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

import duckdb

from esper.karn.mcp.query import rows_to_records
from esper.karn.sanctum.schema import compute_collapse_risk, compute_entropy_velocity


def _execute(
    conn: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any] | None = None
) -> tuple[list[str], list[tuple[Any, ...]]]:
    result = conn.execute(sql, params) if params is not None else conn.execute(sql)
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    return columns, rows


def _one_record(
    conn: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any] | None = None
) -> dict[str, Any] | None:
    columns, rows = _execute(conn, sql, params)
    if not rows:
        return None
    return rows_to_records(columns, [rows[0]])[0]


def _many_records(
    conn: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any] | None = None
) -> list[dict[str, Any]]:
    columns, rows = _execute(conn, sql, params)
    return rows_to_records(columns, rows)


def _resolve_run_dir(
    conn: duckdb.DuckDBPyConnection, run_dir: str | None, group_id: str | None
) -> str | None:
    if run_dir is not None:
        return run_dir

    if group_id is None:
        row = conn.execute(
            "SELECT run_dir FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT run_dir FROM runs WHERE group_id = ? ORDER BY started_at DESC LIMIT 1",
            [group_id],
        ).fetchone()
    if row is not None:
        return str(row[0])

    row = conn.execute(
        "SELECT run_dir FROM raw_events WHERE run_dir IS NOT NULL ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def build_run_overview(
    conn: duckdb.DuckDBPyConnection,
    run_dir: str | None = None,
    group_id: str | None = None,
    recent_limit: int = 20,
) -> dict[str, Any]:
    resolved_run_dir = _resolve_run_dir(conn, run_dir, group_id)
    if resolved_run_dir is None:
        return {
            "ok": False,
            "error": "No telemetry runs found (no event files or no TRAINING_STARTED events).",
        }

    filters: list[str] = ["run_dir = ?"]
    params: list[Any] = [resolved_run_dir]
    if group_id is not None:
        filters.append("group_id = ?")
        params.append(group_id)
    where_sql = " AND ".join(filters)

    run = _one_record(
        conn, f"SELECT * FROM runs WHERE {where_sql} ORDER BY started_at DESC LIMIT 1", params
    )
    latest_batch_epoch = _one_record(
        conn,
        f"SELECT * FROM batch_epochs WHERE {where_sql} ORDER BY timestamp DESC LIMIT 1",
        params,
    )
    latest_batch_stats = _one_record(
        conn,
        f"SELECT * FROM batch_stats WHERE {where_sql} ORDER BY timestamp DESC LIMIT 1",
        params,
    )
    latest_ppo_update = _one_record(
        conn,
        f"SELECT * FROM ppo_updates WHERE {where_sql} ORDER BY timestamp DESC LIMIT 1",
        params,
    )

    env_latest = _many_records(
        conn,
        f"""
        SELECT
            env_id,
            inner_epoch,
            epoch,
            val_accuracy,
            val_loss,
            train_accuracy,
            train_loss,
            timestamp
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY env_id ORDER BY timestamp DESC) AS rn
            FROM epochs
            WHERE {where_sql}
        ) t
        WHERE rn = 1
        ORDER BY env_id
        """,
        params,
    )

    val_accuracies = [
        row["val_accuracy"] for row in env_latest if row["val_accuracy"] is not None
    ]
    mean_val_accuracy = (
        sum(val_accuracies) / len(val_accuracies) if val_accuracies else None
    )

    entropy_rows = conn.execute(
        f"SELECT entropy FROM ppo_updates WHERE {where_sql} ORDER BY timestamp DESC LIMIT 10",
        params,
    ).fetchall()
    entropy_history = [
        float(entropy)
        for (entropy,) in reversed(entropy_rows)
        if entropy is not None
    ]

    policy_health = {
        "entropy_history": entropy_history,
        "entropy_velocity": compute_entropy_velocity(entropy_history),
        "collapse_risk": compute_collapse_risk(entropy_history),
    }

    seed_event_counts = _many_records(
        conn,
        f"""
        SELECT event_type, COUNT(*)::INTEGER AS count
        FROM seed_lifecycle
        WHERE {where_sql}
        GROUP BY event_type
        ORDER BY count DESC
        """,
        params,
    )
    blueprint_lifecycle = _many_records(
        conn,
        f"""
        SELECT
            blueprint_id,
            SUM(CASE WHEN event_type = 'SEED_GERMINATED' THEN 1 ELSE 0 END)::INTEGER AS germinated,
            SUM(CASE WHEN event_type = 'SEED_FOSSILIZED' THEN 1 ELSE 0 END)::INTEGER AS fossilized,
            SUM(CASE WHEN event_type = 'SEED_PRUNED' THEN 1 ELSE 0 END)::INTEGER AS pruned
        FROM seed_lifecycle
        WHERE {where_sql} AND blueprint_id IS NOT NULL
        GROUP BY blueprint_id
        ORDER BY fossilized DESC, germinated DESC
        LIMIT 20
        """,
        params,
    )

    action_mix = _many_records(
        conn,
        f"""
        SELECT
            action_name,
            COUNT(*)::INTEGER AS count,
            AVG(CASE WHEN action_success THEN 1 ELSE 0 END)::DOUBLE AS success_rate,
            AVG(total_reward)::DOUBLE AS avg_total_reward
        FROM decisions
        WHERE {where_sql}
        GROUP BY action_name
        ORDER BY count DESC
        LIMIT 20
        """,
        params,
    )
    recent_decisions = _many_records(
        conn,
        f"""
        SELECT
            timestamp,
            env_id,
            inner_epoch,
            action_name,
            action_success,
            total_reward,
            value_estimate,
            action_confidence,
            decision_entropy
        FROM decisions
        WHERE {where_sql}
        ORDER BY timestamp DESC
        LIMIT {recent_limit}
        """,
        params,
    )

    reward_summary = _one_record(
        conn,
        f"""
        SELECT
            AVG(total_reward)::DOUBLE AS avg_total_reward,
            AVG(base_acc_delta)::DOUBLE AS avg_base_acc_delta,
            AVG(seed_contribution)::DOUBLE AS avg_seed_contribution,
            AVG(compute_rent)::DOUBLE AS avg_compute_rent,
            AVG(stage_bonus)::DOUBLE AS avg_stage_bonus,
            AVG(pbrs_bonus)::DOUBLE AS avg_pbrs_bonus,
            AVG(blending_warning)::DOUBLE AS avg_blending_warning,
            AVG(holding_warning)::DOUBLE AS avg_holding_warning,
            AVG(alpha_shock)::DOUBLE AS avg_alpha_shock,
            AVG(shaped_reward_ratio)::DOUBLE AS avg_shaped_reward_ratio
        FROM (
            SELECT *
            FROM rewards
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT 100
        ) r
        """,
        params,
    )

    recent_anomalies = _many_records(
        conn,
        f"""
        SELECT timestamp, event_type, message
        FROM anomalies
        WHERE {where_sql}
        ORDER BY timestamp DESC
        LIMIT {recent_limit}
        """,
        params,
    )
    recent_trends = _many_records(
        conn,
        f"""
        SELECT
            timestamp,
            event_type,
            batch_idx,
            episodes_completed,
            rolling_delta,
            rolling_avg_accuracy,
            prev_rolling_avg_accuracy
        FROM trends
        WHERE {where_sql}
        ORDER BY timestamp DESC
        LIMIT {recent_limit}
        """,
        params,
    )

    recent_episode_outcomes = _many_records(
        conn,
        f"""
        SELECT
            timestamp,
            env_id,
            episode_idx,
            final_accuracy,
            param_ratio,
            num_fossilized,
            num_contributing_fossilized,
            episode_reward,
            stability_score,
            reward_mode
        FROM episode_outcomes
        WHERE {where_sql}
        ORDER BY timestamp DESC
        LIMIT {recent_limit}
        """,
        params,
    )

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": {"run_dir": resolved_run_dir, "group_id": group_id},
        "run": run,
        "progress": {
            "latest_batch_epoch": latest_batch_epoch,
            "latest_batch_stats": latest_batch_stats,
        },
        "envs": {"latest": env_latest, "mean_val_accuracy": mean_val_accuracy},
        "policy": {"latest_ppo_update": latest_ppo_update, "health": policy_health},
        "seeds": {
            "event_counts": seed_event_counts,
            "blueprints": blueprint_lifecycle,
        },
        "decisions": {"action_mix": action_mix, "recent": recent_decisions},
        "rewards": {"summary": reward_summary},
        "anomalies": recent_anomalies,
        "trends": recent_trends,
        "episode_outcomes": recent_episode_outcomes,
    }
