#!/usr/bin/env python3
"""Generate a reward-efficiency proof packet from Karn telemetry views."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from esper.karn.mcp.views import create_views


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
)


def _rows(conn: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    result = conn.execute(query)
    columns = [col[0] for col in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


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


def build_proof_packet(telemetry_dir: str) -> str:
    """Build a markdown proof packet from telemetry."""
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
                n_envs,
                n_episodes,
                max_epochs,
                max_batches
            FROM runs
            ORDER BY started_at, run_dir, group_id
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
        roi_rows = _rows(
            conn,
            """
            SELECT
                run_dir,
                group_id,
                reward_mode,
                COUNT(*) AS episodes,
                AVG(final_accuracy) AS mean_final_accuracy,
                AVG(param_ratio) AS mean_param_ratio,
                AVG(final_accuracy / NULLIF(param_ratio, 0.0)) AS mean_accuracy_roi
            FROM episode_outcomes
            GROUP BY run_dir, group_id, reward_mode
            ORDER BY run_dir, group_id, reward_mode
            """,
        )
    finally:
        conn.close()

    verdict = "BLOCKED" if blocking_confounders or missing_learnability else "REVIEW"
    lines = [
        "# Reward-Efficiency Proof Packet",
        "",
        f"- Telemetry dir: `{telemetry_dir}`",
        f"- Verdict: `{verdict}`",
        "",
        "## Cohorts",
        "",
    ]

    if runs:
        for row in runs:
            lines.append(
                "- "
                f"`{row['run_dir']}` group `{row['group_id']}`: "
                f"task={row['task']}, reward_mode={row['reward_mode']}, "
                f"envs={row['n_envs']}, env_episodes={row['n_episodes']}, "
                f"episode_length={row['max_epochs']}"
            )
    else:
        lines.append("- No `TRAINING_STARTED` events found.")

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
    lines.append(
        "- Packet command: "
        f"`PYTHONPATH=src uv run python scripts/proof_packet.py --telemetry-dir {telemetry_dir} --output <packet.md>`"
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
    else:
        lines.append("- No `EPISODE_OUTCOME` events found.")

    lines.extend(["", "## Decision", ""])
    if verdict == "BLOCKED":
        lines.append(
            "The proof run cannot support a continue/revise/stop product verdict until blocking confounders are cleared."
        )
    else:
        lines.append(
            "No blocking telemetry confounders were found. Human review can assign continue/revise/stop from the ROI evidence."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Esper proof packet markdown")
    parser.add_argument("--telemetry-dir", required=True, help="Telemetry root containing */events.jsonl")
    parser.add_argument("--output", required=True, help="Markdown output path")
    args = parser.parse_args()

    packet = build_proof_packet(args.telemetry_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet)


if __name__ == "__main__":
    main()
