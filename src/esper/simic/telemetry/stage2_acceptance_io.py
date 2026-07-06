"""Stage-2 MAJOR-1 acceptance wrapper — telemetry I/O boundary (duckdb reader).

Reads Karn ``ppo_updates`` telemetry into the typed ``UpdateRow`` the pure layer
(``stage2_acceptance_packet``) consumes. This module is kept separate from the pure layer so the
statistics stay duckdb-free and synthetic-fixture-testable; it mirrors the ``scripts/proof_packet.py``
structure (``create_views`` + ``scan_ingestion_integrity`` + a ``_rows`` helper).

The dict -> ``UpdateRow`` conversion happens HERE and is the single place that reads named
telemetry columns. It FAILS LOUD on a missing column (schema drift) rather than masking it with a
default — the repo forbids ``.get()`` bug-hiding, and a silently-absent EV column would corrupt
every downstream statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from esper.simic.telemetry.stage2_acceptance_packet import UpdateRow

# The ppo_updates columns the wrapper reads (names verified against the Karn view schema).
# Ordering is chronological by (batch, inner_epoch): PPO collects a rollout (batch), then runs the
# inner optimisation epochs over it — burn-in (drop the first W) depends on this order.
_PPO_UPDATE_COLUMNS: tuple[str, ...] = (
    "inner_epoch",
    "batch",
    "explained_variance",
    "ev_sum",
    "ev_main",
    "ev_cf",
    "ev_return_variance",
    "pre_norm_advantage_std",
    "return_std",
)


def _rows(
    conn: duckdb.DuckDBPyConnection, query: str, params: list[Any] | None = None
) -> list[dict[str, Any]]:
    """Run ``query`` and return each result row as a column-name -> value dict (proof_packet idiom)."""
    result = conn.execute(query, params) if params is not None else conn.execute(query)
    columns = [col[0] for col in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def _require(row: dict[str, Any], key: str) -> Any:
    """A required column: fail loud if absent (schema drift) or NULL (telemetry gap)."""
    if key not in row:
        raise KeyError(f"ppo_updates row missing required column {key!r} (schema drift?)")
    value = row[key]
    if value is None:
        raise ValueError(f"ppo_updates column {key!r} is NULL where a value is required")
    return value


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    """A per-stream EV column: the column MUST exist (else schema drift), but a NULL value is a
    valid OFF leg (the ev_* columns are emitted only under hra_value_decomposition=True)."""
    if key not in row:
        raise KeyError(f"ppo_updates row missing column {key!r} (schema drift?)")
    value = row[key]
    return None if value is None else float(value)


def row_to_update(row: dict[str, Any]) -> UpdateRow:
    """Convert one ``ppo_updates`` row dict to a typed ``UpdateRow`` (fail-loud boundary)."""
    return UpdateRow(
        inner_epoch=int(_require(row, "inner_epoch")),
        batch=int(_require(row, "batch")),
        explained_variance=float(_require(row, "explained_variance")),
        ev_sum=_optional_float(row, "ev_sum"),
        ev_main=_optional_float(row, "ev_main"),
        ev_cf=_optional_float(row, "ev_cf"),
        ev_return_variance=float(_require(row, "ev_return_variance")),
        pre_norm_advantage_std=float(_require(row, "pre_norm_advantage_std")),
        return_std=float(_require(row, "return_std")),
    )


def read_leg(conn: duckdb.DuckDBPyConnection, run_dir: str) -> list[UpdateRow]:
    """Read one leg's ``ppo_updates`` series for ``run_dir``, chronologically ordered and typed.

    ``run_dir`` is passed as a bound parameter (never string-interpolated). An empty result is a
    §1 completeness failure surfaced here as a loud error rather than a silent empty leg.
    """
    query = (
        f"SELECT {', '.join(_PPO_UPDATE_COLUMNS)} FROM ppo_updates "
        "WHERE run_dir = ? ORDER BY batch, inner_epoch"
    )
    rows = _rows(conn, query, [run_dir])
    if not rows:
        raise ValueError(f"no ppo_updates rows for run_dir={run_dir!r}")
    return [row_to_update(row) for row in rows]


# --- episode_outcomes safety readers (§6 G1/G2/G3) ---
#
# DEFINITIONAL CHOICES (flagged for §6 / drl-expert confirmation — a wrong safety definition is the
# silent corruption the gate exists to catch):
#   G1 val-acc  := mean(final_accuracy) at the run's TERMINAL episode_idx, averaged over envs
#                  (its achieved validation accuracy at end of training; percent, 0-100).
#   G2 added-params := host_params * (param_ratio - 1) at the terminal episode. `fossilized_params`
#                  is NOT in the telemetry views (only the Sanctum aggregator), but param_ratio is
#                  defined as (host_params + fossilized_params)/host_params (leyline), so this is an
#                  EXACT derivation, not a proxy.
#   G3 churn    := mean germinate/prune/fossilize count per episode over the WHOLE run.
# G4 (guard channels) + the G3/G4 "materially elevated" thresholds are a separate decision, not read here.


@dataclass(frozen=True)
class ChurnRates:
    """Per-episode germinate/prune/fossilize means for one run (§6 G3 input)."""

    germinate: float
    prune: float
    fossilize: float


def _terminal_episode_scalar(conn: duckdb.DuckDBPyConnection, run_dir: str, expr: str) -> float:
    """Aggregate ``expr`` over the rows at the run's terminal (max) episode_idx; fail loud if none."""
    rows = _rows(
        conn,
        f"SELECT {expr} AS value FROM episode_outcomes WHERE run_dir = ? "
        "AND episode_idx = (SELECT MAX(episode_idx) FROM episode_outcomes WHERE run_dir = ?)",
        [run_dir, run_dir],
    )
    if not rows or rows[0]["value"] is None:
        raise ValueError(f"no episode_outcomes at the terminal episode for run_dir={run_dir!r}")
    return float(rows[0]["value"])


def read_run_val_acc(conn: duckdb.DuckDBPyConnection, run_dir: str) -> float:
    """§6 G1 — the run's validation accuracy: mean ``final_accuracy`` at the terminal episode (pp)."""
    return _terminal_episode_scalar(conn, run_dir, "AVG(final_accuracy)")


def read_run_added_params(
    conn: duckdb.DuckDBPyConnection, run_dir: str, host_params: int
) -> float:
    """§6 G2 — permanently-added params ``host_params * (param_ratio - 1)`` at the terminal episode."""
    param_ratio = _terminal_episode_scalar(conn, run_dir, "AVG(param_ratio)")
    return host_params * (param_ratio - 1.0)


def read_run_churn(conn: duckdb.DuckDBPyConnection, run_dir: str) -> ChurnRates:
    """§6 G3 — per-episode germinate/prune/fossilize means over the whole run."""
    rows = _rows(
        conn,
        "SELECT AVG(germinate_count) AS g, AVG(prune_count) AS p, AVG(fossilize_count) AS f "
        "FROM episode_outcomes WHERE run_dir = ?",
        [run_dir],
    )
    if not rows or rows[0]["g"] is None:
        raise ValueError(f"no episode_outcomes churn rows for run_dir={run_dir!r}")
    return ChurnRates(
        germinate=float(rows[0]["g"]), prune=float(rows[0]["p"]), fossilize=float(rows[0]["f"])
    )
