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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import duckdb

from esper.simic.telemetry.stage2_acceptance_packet import (
    ChurnRates,
    FrozenConfigValue,
    FrozenThresholds,
    GuardChannelCounts,
    Leg,
    RunMeta,
    SafetyEvidence,
    SeedPair,
    Stage2Report,
    UpdateRow,
    Validity,
    calibrate_off,
    g3_hold_from_churn,
    g4_hold_from_counts,
    leg_series,
    render_packet,
    require_hra_signature,
    score,
    validate_pair,
)

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
    "value_main_target_scale",
    "cf_value_target_scale",
    "ev_return_variance",
    "pre_norm_advantage_std",
    "return_std",
    "gradient_cv",
)

_FROZEN_RUN_CONFIG_COLUMNS: tuple[str, ...] = (
    "task",
    "reward_family",
    "n_envs",
    "n_episodes",
    "max_epochs",
    "max_batches",
    "lr",
    "gamma",
    "gae_lambda",
    "ppo_updates_per_batch",
    "recurrent_n_epochs",
    "clip_ratio",
    "entropy_coef",
    "per_head_advantage_norm",
    "return_variance_telemetry",
    "value_coef",
    "value_warmup_batches",
    "value_coef_start",
    "param_budget",
    "param_penalty_weight",
    "sparse_reward_scale",
    "rent_host_params_floor",
    "basic_acc_delta_weight",
    "plateau_threshold",
    "improvement_threshold",
    "gradient_telemetry_stride",
    "lstm_hidden_dim",
    "chunk_length",
    "max_seeds",
    "slot_ids_json",
    "env_devices_json",
    "policy_device",
    "amp_enabled",
    "amp_dtype",
    "compile_enabled",
    "compile_backend",
    "compile_mode",
    "permissive_gates",
    "auto_forward_g1",
    "auto_forward_g2",
    "auto_forward_g3",
    "disable_pbrs",
    "disable_terminal_reward",
    "disable_anti_gaming",
    "max_grad_norm",
    "host_params",
)

_NULLABLE_FROZEN_RUN_CONFIG_COLUMNS: frozenset[str] = frozenset({
    "value_coef_start",
    "max_seeds",
    "amp_dtype",
    "compile_backend",
    "compile_mode",
    "max_grad_norm",
})


def _rows(
    conn: duckdb.DuckDBPyConnection, query: str, params: list[Any] | None = None
) -> list[dict[str, Any]]:
    """Run ``query`` and return each result row as a column-name -> value dict (proof_packet idiom)."""
    result = conn.execute(query, params) if params is not None else conn.execute(query)
    columns = [col[0] for col in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def _exactly_one_row(rows: list[dict[str, Any]], *, source: str, run_dir: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no {source} row for run_dir={run_dir!r}")
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one {source} row for run_dir={run_dir!r}, found {len(rows)} "
            "— duplicate run metadata would make Stage-2 evidence ambiguous"
        )
    return rows[0]


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
        value_main_target_scale=_optional_float(row, "value_main_target_scale"),
        cf_value_target_scale=_optional_float(row, "cf_value_target_scale"),
        ev_return_variance=float(_require(row, "ev_return_variance")),
        pre_norm_advantage_std=float(_require(row, "pre_norm_advantage_std")),
        return_std=float(_require(row, "return_std")),
        gradient_cv=float(_require(row, "gradient_cv")),
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
# DEFINITIONAL CHOICES (§6; owner-ratified where noted; a wrong safety definition is the silent
# corruption the gate exists to catch):
#   G1 val-acc  := mean over EACH env's OWN terminal episode of final_accuracy (percent, 0-100).
#                  `episode_idx` is a DENSE GLOBAL counter (episodes_completed + env_idx), so a plain
#                  MAX(episode_idx) matches ONE env — we take each env's terminal via a per-env window
#                  and average, so the per-seed estimate is an env-average not a single-env blip
#                  (drl-expert Finding 1).
#   G2 added-params := host_params * (per-env terminal param_ratio - 1) = total_params - host_params.
#                  This is the TOTAL added-parameter footprint (fossilized + transient active seeds),
#                  NOT strictly fossilized "permanent" params. `param_ratio` is authoritatively
#                  total_params/host_params (leyline/episode_outcome.py, telemetry.py) — the direct
#                  `fossilized_params` is NOT in the telemetry views (only the Sanctum aggregator), so
#                  this is a conservative PROXY, owner-ratified as "terminal added-parameter footprint".
#   G3 churn    := mean germinate/prune/fossilize count per episode over the WHOLE run.
# G4 (guard channels) + the G3/G4 "materially elevated" thresholds are a separate decision, not read here.


def _per_env_terminal_mean(
    conn: duckdb.DuckDBPyConnection, run_dir: str, column: str
) -> float:
    """Mean of ``column`` over EACH env's terminal (max-episode_idx) row; fail loud if none.

    ``episode_idx`` is a dense global counter, so envs terminate at different indices — a per-env
    window (``ROW_NUMBER() OVER (PARTITION BY env_id ORDER BY episode_idx DESC)``) selects each env's
    own last episode, then we average across envs.
    """
    rows = _rows(
        conn,
        f"""
        WITH per_env AS (
            SELECT {column} AS value,
                   ROW_NUMBER() OVER (PARTITION BY env_id ORDER BY episode_idx DESC) AS rn
            FROM episode_outcomes WHERE run_dir = ?
        )
        SELECT COUNT(*) AS terminal_envs, COUNT(value) AS present_values, AVG(value) AS value
        FROM per_env WHERE rn = 1
        """,
        [run_dir],
    )
    if not rows or rows[0]["terminal_envs"] == 0:
        raise ValueError(f"no episode_outcomes at the terminal episode for run_dir={run_dir!r}")
    if rows[0]["terminal_envs"] != rows[0]["present_values"]:
        raise ValueError(
            f"terminal episode_outcomes.{column} is NULL for "
            f"{rows[0]['terminal_envs'] - rows[0]['present_values']} env(s) in run_dir={run_dir!r}"
        )
    return float(rows[0]["value"])


def read_run_val_acc(conn: duckdb.DuckDBPyConnection, run_dir: str) -> float:
    """§6 G1 — the run's validation accuracy: mean over each env's terminal ``final_accuracy`` (pp)."""
    return _per_env_terminal_mean(conn, run_dir, "final_accuracy")


def read_run_added_params(
    conn: duckdb.DuckDBPyConnection, run_dir: str, host_params: int
) -> float:
    """§6 G2 — terminal added-parameter footprint ``host_params * (param_ratio - 1)``.

    ``param_ratio`` = total_params/host_params (leyline), so this is total added params (fossilized +
    transient), a conservative proxy for §6's fossilized-only quantity (owner-ratified).
    """
    param_ratio = _per_env_terminal_mean(conn, run_dir, "param_ratio")
    return host_params * (param_ratio - 1.0)


def read_run_uses_per_head_norm(conn: duckdb.DuckDBPyConnection, run_dir: str) -> bool:
    """§8F(i) — True if per-head advantage normalization was on for ANY update in the run.

    ``per_head_advantage_norm`` must be False on both legs (§10 frozen); it changes
    ``pre_norm_advantage_std`` semantics and would contaminate the LEG-B comparison. ``BOOL_OR``
    catches a run that toggled it on at any point, not just at the end.
    """
    rows = _rows(
        conn,
        "SELECT BOOL_OR(advantage_per_head_normalized) AS value FROM ppo_updates WHERE run_dir = ?",
        [run_dir],
    )
    if not rows or rows[0]["value"] is None:
        raise ValueError(
            f"no ppo_updates advantage_per_head_normalized rows for run_dir={run_dir!r}"
        )
    return bool(rows[0]["value"])


def read_run_meta(conn: duckdb.DuckDBPyConnection, run_dir: str) -> RunMeta:
    """§0/§1 — the run's provenance/config from the ``runs`` view, plus the §8F(i) scan.

    ``actor_advantage_source`` NULL means pre-S7 telemetry: the §0 scope fence cannot verify
    provenance it does not have — fail loud, never default (the fence would otherwise pass
    on trust, the exact pattern PDR-0039 rejected).
    """
    rows = _rows(
        conn,
        "SELECT seed, reward_mode, actor_advantage_source, resume_path, start_episode, "
        + ", ".join(_FROZEN_RUN_CONFIG_COLUMNS)
        + " FROM runs WHERE run_dir = ?",
        [run_dir],
    )
    row = _exactly_one_row(rows, source="runs", run_dir=run_dir)
    required_columns = (
        "seed",
        "reward_mode",
        "actor_advantage_source",
        "resume_path",
        "start_episode",
        *_FROZEN_RUN_CONFIG_COLUMNS,
    )
    for column in required_columns:
        if row[column] is None and column not in _NULLABLE_FROZEN_RUN_CONFIG_COLUMNS:
            raise ValueError(
                f"runs.{column} is NULL for run_dir={run_dir!r} — "
                + (
                    "pre-S7 telemetry lacks the §0 provenance field; this run cannot be "
                    "scored under the Stage-2 gate"
                    if column == "actor_advantage_source"
                    else "a required provenance field is missing"
                )
            )
    if str(row["resume_path"]) != "":
        raise ValueError(
            f"run_dir={run_dir!r} is not fresh-init telemetry: resume_path={row['resume_path']!r}"
        )
    if int(row["start_episode"]) != 0:
        raise ValueError(
            f"run_dir={run_dir!r} is not fresh-init telemetry: start_episode={row['start_episode']!r}"
        )
    frozen_config = tuple((column, row[column]) for column in _FROZEN_RUN_CONFIG_COLUMNS)
    return RunMeta(
        run_dir=run_dir,
        seed=int(row["seed"]),
        reward_mode=str(row["reward_mode"]),
        actor_advantage_source=str(row["actor_advantage_source"]),
        uses_per_head_norm=read_run_uses_per_head_norm(conn, run_dir),
        frozen_config=frozen_config,
    )


# The §6 G4 guard channels, keyed by the anomalies-view event_type that feeds each count.
# PLATEAU_DETECTED is present in the view but deliberately NOT a guard channel (benign
# training-progress signal).
_GUARD_CHANNEL_EVENT_TYPES: dict[str, str] = {
    "GOVERNOR_ROLLBACK": "governor_rollback",
    "VALUE_COLLAPSE_DETECTED": "value_collapse",
    "RATIO_EXPLOSION_DETECTED": "ratio_explosion",
    "RATIO_COLLAPSE_DETECTED": "ratio_collapse",
    "GRADIENT_ANOMALY": "gradient_anomaly",
    "GRADIENT_PATHOLOGY_DETECTED": "gradient_pathology",
    "NUMERICAL_INSTABILITY_DETECTED": "numerical_instability",
    "REWARD_HACKING_SUSPECTED": "reward_hacking",
}


def read_run_guard_channels(
    conn: duckdb.DuckDBPyConnection, run_dir: str
) -> GuardChannelCounts:
    """§6 G4 — per-channel anomaly counts for one run from the ``anomalies`` view.

    Zero rows is a valid clean run (COUNT semantics), not an error — unlike the meta/safety
    readers, absence of anomalies is the healthy state.
    """
    rows = _rows(
        conn,
        "SELECT event_type, COUNT(*) AS c FROM anomalies WHERE run_dir = ? GROUP BY event_type",
        [run_dir],
    )
    counts = dict.fromkeys(_GUARD_CHANNEL_EVENT_TYPES.values(), 0)
    for row in rows:
        event_type = str(row["event_type"])
        if event_type in _GUARD_CHANNEL_EVENT_TYPES:
            field_name = _GUARD_CHANNEL_EVENT_TYPES[event_type]
            counts[field_name] = int(row["c"])
    return GuardChannelCounts(**counts)


def read_run_churn(conn: duckdb.DuckDBPyConnection, run_dir: str) -> ChurnRates:
    """§6 G3 — per-episode germinate/prune/fossilize means over the whole run."""
    rows = _rows(
        conn,
        "SELECT COUNT(*) AS n, COUNT(germinate_count) AS n_g, COUNT(prune_count) AS n_p, "
        "COUNT(fossilize_count) AS n_f, AVG(germinate_count) AS g, AVG(prune_count) AS p, "
        "AVG(fossilize_count) AS f "
        "FROM episode_outcomes WHERE run_dir = ?",
        [run_dir],
    )
    if not rows or rows[0]["n"] == 0:
        raise ValueError(f"no episode_outcomes churn rows for run_dir={run_dir!r}")
    row = rows[0]
    if row["n"] != row["n_g"] or row["n"] != row["n_p"] or row["n"] != row["n_f"]:
        raise ValueError(
            f"episode_outcomes churn fields contain NULL values for run_dir={run_dir!r}"
        )
    return ChurnRates(
        germinate=float(row["g"]), prune=float(row["p"]), fossilize=float(row["f"])
    )


# --- §6 env-count-completeness validity (backstops Finding 1's per-env-terminal aggregation) ---
#
# The per-env-terminal G1/G2 mean silently averages each env's latest available row. If a leg's
# episode_outcomes is missing an env entirely, or if an env stopped before its terminal episode,
# the paired safety delta is biased over stale/subset evidence. So the pair is rejected unless
# each leg covers all runs.n_envs distinct env_ids at their expected terminal episode_idx AND ON
# n_envs == OFF n_envs. DEFINITIONAL: hard-reject-on-incomplete is the conservative stance.


def read_run_n_envs(conn: duckdb.DuckDBPyConnection, run_dir: str) -> int:
    """The run's configured env count from the ``runs`` view (§6 completeness denominator)."""
    n_envs, _n_episodes = _read_run_episode_shape(conn, run_dir)
    return n_envs


def _read_run_episode_shape(conn: duckdb.DuckDBPyConnection, run_dir: str) -> tuple[int, int]:
    rows = _rows(conn, "SELECT n_envs, n_episodes FROM runs WHERE run_dir = ?", [run_dir])
    row = _exactly_one_row(rows, source="runs", run_dir=run_dir)
    if row["n_envs"] is None:
        raise ValueError(f"runs.n_envs is NULL for run_dir={run_dir!r}")
    if row["n_episodes"] is None:
        raise ValueError(f"runs.n_episodes is NULL for run_dir={run_dir!r}")
    n_envs = int(row["n_envs"])
    n_episodes = int(row["n_episodes"])
    if n_envs <= 0:
        raise ValueError(f"runs.n_envs must be positive for run_dir={run_dir!r}, got {n_envs}")
    if n_episodes <= 0:
        raise ValueError(
            f"runs.n_episodes must be positive for run_dir={run_dir!r}, got {n_episodes}"
        )
    if n_episodes % n_envs != 0:
        raise ValueError(
            f"runs.n_episodes={n_episodes} is not divisible by n_envs={n_envs} for "
            f"run_dir={run_dir!r}; terminal env coverage is ambiguous"
        )
    return n_envs, n_episodes


def _episode_outcome_env_rows(conn: duckdb.DuckDBPyConnection, run_dir: str) -> list[dict[str, Any]]:
    """Per-env outcome coverage for this run: row count and latest episode index."""
    rows = _rows(
        conn,
        "SELECT env_id, COUNT(*) AS n_rows, MAX(episode_idx) AS max_episode_idx "
        "FROM episode_outcomes WHERE run_dir = ? GROUP BY env_id ORDER BY env_id",
        [run_dir],
    )
    return rows


def _terminal_env_coverage_reasons(
    conn: duckdb.DuckDBPyConnection, *, run_dir: str, arm: str
) -> list[str]:
    n_envs, n_episodes = _read_run_episode_shape(conn, run_dir)
    rows = _episode_outcome_env_rows(conn, run_dir)
    expected_env_ids = set(range(n_envs))
    present_env_ids = {int(row["env_id"]) for row in rows}
    reasons: list[str] = []

    missing = sorted(expected_env_ids - present_env_ids)
    if missing:
        reasons.append(
            f"{arm} arm episode_outcomes cover {len(present_env_ids)}/{n_envs} envs — "
            f"missing env_id(s): {', '.join(str(env_id) for env_id in missing)}; the "
            "per-env-terminal G1/G2 mean would be biased over a subset (§6)"
        )

    unexpected = sorted(present_env_ids - expected_env_ids)
    if unexpected:
        reasons.append(
            f"{arm} arm episode_outcomes include unexpected env_id(s): "
            f"{', '.join(str(env_id) for env_id in unexpected)}; expected 0..{n_envs - 1} (§6)"
        )

    stale: list[str] = []
    for row in rows:
        env_id = int(row["env_id"])
        if env_id not in expected_env_ids:
            continue
        expected_terminal_idx = n_episodes - n_envs + env_id
        actual_terminal_idx = int(row["max_episode_idx"])
        if actual_terminal_idx != expected_terminal_idx:
            stale.append(
                f"env {env_id} max episode_idx {actual_terminal_idx} != "
                f"expected terminal {expected_terminal_idx}"
            )
    if stale:
        reasons.append(
            f"{arm} arm episode_outcomes do not cover terminal evidence for every env: "
            + "; ".join(stale)
            + " (§6)"
        )
    return reasons


def env_count_completeness_reasons(
    conn: duckdb.DuckDBPyConnection, *, on_run_dir: str, off_run_dir: str
) -> list[str]:
    """§6 — reasons the pair fails env-count completeness (empty iff both legs are complete + matched).

    Merged into the ``Validity`` passed to ``score`` at the assembly layer (``build_report``), since
    it reads ``runs``/``episode_outcomes`` — telemetry the pure ``validate_pair`` never sees.
    """
    reasons: list[str] = []
    on_expected = read_run_n_envs(conn, on_run_dir)
    off_expected = read_run_n_envs(conn, off_run_dir)
    if on_expected != off_expected:
        reasons.append(
            f"env-count mismatch: ON n_envs={on_expected} != OFF n_envs={off_expected} "
            "(the arms must run the same env count, §6)"
        )
    reasons.extend(_terminal_env_coverage_reasons(conn, run_dir=on_run_dir, arm="ON"))
    reasons.extend(_terminal_env_coverage_reasons(conn, run_dir=off_run_dir, arm="OFF"))
    return reasons


# --- S6: scoring-phase assembly (telemetry -> scored verdict) -----------------------------------


@dataclass(frozen=True)
class SeedPairing:
    """One fresh-init seed's ON/OFF run metadata + host size (the assembler's per-seed input).

    ``on_meta``/``off_meta`` are caller-supplied (option (a)): the wrapper reads no
    ``actor_advantage_source`` column from telemetry — the runs-view ``read_run_meta`` that would is
    bundled with the S7 emission that CREATES that column. ``host_params`` scales the §6 G2
    added-parameter footprint.
    """

    on_meta: RunMeta
    off_meta: RunMeta
    host_params: int


@dataclass(frozen=True)
class BuiltReport:
    """The assembler's output: the scored verdict PLUS the ``Validity`` it computed.

    The ``Validity`` must travel with the report — on an INVALID run the breach list lives ONLY here
    (``Stage2Report`` carries no reasons), so ``render_packet`` needs it to list every §1 breach. A
    caller does ``built = build_report(...); render_packet(built.report, validity=built.validity)``.
    """

    report: Stage2Report
    validity: Validity


def _duplicate_items(values: Sequence[str | int]) -> tuple[str, ...]:
    seen: set[str | int] = set()
    duplicates: set[str | int] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return tuple(str(value) for value in sorted(duplicates, key=str))


def pairset_validity_reasons(pairings: list[SeedPairing]) -> list[str]:
    """§1 pairset-level validity: uniqueness and cross-seed frozen-config homogeneity."""
    reasons: list[str] = []
    seeds = [pairing.on_meta.seed for pairing in pairings]
    duplicate_seeds = _duplicate_items(seeds)
    if duplicate_seeds:
        reasons.append(
            "duplicate seed(s) in Stage-2 pairset: "
            f"{', '.join(duplicate_seeds)} — each seed may contribute one fresh-init pair (§10)"
        )

    on_dirs = [pairing.on_meta.run_dir for pairing in pairings]
    off_dirs = [pairing.off_meta.run_dir for pairing in pairings]
    duplicate_run_dirs = _duplicate_items(on_dirs + off_dirs)
    if duplicate_run_dirs:
        reasons.append(
            "duplicate run_dir(s) in Stage-2 pairset: "
            f"{', '.join(duplicate_run_dirs)} — evidence rows must be unique (§10)"
        )
    overlap = sorted(set(on_dirs).intersection(off_dirs))
    if overlap:
        reasons.append(
            "run_dir(s) appear in both ON and OFF arms: "
            f"{', '.join(overlap)} — arms must be disjoint fresh runs (§10)"
        )

    metas = [meta for pairing in pairings for meta in (pairing.on_meta, pairing.off_meta)]
    if metas:
        reward_mode = metas[0].reward_mode
        if any(meta.reward_mode != reward_mode for meta in metas):
            reasons.append(
                "global reward_mode mismatch across Stage-2 pairset: every run must share "
                "the same frozen reward_mode (§10)"
            )
        frozen_config = metas[0].frozen_config
        if any(meta.frozen_config != frozen_config for meta in metas):
            reasons.append(
                "global frozen run config mismatch across Stage-2 pairset: each pair may match "
                "internally, but the scored seed set must be one homogeneous experiment (§10)"
            )
    return reasons


def _frozen_config_value(meta: RunMeta, field: str) -> FrozenConfigValue:
    for key, value in meta.frozen_config:
        if key == field:
            return value
    raise ValueError(
        f"runs.{field} is missing from frozen_config for run_dir={meta.run_dir!r}; "
        "Stage-2 cannot score this run"
    )


def _run_host_params(meta: RunMeta) -> int:
    value = _frozen_config_value(meta, "host_params")
    if value is None:
        raise ValueError(f"runs.host_params is NULL for run_dir={meta.run_dir!r}")
    host_params = int(value)
    if host_params <= 0:
        raise ValueError(
            f"runs.host_params must be positive for run_dir={meta.run_dir!r}, got {host_params}"
        )
    return host_params


def _host_params_for_pairing(pairing: SeedPairing) -> int:
    on_host_params = _run_host_params(pairing.on_meta)
    off_host_params = _run_host_params(pairing.off_meta)
    if on_host_params != off_host_params:
        raise ValueError(
            f"host_params mismatch for seed {pairing.on_meta.seed}: "
            f"ON {on_host_params} != OFF {off_host_params}; G2 must compare equal-host pairs"
        )
    if pairing.host_params != on_host_params:
        raise ValueError(
            f"spec host_params={pairing.host_params} does not match telemetry "
            f"host_params={on_host_params} for seed {pairing.on_meta.seed}; "
            "G2 added-parameter evidence is telemetry-derived"
        )
    return on_host_params


def build_report(
    conn: duckdb.DuckDBPyConnection,
    pairings: list[SeedPairing],
    thresholds: FrozenThresholds,
    *,
    n: int,
    budget: int,
    floor: float = 1.0,
    floored_asymmetry_max: float = 0.10,
    g3_hold: bool,
    g4_hold: bool,
    safety_evidence: SafetyEvidence | None = None,
) -> BuiltReport:
    """Assemble frozen-config telemetry into a scored Stage-2 verdict (§10 scoring phase).

    Per pairing: read both legs' ``ppo_updates``, run §1 ``validate_pair`` and merge the §6
    env-count-completeness reasons into one ``Validity``; if valid, reduce each leg to a ``LegSeries``,
    read the §6 paired safety deltas (ON − OFF), and ``score`` against the ALREADY-FROZEN thresholds.
    Returns the report AND the ``Validity`` (the latter carries the breach list ``render_packet`` needs).

    It NEVER calibrates (δ is frozen input — "no peeking" is mechanical) and NEVER reads
    ``actor_advantage_source`` from telemetry (caller supplies ``RunMeta``). ``g3_hold``/``g4_hold``
    are injected: the G4 guard-channel reader and the churn threshold are the S7/§6-definitional tail.
    """
    loaded: list[tuple[SeedPairing, int, list[UpdateRow], list[UpdateRow]]] = []
    reasons: list[str] = pairset_validity_reasons(pairings)
    for pairing in pairings:
        host_params = _host_params_for_pairing(pairing)
        on_rows = read_leg(conn, pairing.on_meta.run_dir)
        off_rows = read_leg(conn, pairing.off_meta.run_dir)
        pair_validity = validate_pair(
            pairing.on_meta, on_rows, pairing.off_meta, off_rows,
            w=thresholds.w, budget=budget, floor=floor, floored_asymmetry_max=floored_asymmetry_max,
        )
        reasons.extend(pair_validity.reasons)
        reasons.extend(
            env_count_completeness_reasons(
                conn, on_run_dir=pairing.on_meta.run_dir, off_run_dir=pairing.off_meta.run_dir
            )
        )
        loaded.append((pairing, host_params, on_rows, off_rows))

    validity = Validity(valid=not reasons, reasons=tuple(reasons))
    if not validity.valid:
        # §1: no interpretation on a broken run — score short-circuits before touching pairs.
        report = score(
            [],
            thresholds,
            n=n,
            validity=validity,
            g3_hold=g3_hold,
            g4_hold=g4_hold,
            safety_evidence=safety_evidence,
        )
        return BuiltReport(report=report, validity=validity)

    pairs: list[SeedPair] = []
    for pairing, host_params, on_rows, off_rows in loaded:
        on_ls = leg_series(on_rows, leg=Leg.ON, w=thresholds.w, floor=floor)
        off_ls = leg_series(off_rows, leg=Leg.OFF, w=thresholds.w, floor=floor)
        d_val_acc = read_run_val_acc(conn, pairing.on_meta.run_dir) - read_run_val_acc(
            conn, pairing.off_meta.run_dir
        )
        d_added_params = read_run_added_params(
            conn, pairing.on_meta.run_dir, host_params
        ) - read_run_added_params(conn, pairing.off_meta.run_dir, host_params)
        pairs.append(
            SeedPair(
                seed=pairing.on_meta.seed, on=on_ls, off=off_ls,
                d_val_acc=d_val_acc, d_added_params=d_added_params,
            )
        )
    report = score(
        pairs,
        thresholds,
        n=n,
        validity=validity,
        g3_hold=g3_hold,
        g4_hold=g4_hold,
        safety_evidence=safety_evidence,
    )
    return BuiltReport(report=report, validity=validity)


# --- S7: CLI core (spec -> text), consumed by scripts/stage2_packet.py -------------------------
#
# Spec fields use DIRECT key access (KeyError on a missing field is a spec bug, fail loud).
# Two phases enforce the §10 freeze order at the CLI surface too: `calibrate` reads OFF arms
# only and prints the δ the owner freezes; `score` consumes the frozen thresholds verbatim.


def calibrate_from_spec(conn: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> str:
    """§10 step 2 — freeze δ from the OFF arms only; returns the calibration report text.

    Spec: ``{"off_run_dirs": [...], "w": int, "budget": int}``. An OFF leg scoring fewer
    updates than ``budget`` would freeze δ from an incomplete series — fail loud.
    """
    off_run_dirs = list(spec["off_run_dirs"])
    duplicate_off_dirs = _duplicate_items(off_run_dirs)
    if duplicate_off_dirs:
        raise ValueError(
            "duplicate OFF calibration run_dir(s): "
            f"{', '.join(duplicate_off_dirs)} — δ must be frozen from unique OFF runs (§10)"
        )
    off_legs = []
    for run_dir in off_run_dirs:
        read_run_meta(conn, run_dir)
        rows = read_leg(conn, run_dir)
        require_hra_signature(rows, Leg.OFF, run_dir=run_dir)
        series = leg_series(rows, leg=Leg.OFF, w=spec["w"])
        if series.n_updates_scored < spec["budget"]:
            raise ValueError(
                f"OFF leg {run_dir!r} scores {series.n_updates_scored} updates < budget "
                f"{spec['budget']} — δ must not be frozen from an incomplete series (§1/§10)"
            )
        off_legs.append(series)
    cal = calibrate_off(off_legs)
    lines = [
        "# Stage-2 OFF calibration (§10 step 2)",
        "",
        f"delta (δ, freeze this): {cal.delta:.6f}",
        f"eps_rel (ε_rel, fixed): {cal.eps_rel:.2f}",
        f"OFF seed-to-seed adv-residual spread anchor: {cal.eps_rel_off_spread_anchor:.4f}"
        "  (ε_rel should exceed this, §4)",
        f"OFF ev levels: {', '.join(f'{v:.4f}' for v in cal.off_ev_levels)}",
        f"OFF ev IQRs:   {', '.join(f'{v:.4f}' for v in cal.off_ev_iqrs)}",
        "",
        "Freeze δ + the §11 thresholds in the gate doc BEFORE reading any ON leg.",
    ]
    return "\n".join(lines)


def packet_from_spec(conn: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> str:
    """§10 step 4 — score the pairing spec against FROZEN thresholds; returns the packet.

    Reads RunMeta from telemetry (read_run_meta — §0 provenance verified, not trusted),
    computes the §6 G3/G4 holds via the explicit-threshold predicates (conjunction over
    seeds: one materially-elevated pair fails the pairset's hold), then build_report ->
    render_packet with the Validity the assembler computed.
    """
    thresholds = FrozenThresholds(
        delta=spec["thresholds"]["delta"],
        eps_rel=spec["thresholds"]["eps_rel"],
        tau_acc=spec["thresholds"]["tau_acc"],
        delta_param_max=spec["thresholds"]["delta_param_max"],
        w=spec["thresholds"]["w"],
    )

    pairings: list[SeedPairing] = []
    g3_holds: list[bool] = []
    g4_holds: list[bool] = []
    g3_on: list[ChurnRates] = []
    g3_off: list[ChurnRates] = []
    g4_on: list[GuardChannelCounts] = []
    g4_off: list[GuardChannelCounts] = []
    for pair in spec["pairs"]:
        on_meta = read_run_meta(conn, pair["on_run_dir"])
        off_meta = read_run_meta(conn, pair["off_run_dir"])
        # The spec's declared seed is a cross-check against telemetry, not a trusted input.
        for meta in (on_meta, off_meta):
            if meta.seed != pair["seed"]:
                raise ValueError(
                    f"spec/telemetry seed mismatch for {meta.run_dir!r}: spec says "
                    f"{pair['seed']}, runs view says {meta.seed}"
                )
        pairings.append(
            SeedPairing(on_meta=on_meta, off_meta=off_meta, host_params=spec["host_params"])
        )
        on_churn = read_run_churn(conn, pair["on_run_dir"])
        off_churn = read_run_churn(conn, pair["off_run_dir"])
        g3_on.append(on_churn)
        g3_off.append(off_churn)
        g3_holds.append(
            g3_hold_from_churn(on_churn, off_churn, ratio_max=spec["g3_ratio_max"])
        )
        on_counts = read_run_guard_channels(conn, pair["on_run_dir"])
        off_counts = read_run_guard_channels(conn, pair["off_run_dir"])
        g4_on.append(on_counts)
        g4_off.append(off_counts)
        g4_holds.append(
            g4_hold_from_counts(
                on_counts,
                off_counts,
                ratio_max=spec["g4_ratio_max"],
                abs_floor=spec["g4_abs_floor"],
            )
        )
    safety_evidence = SafetyEvidence(
        g3_churn_on=tuple(g3_on),
        g3_churn_off=tuple(g3_off),
        g3_ratio_max=spec["g3_ratio_max"],
        g4_counts_on=tuple(g4_on),
        g4_counts_off=tuple(g4_off),
        g4_ratio_max=spec["g4_ratio_max"],
        g4_abs_floor=spec["g4_abs_floor"],
    )

    built = build_report(
        conn,
        pairings,
        thresholds,
        n=spec["n"],
        budget=spec["budget"],
        g3_hold=all(g3_holds),
        g4_hold=all(g4_holds),
        safety_evidence=safety_evidence,
    )
    return render_packet(built.report, thresholds=thresholds, validity=built.validity)
