"""Tests for the Stage-2 MAJOR-1 acceptance wrapper's telemetry I/O boundary.

The dict->UpdateRow conversion happens HERE and fails loud on a missing column (schema drift) —
the pure layer never touches raw dicts. read_leg is exercised against an in-memory duckdb
``ppo_updates`` table with the real column names (verified via describe_view), so duckdb's actual
NULL / DOUBLE marshalling is on the test path; the JSONL->view extraction is Karn's own tested
concern and is exercised end-to-end at the CLI slice.
"""

import dataclasses

import duckdb
import pytest

from esper.simic.telemetry.stage2_acceptance import GateResult, Verdict
from esper.simic.telemetry.stage2_acceptance_io import (
    BuiltReport,
    ChurnRates,
    SeedPairing,
    build_report,
    env_count_completeness_reasons,
    off_calibration_validity_reasons,
    pairset_validity_reasons,
    read_leg,
    read_run_added_params,
    read_run_churn,
    read_run_n_envs,
    read_run_uses_per_head_norm,
    read_run_val_acc,
    row_to_update,
)
from esper.leyline.telemetry import ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED
from esper.simic.telemetry.stage2_acceptance_io import (
    calibrate_from_spec,
    packet_from_spec,
    read_run_guard_channels,
    read_run_meta,
)
from esper.simic.telemetry.stage2_acceptance_packet import (
    FrozenThresholds,
    GuardChannelCounts,
    RunMeta,
    UpdateRow,
    g3_hold_from_churn,
    g4_hold_from_counts,
    render_packet,
)
from tests.simic.telemetry.conftest import create_ppo_updates_table, insert_ppo_update


def _full_row(**overrides) -> dict:
    """A ppo_updates row dict as duckdb's _rows would return it (OFF-leg defaults: ev_* null)."""
    row: dict = dict(
        run_dir="/run/off",
        inner_epoch=0,
        batch=0,
        explained_variance=0.5,
        ev_sum=None,
        ev_main=None,
        ev_cf=None,
        value_main_target_scale=None,
        cf_value_target_scale=None,
        cf_value_loss=None,
        ev_return_variance=5.0,
        pre_norm_advantage_std=1.0,
        return_std=3.0,
        gradient_cv=0.10,
        advantage_std_floored=False,
        advantage_per_head_normalized=False,
    )
    row.update(overrides)
    return row


# ---- row_to_update: the dict -> typed boundary ----


def test_row_to_update_maps_all_fields():
    row = _full_row(
        inner_epoch=2, batch=3, explained_variance=0.7, ev_sum=0.7, ev_main=0.2, ev_cf=0.1,
        value_main_target_scale=1.7, cf_value_target_scale=6.3, cf_value_loss=0.9,
        ev_return_variance=8.0, pre_norm_advantage_std=1.5, return_std=2.0, gradient_cv=0.4,
        advantage_std_floored=True,
    )
    assert row_to_update(row) == UpdateRow(
        inner_epoch=2, batch=3, explained_variance=0.7, ev_sum=0.7, ev_main=0.2, ev_cf=0.1,
        value_main_target_scale=1.7, cf_value_target_scale=6.3, cf_value_loss=0.9,
        ev_return_variance=8.0, pre_norm_advantage_std=1.5, return_std=2.0, gradient_cv=0.4,
        advantage_std_floored=True,
    )


def test_row_to_update_preserves_null_per_stream_ev_on_off_leg():
    u = row_to_update(
        _full_row(
            ev_sum=None,
            ev_main=None,
            ev_cf=None,
            value_main_target_scale=None,
            cf_value_target_scale=None,
        )
    )
    assert (
        u.ev_sum,
        u.ev_main,
        u.ev_cf,
        u.value_main_target_scale,
        u.cf_value_target_scale,
    ) == (None, None, None, None, None)


def test_row_to_update_fails_loud_on_missing_required_column():
    row = _full_row()
    del row["explained_variance"]
    with pytest.raises(KeyError):
        row_to_update(row)


def test_row_to_update_fails_loud_on_null_required_value():
    # A NULL where a value is required is a telemetry gap — surface it, do not coerce to 0.0.
    with pytest.raises(ValueError):
        row_to_update(_full_row(return_std=None))


def test_row_to_update_fails_loud_on_missing_optional_column():
    # An ABSENT ev_sum column is schema drift (fail loud); a NULL ev_sum VALUE is a valid OFF leg.
    row = _full_row()
    del row["ev_sum"]
    with pytest.raises(KeyError):
        row_to_update(row)


# ---- read_leg: duckdb query + chronological order + typed conversion ----


def _conn_with_updates(rows: list[dict]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_ppo_updates_table(conn)
    for r in rows:
        insert_ppo_update(conn, r)
    return conn


def test_read_leg_filters_by_run_dir_and_orders_chronologically():
    # Chronology is (batch, inner_epoch): rollout batch outer, optimize inner_epoch inner.
    rows = [
        _full_row(run_dir="/A", batch=1, inner_epoch=0, explained_variance=0.6),
        _full_row(run_dir="/A", batch=0, inner_epoch=1, explained_variance=0.5),
        _full_row(run_dir="/A", batch=0, inner_epoch=0, explained_variance=0.4),
        _full_row(run_dir="/B", batch=0, inner_epoch=0, explained_variance=0.9),
    ]
    legs = read_leg(_conn_with_updates(rows), "/A")
    assert [(u.batch, u.inner_epoch) for u in legs] == [(0, 0), (0, 1), (1, 0)]
    assert [u.explained_variance for u in legs] == [0.4, 0.5, 0.6]


def test_read_leg_returns_typed_update_rows():
    legs = read_leg(_conn_with_updates([_full_row(run_dir="/A")]), "/A")
    assert all(isinstance(u, UpdateRow) for u in legs)


def test_read_leg_marshals_null_ev_columns_to_none():
    conn = _conn_with_updates([_full_row(run_dir="/A", ev_sum=None, ev_main=None, ev_cf=None)])
    u = read_leg(conn, "/A")[0]
    assert (u.ev_sum, u.ev_main, u.ev_cf) == (None, None, None)


def test_read_leg_marshals_populated_ev_columns_on_on_leg():
    conn = _conn_with_updates([
        _full_row(
            run_dir="/A", ev_sum=0.8, ev_main=0.3, ev_cf=0.1,
            value_main_target_scale=1.7, cf_value_target_scale=6.3,
        )
    ])
    u = read_leg(conn, "/A")[0]
    assert (
        u.ev_sum,
        u.ev_main,
        u.ev_cf,
        u.value_main_target_scale,
        u.cf_value_target_scale,
    ) == (0.8, 0.3, 0.1, 1.7, 6.3)


def test_read_leg_fails_loud_on_unknown_run_dir():
    conn = _conn_with_updates([_full_row(run_dir="/A")])
    with pytest.raises(ValueError):
        read_leg(conn, "/does-not-exist")


def test_read_run_uses_per_head_norm_true_if_any_update_normalized():
    # §8F(i): BOOL_OR over the run — True if per-head advantage norm was on for ANY update.
    conn = _conn_with_updates([
        _full_row(run_dir="/A", batch=0, advantage_per_head_normalized=False),
        _full_row(run_dir="/A", batch=1, advantage_per_head_normalized=True),
    ])
    assert read_run_uses_per_head_norm(conn, "/A") is True


def test_read_run_uses_per_head_norm_false_when_all_updates_unnormalized():
    conn = _conn_with_updates([_full_row(run_dir="/A", advantage_per_head_normalized=False)])
    assert read_run_uses_per_head_norm(conn, "/A") is False


# ---- episode_outcomes safety readers (§6 G1/G2/G3) ----


def _outcome_row(**overrides) -> dict:
    row: dict = dict(
        run_dir="/run",
        env_id=0,
        episode_idx=0,
        final_accuracy=50.0,
        param_ratio=1.0,
        germinate_count=0,
        prune_count=0,
        fossilize_count=0,
    )
    row.update(overrides)
    return row


def _conn_with_outcomes(rows: list[dict]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE episode_outcomes (
            run_dir VARCHAR, env_id INTEGER, episode_idx INTEGER, final_accuracy DOUBLE,
            param_ratio DOUBLE, germinate_count INTEGER, prune_count INTEGER, fossilize_count INTEGER
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO episode_outcomes VALUES (?,?,?,?,?,?,?,?)",
            [
                r["run_dir"], r["env_id"], r["episode_idx"], r["final_accuracy"],
                r["param_ratio"], r["germinate_count"], r["prune_count"], r["fossilize_count"],
            ],
        )
    return conn


def test_read_run_val_acc_is_mean_of_per_env_terminal_final_accuracy():
    # §6 G1: episode_idx is a DENSE GLOBAL counter (episodes_completed + env_idx), so envs terminate
    # at DIFFERENT indices. Val-acc = mean over EACH env's OWN terminal episode (not the single
    # global-max row, which would collapse to one env). drl-expert Finding 1.
    rows = [
        _outcome_row(env_id=0, episode_idx=0, final_accuracy=30.0),  # env0 earlier
        _outcome_row(env_id=0, episode_idx=2, final_accuracy=46.0),  # env0 terminal
        _outcome_row(env_id=1, episode_idx=1, final_accuracy=31.0),  # env1 earlier
        _outcome_row(env_id=1, episode_idx=3, final_accuracy=48.0),  # env1 terminal (global max)
    ]
    # mean(env0_terminal=46.0, env1_terminal=48.0) = 47.0 — NOT 48.0 (the single global-max row).
    assert read_run_val_acc(_conn_with_outcomes(rows), "/run") == pytest.approx(47.0)


def test_read_run_added_params_uses_per_env_terminal_param_ratio():
    # §6 G2: per-env terminal param_ratio, averaged over envs; added = host * (mean_ratio - 1).
    rows = [
        _outcome_row(env_id=0, episode_idx=0, param_ratio=1.05),  # env0 earlier
        _outcome_row(env_id=0, episode_idx=2, param_ratio=1.20),  # env0 terminal
        _outcome_row(env_id=1, episode_idx=3, param_ratio=1.30),  # env1 terminal (global max)
    ]
    # mean per-env terminal ratio = (1.20 + 1.30)/2 = 1.25 -> 100_000 * 0.25 = 25_000.
    added = read_run_added_params(_conn_with_outcomes(rows), "/run", host_params=100_000)
    assert added == pytest.approx(25_000.0)


def test_read_run_churn_is_mean_counts_per_episode_over_the_run():
    # §6 G3: churn = germinate/prune/fossilize per-episode means over the whole run.
    rows = [
        _outcome_row(episode_idx=0, germinate_count=4, prune_count=2, fossilize_count=1),
        _outcome_row(episode_idx=1, germinate_count=8, prune_count=4, fossilize_count=3),
    ]
    churn = read_run_churn(_conn_with_outcomes(rows), "/run")
    assert isinstance(churn, ChurnRates)
    assert churn.germinate == pytest.approx(6.0)
    assert churn.prune == pytest.approx(3.0)
    assert churn.fossilize == pytest.approx(2.0)


def test_read_run_val_acc_rejects_null_terminal_accuracy():
    rows = [
        _outcome_row(env_id=0, episode_idx=0, final_accuracy=40.0),
        _outcome_row(env_id=1, episode_idx=1, final_accuracy=None),
    ]
    with pytest.raises(ValueError, match="final_accuracy"):
        read_run_val_acc(_conn_with_outcomes(rows), "/run")


def test_read_run_added_params_rejects_null_terminal_param_ratio():
    rows = [
        _outcome_row(env_id=0, episode_idx=0, param_ratio=1.2),
        _outcome_row(env_id=1, episode_idx=1, param_ratio=None),
    ]
    with pytest.raises(ValueError, match="param_ratio"):
        read_run_added_params(_conn_with_outcomes(rows), "/run", host_params=100_000)


def test_read_run_churn_rejects_null_churn_fields():
    rows = [
        _outcome_row(episode_idx=0, germinate_count=1, prune_count=1, fossilize_count=0),
        _outcome_row(episode_idx=1, germinate_count=2, prune_count=None, fossilize_count=1),
    ]
    with pytest.raises(ValueError, match="churn fields"):
        read_run_churn(_conn_with_outcomes(rows), "/run")


def test_read_run_val_acc_fails_loud_on_unknown_run_dir():
    conn = _conn_with_outcomes([_outcome_row(run_dir="/run")])
    with pytest.raises(ValueError):
        read_run_val_acc(conn, "/missing")


def test_read_run_added_params_fails_loud_on_unknown_run_dir():
    conn = _conn_with_outcomes([_outcome_row(run_dir="/run")])
    with pytest.raises(ValueError):
        read_run_added_params(conn, "/missing", host_params=100_000)


# ---- S6: env-count-completeness (backstops Finding 1's per-env-terminal aggregation) ----
#
# The per-env-terminal G1/G2 mean silently averages over WHATEVER envs are present; if a leg's
# episode_outcomes is missing an env entirely, the paired safety delta is biased. So the pair is
# rejected unless each leg covers all runs.n_envs distinct env_ids AND ON n_envs == OFF n_envs.
# Hard-reject is the conservative stance; the threshold is definitional (pending §6/drl).


def _conn_with_runs_and_outcomes(
    runs: list[dict], outcomes: list[dict]
) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE runs (run_dir VARCHAR, n_envs INTEGER, n_episodes INTEGER)")
    for r in runs:
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            [r["run_dir"], r["n_envs"], r.get("n_episodes", r["n_envs"])],
        )
    conn.execute(
        """
        CREATE TABLE episode_outcomes (
            run_dir VARCHAR, env_id INTEGER, episode_idx INTEGER, final_accuracy DOUBLE,
            param_ratio DOUBLE, germinate_count INTEGER, prune_count INTEGER, fossilize_count INTEGER
        )
        """
    )
    for r in outcomes:
        full = _outcome_row(**r)
        conn.execute(
            "INSERT INTO episode_outcomes VALUES (?,?,?,?,?,?,?,?)",
            [
                full["run_dir"], full["env_id"], full["episode_idx"], full["final_accuracy"],
                full["param_ratio"], full["germinate_count"], full["prune_count"],
                full["fossilize_count"],
            ],
        )
    return conn


def _terminal_outcomes(run_dir: str, env_ids: list[int]) -> list[dict]:
    """One terminal outcome per env (episode_idx = env_id, a distinct dense-global index)."""
    return [{"run_dir": run_dir, "env_id": e, "episode_idx": e} for e in env_ids]


def test_read_run_n_envs_reads_the_runs_view():
    conn = _conn_with_runs_and_outcomes([{"run_dir": "/A", "n_envs": 8}], [])
    assert read_run_n_envs(conn, "/A") == 8


def test_read_run_n_envs_fails_loud_on_unknown_run_dir():
    conn = _conn_with_runs_and_outcomes([{"run_dir": "/A", "n_envs": 8}], [])
    with pytest.raises(ValueError):
        read_run_n_envs(conn, "/missing")


def test_read_run_n_envs_rejects_duplicate_runs_rows():
    conn = _conn_with_runs_and_outcomes(
        [{"run_dir": "/A", "n_envs": 8}, {"run_dir": "/A", "n_envs": 8}], []
    )
    with pytest.raises(ValueError, match="duplicate|exactly one"):
        read_run_n_envs(conn, "/A")


def test_env_count_completeness_empty_when_both_legs_cover_all_envs():
    conn = _conn_with_runs_and_outcomes(
        runs=[{"run_dir": "/on", "n_envs": 3}, {"run_dir": "/off", "n_envs": 3}],
        outcomes=_terminal_outcomes("/on", [0, 1, 2]) + _terminal_outcomes("/off", [0, 1, 2]),
    )
    assert env_count_completeness_reasons(conn, on_run_dir="/on", off_run_dir="/off") == []


def test_env_count_completeness_flags_incomplete_leg():
    # ON ran 3 envs but only 2 appear in episode_outcomes — the per-env mean is over a subset.
    conn = _conn_with_runs_and_outcomes(
        runs=[{"run_dir": "/on", "n_envs": 3}, {"run_dir": "/off", "n_envs": 3}],
        outcomes=_terminal_outcomes("/on", [0, 1]) + _terminal_outcomes("/off", [0, 1, 2]),
    )
    reasons = env_count_completeness_reasons(conn, on_run_dir="/on", off_run_dir="/off")
    assert any("ON" in r and "2/3" in r for r in reasons)


def test_env_count_completeness_flags_n_envs_mismatch_between_arms():
    conn = _conn_with_runs_and_outcomes(
        runs=[{"run_dir": "/on", "n_envs": 4}, {"run_dir": "/off", "n_envs": 3}],
        outcomes=_terminal_outcomes("/on", [0, 1, 2, 3]) + _terminal_outcomes("/off", [0, 1, 2]),
    )
    reasons = env_count_completeness_reasons(conn, on_run_dir="/on", off_run_dir="/off")
    assert any("mismatch" in r.lower() for r in reasons)


def test_env_count_completeness_flags_stale_nonterminal_env_rows():
    conn = _conn_with_runs_and_outcomes(
        runs=[
            {"run_dir": "/on", "n_envs": 2, "n_episodes": 4},
            {"run_dir": "/off", "n_envs": 2, "n_episodes": 4},
        ],
        outcomes=[
            {"run_dir": "/on", "env_id": 0, "episode_idx": 0},
            {"run_dir": "/on", "env_id": 0, "episode_idx": 2},
            {"run_dir": "/on", "env_id": 1, "episode_idx": 1},
            {"run_dir": "/off", "env_id": 0, "episode_idx": 0},
            {"run_dir": "/off", "env_id": 0, "episode_idx": 2},
            {"run_dir": "/off", "env_id": 1, "episode_idx": 1},
            {"run_dir": "/off", "env_id": 1, "episode_idx": 3},
        ],
    )
    reasons = env_count_completeness_reasons(conn, on_run_dir="/on", off_run_dir="/off")
    assert any("ON" in reason and "terminal" in reason for reason in reasons)


# ---- S6: build_report — the scoring-phase assembly (telemetry -> scored verdict) ----
#
# build_report wires read_leg -> leg_series -> validate_pair (+ env-count reasons merged into the
# Validity) -> score, with caller-supplied RunMeta (so it reads no not-yet-emitted column) and
# already-FROZEN thresholds (it never calibrates — "no peeking" is mechanical). g3/g4 holds are
# injected (the G4 reader + churn threshold are S7/§6-definitional).


_IO_THRESHOLDS = FrozenThresholds(delta=0.05, eps_rel=0.10, tau_acc=0.3, delta_param_max=1e9, w=8)


def _build_conn(specs: list[dict]) -> duckdb.DuckDBPyConnection:
    """Full ppo_updates + episode_outcomes + runs fixture. Each spec is one run (ON or OFF)."""
    conn = duckdb.connect(":memory:")
    create_ppo_updates_table(conn)
    conn.execute(
        """
        CREATE TABLE episode_outcomes (
            run_dir VARCHAR, env_id INTEGER, episode_idx INTEGER, final_accuracy DOUBLE,
            param_ratio DOUBLE, germinate_count INTEGER, prune_count INTEGER, fossilize_count INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE runs (run_dir VARCHAR, n_envs INTEGER, n_episodes INTEGER)")
    # Median-preserving jitter over 4 updates so every leg has NONZERO within-run IQR — a
    # constant-EV leg is degenerate (IQR=0) and the scorer rightly rejects it; real telemetry varies.
    ev_jitter = [-0.02, 0.0, 0.02, 0.0]
    std_jitter = [-0.5, 0.0, 0.5, 0.0]
    for s in specs:
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            [s["run_dir"], s["n_envs"], s.get("n_episodes", s["n_envs"])],
        )
        is_on = s["leg"] == "on"
        # 12 updates so the §11 plateau check (trailing-8 window, w=8) is satisfiable; the
        # jitter cycles per 4, so the post-burn-in scored window is one full cycle and every
        # median/IQR matches the original 4-update fixture.
        for b in range(12):
            expl_b = s["expl"] + ev_jitter[b % 4]
            return_std_b = 3.0 + std_jitter[b % 4]
            insert_ppo_update(conn, dict(
                run_dir=s["run_dir"], inner_epoch=0, batch=b, explained_variance=expl_b,
                ev_sum=expl_b if is_on else None,  # total-EV comparand on ON, null on OFF
                ev_main=0.03 if is_on else None,   # feeds §5 MECH, null on OFF
                ev_cf=0.10 if is_on else None,
                ev_return_variance=5.0,
                value_main_target_scale=1.7 if is_on else None,
                cf_value_target_scale=6.3 if is_on else None,
                cf_value_loss=0.02 if is_on else None,  # flat => plateaued by w=8
                pre_norm_advantage_std=1.0,
                return_std=return_std_b,
                gradient_cv=0.10,
                advantage_std_floored=False,
                advantage_per_head_normalized=False,
            ))
        for env_id in s.get("env_ids", list(range(s["n_envs"]))):
            conn.execute(
                "INSERT INTO episode_outcomes VALUES (?,?,?,?,?,?,?,?)",
                [s["run_dir"], env_id, env_id, s.get("final_acc", 50.0),
                 s.get("param_ratio", 1.1), 1, 1, 0],
            )
    return conn


def _meta(run_dir: str, seed: int) -> RunMeta:
    return RunMeta(
        run_dir=run_dir, seed=seed, reward_mode="SHAPED",
        actor_advantage_source=ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED, uses_per_head_norm=False,
        frozen_config=_RUN_META_FROZEN_CONFIG,
        placement=_RUN_META_PLACEMENT,
    )


def _pairset(
    n: int, *, expl_on: float = 0.85, expl_off: float = 0.80,
    on_final: float = 50.0, off_final: float = 50.0, n_envs: int = 1,
    on_env_ids: list[int] | None = None,
) -> tuple[duckdb.DuckDBPyConnection, list[SeedPairing]]:
    specs: list[dict] = []
    pairings: list[SeedPairing] = []
    for seed in range(n):
        on_dir, off_dir = f"/on/{seed}", f"/off/{seed}"
        specs.append({
            "run_dir": on_dir, "leg": "on", "n_envs": n_envs, "expl": expl_on,
            "final_acc": on_final, "env_ids": on_env_ids if on_env_ids is not None else list(range(n_envs)),
        })
        specs.append({
            "run_dir": off_dir, "leg": "off", "n_envs": n_envs, "expl": expl_off,
            "final_acc": off_final, "env_ids": list(range(n_envs)),
        })
        pairings.append(SeedPairing(on_meta=_meta(on_dir, seed), off_meta=_meta(off_dir, seed), host_params=100_000))
    return _build_conn(specs), pairings


def test_build_report_assembles_valid_pairset_to_a_scored_verdict():
    conn, pairings = _pairset(5)
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert isinstance(built, BuiltReport)
    assert built.report.verdict is not Verdict.INVALID
    assert built.validity.valid is True
    assert len(built.report.delta_a) == 5
    # Δ_A = ev_level_on (0.85) − ev_level_off (0.80), read straight from the telemetry.
    assert built.report.delta_a[0] == pytest.approx(0.05)


def test_build_report_rejects_env_incomplete_pair_as_invalid():
    # ON ran 2 envs but only env 0 appears in episode_outcomes — the per-env safety mean is biased.
    conn, pairings = _pairset(5, n_envs=2, on_env_ids=[0])
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.verdict is Verdict.INVALID


def test_build_report_returns_validity_so_render_can_list_the_breaches():
    # The build_report -> render_packet COMPOSE: the assembler must hand back the Validity it
    # computed, else a caller has nothing truthful to pass and an INVALID packet lists no breaches.
    conn, pairings = _pairset(5, n_envs=2, on_env_ids=[0])
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert built.validity.valid is False
    assert any("envs" in reason for reason in built.validity.reasons)
    packet = render_packet(built.report, thresholds=_IO_THRESHOLDS, validity=built.validity)
    assert "INVALID" in packet
    assert "1/2 envs" in packet  # the actual env-count breach is rendered, not an empty body


def test_build_report_rejects_seed_mismatch_as_invalid():
    conn, pairings = _pairset(5)
    bad = pairings[0]
    pairings[0] = SeedPairing(
        on_meta=bad.on_meta, off_meta=_meta(bad.off_meta.run_dir, 999), host_params=100_000
    )
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.verdict is Verdict.INVALID


def test_build_report_rejects_duplicate_pairset_seed_as_invalid():
    conn, pairings = _pairset(5)
    bad = pairings[1]
    pairings[1] = SeedPairing(
        on_meta=_meta(bad.on_meta.run_dir, 0),
        off_meta=_meta(bad.off_meta.run_dir, 0),
        host_params=100_000,
    )
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.verdict is Verdict.INVALID
    assert any("duplicate seed" in reason for reason in built.validity.reasons)


def test_build_report_rejects_duplicate_run_dir_as_invalid():
    conn, pairings = _pairset(5)
    bad = pairings[1]
    pairings[1] = SeedPairing(
        on_meta=_meta(pairings[0].on_meta.run_dir, 1),
        off_meta=bad.off_meta,
        host_params=100_000,
    )
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.verdict is Verdict.INVALID
    assert any("duplicate run_dir" in reason for reason in built.validity.reasons)


def test_build_report_rejects_on_off_run_dir_overlap_as_invalid():
    conn, pairings = _pairset(5)
    bad = pairings[0]
    pairings[0] = SeedPairing(
        on_meta=bad.on_meta,
        off_meta=_meta(bad.on_meta.run_dir, bad.off_meta.seed),
        host_params=100_000,
    )
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.verdict is Verdict.INVALID
    assert any("both ON and OFF" in reason for reason in built.validity.reasons)


def test_build_report_rejects_global_frozen_config_drift_as_invalid():
    conn, pairings = _pairset(5)
    common_config = (("task", "cifar_baseline"), ("gamma", 0.99), ("host_params", 100_000))
    drifted_config = (("task", "cifar_baseline"), ("gamma", 0.90), ("host_params", 100_000))

    updated: list[SeedPairing] = []
    for index, pairing in enumerate(pairings):
        config = drifted_config if index == 1 else common_config
        updated.append(
            SeedPairing(
                on_meta=RunMeta(
                    run_dir=pairing.on_meta.run_dir,
                    seed=pairing.on_meta.seed,
                    reward_mode=pairing.on_meta.reward_mode,
                    actor_advantage_source=pairing.on_meta.actor_advantage_source,
                    uses_per_head_norm=pairing.on_meta.uses_per_head_norm,
                    frozen_config=config,
                ),
                off_meta=RunMeta(
                    run_dir=pairing.off_meta.run_dir,
                    seed=pairing.off_meta.seed,
                    reward_mode=pairing.off_meta.reward_mode,
                    actor_advantage_source=pairing.off_meta.actor_advantage_source,
                    uses_per_head_norm=pairing.off_meta.uses_per_head_norm,
                    frozen_config=config,
                ),
                host_params=pairing.host_params,
            )
        )

    built = build_report(conn, updated, _IO_THRESHOLDS, n=5, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.verdict is Verdict.INVALID
    assert any("global frozen run config" in reason for reason in built.validity.reasons)


def test_build_report_wires_paired_safety_g1_regression_to_reject():
    # ON val-acc 10pp below OFF on every seed -> paired Δ_G1 median = -10 < -τ_acc -> G1 FAIL -> REJECT.
    conn, pairings = _pairset(10, on_final=40.0, off_final=50.0)
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=10, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.g1 is GateResult.FAIL
    assert built.report.verdict is Verdict.REJECT


# ---- S7: read_run_meta — the runs-view provenance reader (§0/§1) ----


_RUN_META_FROZEN_DEFAULTS = {
    "task": "cifar_baseline",
    "reward_family": "contribution",
    "n_envs": 1,
    "n_episodes": 5,
    "max_epochs": 150,
    "max_batches": 5,
    "lr": 0.001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_updates_per_batch": 1,
    "recurrent_n_epochs": 1,
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "per_head_advantage_norm": False,
    "return_variance_telemetry": False,
    "value_coef": 0.5,
    "value_warmup_batches": 0,
    "value_coef_start": None,
    "param_budget": 100_000,
    "param_penalty_weight": 0.1,
    "sparse_reward_scale": 1.0,
    "rent_host_params_floor": 200,
    "basic_acc_delta_weight": 5.0,
    "plateau_threshold": 0.5,
    "improvement_threshold": 2.0,
    "gradient_telemetry_stride": 10,
    "lstm_hidden_dim": 512,
    "chunk_length": 150,
    "max_seeds": None,
    "slot_ids_json": '["r0c1"]',
    "env_devices_json": '["cpu"]',
    "policy_device": "cpu",
    "amp_enabled": False,
    "amp_dtype": None,
    "compile_enabled": False,
    "compile_backend": None,
    "compile_mode": None,
    "permissive_gates": True,
    "auto_forward_g1": False,
    "auto_forward_g2": False,
    "auto_forward_g3": False,
    "disable_pbrs": False,
    "disable_terminal_reward": False,
    "disable_anti_gaming": False,
    "max_grad_norm": None,
    "host_params": 100_000,
}

_RUN_META_PLACEMENT_COLUMNS = ("env_devices_json", "policy_device")
_RUN_META_FROZEN_CONFIG = tuple(
    (key, value)
    for key, value in _RUN_META_FROZEN_DEFAULTS.items()
    if key not in _RUN_META_PLACEMENT_COLUMNS
)
_RUN_META_PLACEMENT = tuple(
    (key, _RUN_META_FROZEN_DEFAULTS[key]) for key in _RUN_META_PLACEMENT_COLUMNS
)

_RUN_META_COLUMN_TYPES = {
    "run_dir": "VARCHAR",
    "seed": "INTEGER",
    "reward_mode": "VARCHAR",
    "actor_advantage_source": "VARCHAR",
    "resume_path": "VARCHAR",
    "start_episode": "INTEGER",
    "task": "VARCHAR",
    "reward_family": "VARCHAR",
    "n_envs": "INTEGER",
    "n_episodes": "INTEGER",
    "max_epochs": "INTEGER",
    "max_batches": "INTEGER",
    "lr": "DOUBLE",
    "gamma": "DOUBLE",
    "gae_lambda": "DOUBLE",
    "ppo_updates_per_batch": "INTEGER",
    "recurrent_n_epochs": "INTEGER",
    "clip_ratio": "DOUBLE",
    "entropy_coef": "DOUBLE",
    "per_head_advantage_norm": "BOOLEAN",
    "return_variance_telemetry": "BOOLEAN",
    "value_coef": "DOUBLE",
    "value_warmup_batches": "INTEGER",
    "value_coef_start": "DOUBLE",
    "param_budget": "INTEGER",
    "param_penalty_weight": "DOUBLE",
    "sparse_reward_scale": "DOUBLE",
    "rent_host_params_floor": "INTEGER",
    "basic_acc_delta_weight": "DOUBLE",
    "plateau_threshold": "DOUBLE",
    "improvement_threshold": "DOUBLE",
    "gradient_telemetry_stride": "INTEGER",
    "lstm_hidden_dim": "INTEGER",
    "chunk_length": "INTEGER",
    "max_seeds": "INTEGER",
    "slot_ids_json": "VARCHAR",
    "env_devices_json": "VARCHAR",
    "policy_device": "VARCHAR",
    "amp_enabled": "BOOLEAN",
    "amp_dtype": "VARCHAR",
    "compile_enabled": "BOOLEAN",
    "compile_backend": "VARCHAR",
    "compile_mode": "VARCHAR",
    "permissive_gates": "BOOLEAN",
    "auto_forward_g1": "BOOLEAN",
    "auto_forward_g2": "BOOLEAN",
    "auto_forward_g3": "BOOLEAN",
    "disable_pbrs": "BOOLEAN",
    "disable_terminal_reward": "BOOLEAN",
    "disable_anti_gaming": "BOOLEAN",
    "max_grad_norm": "DOUBLE",
    "host_params": "INTEGER",
}

_RUN_META_COLUMNS = tuple(_RUN_META_COLUMN_TYPES)


def _create_runs_meta_table(conn: duckdb.DuckDBPyConnection) -> None:
    column_sql = ", ".join(
        f"{column} {_RUN_META_COLUMN_TYPES[column]}" for column in _RUN_META_COLUMNS
    )
    conn.execute(f"CREATE TABLE runs ({column_sql})")


def _insert_run_meta(conn: duckdb.DuckDBPyConnection, row: dict) -> None:
    full = {
        "resume_path": "",
        "start_episode": 0,
        **_RUN_META_FROZEN_DEFAULTS,
        **row,
    }
    placeholders = ", ".join("?" for _ in _RUN_META_COLUMNS)
    conn.execute(
        f"INSERT INTO runs VALUES ({placeholders})",
        [full[column] for column in _RUN_META_COLUMNS],
    )


def _conn_with_runs_meta(rows: list[dict]) -> duckdb.DuckDBPyConnection:
    """runs + ppo_updates tables for read_run_meta (meta + §8F(i) per-head-norm scan)."""
    conn = duckdb.connect(":memory:")
    _create_runs_meta_table(conn)
    conn.execute(
        "CREATE TABLE ppo_updates (run_dir VARCHAR, advantage_per_head_normalized BOOLEAN)"
    )
    for row in rows:
        _insert_run_meta(conn, row)
        conn.execute(
            "INSERT INTO ppo_updates VALUES (?, ?)",
            [row["run_dir"], row.get("per_head_norm", False)],
        )
    return conn


def test_read_run_meta_reads_provenance_from_runs_view():
    conn = _conn_with_runs_meta([{
        "run_dir": "/A", "seed": 7, "reward_mode": "SHAPED",
        "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
    }])
    meta = read_run_meta(conn, "/A")
    assert meta == RunMeta(
        run_dir="/A", seed=7, reward_mode="SHAPED",
        actor_advantage_source=ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        uses_per_head_norm=False,
        frozen_config=_RUN_META_FROZEN_CONFIG,
        placement=_RUN_META_PLACEMENT,
    )


def test_read_run_meta_carries_per_head_norm_scan():
    conn = _conn_with_runs_meta([{
        "run_dir": "/A", "seed": 7, "reward_mode": "SHAPED",
        "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        "per_head_norm": True,
    }])
    assert read_run_meta(conn, "/A").uses_per_head_norm is True


def test_read_run_meta_carries_per_head_norm_config_even_when_update_scan_is_false():
    conn = _conn_with_runs_meta([{
        "run_dir": "/A", "seed": 7, "reward_mode": "SHAPED",
        "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        "per_head_advantage_norm": True,
        "per_head_norm": False,
    }])
    assert read_run_meta(conn, "/A").uses_per_head_norm is True


def test_read_run_meta_fails_loud_on_pre_s7_telemetry():
    # Pre-S7 telemetry has no actor_advantage_source (view extracts NULL): the §0 fence
    # cannot verify provenance it does not have — fail loud, never default.
    conn = _conn_with_runs_meta([{
        "run_dir": "/A", "seed": 7, "reward_mode": "SHAPED", "actor_advantage_source": None,
    }])
    with pytest.raises(ValueError):
        read_run_meta(conn, "/A")


def test_read_run_meta_rejects_resume_run():
    conn = _conn_with_runs_meta([{
        "run_dir": "/A",
        "seed": 7,
        "reward_mode": "SHAPED",
        "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        "resume_path": "/tmp/checkpoint.pt",
    }])
    with pytest.raises(ValueError, match="fresh-init"):
        read_run_meta(conn, "/A")


def test_read_run_meta_rejects_nonzero_start_episode():
    conn = _conn_with_runs_meta([{
        "run_dir": "/A",
        "seed": 7,
        "reward_mode": "SHAPED",
        "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        "start_episode": 1,
    }])
    with pytest.raises(ValueError, match="fresh-init"):
        read_run_meta(conn, "/A")


def test_read_run_meta_fails_loud_on_unknown_run_dir():
    conn = _conn_with_runs_meta([{
        "run_dir": "/A", "seed": 7, "reward_mode": "SHAPED",
        "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
    }])
    with pytest.raises(ValueError):
        read_run_meta(conn, "/missing")


def test_read_run_meta_rejects_duplicate_runs_rows():
    conn = _conn_with_runs_meta([
        {
            "run_dir": "/A", "seed": 7, "reward_mode": "SHAPED",
            "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        },
        {
            "run_dir": "/A", "seed": 7, "reward_mode": "SHAPED",
            "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        },
    ])
    with pytest.raises(ValueError, match="duplicate|exactly one"):
        read_run_meta(conn, "/A")


# ---- S7: G4 guard-channel reader + G3/G4 hold predicates (§6) ----


def _conn_with_anomalies(rows: list[tuple[str, str]]) -> duckdb.DuckDBPyConnection:
    """anomalies view fabricated as a table: (run_dir, event_type) rows."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE anomalies (run_dir VARCHAR, event_type VARCHAR)")
    for run_dir, event_type in rows:
        conn.execute("INSERT INTO anomalies VALUES (?, ?)", [run_dir, event_type])
    return conn


def test_read_run_guard_channels_counts_by_channel():
    conn = _conn_with_anomalies([
        ("/A", "GOVERNOR_ROLLBACK"),
        ("/A", "GOVERNOR_ROLLBACK"),
        ("/A", "VALUE_COLLAPSE_DETECTED"),
        ("/B", "GRADIENT_ANOMALY"),  # other run — not counted
    ])
    counts = read_run_guard_channels(conn, "/A")
    assert isinstance(counts, GuardChannelCounts)
    assert counts.governor_rollback == 2
    assert counts.value_collapse == 1
    assert counts.gradient_anomaly == 0


def test_read_run_guard_channels_excludes_plateau_detected():
    # PLATEAU_DETECTED is a benign training-progress signal, not a guard channel (§6 G4
    # names governor rollbacks / value-collapse / gradient-anomaly / instability detectors).
    conn = _conn_with_anomalies([("/A", "PLATEAU_DETECTED")])
    counts = read_run_guard_channels(conn, "/A")
    assert counts.total() == 0


def test_read_run_guard_channels_counts_reward_hacking_suspicion():
    conn = _conn_with_anomalies([("/A", "REWARD_HACKING_SUSPECTED")])
    counts = read_run_guard_channels(conn, "/A")
    assert counts.reward_hacking == 1
    assert counts.total() == 1


def test_g4_hold_passes_when_on_not_materially_elevated():
    on = GuardChannelCounts(governor_rollback=8, value_collapse=0, ratio_explosion=0,
                            ratio_collapse=0, gradient_anomaly=1, gradient_pathology=0,
                            numerical_instability=0)
    off = GuardChannelCounts(governor_rollback=7, value_collapse=0, ratio_explosion=0,
                             ratio_collapse=0, gradient_anomaly=1, gradient_pathology=0,
                             numerical_instability=0)
    assert g4_hold_from_counts(on, off, ratio_max=2.0, abs_floor=5) is True


def test_g4_hold_fails_on_material_channel_elevation():
    # ON governor rollbacks 3x OFF and +14 absolute -> materially elevated -> hold fails.
    on = GuardChannelCounts(governor_rollback=21, value_collapse=0, ratio_explosion=0,
                            ratio_collapse=0, gradient_anomaly=0, gradient_pathology=0,
                            numerical_instability=0)
    off = GuardChannelCounts(governor_rollback=7, value_collapse=0, ratio_explosion=0,
                             ratio_collapse=0, gradient_anomaly=0, gradient_pathology=0,
                             numerical_instability=0)
    assert g4_hold_from_counts(on, off, ratio_max=2.0, abs_floor=5) is False


def test_g4_hold_fails_on_any_on_reward_hacking_suspicion():
    on = GuardChannelCounts(governor_rollback=0, value_collapse=0, ratio_explosion=0,
                            ratio_collapse=0, gradient_anomaly=0, gradient_pathology=0,
                            numerical_instability=0, reward_hacking=1)
    off = GuardChannelCounts(governor_rollback=0, value_collapse=0, ratio_explosion=0,
                             ratio_collapse=0, gradient_anomaly=0, gradient_pathology=0,
                             numerical_instability=0, reward_hacking=0)
    assert g4_hold_from_counts(on, off, ratio_max=2.0, abs_floor=5) is False


def test_g4_hold_zero_off_baseline_uses_abs_floor():
    # OFF baseline 0: any ratio is infinite, so the absolute floor decides.
    on_small = GuardChannelCounts(governor_rollback=0, value_collapse=2, ratio_explosion=0,
                                  ratio_collapse=0, gradient_anomaly=0, gradient_pathology=0,
                                  numerical_instability=0)
    on_large = GuardChannelCounts(governor_rollback=0, value_collapse=9, ratio_explosion=0,
                                  ratio_collapse=0, gradient_anomaly=0, gradient_pathology=0,
                                  numerical_instability=0)
    off = GuardChannelCounts(governor_rollback=0, value_collapse=0, ratio_explosion=0,
                             ratio_collapse=0, gradient_anomaly=0, gradient_pathology=0,
                             numerical_instability=0)
    assert g4_hold_from_counts(on_small, off, ratio_max=2.0, abs_floor=5) is True
    assert g4_hold_from_counts(on_large, off, ratio_max=2.0, abs_floor=5) is False


def test_g3_hold_from_churn_passes_when_on_within_ratio():
    on = ChurnRates(germinate=13.0, prune=12.0, fossilize=0.4)
    off = ChurnRates(germinate=12.4, prune=11.6, fossilize=0.3)
    assert g3_hold_from_churn(on, off, ratio_max=1.5) is True


def test_g3_hold_from_churn_fails_on_materially_elevated_prune():
    # ON prunes 2x OFF -> germinate/prune churn decoupled from contribution -> hold fails.
    on = ChurnRates(germinate=12.0, prune=24.0, fossilize=0.3)
    off = ChurnRates(germinate=12.0, prune=11.6, fossilize=0.3)
    assert g3_hold_from_churn(on, off, ratio_max=1.5) is False


def test_g3_hold_from_churn_fails_on_materially_elevated_fossilize():
    # Fossilization-heavy reward gaming is churn too; G3 must not ignore it.
    on = ChurnRates(germinate=12.0, prune=12.0, fossilize=3.0)
    off = ChurnRates(germinate=12.0, prune=12.0, fossilize=1.0)
    assert g3_hold_from_churn(on, off, ratio_max=1.5) is False


# ---- S7: CLI core — calibrate_from_spec / packet_from_spec (telemetry -> packet, end to end) ----


def _full_stage2_conn(
    n: int, *, anomaly_rows: list[tuple[str, str]] | None = None
) -> tuple[duckdb.DuckDBPyConnection, dict]:
    """Everything the CLI reads, fabricated: runs (meta + n_envs) / ppo_updates (series +
    per-head flag) / episode_outcomes / anomalies — plus the matching score spec."""
    conn = duckdb.connect(":memory:")
    _create_runs_meta_table(conn)
    create_ppo_updates_table(conn)
    conn.execute(
        """
        CREATE TABLE episode_outcomes (
            run_dir VARCHAR, env_id INTEGER, episode_idx INTEGER, final_accuracy DOUBLE,
            param_ratio DOUBLE, germinate_count INTEGER, prune_count INTEGER, fossilize_count INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE anomalies (run_dir VARCHAR, event_type VARCHAR)")

    ev_jitter = [-0.02, 0.0, 0.02, 0.0]
    std_jitter = [-0.5, 0.0, 0.5, 0.0]
    pairs_spec = []
    for seed in range(n):
        for leg, expl in (("on", 0.85), ("off", 0.80)):
            run_dir = f"/{leg}/{seed}"
            _insert_run_meta(
                conn,
                {
                    "run_dir": run_dir,
                    "seed": seed,
                    "reward_mode": "SHAPED",
                    "actor_advantage_source": ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
                    "n_episodes": n,
                    "max_batches": n,
                },
            )
            is_on = leg == "on"
            # 12 updates (jitter cycled per 4) so the §11 plateau check is satisfiable under
            # the spec's w=8 while the scored window keeps the original cycle's median/IQR.
            for b in range(12):
                insert_ppo_update(conn, dict(
                    run_dir=run_dir, inner_epoch=0, batch=b,
                    explained_variance=expl + ev_jitter[b % 4],
                    ev_sum=expl + ev_jitter[b % 4] if is_on else None,
                    ev_main=0.03 if is_on else None,
                    ev_cf=0.10 if is_on else None,
                    ev_return_variance=5.0,
                    value_main_target_scale=1.7 if is_on else None,
                    cf_value_target_scale=6.3 if is_on else None,
                    cf_value_loss=0.02 if is_on else None,  # flat => plateaued by w=8
                    pre_norm_advantage_std=1.0,
                    return_std=3.0 + std_jitter[b % 4],
                    gradient_cv=0.10,
                    advantage_std_floored=False,
                    advantage_per_head_normalized=False,
                ))
            for episode_idx in range(n):
                conn.execute(
                    "INSERT INTO episode_outcomes VALUES (?,?,?,?,?,?,?,?)",
                    [run_dir, 0, episode_idx, 50.0, 1.1, 1, 1, 0],
                )
        pairs_spec.append({"seed": seed, "on_run_dir": f"/on/{seed}", "off_run_dir": f"/off/{seed}"})

    for run_dir, event_type in anomaly_rows or []:
        conn.execute("INSERT INTO anomalies VALUES (?, ?)", [run_dir, event_type])

    spec = {
        "pairs": pairs_spec,
        "host_params": 100_000,
        "n": n,
        "budget": 2,
        "thresholds": {
            "delta": 0.05, "eps_rel": 0.10, "tau_acc": 0.3,
            "delta_param_max": 1e9, "w": 8,
        },
        "g3_ratio_max": 1.5,
        "g4_ratio_max": 2.0,
        "g4_abs_floor": 5,
    }
    return conn, spec


def test_packet_from_spec_produces_a_scored_markdown_packet():
    conn, spec = _full_stage2_conn(5)
    packet = packet_from_spec(conn, spec)
    assert "# Stage-2 HRA MAJOR-1 acceptance verdict" in packet
    assert "Verdict:" in packet
    assert "INVALID" not in packet.splitlines()[2]  # a clean fixture scores, §1 passes
    assert "value_main_target_scale" in packet
    assert "cf_value_target_scale" in packet
    assert "G3 materiality threshold: ratio_max = 1.5" in packet
    assert "G3 churn rates ON:" in packet
    assert "G4 materiality thresholds: ratio_max = 2, abs_floor = 5" in packet
    assert "reward_hacking=0" in packet


def test_calibrate_from_spec_rejects_on_signature_run_in_off_dirs():
    conn, _spec = _full_stage2_conn(5)
    with pytest.raises(ValueError, match="OFF arm carries"):
        calibrate_from_spec(conn, {"off_run_dirs": ["/on/0"], "w": 0, "budget": 2})


def test_calibrate_from_spec_rejects_duplicate_off_dirs():
    conn, _spec = _full_stage2_conn(5)
    with pytest.raises(ValueError, match="duplicate OFF"):
        calibrate_from_spec(conn, {"off_run_dirs": ["/off/0", "/off/0"], "w": 0, "budget": 2})


def test_calibrate_from_spec_rejects_metadata_invalid_off_run():
    conn, _spec = _full_stage2_conn(5)
    conn.execute("UPDATE runs SET resume_path = '/tmp/checkpoint.pt' WHERE run_dir = '/off/0'")
    with pytest.raises(ValueError, match="fresh-init"):
        calibrate_from_spec(conn, {"off_run_dirs": ["/off/0"], "w": 0, "budget": 2})


def test_calibrate_from_spec_rejects_non_total_actor_source_off_run():
    conn, _spec = _full_stage2_conn(5)
    conn.execute("UPDATE runs SET actor_advantage_source = 'main_only' WHERE run_dir = '/off/0'")
    with pytest.raises(ValueError, match="actor_advantage_source"):
        calibrate_from_spec(
            conn,
            {"off_run_dirs": [f"/off/{seed}" for seed in range(5)], "w": 0, "budget": 2},
        )


def test_calibrate_from_spec_rejects_per_head_norm_off_run():
    conn, _spec = _full_stage2_conn(5)
    conn.execute(
        "UPDATE ppo_updates SET advantage_per_head_normalized = true WHERE run_dir = '/off/0'"
    )
    with pytest.raises(ValueError, match="per-head advantage normalization"):
        calibrate_from_spec(
            conn,
            {"off_run_dirs": [f"/off/{seed}" for seed in range(5)], "w": 0, "budget": 2},
        )


def test_calibrate_from_spec_rejects_duplicate_off_seed():
    conn, _spec = _full_stage2_conn(5)
    conn.execute("UPDATE runs SET seed = 0 WHERE run_dir = '/off/1'")
    with pytest.raises(ValueError, match="duplicate seed"):
        calibrate_from_spec(
            conn,
            {"off_run_dirs": [f"/off/{seed}" for seed in range(5)], "w": 0, "budget": 2},
        )


def test_calibrate_from_spec_rejects_reward_mode_drift_across_off_runs():
    conn, _spec = _full_stage2_conn(5)
    conn.execute("UPDATE runs SET reward_mode = 'SIMPLIFIED' WHERE run_dir = '/off/1'")
    with pytest.raises(ValueError, match="reward_mode mismatch"):
        calibrate_from_spec(
            conn,
            {"off_run_dirs": [f"/off/{seed}" for seed in range(5)], "w": 0, "budget": 2},
        )


def test_calibrate_from_spec_rejects_frozen_config_drift_across_off_runs():
    conn, _spec = _full_stage2_conn(5)
    conn.execute("UPDATE runs SET lr = 0.002 WHERE run_dir = '/off/1'")
    with pytest.raises(ValueError, match="frozen run config mismatch"):
        calibrate_from_spec(
            conn,
            {"off_run_dirs": [f"/off/{seed}" for seed in range(5)], "w": 0, "budget": 2},
        )


def test_calibrate_from_spec_rejects_incomplete_off_series_budget():
    conn, _spec = _full_stage2_conn(5)
    # The fixture emits 12 updates per run; a budget above that is an incomplete OFF series.
    with pytest.raises(ValueError, match="budget"):
        calibrate_from_spec(
            conn,
            {"off_run_dirs": [f"/off/{seed}" for seed in range(5)], "w": 0, "budget": 13},
        )


def test_packet_from_spec_material_guard_elevation_fails_g4():
    # 30 ON-leg governor rollbacks vs 0 OFF on seed 0 -> G4 hold fails for the pairset.
    conn, spec = _full_stage2_conn(5, anomaly_rows=[("/on/0", "GOVERNOR_ROLLBACK")] * 30)
    packet = packet_from_spec(conn, spec)
    assert "G4 guard channels within OFF baseline: False" in packet


def test_packet_from_spec_reward_hacking_suspicion_hard_fails_g4():
    conn, spec = _full_stage2_conn(5, anomaly_rows=[("/on/0", "REWARD_HACKING_SUSPECTED")])
    packet = packet_from_spec(conn, spec)
    assert "G4 guard channels within OFF baseline: False" in packet
    assert "reward_hacking=1" in packet


def test_packet_from_spec_rejects_duplicate_pairset_seed():
    conn, spec = _full_stage2_conn(5)
    spec["pairs"][1]["seed"] = 0
    conn.execute("UPDATE runs SET seed = 0 WHERE run_dir IN ('/on/1', '/off/1')")
    packet = packet_from_spec(conn, spec)
    assert "## §1 Validity — INVALID" in packet
    assert "duplicate seed" in packet


def test_packet_from_spec_rejects_host_params_spec_drift():
    conn, spec = _full_stage2_conn(5)
    spec["host_params"] = 123_456

    with pytest.raises(ValueError, match="host_params"):
        packet_from_spec(conn, spec)


def test_calibrate_from_spec_freezes_delta_from_off_arms_only():
    conn, _ = _full_stage2_conn(5)
    report = calibrate_from_spec(
        conn,
        {
            "off_run_dirs": [f"/off/{seed}" for seed in range(5)],
            "w": 0,
            "budget": 2,
        },
    )
    assert "delta" in report.lower() or "δ" in report
    assert "eps_rel" in report.lower() or "ε_rel" in report


# ---- placement provenance (2026-07-09 owner ruling: device is placement, not frozen config) ----


_PLACEMENT_CUDA0 = (("env_devices_json", '["cuda:0"]'), ("policy_device", "cuda:0"))
_PLACEMENT_CUDA1 = (("env_devices_json", '["cuda:1"]'), ("policy_device", "cuda:1"))


def test_off_calibration_accepts_cross_seed_placement_heterogeneity():
    metas = [
        dataclasses.replace(_meta("/off/41", 41), placement=_PLACEMENT_CUDA0),
        dataclasses.replace(_meta("/off/42", 42), placement=_PLACEMENT_CUDA1),
    ]
    assert off_calibration_validity_reasons(metas) == []


def test_pairset_accepts_cross_seed_placement_heterogeneity():
    pairings = [
        SeedPairing(
            on_meta=dataclasses.replace(_meta("/on/0", 0), placement=_PLACEMENT_CUDA0),
            off_meta=dataclasses.replace(_meta("/off/0", 0), placement=_PLACEMENT_CUDA0),
            host_params=100_000,
        ),
        SeedPairing(
            on_meta=dataclasses.replace(_meta("/on/1", 1), placement=_PLACEMENT_CUDA1),
            off_meta=dataclasses.replace(_meta("/off/1", 1), placement=_PLACEMENT_CUDA1),
            host_params=100_000,
        ),
    ]
    assert pairset_validity_reasons(pairings) == []


# ---- §6 outcome-contract bounds on terminal safety evidence ----


def test_read_run_val_acc_rejects_out_of_range_accuracy():
    for bad in (150.0, -3.0, float("nan")):
        rows = [_outcome_row(env_id=0, episode_idx=0, final_accuracy=bad)]
        with pytest.raises(ValueError, match="outside"):
            read_run_val_acc(_conn_with_outcomes(rows), "/run")


def test_read_run_added_params_rejects_impossible_param_ratio():
    for bad in (0.8, float("nan"), float("inf")):
        rows = [_outcome_row(env_id=0, episode_idx=0, param_ratio=bad)]
        with pytest.raises(ValueError, match="outside"):
            read_run_added_params(_conn_with_outcomes(rows), "/run", host_params=100_000)


def test_terminal_outcome_bounds_accept_boundary_values():
    rows = [
        _outcome_row(env_id=0, episode_idx=0, final_accuracy=0.0),
        _outcome_row(env_id=1, episode_idx=1, final_accuracy=100.0),
    ]
    assert read_run_val_acc(_conn_with_outcomes(rows), "/run") == pytest.approx(50.0)
    rows = [_outcome_row(env_id=0, episode_idx=0, param_ratio=1.0)]
    assert read_run_added_params(
        _conn_with_outcomes(rows), "/run", host_params=100_000
    ) == pytest.approx(0.0)


def test_outcome_bounds_apply_per_env_not_to_the_mean():
    # Two insane values that average to a plausible mean must still be rejected.
    rows = [
        _outcome_row(env_id=0, episode_idx=0, final_accuracy=-40.0),
        _outcome_row(env_id=1, episode_idx=1, final_accuracy=140.0),
    ]
    with pytest.raises(ValueError, match="outside"):
        read_run_val_acc(_conn_with_outcomes(rows), "/run")


def test_terminal_reader_rejects_duplicate_terminal_rows():
    # Two rows at the same (env_id, max episode_idx): ROW_NUMBER would pick one
    # nondeterministically — fail loud instead of letting G1/G2 vary between invocations.
    rows = [
        _outcome_row(env_id=0, episode_idx=0, final_accuracy=60.0),
        _outcome_row(env_id=0, episode_idx=0, final_accuracy=80.0),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        read_run_val_acc(_conn_with_outcomes(rows), "/run")


def test_terminal_reader_rejects_null_env_id():
    # A parseable EPISODE_OUTCOME missing env_id yields SQL NULL — reject with the
    # contract-named ValueError, not an unattributed TypeError.
    rows = [_outcome_row(env_id=None, episode_idx=0, final_accuracy=50.0)]
    with pytest.raises(ValueError, match="env_id"):
        read_run_val_acc(_conn_with_outcomes(rows), "/run")


def test_env_coverage_reports_null_env_id_as_reason_not_crash():
    # A NULL env_id row reaches the coverage pass FIRST (it runs before the terminal
    # reader's guard) — it must land in the reasons list, not abort build_report with
    # an unattributed TypeError.
    conn = _conn_with_runs_and_outcomes(
        runs=[{"run_dir": "/on", "n_envs": 3}, {"run_dir": "/off", "n_envs": 3}],
        outcomes=_terminal_outcomes("/on", [0, 1, 2])
        + [{"run_dir": "/on", "env_id": None, "episode_idx": 5}]
        + _terminal_outcomes("/off", [0, 1, 2]),
    )
    reasons = env_count_completeness_reasons(conn, on_run_dir="/on", off_run_dir="/off")
    assert any("ON" in r and "env_id" in r and "NULL" in r for r in reasons)


def test_env_coverage_flags_duplicate_outcome_rows():
    # Per-env outcome count must equal n_episodes // n_envs — a duplicate row silently
    # re-weights churn and marks a re-emission bug (SIMIC-PROD-001 class).
    conn = _conn_with_runs_and_outcomes(
        runs=[{"run_dir": "/on", "n_envs": 3}, {"run_dir": "/off", "n_envs": 3}],
        outcomes=_terminal_outcomes("/on", [0, 1, 2])
        + [{"run_dir": "/on", "env_id": 0, "episode_idx": 0}]  # duplicate for env 0
        + _terminal_outcomes("/off", [0, 1, 2]),
    )
    reasons = env_count_completeness_reasons(conn, on_run_dir="/on", off_run_dir="/off")
    assert any("ON" in r and ("row count" in r or "duplicate" in r) for r in reasons)
