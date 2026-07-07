"""Tests for the Stage-2 MAJOR-1 acceptance wrapper's telemetry I/O boundary.

The dict->UpdateRow conversion happens HERE and fails loud on a missing column (schema drift) —
the pure layer never touches raw dicts. read_leg is exercised against an in-memory duckdb
``ppo_updates`` table with the real column names (verified via describe_view), so duckdb's actual
NULL / DOUBLE marshalling is on the test path; the JSONL->view extraction is Karn's own tested
concern and is exercised end-to-end at the CLI slice.
"""

import duckdb
import pytest

from esper.simic.telemetry.stage2_acceptance import GateResult, Verdict
from esper.simic.telemetry.stage2_acceptance_io import (
    BuiltReport,
    ChurnRates,
    SeedPairing,
    build_report,
    env_count_completeness_reasons,
    read_leg,
    read_run_added_params,
    read_run_churn,
    read_run_n_envs,
    read_run_uses_per_head_norm,
    read_run_val_acc,
    row_to_update,
)
from esper.simic.telemetry.stage2_acceptance_packet import (
    ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
    FrozenThresholds,
    RunMeta,
    UpdateRow,
    render_packet,
)


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
        ev_return_variance=5.0,
        pre_norm_advantage_std=1.0,
        return_std=3.0,
        advantage_per_head_normalized=False,
    )
    row.update(overrides)
    return row


# ---- row_to_update: the dict -> typed boundary ----


def test_row_to_update_maps_all_fields():
    row = _full_row(
        inner_epoch=2, batch=3, explained_variance=0.7, ev_sum=0.7, ev_main=0.2, ev_cf=0.1,
        ev_return_variance=8.0, pre_norm_advantage_std=1.5, return_std=2.0,
    )
    assert row_to_update(row) == UpdateRow(
        inner_epoch=2, batch=3, explained_variance=0.7, ev_sum=0.7, ev_main=0.2, ev_cf=0.1,
        ev_return_variance=8.0, pre_norm_advantage_std=1.5, return_std=2.0,
    )


def test_row_to_update_preserves_null_per_stream_ev_on_off_leg():
    u = row_to_update(_full_row(ev_sum=None, ev_main=None, ev_cf=None))
    assert (u.ev_sum, u.ev_main, u.ev_cf) == (None, None, None)


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
    conn.execute(
        """
        CREATE TABLE ppo_updates (
            run_dir VARCHAR, inner_epoch INTEGER, batch INTEGER, explained_variance DOUBLE,
            ev_sum DOUBLE, ev_main DOUBLE, ev_cf DOUBLE, ev_return_variance DOUBLE,
            pre_norm_advantage_std DOUBLE, return_std DOUBLE, advantage_per_head_normalized BOOLEAN
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO ppo_updates VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                r["run_dir"], r["inner_epoch"], r["batch"], r["explained_variance"], r["ev_sum"],
                r["ev_main"], r["ev_cf"], r["ev_return_variance"], r["pre_norm_advantage_std"],
                r["return_std"], r["advantage_per_head_normalized"],
            ],
        )
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
    conn = _conn_with_updates([_full_row(run_dir="/A", ev_sum=0.8, ev_main=0.3, ev_cf=0.1)])
    u = read_leg(conn, "/A")[0]
    assert (u.ev_sum, u.ev_main, u.ev_cf) == (0.8, 0.3, 0.1)


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
    conn.execute("CREATE TABLE runs (run_dir VARCHAR, n_envs INTEGER)")
    for r in runs:
        conn.execute("INSERT INTO runs VALUES (?, ?)", [r["run_dir"], r["n_envs"]])
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


# ---- S6: build_report — the scoring-phase assembly (telemetry -> scored verdict) ----
#
# build_report wires read_leg -> leg_series -> validate_pair (+ env-count reasons merged into the
# Validity) -> score, with caller-supplied RunMeta (so it reads no not-yet-emitted column) and
# already-FROZEN thresholds (it never calibrates — "no peeking" is mechanical). g3/g4 holds are
# injected (the G4 reader + churn threshold are S7/§6-definitional).


_IO_THRESHOLDS = FrozenThresholds(delta=0.05, eps_rel=0.10, tau_acc=0.3, delta_param_max=1e9, w=0)


def _build_conn(specs: list[dict]) -> duckdb.DuckDBPyConnection:
    """Full ppo_updates + episode_outcomes + runs fixture. Each spec is one run (ON or OFF)."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE ppo_updates (
            run_dir VARCHAR, inner_epoch INTEGER, batch INTEGER, explained_variance DOUBLE,
            ev_sum DOUBLE, ev_main DOUBLE, ev_cf DOUBLE, ev_return_variance DOUBLE,
            pre_norm_advantage_std DOUBLE, return_std DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE episode_outcomes (
            run_dir VARCHAR, env_id INTEGER, episode_idx INTEGER, final_accuracy DOUBLE,
            param_ratio DOUBLE, germinate_count INTEGER, prune_count INTEGER, fossilize_count INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE runs (run_dir VARCHAR, n_envs INTEGER)")
    # Median-preserving jitter over 4 updates so every leg has NONZERO within-run IQR — a
    # constant-EV leg is degenerate (IQR=0) and the scorer rightly rejects it; real telemetry varies.
    ev_jitter = [-0.02, 0.0, 0.02, 0.0]
    std_jitter = [-0.5, 0.0, 0.5, 0.0]
    for s in specs:
        conn.execute("INSERT INTO runs VALUES (?, ?)", [s["run_dir"], s["n_envs"]])
        is_on = s["leg"] == "on"
        for b in range(4):
            expl_b = s["expl"] + ev_jitter[b]
            return_std_b = 3.0 + std_jitter[b]
            conn.execute(
                "INSERT INTO ppo_updates VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    s["run_dir"], 0, b, expl_b,
                    expl_b if is_on else None,       # ev_sum: total-EV comparand on ON, null on OFF
                    0.03 if is_on else None,         # ev_main (feeds §5 MECH), null on OFF
                    0.10 if is_on else None,         # ev_cf, null on OFF
                    5.0, 1.0, return_std_b,          # ev_return_variance (>floor), adv_std, return_std
                ],
            )
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


def test_build_report_wires_paired_safety_g1_regression_to_reject():
    # ON val-acc 10pp below OFF on every seed -> paired Δ_G1 median = -10 < -τ_acc -> G1 FAIL -> REJECT.
    conn, pairings = _pairset(10, on_final=40.0, off_final=50.0)
    built = build_report(conn, pairings, _IO_THRESHOLDS, n=10, budget=2, g3_hold=True, g4_hold=True)
    assert built.report.g1 is GateResult.FAIL
    assert built.report.verdict is Verdict.REJECT
