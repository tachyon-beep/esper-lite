"""Tests for the Stage-2 MAJOR-1 acceptance wrapper's telemetry I/O boundary.

The dict->UpdateRow conversion happens HERE and fails loud on a missing column (schema drift) —
the pure layer never touches raw dicts. read_leg is exercised against an in-memory duckdb
``ppo_updates`` table with the real column names (verified via describe_view), so duckdb's actual
NULL / DOUBLE marshalling is on the test path; the JSONL->view extraction is Karn's own tested
concern and is exercised end-to-end at the CLI slice.
"""

import duckdb
import pytest

from esper.simic.telemetry.stage2_acceptance_io import (
    ChurnRates,
    read_leg,
    read_run_added_params,
    read_run_churn,
    read_run_val_acc,
    row_to_update,
)
from esper.simic.telemetry.stage2_acceptance_packet import UpdateRow


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
            pre_norm_advantage_std DOUBLE, return_std DOUBLE
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO ppo_updates VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                r["run_dir"], r["inner_epoch"], r["batch"], r["explained_variance"], r["ev_sum"],
                r["ev_main"], r["ev_cf"], r["ev_return_variance"], r["pre_norm_advantage_std"],
                r["return_std"],
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
