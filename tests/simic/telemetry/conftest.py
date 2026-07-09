"""Shared duckdb fixture plumbing for the Stage-2 acceptance telemetry tests.

The ``ppo_updates`` CREATE TABLE / positional INSERT lists used to be hand-copied per
fixture and had already drifted (one copy omitted ``advantage_per_head_normalized``;
column orders differed). Both are now driven by the PRODUCTION read order
(``_PPO_UPDATE_COLUMNS`` — a private name, but this IS that module's test suite), so a
schema change lands in exactly one place and a value can never silently map into an
adjacent DOUBLE column.
"""

import duckdb

from esper.simic.telemetry.stage2_acceptance_io import _PPO_UPDATE_COLUMNS

# advantage_per_head_normalized is not in read_leg's SELECT list — it is read separately
# by read_run_uses_per_head_norm (§8F(i)) — so it rides along as an extra table column.
_EXTRA_COLUMNS: tuple[str, ...] = ("advantage_per_head_normalized",)

_PPO_UPDATES_TABLE_COLUMNS: tuple[str, ...] = ("run_dir", *_PPO_UPDATE_COLUMNS, *_EXTRA_COLUMNS)

_INTEGER_COLUMNS = frozenset({"inner_epoch", "batch"})
_BOOLEAN_COLUMNS = frozenset({"advantage_std_floored", "advantage_per_head_normalized"})


def _column_type(column: str) -> str:
    if column == "run_dir":
        return "VARCHAR"
    if column in _INTEGER_COLUMNS:
        return "INTEGER"
    if column in _BOOLEAN_COLUMNS:
        return "BOOLEAN"
    return "DOUBLE"


def create_ppo_updates_table(
    conn: duckdb.DuckDBPyConnection,
    *,
    extra_columns: tuple[str, ...] = _EXTRA_COLUMNS,
) -> None:
    """CREATE ppo_updates with the production column order (run_dir key + extras appended)."""
    columns = ("run_dir", *_PPO_UPDATE_COLUMNS, *extra_columns)
    column_sql = ", ".join(f"{column} {_column_type(column)}" for column in columns)
    conn.execute(f"CREATE TABLE ppo_updates ({column_sql})")


def insert_ppo_update(conn: duckdb.DuckDBPyConnection, row: dict) -> None:
    """INSERT one ppo_updates row by column name (KeyError on a missing key = fail loud)."""
    placeholders = ", ".join("?" for _ in _PPO_UPDATES_TABLE_COLUMNS)
    conn.execute(
        f"INSERT INTO ppo_updates VALUES ({placeholders})",
        [row[column] for column in _PPO_UPDATES_TABLE_COLUMNS],
    )
