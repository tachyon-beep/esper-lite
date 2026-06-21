from __future__ import annotations

import sqlite3
from pathlib import Path

from esper.utils.weft_parity import (
    LoomweaveCapabilities,
    NormalizedFinding,
    _module_cycles,
    build_phase_a_report,
    inspect_loomweave_db,
    parse_defensive_output,
    parse_leyline_output,
)

HEAD = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _capabilities(
    *,
    database_present: bool = True,
    module_import_edges_present: bool = True,
    module_cycles: list[list[str]] | None = None,
    analyzed_at_commit: str | None = HEAD,
    db_path: Path | None = None,
) -> LoomweaveCapabilities:
    return LoomweaveCapabilities(
        db_path=db_path or Path("loomweave.db"),
        database_present=database_present,
        module_import_edges_present=module_import_edges_present,
        module_cycles=module_cycles or [],
        analyzed_at_commit=analyzed_at_commit,
    )


def _report(
    *,
    wardline_output: Path,
    loomweave: LoomweaveCapabilities,
    head_commit: str = HEAD,
    defensive_findings: list[NormalizedFinding] | None = None,
    defensive_exit_code: int = 0,
    leyline_findings: list[NormalizedFinding] | None = None,
    leyline_exit_code: int = 0,
) -> dict:
    return build_phase_a_report(
        wardline_output=wardline_output,
        loomweave=loomweave,
        head_commit=head_commit,
        defensive_findings=defensive_findings or [],
        defensive_exit_code=defensive_exit_code,
        leyline_findings=leyline_findings or [],
        leyline_exit_code=leyline_exit_code,
    )


def _wardline_with_defensive_rule(tmp_path: Path) -> Path:
    wardline_output = tmp_path / "wardline.jsonl"
    wardline_output.write_text('{"rule_id":"WLN-DEFENSIVE-PATTERN"}\n')
    return wardline_output


def _make_loomweave_db(
    db_path: Path,
    *,
    import_edges: list[tuple[str, str]],
    analyzed_commit: str | None,
    status: str = "completed",
) -> None:
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        create table entities (
            id text primary key,
            kind text not null,
            name text not null,
            properties text not null
        );
        create table edges (
            kind text not null,
            from_id text not null,
            to_id text not null
        );
        create table runs (
            id text primary key,
            started_at text not null,
            completed_at text,
            config text not null,
            stats text not null,
            status text not null,
            analyzed_at_commit text
        );
        """
    )
    nodes = sorted({node for edge in import_edges for node in edge})
    for node in nodes:
        connection.execute(
            "insert into entities values (?, 'module', ?, '{}')",
            (f"python:module:{node}", node),
        )
    for source, target in import_edges:
        connection.execute(
            "insert into edges values ('imports', ?, ?)",
            (f"python:module:{source}", f"python:module:{target}"),
        )
    connection.execute(
        "insert into runs (id, started_at, completed_at, config, stats, status, "
        "analyzed_at_commit) values (?, ?, ?, ?, ?, ?, ?)",
        ("run1", "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "{}", "{}", status, analyzed_commit),
    )
    connection.commit()
    connection.close()


# --------------------------------------------------------------------------- #
# Homegrown lint output parsers
# --------------------------------------------------------------------------- #


def test_parse_leyline_output_records_stale_whitelist_entries() -> None:
    output = """
Checked 163 files, 148 type definitions
Stale whitelist entries: 2

STALE WHITELIST ENTRIES FOUND:

  src/esper/simic/rewards/reward_telemetry.py:dataclass:RewardComponentsTelemetry
  src/esper/simic/telemetry/observation_stats.py:dataclass:ObservationStatsTelemetry

To fix: remove or update these keys in leyline_boundaries.yaml
"""

    findings = parse_leyline_output(output)

    assert [finding.key for finding in findings] == [
        "src/esper/simic/rewards/reward_telemetry.py:dataclass:RewardComponentsTelemetry",
        "src/esper/simic/telemetry/observation_stats.py:dataclass:ObservationStatsTelemetry",
    ]
    assert findings[0].check == "leyline-types"
    assert findings[0].rule == "stale-whitelist"
    assert findings[0].file == "src/esper/simic/rewards/reward_telemetry.py"


def test_parse_leyline_output_records_active_type_boundary_violations() -> None:
    output = """
3 violation(s) found:

  ERROR: src/esper/simic/agent/rollout_buffer.py:51: dataclass 'RollbackPenaltyResult' not in leyline or whitelist

To fix: either move the type to leyline/, or add it to leyline_boundaries.yaml
"""

    findings = parse_leyline_output(output)

    assert [finding.key for finding in findings] == [
        "src/esper/simic/agent/rollout_buffer.py:dataclass:RollbackPenaltyResult"
    ]
    assert findings[0].line == 51
    assert findings[0].rule == "dataclass"
    assert findings[0].message == (
        "dataclass RollbackPenaltyResult not in leyline or whitelist"
    )


def test_parse_defensive_output_records_violation_keys() -> None:
    output = """
1 VIOLATION(S) FOUND:

  src/esper/foo.py:12: getattr in load_config()
    Code: getattr(config, "missing")
    Key:  src/esper/foo.py:load_config:getattr

To fix:
  1. PREFERRED: Remove the defensive pattern and fix the underlying issue
"""

    findings = parse_defensive_output(output)

    assert [finding.key for finding in findings] == [
        "src/esper/foo.py:load_config:getattr"
    ]
    assert findings[0].check == "defensive-patterns"
    assert findings[0].file == "src/esper/foo.py"
    assert findings[0].line == 12
    assert findings[0].rule == "getattr"


def test_parse_defensive_output_records_stale_whitelist_entries() -> None:
    output = """
STALE WHITELIST ENTRIES FOUND:

  src/esper/foo.py:load_config:getattr

To fix: remove or update these keys in defensive_patterns.yaml
"""

    findings = parse_defensive_output(output)

    assert [finding.key for finding in findings] == [
        "src/esper/foo.py:load_config:getattr"
    ]
    assert findings[0].file == "src/esper/foo.py"
    assert findings[0].rule == "stale-whitelist"


# --------------------------------------------------------------------------- #
# Phase A report: burn-in safety invariant
# --------------------------------------------------------------------------- #


def test_build_phase_a_report_never_marks_replacement_ready_during_shadow_burn_in(
    tmp_path: Path,
) -> None:
    wardline_output = _wardline_with_defensive_rule(tmp_path)

    report = _report(wardline_output=wardline_output, loomweave=_capabilities())

    checks = {check["check"]: check for check in report["checks"]}
    # replacement_ready is hard-False for every check throughout Phase A burn-in.
    assert checks["defensive-patterns"]["replacement_ready"] is False
    assert checks["leyline-types"]["replacement_ready"] is False
    assert checks["module-cycles"]["replacement_ready"] is False
    assert all(
        check["retirement_gate"]["satisfied"] is False for check in report["checks"]
    )
    # shadow_signal_ready is exposed separately and CAN be True for checks with a
    # Weft signal (defensive rule present, fresh module-cycle index)...
    assert checks["defensive-patterns"]["shadow_signal_ready"] is True
    assert checks["module-cycles"]["shadow_signal_ready"] is True
    # ...but leyline has no Weft equivalent yet, so it is never shadow-ready.
    assert checks["leyline-types"]["shadow_signal_ready"] is False


def test_leyline_types_is_always_blocked_under_pinned_loomweave(tmp_path: Path) -> None:
    """D3: leyline readiness must never flip ready — Loomweave exposes no per-kind
    class-contract metadata nor stale-whitelist semantics."""
    wardline_output = _wardline_with_defensive_rule(tmp_path)

    report = _report(wardline_output=wardline_output, loomweave=_capabilities())

    checks = {check["check"]: check for check in report["checks"]}
    leyline = checks["leyline-types"]
    assert leyline["shadow_signal_ready"] is False
    assert leyline["blockers"][0] == (
        "Loomweave exposes no per-kind class-contract metadata "
        "(enum/dataclass/protocol/typeddict/namedtuple) and cannot model "
        "leyline_boundaries stale-whitelist semantics; keep lint_leyline_types.py blocking."
    )


def test_defensive_patterns_unready_without_wardline_rule(tmp_path: Path) -> None:
    wardline_output = tmp_path / "wardline.jsonl"
    wardline_output.write_text("")

    report = _report(wardline_output=wardline_output, loomweave=_capabilities())

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["shadow_signal_ready"] is False
    assert checks["defensive-patterns"]["blockers"][0] == (
        "Wardline emitted no defensive-pattern rule mapping; keep lint_defensive_patterns.py blocking."
    )


def test_build_phase_a_report_flags_missing_wardline_artifact(tmp_path: Path) -> None:
    """D2: a genuinely absent wardline artifact surfaces the 'artifact is missing'
    blocker, not the misleading 'no rule mapping' blocker."""
    wardline_output = tmp_path / "missing-wardline.jsonl"

    report = _report(wardline_output=wardline_output, loomweave=_capabilities())

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["blockers"][0] == (
        "Wardline artifact is missing; inspect the Wardline shadow step before evaluating defensive-pattern parity."
    )


# --------------------------------------------------------------------------- #
# D1: a crashed homegrown linter must not read as "zero findings / ready"
# --------------------------------------------------------------------------- #


def test_crashed_defensive_linter_blocks_readiness(tmp_path: Path) -> None:
    wardline_output = _wardline_with_defensive_rule(tmp_path)

    report = _report(
        wardline_output=wardline_output,
        loomweave=_capabilities(),
        defensive_findings=[],
        defensive_exit_code=1,
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["shadow_signal_ready"] is False
    assert any(
        "lint_defensive_patterns.py exited non-zero with no parsed findings" in blocker
        for blocker in checks["defensive-patterns"]["blockers"]
    )


def test_crashed_leyline_linter_adds_failed_run_blocker(tmp_path: Path) -> None:
    wardline_output = _wardline_with_defensive_rule(tmp_path)

    report = _report(
        wardline_output=wardline_output,
        loomweave=_capabilities(),
        leyline_findings=[],
        leyline_exit_code=2,
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert any(
        "lint_leyline_types.py exited non-zero with no parsed findings" in blocker
        for blocker in checks["leyline-types"]["blockers"]
    )


def test_nonzero_exit_with_findings_does_not_add_crash_blocker(tmp_path: Path) -> None:
    """A linter that found violations also exits non-zero, but it emits parseable
    findings which already drive not-ready — so no crash blocker is added and the
    findings are preserved as homegrown_only."""
    wardline_output = _wardline_with_defensive_rule(tmp_path)
    finding = NormalizedFinding(
        check="defensive-patterns",
        file="src/esper/foo.py",
        line=12,
        rule="getattr",
        key="src/esper/foo.py:load_config:getattr",
        source="homegrown",
        message="getattr in load_config()",
    )

    report = _report(
        wardline_output=wardline_output,
        loomweave=_capabilities(),
        defensive_findings=[finding],
        defensive_exit_code=1,
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert not any(
        "exited non-zero with no parsed findings" in blocker
        for blocker in checks["defensive-patterns"]["blockers"]
    )
    assert len(checks["defensive-patterns"]["homegrown_only"]) == 1
    assert checks["defensive-patterns"]["shadow_signal_ready"] is False


# --------------------------------------------------------------------------- #
# D7: stale Loomweave index must block module-cycle evidence
# --------------------------------------------------------------------------- #


def test_module_cycles_blocked_when_index_stale(tmp_path: Path) -> None:
    wardline_output = _wardline_with_defensive_rule(tmp_path)

    report = _report(
        wardline_output=wardline_output,
        loomweave=_capabilities(analyzed_at_commit="0000000000000000000000000000000000000000"),
        head_commit=HEAD,
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["module-cycles"]["shadow_signal_ready"] is False
    assert any(
        "re-run loomweave analyze before trusting module-cycle evidence" in blocker
        for blocker in checks["module-cycles"]["blockers"]
    )


def test_module_cycles_ready_when_index_fresh(tmp_path: Path) -> None:
    wardline_output = _wardline_with_defensive_rule(tmp_path)

    report = _report(
        wardline_output=wardline_output,
        loomweave=_capabilities(analyzed_at_commit=HEAD),
        head_commit=HEAD,
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["module-cycles"]["shadow_signal_ready"] is True
    assert not any(
        "re-run loomweave analyze" in blocker
        for blocker in checks["module-cycles"]["blockers"]
    )


# --------------------------------------------------------------------------- #
# D4: the report must not imply a homegrown-vs-weft comparison it never performs
# --------------------------------------------------------------------------- #


def test_checks_carry_comparison_discriminator(tmp_path: Path) -> None:
    wardline_output = _wardline_with_defensive_rule(tmp_path)

    report = _report(wardline_output=wardline_output, loomweave=_capabilities())

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["comparison"] == "deferred"
    assert checks["leyline-types"]["comparison"] == "deferred"
    assert checks["module-cycles"]["comparison"] == "weft_native"


# --------------------------------------------------------------------------- #
# Loomweave DB introspection (D6 missing-db, D7 freshness marker)
# --------------------------------------------------------------------------- #


def test_inspect_loomweave_db_reports_import_edges_and_analyzed_commit(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "loomweave.db"
    _make_loomweave_db(
        db_path,
        import_edges=[("a", "b")],
        analyzed_commit="abc123",
    )

    capabilities = inspect_loomweave_db(db_path)

    assert capabilities.database_present is True
    assert capabilities.module_import_edges_present is True
    assert capabilities.module_cycles == []
    assert capabilities.analyzed_at_commit == "abc123"


def test_inspect_loomweave_db_detects_module_cycle(tmp_path: Path) -> None:
    db_path = tmp_path / "loomweave.db"
    _make_loomweave_db(
        db_path,
        import_edges=[("a", "b"), ("b", "a")],
        analyzed_commit="abc123",
    )

    capabilities = inspect_loomweave_db(db_path)

    assert capabilities.module_cycles == [["a", "b"]]


def test_inspect_loomweave_db_missing_database(tmp_path: Path) -> None:
    capabilities = inspect_loomweave_db(tmp_path / "nonexistent.db")

    assert capabilities.database_present is False
    assert capabilities.module_import_edges_present is False
    assert capabilities.module_cycles == []
    assert capabilities.analyzed_at_commit is None


# --------------------------------------------------------------------------- #
# D5/D9: Tarjan SCC module-cycle detector
# --------------------------------------------------------------------------- #


def test_module_cycles_detects_simple_cycle() -> None:
    assert _module_cycles([("a", "b"), ("b", "c"), ("c", "a")]) == [["a", "b", "c"]]


def test_module_cycles_detects_self_loop() -> None:
    """D9: a module importing itself is a one-node cycle and must be reported."""
    assert _module_cycles([("a", "a")]) == [["a"]]


def test_module_cycles_separates_disjoint_cycles() -> None:
    assert _module_cycles(
        [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]
    ) == [["a", "b"], ["c", "d"]]


def test_module_cycles_ignores_acyclic_graph() -> None:
    assert _module_cycles([("a", "b"), ("b", "c")]) == []


def test_module_cycles_mixes_self_loop_and_multi_node_cycle() -> None:
    assert _module_cycles([("a", "a"), ("b", "c"), ("c", "b")]) == [["a"], ["b", "c"]]


def test_module_cycles_handles_self_edge_inside_multi_node_cycle() -> None:
    """A self-edge on a node already inside a larger SCC must fold into that SCC,
    not spawn a spurious one-node component, and must not raise (graph[node] exists)."""
    assert _module_cycles([("a", "a"), ("a", "b"), ("b", "a")]) == [["a", "b"]]


def test_module_cycles_output_is_deterministically_sorted() -> None:
    forward = _module_cycles([("a", "b"), ("b", "c"), ("c", "a")])
    shuffled = _module_cycles([("c", "a"), ("a", "b"), ("b", "c")])
    assert forward == shuffled == [["a", "b", "c"]]


# --------------------------------------------------------------------------- #
# D2 (driver-level): ci_weft_parity must not pre-create an empty wardline file
# --------------------------------------------------------------------------- #


def test_ci_driver_reports_missing_wardline_without_creating_file(
    tmp_path: Path,
) -> None:
    """Regression guard for the removed empty-file shim: a genuinely absent wardline
    artifact must surface the 'artifact is missing' blocker and must NOT be created."""
    import json
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "parity.json"
    missing_wardline = tmp_path / "nonexistent-wardline.jsonl"
    missing_db = tmp_path / "nonexistent-loomweave.db"

    subprocess.run(
        [
            sys.executable,
            "scripts/ci_weft_parity.py",
            "--output",
            str(output),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--wardline-output",
            str(missing_wardline),
            "--loomweave-db",
            str(missing_db),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not missing_wardline.exists()
    report = json.loads(output.read_text())
    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["blockers"][0] == (
        "Wardline artifact is missing; inspect the Wardline shadow step before evaluating defensive-pattern parity."
    )
