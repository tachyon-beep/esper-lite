from __future__ import annotations

from pathlib import Path

from esper.utils.weft_parity import (
    LoomweaveCapabilities,
    build_phase_a_report,
    inspect_loomweave_db,
    parse_defensive_output,
    parse_leyline_output,
)


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


def test_build_phase_a_report_blocks_leyline_retirement_without_class_kind_metadata(
    tmp_path: Path,
) -> None:
    wardline_output = tmp_path / "wardline.jsonl"
    wardline_output.write_text("")
    capabilities = LoomweaveCapabilities(
        db_path=tmp_path / "loomweave.db",
        database_present=True,
        class_contract_kind_present=False,
        module_import_edges_present=True,
        module_cycles=[],
    )

    report = build_phase_a_report(
        wardline_output=wardline_output,
        loomweave=capabilities,
        defensive_findings=[],
        leyline_findings=[],
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["leyline-types"]["replacement_ready"] is False
    assert checks["leyline-types"]["blockers"][:1] == [
        "Loomweave class_contract_kind metadata is not present; keep lint_leyline_types.py blocking."
    ]
    assert checks["leyline-types"]["retirement_gate"]["satisfied"] is False


def test_build_phase_a_report_marks_defensive_patterns_unready_without_rule_mapping(
    tmp_path: Path,
) -> None:
    wardline_output = tmp_path / "wardline.jsonl"
    wardline_output.write_text("")
    capabilities = LoomweaveCapabilities(
        db_path=tmp_path / "loomweave.db",
        database_present=True,
        class_contract_kind_present=True,
        module_import_edges_present=True,
        module_cycles=[],
    )

    report = build_phase_a_report(
        wardline_output=wardline_output,
        loomweave=capabilities,
        defensive_findings=[],
        leyline_findings=[],
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["replacement_ready"] is False
    assert checks["defensive-patterns"]["blockers"][:1] == [
        "Wardline emitted no defensive-pattern rule mapping; keep lint_defensive_patterns.py blocking."
    ]
    assert checks["defensive-patterns"]["retirement_gate"]["satisfied"] is False


def test_build_phase_a_report_never_marks_replacement_ready_during_shadow_burn_in(
    tmp_path: Path,
) -> None:
    wardline_output = tmp_path / "wardline.jsonl"
    wardline_output.write_text('{"rule_id":"WLN-DEFENSIVE-PATTERN"}\n')
    capabilities = LoomweaveCapabilities(
        db_path=tmp_path / "loomweave.db",
        database_present=True,
        class_contract_kind_present=True,
        module_import_edges_present=True,
        module_cycles=[],
    )

    report = build_phase_a_report(
        wardline_output=wardline_output,
        loomweave=capabilities,
        defensive_findings=[],
        leyline_findings=[],
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["shadow_signal_ready"] is True
    assert checks["leyline-types"]["shadow_signal_ready"] is True
    assert checks["module-cycles"]["shadow_signal_ready"] is True
    assert checks["defensive-patterns"]["replacement_ready"] is False
    assert checks["leyline-types"]["replacement_ready"] is False
    assert checks["module-cycles"]["replacement_ready"] is False


def test_build_phase_a_report_flags_missing_wardline_artifact(tmp_path: Path) -> None:
    wardline_output = tmp_path / "missing-wardline.jsonl"
    capabilities = LoomweaveCapabilities(
        db_path=tmp_path / "loomweave.db",
        database_present=True,
        class_contract_kind_present=True,
        module_import_edges_present=True,
        module_cycles=[],
    )

    report = build_phase_a_report(
        wardline_output=wardline_output,
        loomweave=capabilities,
        defensive_findings=[],
        leyline_findings=[],
    )

    checks = {check["check"]: check for check in report["checks"]}
    assert checks["defensive-patterns"]["blockers"][0] == (
        "Wardline artifact is missing; inspect the Wardline shadow step before evaluating defensive-pattern parity."
    )


def test_inspect_loomweave_db_reports_import_edges_and_missing_class_kind(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "loomweave.db"
    import sqlite3

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
        insert into entities values
            ('python:module:a', 'module', 'a', '{}'),
            ('python:module:b', 'module', 'b', '{}'),
            ('python:class:a.Payload', 'class', 'a.Payload', '{}');
        insert into edges values
            ('imports', 'python:module:a', 'python:module:b');
        """
    )
    connection.close()

    capabilities = inspect_loomweave_db(db_path)

    assert capabilities.database_present is True
    assert capabilities.module_import_edges_present is True
    assert capabilities.class_contract_kind_present is False
    assert capabilities.module_cycles == []
