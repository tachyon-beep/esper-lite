from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


class NormalizedFinding:
    def __init__(
        self,
        *,
        check: str,
        file: str,
        line: int | None,
        rule: str,
        key: str,
        source: str,
        message: str,
    ) -> None:
        self.check = check
        self.file = file
        self.line = line
        self.rule = rule
        self.key = key
        self.source = source
        self.message = message


class LoomweaveCapabilities:
    def __init__(
        self,
        *,
        db_path: Path,
        database_present: bool,
        class_contract_kind_present: bool,
        module_import_edges_present: bool,
        module_cycles: list[list[str]],
    ) -> None:
        self.db_path = db_path
        self.database_present = database_present
        self.class_contract_kind_present = class_contract_kind_present
        self.module_import_edges_present = module_import_edges_present
        self.module_cycles = module_cycles


def inspect_loomweave_db(db_path: Path) -> LoomweaveCapabilities:
    if not db_path.exists():
        return LoomweaveCapabilities(
            db_path=db_path,
            database_present=False,
            class_contract_kind_present=False,
            module_import_edges_present=False,
            module_cycles=[],
        )

    connection = sqlite3.connect(db_path)
    class_kind_count = connection.execute(
        """
        select count(*)
        from entities
        where kind = 'class'
          and json_extract(properties, '$.python.class_contract_kind') is not null
        """
    ).fetchone()[0]
    import_edge_count = connection.execute(
        """
        select count(*)
        from edges
        where kind = 'imports'
        """
    ).fetchone()[0]
    import_edges = connection.execute(
        """
        select source.name, target.name
        from edges
        join entities source on source.id = edges.from_id
        join entities target on target.id = edges.to_id
        where edges.kind = 'imports'
          and source.kind = 'module'
          and target.kind = 'module'
        """
    ).fetchall()
    connection.close()

    return LoomweaveCapabilities(
        db_path=db_path,
        database_present=True,
        class_contract_kind_present=class_kind_count > 0,
        module_import_edges_present=import_edge_count > 0,
        module_cycles=_module_cycles(import_edges),
    )


def _module_cycles(import_edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, list[str]] = {}
    for source, target in import_edges:
        if source not in graph:
            graph[source] = []
        if target not in graph:
            graph[target] = []
        graph[source].append(target)

    index_by_node: dict[str, int] = {}
    lowlink_by_node: dict[str, int] = {}
    stack: list[str] = []
    nodes_on_stack: set[str] = set()
    components: list[list[str]] = []
    next_index = 0

    def visit(node: str) -> None:
        nonlocal next_index
        index_by_node[node] = next_index
        lowlink_by_node[node] = next_index
        next_index += 1
        stack.append(node)
        nodes_on_stack.add(node)

        for target in graph[node]:
            if target not in index_by_node:
                visit(target)
                lowlink_by_node[node] = min(lowlink_by_node[node], lowlink_by_node[target])
            elif target in nodes_on_stack:
                lowlink_by_node[node] = min(lowlink_by_node[node], index_by_node[target])

        if lowlink_by_node[node] != index_by_node[node]:
            return

        component: list[str] = []
        while True:
            member = stack.pop()
            nodes_on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1:
            components.append(sorted(component))

    for node in sorted(graph):
        if node not in index_by_node:
            visit(node)

    return sorted(components)


def parse_leyline_output(output: str) -> list[NormalizedFinding]:
    findings: list[NormalizedFinding] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("ERROR: "):
            findings.append(_parse_leyline_violation(line))
            continue
        if not _looks_like_leyline_stale_key(line):
            continue
        parts = line.split(":")
        path = parts[0]
        kind = parts[1]
        name = parts[2]
        findings.append(
            NormalizedFinding(
                check="leyline-types",
                file=path,
                line=None,
                rule="stale-whitelist",
                key=line,
                source="homegrown",
                message=f"{kind} {name} is no longer emitted by lint_leyline_types.py",
            )
        )
    return findings


def parse_defensive_output(output: str) -> list[NormalizedFinding]:
    findings: list[NormalizedFinding] = []
    lines = output.splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line.startswith("src/"):
            continue
        if _looks_like_defensive_stale_key(line):
            findings.append(_parse_defensive_stale_key(line))
            continue
        if ": " not in line or " in " not in line:
            continue
        parts = line.split(":")
        path = parts[0]
        line_number = int(parts[1])
        detail = ":".join(parts[2:]).strip()
        rule = detail.split(" in ")[0]
        key = _following_key(lines, index)
        if key == "":
            continue
        findings.append(
            NormalizedFinding(
                check="defensive-patterns",
                file=path,
                line=line_number,
                rule=rule,
                key=key,
                source="homegrown",
                message=detail,
            )
        )
    return findings


def _following_key(lines: list[str], index: int) -> str:
    for raw_line in lines[index + 1 : index + 4]:
        line = raw_line.strip()
        if line.startswith("Key:"):
            return line.removeprefix("Key:").strip()
    return ""


def build_phase_a_report(
    *,
    wardline_output: Path,
    loomweave: LoomweaveCapabilities,
    defensive_findings: list[NormalizedFinding],
    leyline_findings: list[NormalizedFinding],
) -> dict[str, Any]:
    wardline_missing = not wardline_output.exists()
    wardline_has_defensive_rule = (
        False if wardline_missing else _wardline_has_defensive_rule(wardline_output)
    )
    defensive_blockers: list[str] = []
    if wardline_missing:
        defensive_blockers.append(
            "Wardline artifact is missing; inspect the Wardline shadow step before evaluating defensive-pattern parity."
        )
    elif not wardline_has_defensive_rule:
        defensive_blockers.append(
            "Wardline emitted no defensive-pattern rule mapping; keep lint_defensive_patterns.py blocking."
        )

    leyline_blockers: list[str] = []
    if not loomweave.class_contract_kind_present:
        leyline_blockers.append(
            "Loomweave class_contract_kind metadata is not present; keep lint_leyline_types.py blocking."
        )

    module_cycle_blockers: list[str] = []
    if not loomweave.database_present:
        module_cycle_blockers.append(
            "Loomweave database is missing; run loomweave analyze before evaluating module cycles."
        )
    if not loomweave.module_import_edges_present:
        module_cycle_blockers.append(
            "Loomweave module import edges are missing; module-cycle replacement is not evaluable."
        )
    if len(loomweave.module_cycles) > 0:
        module_cycle_blockers.append(
            "Loomweave reported module cycles; keep module-cycle evidence advisory until triaged."
        )

    return {
        "schema": "esper-weft-parity-v1",
        "checks": [
            _phase_a_check(
                check="defensive-patterns",
                shadow_signal_ready=len(defensive_blockers) == 0
                and len(defensive_findings) == 0,
                matches=[],
                weft_only=[],
                homegrown_only=_finding_dicts(defensive_findings),
                blockers=defensive_blockers,
            ),
            _phase_a_check(
                check="leyline-types",
                shadow_signal_ready=len(leyline_blockers) == 0
                and len(leyline_findings) == 0,
                matches=[],
                weft_only=[],
                homegrown_only=_finding_dicts(leyline_findings),
                blockers=leyline_blockers,
            ),
            _phase_a_check(
                check="module-cycles",
                shadow_signal_ready=len(module_cycle_blockers) == 0,
                matches=[],
                weft_only=[
                    {
                        "check": "module-cycles",
                        "rule": "loomweave-module-cycle",
                        "cycle": cycle,
                    }
                    for cycle in loomweave.module_cycles
                ],
                homegrown_only=[],
                blockers=module_cycle_blockers,
            ),
        ],
    }


def _phase_a_check(
    *,
    check: str,
    shadow_signal_ready: bool,
    matches: list[dict[str, Any]],
    weft_only: list[dict[str, Any]],
    homegrown_only: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    phase_a_gate = (
        "Phase A shadow burn-in is not complete; keep the homegrown gate blocking "
        "until 20 PRs or 14 days have zero homegrown-only findings."
    )
    return {
        "check": check,
        "replacement_ready": False,
        "shadow_signal_ready": shadow_signal_ready,
        "matches": matches,
        "weft_only": weft_only,
        "homegrown_only": homegrown_only,
        "blockers": blockers + [phase_a_gate],
        "retirement_gate": {
            "required": "20 PRs or 14 days with zero homegrown-only findings",
            "satisfied": False,
        },
    }


def _looks_like_leyline_stale_key(line: str) -> bool:
    if not line.startswith("src/esper/"):
        return False
    return (
        ":enum:" in line
        or ":dataclass:" in line
        or ":protocol:" in line
        or ":typeddict:" in line
        or ":namedtuple:" in line
    )


def _parse_leyline_violation(line: str) -> NormalizedFinding:
    match = re.fullmatch(
        r"ERROR: (?P<path>[^:]+):(?P<line>\d+): "
        r"(?P<kind>enum|dataclass|protocol|typeddict|namedtuple) "
        r"'(?P<name>[^']+)' not in leyline or whitelist",
        line,
    )
    if match is None:
        raise ValueError(f"Unrecognized leyline violation: {line}")
    kind = match["kind"]
    name = match["name"]
    path = match["path"]
    return NormalizedFinding(
        check="leyline-types",
        file=path,
        line=int(match["line"]),
        rule=kind,
        key=f"{path}:{kind}:{name}",
        source="homegrown",
        message=f"{kind} {name} not in leyline or whitelist",
    )


def _looks_like_defensive_stale_key(line: str) -> bool:
    parts = line.split(":")
    if len(parts) not in (3, 4):
        return False
    return (
        line.startswith("src/esper/")
        and parts[2]
        in {"getattr", "hasattr", "silent_except", "bare_except", "isinstance", "get"}
    )


def _parse_defensive_stale_key(line: str) -> NormalizedFinding:
    parts = line.split(":")
    path = parts[0]
    pattern = parts[2]
    return NormalizedFinding(
        check="defensive-patterns",
        file=path,
        line=None,
        rule="stale-whitelist",
        key=line,
        source="homegrown",
        message=f"{pattern} whitelist entry is stale",
    )


def _finding_dicts(findings: list[NormalizedFinding]) -> list[dict[str, Any]]:
    return [
        {
            "check": finding.check,
            "file": finding.file,
            "line": finding.line,
            "rule": finding.rule,
            "key": finding.key,
            "source": finding.source,
            "message": finding.message,
        }
        for finding in findings
    ]


def _wardline_has_defensive_rule(wardline_output: Path) -> bool:
    for raw_line in wardline_output.read_text().splitlines():
        if raw_line == "":
            continue
        payload = json.loads(raw_line)
        rule_id = str(payload["rule_id"])
        if "DEFENSIVE" in rule_id or "BUG-HIDING" in rule_id:
            return True
    return False
