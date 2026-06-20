"""Assert local locks satisfy live high/critical Dependabot floors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from packaging.version import Version


SEVERITIES = {"critical", "high"}


@dataclass(frozen=True)
class AdvisoryRow:
    number: int
    ecosystem: str
    package: str
    severity: str
    ghsa_id: str
    current_versions: tuple[str, ...]
    first_patched_version: str | None
    manifest_path: str
    disposition: str


def _run_json(command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return cast(dict[str, Any], json.loads(result.stdout))


def _load_python_versions(lock_path: Path) -> dict[str, tuple[str, ...]]:
    lock = tomllib.loads(lock_path.read_text())
    packages = cast(list[dict[str, Any]], lock["package"])
    versions: dict[str, list[str]] = {}
    for package in packages:
        name = cast(str, package["name"]).lower()
        version = cast(str, package["version"])
        if name not in versions:
            versions[name] = []
        versions[name].append(version)
    return {name: tuple(values) for name, values in versions.items()}


def _package_name_from_lock_path(lock_path: str) -> str:
    marker = "node_modules/"
    if marker not in lock_path:
        return ""
    return lock_path.rsplit(marker, maxsplit=1)[1]


def _load_npm_versions(package_lock_path: Path) -> dict[str, tuple[str, ...]]:
    package_lock = cast(dict[str, Any], json.loads(package_lock_path.read_text()))
    packages = cast(dict[str, dict[str, Any]], package_lock["packages"])
    versions: dict[str, list[str]] = {}
    for package_path, package in packages.items():
        if "version" not in package:
            continue
        name = _package_name_from_lock_path(package_path)
        if name == "":
            continue
        if name not in versions:
            versions[name] = []
        versions[name].append(cast(str, package["version"]))
    return {name: tuple(values) for name, values in versions.items()}


def _dedupe_versions(versions: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(versions), key=Version))


def _current_versions(
    ecosystem: str,
    package: str,
    python_versions: dict[str, tuple[str, ...]],
    npm_versions: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if ecosystem == "pip":
        return _dedupe_versions(python_versions[package.lower()])
    if ecosystem == "npm":
        return _dedupe_versions(npm_versions[package])
    raise ValueError(f"Unsupported ecosystem: {ecosystem}")


def _has_package(
    ecosystem: str,
    package: str,
    python_versions: dict[str, tuple[str, ...]],
    npm_versions: dict[str, tuple[str, ...]],
) -> bool:
    if ecosystem == "pip":
        return package.lower() in python_versions
    if ecosystem == "npm":
        return package in npm_versions
    raise ValueError(f"Unsupported ecosystem: {ecosystem}")


def _row_for_alert(
    alert: dict[str, Any],
    python_versions: dict[str, tuple[str, ...]],
    npm_versions: dict[str, tuple[str, ...]],
) -> AdvisoryRow:
    dependency = cast(dict[str, Any], alert["dependency"])
    package_info = cast(dict[str, str], dependency["package"])
    advisory = cast(dict[str, str], alert["security_advisory"])
    vulnerability = cast(dict[str, Any], alert["security_vulnerability"])

    ecosystem = package_info["ecosystem"]
    package = package_info["name"]
    first_patched = cast(dict[str, str] | None, vulnerability["first_patched_version"])
    patched_version = None if first_patched is None else first_patched["identifier"]
    manifest_path = cast(str, dependency["manifest_path"])

    if patched_version is None:
        current_versions: tuple[str, ...] = ()
        disposition = "no fix available"
    elif not _has_package(ecosystem, package, python_versions, npm_versions):
        current_versions = ()
        disposition = "absent optional transitive"
    else:
        current_versions = _current_versions(
            ecosystem, package, python_versions, npm_versions
        )
        vulnerable_versions = [
            version for version in current_versions if Version(version) < Version(patched_version)
        ]
        if vulnerable_versions:
            disposition = "vulnerable present"
        else:
            disposition = "patched"

    return AdvisoryRow(
        number=cast(int, alert["number"]),
        ecosystem=ecosystem,
        package=package,
        severity=advisory["severity"],
        ghsa_id=advisory["ghsa_id"],
        current_versions=current_versions,
        first_patched_version=patched_version,
        manifest_path=manifest_path,
        disposition=disposition,
    )


def _open_high_critical_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        alert
        for alert in alerts
        if alert["state"] == "open"
        and alert["security_advisory"]["severity"] in SEVERITIES
    ]


def _format_versions(versions: tuple[str, ...]) -> str:
    if not versions:
        return "-"
    return ", ".join(versions)


def _print_table(rows: list[AdvisoryRow]) -> None:
    print(
        "| # | ecosystem | package | severity | GHSA | current | first patched | manifest | disposition |"
    )
    print("|---|-----------|---------|----------|------|---------|---------------|----------|-------------|")
    for row in rows:
        patched = row.first_patched_version or "NO_FIX"
        print(
            f"| {row.number} | {row.ecosystem} | {row.package} | {row.severity} | "
            f"{row.ghsa_id} | {_format_versions(row.current_versions)} | {patched} | "
            f"{row.manifest_path} | {row.disposition} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assert local lockfiles satisfy open high/critical Dependabot advisories."
    )
    parser.add_argument(
        "--alerts-json",
        type=Path,
        help="Path to a JSON array from `gh api .../dependabot/alerts --paginate`.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch live alerts with gh instead of reading --alerts-json.",
    )
    parser.add_argument("--repo", default="tachyon-beep/esper-lite")
    parser.add_argument("--uv-lock", type=Path, default=Path("uv.lock"))
    parser.add_argument(
        "--package-lock",
        type=Path,
        default=Path("src/esper/karn/overwatch/web/package-lock.json"),
    )
    args = parser.parse_args()

    if args.fetch:
        alerts_payload = _run_json(
            ["gh", "api", f"/repos/{args.repo}/dependabot/alerts", "--paginate"],
            Path.cwd(),
        )
        alerts = cast(list[dict[str, Any]], alerts_payload)
    else:
        if args.alerts_json is None:
            parser.error("pass --alerts-json or --fetch")
        alerts = cast(list[dict[str, Any]], json.loads(args.alerts_json.read_text()))

    python_versions = _load_python_versions(args.uv_lock)
    npm_versions = _load_npm_versions(args.package_lock)
    rows = [
        _row_for_alert(alert, python_versions, npm_versions)
        for alert in _open_high_critical_alerts(alerts)
    ]
    rows.sort(key=lambda row: row.number, reverse=True)
    _print_table(rows)

    failures = [row for row in rows if row.disposition == "vulnerable present"]
    if failures:
        print("\nVULNERABLE-HIGH-CRITICAL-PACKAGES", file=sys.stderr)
        for row in failures:
            patched = row.first_patched_version or "NO_FIX"
            print(
                f"{row.package} current={_format_versions(row.current_versions)} "
                f"patched_floor={patched} advisory={row.ghsa_id}",
                file=sys.stderr,
            )
        return 1

    print("\nHIGH-CRITICAL-DEPENDABOT-FLOORS-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
