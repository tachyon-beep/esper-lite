"""Tests for the Dependabot advisory floor gate.

Covers the ecosystem-aware version comparison in
``scripts/assert_dependabot_advisories.py``: npm versions follow SemVer 2.0.0
precedence (arbitrary pre-release tags such as ``next``/``canary``/``insiders``),
while pip versions follow PEP 440. The gate must never crash on a valid SemVer
pre-release tag that PEP 440 rejects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "assert_dependabot_advisories.py"
_SPEC = importlib.util.spec_from_file_location("assert_dependabot_advisories", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
# Register before exec so the module's frozen @dataclass can resolve its own module.
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

_semver_key = _MOD._semver_key
_version_key = _MOD._version_key
_dedupe_versions = _MOD._dedupe_versions
_row_for_alert = _MOD._row_for_alert


# The canonical precedence chain from the SemVer 2.0.0 spec, section 11.
SEMVER_PRECEDENCE = [
    "1.0.0-alpha",
    "1.0.0-alpha.1",
    "1.0.0-alpha.beta",
    "1.0.0-beta",
    "1.0.0-beta.2",
    "1.0.0-beta.11",
    "1.0.0-rc.1",
    "1.0.0",
]


def test_semver_key_orders_canonical_precedence_chain() -> None:
    keys = [_semver_key(v) for v in SEMVER_PRECEDENCE]
    assert keys == sorted(keys)
    for lower, higher in zip(keys, keys[1:]):
        assert lower < higher


def test_semver_key_numeric_identifiers_compare_numerically() -> None:
    # 2 < 11 numerically, not lexically.
    assert _semver_key("1.0.0-beta.2") < _semver_key("1.0.0-beta.11")


def test_semver_key_ignores_build_metadata() -> None:
    assert _semver_key("1.2.3+build.99") == _semver_key("1.2.3+build.1")
    assert _semver_key("1.2.3+build.99") == _semver_key("1.2.3")


@pytest.mark.parametrize("tag", ["next", "canary", "insiders"])
def test_semver_key_accepts_npm_prerelease_tags(tag: str) -> None:
    # Valid SemVer but invalid PEP 440 — these must not raise.
    prerelease = _semver_key(f"7.0.0-{tag}.1")
    release = _semver_key("7.0.0")
    assert prerelease < release


def test_dedupe_versions_npm_handles_prerelease_without_crashing() -> None:
    versions = ("7.0.0", "7.0.0-next.1", "7.0.0-next.1", "6.9.0")
    result = _dedupe_versions("npm", versions)
    assert result == ("6.9.0", "7.0.0-next.1", "7.0.0")


def test_version_key_pip_preserves_pep440_ordering() -> None:
    assert _version_key("pip", "2.0.0rc1") < _version_key("pip", "2.0.0")
    assert _version_key("pip", "1.0.0.post1") > _version_key("pip", "1.0.0")
    assert _version_key("pip", "1.9.9") < _version_key("pip", "1.10.0")


def _npm_alert(package: str, patched: str) -> dict[str, Any]:
    return {
        "number": 1,
        "state": "open",
        "dependency": {
            "package": {"ecosystem": "npm", "name": package},
            "manifest_path": "src/esper/karn/overwatch/web/package-lock.json",
        },
        "security_advisory": {"severity": "high", "ghsa_id": "GHSA-test-0000-0000"},
        "security_vulnerability": {"first_patched_version": {"identifier": patched}},
    }


def test_row_for_alert_npm_prerelease_below_floor_is_vulnerable() -> None:
    # Locked at a pre-release below the patched floor → vulnerable, no InvalidVersion.
    alert = _npm_alert("vite", patched="7.0.0")
    row = _row_for_alert(alert, {}, {"vite": ("7.0.0-next.1",)})
    assert row.disposition == "vulnerable present"


def test_row_for_alert_npm_release_at_floor_is_patched() -> None:
    # Patched floor is itself a pre-release; the locked release outranks it.
    alert = _npm_alert("vite", patched="7.0.0-next.1")
    row = _row_for_alert(alert, {}, {"vite": ("7.0.0",)})
    assert row.disposition == "patched"
