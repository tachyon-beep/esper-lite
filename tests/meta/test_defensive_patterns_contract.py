"""Regression tests for the defensive-pattern linter contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINTER_PATH = _REPO_ROOT / "scripts" / "lint_defensive_patterns.py"
_SPEC = importlib.util.spec_from_file_location(
    "lint_defensive_patterns_for_test",
    _LINTER_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_LINTER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LINTER
_SPEC.loader.exec_module(_LINTER)
find_patterns = _LINTER.find_patterns


def test_contract_module_fixtures_flag_strict_defensive_patterns(tmp_path) -> None:
    fixture = tmp_path / "contract_module.py"
    fixture.write_text(
        "\n".join(
            [
                "def read_contract(obj):",
                "    if hasattr(obj, 'field'):",
                "        return getattr(obj, 'field')",
                "    try:",
                "        return obj.field",
                "    except AttributeError:",
                "        pass",
            ]
        )
    )

    hits = find_patterns(fixture)
    patterns = {hit.pattern for hit in hits}
    keys = {hit.key for hit in hits}

    assert {"hasattr", "getattr", "silent_except"} <= patterns
    assert f"{fixture}:read_contract:hasattr" in keys
    assert f"{fixture}:read_contract:getattr" in keys
    assert f"{fixture}:read_contract:silent_except" in keys


def test_contract_module_direct_access_fixture_is_clean(tmp_path) -> None:
    fixture = tmp_path / "contract_module.py"
    fixture.write_text(
        "\n".join(
            [
                "def read_contract(obj):",
                "    return obj.field",
            ]
        )
    )

    assert find_patterns(fixture) == []
