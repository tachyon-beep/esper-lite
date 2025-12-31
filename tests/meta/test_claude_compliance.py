"""CLAUDE.md compliance tests for no-legacy and no-defensive-programming policies."""

import subprocess
import inspect
import pytest
from pathlib import Path


def test_no_legacy_version_toggles():
    """Verify no legacy V2 code remains after V3 implementation (CLAUDE.md)."""
    src_path = Path(__file__).parent.parent.parent / "src"

    forbidden_patterns = [
        (r"if.*obs_version|OBS_VERSION", "Version checks for obs V2/V3"),
        (r"if.*policy_version|POLICY_VERSION", "Version checks for policy V1/V2"),
        (r"@deprecated", "Deprecated decorators"),
        (r"def.*_v2\(|def.*_legacy\(|def.*_old\(", "Legacy function suffixes"),
        (r"_OBS_V2|_POLICY_V1", "Legacy constants"),
    ]

    violations = []
    for pattern, description in forbidden_patterns:
        result = subprocess.run(
            ["grep", "-rE", pattern, str(src_path), "--include=*.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            violations.append(f"{description}: {result.stdout[:200]}")

    assert not violations, (
        "Found legacy code patterns violating CLAUDE.md no-legacy policy:\n" +
        "\n".join(violations)
    )


def test_no_defensive_getattr_in_feature_extraction():
    """Verify feature extraction doesn't use defensive getattr/hasattr (CLAUDE.md)."""
    from esper.tamiyo.policy import features

    source = inspect.getsource(features)

    # Check for defensive patterns
    violations = []

    if "getattr(" in source:
        violations.append("Found getattr() - likely defensive programming")

    if "hasattr(" in source:
        violations.append("Found hasattr() - likely defensive programming")

    # .get() is OK for dictionaries (slot_reports is dict[str, SeedStateReport])
    # but NOT for typed dataclass access
    # Check for suspicious patterns like "report.get(" or "signals.get("
    if "report.get(" in source or "signals.get(" in source:
        violations.append("Found .get() on typed objects - defensive programming")

    assert not violations, (
        "Feature extraction violates CLAUDE.md defensive programming prohibition:\n" +
        "\n".join(violations)
    )


def test_no_defensive_isinstance_in_features():
    """Verify no isinstance checks hiding malformed data (CLAUDE.md)."""
    from esper.tamiyo.policy import features

    source = inspect.getsource(features)

    # isinstance is OK for legitimate type handling (PyTorch tensors, device normalization)
    # NOT OK for hiding bugs like "if isinstance(x, dict) else default"

    # Check for suspicious patterns
    suspicious_patterns = [
        ("isinstance.*dict", "isinstance checking for dict on typed dataclass"),
        ("isinstance.*str.*else", "isinstance with fallback hiding type errors"),
    ]

    violations = []
    for pattern, description in suspicious_patterns:
        if subprocess.run(
            ["grep", "-E", pattern],
            input=source,
            capture_output=True,
            text=True,
        ).returncode == 0:
            violations.append(description)

    # Allow legitimate uses (commented in source)
    # This test catches NEW violations, not documented exceptions

    assert not violations, (
        "Found suspicious isinstance patterns (may violate CLAUDE.md):\n" +
        "\n".join(violations)
    )


def test_no_backwards_compatibility_shims():
    """Verify no compatibility adapters for old APIs exist (CLAUDE.md)."""
    src_path = Path(__file__).parent.parent.parent / "src"

    # Check for adapter/wrapper patterns that support dual APIs
    shim_patterns = [
        (r"class.*Adapter|class.*Wrapper", "Adapter/wrapper classes"),
        (r"def.*_compat\(|def.*_compatibility\(", "Compatibility functions"),
        (r"# Legacy support|# Backwards compatibility", "Legacy support comments"),
    ]

    violations = []
    for pattern, description in shim_patterns:
        result = subprocess.run(
            ["grep", "-rE", pattern, str(src_path), "--include=*.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            violations.append(f"{description}: {result.stdout[:200]}")

    assert not violations, (
        "Found backwards compatibility shims violating CLAUDE.md:\n" +
        "\n".join(violations)
    )
