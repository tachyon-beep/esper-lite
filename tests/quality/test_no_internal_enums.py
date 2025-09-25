from __future__ import annotations

import re
from pathlib import Path


def test_no_custom_enums_outside_allowed_whitelist() -> None:
    root = Path("src/esper")
    enum_pattern = re.compile(r"^class\s+\w+\(.*Enum\):", re.MULTILINE)

    # Whitelisted files that define domain enums not (yet) in Leyline
    # Note: These are flagged for centralization in Leyline (see ADR-003 follow-ups).
    whitelist = {
        root / "karn" / "catalog.py",  # BlueprintTier
    }

    violations: list[str] = []
    for path in root.rglob("*.py"):
        # Skip generated bindings and private caches
        if "_generated" in path.parts:
            continue
        if path in whitelist:
            continue
        text = path.read_text(encoding="utf-8")
        if enum_pattern.search(text):
            violations.append(str(path))

    assert not violations, f"Custom Enum definitions found outside whitelist: {violations}"
