"""Fast import smoke tests for lifecycle contract refactors.

These imports are intentionally broad: during Phase 1 (contracts/naming), a
missing enum member or stale lookup table often manifests as an ImportError at
module import time.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "esper.leyline.stages",
        "esper.leyline.factored_actions",
        "esper.leyline.telemetry",
        "esper.kasmina.slot",
        "esper.simic.rewards",
        "esper.tamiyo.policy.action_masks",
    ],
)
def test_import_smoke(module_name: str) -> None:
    importlib.import_module(module_name)
