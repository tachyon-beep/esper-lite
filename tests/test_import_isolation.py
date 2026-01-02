"""Regression tests for import isolation.

Verifies that importing light modules does NOT trigger heavy imports:
- esper.tolaria should not load torch or telemetry hub at import time
- esper.runtime should not load esper.simic.training.vectorized at import time
- esper.simic should not load training loops at import time

This prevents the import-time side effects that were causing circular dependencies
and slow startup times.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


# Mark all tests in this module to skip torch seeding
# (prevents conftest fixture from auto-loading torch after we delete it)
pytestmark = pytest.mark.no_torch_seeding


def _run_isolated(code: str) -> dict[str, object]:
    """Run code in a fresh Python process and return its JSON stdout."""
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_tolaria_import_isolation():
    """Importing esper.tolaria should NOT load torch or nissa."""
    result = _run_isolated(
        """
import json
import sys

import esper.tolaria

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "nissa_loaded": "esper.nissa" in sys.modules,
    "vectorized_loaded": "esper.simic.training.vectorized" in sys.modules,
}))
""".strip()
    )

    assert result["torch_loaded"] is False
    assert result["nissa_loaded"] is False
    assert result["vectorized_loaded"] is False


def test_runtime_import_isolation():
    """Importing esper.runtime should NOT load heavy training modules."""
    result = _run_isolated(
        """
import json
import sys

import esper.runtime

print(json.dumps({
    "vectorized_loaded": "esper.simic.training.vectorized" in sys.modules,
    "dual_ab_loaded": "esper.simic.training.dual_ab" in sys.modules,
}))
""".strip()
    )

    assert result["vectorized_loaded"] is False
    assert result["dual_ab_loaded"] is False


def test_simic_import_isolation():
    """Importing esper.simic should NOT load heavy training loops."""
    result = _run_isolated(
        """
import json
import sys

import esper.simic

print(json.dumps({
    "vectorized_loaded": "esper.simic.training.vectorized" in sys.modules,
    "dual_ab_loaded": "esper.simic.training.dual_ab" in sys.modules,
}))
""".strip()
    )

    assert result["vectorized_loaded"] is False
    assert result["dual_ab_loaded"] is False


def test_tolaria_lazy_attributes_work():
    """Verify that lazy imports via __getattr__ work correctly."""
    result = _run_isolated(
        """
import json
import sys

import esper.tolaria
from esper.tolaria import create_model

print(json.dumps({
    "callable": callable(create_model),
    "environment_loaded": "esper.tolaria.environment" in sys.modules,
}))
""".strip()
    )

    assert result["callable"] is True
    assert result["environment_loaded"] is True


def test_explicit_training_imports_work():
    """Verify that explicit imports of training functions still work."""
    result = _run_isolated(
        """
import json
import sys

from esper.simic.training.vectorized import train_ppo_vectorized

print(json.dumps({
    "callable": callable(train_ppo_vectorized),
    "vectorized_loaded": "esper.simic.training.vectorized" in sys.modules,
}))
""".strip()
    )

    assert result["callable"] is True
    assert result["vectorized_loaded"] is True


def test_kasmina_import_isolation():
    """Importing esper.kasmina should NOT load torch or heavy modules."""
    result = _run_isolated(
        """
import json
import sys

import esper.kasmina

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "slot_loaded": "esper.kasmina.slot" in sys.modules,
    "blueprints_loaded": "esper.kasmina.blueprints" in sys.modules,
    "host_loaded": "esper.kasmina.host" in sys.modules,
    "isolation_loaded": "esper.kasmina.isolation" in sys.modules,
}))
""".strip()
    )

    assert result["torch_loaded"] is False, "torch should not load from package import"
    assert result["slot_loaded"] is False, "slot (heavy) should not auto-load"
    assert result["blueprints_loaded"] is False, "blueprints (heavy) should not auto-load"
    assert result["host_loaded"] is False, "host (heavy) should not auto-load"
    assert result["isolation_loaded"] is False, "isolation (heavy) should not auto-load"


def test_kasmina_lazy_attributes_work():
    """Verify that lazy kasmina imports work correctly."""
    result = _run_isolated(
        """
import json
import sys

import esper.kasmina
from esper.kasmina import SeedSlot, SeedState

print(json.dumps({
    "SeedSlot_callable": callable(SeedSlot),
    "SeedState_callable": callable(SeedState),
    "SeedSlot_cached": "SeedSlot" in vars(esper.kasmina),
    "SeedState_cached": "SeedState" in vars(esper.kasmina),
    "slot_loaded": "esper.kasmina.slot" in sys.modules,
    "torch_loaded": "torch" in sys.modules,
}))
""".strip()
    )

    assert result["SeedSlot_callable"] is True
    assert result["SeedState_callable"] is True
    assert result["SeedSlot_cached"] is True, "lazy attribute should be cached on module"
    assert result["SeedState_cached"] is True, "lazy attribute should be cached on module"
    assert result["slot_loaded"] is True, "slot should load when SeedSlot accessed"
    assert result["torch_loaded"] is True, "torch should load with slot"


def test_kasmina_dir_exposes_public_api():
    """dir(esper.kasmina) should include __all__ without loading heavy modules."""
    result = _run_isolated(
        """
import json
import sys

import esper.kasmina

missing = sorted([name for name in esper.kasmina.__all__ if name not in dir(esper.kasmina)])

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "missing": missing,
}))
""".strip()
    )

    assert result["torch_loaded"] is False, "dir() must not trigger heavy imports"
    assert result["missing"] == [], "dir() should expose all public kasmina exports"


def test_nissa_import_isolation():
    """Importing esper.nissa should NOT load torch or heavy modules."""
    result = _run_isolated(
        """
import json
import sys

import esper.nissa

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "tracker_loaded": "esper.nissa.tracker" in sys.modules,
}))
""".strip()
    )

    assert result["torch_loaded"] is False, "torch should not load from package import"
    assert result["tracker_loaded"] is False, "tracker (heavy) should not auto-load"


def test_nissa_lazy_attributes_work():
    """Verify that lazy nissa imports work correctly."""
    result = _run_isolated(
        """
import json
import sys

import esper.nissa
from esper.nissa import DiagnosticTracker, get_hub

print(json.dumps({
    "DiagnosticTracker_callable": callable(DiagnosticTracker),
    "get_hub_callable": callable(get_hub),
    "tracker_loaded": "esper.nissa.tracker" in sys.modules,
    "output_loaded": "esper.nissa.output" in sys.modules,
    "torch_loaded": "torch" in sys.modules,
}))
""".strip()
    )

    assert result["DiagnosticTracker_callable"] is True
    assert result["get_hub_callable"] is True
    assert result["tracker_loaded"] is True, "tracker should load when DiagnosticTracker accessed"
    assert result["output_loaded"] is True, "output should load when get_hub accessed"
    assert result["torch_loaded"] is True, "torch should load with tracker"


def test_tamiyo_import_isolation():
    """Importing esper.tamiyo should NOT load torch or heavy modules."""
    result = _run_isolated(
        """
import json
import sys

import esper.tamiyo

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "policy_loaded": "esper.tamiyo.policy" in sys.modules,
    "nissa_tracker_loaded": "esper.nissa.tracker" in sys.modules,
}))
""".strip()
    )

    assert result["torch_loaded"] is False, "torch should not load from package import"
    assert result["policy_loaded"] is False, "policy (heavy) should not auto-load"
    assert result["nissa_tracker_loaded"] is False, "nissa tracker should not auto-load"


def test_tamiyo_lazy_attributes_work():
    """Verify that lazy tamiyo imports work correctly."""
    result = _run_isolated(
        """
import json
import sys

import esper.tamiyo
from esper.tamiyo import PolicyBundle, get_policy

print(json.dumps({
    "PolicyBundle_is_class": str(type(PolicyBundle)),
    "get_policy_callable": callable(get_policy),
    "policy_loaded": "esper.tamiyo.policy" in sys.modules,
    "torch_loaded": "torch" in sys.modules,
}))
""".strip()
    )

    assert "Protocol" in result["PolicyBundle_is_class"] or "ABCMeta" in result["PolicyBundle_is_class"]
    assert result["get_policy_callable"] is True
    assert result["policy_loaded"] is True, "policy should load when accessed"
    assert result["torch_loaded"] is True, "torch should load with policy (LSTM registration)"


def test_simic_contracts_no_torch():
    """Importing esper.simic.contracts should NOT load torch."""
    result = _run_isolated(
        """
import json
import sys

from esper.simic.contracts import SeedSlotProtocol

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "protocol_is_class": str(type(SeedSlotProtocol)),
}))
""".strip()
    )

    assert result["torch_loaded"] is False, "contracts should not load torch (TYPE_CHECKING only)"
    assert ("Protocol" in result["protocol_is_class"] or "ABCMeta" in result["protocol_is_class"]), \
        "SeedSlotProtocol should be a Protocol"


def test_kasmina_host_protocol_no_torch():
    """Importing HostProtocol should NOT load torch (it's a lightweight Protocol).

    HostProtocol is categorized as "lightweight" in kasmina/__init__.py because
    it's just a structural typing contract. It should use TYPE_CHECKING imports
    for torch types, not runtime imports.

    Regression test for: torch import leak via protocol.py unconditional import.
    """
    result = _run_isolated(
        """
import json
import sys

from esper.kasmina import HostProtocol

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "protocol_cached": "HostProtocol" in vars(sys.modules["esper.kasmina"]),
    "leyline_host_protocol_loaded": "esper.leyline.host_protocol" in sys.modules,
    "protocol_is_class": str(type(HostProtocol)),
}))
""".strip()
    )

    assert result["torch_loaded"] is False, (
        "HostProtocol is lightweight - torch should not load. "
        "Fix: move 'from torch import Tensor' to TYPE_CHECKING block in protocol.py"
    )
    assert result["protocol_cached"] is True, "lazy attribute should be cached on module"
    assert result["leyline_host_protocol_loaded"] is True, "host_protocol module should load from leyline"
    assert "Protocol" in result["protocol_is_class"], "HostProtocol should be a Protocol"


def test_kasmina_alpha_controller_no_torch():
    """Importing AlphaController should NOT load torch (it's pure scheduling logic)."""
    result = _run_isolated(
        """
import json
import sys

from esper.kasmina import AlphaController

print(json.dumps({
    "torch_loaded": "torch" in sys.modules,
    "alpha_cached": "AlphaController" in vars(sys.modules["esper.kasmina"]),
    "alpha_controller_loaded": "esper.kasmina.alpha_controller" in sys.modules,
}))
""".strip()
    )

    assert result["torch_loaded"] is False, "AlphaController is lightweight - no torch"
    assert result["alpha_cached"] is True, "lazy attribute should be cached on module"
    assert result["alpha_controller_loaded"] is True, "alpha_controller module should load"
