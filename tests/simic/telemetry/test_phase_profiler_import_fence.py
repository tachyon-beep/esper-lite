"""V3 import-fence regression: the Tier-0 phase profiler is observation-only.

Rule 4 of docs/plans/concepts/2026-06-16-gil-throughput-profiler.md: the
PhaseProfileReport flows ONE-WAY to nissa telemetry. It MUST NOT be read by
reward (simic/rewards), governor (tolaria/governor), scheduler, or PPO, and MUST
NOT enter any snapshot/divergence hash.

We enforce this structurally: importing those decision modules in a fresh
interpreter must NOT pull in either the profiler implementation module
(``esper.simic.telemetry.phase_profiler``) or leave the profiler report types
reachable as a hard dependency of those modules.

The profiler implementation module is the load-bearing fence: if reward/governor/
PPO imported it, they would have a live handle to a profiler. (The leyline report
*types* live in ``esper.leyline.reports`` alongside many other contracts that the
decision modules legitimately import, so we fence the implementation module, which
is the thing that could ever produce a live report a decision path could read.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.no_torch_seeding

_PROFILER_MODULE = "esper.simic.telemetry.phase_profiler"

# Decision-path modules that MUST NOT import the phase profiler implementation.
_FENCED_MODULES = (
    "esper.simic.rewards",
    "esper.tolaria.governor",
    "esper.simic.agent.ppo_update",
    "esper.simic.agent.ppo_agent",
)


def _run_isolated(code: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
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


@pytest.mark.parametrize("module", _FENCED_MODULES)
def test_decision_module_does_not_import_phase_profiler(module: str) -> None:
    """Reward/governor/PPO must not import the phase-profiler implementation."""
    result = _run_isolated(
        f"""
import json
import sys

import {module}  # noqa: F401

print(json.dumps({{
    "profiler_loaded": {_PROFILER_MODULE!r} in sys.modules,
}}))
""".strip()
    )
    assert result["profiler_loaded"] is False, (
        f"{module} transitively imported {_PROFILER_MODULE}; the Tier-0 phase "
        "profiler must be observation-only (Rule 4)."
    )


def test_governor_snapshot_does_not_reference_phase_report() -> None:
    """The governor snapshot/divergence path must not reference PhaseProfileReport.

    Static guard: the governor source must not name the report type. A divergence
    hash that incorporated phase timings would make replay nondeterministic.
    """
    governor_src = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "esper"
        / "tolaria"
        / "governor.py"
    ).read_text()
    assert "PhaseProfileReport" not in governor_src
    assert "phase_profiler" not in governor_src
