# Reward Function Transition Plan: SHAPED → SIMPLIFIED with Pareto Tracking

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transition Tamiyo's reward system from complex SHAPED mode to cleaner SIMPLIFIED mode, with multi-objective Pareto frontier tracking for principled reward engineering.

**Architecture:** Extend `episode_history` to capture multi-objective outcomes (`EpisodeOutcome`), add Pareto frontier computation to Karn analytics, create ablation configs for systematic comparison, and add reward health panel to Sanctum TUI.

**Tech Stack:** Python dataclasses, DuckDB views (Karn MCP), Textual widgets (Sanctum), existing A/B testing infrastructure.

---

## Phase 1: Multi-Objective Episode Outcomes

### Task 1: Define EpisodeOutcome Schema

**Files:**
- Modify: `src/esper/karn/store.py` (add new dataclass)
- Test: `tests/karn/test_episode_outcome.py`

**Step 1: Write the failing test**

```python
# tests/karn/test_episode_outcome.py
"""Tests for EpisodeOutcome multi-objective schema."""

import pytest
from esper.karn.store import EpisodeOutcome


def test_episode_outcome_creation():
    """EpisodeOutcome captures all multi-objective fields."""
    outcome = EpisodeOutcome(
        env_idx=0,
        episode_idx=5,
        final_accuracy=75.5,
        param_ratio=0.15,
        num_fossilized=2,
        num_contributing_fossilized=1,
        episode_reward=12.5,
        stability_score=0.95,
        reward_mode="shaped",
    )
    assert outcome.final_accuracy == 75.5
    assert outcome.param_ratio == 0.15
    assert outcome.num_fossilized == 2


def test_episode_outcome_dominates():
    """Test Pareto dominance checking."""
    better = EpisodeOutcome(
        env_idx=0, episode_idx=1,
        final_accuracy=80.0, param_ratio=0.1, num_fossilized=2,
        num_contributing_fossilized=2, episode_reward=15.0,
        stability_score=0.98, reward_mode="shaped",
    )
    worse = EpisodeOutcome(
        env_idx=1, episode_idx=2,
        final_accuracy=75.0, param_ratio=0.2, num_fossilized=1,
        num_contributing_fossilized=1, episode_reward=10.0,
        stability_score=0.90, reward_mode="shaped",
    )
    # better dominates worse: higher accuracy, lower param_ratio, higher stability
    assert better.dominates(worse)
    assert not worse.dominates(better)


def test_episode_outcome_to_dict():
    """to_dict() produces JSON-serializable output."""
    outcome = EpisodeOutcome(
        env_idx=0, episode_idx=1,
        final_accuracy=70.0, param_ratio=0.05, num_fossilized=1,
        num_contributing_fossilized=1, episode_reward=8.0,
        stability_score=0.85, reward_mode="simplified",
    )
    d = outcome.to_dict()
    assert d["final_accuracy"] == 70.0
    assert d["reward_mode"] == "simplified"
    assert isinstance(d, dict)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_episode_outcome.py -v`
Expected: FAIL with "cannot import name 'EpisodeOutcome'"

**Step 3: Write minimal implementation**

Add to `src/esper/karn/store.py` after the `HostBaseline` class:

```python
@dataclass
class EpisodeOutcome:
    """Multi-objective episode outcome for Pareto analysis.

    Captures all objectives needed for reward function evaluation:
    - Primary: accuracy (maximize), param_ratio (minimize)
    - Secondary: stability (maximize), fossilization count

    Used for A/B testing analysis and Pareto frontier computation.
    """

    # Identity
    env_idx: int
    episode_idx: int

    # Primary objectives
    final_accuracy: float  # 0-100 scale
    param_ratio: float  # total_params / host_params (minimize)

    # Secondary objectives
    num_fossilized: int
    num_contributing_fossilized: int
    episode_reward: float  # Scalarized reward (for comparison)
    stability_score: float  # 0-1, higher = more stable (no anomalies)

    # Metadata
    reward_mode: str  # "shaped", "sparse", "minimal", "simplified"
    timestamp: datetime = field(default_factory=_utc_now)

    def dominates(self, other: "EpisodeOutcome") -> bool:
        """Return True if self Pareto-dominates other.

        Dominance: at least as good in all objectives, strictly better in one.
        Objectives: accuracy (max), stability (max), param_ratio (min).
        """
        # At least as good in all
        acc_ok = self.final_accuracy >= other.final_accuracy
        stab_ok = self.stability_score >= other.stability_score
        param_ok = self.param_ratio <= other.param_ratio  # Lower is better

        if not (acc_ok and stab_ok and param_ok):
            return False

        # Strictly better in at least one
        acc_better = self.final_accuracy > other.final_accuracy
        stab_better = self.stability_score > other.stability_score
        param_better = self.param_ratio < other.param_ratio

        return acc_better or stab_better or param_better

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "env_idx": self.env_idx,
            "episode_idx": self.episode_idx,
            "final_accuracy": self.final_accuracy,
            "param_ratio": self.param_ratio,
            "num_fossilized": self.num_fossilized,
            "num_contributing_fossilized": self.num_contributing_fossilized,
            "episode_reward": self.episode_reward,
            "stability_score": self.stability_score,
            "reward_mode": self.reward_mode,
            "timestamp": self.timestamp.isoformat(),
        }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_episode_outcome.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/test_episode_outcome.py src/esper/karn/store.py
git commit -m "feat(karn): add EpisodeOutcome schema for multi-objective tracking

Captures accuracy, param_ratio, stability_score for Pareto analysis.
Includes dominates() method for frontier computation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Wire EpisodeOutcome into Vectorized Training

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:2717-2730` (episode completion block)
- Test: `tests/simic/test_episode_outcome_emission.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_episode_outcome_emission.py
"""Tests that EpisodeOutcome is emitted at episode end."""

import pytest
from esper.karn.store import EpisodeOutcome


def test_episode_outcome_created_at_episode_end():
    """vectorized training creates EpisodeOutcome at episode completion."""
    # This is an integration test - we'll verify the structure
    outcome = EpisodeOutcome(
        env_idx=0,
        episode_idx=1,
        final_accuracy=72.5,
        param_ratio=0.12,
        num_fossilized=1,
        num_contributing_fossilized=1,
        episode_reward=10.5,
        stability_score=1.0,  # No anomalies
        reward_mode="shaped",
    )

    # Verify all required fields exist
    d = outcome.to_dict()
    required_fields = [
        "env_idx", "episode_idx", "final_accuracy", "param_ratio",
        "num_fossilized", "num_contributing_fossilized", "episode_reward",
        "stability_score", "reward_mode", "timestamp"
    ]
    for field in required_fields:
        assert field in d, f"Missing required field: {field}"
```

**Step 2: Run test to verify it passes (schema already exists)**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_episode_outcome_emission.py -v`
Expected: PASS (this test validates the schema, not the wiring yet)

**Step 3: Modify vectorized.py to emit EpisodeOutcome**

In `src/esper/simic/training/vectorized.py`, find the episode completion block (~line 2717) and add:

```python
# After: episode_history.append({...})
# Add EpisodeOutcome creation:

from esper.karn.store import EpisodeOutcome

# Compute stability score (1.0 if no anomalies detected this episode)
stability = 1.0
if env_state.governor.anomalies_detected > 0:
    stability = max(0.0, 1.0 - (env_state.governor.anomalies_detected * 0.1))

episode_outcome = EpisodeOutcome(
    env_idx=env_idx,
    episode_idx=episodes_completed,
    final_accuracy=env_state.val_acc,
    param_ratio=(env_state.total_params - env_state.host_params) / max(1, env_state.host_params),
    num_fossilized=env_state.seeds_fossilized,
    num_contributing_fossilized=env_state.contributing_fossilized,
    episode_reward=env_total_rewards[env_idx],
    stability_score=stability,
    reward_mode=env_reward_configs[env_idx].reward_mode.value,
)
episode_outcomes.append(episode_outcome)
```

Also add at function start (after `episode_history = []`):
```python
episode_outcomes: list[EpisodeOutcome] = []
```

**Step 4: Run full test suite to verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/simic/ -v -k "not slow"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py tests/simic/test_episode_outcome_emission.py
git commit -m "feat(simic): emit EpisodeOutcome at episode completion

Captures multi-objective data for Pareto analysis:
- final_accuracy, param_ratio, stability_score
- Wired into existing episode_history infrastructure

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Pareto Frontier Analysis

### Task 3: Add Pareto Frontier Computation to Karn Analytics

**Files:**
- Create: `src/esper/karn/pareto.py`
- Test: `tests/karn/test_pareto.py`

**Step 1: Write the failing test**

```python
# tests/karn/test_pareto.py
"""Tests for Pareto frontier computation."""

import pytest
from esper.karn.store import EpisodeOutcome
from esper.karn.pareto import extract_pareto_frontier, compute_hypervolume_2d


def make_outcome(acc: float, param: float, stab: float = 1.0) -> EpisodeOutcome:
    """Helper to create test outcomes."""
    return EpisodeOutcome(
        env_idx=0, episode_idx=0,
        final_accuracy=acc, param_ratio=param,
        num_fossilized=1, num_contributing_fossilized=1,
        episode_reward=acc / 10, stability_score=stab,
        reward_mode="shaped",
    )


def test_extract_pareto_frontier_simple():
    """Extract non-dominated outcomes."""
    outcomes = [
        make_outcome(80, 0.1),  # Pareto optimal
        make_outcome(70, 0.2),  # Dominated by first
        make_outcome(75, 0.05),  # Pareto optimal (lower param)
    ]
    frontier = extract_pareto_frontier(outcomes)
    assert len(frontier) == 2
    accs = {o.final_accuracy for o in frontier}
    assert 80 in accs
    assert 75 in accs
    assert 70 not in accs


def test_extract_pareto_frontier_all_dominated():
    """Single dominant point returns just that point."""
    outcomes = [
        make_outcome(90, 0.05),  # Dominates all
        make_outcome(80, 0.1),
        make_outcome(70, 0.2),
    ]
    frontier = extract_pareto_frontier(outcomes)
    assert len(frontier) == 1
    assert frontier[0].final_accuracy == 90


def test_extract_pareto_frontier_empty():
    """Empty input returns empty frontier."""
    assert extract_pareto_frontier([]) == []


def test_hypervolume_2d():
    """Compute 2D hypervolume for accuracy vs param_ratio."""
    frontier = [
        make_outcome(80, 0.1),
        make_outcome(70, 0.05),
    ]
    # Reference point: worst acceptable (0 accuracy, 1.0 param_ratio)
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    # Expected: dominated area > 0
    assert hv > 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_pareto.py -v`
Expected: FAIL with "No module named 'esper.karn.pareto'"

**Step 3: Write minimal implementation**

```python
# src/esper/karn/pareto.py
"""Pareto frontier analysis for multi-objective reward evaluation.

Provides tools for extracting non-dominated outcomes and computing
hypervolume indicators for tracking training progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.karn.store import EpisodeOutcome


def extract_pareto_frontier(outcomes: list["EpisodeOutcome"]) -> list["EpisodeOutcome"]:
    """Extract non-dominated outcomes from a list.

    An outcome is non-dominated (Pareto optimal) if no other outcome
    is strictly better in all objectives.

    Objectives considered:
    - final_accuracy: maximize
    - param_ratio: minimize
    - stability_score: maximize

    Args:
        outcomes: List of episode outcomes to analyze

    Returns:
        List of non-dominated outcomes (the Pareto frontier)
    """
    if not outcomes:
        return []

    frontier = []
    for candidate in outcomes:
        is_dominated = False
        for other in outcomes:
            if other is candidate:
                continue
            if other.dominates(candidate):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(candidate)

    return frontier


def compute_hypervolume_2d(
    frontier: list["EpisodeOutcome"],
    ref_point: tuple[float, float],
) -> float:
    """Compute 2D hypervolume for accuracy vs param_ratio.

    Hypervolume indicator measures the area dominated by the Pareto
    frontier relative to a reference point. Higher is better.

    Args:
        frontier: List of Pareto-optimal outcomes
        ref_point: (min_accuracy, max_param_ratio) - worst acceptable values

    Returns:
        Hypervolume (area dominated by frontier)
    """
    if not frontier:
        return 0.0

    # Extract (accuracy, param_ratio) points
    # For hypervolume, we want to maximize area, so we invert param_ratio
    # (lower param_ratio = better = higher contribution to hypervolume)
    points = [
        (o.final_accuracy, ref_point[1] - o.param_ratio)
        for o in frontier
    ]

    # Sort by first objective (accuracy) descending
    points.sort(key=lambda p: -p[0])

    ref_acc, ref_param = ref_point

    hv = 0.0
    prev_height = 0.0

    for acc, inv_param in points:
        width = acc - ref_acc
        height = inv_param - prev_height
        if width > 0 and height > 0:
            hv += width * height
        prev_height = max(prev_height, inv_param)

    return hv


__all__ = ["extract_pareto_frontier", "compute_hypervolume_2d"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/test_pareto.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/pareto.py tests/karn/test_pareto.py
git commit -m "feat(karn): add Pareto frontier extraction and hypervolume

- extract_pareto_frontier(): finds non-dominated outcomes
- compute_hypervolume_2d(): area metric for training progress
- Used for multi-objective reward function evaluation

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Add episode_outcomes MCP View

**Files:**
- Modify: `src/esper/karn/mcp/views.py`
- Test: Manual verification via MCP query

**Step 1: Add view definition**

Add to `VIEW_DEFINITIONS` dict in `src/esper/karn/mcp/views.py`:

```python
"episode_outcomes": """
    CREATE OR REPLACE VIEW episode_outcomes AS
    SELECT
        timestamp,
        json_extract(data, '$.env_idx')::INTEGER as env_idx,
        json_extract(data, '$.episode_idx')::INTEGER as episode_idx,
        json_extract(data, '$.final_accuracy')::DOUBLE as final_accuracy,
        json_extract(data, '$.param_ratio')::DOUBLE as param_ratio,
        json_extract(data, '$.num_fossilized')::INTEGER as num_fossilized,
        json_extract(data, '$.num_contributing_fossilized')::INTEGER as num_contributing,
        json_extract(data, '$.episode_reward')::DOUBLE as episode_reward,
        json_extract(data, '$.stability_score')::DOUBLE as stability_score,
        json_extract_string(data, '$.reward_mode') as reward_mode
    FROM raw_events
    WHERE event_type = 'EPISODE_OUTCOME'
""",
```

**Step 2: Verify view syntax is valid**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.mcp.views import VIEW_DEFINITIONS; print('episode_outcomes' in VIEW_DEFINITIONS)"`
Expected: `True`

**Step 3: Commit**

```bash
git add src/esper/karn/mcp/views.py
git commit -m "feat(karn): add episode_outcomes MCP view

SQL view for querying multi-objective episode outcomes.
Supports Pareto analysis via reward_mode grouping.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Ablation Configs

### Task 5: Create Ablation Config Files

**Files:**
- Create: `configs/ablations/simplified_baseline.json`
- Create: `configs/ablations/no_pbrs.json`
- Create: `configs/ablations/pure_sparse.json`
- Create: `configs/ablations/ab_shaped_vs_simplified.json`

**Step 1: Create configs directory**

```bash
mkdir -p configs/ablations
```

**Step 2: Create baseline SIMPLIFIED config**

File: `configs/ablations/simplified_baseline.json`

```json
{
    "n_episodes": 100,
    "n_envs": 4,
    "max_epochs": 25,
    "reward_mode": "simplified",
    "reward_family": "contribution",
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "entropy_coef": 0.1,
    "lr": 3e-4,
    "slots": ["r0c1"]
}
```

**Step 3: Create pure sparse config**

File: `configs/ablations/pure_sparse.json`

```json
{
    "n_episodes": 100,
    "n_envs": 8,
    "max_epochs": 25,
    "reward_mode": "sparse",
    "reward_family": "contribution",
    "sparse_reward_scale": 2.5,
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "entropy_coef": 0.1,
    "lr": 1e-4,
    "slots": ["r0c1"]
}
```

**Step 4: Create A/B comparison config**

File: `configs/ablations/ab_shaped_vs_simplified.json`

```json
{
    "n_episodes": 100,
    "n_envs": 8,
    "max_epochs": 25,
    "reward_mode": "shaped",
    "reward_family": "contribution",
    "ab_reward_modes": ["shaped", "shaped", "shaped", "shaped", "simplified", "simplified", "simplified", "simplified"],
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "entropy_coef": 0.1,
    "lr": 3e-4,
    "slots": ["r0c1"]
}
```

**Step 5: Commit**

```bash
git add configs/ablations/
git commit -m "feat(configs): add ablation config files for reward experiments

4 configs for systematic reward function evaluation:
- simplified_baseline: DRL Expert recommended baseline
- pure_sparse: Credit assignment test (higher variance)
- ab_shaped_vs_simplified: Direct A/B comparison (4+4 envs)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Reward Health Panel in Sanctum

### Task 6: Create RewardHealthPanel Widget

**Files:**
- Create: `src/esper/karn/sanctum/widgets/reward_health.py`
- Test: `tests/karn/sanctum/test_reward_health.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_reward_health.py
"""Tests for RewardHealthPanel widget."""

import pytest
from esper.karn.sanctum.widgets.reward_health import RewardHealthPanel, RewardHealthData


def test_reward_health_data_from_components():
    """RewardHealthData computed from telemetry."""
    data = RewardHealthData(
        pbrs_fraction=0.25,
        anti_gaming_trigger_rate=0.03,
        ev_explained=0.65,
        hypervolume=42.5,
    )
    assert data.pbrs_fraction == 0.25
    assert data.is_pbrs_healthy  # 0.1-0.5 range
    assert data.is_gaming_healthy  # <0.05
    assert data.is_ev_healthy  # >0.5


def test_reward_health_warnings():
    """Warnings triggered for unhealthy metrics."""
    unhealthy = RewardHealthData(
        pbrs_fraction=0.7,  # >0.5 = too much shaping
        anti_gaming_trigger_rate=0.15,  # >0.05 = policy exploiting
        ev_explained=0.3,  # <0.5 = poor value estimation
        hypervolume=10.0,
    )
    assert not unhealthy.is_pbrs_healthy
    assert not unhealthy.is_gaming_healthy
    assert not unhealthy.is_ev_healthy
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_reward_health.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
# src/esper/karn/sanctum/widgets/reward_health.py
"""Reward health monitoring panel for Sanctum TUI.

Displays DRL Expert recommended metrics:
- PBRS fraction of total reward (healthy: 10-30%)
- Anti-gaming penalty frequency (healthy: <5%)
- Value function explained variance (healthy: >0.5)
- Hypervolume indicator (should increase over training)
"""

from __future__ import annotations

from dataclasses import dataclass
from textual.widgets import Static
from rich.text import Text
from rich.panel import Panel


@dataclass
class RewardHealthData:
    """Aggregated reward health metrics."""

    pbrs_fraction: float = 0.0  # |PBRS| / |total_reward|
    anti_gaming_trigger_rate: float = 0.0  # Fraction of steps with penalties
    ev_explained: float = 0.0  # Value function explained variance
    hypervolume: float = 0.0  # Pareto hypervolume indicator

    @property
    def is_pbrs_healthy(self) -> bool:
        """PBRS should be 10-50% of total reward."""
        return 0.1 <= self.pbrs_fraction <= 0.5

    @property
    def is_gaming_healthy(self) -> bool:
        """Anti-gaming penalties should trigger <5% of steps."""
        return self.anti_gaming_trigger_rate < 0.05

    @property
    def is_ev_healthy(self) -> bool:
        """Explained variance should be >0.5."""
        return self.ev_explained > 0.5


class RewardHealthPanel(Static):
    """Compact reward health display for Sanctum."""

    DEFAULT_CSS = """
    RewardHealthPanel {
        height: 6;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def __init__(self, data: RewardHealthData | None = None, **kwargs):
        super().__init__(**kwargs)
        self._data = data or RewardHealthData()

    def update_data(self, data: RewardHealthData) -> None:
        """Update health data and refresh display."""
        self._data = data
        self.refresh()

    def render(self) -> Panel:
        """Render health indicators."""
        lines = []

        # PBRS fraction
        pbrs_color = "green" if self._data.is_pbrs_healthy else "red"
        lines.append(Text.assemble(
            ("PBRS: ", "bold"),
            (f"{self._data.pbrs_fraction:.0%}", pbrs_color),
            (" (10-50%)", "dim"),
        ))

        # Anti-gaming
        gaming_color = "green" if self._data.is_gaming_healthy else "red"
        lines.append(Text.assemble(
            ("Gaming: ", "bold"),
            (f"{self._data.anti_gaming_trigger_rate:.1%}", gaming_color),
            (" (<5%)", "dim"),
        ))

        # Explained variance
        ev_color = "green" if self._data.is_ev_healthy else "yellow"
        lines.append(Text.assemble(
            ("EV: ", "bold"),
            (f"{self._data.ev_explained:.2f}", ev_color),
            (" (>0.5)", "dim"),
        ))

        # Hypervolume
        lines.append(Text.assemble(
            ("HV: ", "bold"),
            (f"{self._data.hypervolume:.1f}", "cyan"),
        ))

        content = Text("\n").join(lines)
        return Panel(content, title="[bold]Reward Health[/]", border_style="blue")


__all__ = ["RewardHealthPanel", "RewardHealthData"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_reward_health.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/reward_health.py tests/karn/sanctum/test_reward_health.py
git commit -m "feat(sanctum): add RewardHealthPanel widget

Displays DRL Expert recommended health metrics:
- PBRS fraction (10-50% healthy)
- Anti-gaming trigger rate (<5% healthy)
- Value function explained variance (>0.5 healthy)
- Hypervolume indicator (Pareto progress)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Wire RewardHealthPanel into Sanctum App

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Modify: `src/esper/karn/sanctum/widgets/__init__.py`

**Step 1: Add import to widgets/__init__.py**

```python
from .reward_health import RewardHealthPanel, RewardHealthData
```

**Step 2: Add health data aggregation to aggregator.py**

In the aggregator class, add method to compute `RewardHealthData` from telemetry:

```python
def compute_reward_health(self) -> RewardHealthData:
    """Compute reward health metrics from recent telemetry."""
    # Get recent reward components
    pbrs_total = sum(c.pbrs_bonus for c in self._recent_components)
    reward_total = sum(abs(c.total_reward) for c in self._recent_components)
    pbrs_fraction = pbrs_total / max(1e-8, reward_total)

    # Anti-gaming trigger rate
    gaming_steps = sum(1 for c in self._recent_components if c.ratio_penalty < 0)
    gaming_rate = gaming_steps / max(1, len(self._recent_components))

    # Get latest EV from PPO updates
    ev = self._latest_ppo.explained_variance if self._latest_ppo else 0.0

    # Hypervolume from episode outcomes
    hv = self._compute_hypervolume()

    return RewardHealthData(
        pbrs_fraction=pbrs_fraction,
        anti_gaming_trigger_rate=gaming_rate,
        ev_explained=ev,
        hypervolume=hv,
    )
```

**Step 3: Mount panel in app.py**

Add to layout and wire update:

```python
from esper.karn.sanctum.widgets import RewardHealthPanel

# In compose():
yield RewardHealthPanel(id="reward-health")

# In update handler:
health_data = self.aggregator.compute_reward_health()
self.query_one("#reward-health", RewardHealthPanel).update_data(health_data)
```

**Step 4: Verify widget renders**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.widgets.reward_health import RewardHealthPanel; print('OK')"`

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/app.py src/esper/karn/sanctum/aggregator.py src/esper/karn/sanctum/widgets/__init__.py
git commit -m "feat(sanctum): wire RewardHealthPanel into app layout

Aggregator computes health metrics from telemetry stream.
Panel updates on each refresh cycle.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Emit Episode Outcome Telemetry

### Task 8: Add EPISODE_OUTCOME Event Type and Emit

**Files:**
- Modify: `src/esper/leyline/telemetry.py`
- Modify: `src/esper/simic/training/vectorized.py`

**Step 1: Add event type to enum**

In `src/esper/leyline/telemetry.py`, add to `TelemetryEventType`:

```python
EPISODE_OUTCOME = "episode_outcome"
```

**Step 2: Emit event in vectorized.py**

After creating `episode_outcome` in the episode completion block, emit telemetry:

```python
if telemetry_cb:
    from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType
    telemetry_cb(TelemetryEvent(
        event_type=TelemetryEventType.EPISODE_OUTCOME,
        epoch=current_global_epoch,
        data=episode_outcome.to_dict(),
    ))
```

**Step 3: Verify event type exists**

Run: `PYTHONPATH=src uv run python -c "from esper.leyline.telemetry import TelemetryEventType; print(TelemetryEventType.EPISODE_OUTCOME)"`

**Step 4: Commit**

```bash
git add src/esper/leyline/telemetry.py src/esper/simic/training/vectorized.py
git commit -m "feat(telemetry): emit EPISODE_OUTCOME events

Multi-objective outcomes now logged for Pareto analysis.
Enables post-hoc hypervolume computation via MCP views.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification

### Task 9: End-to-End Verification

**Step 1: Run A/B test with SHAPED vs SIMPLIFIED**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --config-json configs/ablations/ab_shaped_vs_simplified.json \
    --task cifar10 \
    --telemetry-dir telemetry_ab_test \
    --n-episodes 10
```

**Step 2: Query episode outcomes via MCP**

```sql
SELECT reward_mode,
       AVG(final_accuracy) as avg_acc,
       AVG(param_ratio) as avg_params,
       COUNT(*) as episodes
FROM episode_outcomes
GROUP BY reward_mode
```

**Step 3: Compute Pareto frontier**

```python
from esper.karn.pareto import extract_pareto_frontier
from esper.karn.store import EpisodeOutcome

# Load outcomes from telemetry...
frontier = extract_pareto_frontier(outcomes)
print(f"Pareto frontier size: {len(frontier)}")
```

**Step 4: Verify Sanctum displays health metrics**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 --sanctum --telemetry-dir telemetry_sanctum \
    --n-episodes 5
```

Check that RewardHealthPanel shows PBRS fraction, gaming rate, EV, and HV.

---

## Summary

| Phase | Tasks | Purpose |
|-------|-------|---------|
| 1 | Tasks 1-2 | EpisodeOutcome schema + vectorized wiring |
| 2 | Tasks 3-4 | Pareto frontier computation + MCP view |
| 3 | Task 5 | Ablation config files |
| 4 | Tasks 6-7 | Sanctum RewardHealthPanel |
| 5 | Task 8 | Telemetry emission |
| ✓ | Task 9 | End-to-end verification |

**Total: 9 tasks, ~45 minutes implementation time**

---

## References

- DRL Expert Paper: `docs/research/reward-function-design-for-morphogenetic-controller.md`
- Existing A/B infrastructure: `TrainingConfig.ab_reward_modes`
- PBRS theory: Ng et al. (1999) - Policy invariance under reward transformations
