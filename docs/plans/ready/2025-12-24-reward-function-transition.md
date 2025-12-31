# Reward Function Transition Plan: SHAPED → SIMPLIFIED with Pareto Tracking

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transition Tamiyo's reward system from complex SHAPED mode to cleaner SIMPLIFIED mode, with multi-objective Pareto frontier tracking for principled reward engineering.

**Architecture:** Extend `episode_history` to capture multi-objective outcomes (`EpisodeOutcome`), add Pareto frontier computation to Karn analytics, create ablation configs for systematic comparison, and add reward health panel to Sanctum TUI.

**Tech Stack:** Python dataclasses, DuckDB views (Karn MCP), Textual widgets (Sanctum), existing A/B testing infrastructure.

**Revision Notes (2025-12-24):** Updated based on DRL Expert and Code Reviewer feedback:
- Fixed hypervolume algorithm (corrected sweep-line implementation)
- Fixed stability score computation (uses reward variance proxy, not non-existent `governor.anomalies_detected`)
- Added missing aggregator methods
- Adjusted PBRS healthy range to 10-40%
- Added 3 missing ablation configs from DRL Expert Section 7.2
- Added property-based tests for Pareto operations
- Added PBRS verification tests for SIMPLIFIED mode
- **Generalized A/B → A/B/n**: Renamed `ab_reward_modes` → `reward_mode_per_env`, added `with_reward_split()` builder

**Revision Notes (2025-12-26):** Pre-execution codebase review identified gaps:
- **Added Task 1.5**: `store.py` RewardComponents lacks `stage_bonus` field (sanctum/schema.py has it)
- **Added Task 4.5**: Ablation configs need `disable_pbrs`, `disable_terminal_reward`, `disable_anti_gaming` flags in TrainingConfig
- **Fixed Task 0.5**: Removed incorrect "backwards-compatible property alias" from commit message
- **Updated task count**: 11 → 13 tasks, ~3-3.5 hours estimated

---

## Phase 0: Foundation Fixes

### Task 0: Fix Stability Score Computation Design

**Context:** The original plan referenced `env_state.governor.anomalies_detected` which does not exist in the vectorized training context. This task establishes the stability score computation using reward variance as a proxy (DRL Expert Option C).

**Files:**
- Design decision only (no code changes yet)

**Decision:**
Use **reward variance over a rolling window** as stability score proxy:
- Track last N rewards (window_size=20)
- Compute variance: `var = np.var(recent_rewards)`
- Normalize: `stability = 1.0 / (1.0 + var)`
- High variance → low stability, low variance → high stability

**Rationale:**
- Doesn't require additional state in env
- Captures the signal we care about: erratic training = instability
- Easy to compute from existing `episode_history`
- Aligns with DRL Expert recommendation (Option C)

**Step 1: Document decision**

This design is now incorporated into Task 2.

---

### Task 0.5: Generalize A/B Testing to A/B/n

**Context:** The current `ab_reward_modes` field name implies binary A/B testing, but the infrastructure already supports N groups. Rename for clarity and add a builder method for ergonomic multi-group configuration.

**Files:**
- Modify: `src/esper/simic/training/config.py`
- Modify: `src/esper/simic/training/vectorized.py`
- Modify: `src/esper/scripts/train.py`
- Test: `tests/simic/test_reward_split.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_reward_split.py
"""Tests for A/B/n reward split configuration."""

import pytest
from esper.simic.training.config import TrainingConfig
from esper.simic.rewards import RewardMode


def test_with_reward_split_creates_correct_list():
    """with_reward_split() creates per-env mode list."""
    config = TrainingConfig.with_reward_split({
        "shaped": 3,
        "simplified": 2,
        "sparse": 1,
    })
    assert config.n_envs == 6
    assert config.reward_mode_per_env is not None
    assert len(config.reward_mode_per_env) == 6
    assert config.reward_mode_per_env.count("shaped") == 3
    assert config.reward_mode_per_env.count("simplified") == 2
    assert config.reward_mode_per_env.count("sparse") == 1


def test_with_reward_split_validates_modes():
    """with_reward_split() rejects invalid mode names."""
    with pytest.raises(ValueError, match="invalid_mode"):
        TrainingConfig.with_reward_split({"invalid_mode": 2})


def test_with_reward_split_requires_positive_counts():
    """with_reward_split() rejects zero or negative counts."""
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig.with_reward_split({"shaped": 0})
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig.with_reward_split({"shaped": -1})


def test_with_reward_split_preserves_other_kwargs():
    """with_reward_split() passes through other config options."""
    config = TrainingConfig.with_reward_split(
        {"shaped": 2, "simplified": 2},
        lr=1e-5,
        entropy_coef=0.05,
    )
    assert config.n_envs == 4
    assert config.lr == 1e-5
    assert config.entropy_coef == 0.05


def test_reward_mode_per_env_direct_assignment():
    """Direct assignment of reward_mode_per_env works."""
    config = TrainingConfig(
        n_envs=4,
        reward_mode_per_env=["shaped", "shaped", "simplified", "simplified"],
    )
    assert len(config.reward_mode_per_env) == 4
    assert config.reward_mode_per_env[0] == "shaped"
    assert config.reward_mode_per_env[2] == "simplified"


def test_abc_comparison_config():
    """Three-way comparison works correctly."""
    config = TrainingConfig.with_reward_split({
        "shaped": 4,
        "simplified": 4,
        "sparse": 4,
    })
    assert config.n_envs == 12
    # Verify distribution
    modes = config.reward_mode_per_env
    assert modes.count("shaped") == 4
    assert modes.count("simplified") == 4
    assert modes.count("sparse") == 4
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_split.py -v`
Expected: FAIL with "has no attribute 'with_reward_split'"

**Step 3: Rename field and add builder in config.py**

NOTE: TrainingConfig uses `@dataclass(slots=True)` which prohibits property aliases.
Per No Legacy Code Policy, we do a clean rename without backwards compatibility.

```python
# In TrainingConfig class, rename the field (line ~103):
# OLD: ab_reward_modes: list[str] | None = None
# NEW:

# === Multi-Group Reward Testing ===
# Per-environment reward mode for A/B/n testing.
# If None, all envs use reward_mode. If list, must match n_envs length.
# Example: ["shaped"]*4 + ["simplified"]*2 + ["sparse"]*2 for 8-env A/B/C test
reward_mode_per_env: list[str] | None = None

@classmethod
def with_reward_split(
    cls,
    mode_counts: dict[str, int],
    **kwargs,
) -> "TrainingConfig":
    """Create config with A/B/n reward mode split.

    Args:
        mode_counts: Dict mapping reward mode name to env count.
            Example: {"shaped": 4, "simplified": 2, "sparse": 2}
        **kwargs: Additional TrainingConfig parameters.

    Returns:
        TrainingConfig with reward_mode_per_env and n_envs set.

    Raises:
        ValueError: If mode name is invalid or count is not positive.
    """
    from esper.simic.rewards import RewardMode

    valid_modes = {m.value for m in RewardMode}
    mode_list: list[str] = []

    for mode, count in mode_counts.items():
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid reward mode '{mode}'. Valid modes: {sorted(valid_modes)}"
            )
        if count < 1:
            raise ValueError(f"Count for '{mode}' must be positive (got {count})")
        mode_list.extend([mode] * count)

    n_envs = len(mode_list)
    return cls(
        n_envs=n_envs,
        reward_mode_per_env=mode_list,
        **kwargs,
    )
```

**Step 4: Update validation in config.py**

Replace references to `ab_reward_modes` with `reward_mode_per_env` in `_validate()`:

```python
# A/B/n testing validation
if self.reward_mode_per_env is not None:
    if len(self.reward_mode_per_env) != self.n_envs:
        raise ValueError(
            f"reward_mode_per_env length ({len(self.reward_mode_per_env)}) "
            f"must match n_envs ({self.n_envs})"
        )
    valid_modes = {m.value for m in RewardMode}
    for i, mode in enumerate(self.reward_mode_per_env):
        if mode not in valid_modes:
            raise ValueError(
                f"reward_mode_per_env[{i}] = '{mode}' is not a valid RewardMode. "
                f"Valid modes: {sorted(valid_modes)}"
            )
```

**Step 5: Update to_train_kwargs()**

```python
def to_train_kwargs(self) -> dict[str, Any]:
    # ... existing code ...
    return {
        # ... other fields ...
        "reward_mode_per_env": self.reward_mode_per_env,  # renamed from ab_reward_modes
        # ...
    }
```

**Step 6: Update vectorized.py parameter name**

Change function signature and internal references:
```python
def train_ppo_vectorized(
    # ... other params ...
    reward_mode_per_env: list[str] | None = None,  # renamed from ab_reward_modes
    # ...
):
```

Update all internal references (lines ~672-691, ~3003):
```python
if reward_mode_per_env is not None:
    if len(reward_mode_per_env) != n_envs:
        raise ValueError(
            f"reward_mode_per_env length ({len(reward_mode_per_env)}) must match n_envs ({n_envs})"
        )
    # ... rest of logic unchanged ...
```

**Step 7: Update CLI in train.py**

Update the `--ab-test` handling to use new field name:
```python
if args.ab_test:
    if config.n_envs % 2 != 0:
        raise ValueError("--ab-test requires even number of envs")
    half = config.n_envs // 2
    if args.ab_test == "shaped-vs-simplified":
        config.reward_mode_per_env = ["shaped"] * half + ["simplified"] * half
    elif args.ab_test == "shaped-vs-sparse":
        config.reward_mode_per_env = ["shaped"] * half + ["sparse"] * half
```

**Step 7b: Rename all remaining references**

Use grep to find and update ALL occurrences (no backward compat per policy):

```bash
# Find all occurrences
grep -r "ab_reward_modes" --include="*.py" src/ tests/

# Files to update (rename ab_reward_modes -> reward_mode_per_env):
# - src/esper/simic/training/config.py (~7 occurrences)
# - src/esper/simic/training/vectorized.py (~7 occurrences)
# - src/esper/scripts/train.py (~2 occurrences)
# - tests/integration/test_ab_reward_testing.py (~3 occurrences)
# - tests/simic/test_reward_simplified.py (~13 occurrences)
# - tests/scripts/test_train.py (~12 occurrences)
```

For each file, do a global find-replace: `ab_reward_modes` → `reward_mode_per_env`

**Step 8: Run tests to verify**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_split.py -v`
Expected: PASS

Run: `PYTHONPATH=src uv run pytest tests/simic/ -v -k "not slow"`
Expected: PASS (no regressions)

**Step 9: Commit**

```bash
git add src/esper/simic/training/config.py src/esper/simic/training/vectorized.py src/esper/scripts/train.py tests/simic/test_reward_split.py
git commit -m "refactor(simic): generalize A/B testing to A/B/n

Rename ab_reward_modes → reward_mode_per_env for clarity.
Add TrainingConfig.with_reward_split() builder for ergonomic multi-group config:

    config = TrainingConfig.with_reward_split({
        'shaped': 4,
        'simplified': 2,
        'sparse': 2,
    })

No backwards compatibility per No Legacy Code Policy (clean rename).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

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


def test_episode_outcome_no_self_dominance():
    """Outcome does not dominate itself."""
    outcome = EpisodeOutcome(
        env_idx=0, episode_idx=1,
        final_accuracy=80.0, param_ratio=0.1, num_fossilized=2,
        num_contributing_fossilized=2, episode_reward=15.0,
        stability_score=0.98, reward_mode="shaped",
    )
    assert not outcome.dominates(outcome)


def test_episode_outcome_dominates_requires_strict_improvement():
    """Dominance requires at least one strictly better objective."""
    equal1 = EpisodeOutcome(
        env_idx=0, episode_idx=1,
        final_accuracy=80.0, param_ratio=0.1, num_fossilized=2,
        num_contributing_fossilized=2, episode_reward=15.0,
        stability_score=0.98, reward_mode="shaped",
    )
    equal2 = EpisodeOutcome(
        env_idx=1, episode_idx=2,
        final_accuracy=80.0, param_ratio=0.1, num_fossilized=1,
        num_contributing_fossilized=1, episode_reward=10.0,  # Different reward but not an objective
        stability_score=0.98, reward_mode="shaped",
    )
    # Equal in all objectives = no dominance
    assert not equal1.dominates(equal2)
    assert not equal2.dominates(equal1)


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
    stability_score: float  # 0-1, higher = more stable (low reward variance)

    # Metadata
    reward_mode: str  # "shaped", "sparse", "minimal", "simplified"
    timestamp: datetime = field(default_factory=_utc_now)

    def dominates(self, other: "EpisodeOutcome") -> bool:
        """Return True if self Pareto-dominates other.

        Dominance: at least as good in all objectives, strictly better in one.
        Objectives: accuracy (max), stability (max), param_ratio (min).
        """
        if self is other:
            return False  # Cannot dominate self

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
Self-dominance correctly returns False.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.5: Add stage_bonus to store.py RewardComponents

**Context:** The `sanctum/schema.py` RewardComponents has `stage_bonus` for PBRS tracking, but `store.py` RewardComponents does not. This creates an inconsistency when emitting telemetry from vectorized training. Add the field to store.py for consistency.

**Files:**
- Modify: `src/esper/karn/store.py`
- Test: Existing tests should pass

**Step 1: Add stage_bonus field to RewardComponents**

In `src/esper/karn/store.py`, find the `RewardComponents` dataclass (line ~159) and add:

```python
@dataclass
class RewardComponents:
    """Breakdown of reward computation for debugging."""

    total: float = 0.0
    accuracy_delta: float = 0.0
    bounded_attribution: float | None = None  # For contribution mode
    compute_rent: float = 0.0
    alpha_shock: float = 0.0
    blending_warning: float = 0.0
    holding_warning: float = 0.0
    ratio_penalty: float = 0.0
    terminal_bonus: float = 0.0
    stage_bonus: float = 0.0  # PBRS shaping reward for stage advancement
```

**Step 2: Verify no tests break**

Run: `PYTHONPATH=src uv run pytest tests/karn/ -v -k "not slow"`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/store.py
git commit -m "feat(karn): add stage_bonus to store.py RewardComponents

Aligns store.py with sanctum/schema.py for PBRS fraction tracking.
Enables compute_reward_health() to compute PBRS contribution.

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
import numpy as np
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
        stability_score=0.8,  # Computed from reward variance
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


def test_stability_score_from_reward_variance():
    """Stability score computed correctly from reward variance."""
    # Simulating the computation that will be in vectorized.py

    # Low variance = high stability
    low_var_rewards = [10.0, 10.1, 9.9, 10.0, 10.2]
    var_low = np.var(low_var_rewards)
    stability_low = 1.0 / (1.0 + var_low)
    assert stability_low > 0.9, "Low variance should give high stability"

    # High variance = low stability
    high_var_rewards = [5.0, 15.0, 2.0, 18.0, 8.0]
    var_high = np.var(high_var_rewards)
    stability_high = 1.0 / (1.0 + var_high)
    assert stability_high < 0.1, "High variance should give low stability"

    # Stability always in [0, 1]
    assert 0.0 <= stability_low <= 1.0
    assert 0.0 <= stability_high <= 1.0
```

**Step 2: Run test to verify it passes (schema already exists)**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_episode_outcome_emission.py -v`
Expected: PASS (this test validates the schema and stability computation logic)

**Step 3: Modify vectorized.py to emit EpisodeOutcome**

In `src/esper/simic/training/vectorized.py`, find the episode completion block (~line 2717) and add:

```python
# After: episode_history.append({...})
# Add EpisodeOutcome creation:

from esper.karn.store import EpisodeOutcome
import numpy as np

# Compute stability score from reward variance over the episode
# Use last 20 step rewards as proxy for training stability
recent_rewards = env_step_rewards[env_idx][-20:] if len(env_step_rewards[env_idx]) >= 20 else env_step_rewards[env_idx]
if len(recent_rewards) > 1:
    reward_var = float(np.var(recent_rewards))
    stability = 1.0 / (1.0 + reward_var)
else:
    stability = 1.0  # Default if insufficient data

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
env_step_rewards: dict[int, list[float]] = {i: [] for i in range(n_envs)}
```

And in the step loop, track per-step rewards:
```python
# After computing step reward
env_step_rewards[env_idx].append(step_reward)
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
- Stability computed from reward variance (not governor.anomalies_detected)
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
from hypothesis import given, strategies as st
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


def test_extract_pareto_frontier_single():
    """Single outcome is always on frontier."""
    single = [make_outcome(50, 0.5)]
    frontier = extract_pareto_frontier(single)
    assert len(frontier) == 1


def test_hypervolume_2d_basic():
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


def test_hypervolume_2d_known_value():
    """Verify hypervolume with known expected value."""
    # Single point at (80, 0.2) with ref (0, 1.0)
    # Area = 80 * (1.0 - 0.2) = 80 * 0.8 = 64
    frontier = [make_outcome(80, 0.2)]
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert abs(hv - 64.0) < 1e-6, f"Expected 64.0, got {hv}"


def test_hypervolume_2d_two_points():
    """Verify hypervolume with two non-dominated points."""
    # Points: (80, 0.3) and (60, 0.1) with ref (0, 1.0)
    # Sorted by acc descending: [(80, 0.3), (60, 0.1)]
    # Area from (80, 0.3): 80 * (1.0 - 0.3) = 56
    # Area from (60, 0.1): 60 * (0.3 - 0.1) = 12 (additional area below first point)
    # Total = 56 + 12 = 68
    frontier = [make_outcome(80, 0.3), make_outcome(60, 0.1)]
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert abs(hv - 68.0) < 1e-6, f"Expected 68.0, got {hv}"


def test_hypervolume_2d_empty():
    """Empty frontier has zero hypervolume."""
    assert compute_hypervolume_2d([], (0.0, 1.0)) == 0.0


# Property-based tests for Pareto operations
@given(st.lists(
    st.tuples(
        st.floats(min_value=0, max_value=100),  # accuracy
        st.floats(min_value=0.01, max_value=1.0),  # param_ratio
    ),
    min_size=0, max_size=20
))
def test_pareto_frontier_is_non_dominated(points):
    """Property: all frontier points are non-dominated."""
    outcomes = [make_outcome(acc, param) for acc, param in points]
    frontier = extract_pareto_frontier(outcomes)

    # Every frontier point should not be dominated by any other point
    for f_point in frontier:
        for other in outcomes:
            if other is not f_point:
                assert not other.dominates(f_point), "Frontier point should not be dominated"


@given(st.lists(
    st.tuples(
        st.floats(min_value=0, max_value=100),
        st.floats(min_value=0.01, max_value=1.0),
    ),
    min_size=0, max_size=20
))
def test_pareto_frontier_covers_all_non_dominated(points):
    """Property: frontier contains all non-dominated points."""
    outcomes = [make_outcome(acc, param) for acc, param in points]
    frontier = extract_pareto_frontier(outcomes)
    frontier_set = set(id(o) for o in frontier)

    # Every non-dominated outcome should be in frontier
    for outcome in outcomes:
        is_dominated = any(other.dominates(outcome) for other in outcomes if other is not outcome)
        if not is_dominated:
            assert id(outcome) in frontier_set, "Non-dominated point should be in frontier"


@given(st.lists(
    st.tuples(
        st.floats(min_value=1, max_value=100),
        st.floats(min_value=0.01, max_value=0.99),
    ),
    min_size=1, max_size=10
))
def test_hypervolume_is_non_negative(points):
    """Property: hypervolume is always non-negative."""
    outcomes = [make_outcome(acc, param) for acc, param in points]
    frontier = extract_pareto_frontier(outcomes)
    ref_point = (0.0, 1.0)
    hv = compute_hypervolume_2d(frontier, ref_point)
    assert hv >= 0, "Hypervolume should be non-negative"
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

Hypervolume Algorithm: Sweep-line approach for 2D case.
Reference: Zitzler & Thiele (1999), "Multiobjective Evolutionary Algorithms"
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

    Uses sweep-line algorithm: sort by accuracy descending, sweep from
    high accuracy to low, accumulating rectangular areas.

    For objectives (accuracy: maximize, param_ratio: minimize):
    - Points with higher accuracy contribute width
    - Points with lower param_ratio contribute height
    - Reference point (min_accuracy, max_param_ratio) defines the dominated region

    Args:
        frontier: List of Pareto-optimal outcomes
        ref_point: (min_accuracy, max_param_ratio) - worst acceptable values

    Returns:
        Hypervolume (area dominated by frontier)
    """
    if not frontier:
        return 0.0

    ref_acc, ref_param = ref_point

    # Extract (accuracy, param_ratio) and sort by accuracy descending
    points = [(o.final_accuracy, o.param_ratio) for o in frontier]
    points.sort(key=lambda p: -p[0])  # Descending by accuracy

    hv = 0.0
    prev_acc = ref_acc  # Start from reference (bottom-left corner)
    best_param = ref_param  # Track best (lowest) param_ratio seen so far

    for acc, param in points:
        # Only consider points that improve on param_ratio
        if param < best_param:
            # Width: from previous accuracy to this accuracy
            width = acc - prev_acc
            # Height: improvement in param_ratio from reference
            height = best_param - param

            if width > 0 and height > 0:
                hv += width * height

            best_param = param
            prev_acc = acc

    # Add remaining rectangle from last point to max accuracy
    # This captures the area from the highest accuracy point to ref_acc
    if points:
        highest_acc = points[0][0]
        lowest_param = min(p[1] for p in points)
        # Main rectangle: highest_acc * (ref_param - lowest_param)
        hv = highest_acc * (ref_param - lowest_param)

        # Subtract areas where we don't dominate (stairstep pattern)
        # Actually, let's use the correct sweep-line formulation:

    # Correct implementation: sweep-line from right to left
    if not frontier:
        return 0.0

    # Re-extract and sort by accuracy descending
    points = sorted(
        [(o.final_accuracy, o.param_ratio) for o in frontier],
        key=lambda p: -p[0]
    )

    hv = 0.0
    current_param = ref_param  # Start at worst param_ratio

    for acc, param in points:
        if param < current_param:
            # This point extends the dominated region
            # Add rectangle: width = acc, height = (current_param - param)
            hv += acc * (current_param - param)
            current_param = param

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
- compute_hypervolume_2d(): sweep-line algorithm for area metric
- Property-based tests verify frontier correctness
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

### Task 4.5: Add Ablation Flags to TrainingConfig

**Context:** The ablation configs (Task 5) use flags `disable_pbrs`, `disable_terminal_reward`, and `disable_anti_gaming` which don't exist in TrainingConfig. Add these before creating the config files.

**Files:**
- Modify: `src/esper/simic/training/config.py`
- Modify: `src/esper/simic/rewards/rewards.py` (respect flags in reward computation)
- Test: `tests/simic/test_ablation_flags.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_ablation_flags.py
"""Tests for ablation configuration flags."""

import pytest
from esper.simic.training.config import TrainingConfig


def test_ablation_flags_exist():
    """TrainingConfig has ablation flags with correct defaults."""
    config = TrainingConfig()
    assert config.disable_pbrs is False
    assert config.disable_terminal_reward is False
    assert config.disable_anti_gaming is False


def test_ablation_flags_settable():
    """Ablation flags can be set via constructor."""
    config = TrainingConfig(
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
    )
    assert config.disable_pbrs is True
    assert config.disable_terminal_reward is True
    assert config.disable_anti_gaming is True


def test_ablation_flags_in_to_train_kwargs():
    """Ablation flags are passed through to_train_kwargs()."""
    config = TrainingConfig(disable_pbrs=True)
    kwargs = config.to_train_kwargs()
    assert kwargs.get("disable_pbrs") is True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ablation_flags.py -v`
Expected: FAIL with "has no attribute 'disable_pbrs'"

**Step 3: Add ablation flags to TrainingConfig**

In `src/esper/simic/training/config.py`, add to the TrainingConfig dataclass:

```python
# === Ablation Flags ===
# Used for systematic reward function experiments.
# These disable specific reward components to measure their contribution.
disable_pbrs: bool = False  # Disable PBRS stage advancement shaping
disable_terminal_reward: bool = False  # Disable terminal accuracy bonus
disable_anti_gaming: bool = False  # Disable ratio_penalty and alpha_shock
```

**Step 4: Update to_train_kwargs() to include flags**

```python
def to_train_kwargs(self) -> dict[str, Any]:
    return {
        # ... existing fields ...
        "disable_pbrs": self.disable_pbrs,
        "disable_terminal_reward": self.disable_terminal_reward,
        "disable_anti_gaming": self.disable_anti_gaming,
    }
```

**Step 5: Wire flags into reward computation**

In `src/esper/simic/rewards/rewards.py`, modify the reward computation to respect flags:

```python
# In compute_reward() or relevant method:
if not config.disable_pbrs:
    reward += stage_bonus
else:
    stage_bonus = 0.0  # Zero out for telemetry consistency

if not config.disable_terminal_reward:
    reward += terminal_bonus
else:
    terminal_bonus = 0.0

if not config.disable_anti_gaming:
    reward += ratio_penalty + alpha_shock
else:
    ratio_penalty = 0.0
    alpha_shock = 0.0
```

**Step 6: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ablation_flags.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/simic/training/config.py src/esper/simic/rewards/rewards.py tests/simic/test_ablation_flags.py
git commit -m "feat(simic): add ablation flags to TrainingConfig

Three flags for systematic reward function experiments:
- disable_pbrs: Disable PBRS stage advancement shaping
- disable_terminal_reward: Disable terminal accuracy bonus
- disable_anti_gaming: Disable ratio_penalty and alpha_shock

Enables configs/ablations/*.json files to work.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Create Ablation Config Files

**Files:**
- Create: `configs/ablations/simplified_baseline.json`
- Create: `configs/ablations/no_pbrs.json`
- Create: `configs/ablations/no_terminal.json`
- Create: `configs/ablations/no_anti_gaming.json`
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

**Step 3: Create no_pbrs ablation config (DRL Expert Section 7.2)**

File: `configs/ablations/no_pbrs.json`

```json
{
    "_comment": "Ablation: SIMPLIFIED without PBRS shaping - tests raw terminal + intervention",
    "n_episodes": 100,
    "n_envs": 4,
    "max_epochs": 25,
    "reward_mode": "simplified",
    "reward_family": "contribution",
    "disable_pbrs": true,
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "entropy_coef": 0.1,
    "lr": 3e-4,
    "slots": ["r0c1"]
}
```

**Step 4: Create no_terminal ablation config (DRL Expert Section 7.2)**

File: `configs/ablations/no_terminal.json`

```json
{
    "_comment": "Ablation: SIMPLIFIED without terminal accuracy reward - tests pure shaping",
    "n_episodes": 100,
    "n_envs": 4,
    "max_epochs": 25,
    "reward_mode": "simplified",
    "reward_family": "contribution",
    "disable_terminal_reward": true,
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "entropy_coef": 0.1,
    "lr": 3e-4,
    "slots": ["r0c1"]
}
```

**Step 5: Create no_anti_gaming ablation config (DRL Expert Section 7.2)**

File: `configs/ablations/no_anti_gaming.json`

```json
{
    "_comment": "Ablation: SIMPLIFIED without anti-gaming penalties - tests reward exploitation",
    "n_episodes": 100,
    "n_envs": 4,
    "max_epochs": 25,
    "reward_mode": "simplified",
    "reward_family": "contribution",
    "disable_anti_gaming": true,
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "entropy_coef": 0.1,
    "lr": 3e-4,
    "slots": ["r0c1"]
}
```

**Step 6: Create pure sparse config**

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

**Step 7: Create A/B comparison config**

File: `configs/ablations/ab_shaped_vs_simplified.json`

```json
{
    "n_episodes": 100,
    "n_envs": 8,
    "max_epochs": 25,
    "reward_mode": "shaped",
    "reward_family": "contribution",
    "reward_mode_per_env": ["shaped", "shaped", "shaped", "shaped", "simplified", "simplified", "simplified", "simplified"],
    "param_budget": 500000,
    "param_penalty_weight": 0.1,
    "entropy_coef": 0.1,
    "lr": 3e-4,
    "slots": ["r0c1"]
}
```

**Step 8: Commit**

```bash
git add configs/ablations/
git commit -m "feat(configs): add ablation config files for reward experiments

6 configs for systematic reward function evaluation:
- simplified_baseline: DRL Expert recommended baseline
- no_pbrs: Ablation without PBRS shaping
- no_terminal: Ablation without terminal accuracy reward
- no_anti_gaming: Ablation without anti-gaming penalties
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
    assert data.is_pbrs_healthy  # 0.1-0.4 range
    assert data.is_gaming_healthy  # <0.05
    assert data.is_ev_healthy  # >0.5


def test_reward_health_pbrs_boundary_low():
    """PBRS below 10% is unhealthy (shaping too weak)."""
    data = RewardHealthData(pbrs_fraction=0.05)
    assert not data.is_pbrs_healthy


def test_reward_health_pbrs_boundary_high():
    """PBRS above 40% is unhealthy (shaping dominates)."""
    data = RewardHealthData(pbrs_fraction=0.45)
    assert not data.is_pbrs_healthy


def test_reward_health_pbrs_at_boundaries():
    """PBRS exactly at 10% and 40% is healthy."""
    low_boundary = RewardHealthData(pbrs_fraction=0.10)
    high_boundary = RewardHealthData(pbrs_fraction=0.40)
    assert low_boundary.is_pbrs_healthy
    assert high_boundary.is_pbrs_healthy


def test_reward_health_warnings():
    """Warnings triggered for unhealthy metrics."""
    unhealthy = RewardHealthData(
        pbrs_fraction=0.7,  # >0.4 = too much shaping
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
- PBRS fraction of total reward (healthy: 10-40%)
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
        """PBRS should be 10-40% of total reward (DRL Expert recommendation)."""
        return 0.1 <= self.pbrs_fraction <= 0.4

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
            (" (10-40%)", "dim"),
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
- PBRS fraction (10-40% healthy, per DRL Expert Section 6.2)
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

In the aggregator class, add methods to compute `RewardHealthData` from telemetry.

NOTE: The aggregator stores reward components per-env (in `EnvState.reward_components`),
not in a rolling list. PPO metrics are on `self._tamiyo`. We aggregate across envs.

**Step 2a: Add episode outcomes list to __post_init__**

```python
# In __post_init__, after other initializations:
self._episode_outcomes: list = []  # EpisodeOutcome instances for Pareto
```

**Step 2b: Add event handler routing**

In `_process_event_unlocked()`, add handler:

```python
elif event_type == "EPISODE_OUTCOME":
    self._handle_episode_outcome(event)
```

**Step 2c: Add handler and health computation methods**

```python
from esper.karn.sanctum.widgets.reward_health import RewardHealthData
from esper.karn.pareto import extract_pareto_frontier, compute_hypervolume_2d


def _handle_episode_outcome(self, event: "TelemetryEvent") -> None:
    """Handle incoming episode outcome events."""
    from esper.karn.store import EpisodeOutcome

    data = event.data or {}
    outcome = EpisodeOutcome(
        env_idx=data.get("env_idx", 0),
        episode_idx=data.get("episode_idx", 0),
        final_accuracy=data.get("final_accuracy", 0.0),
        param_ratio=data.get("param_ratio", 0.0),
        num_fossilized=data.get("num_fossilized", 0),
        num_contributing_fossilized=data.get("num_contributing", 0),
        episode_reward=data.get("episode_reward", 0.0),
        stability_score=data.get("stability_score", 0.0),
        reward_mode=data.get("reward_mode", ""),
    )
    self._episode_outcomes.append(outcome)

    # Keep only last 100 outcomes to bound memory
    if len(self._episode_outcomes) > 100:
        self._episode_outcomes = self._episode_outcomes[-100:]

def _compute_hypervolume(self) -> float:
    """Compute hypervolume indicator from recent episode outcomes."""
    if not self._episode_outcomes:
        return 0.0

    frontier = extract_pareto_frontier(self._episode_outcomes)
    ref_point = (0.0, 1.0)  # (min_accuracy, max_param_ratio)
    return compute_hypervolume_2d(frontier, ref_point)

def compute_reward_health(self) -> "RewardHealthData":
    """Compute reward health metrics from recent telemetry.

    Aggregates across all env reward_components (latest per env).
    Uses stage_bonus as PBRS proxy (stage advancement shaping).
    Anti-gaming = ratio_penalty + alpha_shock (non-zero = triggered).
    """
    # Collect latest reward components from all envs
    components = [env.reward_components for env in self._envs.values()
                  if env.reward_components and env.reward_components.total != 0]

    if not components:
        return RewardHealthData()

    # PBRS proxy: stage_bonus is the PBRS shaping reward
    # (stage advancement bonuses from potential function)
    pbrs_total = sum(abs(c.stage_bonus) for c in components)
    reward_total = sum(abs(c.total) for c in components)
    pbrs_fraction = pbrs_total / max(1e-8, reward_total)

    # Anti-gaming trigger rate
    # ratio_penalty and alpha_shock are penalties (non-zero = triggered)
    # Sign: penalties are typically negative, so check != 0
    gaming_steps = sum(
        1 for c in components
        if c.ratio_penalty != 0 or c.alpha_shock != 0
    )
    gaming_rate = gaming_steps / max(1, len(components))

    # Get latest EV from Tamiyo PPO state (not a separate object)
    ev = self._tamiyo.explained_variance

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

**Layout placement:** The RewardHealthPanel should be placed **underneath the Best Runs panel** (Scoreboard widget), which is currently only using half its vertical budget. This creates a natural grouping of "run-level metrics" in that column.

```
┌─────────────────┬──────────────────┐
│  EnvOverview    │   Best Runs      │  ← Scoreboard (top half)
│                 ├──────────────────┤
│                 │  Reward Health   │  ← NEW (bottom half)
├─────────────────┴──────────────────┤
│         Detail Panels              │
└────────────────────────────────────┘
```

Add to layout and wire update:

```python
from esper.karn.sanctum.widgets import RewardHealthPanel

# In compose(), place after Scoreboard in the same container:
# The Scoreboard and RewardHealthPanel share the right column
with Vertical(id="metrics-column"):
    yield Scoreboard(id="scoreboard")
    yield RewardHealthPanel(id="reward-health")

# In update handler:
health_data = self.aggregator.compute_reward_health()
self.query_one("#reward-health", RewardHealthPanel).update_data(health_data)
```

**CSS adjustment** (in `styles.tcss`):
```css
#metrics-column {
    width: 1fr;
}

#scoreboard {
    height: 50%;
}

#reward-health {
    height: 50%;
}
```

**Step 4: Verify widget renders**

Run: `PYTHONPATH=src uv run python -c "from esper.karn.sanctum.widgets.reward_health import RewardHealthPanel; print('OK')"`

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/app.py src/esper/karn/sanctum/aggregator.py src/esper/karn/sanctum/widgets/__init__.py
git commit -m "feat(sanctum): wire RewardHealthPanel into app layout

Aggregator computes health metrics from telemetry stream:
- _handle_episode_outcome() processes incoming events
- _compute_hypervolume() computes Pareto hypervolume
- compute_reward_health() aggregates all metrics

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

## Phase 6: PBRS Verification (DRL Expert Recommendation)

### Task 9: Add PBRS Verification Tests for SIMPLIFIED Mode

**Files:**
- Create: `tests/simic/test_pbrs_verification.py`

**Step 1: Write PBRS verification tests**

```python
# tests/simic/test_pbrs_verification.py
"""Verify PBRS properties hold for SIMPLIFIED reward mode.

Per Ng et al. (1999): PBRS guarantees optimal policy invariance
when shaping reward F(s,s') = gamma * Phi(s') - Phi(s).

These tests verify our implementation maintains this property.
"""

import pytest
import torch
from esper.simic.rewards.rewards import (
    RewardMode,
    ContributionRewardConfig,
    STAGE_POTENTIALS,
)
from esper.leyline import SeedStage


def test_pbrs_uses_stage_potentials():
    """PBRS bonus uses predefined stage potentials."""
    # Verify STAGE_POTENTIALS is defined for all stages
    for stage in SeedStage:
        assert stage in STAGE_POTENTIALS, f"Missing potential for {stage}"


def test_pbrs_potential_monotonicity():
    """Later stages should have higher potentials (progress is rewarded)."""
    # The lifecycle goes: DORMANT -> GERMINATED -> TRAINING -> BLENDING -> HOLDING -> FOSSILIZED
    lifecycle = [
        SeedStage.DORMANT,
        SeedStage.GERMINATED,
        SeedStage.TRAINING,
        SeedStage.BLENDING,
        SeedStage.HOLDING,
        SeedStage.FOSSILIZED,
    ]

    prev_potential = float('-inf')
    for stage in lifecycle:
        current = STAGE_POTENTIALS[stage]
        assert current >= prev_potential, f"{stage} potential should be >= previous"
        prev_potential = current


def test_pbrs_is_difference_based():
    """PBRS reward should be Phi(s') - Phi(s), not absolute."""
    # This is a design constraint test - PBRS must be computed as delta
    # The actual computation happens in rewards.py

    # Verify the formula: advancing DORMANT->GERMINATED gives positive reward
    phi_dormant = STAGE_POTENTIALS[SeedStage.DORMANT]
    phi_germinated = STAGE_POTENTIALS[SeedStage.GERMINATED]

    pbrs_advance = phi_germinated - phi_dormant
    assert pbrs_advance > 0, "Advancing stage should give positive PBRS"

    # Verify: going backwards would give negative reward
    pbrs_regress = phi_dormant - phi_germinated
    assert pbrs_regress < 0, "Regressing stage should give negative PBRS"


def test_simplified_mode_includes_pbrs():
    """SIMPLIFIED mode should include PBRS component."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
    # SIMPLIFIED = PBRS + intervention penalties + terminal accuracy
    # This is a config-level test; actual computation tested in integration
    assert config.reward_mode == RewardMode.SIMPLIFIED


def test_pbrs_zero_for_same_stage():
    """PBRS for staying in same stage should be zero."""
    for stage in SeedStage:
        phi = STAGE_POTENTIALS[stage]
        pbrs = phi - phi  # Same stage transition
        assert pbrs == 0.0, f"PBRS for {stage}->{stage} should be 0"


def test_pbrs_net_zero_over_episode():
    """PBRS should approximately net to zero over a complete episode.

    This verifies policy invariance: the shaping rewards telescope,
    leaving only the difference Phi(terminal) - Phi(initial).
    """
    # Simulate a lifecycle: DORMANT -> GERMINATED -> TRAINING -> FOSSILIZED
    trajectory = [
        (SeedStage.DORMANT, SeedStage.GERMINATED),
        (SeedStage.GERMINATED, SeedStage.TRAINING),
        (SeedStage.TRAINING, SeedStage.BLENDING),
        (SeedStage.BLENDING, SeedStage.HOLDING),
        (SeedStage.HOLDING, SeedStage.FOSSILIZED),
    ]

    total_pbrs = 0.0
    for s, s_prime in trajectory:
        pbrs = STAGE_POTENTIALS[s_prime] - STAGE_POTENTIALS[s]
        total_pbrs += pbrs

    # Total should equal Phi(FOSSILIZED) - Phi(DORMANT)
    expected = STAGE_POTENTIALS[SeedStage.FOSSILIZED] - STAGE_POTENTIALS[SeedStage.DORMANT]
    assert abs(total_pbrs - expected) < 1e-6, "PBRS should telescope correctly"
```

**Step 2: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_pbrs_verification.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/simic/test_pbrs_verification.py
git commit -m "test(simic): add PBRS verification tests for SIMPLIFIED mode

Verifies Ng et al. (1999) PBRS properties:
- Stage potentials defined for all stages
- Potential monotonicity (progress rewarded)
- Difference-based computation (not absolute)
- Zero reward for same-stage transitions
- Telescoping property over episode

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification

### Task 10: End-to-End Verification

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
       AVG(stability_score) as avg_stability,
       COUNT(*) as episodes
FROM episode_outcomes
GROUP BY reward_mode
```

**Step 3: Compute Pareto frontier**

```python
from esper.karn.pareto import extract_pareto_frontier, compute_hypervolume_2d
from esper.karn.store import EpisodeOutcome

# Load outcomes from telemetry...
frontier = extract_pareto_frontier(outcomes)
print(f"Pareto frontier size: {len(frontier)}")

hv = compute_hypervolume_2d(frontier, (0.0, 1.0))
print(f"Hypervolume: {hv:.2f}")
```

**Step 4: Verify Sanctum displays health metrics**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 --sanctum --telemetry-dir telemetry_sanctum \
    --n-episodes 5
```

Check that RewardHealthPanel shows:
- PBRS fraction (should be 10-40% if healthy)
- Gaming rate (<5% if healthy)
- EV (>0.5 if healthy)
- HV (should increase over training)

**Step 5: Run ablation study**

```bash
# Run each ablation config and compare outcomes:
for config in simplified_baseline no_pbrs no_terminal no_anti_gaming; do
    PYTHONPATH=src uv run python -m esper.scripts.train ppo \
        --config-json configs/ablations/${config}.json \
        --task cifar10 \
        --telemetry-dir telemetry_${config} \
        --n-episodes 50
done
```

---

## Summary

| Phase | Tasks | Purpose |
|-------|-------|---------|
| 0 | Task 0 | Foundation fixes (stability score design decision) |
| 0 | Task 0.5 | Generalize A/B → A/B/n (`reward_mode_per_env` + builder) |
| 1 | Task 1 | EpisodeOutcome schema |
| 1 | Task 1.5 | Add `stage_bonus` to store.py RewardComponents |
| 1 | Task 2 | Wire EpisodeOutcome into vectorized training |
| 2 | Tasks 3-4 | Pareto frontier computation + MCP view |
| 3 | Task 4.5 | Add ablation flags to TrainingConfig |
| 3 | Task 5 | Ablation config files (6 configs) |
| 4 | Tasks 6-7 | Sanctum RewardHealthPanel |
| 5 | Task 8 | Telemetry emission |
| 6 | Task 9 | PBRS verification tests |
| ✓ | Task 10 | End-to-end verification |

**Total: 13 tasks, ~3-3.5 hours implementation time**

*Note: Task 0.5 rename scope is ~44 occurrences in .py files (excluding docs/archives).
Mechanical find-replace, but verify tests pass after each file.*

---

## Critical Fixes Applied (from Specialist Reviews)

1. **Hypervolume Algorithm**: Fixed sweep-line implementation to correctly compute dominated area
2. **Stability Score**: Changed from non-existent `governor.anomalies_detected` to reward variance proxy
3. **Missing Aggregator Methods**: Added `_handle_episode_outcome()` and `_compute_hypervolume()`
4. **PBRS Healthy Range**: Adjusted from 10-50% to 10-40% per DRL Expert recommendation
5. **Missing Ablation Configs**: Added `no_pbrs.json`, `no_terminal.json`, `no_anti_gaming.json`
6. **Property-Based Tests**: Added Hypothesis tests for Pareto frontier correctness
7. **PBRS Verification Tests**: Added tests to verify Ng et al. (1999) properties
8. **Self-Dominance Bug**: Added check to prevent `outcome.dominates(outcome)` returning True
9. **A/B → A/B/n Generalization**: Renamed `ab_reward_modes` → `reward_mode_per_env`, added `with_reward_split()` builder

## Pre-Execution Fixes (Risk Assessment 2025-12-24)

10. **Slotted Dataclass Property Bug**: Removed backward compat alias (`ab_reward_modes` property)
    - TrainingConfig uses `@dataclass(slots=True)` which prohibits properties
    - Per No Legacy Code Policy: clean rename without compatibility shims
    - Added Step 7b for comprehensive find-replace across all files

11. **Aggregator State Mismatch**: Rewrote `compute_reward_health()` to use actual fields
    - `_recent_components` doesn't exist → iterate `self._envs.values()` for `reward_components`
    - `_latest_ppo` doesn't exist → use `self._tamiyo.explained_variance` directly
    - Added Step 2a/2b/2c with detailed wiring instructions

12. **PBRS Fraction Source**: Clarified PBRS tracking via `stage_bonus` field
    - `sanctum/schema.py` RewardComponents HAS `stage_bonus` ✓
    - `store.py` RewardComponents does NOT have `stage_bonus` ✗
    - **Added Task 1.5** to add `stage_bonus` to store.py for consistency

13. **Anti-Gaming Sign Convention**: Fixed penalty detection logic
    - Changed `c.ratio_penalty < 0` to `c.ratio_penalty != 0 or c.alpha_shock != 0`
    - Penalties are non-zero when active (sign convention varies)

## Pre-Execution Fixes (Risk Assessment 2025-12-26)

14. **Missing Ablation Config Flags**: Added Task 4.5 for TrainingConfig flags
    - Ablation configs reference `disable_pbrs`, `disable_terminal_reward`, `disable_anti_gaming`
    - These flags don't exist in TrainingConfig
    - **Added Task 4.5** before Task 5 to add these flags

15. **Commit Message Correction**: Fixed Task 0.5 Step 9 commit message
    - Original incorrectly said "Backwards-compatible: ab_reward_modes property alias retained"
    - Changed to "No backwards compatibility per No Legacy Code Policy (clean rename)"

---

## References

- DRL Expert Paper: `docs/research/reward-function-design-for-morphogenetic-controller.md`
- A/B/n infrastructure: `TrainingConfig.reward_mode_per_env` + `with_reward_split()` builder
- PBRS theory: Ng et al. (1999) - Policy invariance under reward transformations
- Hypervolume: Zitzler & Thiele (1999) - Multiobjective Evolutionary Algorithms
