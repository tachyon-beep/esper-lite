# Dead Code Removal + Diagnostic Telemetry Wiring

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove ~2,000 lines of dead code from the Simic module while **keeping and wiring up** `anomaly_detector.py` and `debug_telemetry.py` for automatic crash diagnostics during R&D.

**Architecture:**
- Delete truly dead modules (episodes.py, networks.py, prioritized_buffer.py, ppo_telemetry.py, memory_telemetry.py, sanity.py, evaluate.py)
- Extract live code (`MaskedCategorical`) from `networks.py` to `action_masks.py`
- **Wire `AnomalyDetector` into vectorized.py** to auto-detect training anomalies
- **Wire `debug_telemetry` to escalate** when anomalies detected
- Clean up exports and fix stale references

**Tech Stack:** Python 3.11+, pytest for verification

---

## Pre-Flight Verification

Before starting, establish baseline and create safety tag:

**Step 1: Create rollback tag**

```bash
git tag before-dead-code-cleanup
```

**Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -50
```

Expected: All tests pass (establishes baseline before changes)

---

## Task 1: Extract Live Code from networks.py

**Files:**
- Modify: `src/esper/simic/action_masks.py`
- Read: `src/esper/simic/networks.py:346-429` (MaskedCategorical + InvalidStateMachineError)

**Step 1: Read the code to extract**

Read `src/esper/simic/networks.py` lines 346-429 to copy `InvalidStateMachineError`, `_validate_action_mask`, and `MaskedCategorical`.

**Step 2: Add the extracted code to action_masks.py**

Add to the END of `src/esper/simic/action_masks.py`:

```python
# =============================================================================
# Masked Distribution (moved from networks.py during dead code cleanup)
# =============================================================================

class InvalidStateMachineError(RuntimeError):
    """Raised when action mask has no valid actions (state machine bug)."""
    pass


@torch.compiler.disable
def _validate_action_mask(mask: torch.Tensor) -> None:
    """Validate that at least one action is valid per batch element.

    Isolated from torch.compile to prevent graph breaks in the main forward path.
    The .any() call forces CPU sync, but this safety check is worth the cost.
    """
    valid_count = mask.sum(dim=-1)
    if (valid_count == 0).any():
        raise InvalidStateMachineError(
            f"No valid actions available. Mask: {mask}. "
            "This indicates a bug in the Kasmina state machine."
        )


class MaskedCategorical:
    """Categorical distribution with action masking and correct entropy calculation.

    Masks invalid actions by setting their logits to dtype minimum before softmax.
    Uses torch.finfo().min for float16/bfloat16 compatibility.
    Computes entropy only over valid actions to avoid penalizing restricted states.
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        """Initialize masked categorical distribution.

        Args:
            logits: Raw policy logits [batch, num_actions]
            mask: Binary mask, 1.0 = valid, 0.0 = invalid [batch, num_actions]

        Raises:
            InvalidStateMachineError: If any batch element has no valid actions

        Note:
            The validation check is isolated via @torch.compiler.disable to prevent
            graph breaks in the main forward path while preserving safety checks.
        """
        _validate_action_mask(mask)

        self.mask = mask
        mask_value = torch.finfo(logits.dtype).min
        self.masked_logits = logits.masked_fill(mask < 0.5, mask_value)
        self._dist = Categorical(logits=self.masked_logits)

    @property
    def probs(self) -> torch.Tensor:
        """Action probabilities (masked actions have ~0 probability)."""
        return self._dist.probs

    def sample(self) -> torch.Tensor:
        """Sample actions from the masked distribution."""
        return self._dist.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions."""
        return self._dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        """Compute normalized entropy over valid actions.

        Returns entropy normalized to [0, 1] by dividing by max entropy
        (log of number of valid actions). This makes exploration incentives
        comparable across states with different action restrictions.
        """
        probs = self._dist.probs
        log_probs = self._dist.logits - self._dist.logits.logsumexp(dim=-1, keepdim=True)
        raw_entropy = -(probs * log_probs * self.mask).sum(dim=-1)
        num_valid = self.mask.sum(dim=-1).clamp(min=1)
        max_entropy = torch.log(num_valid)
        return raw_entropy / max_entropy.clamp(min=1e-8)
```

**Step 3: Add required import**

Add to the imports at top of `action_masks.py`:

```python
from torch.distributions import Categorical
```

**Step 4: Update action_masks.py __all__**

Add to `__all__` list in `action_masks.py`:

```python
"MaskedCategorical",
"InvalidStateMachineError",
```

**Step 5: Run tests to verify extraction**

```bash
pytest tests/simic/test_action_masks.py -v
```

Expected: PASS (action_masks still works)

**Step 6: Commit extraction**

```bash
git add src/esper/simic/action_masks.py
git commit -m "refactor(simic): extract MaskedCategorical to action_masks.py

Move live code from networks.py before dead code cleanup.
MaskedCategorical and InvalidStateMachineError are actively used
by ppo.py and vectorized.py for action masking.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Imports in Consumers

**Files:**
- Modify: `src/esper/simic/ppo.py`
- Modify: `src/esper/simic/vectorized.py`

**Step 1: Find current imports**

```bash
grep -n "from esper.simic.networks import\|from esper.simic import.*MaskedCategorical" src/esper/simic/ppo.py src/esper/simic/vectorized.py
```

**Step 2: Update ppo.py imports**

Change import from:
```python
from esper.simic.networks import MaskedCategorical, InvalidStateMachineError
```

To:
```python
from esper.simic.action_masks import MaskedCategorical, InvalidStateMachineError
```

**Step 3: Update vectorized.py imports (if applicable)**

Same pattern - update any imports of MaskedCategorical or InvalidStateMachineError.

**Step 4: Run affected tests**

```bash
pytest tests/simic/test_ppo.py tests/test_simic_vectorized.py -v
```

Expected: PASS

**Step 5: Commit import updates**

```bash
git add src/esper/simic/ppo.py src/esper/simic/vectorized.py
git commit -m "refactor(simic): update MaskedCategorical imports

Point to new location in action_masks.py after extraction.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Delete Dead Source Files (NOT anomaly_detector or debug_telemetry)

**Files to DELETE:**
- `src/esper/scripts/evaluate.py`
- `src/esper/simic/episodes.py`
- `src/esper/simic/prioritized_buffer.py`
- `src/esper/simic/ppo_telemetry.py`
- `src/esper/simic/memory_telemetry.py`
- `src/esper/simic/sanity.py`

**Files to KEEP (will wire up later):**
- `src/esper/simic/anomaly_detector.py` ✓
- `src/esper/simic/debug_telemetry.py` ✓

**Step 1: Delete the dead files only**

```bash
rm src/esper/scripts/evaluate.py
rm src/esper/simic/episodes.py
rm src/esper/simic/prioritized_buffer.py
rm src/esper/simic/ppo_telemetry.py
rm src/esper/simic/memory_telemetry.py
rm src/esper/simic/sanity.py
```

**Step 2: Verify anomaly_detector and debug_telemetry still exist**

```bash
ls src/esper/simic/anomaly_detector.py src/esper/simic/debug_telemetry.py
```

Expected: Both files present

**Step 3: Commit deletions**

```bash
git add -A
git commit -m "chore(simic): delete dead source files

Remove obsolete modules identified by dead code analysis:
- evaluate.py: Incompatible with factored PPO interface
- episodes.py: Legacy 27-dim snapshot format, unused
- prioritized_buffer.py: Off-policy DQN buffer, PPO is on-policy
- ppo_telemetry.py: Duplicates inline metrics in ppo.py
- memory_telemetry.py: Use nvidia-smi or PyTorch profiler instead
- sanity.py: Assertions should be inline

KEPT for wiring:
- anomaly_detector.py: Phase-dependent anomaly detection
- debug_telemetry.py: Per-layer gradient debugging

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Delete networks.py After Extraction

**Files:**
- DELETE: `src/esper/simic/networks.py`

**Step 1: Verify MaskedCategorical was extracted**

```bash
grep -n "class MaskedCategorical" src/esper/simic/action_masks.py
```

Expected: Shows line number (confirms extraction done)

**Step 2: Delete networks.py**

```bash
rm src/esper/simic/networks.py
```

**Step 3: Commit deletion**

```bash
git add -A
git commit -m "chore(simic): delete networks.py after extracting live code

PolicyNetwork, QNetwork, VNetwork were dead (imitation learning stack).
MaskedCategorical and InvalidStateMachineError were extracted to
action_masks.py in previous commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Delete Dead Test Files (Keep anomaly_detector and debug_telemetry tests)

**Files to DELETE:**
- `tests/simic/test_ppo_telemetry.py`
- `tests/simic/test_memory_telemetry.py`
- `tests/simic/test_prioritized_buffer.py`
- `tests/test_sanity_logging.py`

**Files to KEEP:**
- `tests/simic/test_anomaly_detector.py` ✓
- `tests/simic/test_debug_telemetry.py` ✓
- `tests/simic/test_ratio_explosion.py` ✓ (uses debug_telemetry)

**Step 1: Delete dead test files only**

```bash
rm tests/simic/test_ppo_telemetry.py
rm tests/simic/test_memory_telemetry.py
rm tests/simic/test_prioritized_buffer.py
rm tests/test_sanity_logging.py
```

**Step 2: Verify kept test files exist**

```bash
ls tests/simic/test_anomaly_detector.py tests/simic/test_debug_telemetry.py tests/simic/test_ratio_explosion.py
```

Expected: All three present

**Step 3: Commit test deletions**

```bash
git add -A
git commit -m "chore(tests): delete tests for removed dead code

Deleted tests for removed modules:
- test_ppo_telemetry.py
- test_memory_telemetry.py
- test_prioritized_buffer.py
- test_sanity_logging.py

Kept tests for modules being wired up:
- test_anomaly_detector.py
- test_debug_telemetry.py
- test_ratio_explosion.py

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Delete test_simic.py (tests dead code only)

**Files:**
- DELETE: `tests/test_simic.py`

**Step 1: Delete entire file**

The entire file tests dead code (episodes.py types and PolicyNetwork from networks.py):

```bash
rm tests/test_simic.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "chore(tests): delete test_simic.py (tested dead code only)

All tests in this file exercised:
- TrainingSnapshot, Episode, etc. (from deleted episodes.py)
- PolicyNetwork (from deleted networks.py)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Clean Up conftest.py

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Find dead fixtures**

```bash
grep -n "early_training_snapshot\|converged_snapshot\|TrainingSnapshot" tests/conftest.py
```

**Step 2: Verify no other code uses these fixtures**

```bash
grep -r "early_training_snapshot\|converged_snapshot" tests/ --include="*.py"
```

Expected: Only conftest.py (the definition itself)

**Step 3: Edit conftest.py to remove dead fixtures**

Remove the `early_training_snapshot` and `converged_snapshot` fixture functions and any imports from `esper.simic.episodes`.

**Step 4: Run tests to verify**

```bash
pytest tests/conftest.py --collect-only 2>&1 | head -20
```

Expected: No import errors

**Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "chore(tests): remove fixtures depending on deleted code

Remove early_training_snapshot and converged_snapshot fixtures
that imported from deleted episodes.py module.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Clean Up strategies.py

**Files:**
- Modify: `tests/strategies.py`

**Step 1: Find the dead strategy**

```bash
grep -n "episodes\|Episode\|DecisionPoint" tests/strategies.py
```

**Step 2: Check if episodes() strategy is used**

```bash
grep -r "episodes()" tests/ --include="*.py"
```

**Step 3: Delete the episodes() strategy function**

Remove the entire `@st.composite` function `episodes()` and related helpers.

**Step 4: Remove the imports if no longer needed**

Remove the import:
```python
from esper.simic.episodes import Episode, DecisionPoint, StepOutcome, ActionTaken
```

**Step 5: Run test collection**

```bash
pytest tests/ --collect-only 2>&1 | grep -i error | head -10
```

Expected: No import errors

**Step 6: Commit**

```bash
git add tests/strategies.py
git commit -m "chore(tests): remove episodes() strategy for deleted module

The Episode, DecisionPoint, etc. types were deleted from
simic/episodes.py, so the hypothesis strategy is now dead.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Clean Up __init__.py Exports

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Remove dead imports**

Delete the following import blocks:

```python
# DELETE: episodes imports
from esper.simic.episodes import (...)

# DELETE: prioritized_buffer imports
from esper.simic.prioritized_buffer import (...)

# DELETE: networks imports
from esper.simic.networks import (...)

# DELETE: ppo_telemetry imports
from esper.simic.ppo_telemetry import (...)

# DELETE: memory_telemetry imports
from esper.simic.memory_telemetry import (...)
```

**Step 2: KEEP these imports (anomaly_detector and debug_telemetry)**

```python
# KEEP: debug_telemetry imports
from esper.simic.debug_telemetry import (
    LayerGradientStats,
    collect_per_layer_gradients,
    NumericalStabilityReport,
    check_numerical_stability,
    RatioExplosionDiagnostic,
)

# KEEP: anomaly_detector imports
from esper.simic.anomaly_detector import (
    AnomalyDetector,
    AnomalyReport,
)
```

**Step 3: Update __all__ list**

Remove dead exports, keep diagnostic exports:

```python
# DELETE from __all__:
"TrainingSnapshot", "ActionTaken", "StepOutcome", "DecisionPoint", "Episode", "DatasetManager",
"SumTree", "PrioritizedReplayBuffer",
"obs_to_base_features",
"PolicyNetwork", "print_confusion_matrix", "QNetwork", "VNetwork",
"PPOHealthTelemetry", "ValueFunctionTelemetry",
"MemoryMetrics", "collect_memory_metrics",

# KEEP in __all__:
"LayerGradientStats", "collect_per_layer_gradients",
"NumericalStabilityReport", "check_numerical_stability", "RatioExplosionDiagnostic",
"AnomalyDetector", "AnomalyReport",
```

**Step 4: Add new exports for moved code**

Add to imports:
```python
from esper.simic.action_masks import (
    MaskedCategorical,
    InvalidStateMachineError,
)
```

Add to `__all__`:
```python
"MaskedCategorical",
"InvalidStateMachineError",
```

**Step 5: Update module docstring**

Update docstring to reflect kept modules.

**Step 6: Run import test**

```bash
python -c "from esper.simic import *; print('OK')"
```

Expected: "OK"

**Step 7: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "chore(simic): clean up __init__.py exports

Remove exports for deleted modules:
- episodes.py, prioritized_buffer.py, networks.py
- ppo_telemetry.py, memory_telemetry.py

Keep exports for diagnostic modules:
- anomaly_detector.py (AnomalyDetector, AnomalyReport)
- debug_telemetry.py (LayerGradientStats, etc.)

Add exports for moved code:
- MaskedCategorical, InvalidStateMachineError

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Delete Dead Functions from features.py

**Files:**
- Modify: `src/esper/simic/features.py`

**Step 1: Find dead functions**

```bash
grep -n "def obs_to_base_features" src/esper/simic/features.py
```

**Step 2: Delete obs_to_base_features function**

Remove the entire function `obs_to_base_features()`.

**Step 3: Delete obs_to_base_features_tensor function**

Remove the entire function `obs_to_base_features_tensor()`.

**Step 4: Update __all__ in features.py**

Remove these entries:
```python
"obs_to_base_features",
"obs_to_base_features_tensor",
```

**Step 5: Run tests**

```bash
pytest tests/simic/test_features.py tests/test_simic_features.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/features.py
git commit -m "chore(simic): remove dead base feature extractors

obs_to_base_features and obs_to_base_features_tensor are unused.
PPO uses obs_to_multislot_features exclusively.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Fix Stale Docstring Reference

**Files:**
- Modify: `src/esper/simic/rewards.py`

**Step 1: Find stale reference**

```bash
grep -n "iql.py" src/esper/simic/rewards.py
```

**Step 2: Fix stale reference**

Change:
```python
- Offline RL (simic/iql.py)
```

To:
```python
- Vectorized PPO training (simic/vectorized.py)
```

**Step 3: Commit**

```bash
git add src/esper/simic/rewards.py
git commit -m "docs(simic): fix stale iql.py reference in rewards.py

The module simic/iql.py was removed. Update docstring to reference
the actual consumer: vectorized.py

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Add ratio_min to PPO Metrics

**Files:**
- Modify: `src/esper/simic/ppo.py`

**Why:** The `AnomalyDetector.check_all()` method expects `ratio_min` for ratio collapse detection, but ppo.py currently only collects `ratio_mean` and `ratio_max`.

**Step 1: Find where ratio metrics are collected**

```bash
grep -n "ratio_max\|ratio_mean" src/esper/simic/ppo.py
```

**Step 2: Add ratio_min collection**

Near `metrics["ratio_max"].append(...)`, add:

```python
metrics["ratio_min"].append(joint_ratio.min().item())
```

**Step 3: Initialize ratio_min in metrics dict**

Find where metrics dict is initialized (should have `ratio_mean: []` and `ratio_max: []`) and add:

```python
"ratio_min": [],
```

**Step 4: Run tests**

```bash
pytest tests/simic/test_ppo.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "feat(simic): add ratio_min metric to PPO update

Required for anomaly detection - ratio collapse check needs
the minimum ratio value, not just mean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Wire AnomalyDetector and Debug Telemetry into Vectorized Training

**Files:**
- Modify: `src/esper/simic/vectorized.py`

**API Reference (verified against actual code):**

```python
# AnomalyDetector API
detector = AnomalyDetector()
report = detector.check_all(
    ratio_max=...,
    ratio_min=...,
    explained_variance=...,
    has_nan=False,
    has_inf=False,
    current_episode=...,
    total_episodes=...,
)
# report.has_anomaly: bool
# report.anomaly_types: list[str]
# report.details: dict[str, str]

# NumericalStabilityReport API
stability = check_numerical_stability(model)
# stability.has_issues(): bool  (NOT .has_nan/.has_inf)
# stability.to_dict(): dict

# LayerGradientStats API
stats = collect_per_layer_gradients(model)
# stats[i].to_dict(): dict
```

**Existing event types in TelemetryEventType (use these, don't create new):**
- `RATIO_EXPLOSION_DETECTED`
- `RATIO_COLLAPSE_DETECTED`
- `VALUE_COLLAPSE_DETECTED`
- `NUMERICAL_INSTABILITY_DETECTED`
- `GRADIENT_PATHOLOGY_DETECTED`

**Step 1: Add imports**

Add near top of file:

```python
from esper.simic.anomaly_detector import AnomalyDetector
from esper.simic.debug_telemetry import (
    collect_per_layer_gradients,
    check_numerical_stability,
)
```

**Step 2: Add detector instance to train_ppo_vectorized**

After config initialization, add:

```python
# Initialize anomaly detector for automatic diagnostics
anomaly_detector = AnomalyDetector()
```

**Step 3: Add anomaly detection after PPO update**

After the PPO update block (after `ppo_metrics = agent.update(...)`), add:

```python
# === Anomaly Detection ===
# Use check_all() for comprehensive anomaly detection
anomaly_report = anomaly_detector.check_all(
    ratio_max=ppo_metrics.get("ratio_max", [1.0])[-1],
    ratio_min=ppo_metrics.get("ratio_min", [1.0])[-1],
    explained_variance=ppo_metrics.get("explained_variance", [0.0])[-1],
    current_episode=episode + 1,
    total_episodes=episodes,
)

if anomaly_report.has_anomaly:
    print(f"\n⚠️  TRAINING ANOMALY DETECTED at episode {episode + 1}:")
    for anomaly_type in anomaly_report.anomaly_types:
        print(f"   - {anomaly_type}: {anomaly_report.details.get(anomaly_type, '')}")

    # Escalate to debug telemetry - collect diagnostic data
    print("   📊 Collecting debug diagnostics...")
    gradient_stats = collect_per_layer_gradients(agent.network)
    stability_report = check_numerical_stability(agent.network)

    # Log gradient health summary
    vanishing = sum(1 for gs in gradient_stats if gs.zero_fraction > 0.5)
    exploding = sum(1 for gs in gradient_stats if gs.large_fraction > 0.1)
    if vanishing > 0:
        print(f"   ⚠️  {vanishing} layers with vanishing gradients (>50% zeros)")
    if exploding > 0:
        print(f"   ⚠️  {exploding} layers with exploding gradients (>10% large values)")
    if stability_report.has_issues():
        print(f"   🔥 NUMERICAL INSTABILITY detected in weights/gradients")

    # Emit specific telemetry events (use existing event types)
    if hub:
        # Map anomaly types to specific event types
        event_type_map = {
            "ratio_explosion": TelemetryEventType.RATIO_EXPLOSION_DETECTED,
            "ratio_collapse": TelemetryEventType.RATIO_COLLAPSE_DETECTED,
            "value_collapse": TelemetryEventType.VALUE_COLLAPSE_DETECTED,
            "numerical_instability": TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
        }

        for anomaly_type in anomaly_report.anomaly_types:
            event_type = event_type_map.get(
                anomaly_type,
                TelemetryEventType.GRADIENT_ANOMALY  # fallback
            )
            hub.emit(TelemetryEvent(
                event_type=event_type,
                data={
                    "episode": episode + 1,
                    "detail": anomaly_report.details.get(anomaly_type, ""),
                    "gradient_stats": [gs.to_dict() for gs in gradient_stats[:5]],
                    "stability": stability_report.to_dict(),
                }
            ))
```

**Step 4: Run tests**

```bash
pytest tests/test_simic_vectorized.py tests/simic/test_anomaly_detector.py tests/simic/test_debug_telemetry.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(simic): wire anomaly detection into vectorized training

Add automatic training anomaly detection with debug escalation:

Detection (via AnomalyDetector.check_all):
- Ratio explosion (max ratio > 5.0)
- Ratio collapse (min ratio < 0.1)
- Value function collapse (phase-dependent EV thresholds)

Escalation (when anomaly detected):
- Per-layer gradient statistics collection
- Numerical stability analysis (NaN/Inf detection)
- Console warnings with actionable diagnostics
- Specific telemetry events for tracking

Uses existing TelemetryEventType enum values:
- RATIO_EXPLOSION_DETECTED
- RATIO_COLLAPSE_DETECTED
- VALUE_COLLAPSE_DETECTED
- NUMERICAL_INSTABILITY_DETECTED

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 14: Delete Orphaned Fixture Files

**Files:**
- Check: `tests/fixtures/snapshots/`

**Step 1: Check what fixtures exist**

```bash
ls -la tests/fixtures/snapshots/ 2>/dev/null || echo "Directory does not exist"
```

**Step 2: Check if fixtures are used**

```bash
grep -r "early_training_epoch1\|converged_epoch200" tests/ --include="*.py"
```

**Step 3: Delete orphaned fixture files if they exist and are not used**

```bash
rm -f tests/fixtures/snapshots/early_training_epoch1.json
rm -f tests/fixtures/snapshots/converged_epoch200.json
```

**Step 4: Remove empty directory if applicable**

```bash
rmdir tests/fixtures/snapshots 2>/dev/null || true
```

**Step 5: Commit (if files were deleted)**

```bash
git add -A
git diff --cached --quiet || git commit -m "chore(tests): remove orphaned fixture files

JSON fixtures for deleted TrainingSnapshot tests.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 15: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -100
```

Expected: All tests PASS

**Step 2: Verify no import errors**

```bash
python -c "import esper.simic; import esper.scripts.train; print('All imports OK')"
```

Expected: "All imports OK"

**Step 3: Verify anomaly detection imports work**

```bash
python -c "from esper.simic import AnomalyDetector, collect_per_layer_gradients; print('Diagnostics OK')"
```

Expected: "Diagnostics OK"

**Step 4: Run training smoke test**

```bash
python -m esper.scripts.train heuristic --task cifar10 --epochs 1 --dry-run
```

Expected: No errors

**Step 5: Count lines changed**

```bash
git diff --stat before-dead-code-cleanup..HEAD
```

Expected: ~2,000 lines deleted, ~80 lines added for wiring

---

## Summary

| Task | Files Affected | Change |
|------|---------------|--------|
| 1 | action_masks.py | +80 (extraction) |
| 2 | ppo.py, vectorized.py | ~2 (import change) |
| 3 | 6 source files | DELETE ~1,000 lines |
| 4 | networks.py | DELETE ~480 lines |
| 5 | 4 test files | DELETE ~400 lines |
| 6 | test_simic.py | DELETE ~370 lines |
| 7 | conftest.py | -16 lines |
| 8 | strategies.py | -50 lines |
| 9 | __init__.py | Update exports |
| 10 | features.py | -140 lines |
| 11 | rewards.py | Fix docstring |
| 12 | ppo.py | +3 (ratio_min metric) |
| 13 | vectorized.py | +50 (anomaly detection + debug) |
| 14 | fixtures/ | DELETE ~100 lines JSON |

**Net Result:**
- ~2,000 lines of dead code removed
- ~80 lines of diagnostic wiring added
- Automatic anomaly detection during R&D training
- Per-layer gradient debugging on anomaly escalation

---

## What You Get

After this cleanup, training will automatically:

1. **Detect ratio explosions** when policy updates become unstable
2. **Detect ratio collapse** when policy stops updating
3. **Detect value collapse** with phase-aware thresholds
4. **Escalate to debug mode** collecting per-layer gradient stats
5. **Log warnings** with actionable diagnostics
6. **Emit specific telemetry events** for tracking patterns over time

Example output when something goes wrong:

```
⚠️  TRAINING ANOMALY DETECTED at episode 42:
   - value_collapse: explained_variance=-0.15 < -0.2 (at 42% training)
   📊 Collecting debug diagnostics...
   ⚠️  3 layers with vanishing gradients (>50% zeros)
```

---

## Rollback Plan

Safe rollback using the tag created in pre-flight:

```bash
git reset --hard before-dead-code-cleanup
```

Or revert specific commits while keeping later work:

```bash
git revert <commit-hash>
```

After successful completion, clean up the tag:

```bash
git tag -d before-dead-code-cleanup
```
