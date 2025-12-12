# Multislot Features and Max Seeds Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire `obs_to_multislot_features()` (per-slot visibility) and `max_seeds` CLI parameters into the training pipeline.

**Architecture:** Two changes work together - (1) `signals_to_features()` builds per-slot observation dict and calls `obs_to_multislot_features()` instead of `obs_to_base_features()`, (2) `max_seeds` flows from CLI args through training functions to feature extraction and action masking.

**Tech Stack:** PyTorch, esper-lite (kasmina/simic/leyline)

---

## Background

### Current State
- CLI args `--max-seeds` and `--max-seeds-per-slot` exist but are **never passed** to training functions
- `signals_to_features()` in `ppo.py` calls `obs_to_base_features()` (single-slot focused)
- Action masking passes `max_seeds=0` (unlimited), so seed limit checks never trigger

### Target State
- `max_seeds` flows from CLI → training functions → feature extraction + action masking
- Agent sees per-slot state via `obs_to_multislot_features()` (early/mid/late visibility)
- Seed limit masking actually enforces limits when configured

### Dimension Note
Both `obs_to_base_features()` and `obs_to_multislot_features()` return **35 dims**. The difference is **structure**, not dimension:
- `obs_to_base_features()`: 35 dims with single-slot seed state + blueprint one-hot
- `obs_to_multislot_features()`: 35 dims with per-slot states (3 slots × 4 features) + seed_utilization

Network `state_dim` unchanged: 35 base + 10 telemetry = 45 dims.

---

## Task 1: Add max_seeds to train_ppo() signature

**Files:**
- Modify: `src/esper/simic/training.py:445-465`
- Modify: `src/esper/scripts/train.py:180-199`

**Step 1: Add parameters to train_ppo() signature**

In `training.py`, add after `slots` parameter (line 464):

```python
def train_ppo(
    n_episodes: int = 100,
    max_epochs: int = 25,
    update_every: int = 5,
    device: str = "cuda:0",
    task: str = "cifar10",
    use_telemetry: bool = True,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.05,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = 0.01,
    adaptive_entropy_floor: bool = False,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    save_path: str | None = None,
    seed: int | None = None,
    telemetry_config: "TelemetryConfig | None" = None,
    slots: list[str] | None = None,
    max_seeds: int | None = None,           # NEW
    max_seeds_per_slot: int | None = None,  # NEW
):
```

**Step 2: Pass from train.py**

In `train.py`, update the `train_ppo()` call (lines 180-199):

```python
train_ppo(
    n_episodes=args.episodes,
    max_epochs=args.max_epochs,
    update_every=args.update_every,
    device=args.device,
    task=args.task,
    use_telemetry=use_telemetry,
    lr=args.lr,
    clip_ratio=args.clip_ratio,
    entropy_coef=args.entropy_coef,
    entropy_coef_start=args.entropy_coef_start,
    entropy_coef_end=args.entropy_coef_end,
    entropy_coef_min=args.entropy_coef_min,
    entropy_anneal_episodes=args.entropy_anneal_episodes,
    gamma=args.gamma,
    save_path=args.save,
    seed=args.seed,
    telemetry_config=telemetry_config,
    slots=args.slots,
    max_seeds=args.max_seeds,                  # NEW
    max_seeds_per_slot=args.max_seeds_per_slot,  # NEW
)
```

**Step 3: Run existing tests**

```bash
PYTHONPATH=src pytest tests/simic/test_training.py -v
```

Expected: PASS (new params have defaults)

**Step 4: Commit**

```bash
git add src/esper/simic/training.py src/esper/scripts/train.py
git commit -m "feat(simic): add max_seeds params to train_ppo()"
```

---

## Task 2: Add max_seeds to train_ppo_vectorized() signature

**Files:**
- Modify: `src/esper/simic/vectorized.py:163-193`
- Modify: `src/esper/scripts/train.py:154-177`

**Step 1: Add parameters to train_ppo_vectorized() signature**

In `vectorized.py`, add after `slots` parameter (line 192):

```python
def train_ppo_vectorized(
    n_episodes: int = 100,
    n_envs: int = 4,
    max_epochs: int = 25,
    device: str = "cuda:0",
    devices: list[str] | None = None,
    task: str = "cifar10",
    use_telemetry: bool = True,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.05,
    entropy_coef_start: float | None = None,
    entropy_coef_end: float | None = None,
    entropy_coef_min: float = 0.01,
    adaptive_entropy_floor: bool = False,
    entropy_anneal_episodes: int = 0,
    gamma: float = 0.99,
    ppo_updates_per_batch: int = 1,
    save_path: str = None,
    resume_path: str = None,
    seed: int = 42,
    num_workers: int | None = None,
    gpu_preload: bool = False,
    recurrent: bool = False,
    lstm_hidden_dim: int = 128,
    chunk_length: int = 25,
    telemetry_config: "TelemetryConfig | None" = None,
    plateau_threshold: float = 0.5,
    improvement_threshold: float = 2.0,
    slots: list[str] | None = None,
    max_seeds: int | None = None,           # NEW
    max_seeds_per_slot: int | None = None,  # NEW
) -> tuple[PPOAgent, list[dict]]:
```

**Step 2: Pass from train.py**

In `train.py`, update the `train_ppo_vectorized()` call (lines 154-177):

```python
train_ppo_vectorized(
    n_episodes=args.episodes,
    n_envs=args.n_envs,
    max_epochs=args.max_epochs,
    device=args.device,
    devices=args.devices,
    task=args.task,
    use_telemetry=use_telemetry,
    lr=args.lr,
    clip_ratio=args.clip_ratio,
    entropy_coef=args.entropy_coef,
    entropy_coef_start=args.entropy_coef_start,
    entropy_coef_end=args.entropy_coef_end,
    entropy_coef_min=args.entropy_coef_min,
    entropy_anneal_episodes=args.entropy_anneal_episodes,
    gamma=args.gamma,
    save_path=args.save,
    resume_path=args.resume,
    seed=args.seed,
    num_workers=args.num_workers,
    gpu_preload=args.gpu_preload,
    telemetry_config=telemetry_config,
    slots=args.slots,
    max_seeds=args.max_seeds,                  # NEW
    max_seeds_per_slot=args.max_seeds_per_slot,  # NEW
)
```

**Step 3: Run existing tests**

```bash
PYTHONPATH=src pytest tests/simic/test_vectorized.py -v -k "not slow"
```

Expected: PASS (new params have defaults)

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py src/esper/scripts/train.py
git commit -m "feat(simic): add max_seeds params to train_ppo_vectorized()"
```

---

## Task 3: Update signals_to_features() signature and add per-slot obs

**Files:**
- Modify: `src/esper/simic/ppo.py:33-123`
- Test: `tests/simic/test_ppo.py` (new test)

**Step 1: Write the failing test**

Create new test in `tests/simic/test_ppo.py`:

```python
def test_signals_to_features_with_multislot_params():
    """Test signals_to_features accepts total_seeds and max_seeds params."""
    from esper.simic.ppo import signals_to_features
    from esper.simic.features import MULTISLOT_FEATURE_SIZE
    from esper.leyline import SeedTelemetry

    # Create minimal signals mock
    class MockMetrics:
        epoch = 10
        global_step = 100
        train_loss = 0.5
        val_loss = 0.6
        loss_delta = -0.1
        train_accuracy = 85.0
        val_accuracy = 82.0
        accuracy_delta = 0.5
        plateau_epochs = 2
        best_val_accuracy = 83.0
        best_val_loss = 0.55
        grad_norm_host = 1.0

    class MockSignals:
        metrics = MockMetrics()
        loss_history = [0.8, 0.7, 0.6, 0.5, 0.5]
        accuracy_history = [70.0, 75.0, 80.0, 82.0, 85.0]
        active_seeds = []
        available_slots = 3
        seed_stage = 0
        seed_epochs_in_stage = 0
        seed_alpha = 0.0
        seed_improvement = 0.0
        seed_counterfactual = 0.0

    features = signals_to_features(
        signals=MockSignals(),
        model=None,
        use_telemetry=False,
        slots=["mid"],
        total_seeds=1,  # NEW param
        max_seeds=3,    # NEW param
    )

    assert len(features) == MULTISLOT_FEATURE_SIZE
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src pytest tests/simic/test_ppo.py::test_signals_to_features_with_multislot_params -v
```

Expected: FAIL - `unexpected keyword argument 'total_seeds'`

**Step 3: Update signals_to_features() signature**

In `ppo.py`, modify the function signature (line 33):

```python
def signals_to_features(
    signals,
    model,
    use_telemetry: bool = True,
    max_epochs: int = 200,
    slots: list[str] | None = None,
    total_seeds: int = 0,  # NEW
    max_seeds: int = 0,    # NEW
) -> list[float]:
    """Convert training signals to feature vector.

    Args:
        signals: TrainingSignals from tamiyo
        model: MorphogeneticModel
        use_telemetry: Whether to include telemetry features
        max_epochs: Maximum epochs for learning phase normalization
        slots: List of slot names to extract features from
        total_seeds: Current total seeds across all slots (for utilization calc)
        max_seeds: Maximum allowed seeds (for utilization calc)

    Returns:
        Feature vector (35 dims base, +10 if telemetry = 45 dims total)
    """
```

**Step 4: Replace feature extraction with obs_to_multislot_features()**

Replace the import and feature extraction logic:

```python
    if not slots:
        raise ValueError("signals_to_features: slots parameter is required and cannot be empty")

    target_slot = slots[0]
    from esper.simic.features import obs_to_multislot_features

    # Build observation dict
    loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
    while len(loss_hist) < 5:
        loss_hist.insert(0, 0.0)

    acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
    while len(acc_hist) < 5:
        acc_hist.insert(0, 0.0)

    obs = {
        'epoch': signals.metrics.epoch,
        'global_step': signals.metrics.global_step,
        'train_loss': signals.metrics.train_loss,
        'val_loss': signals.metrics.val_loss,
        'loss_delta': signals.metrics.loss_delta,
        'train_accuracy': signals.metrics.train_accuracy,
        'val_accuracy': signals.metrics.val_accuracy,
        'accuracy_delta': signals.metrics.accuracy_delta,
        'plateau_epochs': signals.metrics.plateau_epochs,
        'best_val_accuracy': signals.metrics.best_val_accuracy,
        'best_val_loss': signals.metrics.best_val_loss,
        'loss_history_5': loss_hist,
        'accuracy_history_5': acc_hist,
        'total_params': model.total_params if model else 0,
    }

    # Build per-slot state dict
    slot_states = {}
    for slot_id in ['early', 'mid', 'late']:
        if model and slot_id in model.seed_slots:
            slot = model.seed_slots[slot_id]
            if slot.is_active and slot.state:
                slot_states[slot_id] = {
                    'is_active': 1.0,
                    'stage': slot.state.stage.value,
                    'alpha': slot.state.alpha,
                    'improvement': slot.state.metrics.improvement_since_stage_start,
                }
            else:
                slot_states[slot_id] = {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0}
        else:
            slot_states[slot_id] = {'is_active': 0.0, 'stage': 0, 'alpha': 0.0, 'improvement': 0.0}

    obs['slots'] = slot_states

    features = obs_to_multislot_features(obs, total_seeds=total_seeds, max_seeds=max_seeds)

    if use_telemetry:
        from esper.leyline import SeedTelemetry
        if model and model.has_active_seed:
            seed_state = model.seed_slots[target_slot].state
            features.extend(seed_state.telemetry.to_features())
        else:
            features.extend([0.0] * SeedTelemetry.feature_dim())

    return features
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=src pytest tests/simic/test_ppo.py::test_signals_to_features_with_multislot_params -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo.py
git commit -m "feat(simic): switch signals_to_features to obs_to_multislot_features"
```

---

## Task 4: Thread max_seeds to feature extraction in training.py

**Files:**
- Modify: `src/esper/simic/training.py:358` (signals_to_features call)
- Modify: `src/esper/simic/training.py:366-367` (compute_flat_action_mask call)

**Step 1: Compute effective max_seeds**

After the `slots` validation, add computation (around line 510):

```python
    # Compute effective seed limit
    # max_seeds=None means unlimited (use 0 to indicate no limit)
    effective_max_seeds = max_seeds if max_seeds is not None else 0
```

**Step 2: Update signals_to_features call**

Find the `signals_to_features` call around line 358 and update:

```python
        features = signals_to_features(
            signals,
            model,
            use_telemetry=use_telemetry,
            slots=slots,
            total_seeds=model.count_active_seeds() if model else 0,  # NEW
            max_seeds=effective_max_seeds,  # NEW
        )
```

**Step 3: Update compute_flat_action_mask call**

Find the `compute_flat_action_mask` call (around line 366-367) and update:

```python
        action_mask_list = compute_flat_action_mask(
            slot_states=slot_states,
            total_seeds=model.count_active_seeds() if model else 0,  # UPDATED
            max_seeds=effective_max_seeds,  # UPDATED from 0
            num_germinate_actions=num_germinate_actions,
        )
```

**Step 4: Run tests**

```bash
PYTHONPATH=src pytest tests/simic/test_training.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training.py
git commit -m "feat(simic): thread max_seeds through train_ppo()"
```

---

## Task 5: Thread max_seeds to feature extraction in vectorized.py

**Files:**
- Modify: `src/esper/simic/vectorized.py:975` (signals_to_features call)
- Modify: `src/esper/simic/vectorized.py:980-985` (compute_flat_action_mask call)

**Step 1: Compute effective max_seeds**

After the `slots` validation (around line 232), add:

```python
    # Compute effective seed limit
    effective_max_seeds = max_seeds if max_seeds is not None else 0
```

**Step 2: Update signals_to_features call**

Find the `signals_to_features` call around line 975 and update:

```python
                features = signals_to_features(
                    signals,
                    model,
                    use_telemetry=use_telemetry,
                    slots=slots,
                    total_seeds=model.count_active_seeds() if model else 0,  # NEW
                    max_seeds=effective_max_seeds,  # NEW
                )
```

**Step 3: Update compute_flat_action_mask call**

Find the `compute_flat_action_mask` call (around line 980-985) and update:

```python
                mask = compute_flat_action_mask(
                    slot_states=slot_states,
                    total_seeds=model.count_active_seeds() if model else 0,  # UPDATED
                    max_seeds=effective_max_seeds,  # UPDATED from 0
                    num_germinate_actions=num_germinate_actions,
                )
```

**Step 4: Run tests**

```bash
PYTHONPATH=src pytest tests/simic/test_vectorized.py -v -k "not slow"
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(simic): thread max_seeds through train_ppo_vectorized()"
```

---

## Task 6: Integration test for max_seeds enforcement

**Files:**
- Create: `tests/integration/test_max_seeds_wiring.py`

**Step 1: Write integration test**

```python
"""Integration tests for max_seeds wiring through training pipeline."""
import pytest
import torch

from esper.simic.ppo import signals_to_features
from esper.simic.action_masks import compute_flat_action_mask, build_slot_states


class TestMaxSeedsWiring:
    """Test max_seeds flows correctly through the pipeline."""

    def test_seed_utilization_in_features(self):
        """Verify seed_utilization appears in feature vector."""
        class MockMetrics:
            epoch = 10
            global_step = 100
            train_loss = 0.5
            val_loss = 0.6
            loss_delta = -0.1
            train_accuracy = 85.0
            val_accuracy = 82.0
            accuracy_delta = 0.5
            plateau_epochs = 2
            best_val_accuracy = 83.0
            best_val_loss = 0.55
            grad_norm_host = 1.0

        class MockSignals:
            metrics = MockMetrics()
            loss_history = [0.8, 0.7, 0.6, 0.5, 0.5]
            accuracy_history = [70.0, 75.0, 80.0, 82.0, 85.0]
            active_seeds = []
            available_slots = 3
            seed_stage = 0
            seed_epochs_in_stage = 0
            seed_alpha = 0.0
            seed_improvement = 0.0
            seed_counterfactual = 0.0

        # With 1 seed out of 3 max -> utilization = 0.333...
        features = signals_to_features(
            signals=MockSignals(),
            model=None,
            use_telemetry=False,
            slots=["mid"],
            total_seeds=1,
            max_seeds=3,
        )

        # Feature index 22 is seed_utilization per obs_to_multislot_features layout
        assert abs(features[22] - (1/3)) < 0.01, f"Expected ~0.333, got {features[22]}"

    def test_germinate_masked_at_limit(self):
        """Verify GERMINATE is masked when at seed limit."""
        from esper.simic.action_masks import MaskSeedInfo
        from esper.leyline import SeedStage

        # No active seed (empty slot)
        slot_states = {"mid": None}

        # At limit: 3 seeds out of 3 max
        mask = compute_flat_action_mask(
            slot_states=slot_states,
            total_seeds=3,
            max_seeds=3,
            num_germinate_actions=5,
        )

        # GERMINATE actions (indices 1-5) should all be masked (0.0)
        for i in range(1, 6):
            assert mask[i] == 0.0, f"GERMINATE action {i} should be masked at seed limit"

    def test_germinate_allowed_under_limit(self):
        """Verify GERMINATE is allowed when under seed limit."""
        # No active seed (empty slot)
        slot_states = {"mid": None}

        # Under limit: 2 seeds out of 3 max
        mask = compute_flat_action_mask(
            slot_states=slot_states,
            total_seeds=2,
            max_seeds=3,
            num_germinate_actions=5,
        )

        # GERMINATE actions (indices 1-5) should be allowed (1.0)
        for i in range(1, 6):
            assert mask[i] == 1.0, f"GERMINATE action {i} should be allowed under limit"

    def test_unlimited_seeds_when_max_zero(self):
        """Verify max_seeds=0 means unlimited (legacy behavior)."""
        # No active seed (empty slot)
        slot_states = {"mid": None}

        # max_seeds=0 means unlimited
        mask = compute_flat_action_mask(
            slot_states=slot_states,
            total_seeds=100,  # Many seeds
            max_seeds=0,      # Unlimited
            num_germinate_actions=5,
        )

        # GERMINATE should still be allowed (slot is empty)
        for i in range(1, 6):
            assert mask[i] == 1.0, f"GERMINATE should be allowed with max_seeds=0"
```

**Step 2: Run integration test**

```bash
PYTHONPATH=src pytest tests/integration/test_max_seeds_wiring.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_max_seeds_wiring.py
git commit -m "test(integration): add max_seeds wiring integration tests"
```

---

## Task 7: Clean up outdated comments

**Files:**
- Modify: `src/esper/simic/ppo.py:44` (docstring)
- Modify: `src/esper/simic/vectorized.py:983` (misleading comment)
- Modify: `src/esper/simic/training.py:366` (misleading comment)

**Step 1: Fix ppo.py docstring**

The docstring says "30 dims base" but should say "35 dims base":

```python
    Returns:
        Feature vector (35 dims base, +10 if telemetry = 45 dims total)
```

**Step 2: Fix vectorized.py comment**

Change comment at line 983 from:
```python
max_seeds=0,  # Seed limits handled in reward function
```
To:
```python
max_seeds=effective_max_seeds,
```

(Comment is now removed since we're actually using the value)

**Step 3: Fix training.py comment**

Change comment at line 366 from:
```python
max_seeds=0,  # No limit in non-vectorized path
```
To:
```python
max_seeds=effective_max_seeds,
```

**Step 4: Run full test suite**

```bash
PYTHONPATH=src pytest tests/simic/ tests/integration/ -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py src/esper/simic/vectorized.py src/esper/simic/training.py
git commit -m "docs(simic): fix outdated comments in feature extraction"
```

---

## Task 8: Delete unused legacy constants (DRL Expert)

**Files:**
- Modify: `src/esper/leyline/__init__.py`

**Note:** DRL Expert identified these timing constants as unused after migration to physical-constraints-only masking:
- `MIN_GERMINATE_EPOCH = 5`
- `MIN_PLATEAU_TO_GERMINATE = 3`

**Step 1: Verify constants are unused**

```bash
grep -rn "MIN_GERMINATE_EPOCH\|MIN_PLATEAU_TO_GERMINATE" src/esper --include="*.py"
```

Expected: Only definition in leyline/__init__.py (lines 28-29) and `__all__` export (lines 100-101)

**Step 2: Delete the constants**

Remove these lines from `leyline/__init__.py`:

```python
# Line 28-29: Delete definitions
MIN_GERMINATE_EPOCH = 5        # Let host get easy wins first
MIN_PLATEAU_TO_GERMINATE = 3   # Consecutive epochs with <0.5% improvement

# Line 100-101: Delete from __all__
    "MIN_GERMINATE_EPOCH",
    "MIN_PLATEAU_TO_GERMINATE",
```

**Step 3: Run tests**

```bash
PYTHONPATH=src pytest tests/ -v --tb=short
```

Expected: PASS (if any test imports these, it will fail)

**Step 4: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "refactor(leyline): delete unused timing constants per CLAUDE.md"
```

---

## Task 9: Delete SimicAction deprecated alias (Code Reviewer)

**Files:**
- Modify: `src/esper/leyline/actions.py:101,106`
- Modify: `src/esper/leyline/__init__.py:38,107`
- Modify: `src/esper/simic/__init__.py:18,119`
- Modify: `src/esper/__init__.py:17,21`
- Modify: `tests/test_simic.py:333` (docstring reference)

**Note:** Code Reviewer identified `SimicAction` as a deprecated alias that violates CLAUDE.md. Per CLAUDE.md: "When something is removed or changed, DELETE THE OLD CODE COMPLETELY."

**Step 1: Verify all usages**

```bash
grep -rn "SimicAction" src/ tests/
```

Expected locations:
- `src/esper/leyline/actions.py:101` - Definition: `SimicAction = Action`
- `src/esper/leyline/actions.py:106` - `__all__` export
- `src/esper/leyline/__init__.py:38` - Re-import
- `src/esper/leyline/__init__.py:107` - `__all__` export
- `src/esper/simic/__init__.py:18` - Re-import
- `src/esper/simic/__init__.py:119` - `__all__` export
- `src/esper/__init__.py:17` - Re-import
- `src/esper/__init__.py:21` - `__all__` export
- `tests/test_simic.py:333` - Docstring reference

**Step 2: Delete from leyline/actions.py**

Remove:
```python
# Line 101
SimicAction = Action

# Line 106 in __all__
    "SimicAction",
```

**Step 3: Delete from leyline/__init__.py**

Remove:
```python
# Line 38: Remove SimicAction from import
    SimicAction,  # SimicAction is alias

# Line 107 in __all__
    "SimicAction",  # deprecated alias
```

**Step 4: Delete from simic/__init__.py**

Change line 18 from:
```python
from esper.leyline import Action, SimicAction  # SimicAction is deprecated alias
```
To:
```python
from esper.leyline import Action
```

Remove from `__all__` (line 119):
```python
    "SimicAction",  # deprecated alias
```

**Step 5: Delete from esper/__init__.py**

Change line 17 from:
```python
from esper.leyline import SimicAction, SeedStage, TrainingSignals
```
To:
```python
from esper.leyline import SeedStage, TrainingSignals
```

Remove from `__all__` (line 21):
```python
    "SimicAction",
```

**Step 6: Fix test docstring**

In `tests/test_simic.py:333`, change:
```python
"""Test that predict returns a SimicAction."""
```
To:
```python
"""Test that predict returns an Action."""
```

**Step 7: Run tests**

```bash
PYTHONPATH=src pytest tests/ -v --tb=short
```

Expected: PASS

**Step 8: Commit**

```bash
git add src/esper/leyline/actions.py src/esper/leyline/__init__.py src/esper/simic/__init__.py src/esper/__init__.py tests/test_simic.py
git commit -m "refactor: delete SimicAction deprecated alias per CLAUDE.md"
```

---

## Task 10: Add unit test for build_slot_states() (Code Reviewer)

**Files:**
- Modify: `tests/simic/test_action_masks.py`

**Note:** Code Reviewer noted `build_slot_states()` lacks direct unit test (only exercised through integration).

**Step 1: Write the test**

Add to `tests/simic/test_action_masks.py`:

```python
class TestBuildSlotStates:
    """Tests for build_slot_states() helper function."""

    def test_empty_model_returns_none_states(self):
        """Empty slots return None for each slot."""
        from esper.simic.action_masks import build_slot_states

        # Mock model with empty slots
        class MockSlot:
            is_active = False
            state = None

        class MockModel:
            seed_slots = {"mid": MockSlot()}

        result = build_slot_states(MockModel(), ["mid"])
        assert result == {"mid": None}

    def test_active_seed_returns_mask_seed_info(self):
        """Active seed returns MaskSeedInfo with correct stage and age."""
        from esper.simic.action_masks import build_slot_states, MaskSeedInfo
        from esper.leyline import SeedStage

        class MockMetrics:
            epochs_in_current_stage = 5

        class MockState:
            stage = SeedStage.TRAINING
            metrics = MockMetrics()

        class MockSlot:
            is_active = True
            state = MockState()

        class MockModel:
            seed_slots = {"mid": MockSlot()}

        result = build_slot_states(MockModel(), ["mid"])

        assert "mid" in result
        assert isinstance(result["mid"], MaskSeedInfo)
        assert result["mid"].stage == SeedStage.TRAINING.value
        assert result["mid"].seed_age_epochs == 5

    def test_multiple_slots(self):
        """Multiple slots are all processed."""
        from esper.simic.action_masks import build_slot_states, MaskSeedInfo
        from esper.leyline import SeedStage

        class MockMetrics:
            epochs_in_current_stage = 3

        class MockState:
            stage = SeedStage.BLENDING
            metrics = MockMetrics()

        class MockActiveSlot:
            is_active = True
            state = MockState()

        class MockEmptySlot:
            is_active = False
            state = None

        class MockModel:
            seed_slots = {
                "early": MockEmptySlot(),
                "mid": MockActiveSlot(),
                "late": MockEmptySlot(),
            }

        result = build_slot_states(MockModel(), ["early", "mid", "late"])

        assert result["early"] is None
        assert isinstance(result["mid"], MaskSeedInfo)
        assert result["late"] is None
```

**Step 2: Run test**

```bash
PYTHONPATH=src pytest tests/simic/test_action_masks.py::TestBuildSlotStates -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/simic/test_action_masks.py
git commit -m "test(simic): add unit tests for build_slot_states()"
```

---

## Verification Checklist

After completing all tasks, run:

```bash
# Full test suite
PYTHONPATH=src pytest tests/ -v

# Verify max_seeds flows through
grep -n "effective_max_seeds" src/esper/simic/training.py src/esper/simic/vectorized.py

# Verify signals_to_features uses new params
grep -n "total_seeds=\|max_seeds=" src/esper/simic/training.py src/esper/simic/vectorized.py

# Verify no unused timing constants (DRL Expert cleanup)
grep -rn "MIN_GERMINATE_EPOCH\|MIN_PLATEAU_TO_GERMINATE" src/esper

# Verify SimicAction alias fully removed (Code Reviewer cleanup)
grep -rn "SimicAction" src/esper tests/

# Manual smoke test
python -m esper.scripts.train ppo --task cifar10 --slots mid --max-seeds 1 --episodes 1 --max-epochs 5
```

---

## Summary

| Task | Description | Files | Expert |
|------|-------------|-------|--------|
| 1 | Add max_seeds to train_ppo() | training.py, train.py | - |
| 2 | Add max_seeds to train_ppo_vectorized() | vectorized.py, train.py | - |
| 3 | Update signals_to_features() | ppo.py | - |
| 4 | Thread max_seeds in training.py | training.py | - |
| 5 | Thread max_seeds in vectorized.py | vectorized.py | - |
| 6 | Integration tests | test_max_seeds_wiring.py | - |
| 7 | Clean up comments | ppo.py, training.py, vectorized.py | - |
| 8 | Delete unused timing constants | leyline/__init__.py | DRL Expert |
| 9 | Delete SimicAction deprecated alias | actions.py, __init__.py files | Code Reviewer |
| 10 | Add build_slot_states() unit tests | test_action_masks.py | Code Reviewer |
