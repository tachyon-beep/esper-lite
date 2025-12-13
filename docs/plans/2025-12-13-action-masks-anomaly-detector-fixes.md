# Action Masks & Anomaly Detector Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all P1 and P2 issues identified by code review, PyTorch specialist, and DRL specialist in action_masks.py, anomaly_detector.py, and factored_network.py.

**Architecture:** Targeted fixes maintaining existing patterns. Action masking gets target_slot awareness and FOSSILIZED CULL blocking. Factored network gets mask validation. Anomaly detector gets type safety, configurable thresholds, and gradient norm detection.

**Tech Stack:** Python 3.13, PyTorch 2.9, pytest

---

## Summary of Fixes

| Priority | ID | Issue | File |
|----------|-----|-------|------|
| P1 | 001 | Multi-slot mask uses only first active seed | action_masks.py |
| P1 | 002 | CULL allowed on FOSSILIZED (should be masked) | action_masks.py |
| P1 | 003 | No mask validation for factored actions | factored_network.py |
| P2 | 004 | Missing type annotation for details dict | anomaly_detector.py |
| P2 | 005 | Misleading progress_pct when total_episodes=0 | anomaly_detector.py |
| P2 | 006 | Entropy normalization mismatch | factored_network.py |
| P2 | 007 | Redundant tensor allocations | action_masks.py |
| P3 | 008 | MIN_CULL_AGE re-export ambiguity | action_masks.py |

---

## Task 1: Add Target Slot Parameter to Action Masking (P1-001)

**Files:**
- Modify: `src/esper/simic/action_masks.py:75-145`
- Modify: `src/esper/simic/vectorized.py` (update call sites)
- Test: `tests/simic/test_action_masks.py`

**NOTE:** Per CLAUDE.md "No Legacy Code Policy", we make `target_slot` REQUIRED (no backward
compatibility fallback). All call sites will be updated in this commit.

### Step 1: Write failing test for target_slot parameter

```python
# Add to tests/simic/test_action_masks.py

def test_compute_action_masks_target_slot_determines_fossilize():
    """FOSSILIZE validity should be based on target_slot, not first active seed."""
    slot_states = {
        "early": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,  # Not fossilizable
            seed_age_epochs=10,
        ),
        "mid": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,  # Fossilizable
            seed_age_epochs=10,
        ),
    }

    # Target the PROBATIONARY slot - FOSSILIZE should be valid
    masks_mid = compute_action_masks(slot_states, target_slot="mid")
    assert masks_mid["op"][LifecycleOp.FOSSILIZE] == True

    # Target the TRAINING slot - FOSSILIZE should NOT be valid
    masks_early = compute_action_masks(slot_states, target_slot="early")
    assert masks_early["op"][LifecycleOp.FOSSILIZE] == False


def test_compute_action_masks_target_slot_determines_cull_age():
    """CULL validity should be based on target_slot's seed age."""
    slot_states = {
        "early": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=0,  # Too young to cull
        ),
        "mid": MaskSeedInfo(
            stage=SeedStage.TRAINING.value,
            seed_age_epochs=5,  # Old enough to cull
        ),
    }

    # Target the old seed - CULL should be valid
    masks_mid = compute_action_masks(slot_states, target_slot="mid")
    assert masks_mid["op"][LifecycleOp.CULL] == True

    # Target the young seed - CULL should NOT be valid
    masks_early = compute_action_masks(slot_states, target_slot="early")
    assert masks_early["op"][LifecycleOp.CULL] == False


def test_compute_action_masks_target_slot_empty():
    """Targeting empty slot should disable FOSSILIZE/CULL."""
    slot_states = {
        "early": None,  # Empty
        "mid": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,
            seed_age_epochs=10,
        ),
    }

    # Target the empty slot
    masks = compute_action_masks(slot_states, target_slot="early")
    assert masks["op"][LifecycleOp.FOSSILIZE] == False
    assert masks["op"][LifecycleOp.CULL] == False
    assert masks["op"][LifecycleOp.GERMINATE] == True


def test_compute_action_masks_target_slot_required():
    """target_slot is required - raises TypeError if not provided."""
    slot_states = {
        "mid": MaskSeedInfo(
            stage=SeedStage.PROBATIONARY.value,
            seed_age_epochs=10,
        ),
    }

    # target_slot is required, not optional
    import pytest
    with pytest.raises(TypeError):
        compute_action_masks(slot_states)  # Missing target_slot
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/simic/test_action_masks.py::test_compute_action_masks_target_slot_determines_fossilize -v`
Expected: FAIL (TypeError: unexpected keyword argument 'target_slot')

### Step 3: Implement target_slot parameter (REQUIRED, no default)

Modify `src/esper/simic/action_masks.py`:

```python
def compute_action_masks(
    slot_states: dict[str, MaskSeedInfo | None],
    target_slot: str,  # REQUIRED - no default, no backward compatibility
    total_seeds: int = 0,
    max_seeds: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks based on slot states.

    Only masks PHYSICALLY IMPOSSIBLE actions. Does not mask timing heuristics.

    Args:
        slot_states: Dict mapping slot_id to MaskSeedInfo or None
        target_slot: Slot ID to evaluate FOSSILIZE/CULL against (REQUIRED)
        total_seeds: Total number of active seeds across all slots
        max_seeds: Maximum allowed seeds (0 = unlimited)
        device: Torch device for tensors

    Returns:
        Dict of boolean tensors for each action head
    """
    device = device or torch.device("cpu")

    # Slot mask: all slots always selectable
    slot_mask = torch.ones(NUM_SLOTS, dtype=torch.bool, device=device)

    # Blueprint/blend: always all valid (network learns preferences)
    blueprint_mask = torch.ones(NUM_BLUEPRINTS, dtype=torch.bool, device=device)
    blend_mask = torch.ones(NUM_BLENDS, dtype=torch.bool, device=device)

    # Op mask: depends on slot states
    op_mask = torch.zeros(NUM_OPS, dtype=torch.bool, device=device)
    op_mask[LifecycleOp.WAIT] = True  # WAIT always valid

    # Check slot states
    has_empty_slot = any(info is None for info in slot_states.values())

    # Get target slot's seed info for FOSSILIZE/CULL decisions
    target_seed_info = slot_states.get(target_slot)

    # GERMINATE: valid if empty slot exists AND under seed limit
    if has_empty_slot:
        seed_limit_reached = max_seeds > 0 and total_seeds >= max_seeds
        if not seed_limit_reached:
            op_mask[LifecycleOp.GERMINATE] = True

    # FOSSILIZE/CULL: valid based on TARGET seed state
    if target_seed_info is not None:
        stage = target_seed_info.stage
        age = target_seed_info.seed_age_epochs

        # FOSSILIZE: only from PROBATIONARY
        if stage in _FOSSILIZABLE_STAGES:
            op_mask[LifecycleOp.FOSSILIZE] = True

        # CULL: only if seed age >= MIN_CULL_AGE
        if age >= MIN_CULL_AGE:
            op_mask[LifecycleOp.CULL] = True

    return {
        "slot": slot_mask,
        "blueprint": blueprint_mask,
        "blend": blend_mask,
        "op": op_mask,
    }
```

### Step 3b: Update ALL existing tests to pass target_slot

All existing tests in `test_action_masks.py` must be updated to pass `target_slot`.
Example for `test_compute_action_masks_empty_slots`:

```python
def test_compute_action_masks_empty_slots():
    """Empty slots should allow GERMINATE, not CULL/FOSSILIZE."""
    slot_states = {
        "early": None,
        "mid": None,
        "late": None,
    }

    masks = compute_action_masks(slot_states, target_slot="mid")  # ADD target_slot
    # ... rest unchanged
```

Update ALL tests similarly. Search for `compute_action_masks(` and add `target_slot=` parameter.

### Step 3c: Update call sites in production code

Search for `compute_action_masks` usage in:
- `src/esper/simic/vectorized.py`
- Any other files using this function

Update each call site to pass the appropriate `target_slot`.

### Step 4: Run tests to verify they pass

Run: `pytest tests/simic/test_action_masks.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/esper/simic/action_masks.py src/esper/simic/vectorized.py tests/simic/test_action_masks.py
git commit -m "feat(simic): add required target_slot parameter to compute_action_masks

Fixes P1-001: Multi-slot mask was using first active seed found instead
of the targeted slot. FOSSILIZE/CULL validity now computed from the
specific slot being targeted.

BREAKING: target_slot is now required (no backward compatibility per CLAUDE.md).
All call sites updated in this commit."
```

---

## Task 2: Block CULL on FOSSILIZED Stage (P1-002)

**Files:**
- Modify: `src/esper/simic/action_masks.py:32-35, 136-138`
- Test: `tests/simic/test_action_masks.py`

**NOTE:** We UPDATE the existing test, not create a duplicate (per Code Review feedback).

### Step 1: Add _CULLABLE_STAGES and update mask logic

Modify `src/esper/simic/action_masks.py`:

```python
# After _FOSSILIZABLE_STAGES definition (line 35), add:

# Stages from which a seed can be culled
# Derived as: active stages that have CULLED in VALID_TRANSITIONS
# Equivalently: set(active_stages) - {FOSSILIZED} (terminal success)
# See stages.py VALID_TRANSITIONS for authoritative source
_CULLABLE_STAGES = frozenset({
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
    SeedStage.PROBATIONARY.value,
    # NOT FOSSILIZED - terminal success, no outgoing transitions
})
```

Then update the CULL logic (around line 136-138):

```python
        # CULL: only from cullable stages AND if seed age >= MIN_CULL_AGE
        if stage in _CULLABLE_STAGES and age >= MIN_CULL_AGE:
            op_mask[LifecycleOp.CULL] = True
```

### Step 2: Update existing test (DO NOT create duplicate)

Modify the EXISTING `tests/simic/test_action_masks.py::test_compute_action_masks_fossilized_stage`
(around line 89-110). Change ONLY the CULL assertion and comment:

```python
def test_compute_action_masks_fossilized_stage():
    """FOSSILIZED stage should not allow GERMINATE, FOSSILIZE, or CULL."""
    slot_states = {
        "mid": MaskSeedInfo(
            stage=SeedStage.FOSSILIZED.value,
            seed_age_epochs=20,
        ),
    }

    masks = compute_action_masks(slot_states, target_slot="mid")  # Add target_slot from Task 1

    # WAIT always valid
    assert masks["op"][LifecycleOp.WAIT] == True

    # No GERMINATE (slot occupied)
    assert masks["op"][LifecycleOp.GERMINATE] == False

    # No FOSSILIZE (already fossilized)
    assert masks["op"][LifecycleOp.FOSSILIZE] == False

    # No CULL - FOSSILIZED is terminal success, cannot be removed
    assert masks["op"][LifecycleOp.CULL] == False
```

### Step 3: Run tests to verify the change

Run: `pytest tests/simic/test_action_masks.py::test_compute_action_masks_fossilized_stage -v`
Expected: PASS (after implementation in Step 1)

### Step 4: Run full test suite

Run: `pytest tests/simic/test_action_masks.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/esper/simic/action_masks.py tests/simic/test_action_masks.py
git commit -m "fix(simic): block CULL on FOSSILIZED stage

Fixes P1-002: FOSSILIZED is a terminal success state per stages.py.
Hard masking is superior to penalty-based learning per DRL expert:
- Zero samples needed (vs learning 'don't do this')
- No spurious credit assignment correlations
- Exploration budget focused on meaningful decisions

Added _CULLABLE_STAGES frozenset aligned with VALID_TRANSITIONS."
```

---

## Task 3: Add Mask Validation to Factored Network (P1-003)

**Files:**
- Modify: `src/esper/simic/factored_network.py:96-139`
- Test: `tests/simic/test_factored_network.py`

**NOTE:** Reuse existing `InvalidStateMachineError` from `networks.py` (already exists at line 346)
rather than creating a new errors.py file. This keeps error hierarchy consolidated.

### Step 1: Write failing test for mask validation

```python
# Add to tests/simic/test_factored_network.py

import pytest


def test_factored_actor_critic_raises_on_all_masked():
    """Should raise error if all actions in a head are masked."""
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.networks import InvalidStateMachineError

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)

    # Create masks that block ALL ops
    masks = {
        "slot": torch.ones(4, 3, dtype=torch.bool),
        "blueprint": torch.ones(4, 5, dtype=torch.bool),
        "blend": torch.ones(4, 3, dtype=torch.bool),
        "op": torch.zeros(4, 4, dtype=torch.bool),  # All ops masked!
    }

    with pytest.raises(InvalidStateMachineError, match="op"):
        net(obs, masks=masks)


def test_factored_actor_critic_raises_on_single_env_all_masked():
    """Should raise error if any single env has all actions masked."""
    from esper.simic.factored_network import FactoredActorCritic
    from esper.simic.networks import InvalidStateMachineError

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)

    # Create masks where env 2 has all ops masked
    masks = {
        "slot": torch.ones(4, 3, dtype=torch.bool),
        "blueprint": torch.ones(4, 5, dtype=torch.bool),
        "blend": torch.ones(4, 3, dtype=torch.bool),
        "op": torch.ones(4, 4, dtype=torch.bool),
    }
    masks["op"][2, :] = False  # Env 2 has all ops masked

    with pytest.raises(InvalidStateMachineError, match="op.*env 2"):
        net(obs, masks=masks)
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/simic/test_factored_network.py::test_factored_actor_critic_raises_on_all_masked -v`
Expected: FAIL (InvalidStateMachineError not raised)

### Step 3: Add validation to factored network forward pass

Modify `src/esper/simic/factored_network.py`:

```python
# Add import at top (reuse existing error from networks.py)
from esper.simic.networks import InvalidStateMachineError

# In forward() method, after mask application (around line 129), add:

        # Validate masks - at least one action must be valid per head per env
        if masks:
            for key, logits in [
                ("slot", slot_logits),
                ("blueprint", blueprint_logits),
                ("blend", blend_logits),
                ("op", op_logits),
            ]:
                if key in masks:
                    # Check each env in batch
                    valid_per_env = masks[key].any(dim=-1)  # (batch,)
                    if not valid_per_env.all():
                        invalid_envs = (~valid_per_env).nonzero(as_tuple=True)[0]
                        raise InvalidStateMachineError(
                            f"All actions masked for '{key}' head in env {invalid_envs[0].item()}. "
                            f"This indicates a bug in mask computation."
                        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/simic/test_factored_network.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/esper/simic/factored_network.py tests/simic/test_factored_network.py
git commit -m "fix(simic): add mask validation to FactoredActorCritic

Fixes P1-003: Without validation, all-masked heads produce
Categorical(logits=all_negative_inf) -> NaN probabilities -> silent
training failure.

Reuses InvalidStateMachineError from networks.py for consistency."
```

---

## Task 4: Fix Type Annotation for AnomalyReport.details (P2-004)

**Files:**
- Modify: `src/esper/simic/anomaly_detector.py:17`

### Step 1: No test needed - type annotation only

This is a type annotation fix. Pyright/mypy will validate.

### Step 2: Fix the type annotation

Modify `src/esper/simic/anomaly_detector.py` line 17:

```python
    details: dict[str, str] = field(default_factory=dict)
```

### Step 3: Run type checker

Run: `pyright src/esper/simic/anomaly_detector.py`
Expected: No errors

### Step 4: Commit

```bash
git add src/esper/simic/anomaly_detector.py
git commit -m "fix(simic): add type annotation to AnomalyReport.details

Fixes P2-004: details dict was untyped, now dict[str, str]."
```

---

## Task 5: Fix Misleading progress_pct Display (P2-005)

**Files:**
- Modify: `src/esper/simic/anomaly_detector.py:146`
- Test: `tests/simic/test_anomaly_detector.py`

### Step 1: Write failing test

```python
# Add to tests/simic/test_anomaly_detector.py

def test_value_collapse_detail_shows_unknown_when_no_total():
    """Should show 'unknown' progress when total_episodes is 0."""
    detector = AnomalyDetector()
    report = detector.check_value_function(
        explained_variance=0.05,  # Below threshold
        current_episode=10,
        total_episodes=0,  # Unknown total
    )
    assert report.has_anomaly is True
    assert "unknown" in report.details["value_collapse"].lower()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/simic/test_anomaly_detector.py::test_value_collapse_detail_shows_unknown_when_no_total -v`
Expected: FAIL (AssertionError: 'unknown' not in '... at 0% training')

### Step 3: Fix the progress display

Modify `src/esper/simic/anomaly_detector.py` line 146:

```python
        if explained_variance < threshold:
            if total_episodes > 0:
                progress_pct = current_episode / total_episodes * 100
                progress_str = f"{progress_pct:.0f}%"
            else:
                progress_str = "unknown"
            report.add_anomaly(
                "value_collapse",
                f"explained_variance={explained_variance:.3f} < {threshold} (at {progress_str} training)",
            )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/simic/test_anomaly_detector.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/esper/simic/anomaly_detector.py tests/simic/test_anomaly_detector.py
git commit -m "fix(simic): show 'unknown' progress when total_episodes=0

Fixes P2-005: Was showing '0%' which is misleading when we don't
know the total."
```

---

## Task 6: Normalize Entropy in Factored Network (P2-006)

**Files:**
- Modify: `src/esper/simic/factored_network.py:191-196`
- Test: `tests/simic/test_factored_network.py`

### Step 1: Write failing test for normalized entropy

```python
# Add to tests/simic/test_factored_network.py

def test_factored_actor_critic_entropy_normalized():
    """Entropy should be normalized to [0, 1] range per head, then summed."""
    from esper.simic.factored_network import FactoredActorCritic

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    obs = torch.randn(4, 30)
    actions = {
        "slot": torch.randint(0, 3, (4,)),
        "blueprint": torch.randint(0, 5, (4,)),
        "blend": torch.randint(0, 3, (4,)),
        "op": torch.randint(0, 4, (4,)),
    }

    _, _, entropy = net.evaluate_actions(obs, actions)

    # With 4 heads, normalized entropy should be in [0, 4]
    # (each head contributes 0-1)
    assert (entropy >= 0).all()
    assert (entropy <= 4.0).all()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/simic/test_factored_network.py::test_factored_actor_critic_entropy_normalized -v`
Expected: FAIL (entropy can exceed 4.0 with raw entropy)

### Step 3: Implement normalized entropy

Modify `src/esper/simic/factored_network.py`:

**First, add import at top of file:**
```python
import math  # Add if not already present (it is - line 9)
```

**Then update `evaluate_actions`:**

```python
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            log_probs: Sum of log probs across heads
            values: Value estimates
            entropy: Sum of NORMALIZED entropies across heads (each in [0, 1])

        Note on entropy normalization:
            We normalize by FULL action space (log(num_actions)), not by valid
            actions under the mask. This means:
            - "Uniform over 2 valid actions" reports ~50% entropy (not 100%)
            - entropy_coef has consistent meaning across mask states
            - Exploration bonus is weaker when fewer actions available

            This is intentional: we want consistent regularization strength
            regardless of how restrictive the mask is.
        """
        dists, values = self.forward(states, masks)

        log_probs_list = []
        entropy_list = []

        # Max entropies for normalization (full action space, not masked)
        # See docstring for design rationale
        max_entropies = {
            "slot": math.log(self.num_slots),
            "blueprint": math.log(self.num_blueprints),
            "blend": math.log(self.num_blends),
            "op": math.log(self.num_ops),
        }

        for key, dist in dists.items():
            log_probs_list.append(dist.log_prob(actions[key]))

            # Normalize entropy to [0, 1] by full action space
            raw_entropy = dist.entropy()
            max_ent = max_entropies[key]
            normalized_entropy = raw_entropy / max(max_ent, 1e-8)
            entropy_list.append(normalized_entropy)

        log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
        # Sum normalized entropies (not mean) - each head contributes equally
        # to the exploration budget. Range: [0, num_heads] (here [0, 4])
        entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)

        return log_probs, values, entropy
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/simic/test_factored_network.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/esper/simic/factored_network.py tests/simic/test_factored_network.py
git commit -m "fix(simic): normalize entropy per head in FactoredActorCritic

Fixes P2-006: Raw entropy sum could be ~4.5 nats vs normalized [0,1]
used by MaskedCategorical. Same entropy_coef now has consistent effect.

Sum (not mean) is intentional: each head contributes equally to
exploration budget, and collapsed heads are penalized appropriately."
```

---

## Task 7: Refactor compute_batch_masks (P1-001, P1-002 continued)

**Files:**
- Modify: `src/esper/simic/action_masks.py:148-208`
- Test: `tests/simic/test_action_masks.py`

**NOTE:** Per "no tech debt" policy, we:
1. Make `target_slots` REQUIRED (consistent with `target_slot` in Task 1)
2. Refactor to call `compute_action_masks` internally (single source of truth)

### Step 1: Write failing test for batch masks with target slots

```python
# Add to tests/simic/test_action_masks.py

def test_compute_batch_masks_with_target_slots():
    """Batch masks should respect per-env target slots."""
    batch_slot_states = [
        # Env 0: mid is TRAINING, target mid
        {
            "early": None,
            "mid": MaskSeedInfo(stage=SeedStage.TRAINING.value, seed_age_epochs=5),
        },
        # Env 1: mid is PROBATIONARY, target mid
        {
            "early": None,
            "mid": MaskSeedInfo(stage=SeedStage.PROBATIONARY.value, seed_age_epochs=5),
        },
        # Env 2: mid is FOSSILIZED, target mid
        {
            "early": None,
            "mid": MaskSeedInfo(stage=SeedStage.FOSSILIZED.value, seed_age_epochs=20),
        },
    ]
    target_slots = ["mid", "mid", "mid"]

    masks = compute_batch_masks(batch_slot_states, target_slots=target_slots)

    # Env 0: TRAINING - can CULL, not FOSSILIZE
    assert masks["op"][0, LifecycleOp.CULL] == True
    assert masks["op"][0, LifecycleOp.FOSSILIZE] == False

    # Env 1: PROBATIONARY - can CULL and FOSSILIZE
    assert masks["op"][1, LifecycleOp.CULL] == True
    assert masks["op"][1, LifecycleOp.FOSSILIZE] == True

    # Env 2: FOSSILIZED - cannot CULL or FOSSILIZE
    assert masks["op"][2, LifecycleOp.CULL] == False
    assert masks["op"][2, LifecycleOp.FOSSILIZE] == False


def test_compute_batch_masks_target_slots_required():
    """target_slots is required - raises TypeError if not provided."""
    batch_slot_states = [{"mid": None}]

    import pytest
    with pytest.raises(TypeError):
        compute_batch_masks(batch_slot_states)  # Missing target_slots
```

### Step 2: Run test to verify it fails

Run: `pytest tests/simic/test_action_masks.py::test_compute_batch_masks_with_target_slots -v`
Expected: FAIL

### Step 3: Refactor compute_batch_masks to delegate to compute_action_masks

```python
def compute_batch_masks(
    batch_slot_states: list[dict[str, MaskSeedInfo | None]],
    target_slots: list[str],  # REQUIRED - no backward compatibility
    total_seeds_list: list[int] | None = None,
    max_seeds: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action masks for a batch of observations.

    Delegates to compute_action_masks for each env, then stacks results.
    This ensures single source of truth for masking logic.

    Args:
        batch_slot_states: List of slot state dicts, one per env
        target_slots: List of target slot IDs per env (REQUIRED)
        total_seeds_list: List of total seeds per env (None = all 0)
        max_seeds: Maximum allowed seeds (0 = unlimited)
        device: Torch device for tensors

    Returns:
        Dict of boolean tensors (batch_size, num_actions) for each head
    """
    device = device or torch.device("cpu")

    # Delegate to compute_action_masks for each env
    masks_list = [
        compute_action_masks(
            slot_states=slot_states,
            target_slot=target_slots[i],
            total_seeds=total_seeds_list[i] if total_seeds_list else 0,
            max_seeds=max_seeds,
            device=device,
        )
        for i, slot_states in enumerate(batch_slot_states)
    ]

    # Stack into batch tensors
    return {
        key: torch.stack([m[key] for m in masks_list])
        for key in masks_list[0]
    }
```

### Step 3b: Update ALL existing tests to pass target_slots

All existing tests using `compute_batch_masks` must be updated to pass `target_slots`.
Search for `compute_batch_masks(` and add `target_slots=` parameter.

### Step 4: Run tests to verify they pass

Run: `pytest tests/simic/test_action_masks.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add src/esper/simic/action_masks.py tests/simic/test_action_masks.py
git commit -m "refactor(simic): compute_batch_masks delegates to compute_action_masks

Applies P1-001 and P1-002 fixes to batch function:
- target_slots is now REQUIRED (no backward compatibility)
- Delegates to compute_action_masks for single source of truth
- Eliminates code duplication between single and batch functions"
```

---

## Task 8: Remove MIN_CULL_AGE from __all__ (P3-008)

**Files:**
- Modify: `src/esper/simic/action_masks.py:211-217`
- Modify: `tests/simic/test_action_masks.py:17-18, 334-336` (update imports)

**NOTE:** Must also update test imports to use canonical source (per Code Review feedback).

### Step 1: Update test imports

Modify `tests/simic/test_action_masks.py` imports (around line 13-20):

```python
from esper.simic.action_masks import (
    MaskSeedInfo,
    compute_action_masks,
    compute_batch_masks,
    # MIN_CULL_AGE removed - now import from leyline
)
from esper.leyline import SeedStage, MIN_CULL_AGE  # ADD MIN_CULL_AGE here
from esper.leyline.factored_actions import LifecycleOp, NUM_OPS
```

### Step 2: Remove re-export

Modify `src/esper/simic/action_masks.py` __all__:

```python
__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    # MIN_CULL_AGE removed - import from esper.leyline instead
]
```

### Step 3: Run tests to verify imports work

Run: `pytest tests/simic/test_action_masks.py::test_min_cull_age_constant -v`
Expected: PASS

### Step 4: Commit

```bash
git add src/esper/simic/action_masks.py tests/simic/test_action_masks.py
git commit -m "refactor(simic): remove MIN_CULL_AGE from action_masks exports

Fixes P3-008: Canonical source is esper.leyline, re-export created
import ambiguity. Test imports updated accordingly."
```

---

## Task 9: Run Full Test Suite and Type Checks

### Step 1: Run all simic tests

Run: `pytest tests/simic/ -v`
Expected: All tests PASS

### Step 2: Run type checker

Run: `pyright src/esper/simic/`
Expected: No errors

### Step 3: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

### Step 4: (Optional) Verify torch.compile compatibility

Add a compile verification test to `tests/simic/test_factored_network.py`:

```python
def test_factored_actor_critic_compile_compatible():
    """Verify no graph breaks in forward pass with valid masks."""
    from esper.simic.factored_network import FactoredActorCritic
    import torch

    net = FactoredActorCritic(state_dim=30, num_slots=3, num_blueprints=5, num_blends=3, num_ops=4)

    # Compile with fullgraph=True - will error if any graph breaks
    compiled_net = torch.compile(net, fullgraph=True)

    obs = torch.randn(4, 30)
    masks = {
        "slot": torch.ones(4, 3, dtype=torch.bool),
        "blueprint": torch.ones(4, 5, dtype=torch.bool),
        "blend": torch.ones(4, 3, dtype=torch.bool),
        "op": torch.ones(4, 4, dtype=torch.bool),
    }
    masks["op"][:, 3] = False  # Mask one op (but not all)

    # Should not raise
    dists, value = compiled_net(obs, masks=masks)
    assert value.shape == (4,)
```

Run: `pytest tests/simic/test_factored_network.py::test_factored_actor_critic_compile_compatible -v`
Expected: PASS (confirms no graph breaks)

---

## Task 10: Update Docstrings in Module Header

**Files:**
- Modify: `src/esper/simic/action_masks.py:1-11`

### Step 1: Update module docstring

```python
"""Action Masking for Multi-Slot Control.

Only masks PHYSICALLY IMPOSSIBLE actions:
- GERMINATE: blocked if slot occupied OR at seed limit
- FOSSILIZE: blocked if target slot not PROBATIONARY
- CULL: blocked if target slot has no seed OR seed_age < MIN_CULL_AGE
        OR target slot is FOSSILIZED (terminal state)
- WAIT: always valid

Does NOT mask timing heuristics (epoch, plateau, stabilization).
Tamiyo learns optimal timing from counterfactual reward signals.

The target_slot parameter determines which slot's state is used for
FOSSILIZE/CULL validity checks. This is critical for multi-slot scenarios
where different slots may have seeds at different lifecycle stages.
"""
```

### Step 2: Commit

```bash
git add src/esper/simic/action_masks.py
git commit -m "docs(simic): update action_masks docstring for target_slot"
```

---

## Verification Checklist

After all tasks, verify:

- [ ] `pytest tests/simic/test_action_masks.py -v` - All pass
- [ ] `pytest tests/simic/test_anomaly_detector.py -v` - All pass
- [ ] `pytest tests/simic/test_factored_network.py -v` - All pass
- [ ] `pyright src/esper/simic/` - No errors
- [ ] `pytest tests/ -v` - Full suite passes

---

## Summary of Changes

| File | Changes |
|------|---------|
| `action_masks.py` | Added `target_slot`/`target_slots` params, `_CULLABLE_STAGES`, updated docstrings |
| `anomaly_detector.py` | Fixed type annotation, progress display |
| `factored_network.py` | Added mask validation, normalized entropy |
| `errors.py` | New file with `InvalidActionMaskError` |
| Tests | New tests for all P1/P2 fixes |
