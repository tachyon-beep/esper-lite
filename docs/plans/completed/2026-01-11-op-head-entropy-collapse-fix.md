# Implementation Plan: Op Head Entropy Collapse Fix

## Problem Summary

Esper's PPO agent suffers from **multi-stage entropy collapse** that freezes the policy:

| Stage | Batches | Op Entropy | Blueprint Entropy | GERMINATE Rate | Fossilizations |
|-------|---------|------------|-------------------|----------------|----------------|
| **Learning** | 1-20 | 0.51 → 0.34 | 0.125 → 0.036 | 3.68% | 47 |
| **Collapsing** | 20-40 | 0.34 → 0.16 | 0.036 → 0.008 | 0.15% | 1 |
| **Frozen** | 40+ | ~0.14 | **0.000** | 0.07% | 0 |

### Root Cause Analysis (DRL Expert, 2026-01-11)

**The primary driver is op head entropy collapse, not sparse head entropy.**

1. **Op head collapses to WAIT (98%)** → GERMINATE rarely selected
2. **Blueprint head only trains during GERMINATE** → receives almost no gradients
3. **Blueprint entropy goes to zero** → policy can't explore blueprints
4. **Death spiral**: Low blueprint entropy → Policy avoids GERMINATE → Blueprint gets no gradients → Blueprint entropy stays low

**Critical Finding:** The existing entropy floor (0.15) is too close to the collapse point (0.14). The policy stabilizes just above the floor but in a degenerate state.

### Evidence: Blueprint Selection Collapse

When the frozen policy does germinate, it picks the **worst** blueprint:

| Quartile | conv_small % | Fossilizations |
|----------|--------------|----------------|
| Q1 | 20% | 47 |
| Q2 | 63% | 1 |
| Q3 | 60% | 0 |
| Q4 | **72%** | 0 |

conv_small has **0% fossilization rate** (699 germinated → 616 pruned → 0 fossilized). The policy learned to pick the worst option because the blueprint head stopped receiving gradients before it could learn.

---

## Solution: Two-Pronged Attack

### Prong 1: Higher Probability Floors (Hard Guarantee)

Probability floors guarantee minimum probability mass on all valid actions, ensuring gradients can always flow even when entropy approaches zero.

**Key insight from DRL expert:** The op head floor must guarantee enough GERMINATE actions to keep sparse heads learning.

### Prong 2: Higher Entropy Floors (Soft Pressure)

Increase entropy floor targets and penalty coefficients to push away from collapse earlier.

---

## Implementation Overview

| Phase | Description | Files | Priority |
|-------|-------------|-------|----------|
| 1 | Update constants in leyline | `leyline/__init__.py` | CRITICAL |
| 2 | Modify MaskedCategorical | `tamiyo/policy/action_masks.py` | CRITICAL |
| 3 | Wire through evaluate_actions | `tamiyo/networks/factored_lstm.py` | HIGH |
| 4 | Wire through get_action | `tamiyo/networks/factored_lstm.py` | HIGH |
| 5 | Wire through PPOAgent | `simic/agent/ppo_agent.py` | HIGH |
| 6 | Add conditional entropy metrics | `simic/agent/ppo_agent.py` | MEDIUM |
| 7 | Add unit tests | `tests/tamiyo/policy/test_probability_floor.py` | HIGH |
| 7b | Add property tests | `tests/tamiyo/properties/test_probability_floor_properties.py` | MEDIUM |
| 8 | **Specialist Sign-off** | Review by PyTorch, Quality, Python engineers | Gate |

---

## Phase 1: Update Constants in Leyline

**File:** `src/esper/leyline/__init__.py`

### 1a. Add Probability Floors (NEW)

**Location:** After `ENTROPY_FLOOR_PENALTY_COEF` (around line 225)

```python
# Per-head probability floor (guarantees minimum exploration mass)
# These are HARD floors enforced in MaskedCategorical - probabilities are
# clamped and renormalized, ensuring gradients can always flow.
#
# DRL Expert diagnosis (2026-01-11): Op head collapse to WAIT is the root cause.
# When op chooses WAIT, sparse heads (blueprint, tempo) receive no gradients.
# The op floor of 0.05 guarantees ~5% non-WAIT actions, keeping sparse heads alive.
PROBABILITY_FLOOR_PER_HEAD: dict[str, float] = {
    "op": 0.05,           # CRITICAL: Guarantees ~5% non-WAIT to feed sparse heads
    "slot": 0.03,         # Usually few valid choices anyway
    "blueprint": 0.10,    # GERMINATE only (~5%) - needs high floor when active
    "style": 0.05,        # GERMINATE + SET_ALPHA_TARGET (~7%)
    "tempo": 0.10,        # GERMINATE only (~5%) - needs high floor when active
    "alpha_target": 0.05, # GERMINATE + SET_ALPHA_TARGET (~7%)
    "alpha_speed": 0.05,  # SET_ALPHA_TARGET + PRUNE (~7%)
    "alpha_curve": 0.05,  # SET_ALPHA_TARGET + PRUNE (~7%)
}
```

### 1b. Update Entropy Floors (MODIFY EXISTING)

```python
# Per-head entropy floor targets (normalized entropy, 0-1 scale)
# DRL Expert update (2026-01-11): Increased op floor from 0.15 to 0.25
# to push away from collapse earlier. Previous floor was too close to
# the collapse point (0.14), allowing degenerate equilibrium.
ENTROPY_FLOOR_PER_HEAD: dict[str, float] = {
    "op": 0.25,           # INCREASED from 0.15 - collapse point was 0.14!
    "slot": 0.15,
    "blueprint": 0.20,    # INCREASED from 0.15 - needs room to explore
    "style": 0.15,
    "tempo": 0.20,        # INCREASED from 0.15 - needs room to explore
    "alpha_target": 0.10,
    "alpha_speed": 0.10,
    "alpha_curve": 0.10,
}

# Per-head entropy floor penalty coefficients
# DRL Expert update (2026-01-11): Increased blueprint/tempo from 0.1 to 0.3
# Sparse heads need stronger penalty to overcome gradient starvation.
ENTROPY_FLOOR_PENALTY_COEF: dict[str, float] = {
    "op": 0.2,            # INCREASED from 0.1 - critical head
    "slot": 0.1,
    "blueprint": 0.3,     # INCREASED from 0.1 - sparse head needs strong penalty
    "style": 0.1,
    "tempo": 0.3,         # INCREASED from 0.1 - sparse head needs strong penalty
    "alpha_target": 0.1,
    "alpha_speed": 0.1,
    "alpha_curve": 0.1,
}
```

**Also:** Add `"PROBABILITY_FLOOR_PER_HEAD"` to `__all__` list.

---

## Phase 2: Modify MaskedCategorical

**File:** `src/esper/tamiyo/policy/action_masks.py`

**Changes:**

1. Add `import torch.nn.functional as F` at top

2. Modify `__init__` signature to accept `min_prob: float | None = None`

3. Add `_apply_probability_floor()` method:

```python
def _apply_probability_floor(
    self,
    logits: torch.Tensor,
    mask: torch.Tensor,
    min_prob: float,
) -> torch.Tensor:
    """Apply probability floor to valid actions and renormalize.

    Ensures all valid actions have at least min_prob probability,
    which guarantees gradient flow even when entropy would otherwise collapse.

    This is the HARD floor that prevents the death spiral where:
    1. Low entropy → peaked distribution → near-zero gradients for non-peak actions
    2. No gradients → entropy stays low → policy frozen

    By guaranteeing minimum probability, we ensure gradients always flow to
    all valid actions, allowing the policy to escape degenerate equilibria.
    """
    probs = F.softmax(logits, dim=-1)

    # Cap floor at uniform distribution (can't exceed 1/num_valid)
    num_valid = mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
    max_floor = 1.0 / num_valid
    effective_floor = torch.minimum(
        torch.tensor(min_prob, device=logits.device),
        max_floor * 0.99  # Leave 1% headroom for numerical stability
    )

    # Clamp valid probs to at least effective_floor
    floored_probs = torch.where(
        mask,
        torch.clamp(probs, min=effective_floor),
        probs
    )

    # Renormalize over valid actions
    valid_sum = (floored_probs * mask.float()).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized_probs = floored_probs / valid_sum

    # Convert back to logits (required for torch.distributions compatibility)
    safe_probs = torch.where(mask, normalized_probs.clamp(min=1e-8), torch.ones_like(normalized_probs))
    new_logits = torch.log(safe_probs)
    new_logits = torch.where(mask, new_logits, torch.full_like(new_logits, MASKED_LOGIT_VALUE))

    return new_logits
```

4. Call `_apply_probability_floor()` in `__init__` after masking:

```python
if min_prob is not None and min_prob > 0:
    self.masked_logits = self._apply_probability_floor(
        self.masked_logits, mask, min_prob
    )
```

---

## Phase 3: Wire Through evaluate_actions()

**File:** `src/esper/tamiyo/networks/factored_lstm.py`

**Method:** `evaluate_actions()` (around line 1050)

1. Add parameter: `probability_floor: dict[str, float] | None = None`

2. In the head loop (around line 1092), pass per-head floor:

```python
min_prob = probability_floor.get(key) if probability_floor else None
dist = MaskedCategorical(logits=logits_flat, mask=mask_flat, min_prob=min_prob)
```

---

## Phase 4: Wire Through get_action()

**File:** `src/esper/tamiyo/networks/factored_lstm.py`

**Method:** `get_action()`

1. Add parameter: `probability_floor: dict[str, float] | None = None`

2. Add helper method `_apply_probability_floor_to_probs()` for the fast path

3. Modify `_sample_head()` closure to apply floor before multinomial sampling

4. Ensure log_prob is computed from floored distribution (for rollout consistency)

**Critical:** The floor must be applied in both `get_action()` (rollout collection) and `evaluate_actions()` (training) to ensure consistency. A mismatch would cause importance sampling ratio errors.

---

## Phase 5: Wire Through PPOAgent

**File:** `src/esper/simic/agent/ppo_agent.py`

1. Add import: `from esper.leyline import PROBABILITY_FLOOR_PER_HEAD`

2. Add `__init__` parameter: `probability_floor: dict[str, float] | None = None`

3. Store: `self.probability_floor = probability_floor or dict(PROBABILITY_FLOOR_PER_HEAD)`

4. Pass to `policy.evaluate_actions()` in `update()` method

5. Pass to `policy.get_action()` calls (if any in agent)

6. Add to checkpoint save/load for persistence

---

## Phase 6: Add Conditional Entropy Metrics

**File:** `src/esper/simic/agent/ppo_agent.py`

Track entropy only when head is causally relevant (has >1 valid action):

```python
# In update() epoch loop, after computing entropy:
for key in HEAD_NAMES:
    causal_mask = head_masks[key]
    n_relevant = causal_mask.sum()
    if n_relevant > 0:
        conditional_ent = (entropy[key] * causal_mask).sum() / n_relevant
        # Store for metrics
```

Add to telemetry output as `conditional_{head}_entropy` fields.

**Why this matters:** The current `head_blueprint_entropy` metric averages entropy across ALL samples, including ones where GERMINATE wasn't selected (so blueprint head wasn't active). This dilutes the signal. Conditional entropy shows the true entropy when the head is active.

---

## Phase 7: Unit Tests

**New file:** `tests/tamiyo/policy/test_probability_floor.py`

**Key test cases:**

1. `test_no_floor_preserves_original_behavior` — min_prob=None should be unchanged
2. `test_floor_guarantees_minimum_probability` — all valid probs >= min_prob
3. `test_floor_only_affects_valid_actions` — masked actions stay at ~0
4. `test_probabilities_sum_to_one` — distribution still normalized
5. `test_gradient_flows_at_low_entropy` — **CRITICAL:** gradients non-zero even with peaked logits
6. `test_floor_caps_at_uniform` — can't exceed 1/num_valid
7. `test_log_prob_correct_with_floor` — log_prob matches floored distribution
8. `test_op_floor_guarantees_germinate_probability` — **NEW:** verify op floor prevents WAIT dominance

---

## Phase 7b: Property Tests

**New file:** `tests/tamiyo/properties/test_probability_floor_properties.py`

Property-based tests using Hypothesis to verify mathematical invariants hold across diverse inputs.

### Strategies

```python
@st.composite
def masked_logits(draw, min_actions=2, max_actions=10):
    """Generate logits with valid action masks."""
    n_actions = draw(st.integers(min_value=min_actions, max_value=max_actions))
    logits = draw(pytorch_tensors(shape=(n_actions,), min_value=-10.0, max_value=10.0))

    # At least one action must be valid
    n_valid = draw(st.integers(min_value=1, max_value=n_actions))
    valid_indices = draw(st.permutations(range(n_actions)))[:n_valid]
    mask = torch.zeros(n_actions, dtype=torch.bool)
    mask[list(valid_indices)] = True

    return logits, mask

@st.composite
def peaked_logits(draw, n_actions=5, peak_magnitude=10.0):
    """Generate logits strongly peaked at one action (pathological case)."""
    logits = torch.zeros(n_actions)
    peak_idx = draw(st.integers(min_value=0, max_value=n_actions-1))
    logits[peak_idx] = peak_magnitude
    mask = torch.ones(n_actions, dtype=torch.bool)
    return logits, mask, peak_idx
```

### Property Test Classes

**1. `TestProbabilityFloorDistributionInvariants`**

| Test | Property |
|------|----------|
| `test_probabilities_sum_to_one` | ∑ P(valid actions) = 1.0 (±1e-5) |
| `test_floor_guarantee` | P(a) ≥ effective_floor for all valid a |
| `test_mask_preservation` | P(invalid action) ≈ 0 |
| `test_floor_caps_at_uniform` | effective_floor ≤ 0.99 / num_valid |

**2. `TestGradientFlowProperty`** (CRITICAL - core requirement)

| Test | Property |
|------|----------|
| `test_gradient_nonzero_at_peak` | For peaked logits, ∂log_prob/∂logits ≠ 0 |
| `test_gradient_magnitude_reasonable` | Gradients don't explode (‖∇‖ < 100) |

**3. `TestSamplingProperties`**

| Test | Property |
|------|----------|
| `test_samples_always_valid` | Sampled action always satisfies mask |
| `test_log_prob_matches_floored_distribution` | log_prob(a) = log(floored_probs[a]) |

**4. `TestIdempotenceAndEdgeCases`**

| Test | Property |
|------|----------|
| `test_idempotence` | apply_floor(apply_floor(x)) == apply_floor(x) |
| `test_single_valid_action` | Floor doesn't break with num_valid=1 |
| `test_no_floor_preserves_original` | min_prob=None identical to original |

---

## Phase 8: Specialist Sign-off Gate

**MANDATORY:** Tests must be reviewed and approved by three specialist agents before implementation is considered complete.

### Required Reviewers

| Specialist | Focus Areas | Agent |
|------------|-------------|-------|
| **PyTorch Engineer** | torch.compile compatibility, gradient flow correctness, tensor operations, numerical stability | `pytorch-expert` |
| **Quality Engineer** | Test coverage, property test design, edge case coverage, test pyramid balance | `ordis-quality-engineering` |
| **Python Engineer** | Code patterns, type safety, API design, Hypothesis strategy quality | `axiom-python-engineering` |

### Sign-off Checklist

Each reviewer must confirm:

- [ ] **PyTorch Engineer**
  - [ ] `_apply_probability_floor()` is torch.compile friendly (no graph breaks)
  - [ ] Gradient flow test correctly validates the core requirement
  - [ ] Numerical stability (no NaN/Inf edge cases)
  - [ ] Tensor operations are efficient (no unnecessary allocations)

- [ ] **Quality Engineer**
  - [ ] Property tests cover all mathematical invariants
  - [ ] Edge cases adequately covered (single valid action, floor > 1/n, peaked logits)
  - [ ] Test pyramid balanced (unit + property + integration)
  - [ ] Hypothesis settings appropriate (max_examples, deadline)

- [ ] **Python Engineer**
  - [ ] Type annotations complete and correct
  - [ ] API design follows existing patterns (MaskedCategorical signature)
  - [ ] Hypothesis strategies are reusable and well-documented
  - [ ] No anti-patterns (defensive programming, isinstance abuse)

---

## Critical Files Summary

| File | Purpose |
|------|---------|
| `src/esper/leyline/__init__.py` | Define PROBABILITY_FLOOR_PER_HEAD, update ENTROPY_FLOOR_* |
| `src/esper/tamiyo/policy/action_masks.py` | Core probability floor logic in MaskedCategorical |
| `src/esper/tamiyo/networks/factored_lstm.py` | Wire floor through evaluate_actions() and get_action() |
| `src/esper/simic/agent/ppo_agent.py` | Wire floor through PPOAgent, add conditional metrics |
| `tests/tamiyo/policy/test_probability_floor.py` | Unit tests for probability floor |
| `tests/tamiyo/properties/test_probability_floor_properties.py` | Property tests (Hypothesis) |

---

## Verification Plan

### Unit Tests
```bash
uv run pytest tests/tamiyo/policy/test_probability_floor.py -v
```

### Property Tests
```bash
uv run pytest tests/tamiyo/properties/test_probability_floor_properties.py -v --hypothesis-show-statistics
```

### Integration Test
```bash
# Short training run to verify no crashes
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 \
    --task cifar10 \
    --rounds 5 \
    --envs 2 \
    --episode-length 20 \
    --no-tui
```

### Verify Entropy Collapse Prevention
After implementation, run training and check:
1. **Op head entropy stays above 0.20** (not collapsing to 0.14)
2. **Blueprint/tempo entropy stays above 0.05** (not collapsing to 0.00)
3. **GERMINATE rate stays above 2%** (not collapsing to 0.07%)
4. **Fossilizations continue throughout training** (not just Q1)
5. **Blueprint diversity** - not 72% conv_small

### Telemetry Queries

Check entropy stability:
```sql
SELECT
    batch_number,
    ROUND(head_op_entropy, 3) as op_ent,
    ROUND(head_blueprint_entropy, 3) as bp_ent,
    ROUND(head_tempo_entropy, 3) as tempo_ent
FROM ppo_updates
WHERE run_dir = '<new_run>'
ORDER BY batch_number
```

Check GERMINATE rate:
```sql
SELECT
    batch_number,
    ROUND(100.0 * SUM(CASE WHEN action_name = 'GERMINATE' THEN 1 ELSE 0 END) / COUNT(*), 2) as germinate_pct
FROM decisions
WHERE run_dir = '<new_run>'
GROUP BY batch_number
ORDER BY batch_number
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Op floor = 0.05 | Yes | Guarantees ~5% non-WAIT to feed sparse heads |
| Op entropy floor = 0.25 | Yes | Previous 0.15 was too close to collapse point (0.14) |
| Blueprint/tempo penalty coef = 0.3 | Yes | Sparse heads need stronger penalty to overcome gradient starvation |
| Apply floor in `__init__` | Yes | Single modification point, affects sample/log_prob/entropy consistently |
| Mask first, then floor | Yes | Invalid actions must stay at ~0; floor only affects valid choices |
| Apply in get_action() too | Yes | Rollout collection must match training evaluation |
| Keep entropy floor penalty | Yes | Belt and suspenders; soft penalty + hard floor |

---

## Rollback Plan

If issues arise:
1. Set `PROBABILITY_FLOOR_PER_HEAD` values to 0.0 (disables hard floor)
2. Revert `ENTROPY_FLOOR_PER_HEAD` to previous values
3. Revert `ENTROPY_FLOOR_PENALTY_COEF` to previous values
4. Or pass `probability_floor=None` to PPOAgent constructor
5. Entropy floor penalty system still provides soft backup

---

## Future Considerations

### ESCROW-Specific Exploration Bonus (Not in Scope)

The DRL expert suggested adding an exploration bonus when escrow credit has been stable for N steps:

```python
# Potential future enhancement - NOT implementing now
if escrow_stable_for_n_steps > 10:
    exploration_bonus = 0.1  # Incentivize breaking out of stable state
```

This is deferred until we confirm the probability floor fix is sufficient.

### Availability-Based Entropy Regularization (Not in Scope)

Current entropy floor only applies when head is active. A more aggressive approach would penalize low blueprint/tempo entropy even when GERMINATE wasn't chosen (but was valid). This could prevent the death spiral more directly but adds complexity. Deferred pending results.
