# Scaffold Hindsight Credit - Detailed Design

**Date:** 2025-12-26
**Status:** Revised - Incorporating Specialist Feedback
**Parent Plan:** `docs/plans/2025-12-23-seed-scaffolding-support.md` (Phase 3.2)

## Problem Statement

The `compute_scaffold_hindsight_credit()` function exists but is never called. When a beneficiary seed fossilizes successfully after receiving a boost from a scaffold seed, the scaffold receives no retroactive credit for its contribution.

**Current behavior:**
1. Seed A boosts Seed B (synergy bonus given to both ✅)
2. Seed A gets pruned (no more rewards for A)
3. Seed B fossilizes successfully (A should get credit, but **doesn't**)

**Desired behavior:**
When Seed B fossilizes with positive improvement, any seeds that provided positive `boost_received` to B should receive credit proportional to their contribution.

---

## Specialist Review Summary

### PyTorch Specialist Feedback (2025-12-26)

| Issue | Resolution |
|-------|------------|
| **Wrong dataclass target** | Changed from Karn `EnvState` to `ParallelEnvState` in `parallel_env_state.py` |
| Credit weight 0.2 | ✅ Approved - reasonable relative to synergy bonus (0.1) |
| Symmetric tracking | ✅ Approved for Phase 3.2 (simplicity) |

### DRL Specialist Feedback (2025-12-26)

| Issue | Resolution |
|-------|------------|
| **Temporal discounting missing** | Added γ^delay discount factor |
| **Credit bypasses normalizer** | Changed: add credit to raw reward BEFORE normalization |
| Credit weight 0.2 | ✅ Approved for initial implementation |
| Symmetric tracking | ✅ Approved (directional tracking deferred) |

---

## Design Constraints

### PPO Invariants
- Rewards in the rollout buffer should not be modified retroactively
- Each transition's reward must be known at collection time
- Credit must flow through the standard reward → buffer → PPO update pipeline

### Architectural Constraints
- Scaffolds may be pruned before beneficiary fossilizes (no active seed to credit)
- Multiple scaffolds may boost the same beneficiary
- A seed can be both scaffold (to one) and beneficiary (from another)

### Performance Constraints
- No additional forward passes
- Minimal memory overhead per environment
- O(slots²) interaction tracking already exists; no increase in complexity

---

## Proposed Design: Environment-Level Hindsight Credit

### Core Idea

When a beneficiary fossilizes successfully, add hindsight credit to the **environment's next transition reward** (BEFORE normalization) rather than to a specific seed. This:
1. Avoids retroactive buffer modification
2. Works even when scaffolds are pruned
3. Creates a learning signal that associates scaffolding behavior with delayed success
4. Properly applies temporal discounting for delayed credit

### Data Structures

```python
# Add to ParallelEnvState (parallel_env_state.py)
@dataclass(slots=True)
class ParallelEnvState:
    # ... existing fields ...

    # Scaffold hindsight credit tracking
    # Maps scaffold_slot -> list of (boost_given, beneficiary_slot, epoch_of_boost)
    # Using list instead of set to track each boost interaction with its epoch
    scaffold_boost_ledger: dict[str, list[tuple[float, str, int]]] = field(default_factory=dict)

    # Pending hindsight credit to add to next transition (BEFORE normalization)
    pending_hindsight_credit: float = 0.0

    # Current epoch counter (incremented each step for temporal discount calculation)
    current_epoch: int = 0
```

### Algorithm

**Step 1: Track boost relationships during counterfactual validation**

When computing interaction terms (vectorized.py ~line 2172), also record which seeds boosted which:

```python
# After computing interaction term I_ij
if interaction > 0:
    current_epoch = env_state.current_epoch

    # Seed A boosted Seed B
    if slot_a not in env_state.scaffold_boost_ledger:
        env_state.scaffold_boost_ledger[slot_a] = []
    env_state.scaffold_boost_ledger[slot_a].append((interaction, slot_b, current_epoch))

    # Seed B boosted Seed A (symmetric)
    if slot_b not in env_state.scaffold_boost_ledger:
        env_state.scaffold_boost_ledger[slot_b] = []
    env_state.scaffold_boost_ledger[slot_b].append((interaction, slot_a, current_epoch))
```

**Step 2: Compute hindsight credit on fossilization**

When a seed fossilizes successfully (vectorized.py ~line 2649):

```python
if action_success:  # Fossilization succeeded
    env_state.seeds_fossilized += 1

    # Get the beneficiary's improvement
    beneficiary_improvement = seed_info.total_improvement if seed_info else 0.0

    if beneficiary_improvement > 0:
        current_epoch = env_state.current_epoch
        total_credit = 0.0
        scaffold_count = 0

        # Find all scaffolds that boosted this beneficiary
        for scaffold_slot, boosts in env_state.scaffold_boost_ledger.items():
            for boost_given, beneficiary_slot, epoch_of_boost in boosts:
                if beneficiary_slot == target_slot and boost_given > 0:
                    # Temporal discount: credit decays with distance
                    delay = current_epoch - epoch_of_boost
                    discount = gamma ** delay  # Use PPO's gamma (typically 0.99)

                    # Compute discounted hindsight credit
                    raw_credit = compute_scaffold_hindsight_credit(
                        boost_given=boost_given,
                        beneficiary_improvement=beneficiary_improvement,
                        credit_weight=0.2,
                    )
                    total_credit += raw_credit * discount
                    scaffold_count += 1

        # Cap total credit to prevent runaway values
        # Cap at 2x the synergy bonus maximum (0.1 * 2 = 0.2)
        MAX_HINDSIGHT_CREDIT = 0.2
        total_credit = min(total_credit, MAX_HINDSIGHT_CREDIT)

        env_state.pending_hindsight_credit += total_credit

        # Clear this beneficiary from all ledgers (it's now fossilized)
        for scaffold_slot in list(env_state.scaffold_boost_ledger.keys()):
            env_state.scaffold_boost_ledger[scaffold_slot] = [
                (b, ben, e) for (b, ben, e) in env_state.scaffold_boost_ledger[scaffold_slot]
                if ben != target_slot
            ]
            # Clean up empty entries
            if not env_state.scaffold_boost_ledger[scaffold_slot]:
                del env_state.scaffold_boost_ledger[scaffold_slot]
```

**Step 3: Apply pending credit to next transition (BEFORE normalization)**

When computing the reward for the next transition (vectorized.py ~line 2592):

```python
# Compute raw reward components
reward = compute_reward(**reward_args)

# Add any pending hindsight credit BEFORE normalization
# (DRL Specialist review: credit should go through normalizer for scale consistency)
if env_state.pending_hindsight_credit > 0:
    reward += env_state.pending_hindsight_credit

    # Emit telemetry for hindsight credit
    if use_telemetry:
        hub.emit(TelemetryEventType.REWARD_COMPUTED, {
            "env_idx": env_idx,
            "hindsight_credit": env_state.pending_hindsight_credit,
            # ... other fields
        })
    env_state.pending_hindsight_credit = 0.0

# THEN apply normalization (if reward normalizer enabled)
if reward_normalizer is not None:
    reward = reward_normalizer.normalize(reward)
```

**Step 4: Increment epoch counter**

At the end of each step:

```python
env_state.current_epoch += 1
```

**Step 5: Reset ledger on episode end**

When an episode terminates, clear the ledger:

```python
# In reset_episode_state() method of ParallelEnvState
def reset_episode_state(self, slots: list[str]) -> None:
    # ... existing resets ...
    self.scaffold_boost_ledger.clear()
    self.pending_hindsight_credit = 0.0
    self.current_epoch = 0
```

---

## Temporal Discounting Rationale

The DRL specialist identified that scaffolding interactions from many epochs ago should contribute less than recent ones. This mirrors how eligibility traces work in TD learning:

```
credit_discounted = raw_credit * (gamma ^ epochs_since_scaffold)
```

With γ = 0.99:
- Immediate fossilization: 100% credit
- 10 epochs later: 90% credit
- 50 epochs later: 60% credit
- 100 epochs later: 37% credit

This naturally prioritizes scaffolding that leads to quick fossilization while still rewarding longer-term support.

---

## Known Limitations

### Credit Attribution Dilution

The hindsight credit goes to the "next action after fossilization" rather than to the specific scaffolding action. This is a theoretical limitation:

- **Ideal:** Credit goes to the exact GERMINATE action that created the helpful scaffold
- **Actual:** Credit goes to whatever action is taken after the beneficiary fossilizes

**Why this is acceptable:**
1. Scaffolding is a multi-step behavior (germinate, nurture, maintain synergy)
2. The policy learns that scaffolding-like action sequences lead to delayed bonuses
3. Temporal discounting reduces credit for ancient scaffolding actions anyway

### Not PBRS-Compliant

This hindsight credit is NOT potential-based reward shaping (PBRS). True PBRS guarantees optimal policy invariance, but:

- PBRS requires: `F(s,a,s') = γΦ(s') - Φ(s)` for some potential function Φ
- Hindsight credit violates this: credit depends on future fossilization success

**Why this is acceptable:**
1. The credit is small (capped at 0.2)
2. It only triggers on successful fossilization events
3. Empirical testing will verify training stability

---

## Slot Transformer Compatibility

> **Note:** Per `docs/plans/2025-12-26-slot-transformer-architecture.md`, a Slot Transformer architecture is under consideration for future work. This section documents how Phase 3.2 interacts with that design.

### Why Hindsight Credit Remains Valuable

Even with Slot Transformer, scaffold hindsight credit provides value:

| Aspect | Slot Transformer | Hindsight Credit |
|--------|------------------|------------------|
| **What it captures** | Learns slot-slot interactions via self-attention | Rewards successful scaffolding outcomes |
| **When it acts** | Every forward pass (proactive) | On fossilization (reactive) |
| **Learning signal** | Implicit in attention weights | Explicit reward bonus |

**Key insight:** The transformer learns *how* to scaffold; the hindsight credit learns *when* scaffolding paid off.

### Redundancy Considerations

Phase 2 scaffolding features (from parent plan) include:
- `slot_interaction_matrix` - explicit slot-slot synergy values
- `scaffolding_pressure` - heuristic pressure toward scaffolding

With Slot Transformer, these become redundant (self-attention learns them automatically). However:

- **Phase 3.2 (this design)** is NOT redundant
- Hindsight credit rewards successful fossilization outcomes
- Transformer attention shows *what* interactions exist; credit rewards *successful* ones

### Migration Path

When migrating to Slot Transformer:

1. **Keep:** Scaffold hindsight credit (Phase 3.2)
2. **Remove:** Phase 2 observation features (`slot_interaction_matrix`, `scaffolding_pressure`)
3. **Add:** Attention weight telemetry to visualize learned scaffolding patterns

---

## Alternative Designs Considered

### Option A: Modify Past Buffer Rewards
- **Pros:** Exact credit to the scaffold's transitions
- **Cons:** Violates PPO invariants (rewards should be fixed at collection), complex buffer surgery

### Option B: Credit Only Living Scaffolds
- **Pros:** Simple, credits specific seed
- **Cons:** Misses the key case where scaffold is pruned before beneficiary succeeds

### Option C: Delayed Reward Injection (Chosen)
- **Pros:** PPO-compatible, works for pruned scaffolds, creates learning signal, applies proper temporal discount
- **Cons:** Credit goes to environment, not specific seed; may dilute signal

### Option D: Eligibility Traces
- **Pros:** Theoretically elegant, automatic credit decay
- **Cons:** Requires PPO modifications, hyperparameter tuning (λ for traces)

### Option E: Auxiliary Value Head
- **Pros:** Explicit scaffolding value prediction
- **Cons:** +8K parameters, requires designing prediction target, less valuable with Slot Transformer

---

## Edge Cases

| Case | Handling |
|------|----------|
| Scaffold pruned before beneficiary fossilizes | Credit still tracked in ledger, applied to env on beneficiary success |
| Multiple scaffolds boost same beneficiary | Each receives discounted credit proportional to its `boost_given` |
| Seed is both scaffold and beneficiary | Separate entries in ledger; can receive credit and give credit |
| Beneficiary fossilizes with negative improvement | No credit given (`beneficiary_improvement <= 0` guard) |
| Episode ends before beneficiary fossilizes | Ledger cleared on reset; no credit |
| Scaffold boosts multiple beneficiaries | Boost tracked per beneficiary; credit given for each successful fossilization |
| Very old scaffolding interaction | Heavily discounted by γ^delay (intentional) |
| Total credit exceeds cap | Capped at MAX_HINDSIGHT_CREDIT (0.2) |

---

## Telemetry

Add new telemetry fields to `REWARD_COMPUTED` event:

```python
{
    "hindsight_credit": float,      # Credit from scaffold hindsight (pre-cap)
    "hindsight_credit_capped": float,  # Credit after cap applied
    "scaffold_count": int,          # Number of scaffolds that contributed
    "avg_scaffold_delay": float,    # Average epochs since scaffolding interactions
}
```

---

## Testing Plan

### Unit Tests

```python
def test_hindsight_credit_computed_on_fossilization():
    """Credit is added to pending when beneficiary fossilizes."""

def test_pending_credit_added_to_next_reward():
    """Pending credit flows to next transition's reward."""

def test_no_credit_for_negative_improvement():
    """Beneficiary with negative improvement gives no credit."""

def test_ledger_cleared_on_episode_reset():
    """Scaffold ledger resets between episodes."""

def test_multiple_scaffolds_each_get_credit():
    """Multiple scaffolds boosting same beneficiary each receive credit."""

def test_temporal_discount_applied():
    """Older scaffolding interactions receive less credit."""

def test_credit_capped_at_maximum():
    """Total credit per fossilization is capped."""

def test_credit_goes_through_normalizer():
    """Credit is added before normalization, not after."""
```

### Integration Test

```python
def test_scaffold_hindsight_flow_e2e():
    """End-to-end: scaffold boosts beneficiary, beneficiary fossilizes, credit applied."""
    # 1. Create env with 2 slots
    # 2. Germinate seeds in both slots
    # 3. Simulate positive interaction (mock counterfactual)
    # 4. Wait N epochs (to test temporal discount)
    # 5. Fossilize beneficiary with positive improvement
    # 6. Verify pending_hindsight_credit > 0
    # 7. Verify credit is discounted by gamma^N
    # 8. Step again, verify reward includes credit (before normalization)
```

---

## Implementation Tasks

| Task | Files | Estimated Lines |
|------|-------|-----------------|
| 1. Add `scaffold_boost_ledger`, `pending_hindsight_credit`, `current_epoch` to ParallelEnvState | parallel_env_state.py | +8 |
| 2. Track boost relationships with epoch in counterfactual section | vectorized.py | +25 |
| 3. Compute temporally-discounted hindsight credit on fossilization | vectorized.py | +35 |
| 4. Apply pending credit to reward BEFORE normalization | vectorized.py | +12 |
| 5. Increment epoch counter each step | vectorized.py | +3 |
| 6. Reset ledger on episode end | parallel_env_state.py | +5 |
| 7. Add telemetry fields | vectorized.py, schemas.py | +10 |
| 8. Unit tests | test_scaffold_hindsight.py | +120 |
| 9. Integration test | test_vectorized_scaffolding.py | +60 |

**Total: ~280 lines**

---

## Success Criteria

- [ ] `compute_scaffold_hindsight_credit()` is called on beneficiary fossilization
- [ ] Pending credit flows to next transition's reward (BEFORE normalization)
- [ ] Temporal discount (γ^delay) applied correctly
- [ ] Credit capped at MAX_HINDSIGHT_CREDIT (0.2)
- [ ] Telemetry shows non-zero `hindsight_credit` when scaffolding occurs
- [ ] All tests pass
- [ ] Training completes without errors
