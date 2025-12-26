# Esper Domain Integration Map

> **Generated:** 2025-12-25
> **Status:** Current implementation (not intended design)
> **Scope:** Simic, Tamiyo, Kasmina, Tolaria cross-domain integrations

This document maps the actual integration points between Esper's four core training domains as they exist in the codebase. It was generated through systematic code analysis, not from design documents.

---

## Executive Summary

### Dependency Graph

```
                    ┌─────────────────┐
                    │    LEYLINE      │
                    │  (DNA/Genome)   │
                    │ Shared Contracts│
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │   TAMIYO    │   │   KASMINA   │   │   TOLARIA   │
    │   (Brain)   │   │(Stem Cells) │   │(Metabolism) │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └────────┬────────┴────────┬────────┘
                    │                 │
                    ▼                 ▼
              ┌─────────────────────────────┐
              │          SIMIC             │
              │       (Evolution)          │
              │   Orchestrates Everything  │
              └─────────────────────────────┘
```

### Key Findings

| Property | Status |
|----------|--------|
| Circular dependencies | **NONE** |
| Leyline as foundation | **YES** - all domains import from it |
| Simic as orchestrator | **YES** - imports from all peers |
| Peer cross-imports | **NONE** - Tamiyo/Kasmina/Tolaria don't import each other |
| Dead code | `tolaria/trainer.py` functions orphaned |

---

## Import Matrix

### Cross-Domain Import Counts

| From ↓ To → | Leyline | Tamiyo | Kasmina | Tolaria | Simic |
|-------------|---------|--------|---------|---------|-------|
| **Leyline** | — | 0 | 0 | 0 | 0 |
| **Tamiyo** | ✓ | — | 1 (TYPE_CHECKING) | 0 | 0 |
| **Kasmina** | ✓ | 0 | — | 0 | 1 (TYPE_CHECKING) |
| **Tolaria** | ✓ | 0 | 0 (hasattr) | — | 0 |
| **Simic** | ✓ | 8 modules | 1 module | 2 classes | — |

### Risk Assessment

| Integration | Risk Level | Rationale |
|-------------|------------|-----------|
| Simic → Tamiyo | LOW | Unidirectional, protocol-based |
| Simic → Kasmina | LOW | Single import + protocols |
| Simic → Tolaria | LOW | Utility consumer pattern |
| Tamiyo → Kasmina | NONE | TYPE_CHECKING only |
| All reverse directions | NONE | No imports exist |

---

## Detailed Integration Maps

### 1. Simic → Tamiyo (Brain Interface)

**Coupling Level:** TIGHT (Protocol-based)

Simic depends heavily on Tamiyo for policy decisions, feature extraction, and action masking. This is the primary neural control pathway.

#### Module-Level Imports

| File | Line | Import | Purpose |
|------|------|--------|---------|
| `simic/__init__.py` | 30 | `safe`, `TaskConfig` | Feature utilities re-export |
| `simic/__init__.py` | 31-37 | `MaskedCategorical`, `build_slot_states`, `compute_action_masks`, `compute_batch_masks` | Action masking infrastructure |
| `simic/agent/ppo.py` | 23 | `PolicyBundle` | Policy interface protocol |
| `simic/training/vectorized.py` | 97 | `build_slot_states`, `compute_action_masks` | Vectorized masking |
| `simic/training/vectorized.py` | 113-117 | `MULTISLOT_FEATURE_SIZE`, `get_feature_size`, `batch_obs_to_features` | Batch feature extraction |
| `simic/training/vectorized.py` | 119 | `create_policy` | Policy factory |
| `simic/training/dual_ab.py` | 58 | `get_feature_size` | A/B test dimensioning |

#### Lazy Imports (Inside Functions)

| File | Line | Import | Context |
|------|------|--------|---------|
| `simic/agent/ppo.py` | 80 | `obs_to_multislot_features` | `signals_to_features()` |
| `simic/agent/ppo.py` | 905 | `create_policy` | `load_from_checkpoint()` |
| `simic/training/vectorized.py` | 579 | `SignalTracker` | `train_ppo_vectorized()` |
| `simic/training/helpers.py` | 408 | `SignalTracker` | `_initialize_rl_environment()` |
| `simic/training/helpers.py` | 738-739 | `HeuristicTamiyo`, `HeuristicPolicyConfig` | `train_heuristic()` |

#### PPOAgent ↔ PolicyBundle Contract

| Line | Usage | Type | Data Flow |
|------|-------|------|-----------|
| 195 | `policy: PolicyBundle` | Constructor param | Composition |
| 234-252 | `self.policy.slot_config` | Property access | Configuration |
| 319-325 | `self.policy.network.*` | Direct access | Validation |
| 332 | `self.policy.slot_config.slot_ids` | Assertion | Ordering check |
| 356, 712, 790 | `self.policy.network` | Network extraction | Compile/checkpoint |
| 381 | `self.policy.network.parameters()` | Parameter access | Optimizer creation |
| 546 | `self.policy.evaluate_actions()` | Method call | PPO gradient computation |
| 733 | `clip_grad_norm_(self.policy.network.parameters())` | Gradient clipping | Training |

#### Data Flow

```
ROLLOUT COLLECTION:
  Observations → batch_obs_to_features() → Feature tensor
  Feature tensor → policy.get_action(features, masks) → ActionResult
  ActionResult → actions, log_probs, values, new_hidden

PPO UPDATE:
  Features + Actions → policy.evaluate_actions() → EvalResult
  EvalResult → log_prob, value, entropy (per head)
  Simic computes loss, calls backward(), clips gradients
```

---

### 2. Simic ↔ Kasmina (Stem Cell Control)

**Coupling Level:** TIGHT (Protocol-based via SlottedHostProtocol)

Simic controls seed lifecycle through method calls on MorphogeneticModel and SeedSlot objects. Uses protocols to avoid importing concrete types.

#### Direct Import

| File | Line | Import | Purpose |
|------|------|--------|---------|
| `simic/training/vectorized.py` | 1943 | `BlendCatalog` | Gate network creation for GATE algorithm |

#### Slot State Reading (Simic → Kasmina)

| Field | Lines | Semantic |
|-------|-------|----------|
| `.state` (None check) | 1931, 1939, 2021, 2582 | Presence of active seed |
| `.state.stage` | 215, 672 | Current lifecycle stage |
| `.state.alpha_algorithm` | 1932, 1940 | Alpha update algorithm |
| `.state.metrics` | 2022, 2103, 2176, 2199 | Metrics object for telemetry |
| `.state.seed_id` | 565-567, 598-602 | Unique seed identifier |
| `.state.alpha` | 1828, 2107 | Current alpha value |
| `.alpha` | 1954, 2742 | Slot-level alpha access |
| `.seed` | 82, 1393 | Active seed network module |
| `.is_active` | 2700 | Whether slot has active seed |

#### Metric Writing (Simic → Kasmina)

| Metric | Lines | Value | Semantic |
|--------|-------|-------|----------|
| `counterfactual_contribution` | 2034 | float | Causal seed impact |
| `contribution_velocity` | 2029-2030 | float | EMA of contribution delta |
| `interaction_sum` | 2074, 2082, 1814 | float | Shapley synergy sum |
| `boost_received` | 2075, 2084 | float | Strongest partner synergy |
| `upstream_alpha_sum` | 2113, 1816 | float | Position-aware credit |
| `downstream_alpha_sum` | 2114, 1817 | float | Position-aware credit |
| `seed_gradient_norm_ratio` | 2199 | float | G2 gate health |

#### Lifecycle Operations (Simic → Kasmina)

| Operation | Method | Lines | Data Passed |
|-----------|--------|-------|-------------|
| **Germinate** | `model.germinate_seed()` | 2489 | blueprint_id, seed_id, slot, blend_algorithm, tempo, alpha_algorithm, alpha_target |
| **Prune** | `slot.prune()` | 2524-2526 | reason, initiator |
| **Schedule Prune** | `slot.schedule_prune()` | 2528-2532 | steps, curve, initiator |
| **Set Alpha** | `slot.set_alpha_target()` | 2540-2552 | alpha_target, steps, curve, alpha_algorithm, initiator |
| **Advance** | `slot.advance_stage()` | 2558 | (none - returns GateResult) |
| **Epoch Step** | `slot.step_epoch()` | 596, 2678 | (none - returns auto_prune bool) |
| **Force Alpha** | `slot.force_alpha(value)` | counterfactual | Context manager for validation |

#### Parameter Access

| Method | Lines | Returns | Use Case |
|--------|-------|---------|----------|
| `model.get_seed_parameters(slot_id)` | 1224, 1387 | Iterator[Parameter] | Per-slot optimizer |
| `model.get_seed_parameters()` | 665, 683 | Iterator[Parameter] | All seed params |
| `model.get_host_parameters()` | 1086, 1217, 452 | Iterator[Parameter] | Host optimizer, telemetry |
| `model.has_active_seed_in_slot(slot_id)` | 210, 1255, 2515... | bool | Guard actions |
| `model.active_seed_params` | 2413, 617 | int | Reward scaling |

---

### 3. Simic ↔ Tolaria (Training Infrastructure)

**Coupling Level:** LOOSE (Utility consumer)

Simic uses Tolaria's model factory and governor watchdog. Tolaria provides infrastructure; Simic orchestrates.

#### Imports

| File | Line | Import | Purpose |
|------|------|--------|---------|
| `simic/training/vectorized.py` | 130 | `TolariaGovernor` | Training watchdog |
| `simic/training/vectorized.py` | 578, 627, 1066 | `create_model` (via runtime) | Model instantiation |
| `simic/training/helpers.py` | 407, 424 | `create_model` | Heuristic training |

#### Governor Integration

| Operation | Line | Purpose |
|-----------|------|---------|
| Instantiation | 1124-1130 | Create per-environment watchdog |
| Snapshot | 2158 | Save Last Known Good state (every 5 epochs) |
| Vital Signs | 2161 | Check for NaN/Inf/explosion |
| Rollback | 2375 | Emergency recovery on panic |

#### Governor Configuration

```python
governor = TolariaGovernor(
    model=model,
    sensitivity=DEFAULT_GOVERNOR_SENSITIVITY,      # 6.0 sigma
    absolute_threshold=DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD,  # 10.0
    death_penalty=DEFAULT_GOVERNOR_DEATH_PENALTY,  # -1.0
    history_window=DEFAULT_GOVERNOR_HISTORY_WINDOW,  # 20 epochs
    min_panics_before_rollback=DEFAULT_MIN_PANICS_BEFORE_ROLLBACK,  # 3
    random_guess_loss=random_guess_loss,
)
```

#### Dead Code Alert

The following Tolaria functions are **never called** by Simic:
- `train_epoch_normal()`
- `train_epoch_incubator_mode()`
- `train_epoch_blended()`
- `validate_and_get_metrics()`
- `validate_with_attribution()`

Simic reimplements training inline via `process_train_batch()` in `vectorized.py`.

---

### 4. Tamiyo ↔ Kasmina (Brain → Stem Cells)

**Coupling Level:** LOOSE (Observation only)

Tamiyo observes Kasmina state through SignalTracker but does not control directly. Control flows through Simic.

#### Imports

| File | Line | Import | Type |
|------|------|--------|------|
| `tamiyo/tracker.py` | 24 | `SeedState` | TYPE_CHECKING |
| `tamiyo/heuristic.py` | 38 | `SeedState` | TYPE_CHECKING |

#### SignalTracker Observations

| SeedState Field | Lines | Purpose |
|-----------------|-------|---------|
| `seed.stage` | 199 | Determine lifecycle stage |
| `seed.alpha` | 200 | Current blend weight |
| `seed.metrics.counterfactual_contribution` | 202-203 | Select summary seed |
| `seed.metrics.improvement_since_stage_start` | 211 | Performance delta |
| `seed.seed_id` | 218 | Identify tracked seed |

#### Summary Seed Selection (Deterministic)

```python
# Multi-key sort priority:
1. Highest stage value (FOSSILIZED > HOLDING > BLENDING > TRAINING > GERMINATED)
2. Highest alpha (most integrated wins ties)
3. Most negative counterfactual (prefer helpful seeds)
4. Seed ID (determinism)
```

#### Heuristic Decision Tree

| Stage | Condition | Decision |
|-------|-----------|----------|
| GERMINATED | Always | ADVANCE |
| TRAINING | Failing + min epochs | PRUNE |
| TRAINING | Else | ADVANCE |
| BLENDING | Failing | PRUNE |
| BLENDING | Full blend complete | ADVANCE |
| HOLDING | Ransomware pattern | PRUNE |
| HOLDING | Improvement > threshold | FOSSILIZE |
| HOLDING | Else | PRUNE |

---

### 5. Tolaria ↔ Kasmina (Metabolism → Stem Cells)

**Coupling Level:** MODERATE (Governor fail-safe)

Tolaria's governor monitors training and can prune seeds on catastrophic failure.

#### Governor Access to Kasmina

| File | Line | Access | Purpose |
|------|------|--------|---------|
| `governor.py` | 109-111 | `hasattr(model, 'seed_slots')` | Feature detection |
| `governor.py` | 111-117 | `model.seed_slots.items()` | Filter experimental seeds |
| `governor.py` | 267-271 | `slot.prune()` | Clear live seeds on rollback |

#### Snapshot Filtering

```python
# Governor snapshot excludes non-fossilized seeds
for slot_id, slot in self.model.seed_slots.items():
    if slot.state is not None and slot.state.stage != SeedStage.FOSSILIZED:
        experimental_prefixes.append(f"seed_slots.{slot_id}.seed.")

filtered_state = {
    k: v for k, v in full_state.items()
    if not any(k.startswith(prefix) for prefix in experimental_prefixes)
}
```

#### Rollback Behavior

```python
# On catastrophic failure:
1. Prune ALL live (non-fossilized) seeds
2. Restore host + fossilized seeds from snapshot
3. Reset optimizer momentum (prevent re-crash)
4. Emit telemetry event
```

---

## Data Flow Diagrams

### Training Epoch Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING EPOCH FLOW                          │
└─────────────────────────────────────────────────────────────────┘

1. OBSERVE
   Simic reads → Kasmina slot.state (stage, alpha, metrics)
                → Tolaria governor (vital signs)

2. AGGREGATE
   Simic calls → Tamiyo SignalTracker.update(active_seeds)
              → Returns TrainingSignals

3. DECIDE
   Simic calls → Tamiyo policy.get_action(features, masks)
              → Returns ActionResult (action, log_prob, value)

4. EXECUTE
   Simic calls → Kasmina slot.germinate/prune/advance/set_alpha
              → Tolaria governor.check_vital_signs(loss)

5. LEARN (PPO)
   Simic calls → Tamiyo policy.evaluate_actions(features, actions)
              → Computes loss, calls backward()
              → Clips gradients on policy.network.parameters()

6. FAIL-SAFE
   If panic  → Tolaria governor.execute_rollback()
            → Kasmina slots.prune() (all live seeds)
```

### Forward Pass with Seeds

```
┌──────────────────────────────────────────────────────────────────┐
│  FORWARD PASS (MorphogeneticModel)                               │
└──────────────────────────────────────────────────────────────────┘

Input
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Host.forward_to_segment(slot_0)                                 │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ SeedSlot[0].forward(host_features)                              │
│   ├─ DORMANT: return host_features (no-op)                      │
│   ├─ TRAINING: host + (seed - seed.detach()) [STE]              │
│   └─ BLENDING+: lerp(host, seed, alpha)                         │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Host.forward_to_segment(slot_1)                                 │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
  ... (repeat for each active slot)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Host.forward_from_segment(last_slot) → Classifier               │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
Output
```

### Gradient Flow with Isolation

```
┌──────────────────────────────────────────────────────────────────┐
│  BACKWARD PASS (Gradient Attribution)                            │
└──────────────────────────────────────────────────────────────────┘

loss.backward()
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ BLENDING+ Stage (alpha > 0):                                    │
│   ∂loss/∂host = (1 - α) × ∂loss/∂output                         │
│   ∂loss/∂seed = α × ∂loss/∂output                               │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ TRAINING Stage (alpha = 0, STE):                                │
│   ∂loss/∂host = ∂loss/∂output (normal)                          │
│   ∂loss/∂seed = ∂loss/∂output (via STE, detached path blocked)  │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Gradient Collection:                                            │
│   host_grad_norm = ||∇_host||                                   │
│   seed_grad_norm = ||∇_seed||                                   │
│   ratio = seed_grad_norm / (host_grad_norm + ε)  [For G2 gate]  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quality Gates Integration

Simic monitors quality gates implemented by Kasmina:

| Gate | Stage Transition | Check | Simic Role |
|------|------------------|-------|------------|
| G0 | → GERMINATED | Shape validation | N/A (automatic) |
| G1 | → TRAINING | Always passes | N/A |
| G2 | → BLENDING | Gradient ratio health | Monitor via `seed_gradient_norm_ratio` |
| G3 | → HOLDING | Alpha completion | Policy ensures alpha converges |
| G4/G5 | → FOSSILIZED | Contribution threshold | Policy only advances if `gate_result.passed` |

---

## Counterfactual Validation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  COUNTERFACTUAL PHASE (Causal Attribution)                       │
└──────────────────────────────────────────────────────────────────┘

For each active seed:
  │
  ├─► SOLO VALIDATION
  │     with slot.force_alpha(0.0):
  │       acc_without_seed = validate(model, batch)
  │     contribution = acc_with_seed - acc_without_seed
  │     slot.state.metrics.counterfactual_contribution = contribution
  │
  └─► PAIRWISE VALIDATION (Shapley)
        for other_seed in active_seeds:
          with slot.force_alpha(0.0), other.force_alpha(0.0):
            acc_both_off = validate()
          interaction = acc_both_on - acc_one_off - acc_other_off + acc_both_off
          slot.state.metrics.interaction_sum += interaction
```

---

## Recommendations

### Architectural Health

1. **Clean layering achieved** - No circular dependencies
2. **Protocol isolation working** - Simic uses contracts, not concrete types
3. **Leyline is true foundation** - All domains depend on it

### Technical Debt

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Dead trainer functions | `tolaria/trainer.py` | Archive or remove |
| TYPE_CHECKING import | `kasmina/slot.py:79` → `esper.simic.features.TaskConfig` | Module doesn't exist; remove or create |
| hasattr guards | `tolaria/governor.py` | Already authorized; document in code |

### Future Considerations

1. **Tolaria trainer revival** - If generic training utilities needed, restore and document
2. **TaskConfig type** - Decide whether to create `simic.features` module or remove type hint
3. **Protocol documentation** - Add explicit protocol files with contracts

---

## Appendix: File Manifest

### Simic Files with Cross-Domain Imports

| File | Imports From |
|------|--------------|
| `simic/__init__.py` | Tamiyo (action_masks, features) |
| `simic/agent/ppo.py` | Tamiyo (protocol, features, factory) |
| `simic/training/vectorized.py` | Tamiyo (all), Kasmina (BlendCatalog), Tolaria (Governor) |
| `simic/training/helpers.py` | Tamiyo (SignalTracker, HeuristicTamiyo) |
| `simic/training/dual_ab.py` | Tamiyo (features) |

### Tamiyo Files with Cross-Domain Imports

| File | Imports From |
|------|--------------|
| `tamiyo/tracker.py` | Kasmina (SeedState, TYPE_CHECKING) |
| `tamiyo/heuristic.py` | Kasmina (SeedState, TYPE_CHECKING) |

### Kasmina Files with Cross-Domain Imports

| File | Imports From |
|------|--------------|
| `kasmina/slot.py` | Simic (TaskConfig, TYPE_CHECKING - module doesn't exist) |

### Tolaria Files with Cross-Domain Imports

| File | Imports From |
|------|--------------|
| `tolaria/governor.py` | Kasmina (via hasattr on model.seed_slots) |

---

*This document was generated through systematic codebase analysis on 2025-12-25.*
