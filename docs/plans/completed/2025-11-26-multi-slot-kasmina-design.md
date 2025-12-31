# Multi-Slot Kasmina Design

**Date**: 2025-11-26
**Status**: Brainstormed, Ready for Implementation
**Goal**: Scale from single seed slot to multiple injection points with richer telemetry

## Context

Policy Tamiyo v1 (currently training overnight) learns when to intervene with a single seed slot on CIFAR-10. This design extends to:
- Multiple seed slots at different network depths
- Harder problem (CIFAR-100) that justifies multiple interventions
- Richer observation space with per-slot telemetry
- Factored action space (action type + target slot)

## Design Decisions

### 1. Slot Positions: Fixed Depths (Option A)

Three slots at fixed positions in the network:
- **Slot 0**: After first conv block (early features - edges, textures)
- **Slot 1**: After second conv block (mid features - shapes, parts)
- **Slot 2**: Before classifier (late features - semantic concepts)

Rationale: Simple, interpretable, lets us test "does depth matter for intervention?" Future work can generalize to configurable positions (Option B) once fixed positions feel limiting.

Future vision: "TamiyoExtremeEdition" with dense slots across a sparse scaffold, Karn building custom blocks, Tamiyo placing them precisely.

### 2. Slot Telemetry: Per-Slot Metrics (Option A)

Each slot reports its own telemetry - Tamiyo builds a "mental map" of the network:
- Slots as distributed sensors
- Global metrics as vital signs
- Rich information for pattern detection humans can't see

### 3. Slot Lifecycles: Independent (Option A)

Each slot runs its own DORMANT→GERMINATED→TRAINING→BLENDING→FOSSILIZED lifecycle independently. No coordination constraints - Tamiyo learns coordination through outcomes.

If concurrent seeds cause instability → negative reward → learns to stagger.

### 4. Action Space: Factored (Option B)

Two-headed output:
- Head 1: Action type (WAIT, GERMINATE, ADVANCE, CULL)
- Head 2: Target slot (SLOT_0, SLOT_1, SLOT_2)

Scales cleanly to N slots. Natural fit for future extensions.

### 5. Invalid Actions: Soft Consequences (Purist)

No hard masking - policy can attempt any action:
- Invalid actions treated as WAIT (no-op)
- No explicit penalty (V1 purist approach)
- Policy learns organically that invalid = wasted turn

Future experiments:
- V2: Small penalty for invalid actions
- V3: Log attempted invalid actions for research ("what did she try?")

### 6. Observation Space: Concatenated Vector (Option A)

Simple flat vector, structured for future swap to attention-based:

```
[global_metrics..., slot_0_metrics..., slot_1_metrics..., slot_2_metrics...]
```

## Architecture

### Data Flow

```
Input → Conv1 → [Slot 0] → Conv2 → [Slot 1] → Conv3 → [Slot 2] → Classifier → Output
              ↓              ↓               ↓
         Telemetry      Telemetry       Telemetry
              ↓              ↓               ↓
              └──────────────┴───────────────┘
                              ↓
                      Policy Tamiyo
                              ↓
                    Action (type, slot_id)
```

### Observation Space (~45 features)

**Global metrics (~15 features):**
- epoch, global_step
- train_loss, val_loss, val_accuracy
- plateau_epochs, best_val_accuracy, best_val_loss
- loss_history_5 (5 floats)
- accuracy_history_5 (5 floats)

**Per-slot metrics (~10 features × 3 slots = 30):**
- gradient_norm - learning pressure at this depth
- activation_sparsity - health indicator (dying neurons)
- has_seed - boolean, is slot occupied
- seed_stage - int encoding (0=empty, 1=GERMINATED, etc.)
- seed_alpha - current blending weight
- epochs_in_stage - time in current stage
- seed_improvement - accuracy delta since germination
- (3 reserved for future telemetry)

### Policy Network

```
obs(45) → Linear(128) → ReLU → Linear(64) → ReLU
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              Linear(4)                       Linear(3)
              action_type                     target_slot
              (WAIT/GERM/ADV/CULL)           (SLOT_0/1/2)
```

### Action Execution

```python
def apply_action(action_type, slot_id, slots):
    slot = slots[slot_id]

    if action_type == WAIT:
        return  # Always valid

    if action_type == GERMINATE:
        if slot.is_empty:
            slot.germinate(select_blueprint())
        # else: no-op, wasted turn

    if action_type == ADVANCE:
        if slot.has_active_seed:
            slot.advance_stage()
        # else: no-op, wasted turn

    if action_type == CULL:
        if slot.has_active_seed:
            slot.cull()
        # else: no-op, wasted turn
```

## Training Setup

### Dataset: CIFAR-100
- 100 classes (10× harder than CIFAR-10)
- Same image size (32×32), same data loading infrastructure
- Models plateau harder and earlier
- Creates genuine need for intervention at multiple depths

### Base Model: Deliberately Weak
- Smaller conv blocks, fewer channels than current
- Target baseline accuracy: ~40-50% without seeds
- Struggles enough that seeds can genuinely help
- Multiple depths plateau at different times

### Episode Structure
- Same as current: observe → act → outcome → reward
- Action now includes (type, slot_id) tuple
- Reward based on accuracy delta (unchanged)

### Multi-Slot Coordination
Learned through outcomes, not rules:
- Germinating 3 seeds simultaneously → instability → negative reward
- Staggered intervention → stable improvement → positive reward
- Policy discovers coordination patterns organically

## Component Roster (MTG Theme)

| Component | Role |
|-----------|------|
| **Kasmina** | Seed infrastructure - slots, seeds, lattice, surgical execution |
| **Tamiyo** | Strategic controller - intervention timing and targeting |
| **Simic** | Policy learning infrastructure - data collection, training |
| **Leyline** | Shared contracts - data definitions, interfaces |
| **Karn** | (Future) Autonomous blueprint builder |
| **Emrakul** | (Future) Shadow controller for stable periods - exploratory pruning |

### Emrakul Concept (Future Work)

Risk-adjusted dual controller:
- When instability high → Tamiyo budget 100%, conservative intervention
- When stability high → Budget drips to Emrakul
- Emrakul explores: "What can we *remove* without breaking anything?"
- Chaos engineering for neural networks

## Implementation Roadmap

### Phase 1: Multi-Slot Infrastructure
- [ ] Extend HostCNN with 3 injection points
- [ ] Create SlotManager to coordinate multiple SeedSlots
- [ ] Add per-slot telemetry collection
- [ ] Update MorphogeneticModel for multi-slot forward pass

### Phase 2: Extended Observation Space
- [ ] Define MultiSlotSnapshot dataclass
- [ ] Implement to_vector() for concatenated observation
- [ ] Add gradient_norm and activation_sparsity computation
- [ ] Update TrainingSnapshot → MultiSlotSnapshot in collectors

### Phase 3: Factored Policy Network
- [ ] Implement two-headed PolicyNetwork
- [ ] Add action masking (soft - for logging, not blocking)
- [ ] Update training loop for factored actions
- [ ] Handle invalid actions as no-ops

### Phase 4: CIFAR-100 + Weak Base Model
- [ ] Switch data loaders to CIFAR-100
- [ ] Design weak base architecture
- [ ] Validate baseline accuracy (~40-50%)
- [ ] Run overnight episode generation

### Phase 5: Validation
- [ ] Compare multi-slot vs single-slot on CIFAR-100
- [ ] Analyze which slots get used most
- [ ] Study coordination patterns learned
- [ ] Document findings

## Success Criteria

1. Policy learns to use different slots for different situations
2. Multi-slot outperforms single-slot on CIFAR-100
3. Policy learns not to over-intervene (coordination without rules)
4. Clear telemetry shows which depths benefit most from intervention

## Future Extensions

Three dimensions of scaling, roughly in order of priority:

### Dimension 1: Size (more slots, richer telemetry)

- **Variable slot positions**: Let Tamiyo choose where to inject, not just when
- **Dense slot scaffold**: TamiyoExtremeEdition with slots at every layer
- **Dead neuron cartography**: Map and target dying regions for extraction
- **GNN Architecture**: Full heterogeneous graph processing (see archived 03.1-tamiyo-gnn-architecture.md)

### Dimension 2: Complexity (harder problems, richer blueprints)

- **Karn**: Autonomous blueprint generation based on Tamiyo's requests
- **Blueprint diversity**: Multiple seed architectures, Tamiyo chooses which to germinate
- **Harder datasets**: ImageNet, domain-specific problems
- **Larger base models**: ResNet, ViT - seeds as adapters

### Dimension 3: Control (finer-grained intervention)

- **Blending control**: Tamiyo influences alpha schedule, not just stage transitions
  - `BLEND_FASTER` / `BLEND_SLOWER` actions
  - Direct `SET_ALPHA` for precise control
  - Adaptive schedules: speed up when stable, slow down when volatile
  - Channel-wise or attention-weighted blending
- **Emrakul**: Exploratory pruning during stable periods
- **Learning rate control**: Tamiyo adjusts seed training intensity
- **Gradient gating**: Tamiyo controls which gradients flow where

## References

- Current: `docs/plans/2025-01-25-policy-tamiyo-design.md`
- Archive: `_archive/docs/design/detailed_design/03.1-tamiyo-gnn-architecture.md`
- Archive: `_archive/docs/design/detailed_design/03-tamiyo-unified-design.md`
