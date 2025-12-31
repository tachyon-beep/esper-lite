# Tamiyo Lever Exploration Report

**Date:** 2025-12-19
**Status:** Analysis Complete
**Author:** Claude (PyTorch Engineering Specialist)

## Executive Summary

This report identifies potential levers that could be exposed to Tamiyo, the RL agent controlling seed lifecycle in Esper. The analysis focuses on what's technically feasible from a PyTorch perspective, what would require significant implementation effort, and what telemetry would be needed for the agent to learn effective use of each lever.

The philosophy driving this exploration is: **we don't prescribe what the agent should learn - we give it levers and telemetry, then observe what emerges.**

---

## Current State: Fixed vs. Controllable

### Currently Fixed (Potential Levers)

| Parameter | Current Value | Location | Complexity to Expose |
|-----------|--------------|----------|---------------------|
| Blending duration | 5 epochs (`DEFAULT_BLENDING_TOTAL_STEPS`) | leyline/__init__.py | LOW |
| Blending algorithm | Selected at GERMINATE | factored_actions.BlendAction | Already controllable |
| Sigmoid steepness | 1.0 fixed | blending.py:SigmoidBlend | LOW |
| Gradient isolation | Topology-dependent binary | slot.py:_on_enter_stage | MEDIUM |
| G2 gradient threshold | 0.10 fixed | leyline/__init__.py | MEDIUM |
| Min blending epochs | 3 fixed | leyline/__init__.py | LOW |
| Train-to-blend fraction | 0.1 fixed | features.TaskConfig | LOW |

### Currently Controllable by Tamiyo

| Control | Mechanism | Notes |
|---------|-----------|-------|
| Slot selection | FactoredAction.slot_idx | Multi-slot targeting |
| Blueprint selection | FactoredAction.blueprint | 8 CNN, 8 transformer options |
| Blend algorithm | FactoredAction.blend | Linear, Sigmoid, Gated |
| Lifecycle operation | FactoredAction.op | WAIT, GERMINATE, CULL, FOSSILIZE |

---

## Category 1: Blending Mechanism Levers

### 1.1 Blending Duration (Tempo Lever) - RECOMMENDED

**Current State:**
```python
# leyline/__init__.py
DEFAULT_BLENDING_TOTAL_STEPS = 5

# slot.py:_on_enter_stage()
total_steps = DEFAULT_BLENDING_TOTAL_STEPS
if self.task_config is not None:
    configured_steps = self.task_config.blending_steps
```

**Proposal:**
Add a `tempo` head to the factored action space, allowing Tamiyo to choose blending duration at GERMINATE time.

```python
class TempoAction(IntEnum):
    """Blending tempo selection."""
    FAST = 0      # 3 epochs - rapid integration
    STANDARD = 1  # 5 epochs - default
    SLOW = 2      # 8 epochs - cautious
    GLACIAL = 3   # 12 epochs - very conservative
```

**Implementation Complexity:** LOW
- Modify `FactoredAction` to include tempo
- Pass tempo to `SeedSlot.germinate()` or store for `start_blending()`
- Update action masking (tempo only valid with GERMINATE)

**torch.compile Impact:** NONE
- Blending duration affects schedule stepping, not graph structure
- No graph breaks, no specialization changes

**Telemetry Needed:**
- `blending_progress`: step/total_steps ratio (already tracked as alpha)
- `blending_velocity`: d(alpha)/d(epoch) - how fast blend is progressing
- `contribution_at_alpha_milestones`: counterfactual at 25%, 50%, 75%, 100%

**Learning Signal:**
- Agent discovers optimal tempo per blueprint/slot combination
- Fast tempo = higher compute rent but quicker signal
- Slow tempo = longer evaluation window, better stability assessment

---

### 1.2 Sigmoid Steepness - OPTIONAL

**Current State:**
```python
class SigmoidBlend(BlendAlgorithm):
    def __init__(self, total_steps: int = 10, steepness: float = 1.0):
        self.steepness = steepness

    def _compute_alpha(self, step: int) -> float:
        x = (step / self.total_steps - 0.5) * 12 * self.steepness
        return 1.0 / (1.0 + math.exp(-x))
```

**Proposal:**
Expose steepness as a continuous lever or discretized options.

```python
class SteepnessAction(IntEnum):
    GENTLE = 0   # 0.5 - very gradual transition
    NORMAL = 1   # 1.0 - standard S-curve
    SHARP = 2    # 2.0 - fast transition in middle
```

**Implementation Complexity:** LOW
- Only relevant when BlendAction.SIGMOID selected
- Requires conditional action masking

**torch.compile Impact:** NONE
- Pure Python math in schedule, not in forward path

**Telemetry Needed:**
- `alpha_variance`: variance of alpha across batch (for gated blend comparison)
- `early_contribution_signal`: counterfactual at alpha=0.3

---

### 1.3 GatedBlend Architecture Parameters - ADVANCED

**Current State:**
```python
class GatedBlend(BlendAlgorithm):
    def __init__(self, channels: int, topology: str = "cnn", total_steps: int = 10):
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // 4),  # Fixed reduction ratio
            nn.ReLU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid(),
        )
```

**Potential Levers:**
- Gate reduction ratio (currently channels // 4)
- Gate depth (currently 2 layers)
- Gate activation (currently ReLU)

**Implementation Complexity:** MEDIUM
- Requires dynamic gate construction
- More parameters = more variance in training

**torch.compile Impact:** LOW
- Different gate architectures cause separate specializations
- Acceptable if gate changes are infrequent (only at GERMINATE)

**Recommendation:** DEFER - Too much complexity for limited signal. The gate learns input-dependent alpha, which is already a form of learned blending strategy.

---

## Category 2: Gradient Isolation Levers

### 2.1 Isolation Mode Selection - RECOMMENDED

**Current State:**
```python
# slot.py:_on_enter_stage()
elif new_stage == SeedStage.BLENDING and old_stage == SeedStage.TRAINING:
    # Topology-aware gradient isolation:
    # - CNNs: keep isolation (host learns from loss, not seed feedback)
    # - Transformers: allow co-adaptation (host receives seed gradients)
    topology = self.task_config.topology if self.task_config else "cnn"
    self.isolate_gradients = (topology == "cnn")
```

**Proposal:**
Make isolation mode a choice at GERMINATE or BLENDING transition.

```python
class IsolationAction(IntEnum):
    DEFAULT = 0      # Topology-dependent (current behavior)
    ISOLATED = 1     # Always detach seed input (pure additive)
    COADAPT = 2      # Allow gradients to host (co-adaptation)
```

**Rationale:**
- Some CNN seeds might benefit from co-adaptation
- Some transformer seeds might be too unstable for it
- Let Tamiyo learn which blueprint/slot combinations benefit

**Implementation Complexity:** MEDIUM
- Modify `_on_enter_stage` to check action
- Add isolation mode to factored action space

**torch.compile Impact:** LOW
- `isolate_gradients` is already a branch in `forward()`
- Dynamo already specializes on this boolean

**Telemetry Needed:**
- `host_gradient_norm_via_seed`: gradient flow through seed path
- `host_weight_change_rate`: how much host params change during blend
- `coadaptation_benefit`: (acc with coadapt) - (acc without)

---

### 2.2 Continuous Isolation Strength - ADVANCED

**Current State:** Binary detach/no-detach

**Proposal:**
Instead of `x.detach()`, use `x.detach() + alpha_iso * (x - x.detach())` where `alpha_iso` is a learnable parameter or discrete choice.

**Implementation Complexity:** HIGH
- Requires custom autograd.Function
- Changes gradient scaling, not just flow

**torch.compile Impact:** MEDIUM
- Custom autograd.Function needs careful handling
- May prevent some fusion opportunities

**Recommendation:** DEFER - The binary isolation is already well-understood. Continuous isolation adds complexity without clear benefit over the binary + tempo combination.

---

## Category 3: Stage Timing Levers

### 3.1 Train-to-Blend Dwell Time - RECOMMENDED

**Current State:**
```python
# features.py
@dataclass
class TaskConfig:
    train_to_blend_fraction: float = 0.1  # Fraction of max_epochs to stay in TRAINING

# slot.py:step_epoch()
if stage == SeedStage.TRAINING:
    dwell_epochs = 1
    if self.task_config:
        dwell_epochs = max(1, int(self.task_config.max_epochs * self.task_config.train_to_blend_fraction))
```

**Proposal:**
Let Tamiyo choose dwell time at GERMINATE.

```python
class DwellAction(IntEnum):
    MINIMAL = 0   # 1 epoch - trust early signal
    SHORT = 1     # 2 epochs
    STANDARD = 2  # 3 epochs (10% of 25)
    EXTENDED = 3  # 5 epochs - thorough evaluation
```

**Implementation Complexity:** LOW
- Store dwell preference in SeedState
- Check in step_epoch()

**torch.compile Impact:** NONE
- Pure lifecycle logic, not in forward path

**Telemetry Needed:**
- `training_gradient_stability`: variance of seed gradient norm across epochs
- `g2_gate_headroom`: how far above threshold is gradient ratio
- `early_accuracy_signal`: improvement in first N epochs

---

### 3.2 Probation Timeout - OPTIONAL

**Current State:**
```python
# leyline/__init__.py
DEFAULT_MAX_PROBATION_EPOCHS = 5

# slot.py:step_epoch()
if self.state.metrics.epochs_in_current_stage >= max_probation_epochs:
    self.cull(reason="probation_timeout")
```

**Proposal:**
Let Tamiyo set probation patience at BLENDING completion.

**Rationale:**
- Some seeds may need longer to prove value
- Agent learns which blueprints need more evaluation time

**Implementation Complexity:** LOW
- Add to action space at PROBATIONARY transition

**Recommendation:** OPTIONAL - The current escalating WAIT penalty already creates pressure. This lever adds nuance but may be redundant.

---

## Category 4: Gate Threshold Levers

### 4.1 Per-Seed G2 Threshold - ADVANCED

**Current State:**
```python
# leyline/__init__.py
DEFAULT_GRADIENT_RATIO_THRESHOLD = 0.10

# slot.py:QualityGates._check_g2()
if state.metrics.seed_gradient_norm_ratio >= self.min_seed_gradient_ratio:
    checks_passed.append(f"seed_gradient_active_{state.metrics.seed_gradient_norm_ratio:.2f}")
```

**Proposal:**
Let Tamiyo set a per-seed G2 threshold at GERMINATE.

**Rationale:**
- Different blueprints have different gradient characteristics
- Large MLP seeds may have lower relative gradient ratio
- Agent learns optimal thresholds per blueprint

**Implementation Complexity:** MEDIUM
- Per-seed threshold storage
- Modified gate check logic

**Recommendation:** DEFER - Better to keep gates as safety rails. Let Tamiyo control timing (dwell, tempo) rather than bypassing quality gates.

---

## Category 5: Seed Architecture Levers

### 5.1 Blueprint Hyperparameters - FUTURE

**Potential Levers:**
- Attention reduction ratio (currently fixed at 4)
- LoRA rank (currently 8 or 32)
- Bottleneck reduction (currently 4)
- MLP expansion factor (currently 2 or 4)

**Current State:** All fixed at blueprint definition time.

**Proposal:**
Add a "variant" action head for blueprint-specific hyperparameters.

```python
class VariantAction(IntEnum):
    SMALL = 0   # Minimal version of blueprint
    MEDIUM = 1  # Standard version
    LARGE = 2   # Expanded version
```

**Implementation Complexity:** HIGH
- Requires registry refactoring
- More blueprints = larger action space

**Recommendation:** DEFER - The current blueprint diversity (8 CNN, 8 transformer) provides sufficient exploration. Adding variants would explode action space.

---

## Category 6: Learning Rate Levers

### 6.1 Seed Learning Rate Multiplier - RECOMMENDED

**Current State:**
```python
# leyline/__init__.py
DEFAULT_SEED_LR = 0.01  # Same as host
```

**Proposal:**
Let Tamiyo choose seed LR multiplier at GERMINATE.

```python
class LRAction(IntEnum):
    CAUTIOUS = 0   # 0.5x host LR
    STANDARD = 1   # 1.0x host LR
    AGGRESSIVE = 2 # 2.0x host LR
```

**Rationale:**
- Small seeds (norm, lora) may benefit from higher LR
- Large seeds (mlp) may need lower LR for stability
- Agent learns optimal LR per blueprint/stage

**Implementation Complexity:** MEDIUM
- Requires per-slot optimizer param groups
- Already possible with existing PyTorch optimizer infrastructure

**torch.compile Impact:** NONE
- LR is optimizer state, not graph structure

**Telemetry Needed:**
- `seed_loss_gradient`: |d(loss)/d(seed_params)|
- `seed_weight_magnitude`: L2 norm of seed params
- `lr_stability_ratio`: epochs without gradient explosion / total epochs

---

## Telemetry Summary: What Tamiyo Sees Today

### Current Observation Space (from features.py)

**Base Features (23 dims):**
- Timing: epoch, global_step
- Losses: train_loss, val_loss, loss_delta
- Accuracies: train_accuracy, val_accuracy, accuracy_delta
- Trends: plateau_epochs, best_val_accuracy, best_val_loss
- History: loss_history_5, accuracy_history_5
- Resources: total_params, seed_utilization

**Per-Slot Features (17 dims each):**
- is_active, stage, alpha, improvement
- blueprint_id (13-dim one-hot)

### Additional Telemetry Needed for New Levers

| Lever | Required Telemetry | Existing? |
|-------|-------------------|-----------|
| Tempo | blending_velocity, alpha_milestone_contributions | Partial |
| Isolation | host_gradient_via_seed, coadaptation_delta | NO |
| Dwell | training_gradient_stability, g2_headroom | Partial |
| LR | seed_loss_gradient, weight_magnitude | NO |

### Telemetry Implementation Priority

1. **blending_velocity** (d(alpha)/d(epoch)) - Easy to compute from existing metrics
2. **g2_headroom** (seed_gradient_ratio - threshold) - Already have ratio, just add threshold delta
3. **seed_loss_gradient** - Requires gradient hook during training
4. **coadaptation_delta** - Requires counterfactual with/without isolation

---

## Recommended Implementation Order

### Phase 1: Tempo Lever (Immediate)

1. Add `TempoAction` enum to `factored_actions.py`
2. Modify `FactoredAction` to include tempo
3. Update action masking (tempo only valid with GERMINATE)
4. Pass tempo to `start_blending()` via `TaskConfig.blending_steps`
5. Add `blending_velocity` to SeedTelemetry

**Estimated Effort:** 1-2 days

### Phase 2: Dwell Time Lever

1. Add `DwellAction` enum
2. Store dwell preference in SeedState
3. Modify `step_epoch()` to use per-seed dwell
4. Add `g2_headroom` to SeedTelemetry

**Estimated Effort:** 1 day

### Phase 3: Isolation Mode Lever

1. Add `IsolationAction` enum
2. Modify `_on_enter_stage()` to check action
3. Add `host_gradient_via_seed` telemetry (requires gradient hooks)
4. Add `coadaptation_delta` to counterfactual computation

**Estimated Effort:** 2-3 days (gradient hooks are tricky)

### Phase 4: LR Multiplier Lever (Future)

1. Add `LRAction` enum
2. Modify optimizer setup for per-slot param groups
3. Add `seed_loss_gradient` telemetry

**Estimated Effort:** 2 days

---

## torch.compile Considerations

### Safe Levers (No Compilation Impact)
- Tempo (blending duration)
- Dwell time
- LR multiplier
- Sigmoid steepness

### Minor Impact (Acceptable Specialization)
- Isolation mode (already a boolean branch)
- GatedBlend architecture (changes infrequent)

### Avoid
- Continuous isolation strength (custom autograd.Function)
- Dynamic gate construction at runtime

### Best Practice
All lever selections should happen at GERMINATE time, not during forward pass. This ensures:
1. Graph structure is stable within an episode
2. Specialization overhead is amortized over many forward passes
3. No per-step recompilation

---

## Conclusion

The most impactful levers to expose, in order of recommendation:

1. **Tempo (blending duration)** - Low complexity, high signal, no compile impact
2. **Dwell time (train-to-blend)** - Low complexity, enables learning of per-blueprint patience
3. **Isolation mode** - Medium complexity, enables discovery of beneficial co-adaptation
4. **LR multiplier** - Medium complexity, enables per-blueprint optimization

The philosophy remains: **give Tamiyo the levers, provide the telemetry, observe what emerges.** Just as the agent discovered depthwise separable convolutions are efficient under time pressure, it may discover non-obvious combinations of tempo, dwell, and isolation that human engineers wouldn't prescribe.
