# Blueprint Compiler Appendix: Anti-Pattern Blueprints

> **Parent Plan:** `2026-01-09-blueprint-compiler-and-curriculum-seeds.md`
> **Status:** Deferred to Phase 4 (after Phase 3 curriculum blueprints)

**Expert Review Status:** v1 - Designed by PyTorch specialist

**Purpose:** Intentionally BAD blueprints that Tamiyo should learn to AVOID. By including anti-patterns in the action space, the agent learns architectural boundaries and develops "taste" for what works vs. what doesn't.

---

## Index Allocation

| Index | Blueprint | Category | Status |
|-------|-----------|----------|--------|
| 0-7 | Legacy CNN | - | Existing |
| 8-12 | Legacy Transformer | - | Existing |
| 13-16 | Phase 2 CNN (curriculum) | - | In main plan |
| 17-23 | Phase 3 CNN (curriculum) | - | In appendix |
| **24-33** | **Phase 4 Anti-patterns** | - | **This document** |

---

## Design Philosophy

Anti-patterns serve three curriculum purposes:

1. **Boundary Learning** — The agent discovers what architectures are harmful
2. **Negative Transfer** — Learning "don't do X" is as valuable as "do Y"
3. **Robustness** — Exposure to failure modes makes the agent more conservative

**Curriculum Ordering:** Introduce in order of signal clarity:
1. Catastrophic failures first (immediate punishment, easy to learn)
2. Gradient killers second (delayed but clear signal)
3. Deceptive patterns third (requires cost awareness)
4. Subtle degraders last (requires long-horizon evaluation)

---

## Category 1: Subtle Degraders

*Look reasonable, slowly poison the model over many steps*

### 1. Correlated Noise Injector — Index 24

**Concept:** Adds small Gaussian noise to activations, but the noise is correlated across the batch dimension (same noise vector for all samples in a batch).

**Failure Mode:** Destroys batch independence assumption. BatchNorm statistics become biased, gradients no longer average to true gradient. Loss curves look healthy but test accuracy plateaus 5-15% below optimal.

**Severity:** Moderate

**Lesson:** Batch-correlated operations are toxic. Any intervention breaking sample independence should be avoided.

**Implementation Notes:**
```
Architecture:
  - Generate noise tensor of shape [1, C, 1, 1]
  - Expand to batch size (same noise for all samples)
  - Add noise scaled by learnable sigma (init 0.01)
  - Apply LayerScale

Key detail: noise must be generated ONCE per forward pass and broadcast,
not generated independently per sample.
```

---

### 2. Asymmetric Leaky Bottleneck — Index 25

**Concept:** Bottleneck that reduces channels by 8x, applies ReLU, then expands with a DIFFERENT projection matrix than the compression inverse.

**Failure Mode:** Information bottleneck too aggressive + asymmetric projections mean module cannot learn identity. Representation drift accumulates over thousands of steps as features become increasingly lossy.

**Severity:** Mild to Moderate

**Lesson:** Bottlenecks need skip connections or symmetric encode/decode paths.

**Implementation Notes:**
```
Architecture:
  - down: Conv2d(C, C//8, 1x1)
  - ReLU
  - up: Conv2d(C//8, C, 1x1)  # Different random init, NOT transpose of down
  - NO skip connection
  - LayerScale

Key detail: deliberately omit residual connection that other bottlenecks have.
```

---

### 3. Momentum Dampener — Index 26

**Concept:** Exponential moving average blending current activations with history using high momentum (0.99). Output = 0.99 * EMA + 0.01 * current.

**Failure Mode:** Gradients scaled by 0.01 through module (gradient bottleneck). Creates temporal dependence violating i.i.d. assumption — output for sample N depends on samples 1..N-1. Training becomes order-dependent and non-reproducible.

**Severity:** Moderate

**Lesson:** Stateful modules with high momentum create hidden gradient scaling and temporal coupling.

**Implementation Notes:**
```
Architecture:
  - Maintain EMA buffer (register_buffer, not parameter)
  - On forward: EMA = 0.99 * EMA + 0.01 * x
  - Output: EMA (detached) + LayerScale(x - EMA.detach())

Key detail: The 0.99 momentum means gradients are effectively 0.01x
```

---

## Category 2: Gradient Killers

*Break training dynamics through vanishing/exploding gradients*

### 4. Sigmoid Sandwich — Index 27

**Concept:** Two consecutive sigmoid activations with a linear layer between: Sigmoid → Linear → Sigmoid.

**Failure Mode:** Classic vanishing gradient. Sigmoid saturates at 0/1 where gradients are near-zero. Product of two near-zero gradients = training death. Linear layer weights freeze within epochs.

**Severity:** Severe

**Lesson:** Never chain saturating activations.

**Implementation Notes:**
```
Architecture:
  - Sigmoid
  - Conv2d(C, C, 1x1)
  - Sigmoid
  - LayerScale (won't save it)

Telemetry signature: gradient norm → 0 in this module's parameters
```

---

### 5. Gain Amplifier — Index 28

**Concept:** Learnable scalar gain parameter (init 1.0), unconstrained: output = gain * input.

**Failure Mode:** Positive feedback loop. If loss decreases faster with gain > 1, optimizer increases gain → larger gradients → faster gain updates → explosion to 1e6+ and NaN within 100-500 steps. Alternatively collapses to near-zero.

**Severity:** Catastrophic (delayed)

**Lesson:** Unconstrained multiplicative parameters are unstable. Learnable scalars need constraints.

**Implementation Notes:**
```
Architecture:
  - self.gain = nn.Parameter(torch.ones(1))
  - Output: self.gain * x  # NO clamping, NO spectral norm
  - LayerScale (applied after, doesn't help)

Key detail: deliberately omit any constraint on gain magnitude
```

---

### 6. Deep Linear Stack — Index 29

**Concept:** 8 consecutive linear layers (same hidden dim) with NO nonlinearity. Mathematically equivalent to single linear, but gradient dynamics differ.

**Failure Mode:** Gradient = product of 8 weight matrices. By random matrix theory, product explodes or vanishes exponentially with depth. Unstable within first epoch without orthogonal init. Even with good init, 8 params matrices learning what 1 could do = wasteful.

**Severity:** Severe

**Lesson:** Linear layers must be separated by nonlinearities. Depth without nonlinearity is worse than shallow.

**Implementation Notes:**
```
Architecture:
  - 8 x Conv2d(C, C, 1x1, bias=False)
  - NO activations between them
  - LayerScale at end

Key detail: standard Kaiming init (not orthogonal), no skip connections
```

---

## Category 3: Deceptive Patterns

*Seem helpful but add overhead without proportional benefit*

### 7. Expensive Identity — Index 30

**Concept:** Conv 3x3 → BN → ReLU → Conv 3x3 → BN, with Dirac initialization ensuring it learns to stay near identity. A residual block WITHOUT the residual connection.

**Failure Mode:** Adds 2 convolutions, 2 batchnorms, 2 activations worth of compute while contributing nothing. CAN learn useful features but has no gradient highway, defaults to identity. High FLOPS, zero accuracy benefit.

**Severity:** Mild

**Lesson:** Compute cost matters. Prefer modules with benefit proportional to cost.

**Implementation Notes:**
```
Architecture:
  - Conv2d(C, C, 3x3, padding=1) - Dirac init
  - GroupNorm + ReLU
  - Conv2d(C, C, 3x3, padding=1) - zero init
  - GroupNorm
  - NO residual connection (just output the transformed value)
  - LayerScale

Key detail: without skip connection, gradients must flow through entire path
```

---

### 8. Redundant Ensemble — Index 31

**Concept:** Same input processed by 4 identical 1x1 conv branches (same arch, different random init), then averaged.

**Failure Mode:** Branches learn correlated features (identical inputs + gradients). Averaging correlated predictors provides almost no ensemble benefit. 4x parameters and compute for ~1.1x effective capacity.

**Severity:** Mild

**Lesson:** Ensemble diversity requires architectural or input diversity, not just init diversity.

**Implementation Notes:**
```
Architecture:
  - branches = nn.ModuleList([Conv2d(C, C, 1x1) for _ in range(4)])
  - Output: mean([branch(x) for branch in branches])
  - LayerScale

Key detail: all branches see identical input, receive identical gradients
```

---

## Category 4: Catastrophic Failures

*Immediate model destruction within first forward passes*

### 9. Kaiming Mismatch — Index 32

**Concept:** Conv layer with Kaiming (fan-in) init designed for ReLU, but followed by linear activation (no nonlinearity).

**Failure Mode:** Kaiming init assumes ReLU kills half the activations, scales by sqrt(2/fan_in). Without ReLU, activations 40% larger than expected. Compounds across layers: (1.4)^10 ≈ 29x too large. Immediate gradient explosion or NaN.

**Severity:** Catastrophic

**Lesson:** Initialization must match activation function.

**Implementation Notes:**
```
Architecture:
  - Conv2d(C, C, 3x3, padding=1)
  - nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
  - NO activation function applied
  - LayerScale (too late)

Key detail: init says "relu" but forward has no relu
```

---

### 10. Division By Feature — Index 33

**Concept:** Normalization variant dividing activations by channel-wise mean: output = input / mean(input, dim=1).

**Failure Mode:** If any channel mean is near zero (common with ReLU or centered data), division produces Inf/NaN that propagates instantly. Even surviving, gradient of 1/x at small x is huge → explosion. Fails within first forward pass.

**Severity:** Catastrophic (immediate)

**Lesson:** Division by data-dependent values is extremely dangerous without safeguards.

**Implementation Notes:**
```
Architecture:
  - mean = x.mean(dim=1, keepdim=True)
  - Output: x / mean  # NO epsilon, NO clamping
  - LayerScale (never reached if NaN)

Key detail: deliberately omit the + eps that makes LayerNorm safe
```

---

## Implementation Plan

### Task A.1: Create antipatterns.py module

**Files:**
- Create: `src/esper/kasmina/blueprints/antipatterns.py`
- Test: `tests/kasmina/test_antipattern_blueprints.py`

**Step 1:** Write failing tests for each anti-pattern:
- Shape preservation (must still work as a module)
- Specific failure mode detection (e.g., gradient norm collapse)

**Step 2:** Implement each anti-pattern with:
- `@BlueprintRegistry.register()` decorator
- Explicit `action_index` (24-33)
- `is_antipattern=True` flag (new field for BlueprintSpec)
- Clear docstring explaining why it's bad

**Step 3:** Add antipattern loader to topology modules

---

### Task A.2: Add is_antipattern flag to BlueprintSpec

**Files:**
- Modify: `src/esper/kasmina/blueprints/registry.py`

**Changes:**
```python
@dataclass(frozen=True, slots=True)
class BlueprintSpec:
    name: str
    topology: str
    factory: BlueprintFactory
    param_estimate: int
    description: str = ""
    action_index: int | None = None
    is_antipattern: bool = False  # NEW: marks intentionally bad blueprints
```

This allows Tamiyo's reward function to verify negative outcomes match antipattern selections.

---

### Task A.3: Add telemetry hooks for failure detection

**Files:**
- Modify: `src/esper/simic/rewards.py` (or appropriate rewards module)

**Purpose:** Detect and log characteristic failure signatures:
- Gradient norm collapse → Sigmoid Sandwich, Momentum Dampener
- Gradient explosion → Gain Amplifier, Deep Linear Stack
- NaN/Inf → Division By Feature, Kaiming Mismatch
- Validation plateau → Correlated Noise, Asymmetric Bottleneck

---

## Testing Strategy

Each anti-pattern needs TWO types of tests:

### 1. Functional Tests (must pass)
```python
def test_antipattern_preserves_shape(name):
    """Anti-pattern must still function as a module."""
    seed = BlueprintRegistry.create("cnn", name, dim=64)
    x = torch.randn(2, 64, 32, 32)
    y = seed(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()  # Most should survive one pass
```

### 2. Failure Mode Tests (must demonstrate badness)
```python
def test_sigmoid_sandwich_kills_gradients():
    """Sigmoid Sandwich should have vanishing gradients."""
    seed = BlueprintRegistry.create("cnn", "sigmoid_sandwich", dim=64)
    # ... train for N steps ...
    # Assert gradient norm collapsed

def test_gain_amplifier_explodes():
    """Gain Amplifier should explode within 500 steps."""
    # ... training loop ...
    # Assert gain parameter > 1e4 or NaN occurred
```

---

## Curriculum Integration

### Reward Signal Design

Anti-patterns should produce **clear negative reward** so Tamiyo learns to avoid them:

| Failure Type | Detection | Reward Signal |
|--------------|-----------|---------------|
| NaN/Inf | Immediate check | -100 (catastrophic) |
| Gradient collapse | Grad norm < 1e-7 | -10 per occurrence |
| Gradient explosion | Grad norm > 1e4 | -10 per occurrence |
| Validation plateau | Val loss stagnant 50+ steps | -5 cumulative |
| High cost, low benefit | FLOPS/accuracy ratio | -1 per wasteful selection |

### Ordering (easiest to learn first)

1. **Catastrophic** (24-25): Division By Feature, Kaiming Mismatch
   - Immediate NaN = immediate punishment

2. **Gradient Killers** (26-29): Sigmoid Sandwich, Gain Amplifier, Deep Linear Stack
   - Clear gradient telemetry signal

3. **Deceptive** (30-31): Expensive Identity, Redundant Ensemble
   - Requires cost awareness

4. **Subtle** (32-33): Correlated Noise, Asymmetric Bottleneck, Momentum Dampener
   - Requires long-horizon evaluation

---

## Summary

| Index | Name | Category | Severity | Key Lesson |
|-------|------|----------|----------|------------|
| 24 | correlated_noise | Subtle | Moderate | Batch independence matters |
| 25 | asymmetric_bottleneck | Subtle | Mild-Mod | Bottlenecks need skips |
| 26 | momentum_dampener | Subtle | Moderate | Avoid stateful high-momentum |
| 27 | sigmoid_sandwich | Gradient | Severe | Never chain saturating activations |
| 28 | gain_amplifier | Gradient | Catastrophic | Constrain multiplicative params |
| 29 | deep_linear_stack | Gradient | Severe | Depth requires nonlinearity |
| 30 | expensive_identity | Deceptive | Mild | Compute should match benefit |
| 31 | redundant_ensemble | Deceptive | Mild | Naive ensembling is wasteful |
| 32 | kaiming_mismatch | Catastrophic | Catastrophic | Match init to activation |
| 33 | division_by_feature | Catastrophic | Catastrophic | Division by data is dangerous |

---

## Future Extensions

Once basic anti-patterns are learned, consider:

1. **Composite Anti-patterns** — Combinations that are worse together
2. **Context-dependent Anti-patterns** — Bad in CNN, fine in Transformer (or vice versa)
3. **Adversarial Blueprints** — Designed to exploit specific reward function weaknesses
