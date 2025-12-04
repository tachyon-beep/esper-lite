# SME Report: esper.kasmina

**Package:** Seed Mechanics Layer
**Location:** `src/esper/kasmina/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert (Merged)

---

## 1. Executive Summary

The kasmina package implements a sophisticated seed lifecycle management system with gradient-isolated "incubator mode" training and alpha-scheduled blending across a 10-stage trust escalation model. It provides the mechanical framework for safe seed integration into host neural networks through quality gates (G0-G5), gradient isolation monitoring, and smooth blending schedules.

---

## 2. Key Features & Responsibilities

| Feature | Description |
|---------|-------------|
| **SeedSlot** | Lifecycle container with germination, training, blending, fossilization |
| **Quality Gates** | G0-G5 validation for stage transitions |
| **Gradient Isolation** | Incubator mode STE for safe seed training |
| **Alpha Scheduling** | Sigmoid-based 0→1 blending ramp |
| **Host Protocol** | Structural typing for graftable networks |
| **Blueprint Registry** | Plugin system for seed architectures |

---

## 3. Notable Innovations

### Incubator Mode (Straight-Through Estimator)
```python
# Forward: host features only (alpha=0)
# Backward: gradients flow to both host and seed
output = host_features + (seed_features - seed_features.detach())
```
**Benefit:** Seed learns without contributing to host output during TRAINING stage

### Sigmoid Alpha Scheduling
```python
class AlphaSchedule:
    def __call__(self, step):
        # tanh-based sigmoid for smooth 0→1 transition
        x = (step - self.midpoint) / self.temperature
        return self.start + (self.end - self.start) * (0.5 + 0.5 * math.tanh(x))
```
**Benefit:** Smooth, differentiable blending avoids sudden accuracy drops

### Quality Gates (G0-G5)
| Gate | Stage Transition | Key Checks |
|------|------------------|------------|
| G0 | Any | Sanity (seed_id, blueprint_id present) |
| G1 | GERMINATED → TRAINING | Readiness |
| G2 | TRAINING → BLENDING | Improvement ≥ threshold, isolation OK, age threshold |
| G3 | BLENDING → SHADOWING | Alpha ≥ 0.95 |
| G4 | SHADOWING → PROBATIONARY | Shadowing complete |
| G5 | PROBATIONARY → FOSSILIZED | Positive improvement, healthy |

---

## 4. Complexity Analysis

| Aspect | Rating | Notes |
|--------|--------|-------|
| Overall | MEDIUM | Well-managed state machine |
| State Machine | 10 stages | With explicit transition validation |
| Gradient Isolation | MEDIUM | Multiple isolation mechanisms |
| Blueprint System | LOW | Simple registry pattern |

---

## 5. DRL Specialist Assessment

### Quality Gates as RL Decision Points
- G2 (TRAINING→BLENDING) is the critical decision point
- Agent must learn when improvement is "good enough" to advance
- Quality gate scores could be exposed as additional features

### Seed Lifecycle as MDP State
| Observation | Value |
|-------------|-------|
| seed_stage | Integer (0-10) |
| seed_epochs_in_stage | Integer |
| seed_alpha | Float [0,1] |
| seed_improvement | Float |

### Reward Signal Opportunities
1. **Gate passage**: Bonus for successfully advancing stages
2. **Fossilization**: Large terminal reward
3. **Culling**: Negative reward but avoids wasting compute
4. **Compute rent**: Penalty proportional to seed parameters

---

## 6. PyTorch Specialist Assessment

### nn.Module Patterns

| Aspect | Rating | Notes |
|--------|--------|-------|
| Forward Pass | GOOD | Clean conditional logic for stages |
| Gradient Flow | GOOD | STE and blend_with_isolation verified |
| State Dict | GOOD | extra_state for non-tensor attributes |
| Device Handling | GOOD | to() override propagates correctly |

### torch.compile Compatibility

| Concern | Impact | Notes |
|---------|--------|-------|
| Stage-dependent branching | MEDIUM | May cause graph breaks |
| Dynamic tensor creation | LOW | Seed germination is rare |

### Memory Management

| Aspect | Rating | Notes |
|--------|--------|-------|
| Seed Parameters | GOOD | Counted and tracked |
| History Bounds | GOOD | maxlen=100 on lifecycle history |
| Fast Mode | GOOD | Skip telemetry in PPO rollouts |

---

## 7. Risks & Technical Debt

| Risk | Severity | Description |
|------|----------|-------------|
| hasattr Usage | MEDIUM | slot.py:887 - unauthorized per CLAUDE.md |
| String Topology | LOW | "cnn"/"transformer" should be Enum |
| Redundant .to() | LOW | MorphogeneticModel may call twice |

---

## 8. Opportunities for Improvement

### High Value
1. **Fix hasattr** - Replace with explicit type checking or Protocol
2. **Topology Enum** - Replace string comparisons with enum

### Medium Value
3. **Gate Score Features** - Expose G2 score as observation
4. **Compile Hints** - Add torch.compiler hints for stage branching

### Low Value
5. **Blueprint Caching** - Cache created modules by (topology, name, dim)

---

## 9. Critical Issues

### Unauthorized hasattr (MEDIUM)
```python
# slot.py:887
if hasattr(self, '_isolation_monitor'):
```
**Issue:** Violates CLAUDE.md hasattr policy
**Fix:** Use explicit attribute initialization or Protocol check

---

## 10. Recommendations Summary

| Priority | Recommendation | Effort |
|----------|----------------|--------|
| P0 | Fix unauthorized hasattr | 30 min |
| P1 | Add Topology enum | 1 hour |
| P1 | Expose gate scores as features | 2 hours |
| P2 | Add torch.compile hints | 1 hour |
| P3 | Blueprint caching | 2 hours |

---

**Quality Score:** 8.5/10 - Sophisticated implementation with clean patterns
**Confidence:** HIGH
