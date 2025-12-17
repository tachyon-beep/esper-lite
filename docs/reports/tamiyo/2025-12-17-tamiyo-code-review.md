# Tamiyo Subsystem Code Review Report

**Date:** 2025-12-17
**Reviewers:** PyTorch Specialist Agent, DRL Specialist Agent
**Scope:** Complete Tamiyo subsystem (2,365 lines across 12 files)

---

## Executive Summary

The Tamiyo subsystem implements the "Brain/Cortex" of Esper - strategic decision-making for seed lifecycle management. The review identified **3 critical**, **7 high**, **15 medium**, and **23+ low** severity issues across PyTorch and DRL dimensions.

### Top 5 Critical/High Issues

| # | Severity | File | Issue | Impact |
|---|----------|------|-------|--------|
| 1 | **CRITICAL** | features.py | **Observations lack normalization** - PPO assumes roughly normalized inputs (~[-1,1]). Raw features include unbounded values (epoch, global_step) and poorly scaled values (total_params in millions). | Training instability, value function explosion, poor sample efficiency |
| 2 | **CRITICAL** | action_masks.py | **MaskedCategorical validation forces CUDA sync** on every distribution creation via `.any()` and `.sum()` operations, despite `@torch.compiler.disable` | 100-1000x overhead in tight training loops |
| 3 | **HIGH** | Cross-cutting | **Multi-head log_prob handling for PPO ratio unverified** - factored action space returns per-head log_probs; joint probability computation and clipping behavior needs verification | Incorrect importance weights, training instability |
| 4 | **HIGH** | tracker.py | **Stabilization latch creates non-stationary MDP** - action space availability changes permanently mid-episode when host stabilizes | Value estimation confusion, policy instability |
| 5 | **HIGH** | heuristic.py | **Fossilization threshold allows zero-improvement fossilization** (`min_improvement_to_fossilize=0.0`) enabling reward hacking via marginal fossilization | Agent learns to fossilize quickly for terminal bonuses without real value |

---

## Detailed Findings by Module

### policy/action_masks.py (384 lines)

#### PyTorch Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| **CRITICAL** | Validation forces CUDA sync via `.any()`/`.sum()` | L286-313 | Add validation toggle for production; sample validation (every 100th batch) |
| Medium | MASKED_LOGIT_VALUE tensor created per-call | L347-352 | Use scalar directly with `masked_fill()` - no tensor needed |
| Medium | Entropy uses recomputed log_probs | L377-378 | Use `F.log_softmax(self.masked_logits, dim=-1)` for clarity |
| Low | Batch mask computation creates N*4 tensors then stacks | L209-225 | Pre-allocate batch tensors for high-throughput scenarios |
| Low | `clamp(min=1)` vs `clamp(min=1e-8)` inconsistency | L380-382 | Document different clamping strategies |

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Medium | Optimistic masking creates train/execute discrepancy | L142-172 | Compute joint slot-op mask or document rejection semantics |
| Medium | MASKED_LOGIT_VALUE may cause FP16 instability | L347-352 | Consider `-1e3` or dynamic `finfo(dtype).min / 2` |
| Low | Normalized entropy may cause exploration collapse | L368-384 | Use unnormalized entropy for bonus, normalized only for logging |

---

### policy/lstm_bundle.py (294 lines)

#### PyTorch Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| **HIGH** | `get_value()` missing `@torch.inference_mode()` | L208-224 | Add decorator if inference-only, or document training use case |
| **HIGH** | `getattr` for torch.compile detection needs authorization | L243, 248 | Add authorization comment per CLAUDE.md policy |
| Medium | `device` property uses fragile `next(parameters())` | L254-256 | Add fallback: `next(iter(...), None)` |
| Medium | `to()` may have issues with compiled modules | L258-261 | Document that `to()` must precede external references |
| Medium | Missing dtype consistency check in forward paths | All forward methods | Add assertion if input dtype mismatches policy dtype |
| Low | `expand_mask()` closure recreated on every forward | L134-139 | Make module-level helper |

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| **HIGH** | Hidden state handling during PPO evaluation needs verification | L158-184, 228-236 | Verify sequence-level training or hidden state reconstruction |
| Medium | `get_value()` discards wrong sequence position for bootstrap | L222-224 | Document single-step use or add `timestep` parameter |
| Low | `network` property may return compiled wrapper | L288-291 | Return `_orig_mod` for consistency |

---

### policy/features.py (257 lines)

#### PyTorch Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Low | Returns `list[float]` instead of tensor | L105-209 | Acceptable for flexibility; convert to numpy for production |
| Low | Missing `safe()` wrapper on accuracy values | L174-175 | Apply `safe()` for consistency |

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| **CRITICAL** | **Observations lack normalization** | L105-209 | Implement running mean/std normalization (VecNormalize-style) |
| **HIGH** | Returns list[float] requiring expensive conversion | L105-209 | Return numpy array or pre-allocated tensor buffer |
| Medium | Blueprint one-hot creates sparse features (30% of obs) | L202-207 | Consider learned embeddings |
| Medium | History features lack temporal encoding | L180-181 | Document ordering or add positional encoding |
| Low | `safe()` has asymmetric default handling | L44-59 | Consider sentinel values for "no data" vs "overflow" |

---

### policy/protocol.py (216 lines)

#### PyTorch Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Medium | `initial_hidden()` inference_mode rationale unclear | L147-160 | Document gradient prevention purpose |
| Low | Missing device validation guidance | L183-185 | Note that implementations must handle buffers |

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Medium | No feature dimension validation | Protocol-wide | Add `feature_dim: int` property |
| Low | Protocol requires off-policy methods for on-policy impls | L116-135 | Consider protocol split or document optionality |

---

### policy/heuristic_bundle.py (179 lines)

#### PyTorch Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Low | `dtype` returns hardcoded float32 | L157-159 | Consider ambient dtype from config |

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Medium | Violates PolicyBundle contract semantics | L56-115 | Don't register via `@register_policy` or implement properly |
| Low | Operates on different input modality than learned policies | L26-27 | Document information advantage for fair ablation |

---

### policy/registry.py (124 lines)

#### PyTorch Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Medium | `hasattr` usage requires authorization | L70-71 | Add authorization comment |

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Low | Registry uses global mutable state without thread safety | L11, 80-86 | Add threading.Lock or document import-time-only registration |

---

### heuristic.py (338 lines)

**Note:** This file has no PyTorch dependencies - correctly decoupled from tensor operations.

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| **HIGH** | Fossilization threshold allows zero-improvement fossilization | L62 | Set meaningful threshold (e.g., 0.5% improvement) |
| Medium | Confidence heuristic is ad-hoc and not calibrated | L180, 257 | Remove or document as logging-only |
| Medium | Blueprint penalty decay creates non-Markovian MDP | L292-317 | Expose penalty state in TrainingSignals |
| Medium | Single-seed focus in multi-seed environment | L196-268 | Implement explicit triage function |
| Low | Ransomware detection threshold may be too sensitive | L237-247 | Add minimum epoch requirement |
| Low | Embargo creates hidden temporal credit assignment | L146-152 | Add `embargo_remaining` to observations |
| Low | Round-robin blueprint selection lacks exploration signal | L299-312 | Log blueprint-outcome correlations |

---

### tracker.py (340 lines)

**Note:** This file has no PyTorch dependencies - uses Python deque for efficiency.

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| **HIGH** | Stabilization latch creates non-stationarity | L72-74, 123-156 | Ensure observation includes `host_stabilized` prominently |
| Medium | `peek()` bootstrap semantics may cause GAE bias | L234-256 | Document usage constraints or accept explicit prev values |
| Medium | Summary seed selection loses multi-slot information | L188-214 | Expose per-seed metrics or seed counts |
| Medium | `reset()` doesn't reset `env_id` | L57, 317-329 | Reset or make immutable |
| Low | History truncation loses early episode information | L224-225 | Expose full window or derived features |
| Low | Stabilization uses hard-coded epsilon | L125-128 | Use proper NaN handling |
| Low | Accuracy scale validation is noisy | L103-115 | Suppress for first N epochs |

---

### decisions.py (53 lines)

**Note:** Pure Python dataclass utilities - no PyTorch or complex DRL issues.

#### DRL Findings

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Low | Action classification uses string parsing | L12-22 | Consider explicit ActionType enum |
| Low | `confidence` field appears unused | L35 | Remove or document intended use |
| Low | No validation of action-target consistency | L25-43 | Add `__post_init__` validation |

---

### __init__.py (52 lines - root)

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Low | "Legacy" comment violates No Legacy Code Policy | L41 | Remove comment or clarify as valid baseline |
| Low | Two Policy protocols exported | L37-51 | Consolidate on single interface |

---

### policy/__init__.py (65 lines)

| Severity | Issue | Location | Suggestion |
|----------|-------|----------|------------|
| Low | Import triggers registration side effect | L41-42 | Consider explicit registration function |

---

## Cross-Cutting Concerns

### 1. Observation Normalization (CRITICAL)

**Issue:** No running mean/std normalization anywhere in the pipeline.

**Impact:**
- Early training: High variance features cause unstable value estimates
- Late training: Feature drift causes catastrophic forgetting
- Cross-task transfer: Impossible without normalization

**Recommendation:** Implement VecNormalize-style running statistics in Simic's vectorized training loop. Store in checkpoints.

### 2. Multi-Head Action Factorization (HIGH)

**Issue:** PPO ratio `pi(a|s) / pi_old(a|s)` requires joint probability across all 4 action heads.

**Verification needed:**
1. Are log_probs summed correctly across heads?
2. Is clipping applied to joint ratio or per-head?
3. Are advantages computed for joint actions?

### 3. Observation Space Fragmentation (HIGH)

**Issue:** The heuristic maintains hidden state (blueprint penalties, embargo countdown) not exposed in observations.

**Impact:** Learned policy cannot access the same information the heuristic uses, creating unfair comparison and potential ceiling on RL performance.

**Recommendation:** Audit heuristic information access and expose in TrainingSignals.

### 4. Entropy Coefficient Tuning (MEDIUM)

**Issue:** MaskedCategorical returns normalized entropy in [0,1]. Standard PPO coefficients (~0.01) are calibrated for unnormalized entropy.

**Recommendation:** Document expected coefficient range or implement automatic entropy tuning (SAC-style).

---

## Recommendations

### Immediate Actions (Critical)

1. **Implement observation normalization** in Simic's training loop
   - Running mean/std with exponential moving average
   - Store statistics in checkpoints
   - Apply before policy forward pass

2. **Add validation toggle** to MaskedCategorical
   - Disable in production, enable in debug
   - Or sample validation (every Nth batch)

### High Priority

3. **Verify multi-head log_prob handling** in Simic PPO
   - Check joint probability computation
   - Verify clipping behavior

4. **Increase fossilize threshold** from 0.0 to prevent reward hacking
   - Suggest minimum 0.5% improvement

5. **Add `@torch.inference_mode()`** to `LSTMPolicyBundle.get_value()` if inference-only

6. **Expose heuristic hidden state** (blueprint penalties, embargo countdown) in observations

### Medium Priority

7. **Add `hasattr` authorization comments** per CLAUDE.md policy
8. **Add dtype consistency checks** in forward paths
9. **Document or fix `peek()` bootstrap semantics** in tracker
10. **Expose per-seed metrics** for multi-slot environments

### Low Priority

11. Various documentation improvements
12. Thread safety for registry
13. Minor efficiency optimizations (tensor caching, vectorization)

---

## Positive Findings

The review also identified several well-designed patterns:

- **LSTM initialization** follows best practices (orthogonal init, forget gate bias=1.0)
- **LayerNorm placement** is well-reasoned with extensive documentation
- **`@torch.compiler.disable`** correctly isolates validation from compilable paths
- **`@torch.inference_mode()`** correctly used for action selection
- **Protocol design** is clean and PyTorch-appropriate
- **Feature extraction** correctly avoids heavy imports in hot path
- **Frozen dataclasses** with slots are memory-efficient
- **Heuristic decoupling** from PyTorch is clean

---

## Test Coverage Assessment

Tamiyo has substantial test coverage (5,305 lines):

| Test File | Lines | Purpose |
|-----------|-------|---------|
| test_action_masks.py | 1,069 | Action mask correctness |
| test_mask_properties.py | 520 | Property-based mask testing |
| test_decision_semantics.py | 483 | Decision logic semantics |
| test_state_machine_properties.py | 473 | State machine invariants |
| test_decision_antigaming.py | 446 | Anti-gaming properties |
| test_tamiyo_properties.py | 434 | Cross-domain properties |
| test_features.py | 381 | Feature extraction |
| test_registry.py | 169 | Policy registry |
| test_lstm_bundle.py | 166 | LSTM policy bundle |
| Various integration tests | ~400 | Kasmina/Tolaria/Simic integration |

**Assessment:** Strong property-based testing foundation. Coverage gaps exist around:
- Observation normalization (doesn't exist to test)
- Multi-head log_prob joint computation
- Bootstrap value edge cases

---

## Conclusion

The Tamiyo subsystem demonstrates solid software engineering with clean separation of concerns, good documentation, and extensive testing. However, **the critical observation normalization gap is a blocking issue for stable PPO training**. The high-priority items around multi-head action handling and fossilization thresholds should be addressed before production deployment of the RL policy.

Estimated effort:
- Critical fixes: 2-4 hours
- High priority: 4-8 hours
- Medium priority: 8-16 hours

---

*Report generated by Claude Code with PyTorch and DRL specialist agents.*
