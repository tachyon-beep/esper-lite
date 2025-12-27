# Batch 9 Deep Dive Code Review: Tamiyo Core + Networks

**Reviewer**: DRL Specialist
**Date**: 2025-12-27
**Batch Focus**: Brain/Decision Making - Heuristic policy, LSTM network, decision tracking

---

## Executive Summary

Batch 9 covers Tamiyo's decision-making infrastructure: the heuristic policy baseline, the LSTM-based factored actor-critic network, signal tracking, and decision structures. This is the "brain" of Esper - the component that decides when to germinate, advance, fossilize, or prune seeds.

**Overall Assessment**: Well-designed with solid DRL foundations. The factored action space design is appropriate for the multi-slot morphogenetic control problem. A few integration concerns and minor correctness issues warrant attention.

**Critical Issues**: 0 P0, 2 P1
**Total Issues**: 12 findings across all severity levels

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/tamiyo/decisions.py`

**Purpose**: Defines `TamiyoDecision` - the output structure for strategic decisions.

**DRL Context**: This is the action representation from the heuristic policy (distinct from `FactoredAction` used by the neural policy).

#### Analysis

The design is clean and minimal. The decision structure captures:
- Action (using topology-specific IntEnum from `build_action_enum()`)
- Target seed ID (for targeted operations)
- Reason (for explainability/debugging)
- Confidence (useful for potential future ensemble or exploration tuning)

**Strengths**:
- Uses dynamic action enum from leyline, ensuring consistency with action space definition
- `blueprint_id` property provides clean extraction from germinate actions
- String convention (`GERMINATE_<BLUEPRINT>`) is explicit and self-documenting

**Concerns**:

| ID | Severity | Issue |
|----|----------|-------|
| D1 | P3 | `_is_germinate_action` and `_get_blueprint_from_action` are module-private but logically belong with action utilities. Consider moving to `leyline.actions` for centralized action introspection. |
| D2 | P4 | `confidence` field is set but never read by downstream consumers in the heuristic path. Document its intended use or remove if dead. |

---

### 2. `/home/john/esper-lite/src/esper/tamiyo/heuristic.py`

**Purpose**: Rule-based strategic controller implementing `TamiyoPolicy` protocol. Serves as baseline for comparison with learned policies.

**DRL Context**: This is a hand-crafted policy that encodes domain knowledge about seed lifecycle management. Critical for:
1. Ablation studies (compare learned policy against heuristic)
2. Bootstrapping (warm-start training from heuristic demonstrations)
3. Debugging (isolate network issues from reward/environment issues)

#### Analysis

**Architecture Decisions** (all sound):
- Explicit stage advancement model (GERMINATE -> ADVANCE -> FOSSILIZE/PRUNE)
- Blueprint penalty system with decay (prevents thrashing on failing blueprints)
- Embargo period after prune (anti-thrashing)
- Stabilization gate (prevents germinating during explosive host growth)

**Strengths**:
- Blueprint validation at `__init__` time (lines 102-115) - prevents runtime crashes from misconfigured blueprints
- Ransomware detection (lines 254-266) - identifies seeds with high counterfactual but negative total improvement
- Per-epoch decay (not per-decision) for blueprint penalties - correct temporal semantics
- Clear separation of germination vs. seed management decisions

**Concerns**:

| ID | Severity | Issue |
|----|----------|-------|
| H1 | P1 | **Potential starvation in `_get_next_blueprint`**: If all blueprints are penalized above threshold, the fallback (line 331) picks the minimum-penalty blueprint. However, this doesn't increment `_blueprint_index`, so repeated calls in the same state will always return the same blueprint until penalties decay. This is intentional (documented fallback) but could cause determinism issues in multi-env scenarios if not carefully managed. |
| H2 | P2 | **Missing validation for `improvement` in HOLDING decision**: At line 269, `improvement` can be `None` when both `contribution` and `total_improvement` are None. The comparison `improvement > self.config.min_improvement_to_fossilize` would raise TypeError. The code currently relies on at least one metric being populated, which appears to be enforced upstream, but lacks defensive validation. |
| H3 | P3 | **`_decide_seed_management` processes only first matching seed**: The loop at line 204 returns immediately on any matching condition. With multiple active seeds, only the first (in iteration order) is evaluated per decision call. This is correct for sequential environment stepping but may cause priority inversion if seed order doesn't match lifecycle urgency. |
| H4 | P4 | **Magic number in confidence calculation**: `min(1.0, signals.metrics.plateau_epochs / 5.0)` (line 188) and `min(1.0, improvement / 5.0)` (line 276) use hardcoded 5.0 divisor. Consider extracting to config constants for tuning. |

---

### 3. `/home/john/esper-lite/src/esper/tamiyo/__init__.py`

**Purpose**: Package entry point, exports core types and policy interfaces.

#### Analysis

Clean re-export structure. Notable that it maintains both:
- "Legacy" heuristic interface (`HeuristicTamiyo`, `TamiyoPolicy`)
- New policy bundle interface (`PolicyBundle`, `get_policy`, etc.)

**Concern**:

| ID | Severity | Issue |
|----|----------|-------|
| I1 | P4 | Comment on line 42-44 says "Legacy heuristic (kept for backwards compatibility)" but CLAUDE.md strictly prohibits backwards compatibility code. The heuristic isn't legacy - it's a parallel implementation path. Update comment to clarify it's an alternative policy type, not deprecated code. |

---

### 4. `/home/john/esper-lite/src/esper/tamiyo/tracker.py`

**Purpose**: `SignalTracker` - maintains running statistics and produces `TrainingSignals` for decision-making.

**DRL Context**: This is the observation function that transforms raw training metrics into a structured observation space. Critical for:
1. Plateau detection (triggers germination)
2. Stabilization tracking (gates germination during explosive growth)
3. History tracking (enables temporal pattern detection)

#### Analysis

**Strengths**:
- Latch behavior for stabilization (once True, stays True) - prevents oscillation
- `peek()` method for bootstrap value computation without mutating state
- Clear documentation of scale conventions (0-100 for accuracy)
- Duplicate seed_id detection (lines 202-203) - catches state machine bugs

**Concerns**:

| ID | Severity | Issue |
|----|----------|-------|
| T1 | P1 | **Stabilization can trigger on regression epochs in edge case**: At line 133-137, the check `loss_delta >= 0` prevents counting regression epochs. However, there's a subtle bug: if `relative_improvement` is negative (loss increased) but `loss_delta` is exactly 0.0 (no change), the epoch counts as stable. This is technically correct but may not match intent. More importantly, the condition `val_loss < self._prev_loss * 1.5` allows up to 50% loss increase while still counting toward stabilization if other conditions pass. |
| T2 | P2 | **Summary seed selection rule complexity**: The `summary_key` function (lines 205-211) uses a 4-tuple for priority ordering. The negation on counterfactual (`-counterfactual`) for tie-breaking prefers seeds with *lower* (more negative) counterfactual contribution. Comment says "(safety)" but the rationale is unclear - typically we'd want to prioritize seeds with *higher* positive contribution. |
| T3 | P3 | **`best_val_loss` is "best in window", not global best**: Comment on line 186-187 documents this, but the field name is misleading. `best_val_loss_in_window` or `min_recent_val_loss` would be clearer. |
| T4 | P4 | **env_id conditional telemetry**: Lines 145-161 only emit TAMIYO_INITIATED if `env_id is not None`. This is correct for multi-env scenarios but could silently skip telemetry in single-env testing if env_id isn't set. Consider emitting with env_id=-1 or similar sentinel for the "no env" case. |

---

### 5. `/home/john/esper-lite/src/esper/tamiyo/networks/factored_lstm.py`

**Purpose**: `FactoredRecurrentActorCritic` - the neural policy network with LSTM for temporal reasoning and factored action heads.

**DRL Context**: This is the core network architecture for the learned policy. Design choices are critical for:
1. Credit assignment over 10-25 epoch seed lifecycles
2. Learning correlations between action heads (slot, blueprint, style, op)
3. Maintaining stable training across the large factored action space

#### Analysis

**Architecture Assessment**:

The architecture follows established best practices:
- **Feature extraction before LSTM** (lines 127-132): Reduces dimensionality, stabilizes input
- **Pre-LSTM LayerNorm**: Normalizes input distribution to LSTM gates
- **Post-LSTM LayerNorm** (line 150): Prevents hidden state magnitude drift - critical for 25-epoch sequences
- **Orthogonal initialization** (lines 208-257): Standard for policy gradients
- **Forget gate bias = 1** (lines 246-257): Classic technique from Gers et al. (2000), helps long-term memory
- **Smaller init for policy heads** (gain=0.01): Prevents overconfident initial policies

**Factored Action Space Handling**:

The 8-head structure (slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op) is appropriate:
- All heads share LSTM temporal context (lines 330-337)
- Heads are independent at output (no autoregressive conditioning between heads)
- Action masking uses canonical `MASKED_LOGIT_VALUE` from leyline

**Concerns**:

| ID | Severity | Issue |
|----|----------|-------|
| N1 | P2 | **Style head conditional masking logic is complex and duplicated**: Lines 497-508 (`get_action`) and 613-625 (`evaluate_actions`) both implement the logic "when op is not GERMINATE or SET_ALPHA_TARGET, force style to SIGMOID_ADD". This is correct but duplicated. Consider extracting to a helper. |
| N2 | P2 | **`get_action` runs under `torch.inference_mode()`**: This is correct for inference, but the warning in the docstring (lines 392-396) about non-differentiable log_probs is critical. If anyone tries to backprop through `get_action` results, they'll get silent failures. Consider adding a runtime assertion that detects gradient requirement on inputs. |
| N3 | P3 | **Hidden state memory management is caller responsibility**: The docstring for `get_initial_hidden` (lines 266-283) correctly explains the need to detach at episode boundaries. However, this is easy to forget. Consider adding a wrapper or helper that handles detachment automatically. |
| N4 | P3 | **Missing `return_hidden` for `evaluate_actions`**: The method returns `hidden` as part of the tuple (line 632), but there's no parameter to skip computing it. For PPO updates where hidden isn't needed, this is wasted computation. |
| N5 | P4 | **Inconsistent parameter naming**: Constructor uses `num_slots`, `num_blueprints`, etc. (lines 87-98) but these can be None to use `slot_config` defaults. The fallback logic at lines 109-122 works but creates two paths to configure the same thing. |

---

### 6. `/home/john/esper-lite/src/esper/tamiyo/networks/__init__.py`

**Purpose**: Package entry point for network modules.

**Analysis**: Minimal re-export, correct.

No issues.

---

## Cross-Cutting Integration Concerns

### I1. Heuristic vs Neural Policy Interface Divergence

**Files**: `heuristic.py`, `heuristic_bundle.py`, `networks/factored_lstm.py`

The heuristic policy uses `TrainingSignals` (semantic observations) while the neural policy uses tensor features from `obs_to_multislot_features`. The `HeuristicPolicyBundle` adapter (reviewed in related batch) bridges this by raising `NotImplementedError` for all tensor-based methods.

**Risk**: Training loops must maintain two distinct code paths. If someone tries to use heuristic with the neural training loop or vice versa, failures are explicit (good) but error messages could be clearer about *why* the interface mismatch exists.

**Recommendation**: Add docstring to `HeuristicPolicyBundle` explaining the fundamental architectural difference (semantic vs. tensor observations).

### I2. Action Enum Mismatch Between Heuristic and Neural Paths

**Files**: `heuristic.py` (uses `build_action_enum`), `factored_lstm.py` (uses `LifecycleOp` + factored heads)

The heuristic uses a flat action enum (`Action.GERMINATE_CONV_LIGHT`, `Action.PRUNE`, etc.) while the neural policy uses factored actions (`op=GERMINATE`, `blueprint=CONV_LIGHT`). This is intentional and correct, but:

**Risk**: Comparison between heuristic and neural decisions requires mapping. The `TamiyoDecision` from heuristic vs `GetActionResult` from neural have different structures.

**Recommendation**: Add a utility to convert between representations for logging/comparison purposes.

### I3. MaskedCategorical Validation Toggle

**File**: `action_masks.py` (from policy subpackage, reviewed as dependency)

`MaskedCategorical.validate` is a class-level toggle that disables mask validation. This is intended for production performance, but:

**Risk**: If validation is disabled globally and a state machine bug occurs, the system will sample from an all-masked distribution (undefined behavior) instead of raising `InvalidStateMachineError`.

**Recommendation**: Keep validation enabled by default. Only disable for performance-critical production runs after thorough testing.

### I4. Seed Summary Selection Rule

**File**: `tracker.py` (lines 205-211)

The rule for selecting a "summary seed" from multiple active seeds uses a complex 4-tuple key. This affects what appears in `TrainingSignals.seed_*` fields.

**Risk**: The neural policy observes `TrainingSignals` through feature extraction. If the summary seed selection doesn't match what the policy expects, observations become unreliable.

**Recommendation**: Document the selection rule in `TrainingSignals` docstring. Consider whether multi-slot feature extraction (`obs_to_multislot_features`) should bypass the summary and use per-slot features instead.

---

## Test Coverage Assessment

Reviewed test files:
- `tests/simic/test_tamiyo_network.py`: Comprehensive coverage of `FactoredRecurrentActorCritic`
- `tests/tamiyo/properties/test_tamiyo_properties.py`: Property-based testing with Hypothesis

**Strengths**:
- Property tests cover stabilization latch monotonicity, tracker invariants, decision idempotence
- Network tests cover masking, entropy normalization, hidden state propagation
- Edge cases like single-action heads and all-masked distributions are tested

**Gaps**:

| ID | Gap Description |
|----|-----------------|
| G1 | No tests for ransomware detection logic in `HeuristicTamiyo` |
| G2 | No tests for blueprint penalty fallback behavior (all blueprints above threshold) |
| G3 | No integration tests for heuristic-to-neural decision comparison |
| G4 | Missing test for `tracker.peek()` non-mutation property |

---

## DRL-Specific Recommendations

### R1. Entropy Coefficient Guidance

The `MaskedCategorical.entropy()` returns normalized entropy [0, 1]. The docstring correctly notes that entropy coefficients should be higher than typical values. However, the network doesn't enforce or validate this.

**Recommendation**: Add a `suggested_entropy_coef` property to the network that returns an appropriate range (0.05-0.1) based on normalized entropy semantics.

### R2. LSTM Chunk Length Coordination

The LSTM processes sequences up to 25 epochs (episode length). For proper truncated BPTT, the PPO update's `chunk_length` must align with this.

**Recommendation**: Add assertion in training loop that `chunk_length <= episode_length`.

### R3. Hidden State Detachment Pattern

The comment in `get_initial_hidden` about detaching hidden states at episode boundaries is critical but easy to miss.

**Recommendation**: Add a convenience method `detach_hidden(hidden)` that documents and performs the detachment pattern.

### R4. Value Head Initialization

Value head uses gain=1.0 (line 233), while policy heads use gain=0.01. This is appropriate (value estimates should be unconstrained while policy should be near-uniform initially), but the asymmetry isn't documented.

**Recommendation**: Add comment explaining the different initialization rationale.

---

## Severity-Tagged Findings Summary

| ID | File | Severity | Description |
|----|------|----------|-------------|
| H1 | heuristic.py | P1 | Blueprint selection fallback doesn't increment index - potential determinism issue |
| T1 | tracker.py | P1 | Stabilization edge case: 50% loss increase can count toward stabilization |
| H2 | heuristic.py | P2 | Missing None validation for improvement in HOLDING decision |
| N1 | factored_lstm.py | P2 | Style mask conditional logic duplicated between get_action and evaluate_actions |
| N2 | factored_lstm.py | P2 | No runtime guard against backprop through inference_mode results |
| T2 | tracker.py | P2 | Summary seed selection prefers lower counterfactual (counterintuitive) |
| D1 | decisions.py | P3 | Action introspection helpers should move to leyline.actions |
| H3 | heuristic.py | P3 | First-seed-only processing in multi-seed scenario |
| N3 | factored_lstm.py | P3 | Hidden state detachment is caller responsibility |
| N4 | factored_lstm.py | P3 | No option to skip hidden state computation in evaluate_actions |
| T3 | tracker.py | P3 | best_val_loss naming suggests global best, is actually window best |
| D2 | decisions.py | P4 | confidence field is unused |
| H4 | heuristic.py | P4 | Hardcoded 5.0 divisor in confidence calculations |
| I1 | __init__.py | P4 | "Legacy" comment misleading - heuristic is alternative, not deprecated |
| N5 | factored_lstm.py | P4 | Two paths to configure head sizes (explicit params vs slot_config) |
| T4 | tracker.py | P4 | env_id conditional silently skips telemetry in single-env case |

---

## Conclusion

Tamiyo's decision-making infrastructure is well-architected for the morphogenetic control problem. The LSTM-based factored actor-critic correctly handles temporal credit assignment over seed lifecycles, and the heuristic provides a solid baseline for comparison.

The two P1 issues (blueprint selection determinism, stabilization edge case) should be addressed before production use. The P2 issues are less critical but represent potential sources of subtle bugs.

The separation between heuristic (semantic observation space) and neural (tensor feature space) policies is intentional and well-executed, though the interface differences require careful handling in training loops.

**Next Steps**:
1. Fix P1 issues
2. Add missing test coverage for ransomware detection and blueprint fallback
3. Document the heuristic vs neural interface difference more prominently
