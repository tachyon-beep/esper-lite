# Batch 3 Code Review: Kasmina Blueprints + Slot (Seed Architectures & Lifecycle)

**Reviewer:** DRL Specialist
**Date:** 2025-12-27
**Files Reviewed:**
1. `/home/john/esper-lite/src/esper/kasmina/blueprints/cnn.py`
2. `/home/john/esper-lite/src/esper/kasmina/blueprints/initialization.py`
3. `/home/john/esper-lite/src/esper/kasmina/blueprints/__init__.py`
4. `/home/john/esper-lite/src/esper/kasmina/blueprints/registry.py`
5. `/home/john/esper-lite/src/esper/kasmina/blueprints/transformer.py`
6. `/home/john/esper-lite/src/esper/kasmina/isolation.py`
7. `/home/john/esper-lite/src/esper/kasmina/slot.py`

**Related Files Examined:**
- `/home/john/esper-lite/src/esper/kasmina/blend_ops.py`
- `/home/john/esper-lite/src/esper/kasmina/alpha_controller.py`
- `/home/john/esper-lite/src/esper/leyline/stages.py`
- `/home/john/esper-lite/src/esper/tamiyo/policy/features.py`
- Test files in `tests/kasmina/`

---

## Executive Summary

The Kasmina blueprints and slot system is well-architected for morphogenetic RL. The blueprint registry provides a clean extensibility mechanism, the isolation module correctly implements gradient isolation semantics, and the SeedSlot lifecycle management is thorough. However, I identified several issues ranging from parameter count inaccuracies (affecting rent economy signals) to potential DDP safety gaps.

**Critical Findings:**
- **P1:** Parameter estimate inaccuracies for CNN blueprints affect rent economy observations
- **P2:** FlexAttention block mask cache could accumulate memory in long inference
- **P2:** Gradient ratio normalization uses sqrt scaling that may disadvantage small seeds

**Strengths:**
- Excellent documentation with clear mathematical formulations
- Well-designed identity-at-birth initialization patterns
- Comprehensive DDP synchronization for lifecycle gates
- Clean separation between amplitude scheduling (AlphaController) and per-sample gating

---

## File-by-File Analysis

### 1. `/home/john/esper-lite/src/esper/kasmina/blueprints/cnn.py`

**Purpose:** CNN seed blueprint definitions (noop, norm, attention, depthwise, bottleneck, conv_small, conv_light, conv_heavy).

**Observations:**

The blueprints follow a consistent residual pattern `x + f(x)` with identity-at-birth initialization, which is correct for gradual blending. The GroupNorm adaptation (`get_num_groups`) is mathematically sound and follows Wu & He (2018) recommendations.

**Concerns:**

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| B3-CNN-01 | **P1** | L93-108 | **NormSeed param_estimate=100 is incorrect.** GroupNorm(num_groups, channels) has `2*channels` parameters (gamma+beta) plus 1 scale parameter. For 64 channels: `2*64 + 1 = 129`. The estimate of 100 is 23% off. While the actual count scales with channels, the estimate is fixed. This affects rent economy signals to the policy. |
| B3-CNN-02 | **P2** | L112-163 | **AttentionSeed param_estimate=2000 is significantly off.** For 64 channels with reduction=4: Linear(64,16)+Linear(16,64)+bias = `64*16 + 16*64 + 64 = 2112`. Close enough for small channels, but for 512 channels: `512*128 + 128*512 + 512 = 131,584` vs estimate 2000. The fixed estimate doesn't scale. |
| B3-CNN-03 | **P3** | L166-186 | **DepthwiseSeed param_estimate=4800** - For 64 channels: depthwise(64,64,3,3)=576 + pointwise(64,64)=4096 + GN~128 = 4800. Correct for dim=64, but the estimate is presented as fixed when it's actually `channels*9 + channels^2 + 2*channels`. |
| B3-CNN-04 | **P4** | L19-31 | NoopSeed stores `channels` parameter in constructor but doesn't use it. Minor - could remove the parameter. |
| B3-CNN-05 | **P3** | L104-106 | **Potential NaN from tanh:** While `tanh` is numerically stable, if `self.norm(x) - x` produces very large values (poorly conditioned input), the bound is correct but gradient flow through tanh could saturate. Consider documenting this edge case. |

**Positive:**
- SE attention initialization (bias=3.0 giving sigmoid~0.95) is thoughtful for identity-at-birth
- Zero-init final layer in bottleneck ensures identity start
- GroupNorm instead of BatchNorm avoids running stats drift issues

---

### 2. `/home/john/esper-lite/src/esper/kasmina/blueprints/initialization.py`

**Purpose:** Helper functions for identity-like seed initialization.

**Observations:**

This is a clean utility module with correct handling of the layer search and zero-initialization.

**Concerns:**

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| B3-INIT-01 | **P4** | L52-53 | The `getattr(layer, "bias", None)` is flagged in CLAUDE.md as potentially defensive, but here it's legitimate - some Conv/Linear layers have `bias=False`. The check is correct. |
| B3-INIT-02 | **P3** | L16-26 | `find_final_affine_layer` returns the last affine layer found via iteration. If a module has multiple paths (e.g., parallel branches), this returns an arbitrary "last" one. May not be the semantically correct layer in complex architectures. Document this limitation. |

**Positive:**
- Clean separation of concerns
- Explicit `allow_missing` flag for different use cases

---

### 3. `/home/john/esper-lite/src/esper/kasmina/blueprints/__init__.py`

**Purpose:** Package init that triggers blueprint registration and re-exports.

**Observations:**

Simple and correct. The `# noqa: F401` comments appropriately silence unused import warnings for side-effect imports.

**Concerns:**

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| B3-PKG-01 | **P4** | L11-12 | `from .cnn import ConvBlock` is exported but `SeedConvBlock` and `get_num_groups` are not in `__all__` despite being in `cnn.__all__`. Consider consistency. |

---

### 4. `/home/john/esper-lite/src/esper/kasmina/blueprints/registry.py`

**Purpose:** Plugin registry for seed blueprints with action cache invalidation.

**Observations:**

The registry pattern is well-implemented. The cache invalidation for leyline action enums is a nice touch for dynamic blueprint registration (tests).

**Concerns:**

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| B3-REG-01 | **P3** | L47-50 | `actual_param_count(dim)` instantiates a full module just to count parameters. For large blueprints (transformer MLP with dim=4096), this allocates significant memory. Consider caching or lazy evaluation. |
| B3-REG-02 | **P4** | L56-57 | Class-level `_blueprints` dict is mutable class state. Thread-safe for reads but not for concurrent writes. Tests that register/unregister blueprints should not run in parallel. Document this. |
| B3-REG-03 | **P3** | L103-118 | The `create()` method passes `**kwargs` to factory but the factory signature varies (some take `reduction`, some take `rank`, some take `checkpoint`). This is flexible but makes it hard to validate kwargs at registration time. Type safety gap. |

**Positive:**
- Clear separation between spec (metadata) and factory (instantiation)
- Sorted listing by param_estimate is useful for action space ordering

---

### 5. `/home/john/esper-lite/src/esper/kasmina/blueprints/transformer.py`

**Purpose:** Transformer seed blueprints (norm, lora, lora_large, attention, mlp_small, mlp, flex_attention, noop).

**Observations:**

The transformer blueprints are well-designed with appropriate identity-at-birth initialization (zero-init on output projections).

**Concerns:**

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| B3-TFM-01 | **P2** | L238 | **FlexAttention block mask cache unbounded.** `_MAX_CACHE_SIZE = 8` is the LRU bound, but the cache uses `OrderedDict` which in long inference with varying sequence lengths could accumulate entries up to this limit. The `_apply()` hook clears on device transfer, but not on dtype changes via autocast. Consider clearing on dtype changes too. |
| B3-TFM-02 | **P3** | L265-275 | The `causal` mask function is defined inline inside `_get_causal_block_mask`. This creates a new function object on each cache miss. While `create_block_mask` likely doesn't capture this closure, confirm there's no unintended reference retention. |
| B3-TFM-03 | **P1** | L289-292 | **Potential bug in flex_attention tuple handling.** The code checks `if isinstance(attn_out, tuple)` and extracts `attn_out[0]`. The comment says "flex_attention returns Tensor | tuple" but the official API returns just Tensor. If this is for a specific PyTorch version or custom attention, document it. If not, this dead code path may hide future bugs if the API changes. |
| B3-TFM-04 | **P4** | L125-158 | `mlp_small` and `mlp` are nearly identical except for expansion factor. Consider a shared factory with `expansion` parameter rather than duplicated classes. |
| B3-TFM-05 | **P3** | L24-38 | **TransformerNormSeed param_estimate=800 is incorrect.** LayerNorm(dim) has `2*dim` params. For dim=384: `2*384 + 1(scale) = 769`. The estimate of 800 is reasonable for typical dims but again is fixed vs. scaling. |

**Positive:**
- FlexAttention fallback to SDPA maintains consistent action space across PyTorch versions
- Activation checkpointing option for MLP blueprints (memory-conscious)
- LINEAR amplitude ramp for gated blending to avoid "double dynamics"

---

### 6. `/home/john/esper-lite/src/esper/kasmina/isolation.py`

**Purpose:** Gradient isolation and health monitoring for G2 gate decisions.

**Observations:**

The gradient isolation is correctly implemented via structural `detach()` at the seed input boundary, not via runtime threshold detection. The documentation clearly explains this, which is excellent.

**Concerns:**

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| B3-ISO-01 | **P2** | L69-86 | **ste_forward dtype handling.** The function casts `seed_features.to(host_features.dtype)` but this creates a copy if dtypes differ. Under BF16 autocast, this could happen frequently. The cast is correct for correctness but adds allocation overhead. Consider documenting this as expected behavior. |
| B3-ISO-02 | **P3** | L172 | **Private API usage:** `torch._foreach_norm` is used with a note that it's "stable since PyTorch 1.9". While true, it's still underscore-prefixed. Add a version guard or catch `AttributeError` for future-proofing. |
| B3-ISO-03 | **P4** | L213-225 | The `materialize_gradient_stats` function has branches for `host_sq is None`, `isinstance(host_sq, torch.Tensor)`, and else. The else branch (`host_sq ** 0.5`) implies `host_sq` could be a float, but the async function only returns `torch.Tensor | None`. This else is dead code unless the dict is mutated elsewhere. |

**Positive:**
- Excellent docstrings explaining gradient flow diagrams
- Async/sync gradient computation pattern for PPO hot path
- BF16 compatibility via explicit dtype casting

---

### 7. `/home/john/esper-lite/src/esper/kasmina/slot.py`

**Purpose:** SeedSlot lifecycle management - the core component managing seed germination, training, blending, and fossilization.

**Observations:**

This is a substantial file (~2600 lines) that manages the complete seed lifecycle. The implementation is thorough with good documentation of DDP requirements and torch.compile considerations.

**Concerns:**

| ID | Severity | Location | Issue |
|----|----------|----------|-------|
| B3-SLOT-01 | **P2** | L1798-1827 | **Gradient ratio normalization may disadvantage small seeds.** The formula `(seed_norm / host_norm) * sqrt(host_params / seed_params)` scales up small seeds and down large ones. However, this assumes equal gradient magnitude per parameter, which isn't true for different architectures. A small attention seed may have naturally higher per-param gradients than a large MLP. This could cause the G2 gate to prefer certain architectures over others for non-fundamental reasons. |
| B3-SLOT-02 | **P2** | L1939-1946 | **Memory format handling in isolation.** When `isolate_gradients=True` and input is channels_last, the code calls `.contiguous().detach()` which creates a contiguous copy. This is correct for BUG-005 avoidance but adds memory overhead for channels_last hosts. The comment explains this, but consider measuring the overhead. |
| B3-SLOT-03 | **P3** | L1958-1963 | **STE forward requires_grad assertion.** The `_DEBUG_STE` assertion checks `seed_features.requires_grad` but this flag is env-var controlled. In production, if seed parameters are accidentally frozen, STE would silently produce zero gradients. Consider a one-time warning instead of assertion. |
| B3-SLOT-04 | **P3** | L2537 | **isolate_gradients default in set_extra_state.** Defaults to `False` which is correct for transformers but not CNNs. If a CNN checkpoint is loaded and this field is missing, gradient isolation will be wrong. However, `set_extra_state` should only receive dicts from `get_extra_state`, so this is low risk. |
| B3-SLOT-05 | **P2** | L2191-2256 | **DDP gate sync broadcasts Python objects.** `torch.distributed.broadcast_object_list` pickles objects, which is slower than tensor broadcasts. For hot path gate checks, this adds latency. Consider using tensor-based synchronization for the critical `passed` boolean. |
| B3-SLOT-06 | **P4** | L374-400 | **SeedState slots=True with deque.** The `stage_history` is a `deque[tuple[SeedStage, datetime]]` which has a `maxlen`. This is correct, but `deque` with `maxlen` in a `slots=True` dataclass may have unexpected memory behavior. Test that checkpoint round-trips preserve maxlen. |
| B3-SLOT-07 | **P3** | L672-688 | **QualityGates _check_gate uses match-case.** The match statement dispatches to gate checks. If a new gate level is added to leyline but not handled here, the default case returns `passed=True`. This could accidentally advance seeds. Consider explicit `case _: raise ValueError(...)` for unknown gates. |
| B3-SLOT-08 | **P4** | L1029 | **Shape probe cache key includes topology but not dtype.** If the same slot is used with different dtypes (e.g., fp32 during debug, bf16 during training), the cache may return wrong-dtype probes. The probe is used in eval mode so this is likely fine, but dtype-sensitive. |
| B3-SLOT-09 | **P3** | L143-149 | **SeedMetrics alpha semantics note.** The comment correctly notes that `current_alpha` represents blending progress, not actual per-sample alpha for gated blends. This is important for RL observation construction but could be a subtle source of confusion. Consider renaming to `blending_progress` or `alpha_schedule_progress`. |
| B3-SLOT-10 | **P1** | L361-.get usage in from_dict | **Defensive .get() in from_dict.** Lines 361-365 use `.get()` with defaults for `counterfactual_contribution`, `_prev_contribution`, `contribution_velocity`. The first two being None is legitimate (optional fields), but `contribution_velocity` defaulting to 0.0 silently masks missing data. This is the "optional fields" case from CLAUDE.md, but review whether these should fail loudly on missing. |

**Positive:**
- Comprehensive DDP safety documentation and implementation
- Clear separation of concerns: AlphaController handles timing, BlendCatalog handles per-sample gating
- Good epoch-based lifecycle with explicit gate checks
- Proper cleanup on prune (cache clearing, monitor reset)
- Telemetry integration with lifecycle-only mode for performance

---

## Cross-Cutting Integration Risks

### 1. Parameter Estimate Accuracy for Rent Economy (P1)

**Risk:** The `param_estimate` values in blueprints are fixed constants that don't scale with the `dim` argument. Since the rent economy uses parameter counts as a cost signal, inaccurate estimates could:
- Mislead the policy about the true cost of architectural choices
- Create reward hacking opportunities where seeds with underestimated param counts appear cheaper
- Cause inconsistent rent penalty computation between estimated and actual costs

**Evidence:**
- CNN norm: estimate=100, actual(64)=129 (29% error)
- CNN attention: estimate=2000, actual(512)=131584 (65x error at high dim)

**Recommendation:** Either:
1. Make `param_estimate` a function of `dim` rather than a constant, OR
2. Always use `actual_param_count()` in reward computation (with caching to avoid allocation overhead)

### 2. Observation Space Consistency (P2)

**Risk:** The feature extraction in `tamiyo/policy/features.py` reads slot state directly from observations. Several fields have `.get()` with defaults:
- `alpha_mode` defaults to `AlphaMode.HOLD.value`
- `alpha_algorithm` defaults to `AlphaAlgorithm.ADD` (min value)
- Interaction metrics default to 0.0

If slots are in unexpected states (partial initialization, checkpoint corruption), the observation vector may silently use defaults rather than fail fast.

**Recommendation:** Add validation that active slots have all required fields, or use a typed dataclass for slot observations rather than raw dicts.

### 3. FlexAttention Action Space Stability (P2)

**Risk:** FlexAttention is always registered (even on PyTorch < 2.5) with SDPA fallback. This maintains consistent action space but creates a subtle issue: the learned gate may develop different behaviors on systems with vs. without FlexAttention, since SDPA vs. FlexAttention have different numerical characteristics.

**Recommendation:** Document this as a known reproducibility consideration. Consider adding a test that verifies output equivalence between FlexAttention and SDPA fallback.

### 4. DDP Safety Gap in set_alpha_target (P2)

**Risk:** `set_alpha_target()` and `schedule_prune()` mutate controller state and can trigger stage transitions (HOLDING -> BLENDING). These methods don't have `_sync_gate_decision()` calls like `advance_stage()`. If called asymmetrically across ranks, architecture divergence could occur.

**Evidence:** Only `advance_stage()` calls `_sync_gate_decision()`. Other lifecycle-mutating methods assume symmetric calls.

**Recommendation:** Either:
1. Add DDP sync to `set_alpha_target()` and `schedule_prune()`, OR
2. Document that these must be called symmetrically and add assertions when DDP is initialized

---

## Severity-Tagged Findings Summary

### P0 (Critical) - None identified

### P1 (Correctness)
| ID | File | Issue |
|----|------|-------|
| B3-CNN-01 | cnn.py | NormSeed param_estimate=100 incorrect (~29% error) |
| B3-CNN-02 | cnn.py | AttentionSeed param_estimate=2000 doesn't scale (65x error at high dim) |
| B3-TFM-03 | transformer.py | flex_attention tuple handling may be dead code |
| B3-SLOT-10 | slot.py | from_dict .get() defaults may mask checkpoint corruption |

### P2 (Performance/Safety)
| ID | File | Issue |
|----|------|-------|
| B3-TFM-01 | transformer.py | FlexAttention cache may accumulate with varying seq_len |
| B3-ISO-01 | isolation.py | ste_forward dtype cast allocates under autocast |
| B3-SLOT-01 | slot.py | Gradient ratio normalization may disadvantage small seeds |
| B3-SLOT-02 | slot.py | Memory format handling adds overhead for channels_last |
| B3-SLOT-05 | slot.py | DDP gate sync uses slow pickle-based broadcast |
| Cross-cut | Multiple | Observation space consistency with defaults |
| Cross-cut | Multiple | DDP safety gap in set_alpha_target/schedule_prune |

### P3 (Code Quality)
| ID | File | Issue |
|----|------|-------|
| B3-CNN-03 | cnn.py | DepthwiseSeed param_estimate fixed vs scaling |
| B3-CNN-05 | cnn.py | tanh saturation edge case undocumented |
| B3-INIT-02 | initialization.py | find_final_affine_layer arbitrary in multi-path |
| B3-REG-01 | registry.py | actual_param_count allocates full module |
| B3-REG-03 | registry.py | Factory kwargs not validated at registration |
| B3-ISO-02 | isolation.py | torch._foreach_norm is private API |
| B3-TFM-02 | transformer.py | Inline causal function may retain references |
| B3-TFM-05 | transformer.py | TransformerNormSeed param_estimate=800 fixed |
| B3-SLOT-03 | slot.py | STE assertion only in debug mode |
| B3-SLOT-04 | slot.py | isolate_gradients default may be wrong for CNNs |
| B3-SLOT-07 | slot.py | match-case default passes unknown gates |
| B3-SLOT-09 | slot.py | current_alpha naming ambiguous with gated blends |

### P4 (Style/Minor)
| ID | File | Issue |
|----|------|-------|
| B3-CNN-04 | cnn.py | NoopSeed unused channels parameter |
| B3-PKG-01 | __init__.py | __all__ inconsistent with cnn.__all__ |
| B3-INIT-01 | initialization.py | Legitimate getattr for optional bias (no change) |
| B3-REG-02 | registry.py | Class-level mutable state thread safety |
| B3-TFM-04 | transformer.py | mlp_small/mlp code duplication |
| B3-ISO-03 | isolation.py | Dead else branch in materialize_gradient_stats |
| B3-SLOT-06 | slot.py | deque with slots=True maxlen behavior |
| B3-SLOT-08 | slot.py | Shape probe cache not dtype-keyed |

---

## Recommendations

### High Priority

1. **Fix param_estimate accuracy** - Either make estimates scale with dim or use actual counts in rent economy. This directly affects policy learning signals.

2. **Add DDP sync to set_alpha_target/schedule_prune** - These methods can cause stage transitions without rank synchronization.

3. **Review flex_attention tuple handling** - Confirm whether this is needed or dead code.

### Medium Priority

4. **Add memory overhead metrics** for channels_last + isolation, FlexAttention cache, and dtype casts under autocast.

5. **Consider tensor-based DDP sync** for gate decisions to reduce pickle overhead.

6. **Add explicit unknown gate handling** in QualityGates to fail fast on new gate levels.

### Low Priority

7. Clean up code duplication (mlp_small/mlp, NoopSeed channels).

8. Document edge cases (tanh saturation, multi-path final layer selection).

9. Consider renaming `current_alpha` to `blending_progress` for clarity.

---

## Test Coverage Assessment

The test files in `tests/kasmina/` provide good coverage:
- `test_blueprints.py` - Basic blueprint creation and identity tests
- `test_blueprint_edge_cases.py` - Comprehensive edge case testing (tiny/large channels, mixed precision, shape preservation, gradient flow)
- `test_blueprint_registry.py` - Registry operations
- `test_seed_slot.py` - Lifecycle management
- `test_g2_gradient_readiness.py` - Gradient gate testing

**Missing Test Coverage:**
- FlexAttention vs SDPA fallback numerical equivalence
- param_estimate vs actual_param_count comparison across all blueprints/dims
- DDP gate sync under rank divergence conditions
- Checkpoint round-trip with deque maxlen preservation
- Gradient ratio normalization behavior with different architecture combinations

---

## Conclusion

The Kasmina blueprints and slot system is fundamentally sound with good architectural decisions. The main concerns are:

1. **Rent economy signal accuracy** - Parameter estimates don't scale with dim, potentially misleading the policy about true costs.

2. **DDP safety gaps** - Some lifecycle-mutating methods lack rank synchronization.

3. **Minor performance overhead** - dtype casts, channels_last handling, and cache behavior under edge cases.

The code quality is high with excellent documentation, especially around gradient isolation semantics and DDP requirements. The identity-at-birth initialization pattern is consistently applied across blueprints, which is crucial for stable training.
