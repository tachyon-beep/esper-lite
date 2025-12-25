---
id: "kasmina"
title: "Kasmina - Seed Lifecycle Management"
aliases: ["seed-slots", "morphogenetic-model", "grafting"]
biological_role: "Stem Cell"
layer: "Core Logic"
criticality: "Tier-0"
tech_stack:
  - "Python 3.11+"
  - "PyTorch 2.x"
primary_device: "cuda"
thread_safety: "unsafe"
owners: "core-team"
compliance_tags:
  - "GPU-Resident"
  - "Gradient-Aware"
  - "Checkpoint-Critical"
schema_version: "1.0"
last_updated: "2025-12-14"
last_reviewed_commit: "db3b9c1"
---

# Kasmina Bible

# 1. Prime Directive

**Role:** Manages seed module lifecycle from germination through fossilization, providing gradient-safe integration of dynamically-grown neural network components into host networks.

**Anti-Scope:** Does NOT make strategic decisions about WHEN to germinate, advance, or cull seeds—that's Tamiyo's job. Does NOT run training loops—that's Tolaria's job. Does NOT compute rewards—that's Simic's job.

---

# 2. Interface Contract

## 2.1 Entry Points (Public API)

### Module: `kasmina.slot`

#### `SeedSlot(slot_id, channels, device, gates, on_telemetry, fast_mode, task_config)`
> Manages a single seed through its lifecycle with quality gates.

- **Invariants:**
  - `channels` must match host injection point dimension
  - One seed per slot at a time (except FOSSILIZED which is permanent)
- **Thread Safety:** NOT thread-safe. Do not share across threads.

**Key Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `germinate()` | `(blueprint_id, seed_id, host_module, blend_algorithm_id) -> SeedState` | Create and validate new seed |
| `advance_stage()` | `(target_stage) -> GateResult` | Progress through lifecycle |
| `cull()` | `(reason) -> bool` | Remove non-fossilized seed |
| `forward()` | `(host_features) -> Tensor` | Apply seed transformation |
| `capture_gradient_telemetry()` | `() -> None` | Update G2 gate metrics |
| `step_epoch()` | `() -> None` | Mechanical lifecycle advancement |

#### `SeedState`
> Complete state of a seed through its lifecycle.

**Key Fields:**
- `seed_id`, `blueprint_id`, `slot_id`: Identity
- `stage: SeedStage`: Current lifecycle stage
- `alpha: float`: Blending weight [0.0, 1.0]
- `metrics: SeedMetrics`: Accuracy, gradient stats
- `telemetry: SeedTelemetry`: Observability data

#### `QualityGates`
> Gate checks for stage transitions.

| Gate | Target Stage | Requirements |
|------|--------------|--------------|
| G0 | GERMINATED | seed_id and blueprint_id present |
| G1 | TRAINING | Currently GERMINATED |
| G2 | BLENDING | Global improvement + seed ready + gradient active |
| G3 | HOLDING | Blending complete + alpha ≥ 0.95 |
| G5 | FOSSILIZED | Counterfactual contribution ≥ 1% |

### Module: `kasmina.host`

#### `MorphogeneticModel(host, device, slots, task_config, fast_mode)`
> Wrapper managing multiple SeedSlots attached to a host network.

**Key Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `germinate_seed()` | `(blueprint_id, seed_id, slot, blend_algorithm_id)` | Germinate in specific slot |
| `cull_seed()` | `(slot)` | Cull seed in slot (transitions to PRUNED) |
| `get_seed_parameters()` | `(slot) -> Iterator[Parameter]` | Get seed params |
| `get_host_parameters()` | `() -> Iterator[Parameter]` | Get host-only params |

#### `CNNHost(num_classes, n_blocks, base_channels, pool_layers, memory_format)`
> CNN host with injection points after each block (except first).

#### `TransformerHost(vocab_size, n_embd, n_head, n_layer, block_size, dropout)`
> GPT-style decoder with injection points after each layer.

### Module: `kasmina.isolation`

#### `blend_with_isolation(host_features, seed_features, alpha) -> Tensor`
> Blend using `torch.lerp` with proper gradient attribution.

- **Gradient Flow:** `d_output/d_host = (1-α)`, `d_output/d_seed = α`

#### `ste_forward(host_features, seed_features) -> Tensor`
> Straight-Through Estimator: forward returns host, backward flows to both.

#### `GradientIsolationMonitor`
> Monitors gradient flow for isolation verification.

| Method | Description |
|--------|-------------|
| `register(host, seed)` | Register modules for monitoring |
| `check_isolation_async()` | Return tensor norms (no CPU sync) |
| `materialize_isolation_stats()` | Extract final values (requires sync) |

### Module: `kasmina.blending`

#### `BlendCatalog.create(algorithm_id, **kwargs) -> BlendAlgorithm`
> Factory for blending algorithms.

| Algorithm | Behavior |
|-----------|----------|
| `linear` | Linear ramp 0→1 over steps |
| `sigmoid` | Smooth S-curve transition |
| `gated` | Learned per-sample alpha from features |

## 2.2 Configuration

```python
@dataclass
class QualityGates:
    min_training_improvement: float = 0.5    # % improvement for G2
    min_blending_epochs: int = 3             # Epochs before blending allowed
    max_isolation_violations: int = 10       # Violations before unhealthy
    min_holding_stability: float = 0.95      # Stability threshold
    min_seed_gradient_ratio: float = 0.05    # G2 gradient activity check
```

## 2.3 Events (Emitted via Karn)

| Event | Trigger | Key Payload Fields |
|-------|---------|-------------------|
| `SEED_GERMINATED` | New seed created | `blueprint_id`, `seed_id`, `params` |
| `SEED_STAGE_CHANGED` | Lifecycle transition | `from`, `to` |
| `SEED_FOSSILIZED` | Permanent integration | `improvement`, `counterfactual`, `params_added` |
| `SEED_PRUNED` | Seed removed | `reason`, `counterfactual`, `epochs_total` |

---

# 3. Tensor Contracts

## 3.1 Input Tensors

| Name | Shape | Dtype | Device | Description |
|------|-------|-------|--------|-------------|
| `host_features` (CNN) | `[B, C, H, W]` | float32 | slot.device | Features from host block |
| `host_features` (Transformer) | `[B, T, n_embd]` | float32 | slot.device | Hidden states from layer |
| `x` (CNNHost input) | `[B, 3, H, W]` | float32 | host.device | Raw image input |
| `x` (TransformerHost input) | `[B, T]` | int64 | host.device | Token indices |

## 3.2 Output Tensors

| Name | Shape | Dtype | Device | Description |
|------|-------|-------|--------|-------------|
| `blended_features` | Same as input | float32 | slot.device | After seed transformation |
| `logits` (CNN) | `[B, num_classes]` | float32 | host.device | Classification output |
| `logits` (Transformer) | `[B, T, vocab_size]` | float32 | host.device | Language model output |

## 3.3 Internal Buffers

| Name | Location | Lifetime | Purpose |
|------|----------|----------|---------|
| `_shape_probe_cache` | SeedSlot | Slot lifetime | Cached shape validation tensors |
| `_host_params` | GradientIsolationMonitor | Until reset() | Parameter references for norm computation |
| `stage_history` | SeedState | Bounded (maxlen=100) | Rolling window of stage transitions |

## 3.4 Gradient Flow

```
Loss
  │
  ▼
Output = lerp(host, seed, α)
  │                    │
  │ (1-α) gradient     │ α gradient
  ▼                    ▼
Host Features         Seed Features
  ▲                    │
  │ (BLOCKED if        │ (always flows)
  │  isolate_gradients │
  │  at seed INPUT)    │
  └────────────────────┘

TRAINING stage (α=0): Uses STE pattern
  forward:  host + (seed - seed.detach()) == host
  backward: gradients flow to BOTH host and seed

BLENDING+ stages: Standard lerp gradient attribution
  CNNs: isolate_gradients=True (seed path blocked)
  Transformers: isolate_gradients=False (co-adaptation allowed)
```

**PyTorch Expert Insight:** The STE pattern at `isolation.py:63-72` is the canonical implementation. The `.detach()` creates a gradient bridge—mathematically the `(seed - seed.detach())` term evaluates to zero in forward, but in backward the non-detached `seed` retains its computation graph.

---

# 4. Operational Physics

## 4.1 State Machine

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    ▼                                             │
[DORMANT] ─G0─> [GERMINATED] ─G1─> [TRAINING] ─G2─> [BLENDING]   │
                                        │              │          │
                                        │              │ G3       │
                                        │              ▼          │
                                        │         [HOLDING]       │
                                        │              │          │
                                        │         ┌────┴────┐     │
                                        │         │         │     │
                                        │       G5│      timeout  │
                                        │         ▼     or neg    │
                                        │   [FOSSILIZED] contrib  │
                                        │    (permanent) │        │
                                        │                ▼        │
                                        └──────────> [PRUNED] <───┘
```

**Gate Descriptions:**
- **G0:** Sanity check (identity fields present)
- **G1:** Germination complete
- **G2:** Global improvement + seed training duration + gradient activity
- **G3:** Blending duration + alpha reached target
- **G5:** Counterfactual contribution ≥ 1% (requires HOLDING validation)

**Key Invariants:**
- FOSSILIZED is permanent—cannot be pruned (only future pruning system)
- Negative counterfactual in HOLDING → automatic prune (safety)
- Holding timeout → automatic prune (Tamiyo failed to decide)

## 4.2 Data Governance

### Authoritative (Source of Truth)
- `SeedState`: Canonical lifecycle state for a seed
- `SeedMetrics`: Accuracy tracking, gradient stats, alpha progress
- `stage_history`: Audit trail of transitions (bounded)

### Ephemeral (Cached/Temporary)
- `_shape_probe_cache`: Cleared on device transfer
- Isolation monitor parameter lists: Cleared on reset/cull

### Read-Only (Consumed)
- `task_config`: From Simic via training script
- Quality gate thresholds: Configuration, not modified at runtime

## 4.3 Concurrency Model

- **Thread Safety:** UNSAFE—SeedSlot mutates instance state
- **DDP Pattern:** `_sync_gate_decision()` uses `all_reduce(MIN)` for unanimous consensus
- **Async Gradient Telemetry:** `check_isolation_async()` returns tensors, `materialize_isolation_stats()` performs sync
- **CUDA Streams:** No explicit stream management in Kasmina (handled by PyTorch autograd)

**DDP Critical Requirement:** All ranks MUST call `_sync_gate_decision` in identical order. Stage divergence between ranks causes deadlock.

## 4.4 Memory Lifecycle

- **Seed Allocation:** On `germinate()`, blueprint instantiated and moved to device
- **Shape Probe:** Lazy allocation, cached per topology, cleared on device change
- **Parameter Lists:** Held by `GradientIsolationMonitor`, cleared on `reset()`
- **Cull Cleanup:** Seed set to None, state cleared, monitor reset

---

# 5. Dependencies

## 5.1 Upstream (Modules that call Kasmina)

| Module | Interaction | Failure Impact |
|--------|-------------|----------------|
| `tolaria.Trainer` | Calls `step_epoch()`, `capture_gradient_telemetry()` | Seeds stuck in wrong stage |
| `simic.VectorizedEnv` | Creates `MorphogeneticModel`, calls `germinate_seed()` | Cannot create environments |
| `tamiyo.HeuristicController` | Calls `advance_stage()`, `cull()` | No lifecycle progression |

## 5.2 Downstream (Modules Kasmina depends on)

| Module | Interaction | Failure Handling |
|--------|-------------|------------------|
| `leyline` | Uses `SeedStage`, `GateLevel`, `TelemetryEvent` | **Fatal**—cannot function |
| `kasmina.blueprints` | Creates seed modules | Blueprint not found error |
| `karn` | Emits telemetry events | **Graceful**—continues without metrics |

## 5.3 External Dependencies

| Package | Version | Purpose | Fallback |
|---------|---------|---------|----------|
| `torch` | >=2.0 | Core tensor ops, `torch.lerp`, `torch._foreach_norm` | None (required) |
| `torch.distributed` | (optional) | DDP consensus for gate decisions | Single-rank mode |

---

# 6. Esper Integration

## 6.1 Commandment Compliance

| # | Commandment | Status | Notes |
|---|-------------|--------|-------|
| 1 | Sensors match capabilities | ✅ | Emits stage/germinate/prune events |
| 2 | Complexity pays rent | ✅ | Tracks `active_seed_params` |
| 3 | GPU-first iteration | ✅ | All ops on CUDA, async gradient capture |
| 4 | Progressive curriculum | N/A | Curriculum is Tamiyo's domain |
| 5 | Train Anything protocol | ✅ | `HostProtocol` enables pluggable hosts |
| 6 | Morphogenetic plane | ✅ | **Primary implementer**—SeedSlot IS the morphogenetic plane |
| 7 | Governor prevents catastrophe | ⚠️ | Kasmina has gates, but Governor is Tolaria |
| 8 | Hierarchical scaling | ✅ | Multi-slot architecture (early/mid/late) |
| 9 | Frozen Core economy | N/A | Future consideration |

## 6.2 Biological Role

**Analogy:** Stem Cell / Morphogenetic Field

Kasmina is the biological machinery for growing new neural tissue. Just as stem cells differentiate into specialized tissue, Kasmina's seeds start undifferentiated (DORMANT) and mature through stages until permanently integrated (FOSSILIZED) or removed (PRUNED).

**Responsibilities in the organism:**
- Maintain the "morphogenetic plane" where growth occurs
- Ensure gradient isolation so new growth doesn't destabilize existing tissue
- Gate each developmental transition to prevent malformed growth

**Interaction with other organs:**
- Receives signals from: Tamiyo (lifecycle decisions), Simic (environment creation)
- Sends signals to: Karn (telemetry), Tolaria (via step_epoch integration)

## 6.3 CLI Integration

| Command | Flags | Effect on Kasmina |
|---------|-------|-------------------|
| `esper ppo` | `--slots early mid late` | Configures which SeedSlots are created |
| `esper ppo` | `--max-seeds N` | Limits total seeds across all slots |
| `esper ppo` | `--blend-algorithm X` | Sets blend algorithm (linear/sigmoid/gated) |

---

# 7. Cross-References

## 7.1 Related Bibles

| Bible | Relationship | Integration Point |
|-------|--------------|-------------------|
| [leyline](leyline.md) | **Implements** | Uses `SeedStage`, `GateLevel`, telemetry types |
| [simic](simic.md) | **Consumed by** | Simic creates MorphogeneticModel for envs |
| [tolaria](tolaria.md) | **Called by** | Tolaria calls `step_epoch()` in training loop |
| [tamiyo](tamiyo.md) | **Controlled by** | Tamiyo makes germinate/advance/cull decisions |
| [karn](karn.md) | **Feeds** | Emits lifecycle telemetry events |

## 7.2 Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/esper/kasmina/__init__.py` | 66 | Public exports, re-exports Leyline types |
| `src/esper/kasmina/slot.py` | 1462 | SeedSlot, SeedState, QualityGates |
| `src/esper/kasmina/host.py` | 668 | CNNHost, TransformerHost, MorphogeneticModel |
| `src/esper/kasmina/isolation.py` | 201 | blend_with_isolation, STE, GradientIsolationMonitor |
| `src/esper/kasmina/blending.py` | 206 | BlendAlgorithm, GatedBlend, BlendCatalog |
| `src/esper/kasmina/protocol.py` | 40 | HostProtocol (structural typing) |
| `src/esper/kasmina/blueprints/` | ~300 | BlueprintRegistry, CNN/Transformer blueprints |

## 7.3 Test Coverage

| Test File | Focus |
|-----------|-------|
| `tests/test_kasmina_slot.py` | SeedSlot lifecycle, gates, transitions |
| `tests/test_kasmina_host.py` | Host networks, injection points |
| `tests/test_kasmina_isolation.py` | Gradient isolation, STE correctness |

---

# 8. Tribal Knowledge

## 8.1 Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| One seed per slot at a time | Can't have multiple concurrent seeds in same slot | Use multiple slots (early/mid/late) |
| `force_alpha()` not DDP-safe | Counterfactual evaluation breaks with DataParallel | Use `model.eval()` + single-threaded validation |
| GatedBlend not in `state_dict()` | GatedBlend params only saved via `get_extra_state()` | Ensure checkpoint code handles extra_state |
| FOSSILIZED cannot be undone | Permanent—no way to remove fossilized seed | Make HOLDING decision carefully |

## 8.2 Performance Cliffs

| Operation | Trigger | Why It Happens | Mitigation |
|-----------|---------|----------------|------------|
| `.item()` in CUDA stream | Calling `check_isolation()` during async work | Forces CPU-GPU sync, breaks pipeline | Use `check_isolation_async()` + deferred materialization |
| Shape probe allocation | First germinate per topology | Creates validation tensor | Pre-warm by calling germinate in setup |
| Isinstance check in `blend_with_isolation` | Alpha as tensor vs scalar | Causes Dynamo guard and graph specialization | Accept minor overhead—specializes once per config |
| Memory format conversion | Every forward if not already `channels_last` | `to(memory_format=...)` called repeatedly | Input already in correct format from host |

**PyTorch Expert Insight (isolation.py:126):** `torch._foreach_norm` is a private API but stable since PyTorch 1.9. It launches O(1) CUDA kernels instead of O(n_params). If removed in future PyTorch, fallback to `torch.stack([p.norm() for p in grads])`.

## 8.3 Common Pitfalls

| Pitfall | Why It Happens | Correct Approach |
|---------|----------------|------------------|
| Not calling `capture_gradient_telemetry()` | G2 gate requires gradient ratio | Call after `loss.backward()` in Tolaria |
| Calling `germinate()` without `host_module` | Gradient normalization uses host param count | Always pass host_module for accurate G2 gate |
| Using `force_alpha()` with DDP | Mutates instance state, causes rank divergence | Use explicit alpha parameter or counterfactual method |
| Expecting `advance_stage()` to auto-progress | Some transitions need explicit target | Check current stage, provide target if needed |
| Checking `blending_delta` for attribution | Conflates host drift with seed impact | Use `counterfactual_contribution` for causal attribution |

**PyTorch Expert Insight (slot.py:1134-1139):** When `GatedBlend` computes alpha from `host_features`, there's an implicit gradient path through the gate network. Even with `isolate_gradients=True`, host receives minor gradients through gated alpha. This is expected behavior—gate network is small (~1% of gradient magnitude).

## 8.4 DDP-Specific Gotchas

| Issue | Location | Why It's Dangerous | Mitigation |
|-------|----------|-------------------|------------|
| Gate decision divergence | `slot.py:1200-1244` | Ranks calling different number of gates → deadlock | Ensure all ranks have identical seed lifecycles |
| Stage desync between ranks | Any gate check | Parameter shape mismatch on next forward → crash | Use `_sync_gate_decision()` for all transitions |
| `force_alpha()` under DDP | `slot.py:693-740` | Instance mutation causes different forward outputs | Add runtime guard or use alternative counterfactual method |

## 8.5 torch.compile Behavior

| Pattern | Location | Compile Behavior |
|---------|----------|------------------|
| Stage-dependent control flow | `slot.py:1094-1139` | Creates 6-8 specialized graphs (acceptable—stage changes are epoch-granular) |
| `isinstance(alpha, Tensor)` | `isolation.py:56` | Guard on alpha type, specializes once |
| `OrderedDict.move_to_end()` | `blueprints/transformer.py:166` | **Graph break**—documented in module docstring |

**PyTorch Expert Insight:** The stage-dependent branching in `forward()` is acceptable because stage transitions happen once per epoch. After warmup, execution stays in a single specialized graph. Do NOT use `@torch.compiler.disable`—it completely opts out, which is worse than specialization.

## 8.6 Debugging Tips

- **Symptom:** G2 gate always fails with `seed_gradient_low_0.00`
  - **Cause:** `capture_gradient_telemetry()` not being called
  - **Diagnostic:** Check if Tolaria calls it after `loss.backward()`
  - **Fix:** Add call to `slot.capture_gradient_telemetry()` after backward pass

- **Symptom:** Seeds stuck in GERMINATED, never reach TRAINING
  - **Cause:** `step_epoch()` not being called
  - **Diagnostic:** Check Tolaria's epoch loop
  - **Fix:** Ensure `step_epoch()` called once per epoch

- **Symptom:** DDP ranks deadlock during gate check
  - **Cause:** One rank has seed in different stage than others
  - **Diagnostic:** Add logging before `_sync_gate_decision()` calls
  - **Fix:** Ensure deterministic seed lifecycles across ranks

- **Symptom:** Checkpoint doesn't restore GatedBlend weights
  - **Cause:** GatedBlend is in `extra_state`, not `state_dict`
  - **Diagnostic:** Check if `get_extra_state()` being used in checkpoint code
  - **Fix:** Use `load_state_dict(strict=False)` + manually restore extra_state

---

# 9. Changelog

| Date | Change | Commit | Impact |
|------|--------|--------|--------|
| 2025-12-14 | Bible created with pytorch-expert analysis | `db3b9c1` | Initial documentation |
| 2025-12-10 | Topology-aware gradient isolation | - | CNNs isolate, Transformers co-adapt |
| 2025-12-09 | G2 gate gradient activity check | - | Prevents "free-rider" seeds |
| 2025-12-09 | Counterfactual-only G5 gate | - | No fallback to total_improvement |
| 2025-12-08 | Multi-slot architecture | - | early/mid/late segment support |
| 2025-11-30 | DDP unanimous consensus | - | `_sync_gate_decision()` added |
| 2025-11-25 | Initial SeedSlot implementation | - | Core lifecycle management |
