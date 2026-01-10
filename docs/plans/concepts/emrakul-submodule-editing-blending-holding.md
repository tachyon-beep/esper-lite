# Submodule Editing During BLENDING and HOLDING

**Status:** OPEN (ABANDONED and subsequently revived for consideration as an Emrakul feature)
**Author:** Claude (synthesizing PyTorch + DRL specialist reviews)
**Date:** 2026-01-09
**Abandoned:** 2026-01-09

## Abandonment Note

This proposal was superseded by **Track A+C Microstructured Ladders** after specialist review.

### Why This Approach Was Rejected

**PyTorch Engineering Concerns:**

1. **Tensor surgery complexity** — Downstream cascade repair (`_find_downstream_consumers()`) requires per-blueprint surgery logic for arbitrary module topologies
2. **Optimizer rebuild required** — Every SLIM_CHANNELS operation loses momentum/Adam state
3. **torch.compile impact** — Unbounded graph breaks vs. bounded specialization in ladder approach
4. **Non-reversible** — Pruned channels cannot be recovered

**DRL Learnability Concerns:**

1. **Irreversibility creates risk aversion** — Policy learns "when in doubt, don't touch successful modules"
2. **BLENDING-stage loss aversion trap** — Limiting to BLENDING/HOLDING actually *worsens* exploitation trap ("This finally works! Don't risk it!")
3. **Credit assignment still difficult** — 5-10 epoch horizon during complex blending dynamics
4. **Qualitatively different action type** — SLIM_CHANNELS is not homogeneous with existing lifecycle ops

### Recommended Alternative

Use **Track A+C** from `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md`:

- Microstructured ladder seeds with `GROW_INTERNAL`/`SHRINK_INTERNAL` ops
- Reversible capacity adjustment (GROW undoes SHRINK)
- No tensor surgery (toggle `requires_grad` on sub-blocks)
- 1-2 epoch credit assignment horizon
- Bounded torch.compile graph specialization (L≤4)

### What To Keep From This Proposal

- **Parameterized blueprints** (Tier 1) — `conv_heavy_slim`, `conv_heavy_tiny` variants at germination time
- **Fossilization custody semantics** — Tamiyo cannot modify fossilized modules (custody → Emrakul)
- **Immediate counterfactual reward** — Apply to ladder level changes instead of SLIM_CHANNELS

---

## Original Proposal (Archived)

## Executive Summary

This proposal enables Tamiyo to refine seed architecture during BLENDING and HOLDING stages—the "final shaping before permanent integration" window. Mutations are explicitly **forbidden** on FOSSILIZED modules, which aligns with the custody handoff to Emrakul.

The goal is to solve the "lumpiness problem": when Tamiyo has 2k params of budget remaining but `conv_heavy` costs 4.7k, she currently has no way to efficiently spend that budget.

## Design Principles

### 1. Mutations Are Pre-Integration Refinements

```
GERMINATED ──► TRAINING ──► BLENDING ──► HOLDING ──► FOSSILIZED
                                │            │            │
                          [MUTATE OK]  [MUTATE OK]  [CUSTODY → EMRAKUL]
                                │            │            │
                            Shaping      Final QA      Permanent
```

Rationale:

- **BLENDING**: Module is proving itself. If it's effective but oversized, now is the time to slim it down while alpha is still ramping.
- **HOLDING**: Final validation period. Last chance to optimize before permanent integration.
- **FOSSILIZED**: Custody transfers to Emrakul. Tamiyo can no longer touch it. This is analogous to "code freeze before release."

### 2. Two-Tier Approach: Parameterized Blueprints + Late-Stage Refinement

Rather than runtime surgery alone, we combine:

**Tier 1: Parameterized Blueprints (Primary)**

- Add size variants at germination time: `conv_heavy_full`, `conv_heavy_slim`, `conv_heavy_tiny`
- Tamiyo picks the right size upfront
- No credit assignment problem (immediate budget impact)

**Tier 2: Late-Stage Refinement (Secondary)**

- Channel pruning during BLENDING/HOLDING for fine-tuning
- Used when a module proved more effective than expected and can afford to shrink
- Limited action space: only `SLIM_CHANNELS` with discrete ratios [0.75, 0.85, 0.95]

### 3. No Kernel Size Changes Post-Germination

Per PyTorch specialist review:

- Kernel size changes require full layer rebuilds with reinitialized weights
- This effectively creates a new seed, defeating the purpose of refinement
- **Decision**: Kernel size is fixed at germination. Use parameterized blueprints for kernel variants.

## Detailed Design

### New Blueprint Variants

Extend `BlueprintAction` enum with size variants:

```python
class BlueprintAction(IntEnum):
    # Existing
    CONV_HEAVY = 7          # ~4.7k params (full)

    # New variants
    CONV_HEAVY_SLIM = 13    # ~3.5k params (75% channels)
    CONV_HEAVY_TINY = 14    # ~2.3k params (50% channels)

    CONV_LIGHT_SLIM = 15    # ~2.8k params (75% channels)
    # etc.
```

Implementation in `cnn.py`:

```python
@BlueprintRegistry.register("conv_heavy_slim", "cnn", param_estimate=3500)
def create_conv_heavy_slim_seed(dim: int, **kwargs: Any) -> nn.Module:
    """Conv heavy with 75% channel width."""
    return ConvHeavySeed(dim, channel_multiplier=0.75)


class ConvHeavySeed(nn.Module):
    def __init__(self, dim: int, channel_multiplier: float = 1.0):
        super().__init__()
        hidden = int(dim * channel_multiplier)
        self.block1 = SeedConvBlock(dim, hidden)
        self.block2 = SeedConvBlock(hidden, hidden)
        self.proj = nn.Conv2d(hidden, dim, 1)  # Project back to input dim
        # ... rest of initialization
```

### New LifecycleOp: SLIM_CHANNELS

Add to `factored_actions.py`:

```python
class LifecycleOp(IntEnum):
    NOOP = 0
    GERMINATE = 1
    PRUNE = 2
    FOSSILIZE = 3
    # New
    SLIM_CHANNELS = 4  # Only valid during BLENDING or HOLDING
```

### Slim Factor Action Head

Add a new action head for specifying the slim ratio:

```python
class SlimFactor(IntEnum):
    """Channel retention ratio for SLIM_CHANNELS operation."""
    RETAIN_95 = 0   # Keep 95% of channels (light trim)
    RETAIN_85 = 1   # Keep 85% of channels (moderate trim)
    RETAIN_75 = 2   # Keep 75% of channels (aggressive trim)
```

This keeps the action space discrete and bounded (3 options, not continuous).

### Action Masking

In `action_masks.py`, SLIM_CHANNELS is only valid when:

```python
def can_slim_channels(slot: SeedSlot) -> bool:
    """SLIM_CHANNELS is only valid during BLENDING or HOLDING."""
    if slot.stage not in (SeedStage.BLENDING, SeedStage.HOLDING):
        return False

    # Must have a seed with pruneable channels
    if slot.seed is None:
        return False

    # Don't allow slimming if already at minimum viable size
    if slot.seed_param_count < MIN_VIABLE_SEED_PARAMS:
        return False

    return True
```

### Module Surgery Implementation

New module in `kasmina/surgery.py`:

```python
class ModuleSurgeon:
    """Performs structural modifications on seed modules during BLENDING/HOLDING."""

    @staticmethod
    def slim_channels(
        seed: nn.Module,
        retention_ratio: float,
        importance_fn: Callable[[nn.Parameter], torch.Tensor] | None = None,
    ) -> tuple[nn.Module, int]:
        """Slim output channels across all conv layers in the seed.

        Args:
            seed: The seed module to modify
            retention_ratio: Fraction of channels to keep (0.75, 0.85, 0.95)
            importance_fn: Optional function to score channel importance.
                          If None, uses L1 norm of weights.

        Returns:
            (modified_seed, params_removed)

        Raises:
            SurgeryError: If module structure is unsupported
        """
        if importance_fn is None:
            importance_fn = lambda w: w.abs().sum(dim=(1, 2, 3))  # L1 norm per output channel

        params_before = sum(p.numel() for p in seed.parameters())

        # Find all Conv2d layers and their downstream consumers
        conv_layers = _find_conv_layers(seed)

        for layer_name, conv in conv_layers:
            n_keep = max(1, int(conv.out_channels * retention_ratio))

            # Score channels by importance
            scores = importance_fn(conv.weight)
            _, keep_indices = scores.topk(n_keep)
            keep_indices = keep_indices.sort().values

            # Rebuild the layer with fewer output channels
            new_conv = _rebuild_conv(conv, keep_indices)
            _set_submodule(seed, layer_name, new_conv)

            # Fix downstream layers that expect the old channel count
            downstream = _find_downstream_consumers(seed, layer_name, conv.out_channels)
            for ds_name, ds_layer in downstream:
                _fix_input_channels(ds_layer, keep_indices)

            # Fix normalization layers (GroupNorm, BatchNorm)
            norm_layers = _find_associated_norms(seed, layer_name)
            for norm_name, norm in norm_layers:
                _rebuild_norm(norm, n_keep)

        params_after = sum(p.numel() for p in seed.parameters())
        return seed, params_before - params_after
```

### Optimizer State Handling

After any SLIM_CHANNELS operation, the optimizer must be rebuilt:

```python
def apply_slim_channels(
    slot: SeedSlot,
    retention_ratio: float,
    optimizer: torch.optim.Optimizer,
    optimizer_cls: type,
    optimizer_kwargs: dict,
) -> torch.optim.Optimizer:
    """Apply channel slimming and rebuild optimizer.

    WARNING: This loses momentum/Adam state for the modified parameters.
    This is the expected cost of structural modification.
    """
    # Perform surgery
    slot.seed, params_removed = ModuleSurgeon.slim_channels(
        slot.seed, retention_ratio
    )

    # Rebuild optimizer (fresh state for all params)
    new_optimizer = optimizer_cls(
        slot.host.parameters(),  # Full model params
        **optimizer_kwargs
    )

    # Emit telemetry
    slot._emit_telemetry(TelemetryEventType.SEED_SLIMMED, SeedSlimmedPayload(
        retention_ratio=retention_ratio,
        params_removed=params_removed,
        new_param_count=slot.seed_param_count,
    ))

    return new_optimizer
```

### torch.compile Considerations

SLIM_CHANNELS must happen at epoch boundaries (already the pattern for stage transitions):

```python
def step_epoch(self, metrics: EpochMetrics) -> None:
    """Advance epoch. All structural changes happen here."""
    # ... existing stage transition logic ...

    # SLIM_CHANNELS would be executed here, similar to stage transitions
    # This ensures:
    # 1. No mid-forward structural changes
    # 2. Dynamo can recompile at natural boundaries
    # 3. DDP symmetry is maintained (broadcast slim decision)
```

After slimming, call `torch._dynamo.reset()` to clear stale graphs.

### DDP Safety

Broadcast slim decisions from rank 0:

```python
def _sync_slim_decision(
    self,
    should_slim: bool,
    retention_ratio: float,
) -> tuple[bool, float]:
    """Broadcast slim decision to all ranks for DDP symmetry."""
    if not dist.is_initialized():
        return should_slim, retention_ratio

    # Pack into tensor for broadcast
    decision = torch.tensor(
        [float(should_slim), retention_ratio],
        device=self.device
    )
    dist.broadcast(decision, src=0)

    return bool(decision[0].item()), decision[1].item()
```

### Reward Signal for Slimming

Per DRL specialist recommendation, use **immediate counterfactual feedback**:

```python
def compute_slim_reward(
    slot: SeedSlot,
    pre_params: int,
    post_params: int,
    pre_contribution: float,
    post_contribution: float,  # Measured after 1 forward pass
) -> float:
    """Immediate reward for SLIM_CHANNELS operation.

    Rewards efficiency gains while penalizing contribution loss.
    """
    param_budget = 12000  # Global budget

    # Efficiency gain: fraction of budget freed
    efficiency_gain = (pre_params - post_params) / param_budget

    # Contribution preservation: did we maintain utility?
    contribution_ratio = post_contribution / (pre_contribution + 1e-8)

    if contribution_ratio >= 0.95:  # Contribution held
        return SLIM_SUCCESS_BONUS * efficiency_gain
    elif contribution_ratio >= 0.80:  # Acceptable loss
        return efficiency_gain * contribution_ratio  # Partial credit
    else:  # Unacceptable loss
        return -SLIM_FAILURE_PENALTY * (1.0 - contribution_ratio)
```

## Action Space Summary

### Before (Current)

| Head | Options |
|------|---------|
| slot | 3 |
| blueprint | 13 |
| style | 4 |
| tempo | 3 |
| alpha_target | 3 |
| alpha_speed | 4 |
| alpha_curve | 5 |
| op | 6 |

### After (Proposed)

| Head | Options | Change |
|------|---------|--------|
| slot | 3 | unchanged |
| blueprint | 19 | +6 size variants |
| style | 4 | unchanged |
| tempo | 3 | unchanged |
| alpha_target | 3 | unchanged |
| alpha_speed | 4 | unchanged |
| alpha_curve | 5 | unchanged |
| op | 7 | +1 (SLIM_CHANNELS) |
| slim_factor | 3 | **new head** (only relevant when op=SLIM_CHANNELS) |

The slim_factor head is masked (ignored) unless `op == SLIM_CHANNELS`.

## Implementation Phases

### Phase 1: Parameterized Blueprints (Low Risk)

1. Add `channel_multiplier` parameter to existing blueprint factories
2. Register size variants (`conv_heavy_slim`, `conv_heavy_tiny`, etc.)
3. Extend `BlueprintAction` enum
4. Update action masks and feature extraction
5. **No runtime surgery needed**

### Phase 2: SLIM_CHANNELS Operation (Medium Risk)

1. Implement `ModuleSurgeon` with channel pruning
2. Add `LifecycleOp.SLIM_CHANNELS` with action masking
3. Add `SlimFactor` action head
4. Implement optimizer rebuild logic
5. Add telemetry for slim operations
6. Handle DDP synchronization

### Phase 3: Reward Integration

1. Add immediate slim reward computation
2. Integrate with existing reward shaping infrastructure
3. Add PBRS potential for efficiency (optional)

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Optimizer state loss | Medium | Document as expected cost; consider momentum warmup |
| torch.compile breaks | Medium | Restrict to epoch boundaries; call dynamo.reset() |
| DDP desync | High | Broadcast all slim decisions from rank 0 |
| Credit assignment for slim | Medium | Use immediate counterfactual reward |
| Action space growth | Low | Only +1 op, +6 blueprints, +1 conditional head |

## Alternatives Considered

### A. Full Runtime Mutation (Rejected)

Allowing arbitrary channel/kernel changes at any stage. Rejected because:

- Violates fossilization semantics
- Kernel changes require weight reinitialization (defeats purpose)
- Action space explosion
- Severe credit assignment problems

### B. Automatic Slimming via Emrakul (Deferred)

Let Emrakul automatically slim inefficient modules. This is complementary and planned for Phase 4, but doesn't give Tamiyo agency during the critical BLENDING/HOLDING window.

### C. Continuous Slim Factor (Rejected)

Using a continuous action for slim factor (e.g., [0.5, 1.0]). Rejected because:

- Continuous actions harder to learn than discrete
- Most slimming needs are well-served by 3 discrete options
- Keeps action space tractable

## Success Metrics

1. **Budget Utilization**: % of 12k budget used by fossilized modules should increase
2. **Slim Success Rate**: % of SLIM_CHANNELS operations that maintain >95% contribution
3. **Learning Signal**: Tamiyo should learn to slim oversized modules during BLENDING/HOLDING
4. **No Regression**: Existing lifecycle tests remain green

## Open Questions

1. Should there be a cooldown between SLIM_CHANNELS operations on the same seed?
2. Should we track "original" vs "current" param count for reward computation?
3. How should slim operations interact with the alpha schedule? (Reset alpha? Continue?)
