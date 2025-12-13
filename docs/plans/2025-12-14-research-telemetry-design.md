# Karn: Research Telemetry System Design

**Date:** 2025-12-14
**Status:** Draft
**Subsystem:** Karn (replaces Nissa)
**Goal:** World-class telemetry for proving morphogenetic superiority over traditional training

---

## Executive Summary

Design a telemetry system that enables:
1. **Proving the thesis**: An 80B morphogenetic model can match a 130B traditional model
2. **Debugging the concept**: Deep diagnostics when things don't work
3. **Causal attribution**: Prove seeds are *why* it's better, not correlation

### Primary Metrics
- **Final accuracy delta**: Morphogenetic vs host-only baseline
- **Parameter efficiency**: Accuracy per parameter (the headline number)

### Design Principles
- **Causal Replay**: Capture enough state to reproduce any run and simulate counterfactuals
- **Adaptive Fidelity**: Dense capture during interesting periods, sparse otherwise
- **Current Scale First**: Optimize for CIFAR/TinyStories, scale-aware design for later

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              KARN                                       │
│                    (Research Telemetry System)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐   │
│  │  Emitters   │────▶│ Collector   │────▶│   Analytics Engine      │   │
│  │             │     │             │     │                         │   │
│  │ - Kasmina   │     │ - Validates │     │ - MorphogeneticAnalytics│   │
│  │ - Simic     │     │ - Timestamps│     │ - CounterfactualEngine  │   │
│  │ - Tamiyo    │     │ - Routes    │     │ - HealthMonitor         │   │
│  │ - Tolaria   │     │             │     │ - PolicyAnalyzer        │   │
│  └─────────────┘     └─────────────┘     └───────────┬─────────────┘   │
│                                                      │                  │
│                            ┌─────────────────────────┴──────┐           │
│                            ▼                                ▼           │
│                   ┌─────────────────┐            ┌──────────────────┐   │
│                   │  State Store    │            │  Output Backends │   │
│                   │                 │            │                  │   │
│                   │ - EpisodeContext│            │ - Console (live) │   │
│                   │ - EpochSnapshots│            │ - Structured Log │   │
│                   │ - DenseTraces   │            │ - Future: Files  │   │
│                   │ - Counterfactuals│           │ - Future: W&B    │   │
│                   └─────────────────┘            └──────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model: Three Tiers

### Tier 1: Episode Context (Once Per Run)

Captures everything needed to reproduce the run and establish baselines.

```python
@dataclass(frozen=True)
class EpisodeContext:
    # Identity
    episode_id: str              # UUID for this run
    timestamp: datetime          # When training started
    git_commit: str              # Exact code version

    # Determinism
    base_seed: int               # Master random seed
    torch_seed: int              # PyTorch RNG state
    numpy_seed: int              # NumPy RNG state

    # Architecture
    host_architecture: str       # "cnn_3block" / "gpt_6layer"
    host_params: int             # Param count before any seeds
    slot_config: dict[str, int]  # {"early": 64, "mid": 128, "late": 256}

    # Training config
    max_epochs: int
    task_type: str               # "classification" / "lm"
    hyperparameters: dict        # lr, batch_size, etc.


@dataclass
class HostBaseline:
    # Captured at episode start (epoch 0, before any germination)
    initial_loss: float
    initial_accuracy: float
    initial_checkpoint: bytes | Path  # Full model state for replay

    # Captured at episode end (counterfactual: what if no seeds ever?)
    final_host_only_loss: float | None
    final_host_only_accuracy: float | None
```

### Tier 2: Epoch Snapshots (Every Epoch)

The workhorse tier for analysis.

**Per-Slot State:**

```python
@dataclass
class SlotSnapshot:
    slot_id: str                    # "early", "mid", "late"

    # Lifecycle state
    stage: SeedStage                # DORMANT → GERMINATED → ... → FOSSILIZED
    epochs_in_stage: int            # How long at current stage
    seed_id: str | None             # Current seed identifier
    blueprint_id: str | None        # "conv_light", "attention", etc.

    # Parameters
    seed_params: int                # Parameter count of this seed
    alpha: float                    # Blend coefficient [0, 1]

    # Attribution (the key metrics)
    counterfactual_contribution: float | None  # acc_with - acc_without
    total_improvement: float | None            # acc_now - acc_at_germination
    improvement_this_epoch: float              # delta from last epoch

    # Gate status
    last_gate_attempted: str | None   # "G2", "G3", etc.
    last_gate_passed: bool | None
    last_gate_reason: str | None      # Why it passed/failed
```

**Host State:**

```python
@dataclass
class HostSnapshot:
    epoch: int

    # Performance
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float

    # Parameter accounting
    host_params: int                  # Fixed host parameters
    total_seed_params: int            # Sum across all active seeds
    total_params: int                 # host + seeds
    fossilized_params: int            # Permanently added params

    # Gradient health
    host_grad_norm: float
    seed_grad_norms: dict[str, float]  # Per-slot gradient norms
    grad_isolation_leakage: float | None  # If monitoring isolation
```

**Policy State:**

```python
@dataclass
class PolicySnapshot:
    # Observation (what the agent saw)
    observation_vector: list[float]   # Raw obs fed to policy

    # Action (what the agent did)
    action: FactoredAction            # op, slot, blueprint, blend
    action_log_probs: dict[str, float]  # Per-head log probs
    action_was_masked: bool           # Was this action forced by mask?

    # Value estimates
    value_estimate: float             # Critic's prediction
    advantage: float | None           # Computed during PPO update

    # Reward breakdown
    reward_total: float
    reward_components: RewardComponentsTelemetry
```

### Tier 3: Dense Traces (Adaptive)

Triggered during interesting periods for deep diagnostics.

**Trigger Conditions:**

```python
@dataclass
class DenseTraceTrigger:
    # Stage transitions (always interesting)
    stage_transition: bool = True

    # Anomalies
    loss_spike_threshold: float = 2.0      # > 2x rolling average
    accuracy_drop_threshold: float = 5.0   # > 5% drop epoch-over-epoch
    gradient_explosion: float = 100.0      # grad_norm > 100x typical

    # Gate events
    gate_failure: bool = True              # Any G0-G5 gate fails

    # Policy anomalies
    value_collapse: bool = True            # Critic predicts same value everywhere
    entropy_collapse: bool = True          # Policy becomes deterministic

    # Manual override
    force_dense: bool = False              # Research mode toggle
```

**Dense Trace Content:**

```python
@dataclass
class DenseTrace:
    trigger_reason: str                    # "loss_spike", "gate_failure", etc.
    window_start_epoch: int
    window_end_epoch: int

    # Per-batch metrics
    batch_metrics: list[BatchMetrics]

    # Gradient details
    gradient_histograms: dict[str, list[float]]
    gradient_flow: dict[str, float]

    # Activation statistics
    activation_means: dict[str, float]
    activation_stds: dict[str, float]
    dead_neuron_counts: dict[str, int]

    # Gate internals (if gate event)
    gate_evaluation_details: GateEvaluationTrace | None


@dataclass
class BatchMetrics:
    epoch: int
    batch_idx: int
    loss: float
    accuracy: float
    host_grad_norm: float
    seed_grad_norms: dict[str, float]
    isolation_leakage: float | None
```

---

## Counterfactual Capture

The key mechanism for proving causality with multiple seeds.

### Full Factorial Matrix

For a 3-slot host, capture 8 configurations:

```
Config  | Early | Mid | Late | Val Accuracy
--------|-------|-----|------|-------------
000     |  off  | off | off  | 62.3%  ← host-only baseline
001     |  off  | off | ON   | 64.1%
010     |  off  | ON  | off  | 63.8%
011     |  off  | ON  | ON   | 65.2%
100     |  ON   | off | off  | 63.5%
101     |  ON   | off | ON   | 65.8%
110     |  ON   | ON  | off  | 64.9%
111     |  ON   | ON  | ON   | 66.4%  ← actual training state
```

### Derived Metrics

From the matrix, compute:
- **Marginal contribution**: `acc(1xx) - acc(0xx)` = seed's average effect
- **Interaction effects**: Does Early+Mid together exceed Early + Mid separately?
- **Shapley values**: Fair attribution accounting for all orderings

### Scaling Strategy

**Default behavior** (optimized for reasonable compute):

| Active Seeds | Configurations | Strategy |
|--------------|----------------|----------|
| 1-2          | 2-4            | Full factorial (always) |
| 3-4          | 8-16           | Full factorial |
| 5+           | 32+            | Shapley sampling (~20 samples) |

**Configuration:** The scaling strategy is fully configurable. If you need full factorial on 100 seeds (2^100 configurations) and have the compute budget, that's your choice.

```python
@dataclass
class CounterfactualConfig:
    """User-configurable counterfactual strategy."""

    # Strategy selection
    strategy: Literal["auto", "full_factorial", "shapley", "ablation_only"] = "auto"

    # Auto-strategy thresholds (only used when strategy="auto")
    full_factorial_max_seeds: int = 4      # Switch to Shapley above this
    shapley_samples: int = 20              # Permutation samples for Shapley

    # Force full factorial regardless of seed count (use with caution)
    force_full_factorial: bool = False     # Override auto thresholds

    # Compute budget controls
    max_configurations: int | None = None  # Hard cap, None = unlimited
    timeout_seconds: float | None = None   # Abort if exceeded

    def effective_strategy(self, n_active_seeds: int) -> str:
        """Determine strategy based on config and seed count."""
        if self.force_full_factorial:
            return "full_factorial"
        if self.strategy != "auto":
            return self.strategy
        if n_active_seeds <= self.full_factorial_max_seeds:
            return "full_factorial"
        return "shapley"
```

**Example: Force full factorial for publication-grade results**
```python
# Warning: 2^10 = 1024 validation passes per epoch
config = CounterfactualConfig(
    force_full_factorial=True,
    timeout_seconds=3600.0,  # 1 hour safety cap
)
```

### Data Structures

```python
@dataclass
class CounterfactualResult:
    config: tuple[bool, ...]          # (True, False, True) = early on, mid off, late on
    alpha_settings: dict[str, float]  # {"early": 1.0, "mid": 0.0, "late": 1.0}
    val_loss: float
    val_accuracy: float
    per_class_accuracy: dict[int, float] | None  # Research mode only


@dataclass
class CounterfactualMatrix:
    epoch: int
    configs: list[CounterfactualResult]

    # Derived metrics
    marginal_contributions: dict[str, float]
    interaction_effects: dict[tuple, float]
    shapley_values: dict[str, float] | None
```

---

## Live Analytics

Real-time computation of research metrics during training.

### Analytics API

```python
class MorphogeneticAnalytics:
    # === Claim 1: Accuracy ===
    def accuracy_delta(self) -> float:
        """Current accuracy vs host-only baseline."""
        return self.current_accuracy - self.host_baseline_accuracy

    def accuracy_trajectory(self) -> list[tuple[int, float, float]]:
        """(epoch, morphogenetic_acc, host_baseline_acc) over time."""

    # === Claim 2: Parameter Efficiency ===
    def params_efficiency_ratio(self) -> float:
        """Accuracy per million parameters vs baseline. >1.0 = winning."""
        morph_efficiency = self.current_accuracy / (self.total_params / 1e6)
        baseline_efficiency = self.host_baseline_accuracy / (self.host_params / 1e6)
        return morph_efficiency / baseline_efficiency

    def equivalent_traditional_params(self) -> int:
        """Estimated traditional model size to match our accuracy."""

    # === Causal Attribution ===
    def seed_contributions(self) -> dict[str, SeedContributionSummary]:
        """Per-seed causal attribution from counterfactual matrix."""

    # === Policy Quality ===
    def policy_vs_heuristic(self) -> PolicyComparisonSummary:
        """How is the learned policy doing vs what heuristic would do?"""

    def action_distribution(self) -> dict[str, float]:
        """What actions is the policy taking?"""

    # === Health Monitoring ===
    def training_health(self) -> HealthReport:
        """Aggregate health indicators."""
```

### Live Dashboard Output

```
┌─────────────────────────────────────────────────────────────────┐
│  KARN · MORPHOGENETIC TRAINING MONITOR             Epoch 47/100 │
├─────────────────────────────────────────────────────────────────┤
│  ACCURACY        67.2% (+4.9% vs host-only)              ✓ WIN │
│  PARAM RATIO     1.23x host params                             │
│  EFFICIENCY      1.31x better acc/param                  ✓ WIN │
├─────────────────────────────────────────────────────────────────┤
│  SEED CONTRIBUTIONS (Shapley)                                   │
│    early:  +1.2% (conv_light, 37K params)  efficiency: 0.32    │
│    mid:    +2.8% (attention, 2K params)    efficiency: 14.0 ★  │
│    late:   +0.9% (norm, 100 params)        efficiency: 90.0 ★  │
├─────────────────────────────────────────────────────────────────┤
│  POLICY HEALTH                                                  │
│    Value variance: 0.42 (healthy)                              │
│    Entropy: 1.23 (exploring)                                   │
│    Recent actions: WAIT WAIT GERMINATE(mid) WAIT FOSSILIZE     │
├─────────────────────────────────────────────────────────────────┤
│  ALERTS: None                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration with Existing Subsystems

### Emitter Integration Points

| Subsystem | Current | New Emissions |
|-----------|---------|---------------|
| **Kasmina** (slot.py) | Stage transitions | + Gate evaluation details, alpha changes, isolation metrics |
| **Simic** (vectorized.py) | Rewards | + Full counterfactual matrix, policy internals, batch metrics |
| **Tamiyo** (tracker.py) | Signals | + Shadow heuristic for comparison, decision explanations |
| **Tolaria** (trainer.py) | Loss/acc | + Gradient health, per-component attribution |

### Karn: Nissa's Replacement

The new telemetry subsystem is named **Karn** (the silver golem with perfect memory who witnessed millennia of multiverse history — fitting for a system that captures and recalls every training event).

**Karn improves on Nissa with:**
- **Stateful analytics** (running averages, trajectory tracking)
- **Counterfactual engine** as first-class component
- **Typed contracts** for all emissions (Leyline-style dataclasses)
- **Tiered capture** (adaptive fidelity based on training phase)
- **Research-focused metrics** (Shapley attribution, parameter efficiency)

---

## Nissa → Karn Migration Plan

### Strategy: A/B Cutover

Build Karn as a complete, parallel subsystem. Both coexist during development. Cut over entirely when Karn is validated. Delete Nissa completely (no legacy shims).

```
Phase 1: BUILD          Phase 2: VALIDATE       Phase 3: CUTOVER
─────────────────       ─────────────────       ─────────────────
[Nissa: ACTIVE  ]       [Nissa: ACTIVE  ]       [Nissa: DELETED ]
[Karn:  BUILDING]  ──▶  [Karn:  SHADOW  ]  ──▶  [Karn:  ACTIVE  ]
                        (both emit, compare)
```

### Phase 1: Build Karn (esper/karn/)

**New module structure:**
```
src/esper/karn/
├── __init__.py           # Public API
├── collector.py          # Event collection, validation, routing
├── store.py              # EpisodeContext, EpochSnapshots, DenseTraces
├── counterfactual.py     # Factorial matrix, Shapley estimation
├── analytics.py          # MorphogeneticAnalytics, live computations
├── health.py             # HealthMonitor, anomaly detection
├── output.py             # Console, structured log, future backends
└── triggers.py           # DenseTraceTrigger conditions
```

**API Design:**
```python
# esper/karn/__init__.py
from esper.karn.collector import KarnCollector, emit
from esper.karn.store import EpisodeContext, EpochSnapshot, DenseTrace
from esper.karn.analytics import MorphogeneticAnalytics
from esper.karn.counterfactual import CounterfactualMatrix, CounterfactualResult
from esper.karn.output import ConsoleOutput, StructuredLogOutput
from esper.karn.health import HealthMonitor, HealthReport

def configure(config: KarnConfig) -> KarnCollector:
    """Initialize Karn telemetry for a training run."""
    ...

def get_collector() -> KarnCollector:
    """Get the active collector (analogous to Nissa's get_hub)."""
    ...
```

### Phase 2: Shadow Validation

Run both systems simultaneously to validate Karn captures everything Nissa does, plus more.

**Validation criteria:**
| Check | Method |
|-------|--------|
| Event parity | Assert every Nissa event has Karn equivalent |
| Metric consistency | Compare epoch-level metrics (loss, accuracy) |
| No regression | Karn overhead ≤ Nissa overhead |
| New capabilities | Counterfactual matrix produces valid Shapley values |

**Shadow mode implementation:**
```python
# In training loop during Phase 2
nissa_hub = nissa.get_hub()
karn_collector = karn.get_collector()

# Emit to both
nissa_hub.emit(event)
karn_collector.emit(event)  # Karn converts to its schema

# End of epoch: compare
assert_metrics_match(nissa_hub.epoch_summary(), karn_collector.epoch_snapshot())
```

### Phase 3: Cutover & Nissa Retirement

**Callsite migration (5 files):**

| File | Current | After Cutover |
|------|---------|---------------|
| `tamiyo/tracker.py:13` | `from esper.nissa import get_hub` | `from esper.karn import get_collector` |
| `scripts/train.py:17` | `from esper.nissa import get_hub, ConsoleOutput, ...` | `from esper.karn import configure, ConsoleOutput, ...` |
| `simic/training.py:21` | `from esper.nissa import get_hub` | `from esper.karn import get_collector` |
| `simic/vectorized.py:56` | `from esper.nissa import get_hub, BlueprintAnalytics` | `from esper.karn import get_collector, MorphogeneticAnalytics` |
| `simic/ppo.py:24` | `from esper.nissa import get_hub` | `from esper.karn import get_collector` |

**Retirement steps:**
1. Update all 5 callsites in single PR
2. Run full test suite with Karn-only
3. Delete `src/esper/nissa/` directory entirely
4. Delete `tests/nissa/` directory entirely
5. Remove Nissa from documentation
6. Update architecture docs to reflect Karn

**No transition period.** Once Phase 2 validates Karn, cut over completely. Per project policy: no legacy code, no compatibility shims.

### Timeline

| Phase | Duration | Exit Criteria |
|-------|----------|---------------|
| **Phase 1: Build** | ~1 week | Karn module complete, unit tests pass |
| **Phase 2: Shadow** | ~3-5 training runs | All validation criteria met |
| **Phase 3: Cutover** | 1 PR | Tests green, Nissa deleted |

---

## Operating Modes

### Standard Mode (Default)
- Tier 1: Full capture
- Tier 2: Every epoch
- Tier 3: Trigger-based only
- Counterfactuals: Full factorial up to 4 seeds

### Research Mode (Max Fidelity)
- Tier 1: Full capture
- Tier 2: Every epoch + extended fields
- Tier 3: Every batch (force_dense=True)
- Counterfactuals: Full factorial + per-class breakdowns

### Adaptive Escalation
- Start in Standard mode
- Detect anomaly → escalate to Research mode for window
- Anomaly resolves → return to Standard mode

---

## Future Extensions (Not in Scope Now)

- **File persistence**: Parquet/JSONL for post-hoc analysis
- **W&B integration**: Experiment tracking across runs
- **Distributed telemetry**: Aggregation across DDP ranks
- **Scaling optimizations**: Sampling strategies for 80B+ models

---

## Success Criteria

The telemetry system is successful if it can answer:

1. **"Is morphogenesis winning?"** — Live accuracy delta and efficiency ratio
2. **"Why is it winning?"** — Shapley attribution per seed
3. **"Why did it fail?"** — Dense traces around anomalies with root cause
4. **"Can we reproduce this?"** — Episode context enables exact replay
5. **"What would have happened without seeds?"** — Host baseline comparison

---

## Additional Requirements (Expert Review Findings)

**Review Date:** 2025-12-14
**Reviewers:** DRL Specialist, PyTorch Engineering Specialist

### P0 — Critical (Fix Before Implementation)

#### 1. Add PPO Diagnostic Stats to PolicySnapshot

The design captures `advantage: float` but not aggregate statistics essential for PPO debugging.

```python
@dataclass
class AdvantageStats:
    mean: float
    std: float
    min: float
    max: float
    fraction_clipped: float  # |A| > clip_threshold

@dataclass
class RatioStats:
    mean: float
    fraction_clipped_high: float  # ratio > 1 + eps
    fraction_clipped_low: float   # ratio < 1 - eps
    per_head_clip_rates: dict[str, float]
```

**Rationale:** Without ratio/advantage visibility, PPO debugging is "flying blind." These are the primary health indicators.

#### 2. Add DDP Forward Compatibility Fields

Add rank awareness to telemetry events now to prevent schema migration pain later.

```python
@dataclass
class TelemetryEvent:
    event_type: TelemetryEventType
    rank: int = 0              # Add NOW
    world_size: int = 1        # Add NOW
    is_reduced: bool = False   # True if aggregated across ranks
```

#### 3. Document Counterfactual Validity Scope

The counterfactual matrix measures **removal cost**, not **causal contribution**. This is an important distinction:

- **What we measure:** "How much worse is accuracy when seed A is disabled at epoch T?"
- **What this is NOT:** "How much value did seed A add?" (requires parallel training without seed A)

Add a `CounterfactualValidityNote` to documentation explaining this limitation. The host and other seeds have adapted *assuming all seeds were present*, creating confounding that cannot be eliminated without parallel control runs.

---

### P1 — High Priority (Add During Implementation)

#### 4. Specify Shapley Sampling Algorithm

The design mentions "~20 samples" for 5+ seeds but doesn't specify the algorithm.

**Recommendation:** Use permutation sampling with antithetic pairing (sample permutation and its reverse) for 2x variance reduction.

```python
@dataclass
class ShapleyEstimate:
    mean: float
    std: float  # Required for confidence intervals
    n_samples: int
    algorithm: str = "permutation_antithetic"
```

**Research claims require:** `shapley_mean - 2*shapley_std > 0` before claiming positive contribution.

#### 5. Switch to O(n) Counterfactual Strategy for 3+ Slots

Full factorial at 3 slots = 8x validation overhead (~700% increase). Use single-slot ablation + Shapley:

```python
# O(n) instead of O(2^n)
for slot_id in active_slots:
    with model.seed_slots[slot_id].force_alpha(0.0):
        slot_contribution = val_acc - ablated_acc

# Derive interaction effects from marginal contributions
# using inclusion-exclusion for 2-way interactions only
```

#### 6. Add KL Divergence and Explained Variance

Essential PPO diagnostics missing from the design:

```python
@dataclass
class PolicySnapshot:
    # ... existing fields ...

    # Add these:
    kl_divergence: float  # KL(old || new), critical for LR calibration
    explained_variance: float  # 1 - Var(returns - values) / Var(returns)
    per_head_kl: dict[str, float]  # Per-head KL for factored actions
```

**Thresholds:**
- `KL > 0.1` per update → learning rate too high
- `KL ~ 0` → learning rate too low or degenerate gradients
- `explained_variance < 0` → value function is worse than mean prediction

#### 7. Wrap Dense Trace Collection to Prevent Graph Breaks

Dense trace activation hooks can break `torch.compile` graphs. Use explicit disable:

```python
@torch.compiler.disable
def collect_dense_trace(model: nn.Module, batch_metrics: list) -> DenseTrace:
    """Dense trace collection - explicitly disable compilation."""
    # This prevents graph breaks from propagating to main training loop
    ...
```

#### 8. Use Async Gradient Histogram Collection

Avoid `.item()` sync points in gradient telemetry:

```python
def collect_gradient_histogram_async(grads: list[Tensor], bins: int = 50) -> Tensor:
    """Collect histogram as tensor buckets - no .item() sync."""
    flat = torch.cat([g.view(-1) for g in grads])
    return torch.histc(flat, bins=bins, min=-1.0, max=1.0)  # Returns tensor
```

---

### P2 — Medium Priority (Polish Phase)

#### 9. Add LSTM Hidden State Diagnostics

Recurrent policies have unique failure modes not covered by standard PPO diagnostics:

```python
@dataclass
class LSTMHealth:
    hidden_mean: float
    hidden_std: float
    cell_mean: float
    cell_std: float
    saturation_fraction: float  # |h| > 0.9, indicates potential vanishing gradients
```

#### 10. Store Checkpoints as Paths, Not Bytes

`initial_checkpoint: bytes` could be 50-200MB. Store as path reference instead:

```python
initial_checkpoint: Path  # Reference to torch.save() output
checkpoint_retention_policy: Literal["keep_all", "keep_last", "delete_after_run"]
```

#### 11. Add Ringbuffer for Dense Traces

Bound memory growth from triggered dense traces:

```python
class TelemetryStore:
    def __init__(self, max_dense_traces: int = 10):
        self.dense_traces = deque(maxlen=max_dense_traces)  # Auto-evicts oldest
```

#### 12. Expose Existing GradientIsolationMonitor

Don't duplicate isolation leakage detection — the existing `GradientIsolationMonitor` in `kasmina/isolation.py` already implements this correctly. Route its output through the telemetry system.

---

### P3 — Low Priority (Future Enhancement)

#### 13. Add Cross-Run Statistical Aggregation

For publication-grade claims, need multi-run statistics:

```python
@dataclass
class ExperimentAggregation:
    run_ids: list[str]
    mean_delta: float
    std_delta: float
    p_value: float       # vs null hypothesis of no improvement
    cohens_d: float      # Effect size
```

#### 14. Add Contribution Onset Tracking

Track when seeds start helping, not just current contribution:

```python
def contribution_onset_epoch(self, slot_id: str, threshold: float = 0.5) -> int | None:
    """First epoch where seed's Shapley value exceeded threshold."""
```

#### 15. Consider SHAP-IQ for Interaction Decomposition

Fumagalli et al. (2023) — decomposes Shapley values into pairwise interactions, directly answering "Do early+mid synergize?"

---

### Summary of Expert Assessments

| Aspect | DRL Specialist | PyTorch Specialist |
|--------|---------------|-------------------|
| Overall Rating | GOOD with improvements needed | Architecturally sound |
| Counterfactual Design | Sound but overclaims validity | O(2^n) overhead concern |
| PPO Diagnostics | Critical gaps (ratio/advantage/KL) | N/A |
| Performance | N/A | Manageable with mitigations |
| torch.compile | N/A | Needs explicit disable zones |
| DDP Readiness | N/A | Add schema fields now |

**Bottom Line:** The design is a strong foundation that needs PPO diagnostic augmentation (DRL) and performance/compatibility mitigations (PyTorch) before implementation.
