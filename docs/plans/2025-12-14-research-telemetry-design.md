# Research Telemetry System Design

**Date:** 2025-12-14
**Status:** Draft
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
│                         TELEMETRY SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐   │
│  │  Emitters   │────▶│  Collector  │────▶│   Analytics Engine      │   │
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

| Active Seeds | Configurations | Strategy |
|--------------|----------------|----------|
| 1-2          | 2-4            | Full factorial (always) |
| 3-4          | 8-16           | Full factorial |
| 5+           | 32+            | Shapley sampling (~20 samples) |

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
│  MORPHOGENETIC TRAINING MONITOR                    Epoch 47/100 │
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

### Replaces Nissa

The new telemetry system replaces Nissa's hub with:
- **Stateful analytics** (running averages, trajectory tracking)
- **Counterfactual engine** as first-class component
- **Typed contracts** for all emissions (Leyline-style dataclasses)

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
