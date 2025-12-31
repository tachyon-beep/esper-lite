# Data Generation System Design

**Date**: 2025-11-26
**Status**: Draft (reviewed by drl-expert)
**Goal**: Generate high-quality diverse dataset for offline RL training

## Context

Our current dataset has 87% of state clusters with only one action - Kasmina is deterministic, so same state → same action. This prevents offline RL from learning when to intervene differently.

This design creates a data generation system that produces diverse counterfactual data through:
- Multiple behavior policy variants (including random and deliberately bad policies)
- ε-greedy exploration with explicit probability logging
- Architecture and hyperparameter variation
- Rich metadata for future method compatibility (including return-to-go for Decision Transformer)

## Design Principles

1. **Quality over speed** - We optimize for dataset quality, willing to invest compute
2. **Future-proof metadata** - Store enough to enable methods we haven't tried yet
3. **Adaptive generation** - Start uniform, then fill coverage gaps
4. **Tiered validation** - Blocking errors for critical issues, warnings for minor ones
5. **Contrastive signal** - Include bad policies to establish value estimation floor

---

## Core Data Structures

### EnvironmentConfig

Defines a training environment setup:

```python
@dataclass
class EnvironmentConfig:
    config_id: str              # e.g., "resnet34-adam"
    architecture: str           # "HostCNN", "HostCNN-Wide", "ResNet-18", etc.
    learning_rate: float
    batch_size: int
    optimizer: str              # "SGD" or "Adam"
    momentum: float = 0.9       # For SGD
    weight_decay: float = 0.0
    max_epochs: int = 25
```

### BehaviorPolicyConfig

Defines a Kasmina variant:

```python
@dataclass
class BehaviorPolicyConfig:
    policy_id: str                          # e.g., "eager", "patient"
    min_epochs_before_germinate: int = 5
    plateau_epochs_to_germinate: int = 3
    cull_after_epochs_without_improvement: int = 5
    blueprint_preference: list[str] = None  # None = default order
    epsilon: float = 0.0                    # ε-greedy exploration
    temperature: float = 1.0                # For softmax probabilities
```

### ActionProbabilities

Logged per decision with explicit behavior policy probability for importance sampling:

```python
@dataclass
class ActionProbabilities:
    # Core probabilities
    greedy_probs: dict[str, float]      # π_greedy(a|s) from softmax over scores
    behavior_probs: dict[str, float]    # μ(a|s) = (1-ε) * π_greedy + ε/|A|
    behavior_prob: float                # μ(action_taken|s) - CRITICAL for off-policy

    # Action selection
    greedy_action: str          # What policy would pick without ε
    sampled_action: str         # What was actually taken
    was_exploratory: bool       # Did ε override greedy?
    epsilon: float              # ε value used for this decision

    @staticmethod
    def compute_behavior_prob(greedy_probs: dict, action: str, epsilon: float) -> float:
        """Compute μ(a|s) accounting for ε-greedy."""
        num_actions = len(greedy_probs)
        return (1 - epsilon) * greedy_probs[action] + epsilon / num_actions
```

### RewardComponents

Stored per step for future relabeling:

```python
@dataclass
class RewardComponents:
    accuracy_delta: float       # Raw accuracy change this step
    loss_delta: float           # Raw loss change this step
    potential_prev: float       # Φ(s) before step
    potential_next: float       # Φ(s') after step
    intervention_cost: float    # Action-specific costs (see below)
    sparse: float = 0.0         # Only non-zero at episode end (final accuracy)
    return_to_go: float = 0.0   # Sum of future rewards (for Decision Transformer)

    def total(self, gamma: float = 0.99, shaping_weight: float = 1.0) -> float:
        """Reconstruct reward with configurable shaping."""
        shaping = gamma * self.potential_next - self.potential_prev
        return self.accuracy_delta * 10 + shaping_weight * shaping + self.intervention_cost
```

**Intervention costs** (differentiated by action):
- WAIT: 0.0 (no cost)
- GERMINATE: -0.02 (spawning has overhead)
- ADVANCE: -0.01 (promoting is cheaper)
- CULL: -0.005 (cleanup is cheapest intervention)

**Potential function** (non-monotonic to penalize stagnation):
```python
def compute_potential(state) -> float:
    """Φ(s) that decays if not improving."""
    base = state.best_accuracy
    stagnation_penalty = (1 - state.best_accuracy / 100) * (
        state.steps_since_improvement / state.patience_threshold
    )
    return base - stagnation_penalty * 10  # Decays up to 10 points if stuck
```

### EpisodeMetadata

Stored per episode:

```python
@dataclass
class EpisodeMetadata:
    episode_id: str
    timestamp: datetime
    schema_version: str = "2.0.0"

    # Generation context
    behavior_policy: BehaviorPolicyConfig
    environment: EnvironmentConfig
    random_seed: int

    # Outcome summary
    final_accuracy: float
    best_accuracy: float
    total_return: float
    return_to_go_at_start: float    # For Decision Transformer conditioning
    episode_length: int
    termination_reason: str         # "max_epochs", "early_stop", "diverged"
    done: bool                      # True if natural termination
    truncated: bool                 # True if hit max_epochs

    # Intervention summary
    num_interventions: int
    action_counts: dict[str, int]
    seeds_created: int
    seeds_fossilized: int
    seeds_culled: int

    # Quality labels
    trajectory_quality: str         # "success", "partial", "failure"
    trajectory_hash: str            # For deduplication
```

### StepMetadata

Additional per-step fields beyond observation/action/reward:

```python
@dataclass
class StepMetadata:
    timestep: int                   # Position in episode (0-indexed)
    done: bool                      # Is this a terminal state?
    truncated: bool                 # Was episode truncated here?
    state_hash: str                 # For coverage analysis

    # Training dynamics
    active_seed_count: int
    best_seed_accuracy: float | None
    training_budget_remaining: float  # (max_epochs - current_epoch) / max_epochs
```

---

## Environment Configurations

13 configurations covering architecture, learning rate, batch size, and optimizer variation:

| Config ID | Architecture | LR | Batch | Optimizer | Notes |
|-----------|--------------|-----|-------|-----------|-------|
| baseline | HostCNN | 0.01 | 128 | SGD | Reference configuration |
| fast-lr | HostCNN | 0.1 | 128 | SGD | Fast learning, early plateaus |
| slow-lr | HostCNN | 0.001 | 128 | SGD | Slow learning, gradual improvement |
| wide | HostCNN-Wide | 0.01 | 128 | SGD | 2x channels, different capacity |
| deep | HostCNN-Deep | 0.01 | 128 | SGD | 5 blocks, gradient flow differs |
| adam | HostCNN | 0.001 | 128 | Adam | Adaptive learning rates |
| small-batch | HostCNN | 0.01 | 64 | SGD | High gradient variance |
| large-batch | HostCNN | 0.02 | 256 | SGD | Low variance, needs LR scaling |
| resnet18 | ResNet-18 | 0.01 | 128 | SGD | Skip connections |
| resnet34 | ResNet-34 | 0.01 | 128 | SGD | Deep baseline |
| resnet34-adam | ResNet-34 | 0.001 | 128 | Adam | Deep + adaptive LR |
| resnet34-slow | ResNet-34 | 0.005 | 128 | SGD | Deep + gentle LR |
| resnet34-small-batch | ResNet-34 | 0.01 | 64 | SGD | Deep + high variance |

### Why These Configurations?

- **LR variation** (slow/baseline/fast): Affects plateau timing and intervention urgency
- **Batch size variation** (64/128/256): Affects gradient noise and training stability
- **Architecture variation**: Different models plateau differently, need different interventions
- **ResNet-34 cluster**: Deep networks exercise gradient health telemetry, show vanishing/exploding gradient patterns

---

## Behavior Policy Variants

11 behavior policies covering structured variants, exploration, and contrastive examples:

### Threshold-based Variants (8)

| Variant | min_epochs | plateau_epochs | cull_after | blueprint_pref | Notes |
|---------|------------|----------------|------------|----------------|-------|
| baseline | 5 | 3 | 5 | default | Current Kasmina |
| early-intervener | 3 | 2 | 5 | default | Intervenes sooner |
| late-intervener | 8 | 5 | 5 | default | Waits longer |
| quick-culler | 5 | 3 | 3 | default | Gives up faster |
| patient-culler | 5 | 3 | 8 | default | More patience |
| blueprint-explorer | 5 | 3 | 5 | rotate | Cycles through all blueprints |
| aggressive | 3 | 2 | 3 | default | Early intervention + quick cull |
| conservative | 8 | 5 | 8 | default | Late intervention + patient |

### Exploration/Contrastive Policies (3)

| Variant | Type | Purpose |
|---------|------|---------|
| random | ε = 1.0 | Uniform random actions - baseline coverage guarantee |
| anti-kasmina | Inverted logic | Does opposite of baseline - provides negative examples |
| periodic | Fixed schedule | GERMINATE every N epochs regardless of state |

**Why include bad policies?**
- Establishes a "floor" for value estimation
- Provides contrastive signal (seeing bad outcomes helps learn what to avoid)
- ~10-20% of episodes should come from these policies

### Exploration Mechanism

Threshold-based variants are wrapped with ε-greedy exploration:
- **ε = 0.0**: Pure policy (for some baseline episodes)
- **ε = 0.1**: Light exploration
- **ε = 0.2**: Moderate exploration

Action probabilities are computed via softmax over decision scores (temperature τ = 1.0), then ε-greedy is applied:
```
μ(a|s) = (1-ε) * π_greedy(a|s) + ε/|A|
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Generation System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Environment  │    │   Behavior   │    │   Episode    │       │
│  │   Configs    │    │   Policies   │    │  Collector   │       │
│  │  (13 configs)│    │ (8 variants  │    │ (metadata +  │       │
│  │              │    │  + ε-greedy) │    │  telemetry)  │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────┬───────┴───────────────────┘                │
│                     ▼                                            │
│           ┌─────────────────┐                                    │
│           │   Generation    │                                    │
│           │   Orchestrator  │                                    │
│           │  (parallelized) │                                    │
│           └────────┬────────┘                                    │
│                    ▼                                             │
│           ┌─────────────────┐                                    │
│           │  Health Checks  │──── Errors block, warnings log     │
│           └────────┬────────┘                                    │
│                    ▼                                             │
│           ┌─────────────────┐                                    │
│           │   Data Pack     │                                    │
│           │  Consolidator   │                                    │
│           └─────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Responsibility |
|--------|----------------|
| `EnvironmentConfig` | Dataclass for arch/LR/batch/optimizer |
| `BehaviorPolicy` | Wraps Kasmina variant + ε-greedy + probability logging |
| `EnrichedCollector` | Extended collector with new metadata fields |
| `GenerationOrchestrator` | Schedules policy × environment combinations |
| `DatasetHealthCheck` | Validates coverage, entropy, diversity |
| `PackBuilder` | Consolidates episodes with schema versioning |

---

## Dataset Health Checks

### Tiered Validation

**Blocking errors** (generation fails):
- Single-action clusters > 50%
- Any action < 2% of total data
- Action entropy per state cluster < 0.3

**Warnings** (logged for review):
- Action entropy per state cluster < 0.5
- Behavior policy diversity < 5 policies used
- Return distribution std < 10% of mean
- Any action < 5% of total data
- Late-episode state coverage < 30% of early-episode
- State cluster coverage < 50% of expected clusters
- Reward variance < 1.0 (limited signal)
- All episodes same length (lack of early termination examples)

### State Coverage Check

Cluster states and verify coverage:
```python
def check_state_coverage(episodes, n_clusters=100):
    """Verify we're visiting diverse states."""
    states = extract_all_states(episodes)
    kmeans = KMeans(n_clusters=n_clusters).fit(states)
    cluster_counts = Counter(kmeans.labels_)

    # Check for empty or near-empty clusters
    empty_clusters = sum(1 for c in range(n_clusters) if cluster_counts[c] < 5)
    coverage = 1 - (empty_clusters / n_clusters)

    return {
        "coverage": coverage,
        "empty_clusters": empty_clusters,
        "warning": coverage < 0.5,
        "error": coverage < 0.3,
    }
```

### When Checks Run

1. **Incremental**: After each batch of episodes (every 50-100 episodes)
2. **After generation**: Before consolidating into pack
3. **Before training**: Sanity check that pack is usable

---

## Episode Allocation Strategy

**Matrix size**: 11 policies × 13 environments = 143 combinations

**Target dataset size**: 100K-200K transitions minimum
- At ~100 steps/episode, need 1,000-2,000 episodes
- Minimum 10 episodes per combination = 1,430 episodes baseline

**Adaptive allocation**:
1. Start with baseline: 10 episodes per combination = 1,430 episodes (~143K transitions)
2. Run health checks to identify coverage gaps
3. Generate additional episodes targeting weak areas
4. Repeat until health checks pass

**Contrastive policy allocation**:
- 80-85% from threshold-based variants (structured coverage)
- 15-20% from random/anti-kasmina/periodic (contrastive signal)

This approach avoids wasting compute on well-covered combinations while ensuring adequate coverage everywhere.

---

## State Representation

The state observation must include sufficient history for the MDP to be fully observable.

### Required State Features

**Current metrics** (already captured):
- epoch, global_step
- train_loss, val_loss, loss_delta
- train_accuracy, val_accuracy, accuracy_delta
- plateau_epochs, best_val_accuracy

**History/trend features** (verify captured):
- loss_history_5: Last 5 loss values
- accuracy_history_5: Last 5 accuracy values
- Computed trends: `accuracy_delta_5step`, `loss_trend_slope`

**Seed state** (already captured):
- has_active_seed, seed_stage
- seed_epochs_in_stage, seed_alpha
- seed_improvement

**Training progress**:
- `training_budget_remaining = (max_epochs - epoch) / max_epochs`

### Verifying State Completeness

Before generation, verify:
```python
def verify_state_features(observation):
    """Check that state has all required features."""
    required = [
        'epoch', 'val_accuracy', 'loss_delta', 'accuracy_delta',
        'plateau_epochs', 'loss_history_5', 'accuracy_history_5',
        'has_active_seed', 'seed_stage', 'available_slots'
    ]
    missing = [f for f in required if f not in observation]
    if missing:
        raise ValueError(f"Missing state features: {missing}")
```

---

## Data Schema Version

Schema version: `2.0.0`

Changes from v1:
- Added `behavior_policy` metadata (full config)
- Added `environment` metadata (full config)
- Added `action_probabilities` per step (with explicit μ(a|s))
- Added `reward_components` per step (replaces single reward value)
- Added `return_to_go` per step (for Decision Transformer)
- Added `random_seed` for reproducibility
- Added `step_metadata` with done/truncated/state_hash
- Added `trajectory_quality` label per episode

Backward compatibility: v2 readers can load v1 data by treating missing fields as defaults.

---

## Future Considerations (Deferred)

The following were discussed but deferred to future data generation rounds:

- **Blending variation**: Different alpha schedules for seed integration
- **Dataset variation**: CIFAR-100, distribution shifts
- **Counterfactual branching**: Actually running alternative actions from same state
- **Model checkpointing**: Save model state mid-episode for later counterfactual rollouts

These can be added once we train on this data and identify where Tamiyo struggles.

---

## Resolved Questions

1. **Target episode count**: 1,430 minimum (10 per combination), adaptive up to 2,000+
2. **How to handle diverged episodes**: Log with `termination_reason="diverged"`, include for contrastive signal
3. **Behavior policy probability**: Explicitly compute μ(a|s) = (1-ε) * π_greedy + ε/|A|

---

## Summary

| Dimension | Count |
|-----------|-------|
| Environment configs | 13 |
| Behavior policies | 11 |
| Combinations | 143 |
| Min episodes/combo | 10 |
| Baseline episodes | 1,430 |
| Target transitions | 100K-200K |

This design addresses the core problem (87% single-action coverage) through:
1. Policy variation (11 variants including contrastive)
2. ε-greedy exploration (local counterfactuals)
3. Environment variation (different training dynamics)
4. Rich metadata (importance sampling, reward relabeling, DT support)
