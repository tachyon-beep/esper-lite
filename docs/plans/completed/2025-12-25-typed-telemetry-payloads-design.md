# Typed Telemetry Payloads Design

**Date:** 2025-12-25
**Status:** Approved
**Author:** Claude + DRL Expert + PyTorch Expert review

## Problem Statement

The `TelemetryEvent.data` field is an untyped `dict[str, Any]`:

```python
# leyline/telemetry.py line 120
data: dict[str, Any] = field(default_factory=dict)  # No schema!
```

This causes:
- **156 `.get()` calls** in the aggregator as defensive programming
- **Silent data quality issues** - missing fields return defaults instead of errors
- **No compile-time guarantees** - producer/consumer drift undetected
- **No IDE autocomplete** - developers must read emitter code to know fields

### Bug Example (Fixed)

```python
# BEFORE: Silently drops legitimate zero returns
episode_return = data.get("avg_reward", 0.0)
if episode_return != 0.0:  # Bug! Zero is valid
    self._tamiyo.current_episode_return = episode_return

# AFTER: Only skip if truly missing
episode_return = data.get("avg_reward")
if episode_return is not None:
    self._tamiyo.current_episode_return = float(episode_return)
```

## Solution: Typed Payload Dataclasses

Replace `dict[str, Any]` with a union of strongly-typed dataclasses per event type.

### Design Principles

1. **Required fields raise KeyError** - fail-fast on missing data
2. **Optional fields have explicit defaults** - legitimately nullable
3. **Immutable payloads** - `frozen=True` prevents accidental mutation
4. **Memory efficient** - `slots=True` for high-frequency events
5. **Backward compatible during migration** - `dict` fallback removed when complete

## Payload Specifications

### TrainingStartedPayload

Emitted once at training start.

```python
@dataclass(slots=True, frozen=True)
class TrainingStartedPayload:
    # REQUIRED - training fails without these
    n_envs: int
    max_epochs: int
    task: str
    host_params: int  # Must be post-materialization
    slot_ids: tuple[str, ...]
    seed: int
    n_episodes: int
    lr: float
    clip_ratio: float
    entropy_coef: float
    param_budget: int
    policy_device: str
    env_devices: tuple[str, ...]

    # OPTIONAL - legitimate defaults
    episode_id: str = ""
    resume_path: str = ""
    reward_mode: str = ""
    start_episode: int = 0
    entropy_anneal: dict[str, float] | None = None

    # Distributed training (PyTorch expert recommendation)
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # AMP config (PyTorch expert recommendation)
    amp_enabled: bool = False
    amp_dtype: str | None = None  # "float16" or "bfloat16"

    # torch.compile config (PyTorch expert recommendation)
    compile_enabled: bool = False
    compile_backend: str | None = None
    compile_mode: str | None = None
```

### EpochCompletedPayload

Emitted per environment per epoch.

```python
@dataclass(slots=True, frozen=True)
class EpochCompletedPayload:
    # REQUIRED
    env_id: int
    val_accuracy: float
    val_loss: float
    inner_epoch: int

    # OPTIONAL - per-seed telemetry snapshots
    seeds: dict[str, dict[str, Any]] | None = None
```

### BatchEpochCompletedPayload

Emitted at episode boundary (commit barrier).

```python
@dataclass(slots=True, frozen=True)
class BatchEpochCompletedPayload:
    # REQUIRED (DRL expert: essential for metric normalization)
    episodes_completed: int
    batch_idx: int
    avg_accuracy: float
    avg_reward: float
    total_episodes: int
    n_envs: int

    # OPTIONAL
    rolling_accuracy: float = 0.0
    env_accuracies: tuple[float, ...] | None = None
```

### PPOUpdatePayload

Emitted after each PPO update.

```python
@dataclass(slots=True, frozen=True)
class PPOUpdatePayload:
    # REQUIRED - core PPO health metrics
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float  # Use float('inf') for AMP overflow
    kl_divergence: float
    clip_fraction: float
    nan_grad_count: int  # DRL expert: fail-fast on NaN

    # OPTIONAL - explained_variance can be NaN early training
    explained_variance: float | None = None

    # OPTIONAL - extended diagnostics
    entropy_loss: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    ratio_mean: float = 1.0
    ratio_min: float = 1.0
    ratio_max: float = 1.0
    ratio_std: float = 0.0
    lr: float | None = None
    entropy_coef: float | None = None

    # Gradient health (PyTorch expert: inf separate from nan)
    inf_grad_count: int = 0
    dead_layers: int = 0
    exploding_layers: int = 0
    layer_gradient_health: dict[str, float] | None = None
    entropy_collapsed: bool = False

    # AMP diagnostics (PyTorch expert recommendation)
    loss_scale: float | None = None
    amp_overflow_detected: bool = False
    update_skipped: bool = False

    # Timing
    update_time_ms: float = 0.0
    early_stop_epoch: int | None = None

    # Multi-head entropy (optional, only for factored policies)
    head_slot_entropy: dict[str, float] | None = None
    head_blueprint_entropy: dict[str, float] | None = None
    head_slot_grad_norm: dict[str, float] | None = None
    head_blueprint_grad_norm: dict[str, float] | None = None
    head_style_entropy: dict[str, float] | None = None
    head_tempo_entropy: dict[str, float] | None = None
    head_alpha_target_entropy: dict[str, float] | None = None
    head_alpha_speed_entropy: dict[str, float] | None = None
    head_alpha_curve_entropy: dict[str, float] | None = None
    head_op_entropy: dict[str, float] | None = None

    # PPO inner loop context
    inner_epoch: int = 0
    batch: int = 0

    # Skipped update flag
    skipped: bool = False
```

### RewardComputedPayload

Emitted per RL step with reward breakdown.

```python
@dataclass(slots=True, frozen=True)
class RewardComputedPayload:
    # REQUIRED (DRL expert: value_estimate and action_confidence essential)
    env_id: int
    total_reward: float
    action_name: str
    value_estimate: float
    action_confidence: float

    # OPTIONAL - reward component breakdown
    base_acc_delta: float = 0.0
    bounded_attribution: float = 0.0
    seed_contribution: float = 0.0
    compute_rent: float = 0.0
    alpha_shock: float = 0.0
    ratio_penalty: float = 0.0
    stage_bonus: float = 0.0
    fossilize_terminal_bonus: float = 0.0
    blending_warning: float = 0.0
    holding_warning: float = 0.0
    val_acc: float = 0.0

    # Decision context
    slot_states: dict[str, dict[str, Any]] | None = None
    host_accuracy: float = 0.0
    alternatives: list[dict[str, Any]] | None = None
    decision_entropy: float = 0.0
    ab_group: str | None = None
    action_slot: str | None = None
```

### SeedGerminatedPayload

Emitted when a seed is germinated.

```python
@dataclass(slots=True, frozen=True)
class SeedGerminatedPayload:
    # REQUIRED
    slot_id: str
    env_id: int
    blueprint_id: str
    params: int

    # OPTIONAL
    alpha: float = 0.0
    grad_ratio: float = 0.0
    has_vanishing: bool = False
    has_exploding: bool = False
    epochs_in_stage: int = 0
    blend_tempo_epochs: int = 5
```

### SeedStageChangedPayload

Emitted on seed stage transitions.

```python
@dataclass(slots=True, frozen=True)
class SeedStageChangedPayload:
    # REQUIRED
    slot_id: str
    env_id: int
    from_stage: str
    to_stage: str

    # OPTIONAL
    alpha: float | None = None
    accuracy_delta: float = 0.0
    epochs_in_stage: int = 0
    grad_ratio: float = 0.0
    has_vanishing: bool = False
    has_exploding: bool = False
```

### SeedFossilizedPayload

Emitted when a seed is fossilized (permanently grafted).

```python
@dataclass(slots=True, frozen=True)
class SeedFossilizedPayload:
    # REQUIRED
    slot_id: str
    env_id: int
    blueprint_id: str
    improvement: float
    params_added: int

    # OPTIONAL
    alpha: float = 1.0
    epochs_total: int = 0
    counterfactual: float = 0.0
```

### SeedPrunedPayload

Emitted when a seed is pruned (removed).

```python
@dataclass(slots=True, frozen=True)
class SeedPrunedPayload:
    # REQUIRED
    slot_id: str
    env_id: int
    reason: str

    # OPTIONAL
    blueprint_id: str | None = None
    improvement: float = 0.0
    auto_pruned: bool = False
    epochs_total: int = 0
    counterfactual: float = 0.0
```

### CounterfactualMatrixPayload

Emitted with factorial ablation results.

```python
@dataclass(slots=True, frozen=True)
class CounterfactualMatrixPayload:
    # REQUIRED
    env_id: int
    slot_ids: tuple[str, ...]
    configs: tuple[dict[str, Any], ...]

    # OPTIONAL
    strategy: str = "unavailable"
    compute_time_ms: float = 0.0
```

### AnalyticsSnapshotPayload

Emitted for dashboard sync and action distribution.

```python
@dataclass(slots=True, frozen=True)
class AnalyticsSnapshotPayload:
    # REQUIRED
    kind: str  # "action_distribution" or "last_action"

    # OPTIONAL - depends on kind
    action_counts: dict[str, int] | None = None
    # For kind="last_action", includes RewardComputedPayload fields
    env_id: int | None = None
    total_reward: float | None = None
    action_name: str | None = None
    action_confidence: float | None = None
    value_estimate: float | None = None
```

## TelemetryEvent Changes

```python
# Type alias for the payload union
TelemetryPayload = (
    TrainingStartedPayload
    | EpochCompletedPayload
    | BatchEpochCompletedPayload
    | PPOUpdatePayload
    | RewardComputedPayload
    | SeedGerminatedPayload
    | SeedStageChangedPayload
    | SeedFossilizedPayload
    | SeedPrunedPayload
    | CounterfactualMatrixPayload
    | AnalyticsSnapshotPayload
)

@dataclass
class TelemetryEvent:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: TelemetryEventType = TelemetryEventType.EPOCH_COMPLETED
    timestamp: datetime = field(default_factory=_utc_now)

    # Context
    seed_id: str | None = None
    slot_id: str | None = None
    epoch: int | None = None
    group_id: str = "default"

    # TYPED PAYLOAD (replaces dict[str, Any])
    data: TelemetryPayload | dict[str, Any] = field(default_factory=dict)

    # Metadata
    message: str = ""
    severity: str = "info"
```

**Note:** The `dict[str, Any]` fallback is temporary during migration and MUST be removed once all emitters are updated.

## Migration Strategy

### Phase 1: Add Payload Dataclasses
- Add all 11 payload dataclasses to `leyline/telemetry.py`
- Add `from_dict()` classmethod to each
- Add `TelemetryPayload` type alias
- Update `TelemetryEvent.data` type annotation

### Phase 2: Update Emitters
Update each emitter to construct typed payloads:
- `simic/training/vectorized.py` - PPO updates, rewards, batch epochs
- `nissa/output.py` - Training started, epoch completed
- `kasmina/slot.py` - Seed lifecycle events

### Phase 3: Update Aggregator
Replace `.get()` calls with direct attribute access:
```python
# BEFORE
policy_loss = data.get("policy_loss", 0.0)

# AFTER
if isinstance(event.data, PPOUpdatePayload):
    policy_loss = event.data.policy_loss
```

### Phase 4: Remove Fallback
- Remove `dict[str, Any]` from union type
- Delete any remaining `.get()` calls
- Run full test suite

## Verification

1. **Type checking:** `mypy src/esper/` passes
2. **Tests:** All existing tests pass
3. **Runtime:** Training runs complete without KeyError
4. **Grep audit:** `grep -c "\.get(" aggregator.py` returns 0

## Expert Reviews

### DRL Expert Recommendations (Incorporated)
- `explained_variance` → OPTIONAL (NaN when Var(returns) ≈ 0)
- `nan_grad_count` → REQUIRED with default 0
- `value_estimate`, `action_confidence` → REQUIRED in RewardComputedPayload
- `total_episodes`, `n_envs` → REQUIRED in BatchEpochCompletedPayload

### PyTorch Expert Recommendations (Incorporated)
- Add `world_size`, `rank`, `local_rank` for distributed training
- Add `amp_enabled`, `amp_dtype`, `loss_scale`, `amp_overflow_detected` for AMP
- Add `compile_enabled`, `compile_backend`, `compile_mode` for torch.compile
- Add `inf_grad_count` separate from `nan_grad_count`
- Add `update_skipped` flag for AMP overflow handling
