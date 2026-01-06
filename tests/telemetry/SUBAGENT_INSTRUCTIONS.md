# Telemetry Test Suite - Subagent Instructions

## Your Mission

Write end-to-end tests that verify telemetry metrics flow from their **source point** (where computed) through to **nissa** (where collected for external consumption).

Each test proves: "When X happens in training, metric Y is emitted with correct value Z."

## Skills to Use

Before writing tests, invoke these skills:
- `ordis-quality-engineering:test-suite-reviewer` - Ensure test design follows best practices
- `axiom-python-engineering:python-code-reviewer` - Review your test code

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY FLOW                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SOURCE (where computed)           TRANSPORT              SINK       │
│  ─────────────────────            ─────────              ────       │
│                                                                      │
│  PPOAgent._update()          →   VectorizedEmitter   →   NissaHub   │
│  (simic/agent/ppo_agent.py)      (simic/telemetry/       (nissa/    │
│                                   emitters.py)            output.py) │
│                                                                      │
│  VectorizedTrainer           →   emit_ppo_update()   →   backends   │
│  (simic/training/                                                    │
│   vectorized.py)                                                     │
│                                                                      │
│  Slot.step()                 →   emit_with_env_      →   FileOutput │
│  (kasmina/slot.py)               context()               ConsoleOut │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Test Pattern

Each test should:

1. **Set up a capture backend** - Collect emitted events
2. **Trigger the source** - Run the code that computes/emits the metric
3. **Find the event** - Locate the expected event type
4. **Assert the field** - Verify the metric value is present and correct

### Standard Test Template

```python
"""End-to-end telemetry test for TELE-XXX: metric_name."""

import pytest
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import NissaHub


class CaptureBackend:
    """Test backend that captures all emitted events."""

    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def find_events(self, event_type: TelemetryEventType) -> list[TelemetryEvent]:
        return [e for e in self.events if e.event_type == event_type]


class TestTELE_XXX_MetricName:
    """Verify metric_name flows from source to nissa."""

    def test_metric_emitted_on_trigger(self):
        """TELE-XXX: metric_name is emitted when [trigger condition]."""
        # 1. Set up capture
        backend = CaptureBackend()
        hub = NissaHub()
        hub.add_backend(backend)

        # 2. Trigger the source (create component with hub, call method)
        # ... component-specific setup ...

        # 3. Find event
        events = backend.find_events(TelemetryEventType.PPO_UPDATE_COMPLETED)
        assert len(events) >= 1, "Expected PPO_UPDATE_COMPLETED event"

        # 4. Assert field
        event = events[0]
        assert event.data.metric_name is not None
        assert event.data.metric_name == expected_value
```

## Key Files to Reference

### Telemetry Contracts (leyline)
- `src/esper/leyline/telemetry.py` - All payload dataclasses:
  - `PPOUpdatePayload` - PPO training metrics (entropy, kl, clip_fraction, etc.)
  - `TrainingStartedPayload` - Training config (task, n_envs, etc.)
  - `BatchEpochCompletedPayload` - Episode completion
  - `AnalyticsSnapshotPayload` - Decision context, reward breakdown
  - `SeedGerminatedPayload`, `SeedFossilizedPayload`, etc. - Seed lifecycle

### Emitters (simic)
- `src/esper/simic/telemetry/emitters.py` - `VectorizedEmitter` class
- `src/esper/simic/agent/ppo_agent.py` - PPO metrics computation

### Nissa Hub
- `src/esper/nissa/output.py` - `NissaHub`, `OutputBackend` protocol

### Telemetry Requirements
- `docs/telemetry/telemetry_needs/TELE-*.md` - Each requirement doc

## Payload Field Reference

### PPOUpdatePayload (TelemetryEventType.PPO_UPDATE_COMPLETED)

Core metrics (always required):
- `policy_loss: float`
- `value_loss: float`
- `entropy: float`
- `grad_norm: float`
- `kl_divergence: float`
- `clip_fraction: float`
- `nan_grad_count: int`

Extended diagnostics:
- `advantage_mean`, `advantage_std`, `advantage_skewness`, `advantage_kurtosis`
- `explained_variance: float | None`
- `clip_fraction_positive`, `clip_fraction_negative`
- `gradient_cv: float`
- `dead_layers: int`, `exploding_layers: int`
- `head_*_entropy`, `head_*_grad_norm` (per-head metrics)
- `value_mean`, `value_std`, `value_min`, `value_max`

### TrainingStartedPayload (TelemetryEventType.TRAINING_STARTED)

- `n_envs`, `max_epochs`, `max_batches`, `task`
- `lr`, `clip_ratio`, `entropy_coef`
- `compile_enabled`, `amp_enabled`

### AnalyticsSnapshotPayload (TelemetryEventType.ANALYTICS_SNAPSHOT)

When `kind="last_action"`:
- `action_name`, `action_confidence`, `value_estimate`
- `total_reward`, `reward_components` (nested dataclass)
- `head_telemetry` (per-head confidence/entropy)

When `kind="reward_summary"`:
- `summary: dict[str, float]` - Reward component breakdown

## Test Categories by TELE ID Range

| Range | Category | Event Type | Source File |
|-------|----------|------------|-------------|
| 001-099 | Training | TRAINING_STARTED, BATCH_EPOCH_COMPLETED | vectorized.py |
| 100-199 | Policy | PPO_UPDATE_COMPLETED | ppo_agent.py, emitters.py |
| 200-299 | Value | PPO_UPDATE_COMPLETED | ppo_agent.py |
| 300-399 | Gradient | PPO_UPDATE_COMPLETED | ppo_agent.py, gradient_collector.py |
| 400-499 | Reward | ANALYTICS_SNAPSHOT | reward_telemetry.py |
| 500-599 | Seed | SEED_* events | slot.py, emitters.py |
| 600-699 | Environment | EPOCH_COMPLETED | vectorized.py |
| 700-799 | Infrastructure | TRAINING_STARTED, system queries | train.py |
| 800-899 | Decision | ANALYTICS_SNAPSHOT | emitters.py |

## Test File Naming

```
tests/telemetry/
├── __init__.py
├── conftest.py                    # Shared fixtures (CaptureBackend, hub setup)
├── test_training_metrics.py       # TELE-001 to TELE-099
├── test_policy_metrics.py         # TELE-100 to TELE-199
├── test_value_metrics.py          # TELE-200 to TELE-299
├── test_gradient_metrics.py       # TELE-300 to TELE-399
├── test_reward_metrics.py         # TELE-400 to TELE-499
├── test_seed_lifecycle.py         # TELE-500 to TELE-599
├── test_environment_metrics.py    # TELE-600 to TELE-699
├── test_infrastructure_metrics.py # TELE-700 to TELE-799
├── test_decision_metrics.py       # TELE-800 to TELE-899
└── SUBAGENT_INSTRUCTIONS.md       # This file
```

## Triggering Sources

### For PPO Metrics (TELE-100 to TELE-399)

```python
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import FactoredPolicy

# Create minimal policy and agent
policy = FactoredPolicy(...)  # or use a mock
agent = PPOAgent(policy=policy, ...)

# Inject emitter with capture hub
from esper.simic.telemetry.emitters import VectorizedEmitter
emitter = VectorizedEmitter(env_id=0, device="cpu", hub=hub)

# Call update method to trigger PPO_UPDATE_COMPLETED
agent.update(rollout_buffer, emitter=emitter)
```

### For Seed Lifecycle (TELE-500 to TELE-599)

```python
from esper.kasmina.slot import Slot
from esper.leyline import SeedStage

# Create slot with telemetry callback
slot = Slot(slot_id="r0c0", telemetry_cb=hub.emit)

# Trigger lifecycle events
slot.germinate(blueprint=..., host=...)  # SEED_GERMINATED
slot.step(host_grad_norm=0.5, ...)       # May trigger stage changes
slot.fossilize(host=...)                 # SEED_FOSSILIZED
```

### For Training Started (TELE-001, TELE-700s)

```python
from esper.leyline import TrainingStartedPayload, TelemetryEvent, TelemetryEventType

# Emit directly (testing the payload structure)
payload = TrainingStartedPayload(
    n_envs=4,
    max_epochs=150,
    max_batches=100,
    task="cifar10",
    ...
)
event = TelemetryEvent(
    event_type=TelemetryEventType.TRAINING_STARTED,
    data=payload,
)
hub.emit(event)
```

## What NOT to Test

- Don't test Sanctum aggregator/widgets (those are consumers, not sources)
- Don't test MCP views (already tested in test_reward_telemetry_flow.py)
- Don't duplicate existing tests (check tests/simic/test_telemetry_*.py first)

## Verifying Wiring Gaps

For TELE-600 to TELE-603 (observation stats with wiring gaps):
- Write tests that DOCUMENT the gap (test that field is NOT populated)
- Mark test with `@pytest.mark.xfail(reason="Wiring gap: emitter not implemented")`

```python
@pytest.mark.xfail(reason="TELE-600 wiring gap: obs_nan_count emitter not implemented")
def test_obs_nan_count_emitted(self):
    """TELE-600: obs_nan_count should be emitted but emitter is missing."""
    # Test that would pass if wiring was complete
    ...
```

## Deliverable

Create ONE test file for your assigned category. Include:
1. Module docstring explaining what's tested
2. Shared fixtures in the file (or use conftest.py)
3. One test class per TELE-ID or logical grouping
4. Clear test names: `test_TELE_XXX_field_name_condition`
5. Docstring with TELE-ID for traceability

## Example Complete Test

```python
"""End-to-end tests for policy metrics (TELE-100 to TELE-199).

Verifies PPO training metrics flow from PPOAgent through to nissa.
"""

import pytest
from esper.leyline import TelemetryEvent, TelemetryEventType, PPOUpdatePayload
from esper.nissa.output import NissaHub


class CaptureBackend:
    """Captures all emitted telemetry events for testing."""

    def __init__(self):
        self.events: list[TelemetryEvent] = []

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)


@pytest.fixture
def capture_hub():
    """Create hub with capture backend."""
    hub = NissaHub()
    backend = CaptureBackend()
    hub.add_backend(backend)
    yield hub, backend
    hub.close()


class TestTELE120Entropy:
    """TELE-120: Policy entropy metric."""

    def test_entropy_in_ppo_update_payload(self, capture_hub):
        """TELE-120: entropy field is present in PPO_UPDATE_COMPLETED."""
        hub, backend = capture_hub

        # Create and emit PPO update with known entropy
        payload = PPOUpdatePayload(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=1.234,  # The metric we're testing
            grad_norm=0.8,
            kl_divergence=0.01,
            clip_fraction=0.15,
            nan_grad_count=0,
            pre_clip_grad_norm=0.9,
            advantage_mean=0.0,
            advantage_std=1.0,
            advantage_skewness=0.0,
            advantage_kurtosis=0.0,
            advantage_positive_ratio=0.5,
            ratio_mean=1.0,
            ratio_min=0.9,
            ratio_max=1.1,
            ratio_std=0.05,
            log_prob_min=-5.0,
            log_prob_max=-0.5,
            entropy_collapsed=False,
            update_time_ms=10.0,
            inner_epoch=1,
            batch=0,
            ppo_updates_count=4,
            value_mean=0.5,
            value_std=0.1,
            value_min=0.0,
            value_max=1.0,
            clip_fraction_positive=0.1,
            clip_fraction_negative=0.05,
            gradient_cv=0.3,
            pre_norm_advantage_mean=0.0,
            pre_norm_advantage_std=1.0,
            return_mean=0.5,
            return_std=0.2,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=payload,
        )
        hub.emit(event)

        # Verify
        events = [e for e in backend.events
                  if e.event_type == TelemetryEventType.PPO_UPDATE_COMPLETED]
        assert len(events) == 1
        assert events[0].data.entropy == 1.234
```
