# Esper V1.0 Architecture Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate flat esper module to domain-organized package structure (leyline, kasmina, tamiyo, simic, nissa) using strangler fig pattern.

**Architecture:** Five domain packages reflecting MTG-themed subsystems. Leyline owns contracts, Kasmina owns seed mechanics, Tamiyo owns strategic decisions, Simic owns RL training, Nissa owns system telemetry. Each domain owns its telemetry emissions using leyline-defined contracts.

**Tech Stack:** Python 3.11+, PyTorch, dataclasses, NamedTuples

---

## Target Structure

```
src/esper/
├── leyline/               # Contracts - the invisible substrate
│   ├── __init__.py        # Re-exports all public contracts
│   ├── actions.py         # SimicAction enum
│   ├── signals.py         # TrainingSignals, FastTrainingSignals, TensorSchema
│   ├── stages.py          # SeedStage, transitions, CommandType, RiskLevel
│   ├── schemas.py         # AdaptationCommand, BlueprintSpec, BlueprintProtocol
│   ├── reports.py         # FieldReport, SeedStateReport, TrainingMetrics
│   └── telemetry.py       # TelemetryEvent, TelemetryEventType (contracts only)
│
├── kasmina/               # Seed mechanics
│   ├── __init__.py
│   ├── slot.py            # SeedSlot, SeedState, SeedMetrics, QualityGates
│   ├── blueprints.py      # BlueprintCatalog, ConvEnhanceSeed, AttentionSeed, etc.
│   ├── isolation.py       # GradientIsolationMonitor, AlphaSchedule, blend_with_isolation
│   └── host.py            # HostCNN, MorphogeneticModel, ConvBlock
│
├── tamiyo/                # Strategic decision-making
│   ├── __init__.py
│   ├── heuristic.py       # HeuristicTamiyo, HeuristicPolicyConfig, TamiyoPolicy
│   ├── decisions.py       # TamiyoDecision, TamiyoAction
│   └── tracker.py         # SignalTracker
│
├── simic/                 # RL training
│   ├── __init__.py
│   ├── ppo.py             # PPOAgent, vectorized training loops
│   ├── iql.py             # IQL implementation
│   ├── rewards.py         # compute_shaped_reward, RewardConfig, SeedInfo
│   ├── episodes.py        # Episode, EpisodeCollector, DatasetManager
│   ├── features.py        # snapshot_from_signals, obs_to_base_features
│   └── networks.py        # PolicyNetwork, shared architectures
│
├── nissa/                 # System telemetry hub
│   ├── __init__.py
│   ├── tracker.py         # DiagnosticTracker, GradientStats, EpochSnapshot
│   ├── output.py          # Console/file output backends
│   └── config.py          # TelemetryConfig
│
├── datagen/               # (unchanged)
│
└── scripts/               # Entry points
    ├── train.py           # PPO training entry
    ├── generate.py        # Heuristic data generation
    └── evaluate.py        # Head-to-head comparison
```

---

## Architectural Constraints

### Hot Path Rule (simic/features.py)
The feature extraction module MUST only import from `leyline/`:
- `from esper.leyline.actions import SimicAction`
- `from esper.leyline.signals import FastTrainingSignals, TensorSchema`

**Forbidden imports in simic/features.py:**
- `from esper.kasmina import ...`  # Heavy seed state objects
- `from esper.nissa import ...`    # Telemetry trackers
- `from esper.tamiyo import ...`   # Decision infrastructure

This ensures the vectorized PPO loop stays allocation-free and fast.

### Telemetry Ownership
- Each domain owns telemetry it emits (kasmina emits seed events, simic emits training events)
- Contracts for telemetry events are defined in `leyline/telemetry.py`
- Nissa receives carbon copies and handles output (console/file)

---

## Phase 1: Create Package Structure

### Task 1.1: Create empty package directories

**Files:**
- Create: `src/esper/leyline/__init__.py`
- Create: `src/esper/kasmina/__init__.py`
- Create: `src/esper/tamiyo/__init__.py`
- Create: `src/esper/simic/__init__.py`
- Create: `src/esper/nissa/__init__.py`
- Create: `src/esper/scripts/__init__.py`

**Step 1: Create directories and empty __init__.py files**

```bash
mkdir -p src/esper/leyline src/esper/kasmina src/esper/tamiyo src/esper/simic src/esper/nissa src/esper/scripts
touch src/esper/leyline/__init__.py
touch src/esper/kasmina/__init__.py
touch src/esper/tamiyo/__init__.py
touch src/esper/simic/__init__.py
touch src/esper/nissa/__init__.py
touch src/esper/scripts/__init__.py
```

**Step 2: Verify structure**

```bash
find src/esper -type d | sort
```

Expected:
```
src/esper
src/esper/datagen
src/esper/kasmina
src/esper/leyline
src/esper/nissa
src/esper/scripts
src/esper/simic
src/esper/tamiyo
```

**Step 3: Commit**

```bash
git add src/esper/leyline src/esper/kasmina src/esper/tamiyo src/esper/simic src/esper/nissa src/esper/scripts
git commit -m "chore: create package structure for esper v1.0 architecture"
```

---

## Phase 2: Migrate Leyline (Contracts)

Leyline is the foundation - everything depends on it. We migrate it first.

### Task 2.1: Create leyline/actions.py

**Files:**
- Create: `src/esper/leyline/actions.py`
- Source: `src/esper/simic.py` lines 229-259 (SimicAction enum)

**Step 1: Create actions.py with SimicAction**

```python
"""Leyline Actions - Action space definitions for Esper agents.

Actions represent the discrete choices available to the strategic controller.
"""

from enum import Enum


class SimicAction(Enum):
    """Actions available to the Simic RL agent.

    The action space for seed lifecycle control:
    - WAIT: Continue training without intervention
    - GERMINATE_*: Create a new seed with specific blueprint
    - ADVANCE: Move seed to next lifecycle stage
    - CULL: Remove underperforming seed
    """

    WAIT = 0
    GERMINATE_CONV = 1
    GERMINATE_ATTENTION = 2
    GERMINATE_NORM = 3
    GERMINATE_DEPTHWISE = 4
    ADVANCE = 5
    CULL = 6

    @classmethod
    def is_germinate(cls, action: "SimicAction") -> bool:
        """Check if action is any GERMINATE variant."""
        return action in (
            cls.GERMINATE_CONV,
            cls.GERMINATE_ATTENTION,
            cls.GERMINATE_NORM,
            cls.GERMINATE_DEPTHWISE,
        )

    @classmethod
    def get_blueprint_id(cls, action: "SimicAction") -> str:
        """Get blueprint ID for a GERMINATE action."""
        mapping = {
            cls.GERMINATE_CONV: "conv_enhance",
            cls.GERMINATE_ATTENTION: "attention",
            cls.GERMINATE_NORM: "norm",
            cls.GERMINATE_DEPTHWISE: "depthwise",
        }
        return mapping.get(action, "conv_enhance")
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.leyline.actions import SimicAction; print(SimicAction.WAIT)"
```

Expected: `SimicAction.WAIT`

**Step 3: Commit**

```bash
git add src/esper/leyline/actions.py
git commit -m "feat(leyline): add actions.py with SimicAction enum"
```

---

### Task 2.2: Create leyline/stages.py

**Files:**
- Create: `src/esper/leyline/stages.py`
- Source: `src/esper/leyline.py` lines 182-270 (SeedStage, transitions, CommandType, RiskLevel)

**Step 1: Create stages.py**

```python
"""Leyline Stages - Seed lifecycle stages and transitions.

Defines the state machine for seed development:
DORMANT -> GERMINATING -> TRAINING -> BLENDING -> FOSSILIZED
                                  \-> CULLED (terminal failure)
"""

from enum import Enum, IntEnum


class SeedStage(IntEnum):
    """Lifecycle stages for a seed module.

    Numeric values chosen for:
    - Easy comparison (higher = more mature)
    - Feature vector embedding
    - Ordering in lifecycle progression
    """

    DORMANT = 1        # Not yet activated
    GERMINATING = 2    # Being initialized
    TRAINING = 3       # Actively learning
    BLENDING = 4       # Merging with host
    CULLED = 5         # Removed (terminal)
    FAILED = 6         # Error state (terminal)
    FOSSILIZED = 7     # Fully integrated (terminal success)


# Valid transitions in the lifecycle state machine
VALID_TRANSITIONS: dict[SeedStage, set[SeedStage]] = {
    SeedStage.DORMANT: {SeedStage.GERMINATING},
    SeedStage.GERMINATING: {SeedStage.TRAINING, SeedStage.FAILED},
    SeedStage.TRAINING: {SeedStage.BLENDING, SeedStage.CULLED},
    SeedStage.BLENDING: {SeedStage.FOSSILIZED, SeedStage.CULLED},
    SeedStage.CULLED: set(),      # Terminal
    SeedStage.FAILED: set(),      # Terminal
    SeedStage.FOSSILIZED: set(),  # Terminal
}


def is_valid_transition(from_stage: SeedStage, to_stage: SeedStage) -> bool:
    """Check if a stage transition is valid."""
    return to_stage in VALID_TRANSITIONS.get(from_stage, set())


def is_terminal_stage(stage: SeedStage) -> bool:
    """Check if stage is terminal (no further transitions)."""
    return stage in (SeedStage.CULLED, SeedStage.FAILED, SeedStage.FOSSILIZED)


def is_active_stage(stage: SeedStage) -> bool:
    """Check if stage represents active seed participation.

    Active stages are where the seed contributes to training.
    """
    return stage in (SeedStage.TRAINING, SeedStage.BLENDING)


def is_failure_stage(stage: SeedStage) -> bool:
    """Check if stage represents a failure outcome."""
    return stage in (SeedStage.CULLED, SeedStage.FAILED)


class CommandType(Enum):
    """Types of commands Tamiyo can issue to Kasmina."""

    WAIT = "wait"                    # No action this cycle
    GERMINATE = "germinate"          # Start a new seed
    ADVANCE_STAGE = "advance_stage"  # Move to next lifecycle stage
    CULL = "cull"                    # Remove seed
    ADJUST_BLEND = "adjust_blend"    # Modify blending parameters


class RiskLevel(IntEnum):
    """Risk assessment for adaptation commands."""

    NONE = 0       # No risk (WAIT)
    LOW = 1        # Safe operations
    MEDIUM = 2     # Moderate risk
    HIGH = 3       # Significant risk
    CRITICAL = 4   # Could destabilize training
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.leyline.stages import SeedStage, is_valid_transition; print(is_valid_transition(SeedStage.TRAINING, SeedStage.BLENDING))"
```

Expected: `True`

**Step 3: Commit**

```bash
git add src/esper/leyline/stages.py
git commit -m "feat(leyline): add stages.py with SeedStage lifecycle"
```

---

### Task 2.3: Create leyline/signals.py

**Files:**
- Create: `src/esper/leyline/signals.py`
- Source: `src/esper/leyline.py` lines 40-173 (TensorSchema, FastTrainingSignals)
- Source: `src/esper/leyline.py` lines 399-477 (TrainingSignals)

**Step 1: Create signals.py**

```python
"""Leyline Signals - Training state observations.

Two tiers of signal representation:
- FastTrainingSignals: NamedTuple for hot path (no GC pressure)
- TrainingSignals: Full dataclass for rich context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple


class TensorSchema(IntEnum):
    """Feature indices for the observation vector.

    Maps feature names to tensor indices for vectorized PPO training.
    Use this to slice state vectors by name without string lookups.

    Total: 27 base features (V1 compatible)
    """

    # Core state (2)
    EPOCH = 0
    GLOBAL_STEP = 1

    # Loss metrics (3)
    TRAIN_LOSS = 2
    VAL_LOSS = 3
    LOSS_DELTA = 4

    # Accuracy metrics (4)
    TRAIN_ACCURACY = 5
    VAL_ACCURACY = 6
    ACCURACY_DELTA = 7
    BEST_VAL_ACCURACY = 8

    # Trend indicators (2)
    PLATEAU_EPOCHS = 9
    BEST_VAL_LOSS = 10

    # History windows (10) - 5 loss + 5 accuracy
    LOSS_HISTORY_0 = 11
    LOSS_HISTORY_1 = 12
    LOSS_HISTORY_2 = 13
    LOSS_HISTORY_3 = 14
    LOSS_HISTORY_4 = 15
    ACC_HISTORY_0 = 16
    ACC_HISTORY_1 = 17
    ACC_HISTORY_2 = 18
    ACC_HISTORY_3 = 19
    ACC_HISTORY_4 = 20

    # Seed state (6)
    HAS_ACTIVE_SEED = 21
    SEED_STAGE = 22
    SEED_EPOCHS_IN_STAGE = 23
    SEED_ALPHA = 24
    SEED_IMPROVEMENT = 25
    AVAILABLE_SLOTS = 26


TENSOR_SCHEMA_SIZE = 27


class FastTrainingSignals(NamedTuple):
    """Lightweight training signals for hot path.

    This is the PRIMARY data structure for PPO vectorized training.
    Uses NamedTuple for zero-allocation access patterns.

    All fields are primitives - no objects, no allocations.
    """

    # Core state
    epoch: int
    global_step: int

    # Loss metrics
    train_loss: float
    val_loss: float
    loss_delta: float

    # Accuracy metrics
    train_accuracy: float
    val_accuracy: float
    accuracy_delta: float
    best_val_accuracy: float

    # Trend indicators
    plateau_epochs: int
    best_val_loss: float

    # History windows (flattened)
    loss_history_0: float
    loss_history_1: float
    loss_history_2: float
    loss_history_3: float
    loss_history_4: float
    acc_history_0: float
    acc_history_1: float
    acc_history_2: float
    acc_history_3: float
    acc_history_4: float

    # Seed state
    has_active_seed: bool
    seed_stage: int
    seed_epochs_in_stage: int
    seed_alpha: float
    seed_improvement: float
    available_slots: int

    def to_tensor_list(self) -> list[float]:
        """Convert to flat list for tensor creation."""
        return [
            float(self.epoch),
            float(self.global_step),
            self.train_loss,
            self.val_loss,
            self.loss_delta,
            self.train_accuracy,
            self.val_accuracy,
            self.accuracy_delta,
            self.best_val_accuracy,
            float(self.plateau_epochs),
            self.best_val_loss,
            self.loss_history_0,
            self.loss_history_1,
            self.loss_history_2,
            self.loss_history_3,
            self.loss_history_4,
            self.acc_history_0,
            self.acc_history_1,
            self.acc_history_2,
            self.acc_history_3,
            self.acc_history_4,
            float(self.has_active_seed),
            float(self.seed_stage),
            float(self.seed_epochs_in_stage),
            self.seed_alpha,
            self.seed_improvement,
            float(self.available_slots),
        ]


@dataclass
class TrainingSignals:
    """Full training signals with rich context.

    Use this for:
    - Heuristic decision making (needs full context)
    - Logging and telemetry
    - Episode recording

    For hot-path PPO, use FastTrainingSignals instead.
    """

    # Core training state
    epoch: int
    global_step: int
    max_epochs: int

    # Loss metrics
    train_loss: float
    val_loss: float
    loss_delta: float = 0.0

    # Accuracy metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    accuracy_delta: float = 0.0
    best_val_accuracy: float = 0.0

    # Trend tracking
    plateau_epochs: int = 0
    best_val_loss: float = float("inf")

    # History windows
    loss_history: list[float] = field(default_factory=lambda: [0.0] * 5)
    accuracy_history: list[float] = field(default_factory=lambda: [0.0] * 5)

    # Seed state (filled by Kasmina)
    has_active_seed: bool = False
    seed_stage: int = 0
    seed_epochs_in_stage: int = 0
    seed_alpha: float = 0.0
    seed_improvement: float = 0.0
    available_slots: int = 1

    def to_fast(self) -> FastTrainingSignals:
        """Convert to FastTrainingSignals for hot path."""
        return FastTrainingSignals(
            epoch=self.epoch,
            global_step=self.global_step,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            loss_delta=self.loss_delta,
            train_accuracy=self.train_accuracy,
            val_accuracy=self.val_accuracy,
            accuracy_delta=self.accuracy_delta,
            best_val_accuracy=self.best_val_accuracy,
            plateau_epochs=self.plateau_epochs,
            best_val_loss=self.best_val_loss,
            loss_history_0=self.loss_history[0] if len(self.loss_history) > 0 else 0.0,
            loss_history_1=self.loss_history[1] if len(self.loss_history) > 1 else 0.0,
            loss_history_2=self.loss_history[2] if len(self.loss_history) > 2 else 0.0,
            loss_history_3=self.loss_history[3] if len(self.loss_history) > 3 else 0.0,
            loss_history_4=self.loss_history[4] if len(self.loss_history) > 4 else 0.0,
            acc_history_0=self.accuracy_history[0] if len(self.accuracy_history) > 0 else 0.0,
            acc_history_1=self.accuracy_history[1] if len(self.accuracy_history) > 1 else 0.0,
            acc_history_2=self.accuracy_history[2] if len(self.accuracy_history) > 2 else 0.0,
            acc_history_3=self.accuracy_history[3] if len(self.accuracy_history) > 3 else 0.0,
            acc_history_4=self.accuracy_history[4] if len(self.accuracy_history) > 4 else 0.0,
            has_active_seed=self.has_active_seed,
            seed_stage=self.seed_stage,
            seed_epochs_in_stage=self.seed_epochs_in_stage,
            seed_alpha=self.seed_alpha,
            seed_improvement=self.seed_improvement,
            available_slots=self.available_slots,
        )
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.leyline.signals import TensorSchema, FastTrainingSignals, TrainingSignals; print(f'Schema size: {len(TensorSchema)}')"
```

Expected: `Schema size: 27`

**Step 3: Commit**

```bash
git add src/esper/leyline/signals.py
git commit -m "feat(leyline): add signals.py with TensorSchema and TrainingSignals"
```

---

### Task 2.4: Create leyline/schemas.py

**Files:**
- Create: `src/esper/leyline/schemas.py`
- Source: `src/esper/leyline.py` lines 302-368 (AdaptationCommand, SeedOperation)
- Source: `src/esper/leyline.py` lines 589-660 (BlueprintProtocol, BlueprintSpec, GateLevel, GateResult)

**Step 1: Create schemas.py**

```python
"""Leyline Schemas - Command and blueprint specifications.

Defines the structure of commands issued by controllers
and specifications for seed blueprints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from esper.leyline.stages import CommandType, RiskLevel


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class SeedOperation(Enum):
    """Specific operations for seed manipulation."""

    CREATE = "create"
    ADVANCE = "advance"
    BLEND = "blend"
    CULL = "cull"
    FOSSILIZE = "fossilize"


@dataclass
class AdaptationCommand:
    """Command from Tamiyo to Kasmina.

    Represents an instruction to modify the model's adaptation state.
    """

    command_type: CommandType
    command_id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = field(default_factory=_utc_now)

    # Optional parameters based on command type
    seed_operation: SeedOperation | None = None
    blueprint_id: str | None = None
    target_seed_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    rationale: str = ""


class GateLevel(IntEnum):
    """Quality gate strictness levels."""

    PERMISSIVE = 0   # Allow almost anything
    STANDARD = 1     # Normal quality checks
    STRICT = 2       # High quality requirements
    PARANOID = 3     # Maximum safety checks


@dataclass
class GateResult:
    """Result of a quality gate check."""

    passed: bool
    gate_name: str
    level: GateLevel
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class BlueprintProtocol(Protocol):
    """Protocol that all seed blueprints must implement.

    Blueprints define HOW to create a seed module.
    """

    @property
    def blueprint_id(self) -> str:
        """Unique identifier for this blueprint type."""
        ...

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters this blueprint adds."""
        ...

    def create_module(self, host_channels: int) -> Any:
        """Create the actual nn.Module for this seed."""
        ...


@dataclass
class BlueprintSpec:
    """Specification for a seed blueprint.

    Metadata about a blueprint without the implementation.
    """

    blueprint_id: str
    name: str
    description: str
    parameter_count: int
    risk_level: RiskLevel = RiskLevel.LOW
    recommended_stages: int = 10  # Typical epochs in TRAINING
    tags: list[str] = field(default_factory=list)
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.leyline.schemas import AdaptationCommand, BlueprintSpec, CommandType; print(AdaptationCommand(command_type=CommandType.WAIT))"
```

Expected: `AdaptationCommand(command_type=<CommandType.WAIT: 'wait'>, ...)`

**Step 3: Commit**

```bash
git add src/esper/leyline/schemas.py
git commit -m "feat(leyline): add schemas.py with AdaptationCommand and BlueprintSpec"
```

---

### Task 2.5: Create leyline/reports.py

**Files:**
- Create: `src/esper/leyline/reports.py`
- Source: `src/esper/leyline.py` lines 369-397 (TrainingMetrics)
- Source: `src/esper/leyline.py` lines 479-543 (SeedMetrics, SeedStateReport)
- Source: `src/esper/leyline.py` lines 545-587 (FieldReport)

**Step 1: Create reports.py**

```python
"""Leyline Reports - Structured reporting contracts.

Defines the shape of reports emitted by various subsystems:
- TrainingMetrics: Per-epoch training statistics
- SeedMetrics: Seed module performance
- SeedStateReport: Full seed state snapshot
- FieldReport: Episode summary for RL training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from esper.leyline.stages import SeedStage


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


@dataclass
class TrainingMetrics:
    """Metrics from a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    epoch_time_seconds: float = 0.0


@dataclass
class SeedMetrics:
    """Performance metrics for a seed module."""

    accuracy_at_stage_start: float = 0.0
    current_val_accuracy: float = 0.0
    loss_at_stage_start: float = float("inf")
    current_val_loss: float = float("inf")
    gradient_norm: float = 0.0
    parameter_delta: float = 0.0  # L2 distance from init


@dataclass
class SeedStateReport:
    """Complete state report for a seed.

    Snapshot of seed state for telemetry and decision-making.
    """

    seed_id: str
    blueprint_id: str
    stage: SeedStage
    epochs_in_stage: int
    total_epochs: int
    metrics: SeedMetrics
    alpha: float = 0.0  # Blending weight
    timestamp: datetime = field(default_factory=_utc_now)

    # Quality indicators
    is_healthy: bool = True
    health_notes: list[str] = field(default_factory=list)


@dataclass
class FieldReport:
    """Episode summary for Simic RL training.

    Captures everything needed to learn from an episode:
    - What the agent observed
    - What action it took
    - What reward it received
    - The outcome
    """

    # Episode identification
    episode_id: str
    step: int
    timestamp: datetime = field(default_factory=_utc_now)

    # Observation (what agent saw)
    observation: dict[str, Any] = field(default_factory=dict)

    # Action taken
    action: int = 0
    action_name: str = ""

    # Reward received
    reward: float = 0.0
    shaped_reward: float = 0.0

    # Outcome metrics
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    accuracy_delta: float = 0.0

    # Seed state at decision time
    seed_stage: int = 0
    seed_alpha: float = 0.0

    # Episode terminal info
    is_terminal: bool = False
    terminal_reason: str = ""
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.leyline.reports import FieldReport, SeedStateReport; print(FieldReport(episode_id='test', step=0))"
```

Expected: `FieldReport(episode_id='test', step=0, ...)`

**Step 3: Commit**

```bash
git add src/esper/leyline/reports.py
git commit -m "feat(leyline): add reports.py with FieldReport and SeedStateReport"
```

---

### Task 2.6: Create leyline/telemetry.py

**Files:**
- Create: `src/esper/leyline/telemetry.py`
- Source: `src/esper/leyline.py` lines 665-734 (TelemetryEventType, TelemetryEvent, PerformanceBudgets)

**Step 1: Create telemetry.py**

```python
"""Leyline Telemetry Contracts - Event definitions for system monitoring.

These are CONTRACTS only - the actual emission and handling
is done by domain-specific modules and Nissa.

Each domain emits events using these contracts:
- Kasmina: Seed lifecycle events
- Tamiyo: Decision events
- Simic: Training progress events
- Nissa: System health events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TelemetryEventType(Enum):
    """Categories of telemetry events."""

    # Seed lifecycle (Kasmina)
    SEED_GERMINATED = "seed.germinated"
    SEED_STAGE_CHANGED = "seed.stage_changed"
    SEED_CULLED = "seed.culled"
    SEED_FOSSILIZED = "seed.fossilized"

    # Decision events (Tamiyo)
    DECISION_MADE = "decision.made"
    DECISION_RATIONALE = "decision.rationale"

    # Training events (Simic)
    EPOCH_COMPLETED = "training.epoch_completed"
    EPISODE_COMPLETED = "training.episode_completed"
    BATCH_COMPLETED = "training.batch_completed"

    # System health (Nissa)
    GPU_UTILIZATION = "system.gpu_utilization"
    MEMORY_PRESSURE = "system.memory_pressure"
    THROUGHPUT_SAMPLE = "system.throughput"
    GRADIENT_HEALTH = "system.gradient_health"


@dataclass
class TelemetryEvent:
    """Base telemetry event structure.

    All telemetry events share this shape, making it easy
    for Nissa to collect and route them.
    """

    event_type: TelemetryEventType
    source: str  # e.g., "kasmina", "tamiyo", "simic"
    timestamp: datetime = field(default_factory=_utc_now)

    # Event-specific payload
    payload: dict[str, Any] = field(default_factory=dict)

    # Optional context
    episode_id: str | None = None
    epoch: int | None = None
    step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "episode_id": self.episode_id,
            "epoch": self.epoch,
            "step": self.step,
        }


@dataclass
class PerformanceBudgets:
    """Performance budgets for real-time training.

    Defines acceptable latency bounds for operations.
    """

    # Per-epoch budgets (milliseconds)
    epoch_forward_ms: float = 100.0
    epoch_backward_ms: float = 150.0
    epoch_total_ms: float = 300.0

    # Decision budgets
    decision_latency_ms: float = 10.0
    feature_extraction_ms: float = 5.0

    # Memory budgets (MB)
    seed_memory_mb: float = 50.0
    telemetry_buffer_mb: float = 10.0


DEFAULT_BUDGETS = PerformanceBudgets()
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType; e = TelemetryEvent(event_type=TelemetryEventType.EPOCH_COMPLETED, source='simic'); print(e.to_dict())"
```

Expected: Dictionary with event_type, source, timestamp, etc.

**Step 3: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "feat(leyline): add telemetry.py with event contracts"
```

---

### Task 2.7: Create leyline/__init__.py with re-exports

**Files:**
- Modify: `src/esper/leyline/__init__.py`

**Step 1: Add re-exports for public API**

```python
"""Leyline - The invisible substrate of Esper.

Leyline defines the data contracts that flow between all Esper components.
Import from here for the public API.

Example:
    from esper.leyline import SimicAction, SeedStage, TrainingSignals
"""

# Actions
from esper.leyline.actions import SimicAction

# Stages and transitions
from esper.leyline.stages import (
    SeedStage,
    CommandType,
    RiskLevel,
    VALID_TRANSITIONS,
    is_valid_transition,
    is_terminal_stage,
    is_active_stage,
    is_failure_stage,
)

# Signals (hot path)
from esper.leyline.signals import (
    TensorSchema,
    TENSOR_SCHEMA_SIZE,
    FastTrainingSignals,
    TrainingSignals,
)

# Schemas and specifications
from esper.leyline.schemas import (
    SeedOperation,
    AdaptationCommand,
    GateLevel,
    GateResult,
    BlueprintProtocol,
    BlueprintSpec,
)

# Reports
from esper.leyline.reports import (
    TrainingMetrics,
    SeedMetrics,
    SeedStateReport,
    FieldReport,
)

# Telemetry contracts
from esper.leyline.telemetry import (
    TelemetryEventType,
    TelemetryEvent,
    PerformanceBudgets,
    DEFAULT_BUDGETS,
)

__all__ = [
    # Actions
    "SimicAction",
    # Stages
    "SeedStage",
    "CommandType",
    "RiskLevel",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "is_terminal_stage",
    "is_active_stage",
    "is_failure_stage",
    # Signals
    "TensorSchema",
    "TENSOR_SCHEMA_SIZE",
    "FastTrainingSignals",
    "TrainingSignals",
    # Schemas
    "SeedOperation",
    "AdaptationCommand",
    "GateLevel",
    "GateResult",
    "BlueprintProtocol",
    "BlueprintSpec",
    # Reports
    "TrainingMetrics",
    "SeedMetrics",
    "SeedStateReport",
    "FieldReport",
    # Telemetry
    "TelemetryEventType",
    "TelemetryEvent",
    "PerformanceBudgets",
    "DEFAULT_BUDGETS",
]
```

**Step 2: Verify all imports work through package**

```bash
PYTHONPATH=src python -c "from esper.leyline import SimicAction, SeedStage, TrainingSignals, FieldReport, TelemetryEvent; print('All leyline imports OK')"
```

Expected: `All leyline imports OK`

**Step 3: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "feat(leyline): add __init__.py with public API re-exports"
```

---

## Phase 3: Migrate Kasmina (Seed Mechanics)

### Task 3.1: Create kasmina/slot.py

**Files:**
- Create: `src/esper/kasmina/slot.py`
- Source: `src/esper/kasmina.py` lines 52-189 (SeedMetrics, SeedState)
- Source: `src/esper/kasmina.py` lines 190-376 (QualityGates)
- Source: `src/esper/kasmina.py` lines 602-860 (SeedSlot)

**Step 1: Create slot.py**

Copy the SeedMetrics, SeedState, QualityGates, and SeedSlot classes from `src/esper/kasmina.py`, updating imports to use `esper.leyline`:

```python
"""Kasmina Slot - Seed lifecycle management.

The SeedSlot manages a single seed module through its lifecycle:
germination -> training -> blending -> fossilization/culling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from esper.leyline import SeedStage, is_valid_transition, is_terminal_stage

if TYPE_CHECKING:
    from esper.kasmina.blueprints import BlueprintCatalog


@dataclass
class SeedMetrics:
    """Runtime metrics for a seed module."""

    accuracy_at_stage_start: float = 0.0
    current_val_accuracy: float = 0.0
    loss_at_stage_start: float = float("inf")
    current_val_loss: float = float("inf")
    gradient_norm: float = 0.0
    parameter_delta: float = 0.0
    epochs_since_improvement: int = 0
    best_accuracy_in_stage: float = 0.0


# ... (copy remaining classes from kasmina.py, updating imports)
```

**Note:** This file is large (~400 lines). Copy the full implementation from `src/esper/kasmina.py` lines 52-189 (SeedMetrics, SeedState), 190-376 (QualityGates), and 602-860 (SeedSlot), updating all imports to use `esper.leyline` instead of relative imports.

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.kasmina.slot import SeedSlot, SeedState; print('SeedSlot imported')"
```

**Step 3: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "feat(kasmina): add slot.py with SeedSlot and SeedState"
```

---

### Task 3.2: Create kasmina/blueprints.py

**Files:**
- Create: `src/esper/kasmina/blueprints.py`
- Source: `src/esper/kasmina.py` lines 477-600 (ConvBlock, seed modules, BlueprintCatalog)

**Step 1: Create blueprints.py**

Copy the ConvBlock, ConvEnhanceSeed, AttentionSeed, NormSeed, DepthwiseSeed, and BlueprintCatalog from `src/esper/kasmina.py`:

```python
"""Kasmina Blueprints - Seed module implementations.

Blueprints define the architecture of injectable seed modules.
Each blueprint creates a specific type of enhancement:
- ConvEnhance: Additional convolutional capacity
- Attention: Self-attention mechanism
- Norm: Normalization layers
- Depthwise: Depthwise separable convolutions
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


# ... (copy remaining classes from kasmina.py lines 492-600)
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.kasmina.blueprints import BlueprintCatalog, ConvEnhanceSeed; print(BlueprintCatalog.list_blueprints())"
```

**Step 3: Commit**

```bash
git add src/esper/kasmina/blueprints.py
git commit -m "feat(kasmina): add blueprints.py with seed module implementations"
```

---

### Task 3.3: Create kasmina/isolation.py

**Files:**
- Create: `src/esper/kasmina/isolation.py`
- Source: `src/esper/kasmina.py` lines 378-475 (AlphaSchedule, blend_with_isolation, GradientIsolationMonitor)

**Step 1: Create isolation.py**

```python
"""Kasmina Isolation - Gradient isolation and blending.

Ensures seed modules don't destabilize the host network during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class AlphaSchedule:
    """Schedule for blending alpha during seed integration."""

    start_alpha: float = 0.0
    end_alpha: float = 1.0
    total_steps: int = 10
    current_step: int = 0
    schedule_type: str = "linear"  # "linear", "cosine", "exponential"

    @property
    def current_alpha(self) -> float:
        """Get current blending alpha."""
        if self.total_steps <= 0:
            return self.end_alpha

        progress = min(self.current_step / self.total_steps, 1.0)

        if self.schedule_type == "cosine":
            # Smooth cosine annealing
            alpha = self.start_alpha + (self.end_alpha - self.start_alpha) * (
                1 - (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
            ).item()
        elif self.schedule_type == "exponential":
            # Exponential ramp
            alpha = self.start_alpha + (self.end_alpha - self.start_alpha) * (
                1 - (1 - progress) ** 2
            )
        else:
            # Linear
            alpha = self.start_alpha + (self.end_alpha - self.start_alpha) * progress

        return alpha

    def step(self) -> float:
        """Advance schedule and return new alpha."""
        self.current_step += 1
        return self.current_alpha


def blend_with_isolation(
    host_output: torch.Tensor,
    seed_output: torch.Tensor,
    alpha: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Blend host and seed outputs with gradient isolation."""
    # Detach seed gradient flow when alpha is low (exploration phase)
    if alpha < 0.5:
        seed_contribution = seed_output.detach() * alpha
    else:
        seed_contribution = seed_output * alpha

    # Temperature scaling for sharper/softer blending
    if temperature != 1.0:
        seed_contribution = seed_contribution * temperature

    return host_output * (1 - alpha) + seed_contribution


# ... (copy GradientIsolationMonitor from kasmina.py lines 421-475)
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.kasmina.isolation import AlphaSchedule, blend_with_isolation; s = AlphaSchedule(); print(f'Alpha: {s.current_alpha}')"
```

**Step 3: Commit**

```bash
git add src/esper/kasmina/isolation.py
git commit -m "feat(kasmina): add isolation.py with gradient isolation"
```

---

### Task 3.4: Create kasmina/host.py

**Files:**
- Create: `src/esper/kasmina/host.py`
- Source: `src/esper/poc_tamiyo.py` lines 78-177 (ConvBlock, HostCNN, MorphogeneticModel)

**Step 1: Create host.py**

Copy the host network classes from `poc_tamiyo.py`, updating imports:

```python
"""Kasmina Host - The graftable host network.

The MorphogeneticModel is the host network that accepts seed grafts.
It manages the injection points where seeds can be attached.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.leyline import SeedStage
from esper.kasmina.slot import SeedSlot, SeedState
from esper.kasmina.blueprints import BlueprintCatalog


class ConvBlock(nn.Module):
    """Basic conv-bn-relu block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


# ... (copy HostCNN and MorphogeneticModel from poc_tamiyo.py)
```

**Step 2: Verify import works**

```bash
PYTHONPATH=src python -c "from esper.kasmina.host import MorphogeneticModel; print('MorphogeneticModel imported')"
```

**Step 3: Commit**

```bash
git add src/esper/kasmina/host.py
git commit -m "feat(kasmina): add host.py with MorphogeneticModel"
```

---

### Task 3.5: Create kasmina/__init__.py

**Files:**
- Modify: `src/esper/kasmina/__init__.py`

**Step 1: Add re-exports**

```python
"""Kasmina - Seed mechanics for Esper.

Kasmina manages the lifecycle of seed modules:
- Germination: Creating new seeds from blueprints
- Training: Growing seeds with gradient flow
- Blending: Integrating seeds with the host
- Fossilization: Permanent integration
"""

from esper.kasmina.slot import (
    SeedMetrics,
    SeedState,
    QualityGates,
    SeedSlot,
)
from esper.kasmina.blueprints import (
    ConvBlock,
    ConvEnhanceSeed,
    AttentionSeed,
    NormSeed,
    DepthwiseSeed,
    BlueprintCatalog,
)
from esper.kasmina.isolation import (
    AlphaSchedule,
    blend_with_isolation,
    GradientIsolationMonitor,
)
from esper.kasmina.host import (
    HostCNN,
    MorphogeneticModel,
)

__all__ = [
    # Slot management
    "SeedMetrics",
    "SeedState",
    "QualityGates",
    "SeedSlot",
    # Blueprints
    "ConvBlock",
    "ConvEnhanceSeed",
    "AttentionSeed",
    "NormSeed",
    "DepthwiseSeed",
    "BlueprintCatalog",
    # Isolation
    "AlphaSchedule",
    "blend_with_isolation",
    "GradientIsolationMonitor",
    # Host
    "HostCNN",
    "MorphogeneticModel",
]
```

**Step 2: Verify all imports**

```bash
PYTHONPATH=src python -c "from esper.kasmina import SeedSlot, BlueprintCatalog, MorphogeneticModel; print('All kasmina imports OK')"
```

**Step 3: Commit**

```bash
git add src/esper/kasmina/__init__.py
git commit -m "feat(kasmina): add __init__.py with public API"
```

---

## Phase 4: Migrate Tamiyo (Strategic Decisions)

### Task 4.1: Create tamiyo/decisions.py

**Files:**
- Create: `src/esper/tamiyo/decisions.py`
- Source: `src/esper/tamiyo.py` lines 181-262 (TamiyoAction, TamiyoDecision)

**Step 1: Create decisions.py**

```python
"""Tamiyo Decisions - Strategic decision structures.

Defines the decisions made by strategic controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TamiyoAction(Enum):
    """High-level strategic actions (maps to SimicAction for RL)."""

    WAIT = "wait"
    GERMINATE = "germinate"
    ADVANCE = "advance"
    CULL = "cull"


@dataclass
class TamiyoDecision:
    """A strategic decision from Tamiyo.

    Captures not just what action to take, but why.
    """

    action: TamiyoAction
    confidence: float = 1.0
    rationale: str = ""
    timestamp: datetime = field(default_factory=_utc_now)

    # Optional parameters
    blueprint_id: str | None = None
    target_seed_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    # Observation context (for learning)
    observation_hash: str | None = None
```

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.tamiyo.decisions import TamiyoAction, TamiyoDecision; print(TamiyoAction.GERMINATE)"
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/decisions.py
git commit -m "feat(tamiyo): add decisions.py with TamiyoDecision"
```

---

### Task 4.2: Create tamiyo/tracker.py

**Files:**
- Create: `src/esper/tamiyo/tracker.py`
- Source: `src/esper/tamiyo.py` lines 94-179 (SignalTracker)

**Step 1: Create tracker.py**

```python
"""Tamiyo Tracker - Training signal observation.

SignalTracker maintains running statistics for decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from esper.leyline import TrainingSignals


@dataclass
class SignalTracker:
    """Tracks training signals over time for trend analysis."""

    # History windows
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    window_size: int = 5

    # Best values seen
    best_val_accuracy: float = 0.0
    best_val_loss: float = float("inf")

    # Plateau detection
    plateau_epochs: int = 0
    plateau_threshold: float = 0.5  # Accuracy improvement threshold

    # Previous values for delta computation
    _prev_val_accuracy: float = 0.0
    _prev_val_loss: float = float("inf")

    def update(self, val_accuracy: float, val_loss: float) -> None:
        """Update tracker with new epoch results."""
        # Update histories
        self.loss_history.append(val_loss)
        self.accuracy_history.append(val_accuracy)

        # Trim to window size
        if len(self.loss_history) > self.window_size:
            self.loss_history = self.loss_history[-self.window_size:]
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history = self.accuracy_history[-self.window_size:]

        # Update best values
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.plateau_epochs = 0
        else:
            self.plateau_epochs += 1

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Store for delta computation
        self._prev_val_accuracy = val_accuracy
        self._prev_val_loss = val_loss

    def get_accuracy_delta(self, current: float) -> float:
        """Get accuracy change from previous epoch."""
        return current - self._prev_val_accuracy

    def get_loss_delta(self, current: float) -> float:
        """Get loss change from previous epoch."""
        return current - self._prev_val_loss

    def reset(self) -> None:
        """Reset tracker for new episode."""
        self.loss_history = []
        self.accuracy_history = []
        self.best_val_accuracy = 0.0
        self.best_val_loss = float("inf")
        self.plateau_epochs = 0
        self._prev_val_accuracy = 0.0
        self._prev_val_loss = float("inf")
```

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.tamiyo.tracker import SignalTracker; t = SignalTracker(); t.update(75.0, 0.5); print(f'Best acc: {t.best_val_accuracy}')"
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/tracker.py
git commit -m "feat(tamiyo): add tracker.py with SignalTracker"
```

---

### Task 4.3: Create tamiyo/heuristic.py

**Files:**
- Create: `src/esper/tamiyo/heuristic.py`
- Source: `src/esper/tamiyo.py` lines 263-493 (TamiyoPolicy, HeuristicPolicyConfig, HeuristicTamiyo)

**Step 1: Create heuristic.py**

Copy the HeuristicTamiyo implementation, updating imports to use new package structure.

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig; print('HeuristicTamiyo imported')"
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/heuristic.py
git commit -m "feat(tamiyo): add heuristic.py with HeuristicTamiyo"
```

---

### Task 4.4: Create tamiyo/__init__.py

**Files:**
- Modify: `src/esper/tamiyo/__init__.py`

**Step 1: Add re-exports**

```python
"""Tamiyo - Strategic decision-making for Esper.

Tamiyo observes training signals and makes strategic decisions
about seed lifecycle management.
"""

from esper.tamiyo.decisions import TamiyoAction, TamiyoDecision
from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.heuristic import (
    TamiyoPolicy,
    HeuristicPolicyConfig,
    HeuristicTamiyo,
)

__all__ = [
    "TamiyoAction",
    "TamiyoDecision",
    "SignalTracker",
    "TamiyoPolicy",
    "HeuristicPolicyConfig",
    "HeuristicTamiyo",
]
```

**Step 2: Verify all imports**

```bash
PYTHONPATH=src python -c "from esper.tamiyo import HeuristicTamiyo, SignalTracker, TamiyoDecision; print('All tamiyo imports OK')"
```

**Step 3: Commit**

```bash
git add src/esper/tamiyo/__init__.py
git commit -m "feat(tamiyo): add __init__.py with public API"
```

---

## Phase 5: Migrate Simic (RL Training)

### Task 5.1: Create simic/features.py (HOT PATH - leyline imports only!)

**Files:**
- Create: `src/esper/simic/features.py`
- Source: `src/esper/simic_train.py` lines 45-130 (safe, obs_to_base_features, telemetry_to_features)

**Step 1: Create features.py with ONLY leyline imports**

```python
"""Simic Features - Observation feature extraction.

!!! HOT PATH - LEYLINE IMPORTS ONLY !!!

This module is used in the vectorized PPO loop.
It MUST NOT import from kasmina, tamiyo, or nissa.
"""

from __future__ import annotations

import math

from esper.leyline import TensorSchema, TENSOR_SCHEMA_SIZE


def safe(v, default: float = 0.0, max_val: float = 100.0) -> float:
    """Safely convert value to float, handling None/inf/nan."""
    if v is None:
        return default
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return default
    return max(-max_val, min(float(v), max_val))


def obs_to_base_features(obs: dict) -> list[float]:
    """Extract V1-style base features (27 dims) from observation dict.

    Returns features in TensorSchema order for direct tensor creation.
    """
    return [
        float(obs["epoch"]),
        float(obs["global_step"]),
        safe(obs["train_loss"], 10.0),
        safe(obs["val_loss"], 10.0),
        safe(obs["loss_delta"], 0.0),
        obs["train_accuracy"],
        obs["val_accuracy"],
        safe(obs["accuracy_delta"], 0.0),
        obs["best_val_accuracy"],
        float(obs["plateau_epochs"]),
        safe(obs["best_val_loss"], 10.0),
        *[safe(v, 10.0) for v in obs["loss_history_5"]],
        *obs["accuracy_history_5"],
        float(obs["has_active_seed"]),
        float(obs["seed_stage"]),
        float(obs["seed_epochs_in_stage"]),
        obs["seed_alpha"],
        obs["seed_improvement"],
        float(obs["available_slots"]),
    ]


def telemetry_to_features(telem: dict) -> list[float]:
    """Extract V2 telemetry features (27 dims) from telemetry snapshot."""
    features = []

    # Gradient health (6)
    grad = telem.get("gradient_health", {})
    features.append(safe(grad.get("host_grad_norm", 0.0)))
    features.append(safe(grad.get("seed_grad_norm", 0.0)))
    features.append(safe(grad.get("grad_similarity", 0.0), 0.0, 1.0))
    features.append(float(grad.get("is_healthy", True)))
    features.append(safe(grad.get("max_grad", 0.0)))
    features.append(safe(grad.get("min_grad", 0.0)))

    # Loss components (6)
    loss = telem.get("loss_components", {})
    features.append(safe(loss.get("ce_loss", 0.0), 10.0))
    features.append(safe(loss.get("reg_loss", 0.0), 1.0))
    features.append(safe(loss.get("seed_loss", 0.0), 10.0))
    features.append(safe(loss.get("host_loss", 0.0), 10.0))
    features.append(safe(loss.get("blend_loss", 0.0), 10.0))
    features.append(safe(loss.get("total_loss", 0.0), 10.0))

    # Learning dynamics (6)
    dyn = telem.get("learning_dynamics", {})
    features.append(safe(dyn.get("lr_current", 0.0), 1.0))
    features.append(safe(dyn.get("momentum", 0.0), 1.0))
    features.append(safe(dyn.get("weight_decay", 0.0), 0.1))
    features.append(float(dyn.get("scheduler_step", 0)))
    features.append(safe(dyn.get("effective_batch_size", 128), 1024))
    features.append(safe(dyn.get("samples_seen", 0), 1e6))

    # Activation stats (6)
    act = telem.get("activation_stats", {})
    features.append(safe(act.get("mean_activation", 0.0), 10.0))
    features.append(safe(act.get("std_activation", 0.0), 10.0))
    features.append(safe(act.get("sparsity", 0.0), 1.0))
    features.append(safe(act.get("dead_neurons_pct", 0.0), 1.0))
    features.append(safe(act.get("saturation_pct", 0.0), 1.0))
    features.append(safe(act.get("layer_imbalance", 0.0), 10.0))

    # Red flags (3)
    rf = telem.get("red_flags", [])
    features.append(1.0 if "severe_class_imbalance" in rf else 0.0)
    features.append(1.0 if "sharp_minimum" in rf else 0.0)
    features.append(1.0 if "gradient_issues" in rf else 0.0)

    return features
```

**Step 2: Verify ONLY leyline imports (critical!)**

```bash
PYTHONPATH=src python -c "
import ast
import sys

with open('src/esper/simic/features.py') as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        if node.module and 'esper.' in node.module:
            if not node.module.startswith('esper.leyline'):
                print(f'ERROR: Forbidden import: {node.module}')
                sys.exit(1)

print('HOT PATH CHECK PASSED: Only leyline imports found')
"
```

Expected: `HOT PATH CHECK PASSED: Only leyline imports found`

**Step 3: Commit**

```bash
git add src/esper/simic/features.py
git commit -m "feat(simic): add features.py with hot-path feature extraction"
```

---

### Task 5.2: Create simic/rewards.py

**Files:**
- Move: `src/esper/rewards.py` -> `src/esper/simic/rewards.py`

**Step 1: Move and update imports**

```bash
mv src/esper/rewards.py src/esper/simic/rewards.py
```

Update imports in the file to use `esper.leyline` for SeedStage.

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.simic.rewards import compute_shaped_reward, RewardConfig, SeedInfo; print('Rewards imported')"
```

**Step 3: Commit**

```bash
git add src/esper/simic/rewards.py
git rm src/esper/rewards.py 2>/dev/null || true
git commit -m "feat(simic): move rewards.py to simic package"
```

---

### Task 5.3: Create simic/episodes.py

**Files:**
- Create: `src/esper/simic/episodes.py`
- Source: `src/esper/simic.py` lines 30-228 (TrainingSnapshot)
- Source: `src/esper/simic.py` lines 261-490 (ActionTaken, StepOutcome, DecisionPoint, Episode)
- Source: `src/esper/simic.py` lines 491-672 (EpisodeCollector, snapshot_from_signals, action_from_decision)

**Step 1: Create episodes.py**

Copy the episode-related classes from `simic.py`, updating imports.

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.simic.episodes import Episode, EpisodeCollector, TrainingSnapshot; print('Episodes imported')"
```

**Step 3: Commit**

```bash
git add src/esper/simic/episodes.py
git commit -m "feat(simic): add episodes.py with Episode and EpisodeCollector"
```

---

### Task 5.4: Create simic/networks.py

**Files:**
- Create: `src/esper/simic/networks.py`
- Source: `src/esper/simic.py` lines 736-972 (PolicyNetwork)

**Step 1: Create networks.py**

Copy the PolicyNetwork class from `simic.py`.

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.simic.networks import PolicyNetwork; print('PolicyNetwork imported')"
```

**Step 3: Commit**

```bash
git add src/esper/simic/networks.py
git commit -m "feat(simic): add networks.py with PolicyNetwork"
```

---

### Task 5.5: Create simic/ppo.py

**Files:**
- Create: `src/esper/simic/ppo.py`
- Source: `src/esper/simic_ppo.py` (entire file, ~1590 lines)

**Step 1: Copy and update imports**

Copy `simic_ppo.py` to `simic/ppo.py`, updating all imports:

```python
# Old imports
from esper.simic import SimicAction
from esper.rewards import compute_shaped_reward, SeedInfo
from esper.tamiyo import SignalTracker
from esper.kasmina import SeedStage, SeedSlot, BlueprintCatalog

# New imports
from esper.leyline import SimicAction, SeedStage
from esper.simic.rewards import compute_shaped_reward, SeedInfo
from esper.tamiyo import SignalTracker
from esper.kasmina import SeedSlot, BlueprintCatalog
```

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.simic.ppo import PPOAgent; print('PPOAgent imported')"
```

**Step 3: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "feat(simic): add ppo.py with PPOAgent and training loops"
```

---

### Task 5.6: Create simic/iql.py

**Files:**
- Create: `src/esper/simic/iql.py`
- Source: `src/esper/simic_iql.py` (entire file, ~1325 lines)

**Step 1: Copy and update imports**

Same process as PPO - copy and update imports to use new package structure.

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.simic.iql import IQLAgent; print('IQLAgent imported')" 2>/dev/null || echo "IQL has additional dependencies - verify manually"
```

**Step 3: Commit**

```bash
git add src/esper/simic/iql.py
git commit -m "feat(simic): add iql.py with IQL implementation"
```

---

### Task 5.7: Create simic/__init__.py

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Add re-exports**

```python
"""Simic - RL training infrastructure for Esper.

The Simic Combine experiments with evolution and adaptation.
This package contains all RL algorithms and training infrastructure.
"""

from esper.simic.features import (
    safe,
    obs_to_base_features,
    telemetry_to_features,
)
from esper.simic.rewards import (
    compute_shaped_reward,
    RewardConfig,
    SeedInfo,
)
from esper.simic.episodes import (
    TrainingSnapshot,
    ActionTaken,
    StepOutcome,
    DecisionPoint,
    Episode,
    EpisodeCollector,
    snapshot_from_signals,
    action_from_decision,
)
from esper.simic.networks import PolicyNetwork

# Heavy imports - only import if needed
# from esper.simic.ppo import PPOAgent
# from esper.simic.iql import IQLAgent

__all__ = [
    # Features (hot path)
    "safe",
    "obs_to_base_features",
    "telemetry_to_features",
    # Rewards
    "compute_shaped_reward",
    "RewardConfig",
    "SeedInfo",
    # Episodes
    "TrainingSnapshot",
    "ActionTaken",
    "StepOutcome",
    "DecisionPoint",
    "Episode",
    "EpisodeCollector",
    "snapshot_from_signals",
    "action_from_decision",
    # Networks
    "PolicyNetwork",
]
```

**Step 2: Verify imports**

```bash
PYTHONPATH=src python -c "from esper.simic import Episode, compute_shaped_reward, obs_to_base_features; print('All simic imports OK')"
```

**Step 3: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "feat(simic): add __init__.py with public API"
```

---

## Phase 6: Create Nissa (System Telemetry)

### Task 6.1: Create nissa/config.py

**Files:**
- Create: `src/esper/nissa/config.py`
- Source: `src/esper/telemetry_config.py` (entire file)

**Step 1: Move and rename**

```bash
mv src/esper/telemetry_config.py src/esper/nissa/config.py
mv src/esper/telemetry_profiles.yaml src/esper/nissa/profiles.yaml
```

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.nissa.config import TelemetryConfig; print('TelemetryConfig imported')"
```

**Step 3: Commit**

```bash
git add src/esper/nissa/config.py src/esper/nissa/profiles.yaml
git rm src/esper/telemetry_config.py src/esper/telemetry_profiles.yaml 2>/dev/null || true
git commit -m "feat(nissa): add config.py with TelemetryConfig"
```

---

### Task 6.2: Create nissa/tracker.py

**Files:**
- Create: `src/esper/nissa/tracker.py`
- Source: `src/esper/telemetry.py` (entire file)

**Step 1: Move and update imports**

```bash
mv src/esper/telemetry.py src/esper/nissa/tracker.py
```

Update imports to use `esper.leyline` for TelemetryEvent types.

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.nissa.tracker import DiagnosticTracker; print('DiagnosticTracker imported')"
```

**Step 3: Commit**

```bash
git add src/esper/nissa/tracker.py
git rm src/esper/telemetry.py 2>/dev/null || true
git commit -m "feat(nissa): add tracker.py with DiagnosticTracker"
```

---

### Task 6.3: Create nissa/output.py

**Files:**
- Create: `src/esper/nissa/output.py`

**Step 1: Create simple output backend**

```python
"""Nissa Output - Telemetry output backends.

Simple console and file output for telemetry events.
Designed to be extended with more sophisticated backends later.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TextIO

from esper.leyline import TelemetryEvent


class OutputBackend(ABC):
    """Base class for telemetry output backends."""

    @abstractmethod
    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event."""
        ...

    def flush(self) -> None:
        """Flush any buffered output."""
        pass

    def close(self) -> None:
        """Close the backend."""
        pass


class ConsoleOutput(OutputBackend):
    """Output telemetry events to console."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def emit(self, event: TelemetryEvent) -> None:
        if self.verbose:
            print(f"[{event.source}] {event.event_type.value}: {event.payload}")
        else:
            # Compact format
            ts = event.timestamp.strftime("%H:%M:%S")
            print(f"[{ts}] {event.event_type.value}")


class FileOutput(OutputBackend):
    """Output telemetry events to JSON lines file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None

    def _ensure_open(self) -> TextIO:
        if self._file is None:
            self._file = open(self.path, "a")
        return self._file

    def emit(self, event: TelemetryEvent) -> None:
        f = self._ensure_open()
        f.write(json.dumps(event.to_dict()) + "\n")

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


class NissaHub:
    """Central telemetry hub that routes events to backends.

    Nissa receives carbon copies of events from all domains
    and routes them to configured output backends.
    """

    def __init__(self):
        self.backends: list[OutputBackend] = []
        self._event_count = 0

    def add_backend(self, backend: OutputBackend) -> None:
        """Add an output backend."""
        self.backends.append(backend)

    def emit(self, event: TelemetryEvent) -> None:
        """Emit event to all backends."""
        self._event_count += 1
        for backend in self.backends:
            backend.emit(event)

    def flush(self) -> None:
        """Flush all backends."""
        for backend in self.backends:
            backend.flush()

    def close(self) -> None:
        """Close all backends."""
        for backend in self.backends:
            backend.close()

    @property
    def event_count(self) -> int:
        return self._event_count


# Global hub instance (optional convenience)
_hub: NissaHub | None = None


def get_hub() -> NissaHub:
    """Get or create the global Nissa hub."""
    global _hub
    if _hub is None:
        _hub = NissaHub()
    return _hub


def emit(event: TelemetryEvent) -> None:
    """Emit event to global hub."""
    get_hub().emit(event)
```

**Step 2: Verify import**

```bash
PYTHONPATH=src python -c "from esper.nissa.output import NissaHub, ConsoleOutput; hub = NissaHub(); hub.add_backend(ConsoleOutput()); print('NissaHub ready')"
```

**Step 3: Commit**

```bash
git add src/esper/nissa/output.py
git commit -m "feat(nissa): add output.py with console and file backends"
```

---

### Task 6.4: Create nissa/__init__.py

**Files:**
- Modify: `src/esper/nissa/__init__.py`

**Step 1: Add re-exports**

```python
"""Nissa - System telemetry hub for Esper.

Nissa senses the health and flow of the training system.
She receives carbon copies of events from all domains and
routes them to configured output backends.
"""

from esper.nissa.config import TelemetryConfig
from esper.nissa.tracker import (
    DiagnosticTracker,
    GradientStats,
    GradientHealth,
    EpochSnapshot,
)
from esper.nissa.output import (
    OutputBackend,
    ConsoleOutput,
    FileOutput,
    NissaHub,
    get_hub,
    emit,
)

__all__ = [
    # Config
    "TelemetryConfig",
    # Tracker
    "DiagnosticTracker",
    "GradientStats",
    "GradientHealth",
    "EpochSnapshot",
    # Output
    "OutputBackend",
    "ConsoleOutput",
    "FileOutput",
    "NissaHub",
    "get_hub",
    "emit",
]
```

**Step 2: Verify imports**

```bash
PYTHONPATH=src python -c "from esper.nissa import NissaHub, DiagnosticTracker, TelemetryConfig; print('All nissa imports OK')"
```

**Step 3: Commit**

```bash
git add src/esper/nissa/__init__.py
git commit -m "feat(nissa): add __init__.py with public API"
```

---

## Phase 7: Create Entry Points

### Task 7.1: Create scripts/train.py

**Files:**
- Create: `src/esper/scripts/train.py`

**Step 1: Create PPO training entry point**

```python
"""Esper Training Script - PPO entry point.

Usage:
    python -m esper.scripts.train --episodes 100 --device cuda:0
    python -m esper.scripts.train --episodes 100 --vectorized --n-envs 6
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Train Esper PPO agent")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vectorized", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.1)

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from esper.simic.ppo import train_vectorized, train_sequential

    if args.vectorized:
        train_vectorized(
            n_episodes=args.episodes,
            max_epochs=args.max_epochs,
            n_envs=args.n_envs,
            device=args.device,
            save_path=args.save,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
        )
    else:
        train_sequential(
            n_episodes=args.episodes,
            max_epochs=args.max_epochs,
            device=args.device,
            save_path=args.save,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
        )


if __name__ == "__main__":
    main()
```

**Step 2: Verify script runs**

```bash
PYTHONPATH=src python -m esper.scripts.train --help
```

**Step 3: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "feat(scripts): add train.py entry point"
```

---

### Task 7.2: Create scripts/generate.py

**Files:**
- Create: `src/esper/scripts/generate.py`

**Step 1: Create data generation entry point**

```python
"""Esper Generate Script - Heuristic data generation.

Usage:
    python -m esper.scripts.generate --episodes 1000 --output data/episodes/
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate training data with heuristic")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    from esper.tamiyo import HeuristicTamiyo
    from esper.simic.episodes import EpisodeCollector

    print(f"Generating {args.episodes} episodes to {args.output}")
    # TODO: Implement generation loop


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/esper/scripts/generate.py
git commit -m "feat(scripts): add generate.py entry point (stub)"
```

---

### Task 7.3: Create scripts/evaluate.py

**Files:**
- Create: `src/esper/scripts/evaluate.py`

**Step 1: Create evaluation entry point**

```python
"""Esper Evaluate Script - Head-to-head comparison.

Usage:
    python -m esper.scripts.evaluate --policy models/ppo.pt --episodes 10
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy vs heuristic")
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    print(f"Evaluating {args.policy} for {args.episodes} episodes")
    # TODO: Implement evaluation loop


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/esper/scripts/evaluate.py
git commit -m "feat(scripts): add evaluate.py entry point (stub)"
```

---

## Phase 8: Update Root Imports and Cleanup

### Task 8.1: Update src/esper/__init__.py

**Files:**
- Modify: `src/esper/__init__.py`

**Step 1: Add package-level imports**

```python
"""Esper - Adaptive Neural Architecture System.

Esper implements dynamic model adaptation through the seed lifecycle:
germination -> training -> blending -> fossilization.

Subpackages:
- leyline: Data contracts and schemas
- kasmina: Seed mechanics and host models
- tamiyo: Strategic decision-making
- simic: RL training infrastructure
- nissa: System telemetry
"""

__version__ = "1.0.0"

# Re-export key types for convenience
from esper.leyline import SimicAction, SeedStage, TrainingSignals
from esper.kasmina import MorphogeneticModel, SeedSlot
from esper.tamiyo import HeuristicTamiyo
```

**Step 2: Commit**

```bash
git add src/esper/__init__.py
git commit -m "feat(esper): update root __init__.py with v1.0 exports"
```

---

### Task 8.2: Delete old root-level files

**Files:**
- Delete: `src/esper/leyline.py`
- Delete: `src/esper/kasmina.py`
- Delete: `src/esper/tamiyo.py`
- Delete: `src/esper/simic.py`
- Delete: `src/esper/simic_ppo.py`
- Delete: `src/esper/simic_iql.py`
- Delete: `src/esper/simic_train.py`
- Delete: `src/esper/simic_overnight.py`
- Delete: `src/esper/poc_tamiyo.py`

**Step 1: Remove old files**

```bash
git rm src/esper/leyline.py
git rm src/esper/kasmina.py
git rm src/esper/tamiyo.py
git rm src/esper/simic.py
git rm src/esper/simic_ppo.py
git rm src/esper/simic_iql.py
git rm src/esper/simic_train.py
git rm src/esper/simic_overnight.py
git rm src/esper/poc_tamiyo.py
```

**Step 2: Commit**

```bash
git commit -m "chore: remove old root-level module files"
```

---

### Task 8.3: Update external scripts

**Files:**
- Modify: `scripts/train_ppo.sh`

**Step 1: Update script to use new module path**

Change:
```bash
python -m esper.simic_ppo
```

To:
```bash
python -m esper.scripts.train
```

**Step 2: Commit**

```bash
git add scripts/train_ppo.sh
git commit -m "chore: update train_ppo.sh to use new module path"
```

---

### Task 8.4: Final verification

**Step 1: Run import tests**

```bash
PYTHONPATH=src python -c "
from esper.leyline import SimicAction, SeedStage, TrainingSignals, FieldReport
from esper.kasmina import MorphogeneticModel, SeedSlot, BlueprintCatalog
from esper.tamiyo import HeuristicTamiyo, SignalTracker
from esper.simic import Episode, compute_shaped_reward, obs_to_base_features
from esper.simic.ppo import PPOAgent
from esper.nissa import NissaHub, DiagnosticTracker

print('=== All imports successful! ===')
print(f'SimicAction: {SimicAction.GERMINATE_CONV}')
print(f'SeedStage: {SeedStage.TRAINING}')
print('Esper V1.0 migration complete!')
"
```

**Step 2: Run hot path check**

```bash
PYTHONPATH=src python -c "
import ast

with open('src/esper/simic/features.py') as f:
    tree = ast.parse(f.read())

forbidden = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        if node.module and 'esper.' in node.module:
            if not node.module.startswith('esper.leyline'):
                forbidden.append(node.module)

if forbidden:
    print(f'HOT PATH VIOLATION: {forbidden}')
    exit(1)
else:
    print('HOT PATH CHECK PASSED')
"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Esper V1.0 architecture migration

- Leyline: Data contracts and schemas
- Kasmina: Seed mechanics and host models
- Tamiyo: Strategic decision-making
- Simic: RL training infrastructure
- Nissa: System telemetry hub

Hot path constraint enforced: simic/features.py imports only from leyline."
```

---

## Summary

This plan migrates Esper from a flat module structure to a domain-organized architecture:

| Phase | Tasks | Purpose |
|-------|-------|---------|
| 1 | 1 | Create package directories |
| 2 | 7 | Migrate Leyline (contracts) |
| 3 | 5 | Migrate Kasmina (seed mechanics) |
| 4 | 4 | Migrate Tamiyo (decisions) |
| 5 | 7 | Migrate Simic (RL training) |
| 6 | 4 | Create Nissa (telemetry) |
| 7 | 3 | Create entry points |
| 8 | 4 | Cleanup and verification |

**Total: 35 tasks**

Key architectural constraints:
- Hot path (`simic/features.py`) imports only from `leyline`
- Each domain owns its telemetry emissions
- Nissa receives carbon copies and handles output
- MTG-themed names preserved at package level
