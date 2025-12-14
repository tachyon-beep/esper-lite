"""Leyline - The invisible substrate of Esper.

Leyline defines the data contracts that flow between all Esper components.
Import from here for the public API.

Example:
    from esper.leyline import SeedStage, TrainingSignals
    from esper.leyline.factored_actions import FactoredAction, LifecycleOp
"""

# Version
LEYLINE_VERSION = "0.2.0"

# =============================================================================
# Lifecycle Constants (shared across simic modules)
# =============================================================================

# Minimum seed age before CULL is allowed (need at least one counterfactual measurement)
# Reduced from 10 to 1: let agent LEARN optimal timing via rewards, not hard masks
MIN_CULL_AGE = 1

# Epochs needed for confident seed quality assessment
FULL_EVALUATION_AGE = 10

# Minimum epochs in PROBATIONARY to earn full fossilize bonus
MIN_PROBATION_EPOCHS = 5

# Seed limits (None = unlimited)
DEFAULT_MAX_SEEDS = None           # Global limit across all slots

# =============================================================================
# PPO/PBRS Constants (shared between ppo.py, rewards.py, buffer, vectorized)
# =============================================================================

# Discount factor for PPO and PBRS reward shaping.
# CRITICAL: PPO gamma MUST equal PBRS gamma for policy invariance (Ng et al., 1999).
# If they differ, reward shaping can change the optimal policy.
# Value 0.995 optimized for 25-epoch episodes: gamma^25 ≈ 0.88 preserves meaningful
# credit for early actions while still providing appropriate temporal discounting.
DEFAULT_GAMMA = 0.995

# =============================================================================
# Episode & Architecture Constants (MUST sync across simic modules)
# =============================================================================

# Episode length in epochs - determines LSTM sequence length and max_steps_per_env.
# CRITICAL: Must sync with chunk_length (BPTT window) for proper credit assignment.
# Used by: config.py, vectorized.py, ppo.py (chunk_length, max_steps_per_env)
DEFAULT_EPISODE_LENGTH = 25

# LSTM hidden dimension - architecture constant for temporal memory.
# Must match across network construction and buffer state tracking.
# Used by: config.py, vectorized.py, ppo.py, tamiyo_buffer.py, tamiyo_network.py
DEFAULT_LSTM_HIDDEN_DIM = 128

# Parallel environments for vectorized training.
# Affects batch size (n_envs * episode_length) and GPU utilization.
# Used by: config.py, vectorized.py, ppo.py, train.py CLI
DEFAULT_N_ENVS = 4

# =============================================================================
# PPO Hyperparameters (tuning knobs for policy gradient training)
# =============================================================================

# Learning rate for PPO optimizer (Adam).
# 3e-4 is the "safe default" from Schulman et al. PPO paper.
DEFAULT_LEARNING_RATE = 3e-4

# PPO clip ratio - limits policy update magnitude per step.
# 0.2 is standard; lower = more conservative updates.
DEFAULT_CLIP_RATIO = 0.2

# GAE lambda for advantage estimation bias-variance tradeoff.
# 0.97 = less bias (good for long delays like 25-epoch episodes).
# Standard value is 0.95; higher reduces bias at cost of variance.
DEFAULT_GAE_LAMBDA = 0.97

# Value function loss coefficient in combined PPO loss.
# 0.5 is standard; higher = prioritize value accuracy over policy.
DEFAULT_VALUE_COEF = 0.5

# Maximum gradient norm for clipping (prevents exploding gradients).
# 0.5 is standard; lower = more aggressive clipping.
DEFAULT_MAX_GRAD_NORM = 0.5

# Number of PPO epochs per batch of experience.
# More epochs = more sample efficiency, but risks overfitting.
DEFAULT_N_PPO_EPOCHS = 10

# Mini-batch size for PPO updates.
# 64 is standard; larger = more stable but less frequent updates.
DEFAULT_BATCH_SIZE = 64

# Entropy coefficient for exploration bonus in policy loss.
# Higher = more exploration; 0.05 prevents premature convergence.
DEFAULT_ENTROPY_COEF = 0.05

# Minimum entropy coefficient floor (prevents exploration collapse).
DEFAULT_ENTROPY_COEF_MIN = 0.01

# =============================================================================
# Reward Shaping Constants (tunable reward function weights)
# =============================================================================

# Primary signal: seed contribution weight in reward calculation.
# Affects how much counterfactual seed contribution impacts rewards.
DEFAULT_CONTRIBUTION_WEIGHT = 1.0

# PBRS (Potential-Based Reward Shaping) weight for stage progression bonuses.
# Balances exploration incentives with primary contribution signal.
DEFAULT_PBRS_WEIGHT = 0.3

# Compute rent weight (penalizes parameter overhead from seeds).
# Logarithmic scaling prevents excessive penalty for large seeds.
DEFAULT_RENT_WEIGHT = 0.5

# Terminal bonus multiplier for fossilized seeds.
# Higher values incentivize completion over seed farming.
DEFAULT_FOSSILIZE_TERMINAL_SCALE = 3.0

# =============================================================================
# Lifecycle Gate Thresholds (seed state machine gates)
# =============================================================================

# G5 Gate: Minimum causal contribution (%) required for fossilization.
# Seeds with contribution < this threshold cannot fossilize successfully.
DEFAULT_MIN_FOSSILIZE_CONTRIBUTION = 1.0

# G2 Gate: Seed gradient ratio threshold for activity detection.
# Seeds with ratio below threshold may be considered inactive.
DEFAULT_GRADIENT_RATIO_THRESHOLD = 0.05

# G3 Gate: Minimum stability required for fossilization.
# Higher = stricter stability requirements before fossilization allowed.
DEFAULT_MIN_PROBATION_STABILITY = 0.95

# EMA decay for gradient ratio smoothing (reduces noise in G2 gate).
DEFAULT_GRADIENT_EMA_DECAY = 0.9

# =============================================================================
# Host Stabilization Constants (training convergence detection)
# =============================================================================

# Relative improvement threshold for considering training "stable".
# Lower = stricter = host stabilizes later (more conservative).
DEFAULT_STABILIZATION_THRESHOLD = 0.03  # 3% relative improvement

# Consecutive epochs below threshold required to declare stability.
DEFAULT_STABILIZATION_EPOCHS = 3

# =============================================================================
# Governor (Tolaria) Constants (catastrophic failure detection)
# =============================================================================

# Sensitivity multiplier for loss anomaly detection (std devs above mean).
# Higher = more tolerant of loss spikes; lower = triggers faster.
DEFAULT_GOVERNOR_SENSITIVITY = 6.0

# Absolute loss threshold that triggers panic regardless of statistics.
# Task-dependent: CIFAR=12.0, TinyStories=15.0 (higher loss scales).
DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD = 12.0

# Penalty applied when governor triggers a rollback (negative reward).
DEFAULT_GOVERNOR_DEATH_PENALTY = 10.0

# Rolling window size for loss history statistics.
DEFAULT_GOVERNOR_HISTORY_WINDOW = 20

# Consecutive panics required before triggering a rollback.
# Higher = more conservative (avoids false positives from transients).
DEFAULT_MIN_PANICS_BEFORE_ROLLBACK = 3

# =============================================================================
# Heuristic Policy (Tamiyo) Constants
# =============================================================================

# Number of plateau epochs before triggering germination.
# Plateau = no significant accuracy improvement (< plateau_threshold).
DEFAULT_PLATEAU_EPOCHS_TO_GERMINATE = 3

# Minimum training epochs before first germination allowed.
# Prevents premature seed creation during initial training.
DEFAULT_MIN_EPOCHS_BEFORE_GERMINATE = 5

# Epochs without improvement before culling a seed.
DEFAULT_CULL_AFTER_EPOCHS_WITHOUT_IMPROVEMENT = 5

# Accuracy drop (%) that triggers immediate cull.
DEFAULT_CULL_IF_ACCURACY_DROPS_BY = 2.0

# Cooldown epochs after a cull before next germination allowed (anti-thrashing).
DEFAULT_EMBARGO_EPOCHS_AFTER_CULL = 5

# Default number of steps for alpha ramp during BLENDING stage.
# Controls how gradually seed influence is increased.
DEFAULT_BLENDING_TOTAL_STEPS = 5

# =============================================================================
# Task Training Defaults
# =============================================================================

# Host model learning rate (backbone network training).
# 0.01 is typical for CNN backbones like ResNet; lower for transformers.
DEFAULT_HOST_LR = 0.01

# Seed module learning rate (new capacity parameters).
# Same as host by default; can be lowered for more conservative seed training.
DEFAULT_SEED_LR = 0.01

# Default batch size for training (CIFAR-10 optimized).
# Affects memory usage and gradient noise. Larger = more stable, more memory.
DEFAULT_BATCH_SIZE_TRAINING = 128

# Default dropout rate for regularization.
# Higher = more regularization; 0.1 is standard for transformers.
DEFAULT_DROPOUT = 0.1

# =============================================================================
# PPO Network Architecture
# =============================================================================

# Value function clip range (separate from policy clip_ratio).
# Larger range allows value estimates to update more freely.
DEFAULT_VALUE_CLIP = 10.0

# Feature extraction dimension before LSTM in Tamiyo network.
# Larger = more capacity but slower training.
DEFAULT_FEATURE_DIM = 128

# =============================================================================
# Blueprint Penalty System (Anti-Thrashing)
# =============================================================================

# Penalty added to a blueprint when its seed is culled.
# Higher = more aggressive avoidance of failed blueprints.
DEFAULT_BLUEPRINT_PENALTY_ON_CULL = 2.0

# Multiplicative decay applied to blueprint penalties each epoch.
# Lower = penalties persist longer; 0.5 means ~10 epoch half-life.
DEFAULT_BLUEPRINT_PENALTY_DECAY = 0.5

# Accumulated penalty threshold above which blueprint is skipped.
# Lower = more aggressive exclusion of penalized blueprints.
DEFAULT_BLUEPRINT_PENALTY_THRESHOLD = 3.0

# =============================================================================
# Lifecycle Gate Thresholds (QualityGates)
# =============================================================================

# Minimum training improvement required to pass G1 gate.
# Seeds must show this much improvement before advancing.
DEFAULT_MIN_TRAINING_IMPROVEMENT = 0.5

# Minimum epochs in BLENDING stage before advancement allowed.
# Ensures seed has time to demonstrate stable blending.
DEFAULT_MIN_BLENDING_EPOCHS = 3

# Maximum gradient isolation violations before gate failure.
# Prevents seeds that destabilize host gradients from advancing.
DEFAULT_MAX_ISOLATION_VIOLATIONS = 10

# Alpha threshold for considering blending "complete" (G3 gate).
# Seeds must reach this alpha level to be considered fully blended.
DEFAULT_ALPHA_COMPLETE_THRESHOLD = 0.95

# Maximum epochs in PROBATIONARY stage before auto-cull timeout.
# Prevents indecisive policies from holding seeds indefinitely.
DEFAULT_MAX_PROBATION_EPOCHS = 5

# =============================================================================
# Governor Anomaly Detection
# =============================================================================

# Loss multiplier threshold for governor panic.
# Loss must exceed (average * multiplier) to trigger anomaly detection.
DEFAULT_GOVERNOR_LOSS_MULTIPLIER = 3.0

# Actions (build_action_enum used by HeuristicTamiyo for flat action mapping)
from esper.leyline.actions import build_action_enum

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
    TrainingMetrics,
    TrainingSignals,
)

# Schemas and specifications
from esper.leyline.schemas import (
    SeedOperation,
    OPERATION_TARGET_STAGE,
    AdaptationCommand,
    GateLevel,
    GateResult,
    BlueprintProtocol,
    BlueprintSpec,
)

# Reports
from esper.leyline.reports import (
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
    SeedTelemetry,
)

__all__ = [
    # Version
    "LEYLINE_VERSION",

    # Lifecycle constants
    "MIN_CULL_AGE",
    "FULL_EVALUATION_AGE",
    "MIN_PROBATION_EPOCHS",
    "DEFAULT_MAX_SEEDS",
    "DEFAULT_GAMMA",

    # Episode & Architecture constants
    "DEFAULT_EPISODE_LENGTH",
    "DEFAULT_LSTM_HIDDEN_DIM",
    "DEFAULT_N_ENVS",

    # PPO Hyperparameters
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_CLIP_RATIO",
    "DEFAULT_GAE_LAMBDA",
    "DEFAULT_VALUE_COEF",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_N_PPO_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ENTROPY_COEF",
    "DEFAULT_ENTROPY_COEF_MIN",

    # Reward Shaping Constants
    "DEFAULT_CONTRIBUTION_WEIGHT",
    "DEFAULT_PBRS_WEIGHT",
    "DEFAULT_RENT_WEIGHT",
    "DEFAULT_FOSSILIZE_TERMINAL_SCALE",

    # Lifecycle Gate Thresholds
    "DEFAULT_MIN_FOSSILIZE_CONTRIBUTION",
    "DEFAULT_GRADIENT_RATIO_THRESHOLD",
    "DEFAULT_MIN_PROBATION_STABILITY",
    "DEFAULT_GRADIENT_EMA_DECAY",

    # Host Stabilization
    "DEFAULT_STABILIZATION_THRESHOLD",
    "DEFAULT_STABILIZATION_EPOCHS",

    # Governor (Tolaria)
    "DEFAULT_GOVERNOR_SENSITIVITY",
    "DEFAULT_GOVERNOR_ABSOLUTE_THRESHOLD",
    "DEFAULT_GOVERNOR_DEATH_PENALTY",
    "DEFAULT_GOVERNOR_HISTORY_WINDOW",
    "DEFAULT_MIN_PANICS_BEFORE_ROLLBACK",

    # Heuristic Policy (Tamiyo)
    "DEFAULT_PLATEAU_EPOCHS_TO_GERMINATE",
    "DEFAULT_MIN_EPOCHS_BEFORE_GERMINATE",
    "DEFAULT_CULL_AFTER_EPOCHS_WITHOUT_IMPROVEMENT",
    "DEFAULT_CULL_IF_ACCURACY_DROPS_BY",
    "DEFAULT_EMBARGO_EPOCHS_AFTER_CULL",
    "DEFAULT_BLENDING_TOTAL_STEPS",

    # Task Training Defaults
    "DEFAULT_HOST_LR",
    "DEFAULT_SEED_LR",
    "DEFAULT_BATCH_SIZE_TRAINING",
    "DEFAULT_DROPOUT",

    # PPO Network Architecture
    "DEFAULT_VALUE_CLIP",
    "DEFAULT_FEATURE_DIM",

    # Blueprint Penalty System
    "DEFAULT_BLUEPRINT_PENALTY_ON_CULL",
    "DEFAULT_BLUEPRINT_PENALTY_DECAY",
    "DEFAULT_BLUEPRINT_PENALTY_THRESHOLD",

    # Lifecycle Gate Thresholds (QualityGates)
    "DEFAULT_MIN_TRAINING_IMPROVEMENT",
    "DEFAULT_MIN_BLENDING_EPOCHS",
    "DEFAULT_MAX_ISOLATION_VIOLATIONS",
    "DEFAULT_ALPHA_COMPLETE_THRESHOLD",
    "DEFAULT_MAX_PROBATION_EPOCHS",

    # Governor Anomaly Detection
    "DEFAULT_GOVERNOR_LOSS_MULTIPLIER",

    # Actions (build_action_enum used by HeuristicTamiyo)
    "build_action_enum",

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
    "TrainingMetrics",
    "TrainingSignals",

    # Schemas
    "SeedOperation",
    "OPERATION_TARGET_STAGE",
    "AdaptationCommand",
    "GateLevel",
    "GateResult",
    "BlueprintProtocol",
    "BlueprintSpec",

    # Reports
    "SeedMetrics",
    "SeedStateReport",
    "FieldReport",

    # Telemetry
    "TelemetryEventType",
    "TelemetryEvent",
    "PerformanceBudgets",
    "DEFAULT_BUDGETS",
    "SeedTelemetry",
]
