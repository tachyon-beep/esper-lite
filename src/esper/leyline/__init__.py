"""Leyline - The invisible substrate of Esper.

Leyline defines the data contracts that flow between all Esper components.
Import from here for the public API.

OWNERSHIP BOUNDARY:
    This module owns all TRAINING BEHAVIOR constants - anything that affects:
    - Model updates (PPO hyperparameters, learning rates, clipping)
    - Reward calculation (PBRS weights, terminal bonuses)
    - Lifecycle gates (fossilization thresholds, prune criteria)
    - Anomaly detection thresholds for training (entropy collapse, ratio explosion)
    - Architecture constants (LSTM dim, episode length, batch sizes)

    TUI/display thresholds belong in karn/constants.py instead.
    When in doubt: if it affects training outcomes, it belongs here.

Example:
    from esper.leyline import SeedStage, TrainingSignals, FactoredAction, LifecycleOp
"""

# ruff: noqa: E402  # Imports intentionally follow constant definitions for clarity

from typing import Any

# Version
LEYLINE_VERSION = "0.2.0"

# =============================================================================
# Lifecycle Constants (shared across simic modules)
# =============================================================================

# Minimum seed age before PRUNE is allowed.
# Set to 5 epochs: seeds need training time to demonstrate value before being prunable.
# The original "let agent learn via rewards" approach (MIN_PRUNE_AGE=1) failed because
# short-term risk-aversion dominates - the agent learns PRUNE is "safe" and kills seeds
# before they can prove themselves (171/318 seeds pruned after just 1 epoch).
MIN_PRUNE_AGE = 5

# Epochs needed for confident seed quality assessment
FULL_EVALUATION_AGE = 10

# Minimum epochs in HOLDING to earn full fossilize bonus
MIN_HOLDING_EPOCHS = 5

# Seed limits (None = unlimited)
DEFAULT_MAX_SEEDS = None           # Global limit across all slots

# Episode outcome classification threshold (percent accuracy).
EPISODE_SUCCESS_THRESHOLD = 80.0

# =============================================================================
# PPO/PBRS Constants (shared between ppo_agent.py, rewards.py, buffer, vectorized)
# =============================================================================

# Discount factor for PPO and PBRS reward shaping.
# CRITICAL: PPO gamma MUST equal PBRS gamma for policy invariance (Ng et al., 1999).
# If they differ, reward shaping can change the optimal policy.
#
# Value 0.995 optimized for 150-epoch sequential scaffolding:
# - gamma^150 ≈ 0.47 preserves ~47% credit at episode end
# - gamma^100 ≈ 0.61 preserves ~61% credit at 2/3 point
# - gamma^50 ≈ 0.78 preserves ~78% credit at 1/3 point
#
# For 3-seed sequential scaffolding, Tamiyo needs to assign credit from
# epoch ~140 (Seed C fossilizes) back to epoch ~5 (Seed A germination decision).
# 0.995^135 ≈ 0.51 means early decisions still receive meaningful signal.
DEFAULT_GAMMA = 0.995

# =============================================================================
# Episode & Architecture Constants (MUST sync across simic modules)
# =============================================================================

# Episode length for CIFAR environments.
# This is the "rollout length" for Tamiyo - how many timesteps each env
# contributes to one Tamiyo training batch.
#
# 150 epochs is the MINIMUM viable horizon for SEQUENTIAL 3-seed scaffolding:
# - Germinate seed A at epoch 5
# - A trains/stabilizes by epoch 40, fossilizes by epoch 50
# - Germinate seed B at epoch 55 (benefiting from A's learned features)
# - B trains/stabilizes by epoch 95, fossilizes by epoch 105
# - Germinate seed C at epoch 110 (building on A+B)
# - C trains/stabilizes by epoch 145
#
# The 150-200 epoch range allows Tamiyo to learn the *strategic* value of
# sequential planting - that germinating Seed C after Seed B stabilizes
# produces better synergies than parallel germination.
#
# Used by: config.py, vectorized.py, ppo_agent.py (chunk_length, max_steps_per_env)
DEFAULT_EPISODE_LENGTH = 150

# Maximum epochs a seed can spend in a single stage for normalization purposes.
# Derived from episode length - a seed could theoretically stay in one stage
# for the entire episode. Used for epochs_in_stage_norm feature.
# Used by: tamiyo/policy/features.py
MAX_EPOCHS_IN_STAGE = DEFAULT_EPISODE_LENGTH

# LSTM hidden dimension - architecture constant for temporal memory.
# Must match across network construction and buffer state tracking.
#
# 512 is required for 150-epoch sequential 3-seed scaffolding due to:
# - "Accumulating Context": LSTM must remember Seed A's archival info while
#   processing Seed B and C's active gradients over 100+ timesteps
# - "PPO Horizon Cut Risk": Hidden state is the only bridge connecting
#   step 150 to step 1 across rollout boundaries
#
# 256 dims risk "Catastrophic Overwrite" - Seed C's gradient flood may
# evict earlier seeds' learned representations.
#
# Used by: config.py, vectorized.py, ppo_agent.py, rollout_buffer.py, network.py
DEFAULT_LSTM_HIDDEN_DIM = 512

# Number of LSTM layers in the host model.
# Used by: Karn TUI widgets (gradient health display), telemetry dashboards.
# NOTE: Reduced from 12 to 4 to fix vanishing gradient problem.
# With 12 stacked layers (no residuals), only ~3% of gradient reached layer 1.
# 4 layers with residual connections (see ResidualLSTM) provides better gradient flow.
DEFAULT_HOST_LSTM_LAYERS = 4

# Parallel environments for vectorized training.
# This controls sample DIVERSITY per Tamiyo update, not training quantity.
# More envs = richer/more varied experience per PPO batch, but same number
# of Tamiyo gradient updates. Affects GPU memory usage.
# Used by: config.py, vectorized.py, ppo_agent.py, train.py CLI
DEFAULT_N_ENVS = 4

# Default number of injection slots (Kasmina capacity points).
# 3 slots allows sequential scaffolding: Seed A stabilizes, then B, then C.
# Used by: SlotConfig defaults, feature size calculations, buffer shapes
DEFAULT_NUM_SLOTS = 3

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
# DRL expert recommendation for scaffolding: 0.98 for longer credit horizon
# (was 0.97; increased to capture delayed scaffold effects)
# Standard value is 0.95; higher reduces bias at cost of variance.
DEFAULT_GAE_LAMBDA = 0.98

# D4: Advantage standard deviation floor (prevents gradient amplification).
# During slot saturation (forced WAIT corridors), advantage variance can collapse
# because all steps have similar value estimates. When std drops below this floor,
# we clamp it to prevent normalization from amplifying noise into huge gradients.
# Typical healthy std: 0.5-2.0; below 0.1 indicates a degenerate batch.
ADVANTAGE_STD_FLOOR: float = 0.1

# Value function loss coefficient in combined PPO loss.
# 1.0 gives critic equal weight with policy, important when value head
# is underfitting (negative explained variance from batch 1).
DEFAULT_VALUE_COEF = 1.0

# Maximum gradient norm for clipping (prevents exploding gradients).
# History: 0.5 caused 100% saturation with 12-layer LSTM; 1.0 still caused
# 14-27x clipping which killed learning signal (ratio=1.0, KL=0, q_var=1e-7).
# 5.0 allows ~60-80% of gradient magnitude through when pre-clip is 14-27,
# preserving direction while preventing true explosions (>100 norm).
DEFAULT_MAX_GRAD_NORM = 5.0

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

# M11: Entropy collapse detection thresholds.
# These are normalized entropy values from MaskedCategorical.entropy().
# - Collapse: Policy is nearly deterministic (<10% of max entropy)
# - Warning: Policy is converging, may need entropy boost (<30% of max)
DEFAULT_ENTROPY_COLLAPSE_THRESHOLD = 0.1
DEFAULT_ENTROPY_WARNING_THRESHOLD = 0.3

# Per-head entropy floor targets (normalized entropy, 0-1 scale)
# DRL Expert update (2026-01-11): Increased op floor from 0.25 to 0.30
# to push further from collapse point. With 6 actions, 0.30 normalized
# means minimum ~52% of maximum entropy - enough to maintain exploration.
ENTROPY_FLOOR_PER_HEAD: dict[str, float] = {
    "op": 0.30,           # INCREASED from 0.25 - push further from collapse
    "slot": 0.15,
    "blueprint": 0.20,    # INCREASED from 0.15 - needs room to explore
    "style": 0.15,
    "tempo": 0.20,        # INCREASED from 0.15 - needs room to explore
    "alpha_target": 0.10,
    "alpha_speed": 0.10,
    "alpha_curve": 0.10,
}

# Per-head entropy collapse thresholds (stricter than floor for detection)
# Entropy below this triggers anomaly detection
ENTROPY_COLLAPSE_PER_HEAD: dict[str, float] = {
    "op": 0.08,
    "slot": 0.10,
    "blueprint": 0.05,  # Lower threshold but still detect collapse
    "style": 0.08,
    "tempo": 0.05,
    "alpha_target": 0.08,
    "alpha_speed": 0.08,
    "alpha_curve": 0.08,
}

# Per-head entropy floor penalty coefficients
# DRL Expert update (2026-01-11): Increased op from 0.2 to 0.3, blueprint/tempo to 0.3
# Sparse heads need stronger penalty to overcome gradient starvation.
# Op head is critical - collapse there cascades to all other heads.
ENTROPY_FLOOR_PENALTY_COEF: dict[str, float] = {
    "op": 0.3,            # INCREASED from 0.2 - critical head, stronger enforcement
    "slot": 0.1,
    "blueprint": 0.3,     # INCREASED from 0.1 - sparse head needs strong penalty
    "style": 0.1,
    "tempo": 0.3,         # INCREASED from 0.1 - sparse head needs strong penalty
    "alpha_target": 0.1,
    "alpha_speed": 0.1,
    "alpha_curve": 0.1,
}

# Per-head probability floor (guarantees minimum exploration mass)
# These are HARD floors enforced in MaskedCategorical - probabilities are
# clamped and renormalized, ensuring gradients can always flow.
#
# DRL Expert diagnosis (2026-01-11): Op head collapse to WAIT is the root cause.
# When op chooses WAIT, sparse heads (blueprint, tempo) receive no gradients.
# AGGRESSIVE FLOORS: Previous 8% floor was insufficient - runs still collapsed
# to 99.9% WAIT within 24 batches. Increased to 15% op floor.
PROBABILITY_FLOOR_PER_HEAD: dict[str, float] = {
    "op": 0.15,           # INCREASED from 0.08 - guarantees ~15% non-WAIT exploration
    "slot": 0.05,         # Increased from 0.03 for more exploration
    "blueprint": 0.12,    # GERMINATE only (~5%) - needs high floor when active
    "style": 0.08,        # GERMINATE + SET_ALPHA_TARGET (~7%)
    "tempo": 0.12,        # GERMINATE only (~5%) - needs high floor when active
    "alpha_target": 0.08, # GERMINATE + SET_ALPHA_TARGET (~7%)
    "alpha_speed": 0.06,  # SET_ALPHA_TARGET + PRUNE (~7%)
    "alpha_curve": 0.06,  # SET_ALPHA_TARGET + PRUNE (~7%)
}

# M21: PPO ratio anomaly detection thresholds.
# ratio = exp(new_log_prob - old_log_prob). Healthy ratio is close to 1.0.
# - Explosion (>5.0): Policy changed too much, trust region violated
# - Collapse (<0.1): Policy severely underweights old actions (potential bug)
DEFAULT_RATIO_EXPLOSION_THRESHOLD = 5.0
DEFAULT_RATIO_COLLAPSE_THRESHOLD = 0.1

# =============================================================================
# Factored Action Space Constants
# =============================================================================

import math

# Head names for factored action space (slot selection, blueprint, blend algorithm, tempo,
# alpha target/speed/curve/algorithm, lifecycle op).
# Order matters: slot → blueprint → blend → tempo → alpha_target → alpha_speed → alpha_curve
# → alpha_algorithm → op is the causal chain.
from esper.leyline.factored_actions import (
    ACTION_HEAD_NAMES,
    ACTION_HEAD_SPECS,
    ActionHeadSpec,
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    ALPHA_CURVE_GLYPHS,
    ALPHA_CURVE_NAMES,
    ALPHA_SPEED_NAMES,
    ALPHA_SPEED_TO_STEPS,
    ALPHA_TARGET_NAMES,
    ALPHA_TARGET_VALUES,
    BLUEPRINT_IDS,
    BLUEPRINT_ID_TO_INDEX,
    BlueprintAction,
    CNN_BLUEPRINTS,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    OP_ADVANCE,
    OP_FOSSILIZE,
    OP_GERMINATE,
    OP_NAMES,
    OP_PRUNE,
    OP_SET_ALPHA_TARGET,
    OP_WAIT,
    STYLE_ALPHA_ALGORITHMS,
    STYLE_BLEND_IDS,
    STYLE_NAMES,
    STYLE_TO_KASMINA,
    TEMPO_NAMES,
    TEMPO_TO_EPOCHS,
    Topology,
    TRANSFORMER_BLUEPRINTS,
    VALID_TOPOLOGIES,
    TempoAction,
    get_action_head_sizes,
)

HEAD_NAMES: tuple[str, ...] = ACTION_HEAD_NAMES

# Max entropy per action head (ln(N) where N = action space size)
# Derived from enum sizes to prevent drift. Used by Karn UI for normalization.
# Keys are lowercase to match causal_masks.py head names.
HEAD_MAX_ENTROPIES: dict[str, float] = {
    "op": math.log(NUM_OPS),
    "slot": math.log(DEFAULT_NUM_SLOTS),  # Runtime config, default=3
    "blueprint": math.log(NUM_BLUEPRINTS),
    "style": math.log(NUM_STYLES),
    "tempo": math.log(NUM_TEMPO),
    "alpha_target": math.log(NUM_ALPHA_TARGETS),
    "alpha_speed": math.log(NUM_ALPHA_SPEEDS),
    "alpha_curve": math.log(NUM_ALPHA_CURVES),
}

# Action masking constant - safe for FP16/BF16, avoids softmax overflow
# Used by MaskedCategorical to zero out invalid action probabilities
MASKED_LOGIT_VALUE: float = -1e4

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

# Maximum scaffold hindsight credit per fossilization event.
# Credit is capped at 2x synergy bonus (0.1) to prevent runaway values.
# Used by Phase 3.2 scaffold hindsight credit mechanism.
MAX_HINDSIGHT_CREDIT = 0.2

# Per-scaffold credit weight (half of max to allow 2+ scaffolds to contribute).
# When multiple scaffolds help a beneficiary, each receives this weight,
# allowing meaningful contribution from multiple helpers before hitting the cap.
HINDSIGHT_CREDIT_WEIGHT = 0.1

# =============================================================================
# Lifecycle Gate Thresholds (seed state machine gates)
# =============================================================================

# G5 Gate: Minimum causal contribution (%) required for fossilization.
# Seeds with contribution < this threshold cannot fossilize successfully.
DEFAULT_MIN_FOSSILIZE_CONTRIBUTION = 1.0

# G2 Gate: Seed gradient ratio threshold for activity detection.
# Seeds with ratio below threshold may be considered inactive.
# (DRL Expert review 2025-12-17: increased from 0.05 to 0.10 to prevent
# seeds with transient gradient activity from passing G2 prematurely.
# Combined with EMA smoothing, this ensures meaningful sustained learning.)
DEFAULT_GRADIENT_RATIO_THRESHOLD = 0.10

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

# Minimum samples required for statistical anomaly detection.
# The governor silently skips detection if history has fewer samples.
# INVARIANT: history_window MUST be >= this value, or detection is disabled.
MIN_GOVERNOR_HISTORY_SAMPLES = 10

# Consecutive panics required before triggering a rollback.
# Higher = more conservative (avoids false positives from transients).
DEFAULT_MIN_PANICS_BEFORE_ROLLBACK = 3

# Loss multiplier threshold for statistical anomaly detection.
# Loss must be Nx the rolling average to trigger panic.
DEFAULT_GOVERNOR_LOSS_MULTIPLIER = 3.0

# =============================================================================
# Device Constants (Tolaria validation)
# =============================================================================

# Supported device types for Esper training.
# cpu: Universal fallback, always available
# cuda: NVIDIA GPU via CUDA
# mps: Apple Silicon GPU via Metal Performance Shaders
#
# Device types NOT supported (and why):
# - meta: Fake device with no storage - forward passes work but training silently fails
# - xla: TPU backend - requires separate torch_xla package and different training patterns
# - xpu: Intel GPUs - still maturing in PyTorch, needs explicit testing/support
# - hpu: Habana Gaudi - requires Intel Gaudi SDK
# - privateuseone: Custom backend - undefined behavior
SUPPORTED_DEVICE_TYPES: frozenset[str] = frozenset({"cpu", "cuda", "mps"})

# =============================================================================
# Display Thresholds (Karn UI)
# =============================================================================

# Growth ratio: (host+fossilized_params) / host_params
# Controls color coding in env_overview and scoreboard widgets
DEFAULT_GROWTH_RATIO_GREEN_MAX = 2.0   # <2x = green (efficient)
DEFAULT_GROWTH_RATIO_YELLOW_MAX = 5.0  # 2-5x = yellow (moderate), >5x = red (heavy)

# Seed lifecycle stage colors (Rich markup)
# Used across all Sanctum widgets for consistent visual language
STAGE_COLORS: dict[str, str] = {
    "DORMANT": "dim",
    "GERMINATED": "bright_blue",
    "TRAINING": "cyan",
    "HOLDING": "magenta",
    "BLENDING": "yellow",
    "FOSSILIZED": "green",
    "PRUNED": "red",
    "EMBARGOED": "bright_red",
    "RESETTING": "dim",
}

# Stage abbreviations for compact display
STAGE_ABBREVIATIONS: dict[str, str] = {
    "DORMANT": "Dorm",
    "GERMINATED": "Germ",
    "TRAINING": "Train",
    "HOLDING": "Hold",
    "BLENDING": "Blend",
    "FOSSILIZED": "Foss",
    "PRUNED": "Prune",
    "EMBARGOED": "Embar",
    "RESETTING": "Reset",
}

# =============================================================================
# Heuristic Policy (Tamiyo) Constants
# =============================================================================

# Number of plateau epochs before triggering germination.
# Plateau = no significant accuracy improvement (< plateau_threshold).
DEFAULT_PLATEAU_EPOCHS_TO_GERMINATE = 3

# Minimum training epochs before first germination allowed.
# Prevents premature seed creation during initial training.
DEFAULT_MIN_EPOCHS_BEFORE_GERMINATE = 5

# Epochs without improvement before pruning a seed.
DEFAULT_PRUNE_AFTER_EPOCHS_WITHOUT_IMPROVEMENT = 5

# Accuracy drop (%) that triggers immediate prune.
DEFAULT_PRUNE_IF_ACCURACY_DROPS_BY = 2.0

# Cooldown epochs after a prune before next germination allowed (anti-thrashing).
DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE = 5

# Fossilization threshold: minimum improvement required to fossilize a seed.
# Set to 0.5% to prevent reward hacking via marginal fossilization.
# A seed must demonstrate meaningful contribution before permanent integration.
DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE = 0.5

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
# Matches LSTM hidden dim to prevent information bottleneck.
# With 512 LSTM hidden, 128 feature dim would compress information
# unnecessarily before the LSTM can process it.
# Used by: tamiyo/policy/network.py
DEFAULT_FEATURE_DIM = 512

# =============================================================================
# Blueprint Embedding (Obs V3 Neural Encoding)
# =============================================================================

# Number of valid blueprints (indices 0-(NUM_BLUEPRINTS - 1)). Used for embedding table size.
# The embedding table has NUM_BLUEPRINTS + 1 entries to accommodate the null index.

# Index used for "no blueprint" / null embedding in BlueprintEmbedding.
# Must be >= NUM_BLUEPRINTS to avoid collision with valid blueprint indices.
# Used in: torch.where(blueprint_indices < 0, _null_idx, blueprint_indices)
# Stored as register_buffer to avoid torch.tensor() allocation per forward call.
BLUEPRINT_NULL_INDEX = NUM_BLUEPRINTS

# Embedding dimension for blueprint vectors.
# Small (4) because blueprints are low-cardinality (13 types).
# Larger dims would overfit; 4 is sufficient for type discrimination.
# Total embedding params: (NUM_BLUEPRINTS + 1) * EMBED_DIM = 14 * 4 = 56
DEFAULT_BLUEPRINT_EMBED_DIM = 4

# Obs V3 base observation dimension (before blueprint embeddings are concatenated).
# For DEFAULT_NUM_SLOTS=3: 23 + (31 × 3) = 116 dims.
# The full Obs V3 input to the network is:
#   OBS_V3_NON_BLUEPRINT_DIM + (DEFAULT_NUM_SLOTS × DEFAULT_BLUEPRINT_EMBED_DIM) = 128 dims.
OBS_V3_BASE_FEATURE_SIZE = 23
OBS_V3_SLOT_FEATURE_SIZE = 31
OBS_V3_NON_BLUEPRINT_DIM = OBS_V3_BASE_FEATURE_SIZE + (OBS_V3_SLOT_FEATURE_SIZE * DEFAULT_NUM_SLOTS)

# Number of independent action heads in Tamiyo's factored action space.
# Heads: op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve.
NUM_ACTION_HEADS = 8

# Minimum log probability clamp for numerical stability.
# Prevents log(0) = -inf and ensures gradients remain finite.
# Used in: MaskedCategorical distribution log_prob calculations.
LOG_PROB_MIN = -100.0

# =============================================================================
# Blueprint Penalty System (Anti-Thrashing)
# =============================================================================

# Penalty added to a blueprint when its seed is pruned.
# Higher = more aggressive avoidance of failed blueprints.
DEFAULT_BLUEPRINT_PENALTY_ON_PRUNE = 2.0

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

# Minimum epochs in TRAINING stage before advancement to BLENDING allowed.
# Ensures seed has time to exit the "initial chaos" phase of training.
# PyTorch expert recommendation: 10-20 epochs for gradient stability.
DEFAULT_MIN_BLENDING_EPOCHS = 10

# Minimum gradient health (0-1) for safe blending (G2 gate, permissive mode).
# Seeds with gradient health below this threshold may destabilize the host.
# PyTorch expert recommendation: >= 0.7 for safety.
DEFAULT_MIN_GRADIENT_HEALTH_FOR_BLENDING = 0.7

# Alpha threshold for considering blending "complete" (G3 gate).
# Seeds must reach this alpha level to be considered fully blended.
DEFAULT_ALPHA_COMPLETE_THRESHOLD = 0.95

# Maximum epochs in HOLDING stage before auto-prune timeout.
# Prevents indecisive policies from holding seeds indefinitely.
DEFAULT_MAX_PROBATION_EPOCHS = 5

# =============================================================================
# Governor Anomaly Detection
# =============================================================================

# Loss multiplier threshold for governor panic.
# Loss must exceed (average * multiplier) to trigger anomaly detection.
DEFAULT_GOVERNOR_LOSS_MULTIPLIER = 3.0

# Slot ID formatting and parsing
from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
    parse_slot_id,
    validate_slot_id,
    slot_sort_key,
    validate_slot_ids,
)

# Slot configuration
from esper.leyline.injection_spec import InjectionSpec, SurfaceType
from esper.leyline.slot_config import SlotConfig

# Action name parsing utilities (build_action_enum moved to tamiyo.action_enums)
from esper.leyline.actions import (
    GERMINATE_PREFIX,
    get_blueprint_from_action_name,
    is_germinate_action_name,
)

# Stages and transitions
from esper.leyline.stages import (
    SeedStage,
    VALID_TRANSITIONS,
    is_valid_transition,
    is_terminal_stage,
    is_active_stage,
    is_failure_stage,
)

# Stage schema (centralized stage encoding contract)
from esper.leyline.stage_schema import (
    STAGE_SCHEMA_VERSION,
    VALID_STAGES,
    NUM_STAGES,
    STAGE_TO_INDEX,
    INDEX_TO_STAGE,
    VALID_STAGE_VALUES,
    RESERVED_STAGE_VALUES,
    stage_to_one_hot,
    stage_to_index,
    validate_stage_value,
)

# Signals
from esper.leyline.signals import (
    TrainingMetrics,
    TrainingSignals,
)

# Schemas and specifications
from esper.leyline.schemas import (
    SeedOperation,
    OPERATION_TARGET_STAGE,
    GateLevel,
    GateResult,
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
    TelemetryCallback,
    PerformanceBudgets,
    DEFAULT_BUDGETS,
    SeedTelemetry,
    # Typed payloads (see docs/plans/2025-12-25-typed-telemetry-payloads-design.md)
    TelemetryPayload,
    TrainingStartedPayload,
    CheckpointLoadedPayload,
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    TrendDetectedPayload,
    PPOUpdatePayload,
    MemoryWarningPayload,
    RewardHackingSuspectedPayload,
    TamiyoInitiatedPayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedGateEvaluatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    CounterfactualMatrixPayload,
    AnalyticsSnapshotPayload,
    HeadTelemetry,
    AnomalyDetectedPayload,
    PerformanceDegradationPayload,
    EpisodeOutcomePayload,
    GovernorRollbackPayload,
    GovernorPanicReason,
)

# Alpha controller contracts
from esper.leyline.alpha import (
    AlphaMode,
    AlphaCurve,
    AlphaAlgorithm,
)

# Type contracts for observations
from esper.leyline.types import (
    SeedObservationFields,
    SlotObservationFields,
)

# Causal masks for credit assignment (used by PPO + Karn UI)
# NOTE: Lazy-loaded to avoid torch import at module level.
# Access via module attribute (e.g., leyline.compute_causal_masks) or explicit import.
_CAUSAL_MASK_EXPORTS = ("compute_causal_masks", "compute_availability_masks", "HEAD_RELEVANCE_BY_OP", "is_head_relevant")

# Host protocol (Train Anything principle - ROADMAP #5)
from esper.leyline.host_protocol import HostProtocol

# Policy protocol (swappable Tamiyo policies)
# NOTE: Lazy-loaded to avoid torch import at module level.
_POLICY_PROTOCOL_EXPORTS = ("ActionResult", "EvalResult", "ForwardResult", "PolicyBundle")

# Seed protocols (decouple training from seed implementation)
from esper.leyline.seed_protocols import (
    SeedStateProtocol,
    SeedSlotProtocol,
    SlottedHostProtocol,
)

# Task configuration (cross-subsystem training config)
from esper.leyline.task_config import TaskConfig

# Reward configuration (cross-subsystem reward hyperparameters)
from esper.leyline.reward_config import LossRewardConfig

# Episode outcome (cross-subsystem Pareto analysis)
from esper.leyline.episode_outcome import EpisodeOutcome

# Output protocol (telemetry backend contract)
from esper.leyline.output_protocol import OutputBackend

# Governor protocol (fail-safe training watchdog)
from esper.leyline.governor_protocol import GovernorProtocol, GovernorReport

# Utility functions (cross-subsystem pure functions)
from esper.leyline.utils import safe

__all__ = [
    # Version
    "LEYLINE_VERSION",

    # Lifecycle constants
    "MIN_PRUNE_AGE",
    "FULL_EVALUATION_AGE",
    "MIN_HOLDING_EPOCHS",
    "DEFAULT_MAX_SEEDS",
    "EPISODE_SUCCESS_THRESHOLD",
    "DEFAULT_GAMMA",

    # Episode & Architecture constants
    "DEFAULT_EPISODE_LENGTH",
    "MAX_EPOCHS_IN_STAGE",
    "DEFAULT_LSTM_HIDDEN_DIM",
    "DEFAULT_HOST_LSTM_LAYERS",
    "DEFAULT_N_ENVS",

    # PPO Hyperparameters
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_CLIP_RATIO",
    "DEFAULT_GAE_LAMBDA",
    "ADVANTAGE_STD_FLOOR",
    "DEFAULT_VALUE_COEF",
    "DEFAULT_MAX_GRAD_NORM",
    "DEFAULT_N_PPO_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ENTROPY_COEF",
    "DEFAULT_ENTROPY_COEF_MIN",
    "DEFAULT_ENTROPY_COLLAPSE_THRESHOLD",
    "DEFAULT_ENTROPY_WARNING_THRESHOLD",
    "ENTROPY_FLOOR_PER_HEAD",
    "ENTROPY_COLLAPSE_PER_HEAD",
    "ENTROPY_FLOOR_PENALTY_COEF",
    "PROBABILITY_FLOOR_PER_HEAD",
    "DEFAULT_RATIO_EXPLOSION_THRESHOLD",
    "DEFAULT_RATIO_COLLAPSE_THRESHOLD",

    # Factored Action Space
    "ACTION_HEAD_NAMES",
    "ACTION_HEAD_SPECS",
    "ActionHeadSpec",
    "AlphaCurveAction",
    "AlphaSpeedAction",
    "AlphaTargetAction",
    "ALPHA_CURVE_GLYPHS",
    "ALPHA_CURVE_NAMES",
    "ALPHA_SPEED_NAMES",
    "ALPHA_SPEED_TO_STEPS",
    "ALPHA_TARGET_NAMES",
    "ALPHA_TARGET_VALUES",
    "BLUEPRINT_IDS",
    "BLUEPRINT_ID_TO_INDEX",
    "BlueprintAction",
    "CNN_BLUEPRINTS",
    "FactoredAction",
    "GerminationStyle",
    "get_action_head_sizes",
    "HEAD_NAMES",
    "HEAD_MAX_ENTROPIES",
    "HEAD_RELEVANCE_BY_OP",
    "is_head_relevant",
    "compute_causal_masks",
    "compute_availability_masks",
    "LifecycleOp",
    "MASKED_LOGIT_VALUE",
    "NUM_ALPHA_CURVES",
    "NUM_ALPHA_SPEEDS",
    "NUM_ALPHA_TARGETS",
    "NUM_BLUEPRINTS",
    "NUM_OPS",
    "NUM_STYLES",
    "NUM_TEMPO",
    "OP_ADVANCE",
    "OP_FOSSILIZE",
    "OP_GERMINATE",
    "OP_NAMES",
    "OP_PRUNE",
    "OP_SET_ALPHA_TARGET",
    "OP_WAIT",
    "STYLE_ALPHA_ALGORITHMS",
    "STYLE_BLEND_IDS",
    "STYLE_NAMES",
    "STYLE_TO_KASMINA",
    "TEMPO_NAMES",
    "TEMPO_TO_EPOCHS",
    "TempoAction",
    "Topology",
    "TRANSFORMER_BLUEPRINTS",
    "VALID_TOPOLOGIES",

    # Reward Shaping Constants
    "DEFAULT_CONTRIBUTION_WEIGHT",
    "DEFAULT_PBRS_WEIGHT",
    "DEFAULT_RENT_WEIGHT",
    "DEFAULT_FOSSILIZE_TERMINAL_SCALE",
    "MAX_HINDSIGHT_CREDIT",
    "HINDSIGHT_CREDIT_WEIGHT",

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
    "MIN_GOVERNOR_HISTORY_SAMPLES",
    "DEFAULT_MIN_PANICS_BEFORE_ROLLBACK",
    "DEFAULT_GOVERNOR_LOSS_MULTIPLIER",

    # Device Constants (Tolaria validation)
    "SUPPORTED_DEVICE_TYPES",

    # Display Thresholds (Karn UI)
    "DEFAULT_GROWTH_RATIO_GREEN_MAX",
    "DEFAULT_GROWTH_RATIO_YELLOW_MAX",
    "STAGE_COLORS",
    "STAGE_ABBREVIATIONS",

    # Heuristic Policy (Tamiyo)
    "DEFAULT_PLATEAU_EPOCHS_TO_GERMINATE",
    "DEFAULT_MIN_EPOCHS_BEFORE_GERMINATE",
    "DEFAULT_PRUNE_AFTER_EPOCHS_WITHOUT_IMPROVEMENT",
    "DEFAULT_PRUNE_IF_ACCURACY_DROPS_BY",
    "DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE",
    "DEFAULT_MIN_IMPROVEMENT_TO_FOSSILIZE",
    "DEFAULT_BLENDING_TOTAL_STEPS",

    # Task Training Defaults
    "DEFAULT_HOST_LR",
    "DEFAULT_SEED_LR",
    "DEFAULT_BATCH_SIZE_TRAINING",
    "DEFAULT_DROPOUT",

    # PPO Network Architecture
    "DEFAULT_VALUE_CLIP",
    "DEFAULT_FEATURE_DIM",

    # Blueprint Embedding (Obs V3)
    "NUM_BLUEPRINTS",
    "BLUEPRINT_NULL_INDEX",
    "DEFAULT_BLUEPRINT_EMBED_DIM",
    "OBS_V3_BASE_FEATURE_SIZE",
    "OBS_V3_SLOT_FEATURE_SIZE",
    "OBS_V3_NON_BLUEPRINT_DIM",
    "NUM_ACTION_HEADS",
    "LOG_PROB_MIN",

    # Blueprint Penalty System
    "DEFAULT_BLUEPRINT_PENALTY_ON_PRUNE",
    "DEFAULT_BLUEPRINT_PENALTY_DECAY",
    "DEFAULT_BLUEPRINT_PENALTY_THRESHOLD",

    # Lifecycle Gate Thresholds (QualityGates)
    "DEFAULT_MIN_TRAINING_IMPROVEMENT",
    "DEFAULT_MIN_BLENDING_EPOCHS",
    "DEFAULT_MIN_GRADIENT_HEALTH_FOR_BLENDING",
    "DEFAULT_ALPHA_COMPLETE_THRESHOLD",
    "DEFAULT_MAX_PROBATION_EPOCHS",

    # Governor Anomaly Detection
    "DEFAULT_GOVERNOR_LOSS_MULTIPLIER",

    # Slot ID
    "SlotIdError",
    "format_slot_id",
    "parse_slot_id",
    "validate_slot_id",
    "slot_sort_key",
    "validate_slot_ids",

    # Slot configuration
    "InjectionSpec",
    "SurfaceType",
    "SlotConfig",

    # Action name parsing (build_action_enum moved to tamiyo.action_enums)
    "GERMINATE_PREFIX",
    "get_blueprint_from_action_name",
    "is_germinate_action_name",

    # Stages
    "SeedStage",
    "VALID_TRANSITIONS",
    "is_valid_transition",
    "is_terminal_stage",
    "is_active_stage",
    "is_failure_stage",

    # Stage Schema
    "STAGE_SCHEMA_VERSION",
    "VALID_STAGES",
    "NUM_STAGES",
    "STAGE_TO_INDEX",
    "INDEX_TO_STAGE",
    "VALID_STAGE_VALUES",
    "RESERVED_STAGE_VALUES",
    "stage_to_one_hot",
    "stage_to_index",
    "validate_stage_value",

    # Signals
    "TrainingMetrics",
    "TrainingSignals",

    # Schemas
    "SeedOperation",
    "OPERATION_TARGET_STAGE",
    "GateLevel",
    "GateResult",

    # Reports
    "SeedMetrics",
    "SeedStateReport",
    "FieldReport",

    # Telemetry
    "TelemetryEventType",
    "TelemetryEvent",
    "TelemetryCallback",
    "PerformanceBudgets",
    "DEFAULT_BUDGETS",
    "SeedTelemetry",
    # Typed payloads
    "TelemetryPayload",
    "TrainingStartedPayload",
    "CheckpointLoadedPayload",
    "EpochCompletedPayload",
    "BatchEpochCompletedPayload",
    "TrendDetectedPayload",
    "PPOUpdatePayload",
    "MemoryWarningPayload",
    "RewardHackingSuspectedPayload",
    "TamiyoInitiatedPayload",
    "SeedGerminatedPayload",
    "SeedStageChangedPayload",
    "SeedGateEvaluatedPayload",
    "SeedFossilizedPayload",
    "SeedPrunedPayload",
    "CounterfactualMatrixPayload",
    "AnalyticsSnapshotPayload",
    "HeadTelemetry",
    "AnomalyDetectedPayload",
    "PerformanceDegradationPayload",
    "EpisodeOutcomePayload",
    "GovernorRollbackPayload",
    "GovernorPanicReason",

    # Alpha controller
    "AlphaMode",
    "AlphaCurve",
    "AlphaAlgorithm",

    # Type contracts
    "SeedObservationFields",
    "SlotObservationFields",

    # Host protocol
    "HostProtocol",

    # Policy protocol
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    "PolicyBundle",

    # Seed protocols
    "SeedStateProtocol",
    "SeedSlotProtocol",
    "SlottedHostProtocol",

    # Task configuration
    "TaskConfig",

    # Reward configuration
    "LossRewardConfig",

    # Episode outcome (Pareto analysis)
    "EpisodeOutcome",

    # Output protocol (telemetry backends)
    "OutputBackend",

    # Governor protocol (fail-safe watchdog)
    "GovernorProtocol",
    "GovernorReport",

    # Utility functions
    "safe",
]


def __getattr__(name: str) -> Any:
    """Lazy import for heavy modules (torch-dependent).

    This enables `from esper.leyline import compute_causal_masks` to work
    while deferring torch import until actually needed.
    """
    if name in _CAUSAL_MASK_EXPORTS:
        from esper.leyline.causal_masks import (
            compute_causal_masks,
            compute_availability_masks,
            HEAD_RELEVANCE_BY_OP,
            is_head_relevant,
        )
        # Cache in module globals for subsequent access
        globals()["compute_causal_masks"] = compute_causal_masks
        globals()["compute_availability_masks"] = compute_availability_masks
        globals()["HEAD_RELEVANCE_BY_OP"] = HEAD_RELEVANCE_BY_OP
        globals()["is_head_relevant"] = is_head_relevant
        return globals()[name]

    if name in _POLICY_PROTOCOL_EXPORTS:
        from esper.leyline.policy_protocol import (
            ActionResult,
            EvalResult,
            ForwardResult,
            PolicyBundle,
        )
        # Cache in module globals for subsequent access
        globals()["ActionResult"] = ActionResult
        globals()["EvalResult"] = EvalResult
        globals()["ForwardResult"] = ForwardResult
        globals()["PolicyBundle"] = PolicyBundle
        return globals()[name]

    raise AttributeError(f"module 'esper.leyline' has no attribute {name!r}")
