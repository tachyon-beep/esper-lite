"""Hypothesis strategies for reward function property tests.

These strategies generate valid inputs for compute_contribution_reward()
and related functions, enabling property-based testing across the full
input space.
"""

from hypothesis import strategies as st
from hypothesis.strategies import composite, sampled_from

from esper.leyline import LifecycleOp, MIN_PRUNE_AGE, SeedStage
from esper.simic.rewards import SeedInfo

# Stage values for strategies
ACTIVE_STAGES = [
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
    SeedStage.HOLDING.value,
    SeedStage.FOSSILIZED.value,
]

PRE_BLENDING_STAGES = [
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
]

BLENDING_PLUS_STAGES = [
    SeedStage.BLENDING.value,
    SeedStage.HOLDING.value,
    SeedStage.FOSSILIZED.value,
]


@composite
def seed_infos(draw, stage=None):
    """Generate arbitrary valid SeedInfo objects.

    Args:
        stage: If provided, fixes the stage to this value.
               Otherwise draws from all active stages.
    """
    if stage is None:
        stage = draw(sampled_from(ACTIVE_STAGES))

    # Previous stage must be valid predecessor (or same for epochs_in_stage > 0)
    previous_stage = draw(st.integers(0, stage))

    # Epochs in stage: 0 means just transitioned
    epochs_in_stage = draw(st.integers(0, 25))

    # Previous epochs: only meaningful if epochs_in_stage == 0 (just transitioned)
    previous_epochs = draw(st.integers(0, 20)) if epochs_in_stage == 0 else 0

    # Seed age must be >= epochs_in_stage
    min_age = max(epochs_in_stage, 1)
    seed_age = draw(st.integers(min_age, 50))

    return SeedInfo(
        stage=stage,
        improvement_since_stage_start=draw(st.floats(-10.0, 10.0, allow_nan=False)),
        total_improvement=draw(st.floats(-5.0, 10.0, allow_nan=False)),
        epochs_in_stage=epochs_in_stage,
        seed_params=draw(st.integers(0, 1_000_000)),
        previous_stage=previous_stage,
        previous_epochs_in_stage=previous_epochs,
        seed_age_epochs=seed_age,
    )


@composite
def seed_infos_at_stage(draw, stage: int):
    """Generate SeedInfo fixed at a specific stage."""
    return draw(seed_infos(stage=stage))


def lifecycle_ops():
    """Strategy for lifecycle operations."""
    return sampled_from([
        LifecycleOp.WAIT,
        LifecycleOp.GERMINATE,
        LifecycleOp.PRUNE,
        LifecycleOp.FOSSILIZE,
        LifecycleOp.ADVANCE,
    ])


@composite
def reward_inputs(draw, with_seed: bool | None = None):
    """Generate complete inputs for compute_contribution_reward().

    Args:
        with_seed: If True, always include seed. If False, never include seed.
                   If None, randomly decide.
    """
    # Decide if we have a seed
    if with_seed is None:
        has_seed = draw(st.booleans())
    else:
        has_seed = with_seed

    seed_info = draw(seed_infos()) if has_seed else None

    # Seed contribution only available for BLENDING+ stages
    seed_contribution = None
    if seed_info is not None and seed_info.stage in BLENDING_PLUS_STAGES:
        seed_contribution = draw(st.floats(-5.0, 15.0, allow_nan=False))

    # Action selection
    action = draw(lifecycle_ops())

    # Epoch within episode
    max_epochs = 25
    epoch = draw(st.integers(1, max_epochs))

    # Accuracy values
    val_acc = draw(st.floats(0.0, 100.0, allow_nan=False))
    acc_at_germination = draw(st.floats(0.0, val_acc, allow_nan=False)) if has_seed else None
    acc_delta = draw(st.floats(-5.0, 5.0, allow_nan=False))

    # Parameter counts
    host_params = draw(st.integers(1, 10_000_000))
    seed_params = seed_info.seed_params if seed_info else 0
    total_params = host_params + seed_params

    # Fossilized seed counts (for terminal bonus)
    num_fossilized = draw(st.integers(0, 10))
    num_contributing = draw(st.integers(0, num_fossilized))

    return {
        "action": action,
        "seed_contribution": seed_contribution,
        "val_acc": val_acc,
        "seed_info": seed_info,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "total_params": total_params,
        "host_params": host_params,
        "acc_at_germination": acc_at_germination,
        "acc_delta": acc_delta,
        "num_fossilized_seeds": num_fossilized,
        "num_contributing_fossilized": num_contributing,
    }


@composite
def reward_inputs_with_seed(draw):
    """Generate inputs that always have an active seed."""
    return draw(reward_inputs(with_seed=True))


@composite
def reward_inputs_without_seed(draw):
    """Generate inputs without a seed (DORMANT slot)."""
    return draw(reward_inputs(with_seed=False))


@composite
def ransomware_seed_inputs(draw):
    """Generate inputs matching the ransomware signature.

    Ransomware pattern: high counterfactual contribution but negative
    total_improvement - the seed created dependencies without adding value.
    """
    # Force BLENDING or HOLDING stage (where counterfactual is available)
    stage = draw(sampled_from([SeedStage.BLENDING.value, SeedStage.HOLDING.value]))

    seed_info = SeedInfo(
        stage=stage,
        improvement_since_stage_start=draw(st.floats(-3.0, 0.0, allow_nan=False)),
        total_improvement=draw(st.floats(-2.0, -0.2, allow_nan=False)),  # Negative!
        epochs_in_stage=draw(st.integers(1, 10)),
        seed_params=draw(st.integers(10_000, 500_000)),
        previous_stage=stage - 1,
        previous_epochs_in_stage=draw(st.integers(1, 5)),
        seed_age_epochs=draw(st.integers(5, 20)),
    )

    # High counterfactual contribution (the "ransom")
    seed_contribution = draw(st.floats(1.0, 10.0, allow_nan=False))

    val_acc = draw(st.floats(50.0, 90.0, allow_nan=False))
    acc_at_germination = min(100.0, val_acc + abs(seed_info.total_improvement))  # Was better before

    return {
        "action": LifecycleOp.WAIT,  # Default action
        "seed_contribution": seed_contribution,
        "val_acc": val_acc,
        "seed_info": seed_info,
        "epoch": draw(st.integers(10, 25)),
        "max_epochs": 25,
        "total_params": draw(st.integers(100_000, 1_000_000)),
        "host_params": draw(st.integers(100_000, 500_000)),
        "acc_at_germination": acc_at_germination,
        "acc_delta": draw(st.floats(-1.0, 0.5, allow_nan=False)),
        "num_fossilized_seeds": 0,
        "num_contributing_fossilized": 0,
    }


@composite
def fossilize_inputs(draw, valid: bool = True):
    """Generate inputs for FOSSILIZE action.

    Args:
        valid: If True, generate valid fossilize context (HOLDING stage).
               If False, generate invalid context.
    """
    if valid:
        stage = SeedStage.HOLDING.value
    else:
        stage = draw(sampled_from([
            SeedStage.GERMINATED.value,
            SeedStage.TRAINING.value,
            SeedStage.BLENDING.value,
            SeedStage.FOSSILIZED.value,  # Already fossilized
        ]))

    seed_info = draw(seed_infos(stage=stage))

    # Only generate seed_contribution for BLENDING+ stages
    seed_contribution = None
    if stage in BLENDING_PLUS_STAGES:
        seed_contribution = draw(st.floats(-2.0, 10.0, allow_nan=False))

    return {
        "action": LifecycleOp.FOSSILIZE,
        "seed_contribution": seed_contribution,
        "val_acc": draw(st.floats(50.0, 95.0, allow_nan=False)),
        "seed_info": seed_info,
        "epoch": draw(st.integers(5, 25)),
        "max_epochs": 25,
        "total_params": draw(st.integers(100_000, 500_000)),
        "host_params": draw(st.integers(100_000, 400_000)),
        "acc_at_germination": draw(st.floats(40.0, 70.0, allow_nan=False)),
        "acc_delta": draw(st.floats(-1.0, 2.0, allow_nan=False)),
        "num_fossilized_seeds": draw(st.integers(0, 5)),
        "num_contributing_fossilized": draw(st.integers(0, 3)),
    }


@composite
def prune_inputs(draw, valid: bool = True):
    """Generate inputs for PRUNE action.

    Args:
        valid: If True, generate valid prune context (not FOSSILIZED, meets age).
               If False, generate invalid context.
    """
    if valid:
        stage = draw(sampled_from([
            SeedStage.GERMINATED.value,
            SeedStage.TRAINING.value,
            SeedStage.BLENDING.value,
            SeedStage.HOLDING.value,
        ]))
        seed_age = draw(st.integers(MIN_PRUNE_AGE, 25))
    else:
        # Invalid: either fossilized or too young
        if draw(st.booleans()):
            stage = SeedStage.FOSSILIZED.value
            seed_age = draw(st.integers(5, 25))
        else:
            stage = draw(sampled_from(ACTIVE_STAGES[:4]))  # Non-fossilized
            seed_age = 0  # Too young

    seed_info = SeedInfo(
        stage=stage,
        improvement_since_stage_start=draw(st.floats(-5.0, 5.0, allow_nan=False)),
        total_improvement=draw(st.floats(-3.0, 5.0, allow_nan=False)),
        epochs_in_stage=draw(st.integers(0, 10)),
        seed_params=draw(st.integers(10_000, 200_000)),
        previous_stage=max(0, stage - 1),
        previous_epochs_in_stage=draw(st.integers(0, 5)),
        seed_age_epochs=seed_age,
    )

    seed_contribution = None
    if stage in BLENDING_PLUS_STAGES:
        seed_contribution = draw(st.floats(-5.0, 5.0, allow_nan=False))

    return {
        "action": LifecycleOp.PRUNE,
        "seed_contribution": seed_contribution,
        "val_acc": draw(st.floats(50.0, 90.0, allow_nan=False)),
        "seed_info": seed_info,
        "epoch": draw(st.integers(1, 25)),
        "max_epochs": 25,
        "total_params": draw(st.integers(100_000, 300_000)),
        "host_params": draw(st.integers(100_000, 250_000)),
        "acc_at_germination": draw(st.floats(40.0, 70.0, allow_nan=False)),
        "acc_delta": draw(st.floats(-2.0, 2.0, allow_nan=False)),
        "num_fossilized_seeds": draw(st.integers(0, 3)),
        "num_contributing_fossilized": draw(st.integers(0, 2)),
    }


@composite
def stage_sequences(draw, min_length: int = 3, max_length: int = 15):
    """Generate valid stage transition sequences for PBRS telescoping tests.

    Returns a list of (stage, epochs_in_stage) tuples representing a valid
    seed lifecycle trajectory.
    """
    # Start from GERMINATED
    sequence = [(SeedStage.GERMINATED.value, draw(st.integers(1, 5)))]

    # Progress through stages (may skip some)
    stage_order = [
        SeedStage.TRAINING.value,
        SeedStage.BLENDING.value,
        SeedStage.HOLDING.value,
        SeedStage.FOSSILIZED.value,
    ]

    current_idx = 0
    length = draw(st.integers(min_length, max_length))

    while len(sequence) < length and current_idx < len(stage_order):
        # Spend some epochs in current stage
        epochs = draw(st.integers(1, 5))
        sequence.append((stage_order[current_idx], epochs))

        # Maybe advance to next stage
        if draw(st.booleans()):
            current_idx += 1

    return sequence
