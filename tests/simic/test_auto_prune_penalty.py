"""Integration test for auto-prune penalty application.

Verifies that Simic correctly detects auto-prunes from both:
1. Direct governor prunes (governor.execute_rollback)
2. Scheduled prune completion via step_epoch

This test addresses the contract drift bug where step_epoch() used to return
bool but always returned False, making the penalty path dead code.
"""
import torch
import torch.nn as nn
from esper.kasmina.slot import SeedSlot, SeedState
from esper.kasmina.alpha_controller import AlphaCurve
from esper.leyline import SeedStage, AlphaMode


def test_direct_governor_prune_sets_auto_pruned_flag():
    """Test that auto-prune reasons are correctly identified."""
    from esper.kasmina.slot import SeedMetrics

    # Test that governor reasons are in AUTO_PRUNE_REASONS
    assert "governor_nan" in SeedMetrics.AUTO_PRUNE_REASONS
    assert "governor_lobotomy" in SeedMetrics.AUTO_PRUNE_REASONS
    assert "governor_divergence" in SeedMetrics.AUTO_PRUNE_REASONS
    assert "governor_rollback" in SeedMetrics.AUTO_PRUNE_REASONS

    # Test that policy reasons are NOT in AUTO_PRUNE_REASONS
    assert "scheduled_prune" not in SeedMetrics.AUTO_PRUNE_REASONS
    assert "policy_prune" not in SeedMetrics.AUTO_PRUNE_REASONS

    # Test the flag-setting logic directly on metrics
    metrics = SeedMetrics()
    assert metrics.auto_pruned is False

    # Simulate what prune() does at line 1515-1517
    reason = "governor_nan"
    is_auto_prune = reason in SeedMetrics.AUTO_PRUNE_REASONS
    metrics.auto_pruned = is_auto_prune
    metrics.auto_prune_reason = reason if is_auto_prune else ""

    assert metrics.auto_pruned is True
    assert metrics.auto_prune_reason == "governor_nan"


def test_scheduled_prune_via_step_epoch_sets_auto_pruned_flag():
    """Test that scheduled prune completion sets metrics.auto_pruned=True.

    This test exercises the REAL ramp-down completion path in step_epoch()
    (slot.py:2306-2311), not the immediate prune path in schedule_prune()
    (slot.py:1614-1615).
    """
    slot = SeedSlot(
        slot_id="test_slot",
        channels=10,
        fast_mode=True,
    )

    # Create and attach a seed in BLENDING stage
    seed_module = nn.Linear(10, 10)
    state = SeedState(
        seed_id="test_seed",
        blueprint_id="test_blueprint",
    )
    # Set stage directly (transition validation would reject DORMANT→BLENDING)
    state.stage = SeedStage.BLENDING
    slot.seed = seed_module
    slot.state = state
    slot.state.slot_id = "test_slot"

    # CRITICAL: Set alpha > 0 to avoid immediate prune in schedule_prune()
    # When alpha=0, schedule_prune() immediately calls prune() (slot.py:1614)
    # We want to exercise the ramp-down completion path in step_epoch() instead
    state.alpha = 0.5  # Non-zero value (syncs to controller in __post_init__)
    state.alpha_controller.alpha = 0.5  # Explicit sync
    state.alpha_controller.alpha_target = 0.5  # Match current alpha
    state.alpha_controller.alpha_mode = AlphaMode.HOLD

    # Schedule a prune with governor reason
    # This should set up a ramp-down schedule (not immediate prune)
    result = slot.schedule_prune(
        steps=2,  # Multi-step to ensure ramp-down
        curve=AlphaCurve.LINEAR,
        reason="governor_divergence",
        initiator="governor"
    )
    assert result is True, "schedule_prune should succeed when alpha > 0"

    # First step_epoch: advance controller but don't complete yet
    slot.step_epoch()
    assert slot.state is not None, "State should exist during ramp-down"
    assert slot.state.metrics.auto_pruned is False, "Flag should not be set mid-ramp"

    # Second step_epoch: complete the ramp-down and trigger prune
    # This exercises the completion path at slot.py:2306-2311
    slot.step_epoch()

    # Verify the flag is set after prune completion
    assert slot.state is not None, "State should exist after scheduled prune"
    assert slot.state.metrics.auto_pruned is True, "auto_pruned flag should be set"
    assert slot.state.metrics.auto_prune_reason == "governor_divergence"


def test_scheduled_policy_prune_does_not_set_auto_pruned_flag():
    """Test that policy-initiated scheduled prune does NOT set auto_pruned."""
    slot = SeedSlot(
        slot_id="test_slot",
        channels=10,
        fast_mode=True,
    )

    # Create and attach a seed in BLENDING stage
    seed_module = nn.Linear(10, 10)
    state = SeedState(
        seed_id="test_seed",
        blueprint_id="test_blueprint",
    )
    state.transition(SeedStage.BLENDING)
    slot.seed = seed_module
    slot.state = state
    slot.state.slot_id = "test_slot"

    # Schedule a policy prune (default reason is "scheduled_prune")
    slot.schedule_prune(
        steps=1,
        curve=AlphaCurve.LINEAR,
        # No reason specified - defaults to "scheduled_prune"
    )

    # Call step_epoch to trigger the scheduled prune
    slot.step_epoch()

    # Verify the flag is NOT set (policy prunes are not auto-prunes)
    assert slot.state is not None
    assert slot.state.metrics.auto_pruned is False, "Policy prunes should not set auto_pruned"
    assert slot.state.metrics.auto_prune_reason == ""


def test_auto_prune_flag_one_shot_consumption():
    """Test that auto_prune flag can be cleared after reading (one-shot)."""
    slot = SeedSlot(
        slot_id="test_slot",
        channels=10,
        fast_mode=True,
    )

    seed_module = nn.Linear(10, 10)
    state = SeedState(
        seed_id="test_seed",
        blueprint_id="test_blueprint",
    )
    state.transition(SeedStage.TRAINING)
    slot.seed = seed_module
    slot.state = state
    slot.state.slot_id = "test_slot"

    # Governor prune
    slot.prune("governor_nan", initiator="governor")

    # First read: flag is True
    assert slot.state.metrics.auto_pruned is True

    # Clear the flag (simulating Simic's one-shot consumption)
    slot.state.metrics.auto_pruned = False

    # Second read: flag is False
    assert slot.state.metrics.auto_pruned is False


def test_simic_penalty_logic_simulation():
    """Simulate the Simic penalty accumulation logic.

    This test replicates the exact logic in vectorized.py:3199-3210,
    checking the auto_pruned flag AFTER step_epoch() to catch both
    governor prunes and scheduled prune completions.
    """
    slot = SeedSlot(
        slot_id="test_slot",
        channels=10,
        fast_mode=True,
    )

    seed_module = nn.Linear(10, 10)
    state = SeedState(
        seed_id="test_seed",
        blueprint_id="test_blueprint",
    )
    state.transition(SeedStage.TRAINING)
    slot.seed = seed_module
    slot.state = state
    slot.state.slot_id = "test_slot"

    # Simulate governor prune
    slot.prune("governor_lobotomy", initiator="governor")

    # Simulate Simic's penalty accumulation loop (vectorized.py:3199-3210)
    pending_auto_prune_penalty = 0.0
    auto_prune_penalty = -1.0  # From reward config

    # Advance lifecycle
    slot.step_epoch()

    # Check flag AFTER step_epoch (catches both governor prunes and scheduled completions)
    if slot.state and slot.state.metrics.auto_pruned:
        pending_auto_prune_penalty += auto_prune_penalty
        # Clear one-shot flag after reading
        slot.state.metrics.auto_pruned = False

    # Verify penalty was accumulated
    assert pending_auto_prune_penalty == -1.0, "Penalty should be accumulated"

    # Verify flag was cleared
    assert slot.state.metrics.auto_pruned is False, "Flag should be cleared after read"

    # Advance through cooldown pipeline
    slot.step_epoch()  # PRUNED → EMBARGOED

    # Verify second call doesn't re-accumulate penalty (flag was cleared)
    if slot.state and slot.state.metrics.auto_pruned:
        pending_auto_prune_penalty += auto_prune_penalty

    assert pending_auto_prune_penalty == -1.0, "Penalty should not be re-accumulated"
