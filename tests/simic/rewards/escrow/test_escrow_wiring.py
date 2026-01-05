"""Integration-ish tests for ESCROW wiring.

These tests intentionally touch the same seams that escrow systems usually break:

- stable accuracy timing (off-by-one in history)
- slot targeting vs escrow ledger keys
- terminal clawback excluding fossilized seeds
"""

import pytest

from esper.leyline.slot_config import SlotConfig
from esper.simic.rewards import compute_contribution_reward
from esper.simic.training.vectorized import _resolve_target_slot
from esper.tamiyo.tracker import SignalTracker

from tests.simic.rewards.escrow.harness import (
    LifecycleOp,
    SeedStage,
    apply_terminal_escrow_forfeit,
    escrow_config,
    seed_info,
    stable_val_acc_from_history,
)


def test_stable_accuracy_history_includes_current_val_accuracy():
    tracker = SignalTracker(history_window=10)
    config_window = 3

    # Feed three epochs; the last value is LOWER so a missing-current bug would be obvious.
    for epoch, val_acc in enumerate([70.0, 80.0, 60.0], start=1):
        signals = tracker.update(
            epoch=epoch,
            global_step=epoch,
            train_loss=1.0,
            train_accuracy=0.0,
            val_loss=1.0,
            val_accuracy=val_acc,
            active_seeds=[],
            available_slots=1,
        )

    stable = stable_val_acc_from_history(signals.accuracy_history, window=config_window)
    assert stable == pytest.approx(60.0)


def test_terminal_clawback_excludes_fossilized_credit_and_charges_nonfossilized():
    config = escrow_config()

    credits = {"r0c0": 1.0, "r0c1": 2.0}
    fossilized = {"r0c0": True, "r0c1": False}

    # Target slot is fossilized: escrow delta must be 0 and credit must remain unchanged.
    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=3.0,
        val_acc=80.0,
        seed_info=seed_info(stage=SeedStage.FOSSILIZED),
        epoch=10,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=80.0,
        escrow_credit_prev=credits["r0c0"],
        acc_at_germination=70.0,
    )
    credits["r0c0"] = components.escrow_credit_next
    assert credits["r0c0"] == pytest.approx(1.0)

    reward_after, forfeit_component = apply_terminal_escrow_forfeit(
        reward=reward,
        credits_by_slot=credits,
        is_fossilized_by_slot=fossilized,
    )
    assert forfeit_component == pytest.approx(-2.0)
    assert reward_after == pytest.approx(reward - 2.0)


def test_disabled_slot_selected_is_not_in_escrow_ledger_keys():
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    enabled_slots = ["r0c0"]  # r0c1 is disabled

    target_slot, is_enabled = _resolve_target_slot(
        1, enabled_slots=enabled_slots, slot_config=slot_config
    )
    assert target_slot == "r0c1"
    assert is_enabled is False

    escrow_credit = {"r0c0": 0.0}
    with pytest.raises(KeyError):
        _ = escrow_credit[target_slot]


def test_out_of_range_slot_idx_falls_back_to_enabled_slot_zero():
    slot_config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    enabled_slots = ["r0c0"]

    target_slot, is_enabled = _resolve_target_slot(
        999, enabled_slots=enabled_slots, slot_config=slot_config
    )
    assert target_slot == "r0c0"
    assert is_enabled is False  # marked invalid even though target is deterministic

    escrow_credit = {"r0c0": 0.0}
    assert escrow_credit[target_slot] == 0.0


def test_invalid_op_mapping_sampled_prune_rewarded_wait_does_not_refund_credit():
    # This simulates the training loop behaviour where invalid ops are mapped to
    # `action_for_reward = WAIT` (e.g., PRUNE sampled but ineligible).
    config = escrow_config()
    credit_prev = 1.0

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,  # invalid PRUNE → rewarded WAIT
        seed_contribution=None,   # pre-counterfactual: escrow should not mint or claw back
        val_acc=70.0,
        seed_info=seed_info(stage=SeedStage.TRAINING),
        epoch=4,
        max_epochs=10,
        total_params=100_000,
        host_params=100_000,
        config=config,
        return_components=True,
        stable_val_acc=70.0,
        escrow_credit_prev=credit_prev,
        acc_at_germination=65.0,
    )

    assert components.escrow_delta == pytest.approx(0.0)
    assert components.escrow_credit_next == pytest.approx(credit_prev)
    assert reward == pytest.approx(0.0)

