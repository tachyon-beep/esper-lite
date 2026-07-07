"""Stateful escrow ledger tests.

These tests simulate multiple consecutive reward steps and assert the "ledger
conservation laws":

- `credit_next == credit_prev + escrow_delta` (always)
- positive branch jumps to the full target difference; no reward-path clipping
"""

import pytest

from esper.simic.rewards import compute_contribution_reward

from tests.simic.rewards.escrow.harness import LifecycleOp, SeedStage, escrow_config, seed_info


def test_credit_continuity_invariant_credit_next_equals_prev_plus_delta():
    config = escrow_config()
    credit = 0.0

    # Fixed target: progress None -> target = contribution * 0.5 = 0.5
    for _ in range(8):
        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=1.0,
            val_acc=60.0,
            seed_info=seed_info(stage=SeedStage.TRAINING),
            epoch=1,
            max_epochs=10,
            total_params=100_000,
            host_params=100_000,
            config=config,
            return_components=True,
            stable_val_acc=60.0,
            escrow_credit_prev=credit,
            acc_at_germination=None,
        )
        assert components.escrow_credit_next == pytest.approx(
            credit + components.escrow_delta
        )
        credit = components.escrow_credit_next


def test_positive_branch_jumps_to_target_without_delta_clip():
    config = escrow_config()

    target = 0.5  # seed_contribution=1.0, progress None -> 0.5
    credit = 0.0

    expected = [0.5, 0.5]
    for step_idx, expected_credit in enumerate(expected, start=1):
        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=1.0,
            val_acc=60.0,
            seed_info=seed_info(stage=SeedStage.TRAINING),
            epoch=step_idx,
            max_epochs=10,
            total_params=100_000,
            host_params=100_000,
            config=config,
            return_components=True,
            stable_val_acc=60.0,
            escrow_credit_prev=credit,
            acc_at_germination=None,
        )

        assert components.escrow_credit_target == pytest.approx(target)
        assert components.escrow_credit_next == pytest.approx(expected_credit)
        assert components.escrow_credit_next <= target + 1e-9
        assert components.escrow_delta == pytest.approx(target - credit)

        credit = components.escrow_credit_next
