"""D2 Capacity Economics: Tests for slot saturation prevention.

These tests verify the D2 capacity economics feature which prevents early slot
saturation through occupancy rent and first-germinate bonus.

Bug fixed (2025-01-08): n_active_seeds and seeds_germinated_this_episode were
not wired through the reward pipeline, causing:
1. Occupancy rent to only count fossilized seeds (active seeds escaped the cost)
2. First-germinate bonus to fire on EVERY germinate action, not just the first
"""

from __future__ import annotations

import pytest

from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import (
    compute_contribution_reward,
    compute_reward,
    ContributionRewardConfig,
    RewardMode,
    SeedInfo,
)
from esper.simic.rewards.types import ContributionRewardInputs


def capacity_config(
    *,
    seed_occupancy_cost: float = 0.01,
    free_slots: int = 1,
    fossilized_maintenance_cost: float = 0.002,
    first_germinate_bonus: float = 0.2,
) -> ContributionRewardConfig:
    """Config that isolates D2 capacity economics from other reward components."""
    return ContributionRewardConfig(
        reward_mode=RewardMode.SHAPED,
        # Zero out other shaping to isolate capacity economics
        contribution_weight=0.0,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        germinate_cost=0.0,
        fossilize_cost=0.0,
        prune_cost=0.0,
        set_alpha_target_cost=0.0,
        germinate_with_seed_penalty=0.0,
        # D2 capacity economics - values under test
        seed_occupancy_cost=seed_occupancy_cost,
        free_slots=free_slots,
        fossilized_maintenance_cost=fossilized_maintenance_cost,
        first_germinate_bonus=first_germinate_bonus,
    )


class TestOccupancyRent:
    """Tests for per-epoch occupancy rent above free_slots threshold."""

    def test_no_rent_when_below_free_slots_threshold(self) -> None:
        """No occupancy rent when total occupied slots <= free_slots."""
        config = capacity_config(free_slots=2, seed_occupancy_cost=0.1)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            n_active_seeds=1,  # 1 active + 0 fossilized = 1 total, below threshold of 2
            num_fossilized_seeds=0,
        )

        assert components.occupancy_rent == 0.0

    def test_rent_applies_when_above_free_slots_threshold(self) -> None:
        """Occupancy rent applies for slots above the free threshold."""
        config = capacity_config(free_slots=1, seed_occupancy_cost=0.1)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            n_active_seeds=2,  # 2 active + 0 fossilized = 2 total, 1 above threshold
            num_fossilized_seeds=0,
        )

        # 2 occupied - 1 free = 1 excess * 0.1 cost = 0.1 rent
        assert components.occupancy_rent == pytest.approx(0.1)
        assert reward == pytest.approx(-0.1)

    def test_rent_counts_both_active_and_fossilized(self) -> None:
        """Occupancy rent counts both active AND fossilized seeds.

        ChatGPT Pro review 2025-01-08: Using only n_active made fossilizing an
        "escape hatch" from occupancy cost. Fossilized seeds still consume slots.
        """
        config = capacity_config(free_slots=1, seed_occupancy_cost=0.1)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            n_active_seeds=1,  # 1 active + 2 fossilized = 3 total, 2 above threshold
            num_fossilized_seeds=2,
        )

        # 3 occupied - 1 free = 2 excess * 0.1 cost = 0.2 rent
        assert components.occupancy_rent == pytest.approx(0.2)

    def test_fossilized_maintenance_rent_separate_from_occupancy(self) -> None:
        """Fossilized seeds incur additional maintenance rent (frozen compute cost)."""
        config = capacity_config(
            free_slots=10,  # High threshold so no occupancy rent
            seed_occupancy_cost=0.1,
            fossilized_maintenance_cost=0.05,
        )

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            n_active_seeds=0,
            num_fossilized_seeds=3,
        )

        # 3 fossilized * 0.05 = 0.15 fossilized rent
        assert components.occupancy_rent == 0.0  # Below free_slots threshold
        assert components.fossilized_rent == pytest.approx(0.15)
        assert reward == pytest.approx(-0.15)


class TestFirstGerminateBonus:
    """Tests for one-time first-germination bonus that breaks 'do nothing' symmetry."""

    def test_first_germinate_gets_bonus(self) -> None:
        """First germination in an episode receives the one-time bonus."""
        config = capacity_config(first_germinate_bonus=0.5)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,  # No existing seed
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            seeds_germinated_this_episode=0,  # FIRST germination
        )

        assert components.first_germinate_bonus == pytest.approx(0.5)
        assert reward == pytest.approx(0.5)

    def test_second_germinate_no_bonus(self) -> None:
        """Second germination does NOT receive the bonus.

        Bug fixed (2025-01-08): Previously seeds_germinated_this_episode was
        not wired, so it defaulted to 0 and every germinate got the bonus.
        """
        config = capacity_config(first_germinate_bonus=0.5)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            seeds_germinated_this_episode=1,  # NOT the first
        )

        assert components.first_germinate_bonus == 0.0
        assert reward == pytest.approx(0.0)

    def test_subsequent_germinates_no_bonus(self) -> None:
        """No bonus for third, fourth, etc. germinations."""
        config = capacity_config(first_germinate_bonus=0.5)

        for prev_germinates in [2, 3, 5, 10]:
            reward, components = compute_contribution_reward(
                action=LifecycleOp.GERMINATE,
                seed_contribution=None,
                val_acc=50.0,
                seed_info=None,
                epoch=1,
                max_epochs=10,
                config=config,
                return_components=True,
                seeds_germinated_this_episode=prev_germinates,
            )

            assert components.first_germinate_bonus == 0.0, (
                f"Should not get bonus when seeds_germinated={prev_germinates}"
            )


class TestContributionRewardInputsWiring:
    """Tests verifying the fix wires D2 fields through compute_reward()."""

    def test_compute_reward_passes_n_active_seeds(self) -> None:
        """compute_reward() correctly passes n_active_seeds to compute_contribution_reward()."""
        config = capacity_config(free_slots=0, seed_occupancy_cost=0.1)

        inputs = ContributionRewardInputs(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            total_params=1000,
            host_params=500,
            acc_at_germination=None,
            acc_delta=0.0,
            config=config,
            return_components=True,
            n_active_seeds=3,  # Field must be wired
            num_fossilized_seeds=0,
        )

        reward, components = compute_reward(inputs)

        # 3 active * 0.1 cost = 0.3 rent
        assert components.occupancy_rent == pytest.approx(0.3)

    def test_compute_reward_passes_seeds_germinated_this_episode(self) -> None:
        """compute_reward() correctly passes seeds_germinated_this_episode."""
        config = capacity_config(first_germinate_bonus=0.5)

        # First germinate should get bonus
        inputs_first = ContributionRewardInputs(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            total_params=1000,
            host_params=500,
            acc_at_germination=None,
            acc_delta=0.0,
            config=config,
            return_components=True,
            seeds_germinated_this_episode=0,
        )

        reward_first, components_first = compute_reward(inputs_first)
        assert components_first.first_germinate_bonus == pytest.approx(0.5)

        # Second germinate should NOT get bonus
        inputs_second = ContributionRewardInputs(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            total_params=1000,
            host_params=500,
            acc_at_germination=None,
            acc_delta=0.0,
            config=config,
            return_components=True,
            seeds_germinated_this_episode=1,  # Field must be wired
        )

        reward_second, components_second = compute_reward(inputs_second)
        assert components_second.first_germinate_bonus == 0.0


class TestCapacityEconomicsRLImplications:
    """Tests for RL-specific implications of capacity economics."""

    def test_occupancy_rent_creates_pressure_to_fossilize_or_prune(self) -> None:
        """High occupancy should create negative reward pressure.

        This encourages the agent to make decisions (fossilize or prune) rather
        than accumulating active seeds indefinitely.
        """
        config = capacity_config(
            free_slots=1,
            seed_occupancy_cost=0.05,
            fossilized_maintenance_cost=0.01,
        )

        # Many active seeds: high pressure
        _, high_active = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            n_active_seeds=5,
            num_fossilized_seeds=0,
        )

        # Same count but fossilized: lower pressure (fossilized_maintenance < occupancy)
        _, high_fossilized = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            n_active_seeds=0,
            num_fossilized_seeds=5,
        )

        # Both should have some rent
        assert high_active.occupancy_rent > 0
        assert high_fossilized.occupancy_rent > 0

        # Fossilized also has maintenance cost
        assert high_fossilized.fossilized_rent == pytest.approx(5 * 0.01)

    def test_first_germinate_bonus_breaks_do_nothing_symmetry(self) -> None:
        """First germinate bonus should make GERMINATE preferable to WAIT at start.

        Without this bonus, an agent could learn to always WAIT because there's
        no immediate reward for taking the first action.
        """
        config = capacity_config(first_germinate_bonus=0.2)

        # WAIT action
        wait_reward, _ = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            seeds_germinated_this_episode=0,
        )

        # GERMINATE action (first one)
        germinate_reward, _ = compute_contribution_reward(
            action=LifecycleOp.GERMINATE,
            seed_contribution=None,
            val_acc=50.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
            seeds_germinated_this_episode=0,
        )

        # First GERMINATE should be more rewarding than WAIT
        assert germinate_reward > wait_reward
        assert germinate_reward == pytest.approx(0.2)
        assert wait_reward == pytest.approx(0.0)
