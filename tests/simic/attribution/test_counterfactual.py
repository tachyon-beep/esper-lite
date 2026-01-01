"""Tests for counterfactual attribution engine.

B5-CR-01 regression tests: verify seeded RNG produces reproducible results.
"""

from esper.simic.attribution.counterfactual import (
    CounterfactualConfig,
    CounterfactualEngine,
    CounterfactualMatrix,
    CounterfactualResult,
)


class TestShapleyReproducibility:
    """B5-CR-01: Verify seeded RNG produces reproducible Shapley values."""

    def test_generate_configs_reproducible_with_seed(self):
        """Same seed should produce identical config sequences."""
        slot_ids = ["r0c0", "r0c1", "r0c2", "r0c3", "r0c4"]  # 5 slots -> Shapley

        config = CounterfactualConfig(strategy="shapley", shapley_samples=20, seed=42)
        engine1 = CounterfactualEngine(config)
        engine2 = CounterfactualEngine(config)

        configs1 = engine1.generate_configs(slot_ids)
        configs2 = engine2.generate_configs(slot_ids)

        assert configs1 == configs2, "Same seed should produce identical configs"

    def test_generate_configs_different_without_seed(self):
        """Without seed, different engines may produce different sequences.

        Note: This test is probabilistic - there's a small chance two unseeded
        RNGs produce the same sequence. We use enough samples to make this unlikely.
        """
        slot_ids = ["r0c0", "r0c1", "r0c2", "r0c3", "r0c4", "r0c5"]  # 6 slots

        config = CounterfactualConfig(
            strategy="shapley", shapley_samples=50, seed=None
        )
        engine1 = CounterfactualEngine(config)
        engine2 = CounterfactualEngine(config)

        configs1 = engine1.generate_configs(slot_ids)
        configs2 = engine2.generate_configs(slot_ids)

        # With 50 samples and no seed, sequences should differ
        # (extremely unlikely to be identical by chance)
        assert configs1 != configs2, "Unseeded engines should produce different configs"

    def test_compute_shapley_reproducible_with_seed(self):
        """Same seed should produce identical Shapley value estimates."""
        slot_ids = ["r0c0", "r0c1", "r0c2"]

        # Create a mock matrix with known values
        matrix = CounterfactualMatrix(epoch=1, strategy_used="shapley")
        # All possible configs for 3 slots (2^3 = 8)
        for i in range(8):
            config_tuple = tuple(bool(i & (1 << j)) for j in range(3))
            # Accuracy proportional to number of active slots
            accuracy = sum(config_tuple) * 0.1 + 0.5
            matrix.configs.append(
                CounterfactualResult(
                    config=config_tuple,
                    slot_ids=tuple(slot_ids),
                    val_accuracy=accuracy,
                    val_loss=1.0 - accuracy,
                )
            )

        config = CounterfactualConfig(shapley_samples=100, seed=12345)
        engine1 = CounterfactualEngine(config)
        engine2 = CounterfactualEngine(config)

        shapley1 = engine1.compute_shapley_values(matrix)
        shapley2 = engine2.compute_shapley_values(matrix)

        # Shapley values should be identical with same seed
        for slot_id in slot_ids:
            assert shapley1[slot_id].mean == shapley2[slot_id].mean, (
                f"Shapley mean for {slot_id} differs: "
                f"{shapley1[slot_id].mean} vs {shapley2[slot_id].mean}"
            )
            assert shapley1[slot_id].std == shapley2[slot_id].std, (
                f"Shapley std for {slot_id} differs"
            )

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different Shapley estimates."""
        slot_ids = ["r0c0", "r0c1", "r0c2", "r0c3", "r0c4"]

        config1 = CounterfactualConfig(strategy="shapley", shapley_samples=30, seed=1)
        config2 = CounterfactualConfig(strategy="shapley", shapley_samples=30, seed=2)

        engine1 = CounterfactualEngine(config1)
        engine2 = CounterfactualEngine(config2)

        configs1 = engine1.generate_configs(slot_ids)
        configs2 = engine2.generate_configs(slot_ids)

        assert configs1 != configs2, "Different seeds should produce different configs"


class TestCounterfactualConfigDefaults:
    """Test CounterfactualConfig default behavior."""

    def test_seed_default_is_none(self):
        """Seed should default to None for backwards compatibility."""
        config = CounterfactualConfig()
        assert config.seed is None

    def test_seed_can_be_set(self):
        """Seed should be settable in config."""
        config = CounterfactualConfig(seed=42)
        assert config.seed == 42


class TestCounterfactualHelperReset:
    """B8-CR-04: Test CounterfactualHelper.reset() behavior.

    Verifies that the public reset() method properly clears cached state,
    fixing the encapsulation violation where ParallelEnvState was directly
    accessing _last_matrix.
    """

    def test_reset_clears_last_matrix(self):
        """reset() should clear _last_matrix to None."""
        from esper.simic.attribution.counterfactual_helper import CounterfactualHelper

        helper = CounterfactualHelper(emit_events=False)

        # Manually create a matrix to set cached state
        matrix = CounterfactualMatrix(epoch=1, strategy_used="ablation_only")
        matrix.configs.append(
            CounterfactualResult(
                config=(True,),
                slot_ids=("r0c0",),
                val_accuracy=0.5,
                val_loss=0.5,
            )
        )
        helper._last_matrix = matrix  # Simulate a computed matrix

        # Verify state is set
        assert helper.last_matrix is not None
        assert helper.last_matrix is matrix

        # Call public reset method
        helper.reset()

        # Verify state is cleared
        assert helper.last_matrix is None

    def test_reset_idempotent(self):
        """reset() should be safe to call multiple times."""
        from esper.simic.attribution.counterfactual_helper import CounterfactualHelper

        helper = CounterfactualHelper(emit_events=False)

        # Call reset on fresh helper (already None)
        helper.reset()
        assert helper.last_matrix is None

        # Call reset again
        helper.reset()
        assert helper.last_matrix is None

    def test_reset_preserves_engine_config(self):
        """reset() should only clear cached results, not engine config."""
        from esper.simic.attribution.counterfactual_helper import CounterfactualHelper

        helper = CounterfactualHelper(
            strategy="shapley",
            shapley_samples=50,
            emit_events=False,
            seed=42,
        )

        # Set some cached state
        matrix = CounterfactualMatrix(epoch=1, strategy_used="shapley")
        helper._last_matrix = matrix

        # Reset
        helper.reset()

        # Engine config should be preserved
        assert helper.engine.config.strategy == "shapley"
        assert helper.engine.config.shapley_samples == 50
        assert helper.engine.config.seed == 42
