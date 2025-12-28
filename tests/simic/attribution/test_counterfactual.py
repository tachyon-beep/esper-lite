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
