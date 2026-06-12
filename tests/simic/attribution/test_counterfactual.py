"""Tests for counterfactual attribution engine.

B5-CR-01 regression tests: verify seeded RNG produces reproducible results.
"""

import pytest

from esper.simic.attribution.counterfactual import (
    CounterfactualConfig,
    CounterfactualEngine,
    CounterfactualMatrix,
    CounterfactualResult,
)
from esper.simic.attribution.counterfactual_helper import compute_simple_ablation


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


class TestShapleyPermutationCap:
    """B5-DRL-04: Verify warning when shapley_samples exceeds cap."""

    def test_warning_logged_when_samples_exceed_cap(self, caplog):
        """Warning should be logged when shapley_samples > 100."""
        import logging

        slot_ids = ["r0c0", "r0c1", "r0c2"]

        # Create a matrix with full factorial configs
        matrix = CounterfactualMatrix(epoch=1, strategy_used="shapley")
        for i in range(8):
            config_tuple = tuple(bool(i & (1 << j)) for j in range(3))
            matrix.configs.append(
                CounterfactualResult(
                    config=config_tuple,
                    slot_ids=tuple(slot_ids),
                    val_accuracy=0.5,
                    val_loss=0.5,
                )
            )

        # Configure with samples exceeding the cap
        config = CounterfactualConfig(shapley_samples=200, seed=42)
        engine = CounterfactualEngine(config)

        with caplog.at_level(logging.WARNING):
            engine.compute_shapley_values(matrix)

        # Verify warning was logged
        assert any(
            "Shapley permutations capped at 100" in record.message
            and "requested 200" in record.message
            for record in caplog.records
        ), "Expected warning about Shapley permutation cap"

    def test_no_warning_when_samples_within_cap(self, caplog):
        """No warning should be logged when shapley_samples <= 100."""
        import logging

        slot_ids = ["r0c0", "r0c1", "r0c2"]

        matrix = CounterfactualMatrix(epoch=1, strategy_used="shapley")
        for i in range(8):
            config_tuple = tuple(bool(i & (1 << j)) for j in range(3))
            matrix.configs.append(
                CounterfactualResult(
                    config=config_tuple,
                    slot_ids=tuple(slot_ids),
                    val_accuracy=0.5,
                    val_loss=0.5,
                )
            )

        # Configure within cap
        config = CounterfactualConfig(shapley_samples=50, seed=42)
        engine = CounterfactualEngine(config)

        with caplog.at_level(logging.WARNING):
            engine.compute_shapley_values(matrix)

        # Verify no warning was logged
        assert not any(
            "Shapley permutations capped" in record.message
            for record in caplog.records
        ), "No warning expected when within cap"


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


class TestZeroSeedEdgeCases:
    """Edge case tests for zero-seed attribution (from audit recommendations)."""

    def test_generate_configs_empty_slots(self):
        """generate_configs with empty slot list returns single empty config."""
        engine = CounterfactualEngine()
        configs = engine.generate_configs([])
        # Empty slot list should return single config (the empty configuration)
        assert len(configs) == 1
        assert configs[0] == ()

    def test_compute_matrix_empty_slots(self):
        """compute_matrix with empty slots returns valid empty matrix."""
        engine = CounterfactualEngine()

        def mock_evaluate(alpha_settings: dict) -> tuple[float, float]:
            # Empty alpha_settings means no seeds to evaluate
            assert alpha_settings == {}
            return (0.5, 0.7)  # (loss, accuracy)

        matrix = engine.compute_matrix([], mock_evaluate)

        assert matrix.strategy_used == "full_factorial"
        assert len(matrix.configs) == 1
        assert matrix.configs[0].config == ()
        assert matrix.configs[0].slot_ids == ()
        assert matrix.configs[0].val_accuracy == 0.7
        assert matrix.configs[0].val_loss == 0.5

    def test_compute_shapley_empty_matrix(self):
        """compute_shapley_values with empty matrix returns empty dict."""
        engine = CounterfactualEngine()
        empty_matrix = CounterfactualMatrix(epoch=0, strategy_used="shapley")

        shapley = engine.compute_shapley_values(empty_matrix)

        assert shapley == {}

    def test_compute_interaction_terms_empty_matrix(self):
        """compute_interaction_terms with empty matrix returns empty dict."""
        engine = CounterfactualEngine()
        empty_matrix = CounterfactualMatrix(epoch=0, strategy_used="full_factorial")

        interactions = engine.compute_interaction_terms(empty_matrix)

        assert interactions == {}

    def test_marginal_contribution_empty_matrix(self):
        """marginal_contribution on empty matrix returns empty dict."""
        matrix = CounterfactualMatrix(epoch=0)
        # Accessing internal method via property should work
        matrix._compute_marginal_contributions()

        assert matrix._marginal_contributions == {}

    def test_baseline_accuracy_no_all_false_config(self):
        """baseline_accuracy returns 0.0 when no all-false config exists."""
        matrix = CounterfactualMatrix(epoch=0)
        # Add only configs with at least one True
        matrix.configs.append(
            CounterfactualResult(
                config=(True,),
                slot_ids=("r0c0",),
                val_accuracy=0.8,
                val_loss=0.2,
            )
        )

        # No (False,) config exists
        assert matrix.baseline_accuracy == 0.0

    def test_full_accuracy_no_all_true_config(self):
        """full_accuracy returns 0.0 when no all-true config exists."""
        matrix = CounterfactualMatrix(epoch=0)
        # Add only configs with at least one False
        matrix.configs.append(
            CounterfactualResult(
                config=(False,),
                slot_ids=("r0c0",),
                val_accuracy=0.6,
                val_loss=0.4,
            )
        )

        # No (True,) config exists
        assert matrix.full_accuracy == 0.0

    def test_single_seed_shapley(self):
        """Shapley with single seed should return valid contribution."""
        slot_ids = ["r0c0"]
        engine = CounterfactualEngine(CounterfactualConfig(seed=42))

        matrix = CounterfactualMatrix(epoch=1, strategy_used="full_factorial")
        # Two configs: seed off and seed on
        matrix.configs.append(
            CounterfactualResult(
                config=(False,),
                slot_ids=tuple(slot_ids),
                val_accuracy=0.5,
                val_loss=0.5,
            )
        )
        matrix.configs.append(
            CounterfactualResult(
                config=(True,),
                slot_ids=tuple(slot_ids),
                val_accuracy=0.7,
                val_loss=0.3,
            )
        )

        shapley = engine.compute_shapley_values(matrix)

        # With single seed, Shapley = marginal contribution = 0.7 - 0.5 = 0.2
        assert "r0c0" in shapley
        assert abs(shapley["r0c0"].mean - 0.2) < 0.01


class TestCounterfactualInteractions:
    """Regression coverage for interaction term validity."""

    def test_compute_matrix_propagates_evaluation_failure(self):
        """Failed required evaluations should not return a partial matrix."""
        engine = CounterfactualEngine()

        def failing_evaluate(_alpha_settings: dict[str, float]) -> tuple[float, float]:
            raise RuntimeError("required config failed")

        with pytest.raises(RuntimeError, match="required config failed"):
            engine.compute_matrix(["r0c0", "r0c1"], failing_evaluate)

    def test_interaction_terms_reject_partial_matrix(self):
        """Missing required coalitions should not be substituted with 0.0."""
        engine = CounterfactualEngine()
        matrix = CounterfactualMatrix(epoch=1, strategy_used="full_factorial")
        matrix.configs.append(
            CounterfactualResult(
                config=(False, False),
                slot_ids=("r0c0", "r0c1"),
                val_accuracy=0.50,
                val_loss=0.50,
            )
        )
        matrix.configs.append(
            CounterfactualResult(
                config=(True, True),
                slot_ids=("r0c0", "r0c1"),
                val_accuracy=0.80,
                val_loss=0.20,
            )
        )

        with pytest.raises(ValueError, match="missing required configs"):
            engine.compute_interaction_terms(matrix)


class TestCounterfactualPrecomputedTiming:
    """Regression coverage for precomputed-matrix timing telemetry."""

    def test_compute_matrix_from_results_records_elapsed_time(self):
        """Fused validation callers must propagate measured compute time."""
        engine = CounterfactualEngine()

        matrix = engine.compute_matrix_from_results(
            slot_ids=["r0c0"],
            results={
                (False,): (0.5, 0.50),
                (True,): (0.3, 0.70),
            },
            compute_time_seconds=1.25,
        )

        assert matrix.source == "precomputed"
        assert matrix.compute_time_seconds == pytest.approx(1.25)

    def test_compute_matrix_from_results_rejects_negative_elapsed_time(self):
        """Negative elapsed time is invalid telemetry input."""
        engine = CounterfactualEngine()

        with pytest.raises(ValueError, match="compute_time_seconds"):
            engine.compute_matrix_from_results(
                slot_ids=["r0c0"],
                results={
                    (False,): (0.5, 0.50),
                    (True,): (0.3, 0.70),
                },
                compute_time_seconds=-0.1,
            )


class TestCounterfactualHelperShapley:
    """Regression coverage for helper-level Shapley processing."""

    def test_single_slot_results_compute_shapley(self):
        """Single-slot full-factorial results should produce Shapley telemetry data."""
        from esper.leyline import TelemetryEvent
        from esper.simic.attribution.counterfactual_helper import CounterfactualHelper

        events: list[TelemetryEvent] = []
        helper = CounterfactualHelper(seed=42, emit_callback=events.append)

        contributions = helper.compute_contributions_from_results(
            slot_ids=["r0c0"],
            results={
                (False,): (0.5, 0.50),
                (True,): (0.3, 0.70),
            },
            epoch=7,
            compute_time_seconds=0.5,
        )

        assert contributions["r0c0"].contribution == pytest.approx(0.20)
        assert contributions["r0c0"].shapley_mean == pytest.approx(0.20)
        assert contributions["r0c0"].shapley_std == pytest.approx(0.0)
        assert len(events) == 1
        assert events[0].data.kind == "shapley_computed"
        assert events[0].data.shapley_values["r0c0"]["mean"] == pytest.approx(0.20)

    def test_precomputed_matrix_marker_matches_slot_ids_and_epoch(self):
        """Episode-end code needs to know when exact fused results already landed."""
        from esper.simic.attribution.counterfactual_helper import CounterfactualHelper

        helper = CounterfactualHelper(seed=42, emit_callback=None)
        helper.compute_contributions_from_results(
            slot_ids=["r0c0", "r0c1"],
            results={
                (False, False): (0.5, 0.50),
                (True, False): (0.4, 0.60),
                (False, True): (0.4, 0.65),
                (True, True): (0.3, 0.75),
            },
            epoch=7,
            compute_time_seconds=0.5,
        )

        assert helper.has_precomputed_matrix_for(["r0c0", "r0c1"], epoch=7)
        assert not helper.has_precomputed_matrix_for(["r0c1", "r0c0"], epoch=7)
        assert not helper.has_precomputed_matrix_for(["r0c0", "r0c1"], epoch=8)


class TestSimpleAblation:
    """Regression coverage for simple ablation input contracts."""

    def test_compute_simple_ablation_raises_for_missing_slot_accuracy(self):
        """Missing required slot measurements must fail loudly."""
        with pytest.raises(KeyError, match="r0c1"):
            compute_simple_ablation(
                slot_ids=["r0c0", "r0c1"],
                full_accuracy=80.0,
                per_slot_accuracy={"r0c0": 70.0},
            )

    def test_compute_simple_ablation_uses_required_measurements(self):
        """Removal costs are computed from explicit per-slot measurements."""
        contributions = compute_simple_ablation(
            slot_ids=["r0c0", "r0c1"],
            full_accuracy=80.0,
            per_slot_accuracy={
                "r0c0": 70.0,
                "r0c1": 75.0,
            },
        )

        assert contributions == {
            "r0c0": pytest.approx(10.0),
            "r0c1": pytest.approx(5.0),
        }


class TestCounterfactualHelperReset:
    """B8-CR-04: Test CounterfactualHelper.reset() behavior.

    Verifies that the public reset() method properly clears cached state,
    fixing the encapsulation violation where ParallelEnvState was directly
    accessing _last_matrix.
    """

    def test_reset_clears_last_matrix(self):
        """reset() should clear _last_matrix to None."""
        from esper.simic.attribution.counterfactual_helper import CounterfactualHelper

        helper = CounterfactualHelper(emit_callback=None)

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

        helper = CounterfactualHelper(emit_callback=None)

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
            emit_callback=None,
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
