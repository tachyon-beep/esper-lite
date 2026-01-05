"""Property-based tests for Simic counterfactual attribution math."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.simic.attribution.counterfactual import (
    CounterfactualConfig,
    CounterfactualEngine,
    CounterfactualMatrix,
    CounterfactualResult,
)

pytestmark = pytest.mark.property


@st.composite
def _slot_ids(draw: st.DrawFn, *, min_slots: int = 2, max_slots: int = 6) -> list[str]:
    n = draw(st.integers(min_value=min_slots, max_value=max_slots))
    return [f"r0c{i}" for i in range(n)]


@given(slot_ids=_slot_ids())
@settings(max_examples=50)
def test_shapley_generate_configs_includes_baseline_and_full(slot_ids: list[str]) -> None:
    engine = CounterfactualEngine(
        CounterfactualConfig(strategy="shapley", shapley_samples=10, seed=123)
    )
    configs = engine.generate_configs(slot_ids)
    n = len(slot_ids)
    assert tuple(False for _ in range(n)) in configs
    assert tuple(True for _ in range(n)) in configs


@given(slot_ids=_slot_ids(min_slots=1, max_slots=5))
@settings(max_examples=50)
def test_full_factorial_generate_configs_produces_2_to_the_n_unique(slot_ids: list[str]) -> None:
    engine = CounterfactualEngine(CounterfactualConfig(strategy="full_factorial"))
    configs = engine.generate_configs(slot_ids)
    n = len(slot_ids)
    assert len(configs) == 2**n
    assert len(set(configs)) == len(configs)
    assert all(len(cfg) == n for cfg in configs)


@given(
    weights=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=4,
    ),
    baseline=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    shapley_samples=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=60)
def test_shapley_values_exact_for_additive_game(
    weights: list[float], baseline: float, shapley_samples: int
) -> None:
    n = len(weights)
    slot_ids = [f"r0c{i}" for i in range(n)]

    matrix = CounterfactualMatrix(epoch=0, strategy_used="full_factorial")
    for i in range(2**n):
        config_tuple = tuple(bool(i & (1 << j)) for j in range(n))
        accuracy = baseline + sum(w for w, enabled in zip(weights, config_tuple) if enabled)
        matrix.configs.append(
            CounterfactualResult(
                config=config_tuple,
                slot_ids=tuple(slot_ids),
                val_accuracy=accuracy,
                val_loss=0.0,
            )
        )

    engine = CounterfactualEngine(
        CounterfactualConfig(strategy="shapley", shapley_samples=shapley_samples, seed=7)
    )
    shapley = engine.compute_shapley_values(matrix)

    for slot_id, weight in zip(slot_ids, weights, strict=False):
        estimate = shapley[slot_id]
        assert estimate.n_samples > 0
        assert estimate.mean == pytest.approx(weight, rel=1e-7, abs=1e-7)
        assert estimate.std == pytest.approx(0.0, abs=1e-12)

        assert matrix.marginal_contribution(slot_id) == pytest.approx(weight, rel=1e-7, abs=1e-7)


@given(
    weights=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=3,
    ),
    baseline=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=60)
def test_interaction_terms_zero_for_additive_game(weights: list[float], baseline: float) -> None:
    n = len(weights)
    slot_ids = [f"r0c{i}" for i in range(n)]

    matrix = CounterfactualMatrix(epoch=0, strategy_used="full_factorial")
    for i in range(2**n):
        config_tuple = tuple(bool(i & (1 << j)) for j in range(n))
        accuracy = baseline + sum(w for w, enabled in zip(weights, config_tuple) if enabled)
        matrix.configs.append(
            CounterfactualResult(
                config=config_tuple,
                slot_ids=tuple(slot_ids),
                val_accuracy=accuracy,
                val_loss=0.0,
            )
        )

    engine = CounterfactualEngine(CounterfactualConfig(strategy="full_factorial", seed=0))
    interactions = engine.compute_interaction_terms(matrix)
    for term in interactions.values():
        assert term.interaction == pytest.approx(0.0, abs=1e-12)

