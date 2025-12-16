"""Property-based tests for SlotConfig and InjectionSpec.

Tests invariants for slot configuration that must hold for ALL valid inputs:
- Index ↔ slot_id bijection
- Canonical row-major ordering
- Position-based ordering from specs
- Feature size formula correctness
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from esper.leyline.slot_config import SlotConfig
from esper.leyline.injection_spec import InjectionSpec
from esper.simic.features import (
    get_feature_size,
    obs_to_multislot_features,
    BASE_FEATURE_SIZE,
    SLOT_FEATURE_SIZE,
)
from tests.strategies import slot_configs, injection_specs


class TestSlotConfigIndexRoundTrip:
    """Property: index_for_slot_id(slot_id_for_index(i)) == i for all valid i."""

    @given(config=slot_configs())
    def test_index_to_slot_to_index(self, config: SlotConfig):
        """Property: index → slot_id → index is identity."""
        for i in range(config.num_slots):
            slot_id = config.slot_id_for_index(i)
            recovered_index = config.index_for_slot_id(slot_id)
            assert recovered_index == i, (
                f"Round trip failed: index {i} → slot_id '{slot_id}' → index {recovered_index}"
            )

    @given(config=slot_configs())
    def test_slot_to_index_to_slot(self, config: SlotConfig):
        """Property: slot_id → index → slot_id is identity."""
        for slot_id in config.slot_ids:
            idx = config.index_for_slot_id(slot_id)
            recovered_slot = config.slot_id_for_index(idx)
            assert recovered_slot == slot_id, (
                f"Round trip failed: slot_id '{slot_id}' → index {idx} → slot_id '{recovered_slot}'"
            )

    @given(config=slot_configs())
    def test_all_indices_valid(self, config: SlotConfig):
        """Property: all indices 0..num_slots-1 map to valid slot_ids."""
        for i in range(config.num_slots):
            slot_id = config.slot_id_for_index(i)
            # Slot ID should match canonical format
            assert slot_id.startswith("r"), f"Invalid slot_id format: {slot_id}"
            assert "c" in slot_id, f"Invalid slot_id format: {slot_id}"


class TestSlotConfigOrdering:
    """Property: SlotConfig.for_grid() produces canonical row-major ordering."""

    @given(
        rows=st.integers(min_value=1, max_value=10),
        cols=st.integers(min_value=1, max_value=10),
    )
    def test_for_grid_row_major_order(self, rows: int, cols: int):
        """Property: for_grid produces slots in row-major order."""
        config = SlotConfig.for_grid(rows, cols)

        expected_order = []
        for r in range(rows):
            for c in range(cols):
                expected_order.append(f"r{r}c{c}")

        assert list(config.slot_ids) == expected_order, (
            f"Expected row-major order {expected_order}, got {list(config.slot_ids)}"
        )

    @given(
        rows=st.integers(min_value=1, max_value=10),
        cols=st.integers(min_value=1, max_value=10),
    )
    def test_for_grid_correct_count(self, rows: int, cols: int):
        """Property: for_grid produces rows * cols slots."""
        config = SlotConfig.for_grid(rows, cols)
        assert config.num_slots == rows * cols

    @given(config=slot_configs())
    def test_slot_ids_unique(self, config: SlotConfig):
        """Property: all slot_ids are unique."""
        assert len(config.slot_ids) == len(set(config.slot_ids))


class TestSlotConfigFromSpecs:
    """Property: from_specs preserves position ordering."""

    @given(
        n_specs=st.integers(min_value=1, max_value=10),
        data=st.data(),
    )
    def test_from_specs_orders_by_position(self, n_specs: int, data):
        """Property: from_specs orders specs by position."""
        # Generate n specs with distinct positions
        positions = sorted(data.draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n_specs,
                max_size=n_specs,
                unique=True,
            )
        ))

        # Create specs with shuffled positions
        specs = []
        for i, pos in enumerate(positions):
            specs.append(InjectionSpec(
                slot_id=f"r0c{i}",
                channels=64 * (i + 1),
                position=pos,
                layer_range=(i, i + 1),
            ))

        # Shuffle to simulate out-of-order input
        shuffled = data.draw(st.permutations(specs))
        config = SlotConfig.from_specs(list(shuffled))

        # Result should be ordered by position
        for i, slot_id in enumerate(config.slot_ids):
            expected_slot_id = f"r0c{positions.index(sorted(positions)[i])}"
            # Actually we want to verify that positions are sorted
            pass

        # Verify positions are in order
        result_positions = []
        for slot_id in config.slot_ids:
            for spec in specs:
                if spec.slot_id == slot_id:
                    result_positions.append(spec.position)
                    break

        assert result_positions == sorted(result_positions), (
            f"Specs not ordered by position: {result_positions}"
        )

    @given(
        n_specs=st.integers(min_value=1, max_value=10),
        data=st.data(),
    )
    def test_from_specs_preserves_channels(self, n_specs: int, data):
        """Property: from_specs preserves channel information."""
        specs = []
        for i in range(n_specs):
            channels = data.draw(st.integers(min_value=1, max_value=1024))
            specs.append(InjectionSpec(
                slot_id=f"r{i}c0",
                channels=channels,
                position=i / max(n_specs, 1),
                layer_range=(0, 1),
            ))

        config = SlotConfig.from_specs(specs)

        for spec in specs:
            retrieved = config.channels_for_slot(spec.slot_id)
            assert retrieved == spec.channels, (
                f"Channel mismatch for {spec.slot_id}: expected {spec.channels}, got {retrieved}"
            )


class TestFeatureSizeFormula:
    """Property: get_feature_size follows the formula BASE + num_slots * SLOT."""

    @given(config=slot_configs(max_slots=25))  # Up to 5x5 grid
    def test_feature_size_matches_formula(self, config: SlotConfig):
        """Property: feature size = 23 + num_slots * 9."""
        expected = BASE_FEATURE_SIZE + config.num_slots * SLOT_FEATURE_SIZE
        actual = get_feature_size(config)
        assert actual == expected, (
            f"Feature size mismatch: expected {expected} "
            f"(23 + {config.num_slots} * 9), got {actual}"
        )

    @given(config=slot_configs(max_slots=10))
    @settings(max_examples=50)  # Reduce examples - obs_to_multislot_features is slower
    def test_actual_features_match_declared_size(self, config: SlotConfig):
        """Property: actual feature extraction produces declared size."""
        # Minimal observation dict
        obs = {
            'epoch': 0,
            'global_step': 0,
            'train_loss': 2.3,
            'val_loss': 2.3,
            'loss_delta': 0.0,
            'train_accuracy': 10.0,
            'val_accuracy': 10.0,
            'accuracy_delta': 0.0,
            'plateau_epochs': 0,
            'best_val_accuracy': 10.0,
            'best_val_loss': 2.3,
            'loss_history_5': [2.3] * 5,
            'accuracy_history_5': [10.0] * 5,
            'total_params': 1000,
            'slots': {},
        }

        features = obs_to_multislot_features(obs, slot_config=config)
        declared_size = get_feature_size(config)

        assert len(features) == declared_size, (
            f"Feature length mismatch: got {len(features)}, declared {declared_size} "
            f"for {config.num_slots} slots"
        )


class TestInjectionSpecInvariants:
    """Property: InjectionSpec validates its invariants."""

    @given(spec=injection_specs())
    def test_channels_positive(self, spec: InjectionSpec):
        """Property: all generated specs have positive channels."""
        assert spec.channels > 0

    @given(spec=injection_specs())
    def test_position_in_range(self, spec: InjectionSpec):
        """Property: position is in [0, 1]."""
        assert 0.0 <= spec.position <= 1.0

    @given(spec=injection_specs())
    def test_layer_range_valid(self, spec: InjectionSpec):
        """Property: layer_range start <= end and start >= 0."""
        start, end = spec.layer_range
        assert start >= 0
        assert start <= end

    def test_invalid_channels_rejected(self):
        """Validation: zero/negative channels are rejected."""
        with pytest.raises(ValueError, match="positive"):
            InjectionSpec(slot_id="r0c0", channels=0, position=0.5, layer_range=(0, 1))

        with pytest.raises(ValueError, match="positive"):
            InjectionSpec(slot_id="r0c0", channels=-1, position=0.5, layer_range=(0, 1))

    def test_invalid_position_rejected(self):
        """Validation: position outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=-0.1, layer_range=(0, 1))

        with pytest.raises(ValueError, match="between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=1.1, layer_range=(0, 1))

    def test_invalid_layer_range_rejected(self):
        """Validation: invalid layer_range is rejected."""
        with pytest.raises(ValueError, match="start"):
            InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(5, 3))

        with pytest.raises(ValueError, match="non-negative"):
            InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(-1, 3))


class TestSlotConfigValidation:
    """Property: SlotConfig validates its invariants."""

    def test_empty_slot_ids_rejected(self):
        """Validation: empty slot_ids raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            SlotConfig(slot_ids=())

    def test_duplicate_slot_ids_rejected(self):
        """Validation: duplicate slot_ids raises ValueError."""
        with pytest.raises(ValueError, match="unique"):
            SlotConfig(slot_ids=("r0c0", "r0c0"))

        with pytest.raises(ValueError, match="unique"):
            SlotConfig(slot_ids=("r0c0", "r0c1", "r0c0"))
