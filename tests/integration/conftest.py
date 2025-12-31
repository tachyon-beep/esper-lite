"""Integration test configuration.

Auto-marks all tests in this directory as integration tests.
These tests are excluded by default (run with: pytest -m integration).
"""

import pytest
import torch
from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)


def pytest_collection_modifyitems(items):
    """Auto-mark all tests in integration directory."""
    for item in items:
        # Check if test is in integration directory
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def create_all_valid_masks(batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Create all-valid per-head action masks for testing."""
    return {
        "slot": torch.ones(batch_size, 3, dtype=torch.bool),
        "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool),
        "style": torch.ones(batch_size, NUM_STYLES, dtype=torch.bool),
        "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool),
        "alpha_target": torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool),
        "alpha_speed": torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool),
        "alpha_curve": torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool),
        "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool),
    }
