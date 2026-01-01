"""Shared helpers for tests.

Keep this module small and dependency-light. It exists to avoid duplicating
cross-suite test utilities (e.g., action mask builders) and to prevent
test-only helpers from living inside `tests/integration/` (which is excluded
by default).
"""

from __future__ import annotations

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


def create_all_valid_masks(batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Create all-valid per-head action masks for factored action policies."""

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

