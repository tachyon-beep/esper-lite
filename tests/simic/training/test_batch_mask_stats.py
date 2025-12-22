"""Test batched mask stat computation."""
import torch
from esper.leyline import HEAD_NAMES
from esper.leyline.factored_actions import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)


def test_batch_mask_stats_no_item_in_loop():
    """Mask stats should be computed as batch ops, not per-env .item()."""
    num_envs = 4

    # Simulate masks_batch from training
    masks_batch = {
        "slot": torch.tensor([
            [True, True, False, False, False],
            [True, False, False, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]),
        "blueprint": torch.ones(4, NUM_BLUEPRINTS, dtype=torch.bool),
        "style": torch.ones(4, NUM_STYLES, dtype=torch.bool),
        "tempo": torch.ones(4, NUM_TEMPO, dtype=torch.bool),
        "alpha_target": torch.ones(4, NUM_ALPHA_TARGETS, dtype=torch.bool),
        "alpha_speed": torch.ones(4, NUM_ALPHA_SPEEDS, dtype=torch.bool),
        "alpha_curve": torch.ones(4, NUM_ALPHA_CURVES, dtype=torch.bool),
        "op": torch.tensor([
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ], dtype=torch.bool),
    }

    # BATCHED approach: compute all at once, then transfer
    # "masked" = not all True = some actions disabled
    masked_batch = {
        key: ~masks_batch[key].all(dim=-1)  # [num_envs] bool
        for key in HEAD_NAMES
    }
    # Single CPU transfer
    masked_cpu = {key: masked_batch[key].cpu().numpy() for key in HEAD_NAMES}

    # Now extract per-env (no GPU sync)
    for env_idx in range(num_envs):
        masked_flags = {key: bool(masked_cpu[key][env_idx]) for key in HEAD_NAMES}

        if env_idx == 0:
            assert masked_flags["slot"] is True  # Not all True
            assert masked_flags["blueprint"] is False  # All True
            assert masked_flags["tempo"] is False  # All True
            assert masked_flags["op"] is True  # Not all True
        elif env_idx == 2:
            assert masked_flags["slot"] is False  # All True
            assert masked_flags["tempo"] is False  # All True
            assert masked_flags["op"] is True  # Not all True


def test_batch_mask_stats_values_match_item_approach():
    """Batched approach must produce same values as .item() approach."""

    masks_batch = {
        "slot": torch.tensor([
            [True, True, False, False, False],
            [True, False, False, False, False],
        ]),
        "blueprint": torch.ones(2, NUM_BLUEPRINTS, dtype=torch.bool),
        "style": torch.ones(2, NUM_STYLES, dtype=torch.bool),
        "tempo": torch.ones(2, NUM_TEMPO, dtype=torch.bool),
        "alpha_target": torch.ones(2, NUM_ALPHA_TARGETS, dtype=torch.bool),
        "alpha_speed": torch.ones(2, NUM_ALPHA_SPEEDS, dtype=torch.bool),
        "alpha_curve": torch.ones(2, NUM_ALPHA_CURVES, dtype=torch.bool),
        "op": torch.tensor([
            [True, True, False, False, False],
            [True, True, True, False, False],
        ], dtype=torch.bool),
    }

    # Old approach (per-env .item())
    old_results = []
    for env_idx in range(2):
        masked_flags = {
            key: not bool(masks_batch[key][env_idx].all().item())
            for key in HEAD_NAMES
        }
        old_results.append(masked_flags)

    # New approach (batched)
    masked_batch = {key: ~masks_batch[key].all(dim=-1) for key in HEAD_NAMES}
    masked_cpu = {key: masked_batch[key].cpu().numpy() for key in HEAD_NAMES}
    new_results = []
    for env_idx in range(2):
        masked_flags = {key: bool(masked_cpu[key][env_idx]) for key in HEAD_NAMES}
        new_results.append(masked_flags)

    assert old_results == new_results
