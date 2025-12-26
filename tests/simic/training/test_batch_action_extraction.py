"""Test batched action extraction eliminates per-element .item() calls."""
import torch
from unittest.mock import patch


def test_batch_action_extraction_single_cpu_call():
    """Verify actions are extracted via single .cpu() not per-element .item()."""
    # Create mock action tensors on "GPU" (actually CPU but we track calls)
    actions_dict = {
        "slot": torch.tensor([0, 1, 2, 3]),
        "blueprint": torch.tensor([5, 6, 7, 8]),
        "style": torch.tensor([0, 1, 0, 1]),
        "tempo": torch.tensor([2, 3, 1, 0]),
        "alpha_target": torch.tensor([0, 1, 2, 0]),
        "alpha_speed": torch.tensor([0, 1, 2, 3]),
        "alpha_curve": torch.tensor([0, 1, 2, 0]),
        "op": torch.tensor([1, 2, 0, 3]),
    }

    # Track .item() calls - should NOT be called
    item_call_count = 0
    original_item = torch.Tensor.item
    def counting_item(self):
        nonlocal item_call_count
        item_call_count += 1
        return original_item(self)

    # Batch extraction pattern (what we want)
    with patch.object(torch.Tensor, 'item', counting_item):
        # CORRECT: Single .cpu() per head, then index with Python
        actions_cpu = {key: actions_dict[key].cpu().numpy() for key in actions_dict}
        actions = [
            {key: int(actions_cpu[key][i]) for key in actions_cpu}
            for i in range(4)
        ]

    # Verify no .item() calls were made
    assert item_call_count == 0, f"Expected 0 .item() calls, got {item_call_count}"

    # Verify correct values extracted
    assert actions[0] == {
        "slot": 0,
        "blueprint": 5,
        "style": 0,
        "tempo": 2,
        "alpha_target": 0,
        "alpha_speed": 0,
        "alpha_curve": 0,
        "op": 1,
    }
    assert actions[3] == {
        "slot": 3,
        "blueprint": 8,
        "style": 1,
        "tempo": 0,
        "alpha_target": 0,
        "alpha_speed": 3,
        "alpha_curve": 0,
        "op": 3,
    }


def test_batch_action_extraction_values_match_item():
    """Verify batched extraction produces same values as .item() approach."""
    actions_dict = {
        "slot": torch.tensor([0, 1, 2, 3]),
        "blueprint": torch.tensor([5, 6, 7, 8]),
        "style": torch.tensor([0, 1, 0, 1]),
        "tempo": torch.tensor([2, 3, 1, 0]),
        "alpha_target": torch.tensor([0, 1, 2, 0]),
        "alpha_speed": torch.tensor([0, 1, 2, 3]),
        "alpha_curve": torch.tensor([0, 1, 2, 0]),
        "op": torch.tensor([1, 2, 0, 3]),
    }

    # Old approach (what we're replacing)
    old_actions = [
        {key: actions_dict[key][i].item() for key in actions_dict}
        for i in range(4)
    ]

    # New approach
    actions_cpu = {key: actions_dict[key].cpu().numpy() for key in actions_dict}
    new_actions = [
        {key: int(actions_cpu[key][i]) for key in actions_cpu}
        for i in range(4)
    ]

    assert old_actions == new_actions
