from __future__ import annotations

from typing import Any

import torch

from esper.leyline import SlotConfig
from esper.tamiyo.policy.features import batch_obs_to_features

from .parallel_env_state import ParallelEnvState


def batch_signals_to_features(
    batch_signals: list[Any],
    batch_slot_reports: list[dict[str, Any]],
    slot_config: SlotConfig,
    env_states: list[ParallelEnvState],
    device: torch.device,
    max_epochs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consolidated signals-to-features conversion for all environments."""
    return batch_obs_to_features(
        batch_signals=batch_signals,
        batch_slot_reports=batch_slot_reports,
        batch_env_states=env_states,
        slot_config=slot_config,
        device=device,
        max_epochs=max_epochs,
    )
