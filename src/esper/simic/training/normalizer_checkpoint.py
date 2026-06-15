from __future__ import annotations

from typing import Any

import torch

from esper.leyline import (
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_FEATURE_SCHEMA_VERSION,
    OBS_V3_SLOT_FEATURE_SIZE,
    OBS_V3_UNKNOWN_SENTINEL,
    SlotConfig,
)
from esper.simic.control import RunningMeanStd
from esper.tamiyo.policy.features import get_feature_size


def obs_normalizer_contract(slot_config: SlotConfig) -> dict[str, Any]:
    """Return the Obs V3 contract that fitted normalizer stats must match."""
    return {
        "schema_version": OBS_V3_FEATURE_SCHEMA_VERSION,
        "slot_ids": list(slot_config.slot_ids),
        "base_feature_size": OBS_V3_BASE_FEATURE_SIZE,
        "slot_feature_size": OBS_V3_SLOT_FEATURE_SIZE,
        "state_dim": get_feature_size(slot_config),
        "unknown_sentinel": OBS_V3_UNKNOWN_SENTINEL,
    }


def _validate_obs_normalizer_shape(
    obs_normalizer: RunningMeanStd,
    *,
    expected_state_dim: int,
) -> None:
    expected_shape = torch.Size((expected_state_dim,))
    if obs_normalizer.mean.shape != expected_shape:
        raise RuntimeError(
            "Observation normalizer mean shape does not match Obs V3 contract: "
            f"mean_shape={tuple(obs_normalizer.mean.shape)} expected={tuple(expected_shape)}"
        )
    if obs_normalizer.var.shape != expected_shape:
        raise RuntimeError(
            "Observation normalizer variance shape does not match Obs V3 contract: "
            f"var_shape={tuple(obs_normalizer.var.shape)} expected={tuple(expected_shape)}"
        )


def obs_normalizer_metadata(
    obs_normalizer: RunningMeanStd,
    *,
    slot_config: SlotConfig,
) -> dict[str, Any]:
    """Serialize observation normalizer state with its fitted feature contract."""
    contract = obs_normalizer_contract(slot_config)
    state_dim = int(contract["state_dim"])
    _validate_obs_normalizer_shape(obs_normalizer, expected_state_dim=state_dim)
    return {
        "obs_normalizer_contract": contract,
        "obs_normalizer_mean": obs_normalizer.mean.tolist(),
        "obs_normalizer_var": obs_normalizer.var.tolist(),
        "obs_normalizer_count": obs_normalizer.count.item(),
        "obs_normalizer_momentum": obs_normalizer.momentum,
    }


def restore_obs_normalizer_from_metadata(
    metadata: dict[str, Any],
    *,
    obs_normalizer: RunningMeanStd,
    slot_config: SlotConfig,
    device: str,
) -> None:
    """Restore observation normalizer state only when the Obs V3 contract matches."""
    expected_contract = obs_normalizer_contract(slot_config)
    try:
        checkpoint_contract = metadata["obs_normalizer_contract"]
        mean = torch.tensor(metadata["obs_normalizer_mean"], device=device)
        var = torch.tensor(metadata["obs_normalizer_var"], device=device)
        count = torch.tensor(metadata["obs_normalizer_count"], device=device)
        momentum = metadata["obs_normalizer_momentum"]
    except KeyError as exc:
        raise RuntimeError(
            "Incompatible checkpoint metadata: missing required observation "
            f"normalizer field {exc}. Please retrain or resume from a checkpoint "
            "created with the current Obs V3 contract."
        ) from exc

    if checkpoint_contract != expected_contract:
        raise RuntimeError(
            "Observation normalizer contract mismatch: "
            f"checkpoint={checkpoint_contract}, runtime={expected_contract}. "
            "Please retrain or reset the observation normalizer before resume."
        )

    obs_normalizer.mean = mean
    obs_normalizer.var = var
    obs_normalizer.count = count
    obs_normalizer.momentum = momentum
    obs_normalizer._device = device
    _validate_obs_normalizer_shape(
        obs_normalizer,
        expected_state_dim=int(expected_contract["state_dim"]),
    )


__all__ = [
    "obs_normalizer_contract",
    "obs_normalizer_metadata",
    "restore_obs_normalizer_from_metadata",
]
