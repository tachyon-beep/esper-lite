"""Policy factory for creating configured PolicyBundle instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from esper.tamiyo.policy.protocol import PolicyBundle

# Valid torch.compile modes
VALID_COMPILE_MODES = frozenset({"off", "default", "reduce-overhead", "max-autotune"})


def create_policy(
    policy_type: str = "lstm",
    state_dim: int | None = None,
    num_slots: int = 4,
    device: torch.device | str = "cpu",
    compile_mode: str = "off",
    lstm_hidden_dim: int = 256,
    **kwargs,
) -> PolicyBundle:
    """Create a policy instance with optional torch.compile.

    This is the recommended entry point for creating policies. It handles:
    - Policy instantiation via the registry
    - Device placement
    - torch.compile of the inner network (not the wrapper)

    Args:
        policy_type: Registered policy name (default: "lstm")
        state_dim: Observation feature dimension. If None, computed from num_slots.
        num_slots: Number of seed slots (used to compute state_dim if not provided)
        device: Target device for the policy
        compile_mode: torch.compile mode ("off", "default", "reduce-overhead", "max-autotune")
        lstm_hidden_dim: Hidden dimension for LSTM policies
        **kwargs: Additional arguments passed to policy constructor

    Returns:
        Configured PolicyBundle instance on the target device

    Raises:
        ValueError: If compile_mode is not valid

    Example:
        >>> policy = create_policy(
        ...     policy_type="lstm",
        ...     state_dim=64,
        ...     num_slots=4,
        ...     device="cuda:0",
        ...     compile_mode="default",
        ... )
    """
    from esper.tamiyo.policy.registry import get_policy
    from esper.tamiyo.policy.features import get_feature_size
    from esper.leyline.slot_config import SlotConfig

    # Validate compile_mode
    if compile_mode not in VALID_COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of {sorted(VALID_COMPILE_MODES)}, got {compile_mode!r}"
        )

    # Create slot config (single-row grid by default)
    slot_config = SlotConfig.for_grid(rows=1, cols=num_slots)

    # Compute state_dim if not provided
    if state_dim is None:
        state_dim = get_feature_size(slot_config)

    # Build config for registry
    # Note: LSTMPolicyBundle expects feature_dim, not state_dim
    config = {
        "feature_dim": state_dim,
        "hidden_dim": lstm_hidden_dim,
        "slot_config": slot_config,
        **kwargs,
    }

    # Create policy via registry
    policy = get_policy(policy_type, config)

    # Move to device (before compile - compile traces on target device)
    policy = policy.to(device)

    # Compile the inner network (not the wrapper)
    # PolicyBundle exposes .network property for this purpose
    # Device placement must happen before compile for correct tracing
    if compile_mode != "off":
        policy._network = torch.compile(
            policy.network,  # Use public property to get network
            mode=compile_mode,
            dynamic=True,
        )

    return policy


__all__ = ["create_policy"]
